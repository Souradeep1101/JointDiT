#!/usr/bin/env python
# scripts/data/cache_latents.py
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.io import read_video

# audio
try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False

# diffusers VAE (robust import: temporal first, then plain)
HAVE_DIFFUSERS = True
try:
    from diffusers import AutoencoderKLTemporalDecoder
except Exception:
    AutoencoderKLTemporalDecoder = None

try:
    from diffusers import AutoencoderKL
except Exception:
    HAVE_DIFFUSERS = False
    AutoencoderKL = None

# CLIP (image + text)
try:
    import open_clip

    HAVE_OPENCLIP = True
except Exception:
    HAVE_OPENCLIP = False


def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def find_videos(root, exts):
    root = Path(root)
    files = []
    for e in exts:
        files.extend([str(p) for p in root.rglob(f"*{e}") if p.is_file()])
    return sorted(files)


def ensure_dirs(base, split):
    out = {
        "video_latents": Path(base) / "video_latents" / split,
        "audio_latents": Path(base) / "audio_latents" / split,
        "img_firstframe": Path(base) / "img_firstframe" / split,
        "img_clip": Path(base) / "img_clip" / split,
        "txt_tokens": Path(base) / "txt_tokens" / split,  # <-- NEW (CLIP TEXT)
        "txt_clip": Path(base) / "txt_clip" / split,  # <-- NEW (CLIP TEXT)
        "meta": Path(base) / "meta" / split,
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def to_device_dtype(x, device="cuda", dtype=torch.float16):
    return x.to(device=device, dtype=dtype, non_blocking=True)


def resize_keep_short_side(img_t, target_short_side):
    # img_t: (C,H,W) in [0,1]
    _, H, W = img_t.shape
    if target_short_side is None or min(H, W) == target_short_side:
        return img_t
    if H < W:
        new_h = target_short_side
        new_w = int(W * (target_short_side / H))
    else:
        new_w = target_short_side
        new_h = int(H * (target_short_side / W))
    img_t = transforms.functional.resize(img_t, [new_h, new_w], antialias=True)
    return img_t


def frame_batch_encode(frames, vae, batch=4, device="cuda", dtype=torch.float16):
    # frames: list of (C,H,W) in [0,1]
    zs = []
    for i in range(0, len(frames), batch):
        ft = torch.stack(frames[i : i + batch], 0)  # (b, C, H, W)
        ft = to_device_dtype(ft * 2 - 1, device, dtype)  # -> [-1,1]
        with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)), torch.no_grad():
            z = vae.encode(ft).latent_dist.sample() * vae.config.scaling_factor
        zs.append(z.detach().cpu())
    return torch.cat(zs, 0)  # (T, c, h, w)


def mel_spectrogram(wav, sr, n_mels, hop_length, win_length, fmin, fmax):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        power=2.0,
    )(wav)
    spec = torch.clamp(spec, min=1e-10).log()  # log-mel
    return spec  # (1, n_mels, T)


def encode_audio_spec_to_latents(spec_img, vae, device="cuda", dtype=torch.float16):
    # spec_img: (1, n_mels, T) -> treat as (B=1, C=1, H=n_mels, W=T) in [0,1] normalized
    x = spec_img
    x = (x - x.amin(dim=(-2, -1), keepdim=True)) / (
        x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + 1e-6
    )
    x = x.unsqueeze(0)  # (1,1,H,W)
    x = to_device_dtype(x * 2 - 1, device, dtype)
    with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)), torch.no_grad():
        z = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    return z.detach().cpu()  # (1, c, h, w)


def load_video_vae(video_vae_path, device):
    """
    Robust loader for SVD VAE:
      - Try AutoencoderKLTemporalDecoder (correct for SVD)
      - Fallback to AutoencoderKL
    """
    if not HAVE_DIFFUSERS:
        raise RuntimeError("diffusers not available. Install it in your venv.")
    print("[load] video VAE:", video_vae_path)

    last_err = None
    v_vae = None

    # 1) temporal decoder (preferred for SVD)
    if AutoencoderKLTemporalDecoder is not None:
        try:
            v_vae = AutoencoderKLTemporalDecoder.from_pretrained(video_vae_path)
            print("[ok] loaded AutoencoderKLTemporalDecoder")
        except Exception as e:
            last_err = e
            print(f"[warn] temporal VAE failed: {e}")

    # 2) fallback to plain KL VAE
    if v_vae is None:
        try:
            v_vae = AutoencoderKL.from_pretrained(video_vae_path)
            print("[ok] loaded AutoencoderKL")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load video VAE from {video_vae_path}. "
                f"Temporal error: {last_err}. Plain error: {e}"
            )

    v_vae = v_vae.to(device)
    v_vae.eval()
    return v_vae


# ----------------- CLIP helpers (image + text) -----------------
def load_clip_model(pretrained: str, variant: str, device: str):
    """
    Robust CLIP loader for open_clip.
    Returns (model, preprocess, tokenizer) or (None, None, None).
    """
    if not HAVE_OPENCLIP:
        print("[warn] CLIP requested but open_clip not installed; skipping CLIP.")
        return None, None, None

    # --- normalize common aliases like "ViT-L/14" -> "ViT-L-14"
    alias = {
        "ViT-B/32": "ViT-B-32",
        "ViT-B/16": "ViT-B-16",
        "ViT-L/14": "ViT-L-14",
        "ViT-L/14-336": "ViT-L-14-336",
        "ViT-H/14": "ViT-H-14",
        "ViT-g/14": "ViT-g-14",
    }
    variant = alias.get(variant, variant).replace("/", "-")

    def _create(pre):
        return open_clip.create_model_and_transforms(variant, pretrained=pre, device=device)

    try:
        if os.path.isdir(pretrained):
            cands = []
            for ext in (".pt", ".pth"):
                cands.extend(glob.glob(os.path.join(pretrained, f"*{ext}")))
            if not cands:
                raise FileNotFoundError(f"No .pt/.pth under {pretrained}")
            ckpt = sorted(cands)[0]
            print(f"[load] CLIP local checkpoint: {ckpt} (model={variant})")
            model, _, preprocess = _create(ckpt)
        elif os.path.isfile(pretrained):
            print(f"[load] CLIP local checkpoint file: {pretrained} (model={variant})")
            model, _, preprocess = _create(pretrained)
        else:
            print(f"[load] CLIP tag '{pretrained}' for model {variant}")
            model, _, preprocess = _create(pretrained)
        tokenizer = open_clip.get_tokenizer(variant)
        model.eval()
        print(f"[ok] CLIP loaded with '{pretrained}'")
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"[warn] CLIP load failed for pretrained='{pretrained}': {e}")
        try:
            print("[info] Falling back to pretrained='openai'")
            model, _, preprocess = _create("openai")
            tokenizer = open_clip.get_tokenizer(variant)
            model.eval()
            print("[ok] CLIP loaded with 'openai'")
            return model, preprocess, tokenizer
        except Exception as e2:
            print(f"[warn] CLIP fallback 'openai' also failed: {e2}")
            print("[warn] Disabling CLIP for this run.")
            return None, None, None


def _read_existing_meta(meta_dir: Path, stem: str):
    p = meta_dir / f"{stem}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _pick_caption(existing_meta: dict, vpath: str, cfg_clip: dict):
    """
    Choose a text prompt for this sample.
    Priority:
      1) existing_meta['captions'] (list[str]) -> join_with
      2) class name from path parent (e.g., 'dog_barking') -> map to phrase
      3) fallback to stem
    """
    join_with = str(cfg_clip.get("text", {}).get("join_with", ", "))
    # 1) meta captions
    caps = existing_meta.get("captions", None)
    if isinstance(caps, list) and caps:
        return join_with.join([str(c) for c in caps])

    # 2) derive class from folder name and map
    classname = Path(vpath).parent.name
    mapping = dict(cfg_clip.get("text", {}).get("classname_map", {}))
    if classname in mapping:
        return str(mapping[classname])

    # nicely spaced version of classname if it’s like dog_barking
    if "_" in classname:
        return classname.replace("_", " ")

    # 3) fallback: stem
    return Path(vpath).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="override cfg.runtime.limit")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = cfg["runtime"]["device"]
    dtype = torch.float16 if str(cfg["runtime"]["dtype"]).lower() == "fp16" else torch.float32

    raw_dir = Path(cfg["raw"]["dir"]) / args.split
    cache_dirs = ensure_dirs(cfg["cache"]["dir"], args.split)
    files = find_videos(raw_dir, cfg["raw"]["exts"])
    if not files:
        print(f"[warn] No videos found under {raw_dir} with exts {cfg['raw']['exts']}")
        return

    limit = args.limit if args.limit is not None else cfg["runtime"]["limit"]
    if limit is not None and limit > 0:
        files = files[:limit]

    # Load models
    v_vae = load_video_vae(cfg["video"]["vae_path"], device)

    print("[load] audio VAE:", cfg["audio"]["vae_path"])
    a_vae = AutoencoderKL.from_pretrained(cfg["audio"]["vae_path"]).to(device)
    a_vae.eval()

    # ---- CLIP config ----
    clip_cfg = cfg.get("clip", {})
    clip_enabled = bool(clip_cfg.get("enabled", False))
    model = preprocess = tokenizer = None
    if clip_enabled:
        pretrained = clip_cfg.get("model_path", "openai")
        variant = clip_cfg.get("variant", "ViT-L/14")
        print("[load] CLIP:", pretrained, variant)
        model, preprocess, tokenizer = load_clip_model(pretrained, variant, device)
        if model is None:
            clip_enabled = False

    # Transforms
    to_tensor = transforms.ToTensor()  # PIL/np -> [0,1] tensor
    resize_policy = cfg["video"]["resize_policy"]
    short_side = cfg["video"]["target_short_side"]
    max_frames = int(cfg["video"]["max_frames"])
    fps_target = cfg["video"]["target_fps"]
    frame_batch = int(cfg["video"]["frame_batch"])

    sr = int(cfg["audio"]["sample_rate"])
    n_mels = int(cfg["audio"]["n_mels"])
    hop = int(cfg["audio"]["hop_length"])
    win = int(cfg["audio"]["win_length"])
    fmin = float(cfg["audio"]["fmin"])
    fmax = float(cfg["audio"]["fmax"])
    max_seconds = float(cfg["audio"]["max_seconds"])

    # CLIP TEXT toggles
    save_img_emb = bool(clip_cfg.get("image", {}).get("save_image_emb", True))
    save_txt_tokens = bool(clip_cfg.get("text", {}).get("save_tokens", True))
    save_txt_emb = bool(clip_cfg.get("text", {}).get("save_text_emb", True))

    stats = {"ok": 0, "fail": 0, "items": []}

    for vi, vpath in enumerate(files, 1):
        stem = Path(vpath).stem
        print(f"\n[{vi}/{len(files)}] {vpath}")
        try:
            # --- read existing meta (from fetch script) if present ---
            existing_meta = _read_existing_meta(cache_dirs["meta"], stem)

            # --- read video (frames + audio waveform) ---
            video, audio, info = read_video(vpath, pts_unit="sec")
            src_fps = info["video_fps"]

            # first frame image
            first = Image.fromarray(video[0].numpy(), mode="RGB")
            first_out = cache_dirs["img_firstframe"] / f"{stem}.png"
            first.save(first_out)

            # optional CLIP (image)
            if clip_enabled and save_img_emb:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
                    clip_img = preprocess(first).unsqueeze(0).to(device)
                    clip_emb = model.encode_image(clip_img)
                    clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
                torch.save(
                    {"embedding": clip_emb.detach().cpu()}, cache_dirs["img_clip"] / f"{stem}.pt"
                )

            # ---- CLIP TEXT: choose caption and cache tokens/emb ----
            caption_text = ""
            if clip_enabled and (save_txt_tokens or save_txt_emb):
                caption_text = _pick_caption(existing_meta, vpath, clip_cfg)
                if save_txt_tokens and tokenizer is not None:
                    tok = tokenizer([caption_text])  # (1,77) Long
                    torch.save(
                        {"tokens": tok.cpu(), "text": caption_text},
                        cache_dirs["txt_tokens"] / f"{stem}.pt",
                    )
                if save_txt_emb:
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
                        tok = tokenizer([caption_text]).to(device)
                        txt_emb = model.encode_text(tok)
                        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                    torch.save(
                        {"embedding": txt_emb.detach().cpu(), "text": caption_text},
                        cache_dirs["txt_clip"] / f"{stem}.pt",
                    )

            # --- select frames uniformly up to max_frames (optionally fps clamp) ---
            T = video.shape[0]
            if fps_target is not None and fps_target > 0:
                stride = max(1, int(round(src_fps / fps_target)))
                idxs = np.arange(0, T, stride)
            else:
                idxs = np.arange(T)
            if len(idxs) > max_frames:
                idxs = np.linspace(0, len(idxs) - 1, max_frames).round().astype(int)
            frames = video[idxs]  # (t, H, W, C) u8

            # to list of C,H,W float
            frame_tensors = []
            for k in range(frames.shape[0]):
                pt = to_tensor(Image.fromarray(frames[k].numpy(), mode="RGB"))  # [0,1]
                if resize_policy == "short_side":
                    pt = resize_keep_short_side(pt, short_side)
                frame_tensors.append(pt)

            # --- encode frames with video VAE ---
            v_lat = frame_batch_encode(
                frame_tensors, v_vae, batch=frame_batch, device=device, dtype=dtype
            )
            torch.save(
                {
                    "latents": v_lat.half() if dtype == torch.float16 else v_lat.float(),
                    "scaling_factor": v_vae.config.scaling_factor,
                    "idxs": idxs.tolist(),
                    "src_fps": float(src_fps),
                },
                cache_dirs["video_latents"] / f"{stem}.pt",
            )

            # --- audio to mono + resample ---
            if not HAVE_TORCHAUDIO:
                raise RuntimeError("torchaudio not available; install torchaudio to cache audio.")
            if audio.numel() == 0:
                approx_vid_secs = len(idxs) / (fps_target if fps_target else src_fps)
                target_len = int(sr * min(approx_vid_secs, max_seconds))
                wav = torch.zeros((1, target_len))
                src_sr = sr
            else:
                src_sr = int(info["audio_fps"])
                a = audio  # (S, C)
                a = a.mean(-1) if a.ndim == 2 and a.shape[-1] > 1 else a.squeeze(-1)  # mono
                a = a.contiguous().t().contiguous() if a.ndim == 2 else a.unsqueeze(0)  # (1, S)
                if src_sr != sr:
                    a = torchaudio.functional.resample(a, src_sr, sr)
                approx_vid_secs = len(idxs) / (fps_target if fps_target else src_fps)
                target_len = int(sr * min(approx_vid_secs, max_seconds))
                if a.shape[-1] >= target_len:
                    a = a[..., :target_len]
                else:
                    pad = target_len - a.shape[-1]
                    a = F.pad(a, (0, pad))
                wav = a  # (1, S)
            # --- mel spec ---
            spec = mel_spectrogram(wav, sr, n_mels, hop, win, fmin, fmax)  # (1, n_mels, Tm)
            # --- encode with audio VAE ---
            a_lat = encode_audio_spec_to_latents(
                spec, a_vae, device=device, dtype=dtype
            )  # (1, c, h, w)
            torch.save(
                {
                    "latents": a_lat.half() if dtype == torch.float16 else a_lat.float(),
                    "scaling_factor": a_vae.config.scaling_factor,
                    "sr": sr,
                    "n_mels": n_mels,
                    "hop_length": hop,
                    "win_length": win,
                    "fmin": fmin,
                    "fmax": fmax,
                },
                cache_dirs["audio_latents"] / f"{stem}.pt",
            )

            # --- meta (MERGE with any existing, keep captions if present) ---
            meta = {
                "stem": stem,
                "video_file": vpath,
                "img_firstframe": str(first_out),
                "clip_file": (
                    (cache_dirs["img_clip"] / f"{stem}.pt").as_posix()
                    if (clip_enabled and save_img_emb)
                    else existing_meta.get("clip_file")
                ),
                "video_latents": (cache_dirs["video_latents"] / f"{stem}.pt").as_posix(),
                "audio_latents": (cache_dirs["audio_latents"] / f"{stem}.pt").as_posix(),
                "frame_count": int(len(idxs)),
                "src_fps": float(existing_meta.get("src_fps", src_fps)),
                "fps_used": float(
                    fps_target if fps_target else existing_meta.get("src_fps", src_fps)
                ),
                "sr": sr,
            }
            # preserve or set captions
            if "captions" in existing_meta and isinstance(existing_meta["captions"], list):
                meta["captions"] = existing_meta["captions"]
            elif caption_text:
                meta["captions"] = [caption_text]

            # record locations for text artifacts if we made them
            if clip_enabled and save_txt_tokens:
                meta["clip_text_tokens"] = (cache_dirs["txt_tokens"] / f"{stem}.pt").as_posix()
            if clip_enabled and save_txt_emb:
                meta["clip_text_emb"] = (cache_dirs["txt_clip"] / f"{stem}.pt").as_posix()

            # write merged meta
            with open(cache_dirs["meta"] / f"{stem}.json", "w") as f:
                json.dump(meta, f, indent=2)

            stats["ok"] += 1
            stats["items"].append(stem)

        except Exception as e:
            stats["fail"] += 1
            print(f"[ERROR] {stem}: {e}")

    print("\n======== cache summary ========")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
