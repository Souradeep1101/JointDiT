#!/usr/bin/env python
# scripts/infer/infer_joint.py
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# repo imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.cond.clip_cond import CLIPCondEncoder
from models.jointdit import JointDiT

# -------------------- Optional deps (VAEs & IO) --------------------
try:
    from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder

    HAVE_DIFFUSERS = True
except Exception:
    AutoencoderKLTemporalDecoder = None
    AutoencoderKL = None
    HAVE_DIFFUSERS = False

try:
    import torchaudio

    HAVE_TA = True
except Exception:
    HAVE_TA = False

try:
    from torchvision import io as tvio

    HAVE_TVIO = True
except Exception:
    HAVE_TVIO = False


# -------------------- Helpers --------------------
def load_yaml(p: str):
    return yaml.safe_load(Path(p).read_text())


def load_meta(p: str):
    return json.loads(Path(p).read_text())


def prepare_shapes_from_meta(meta: dict):
    """
    Returns:
      (T, vc, vh, vw), (ac, ah, aw), fps, audio_cfg, v_scale, a_scale
    """
    v = torch.load(Path(meta["video_latents"]), map_location="cpu")
    a = torch.load(Path(meta["audio_latents"]), map_location="cpu")

    T = int(meta.get("frame_count", v["latents"].shape[0]))
    fps = float(meta.get("fps_used", meta.get("src_fps", 12.0)))

    vc, vh, vw = int(v["latents"].shape[1]), int(v["latents"].shape[2]), int(v["latents"].shape[3])
    ac, ah, aw = int(a["latents"].shape[1]), int(a["latents"].shape[2]), int(a["latents"].shape[3])

    audio_cfg = {
        "sr": int(a.get("sr", 16000)),
        "n_mels": int(a.get("n_mels", 80)),
        "hop_length": int(a.get("hop_length", 256)),
        "win_length": int(a.get("win_length", 1024)),
        "fmin": float(a.get("fmin", 0)),
        "fmax": float(a.get("fmax", 8000)),
    }
    v_scale = float(v.get("scaling_factor", 0.18215))
    a_scale = float(a.get("scaling_factor", 0.18215))
    return (T, vc, vh, vw), (ac, ah, aw), fps, audio_cfg, v_scale, a_scale


def load_video_vae(path: str, device: torch.device):
    if not HAVE_DIFFUSERS:
        raise RuntimeError("diffusers not installed")
    last_err = None
    v_vae = None
    if AutoencoderKLTemporalDecoder is not None:
        try:
            v_vae = AutoencoderKLTemporalDecoder.from_pretrained(path)
            print("[VAE-video] loaded AutoencoderKLTemporalDecoder")
        except Exception as e:
            last_err = e
            print(f"[VAE-video][warn] temporal decoder failed: {e}")
    if v_vae is None:
        try:
            v_vae = AutoencoderKL.from_pretrained(path)
            print("[VAE-video] loaded plain AutoencoderKL")
        except Exception as e:
            raise RuntimeError(f"[VAE-video] failed both: temporal={last_err} plain={e}")
    return v_vae.to(device).eval()


def load_audio_vae(path: str, device: torch.device):
    if not HAVE_DIFFUSERS:
        raise RuntimeError("diffusers not installed")
    a_vae = AutoencoderKL.from_pretrained(path).to(device)
    a_vae.eval()
    return a_vae


@torch.no_grad()
def joint_guided_pred(
    model: JointDiT,
    v_t: torch.Tensor,
    a_t: torch.Tensor,
    w_v: float = 1.0,
    w_a: float = 1.0,
    w_txt: float = 0.0,
    w_neg_txt: float = 0.0,
    w_img: float = 0.0,
    t_pos: Optional[torch.Tensor] = None,
    t_neg: Optional[torch.Tensor] = None,
    i_pos: Optional[torch.Tensor] = None,
):
    """
    Guidance terms:
      - video/audio: pos vs iso
      - text positive: pos vs no-text
      - text negative: pos vs neg-text
      - image positive: pos vs no-image

    Implements the "JointCFG" style by comparing full vs isolated paths.
    """
    # positive contexts
    v_pos, a_pos = model(v_t, a_t, mode="full", t_ctx=t_pos, i_ctx=i_pos)

    # iso baselines (no cross)
    v_iso, _ = model(v_t, a_t, mode="iso_v", t_ctx=t_pos, i_ctx=i_pos)
    _, a_iso = model(v_t, a_t, mode="iso_a", t_ctx=t_pos, i_ctx=i_pos)

    # text (pos vs no-text)
    if w_txt > 0 and t_pos is not None:
        v_no_txt, a_no_txt = model(v_t, a_t, mode="full", t_ctx=None, i_ctx=i_pos)
    else:
        v_no_txt, a_no_txt = v_pos, a_pos

    # text (pos vs neg-text)
    if w_neg_txt > 0 and t_neg is not None:
        v_neg_txt, a_neg_txt = model(v_t, a_t, mode="full", t_ctx=t_neg, i_ctx=i_pos)
    else:
        v_neg_txt, a_neg_txt = v_pos, a_pos

    # image (pos vs no-image)
    if w_img > 0 and i_pos is not None:
        v_no_img, a_no_img = model(v_t, a_t, mode="full", t_ctx=t_pos, i_ctx=None)
    else:
        v_no_img, a_no_img = v_pos, a_pos

    v_g = (
        v_pos
        + w_v * (v_pos - v_iso)
        + w_txt * (v_pos - v_no_txt)
        + w_neg_txt * (v_pos - v_neg_txt)
        + w_img * (v_pos - v_no_img)
    )
    a_g = (
        a_pos
        + w_a * (a_pos - a_iso)
        + w_txt * (a_pos - a_no_txt)
        + w_neg_txt * (a_pos - a_neg_txt)
        + w_img * (a_pos - a_no_img)
    )
    return v_g, a_g


@torch.no_grad()
def x0_to_next_xt(
    x_t: torch.Tensor, x0_hat: torch.Tensor, sigma_t: torch.Tensor, sigma_next: torch.Tensor
):
    eps_hat = (x_t - x0_hat) / (sigma_t.clamp(min=1e-8))
    return x0_hat + sigma_next * eps_hat


# @torch.no_grad()
# def decode_video_latents(v_vae, v_latents: torch.Tensor, scaling: float) -> torch.Tensor:
#     """
#     v_latents: (T, C, H, W)
#     Returns frames: (T, 3, H', W')
#     """
#     device = next(v_vae.parameters()).device
#     weight_dtype = next(v_vae.parameters()).dtype
#     T = v_latents.shape[0]


#     if AutoencoderKLTemporalDecoder is not None and isinstance(v_vae, AutoencoderKLTemporalDecoder):
#         # decode as a single temporal chunk for speed/consistency
#         z = (v_latents.unsqueeze(0) / scaling).to(device=device, dtype=weight_dtype)  # (1,T,C,H,W)
#         x = v_vae.decode(z).sample  # expects (B,T,C,H,W)
#         x = (x.clamp(-1, 1) + 1) * 0.5  # (1,T,3,H',W')
#         frames = x[0].detach().cpu()
#     else:
#         # per-frame (plain AutoencoderKL)
#         frames = []
#         for i in range(T):
#             zi = (v_latents[i : i + 1] / scaling).to(device=device, dtype=weight_dtype)  # (1,C,H,W)
#             x = v_vae.decode(zi).sample  # (1,3,H',W')
#             x = (x.clamp(-1, 1) + 1) * 0.5
#             frames.append(x[0].detach().cpu())
#         frames = torch.stack(frames, 0)
#     return frames
@torch.no_grad()
def decode_video_latents(v_vae, v_latents, scaling):
    """
    Decode video latents to float tensor in [0,1].
    - Works with AutoencoderKLTemporalDecoder (SVD) and plain AutoencoderKL.
    - Accepts (T,C,H,W) or (B,T,C,H,W) with B==1.
    - Optional env: JOINTDIT_VID_DECODE_CHUNK (default: 8).
    Returns: (T, 3, H, W)
    """
    device = next(v_vae.parameters()).device
    weight_dtype = next(v_vae.parameters()).dtype

    # Normalize to (T,C,H,W)
    if v_latents.dim() == 5:  # (B,T,C,H,W)
        assert v_latents.size(0) == 1, "Batch>1 not supported here"
        v_latents = v_latents[0]
    assert v_latents.dim() == 4, f"expected (T,C,H,W), got {tuple(v_latents.shape)}"

    T, C, H, W = v_latents.shape
    z = (v_latents / scaling).to(device=device, dtype=weight_dtype, non_blocking=True)

    # Chunk size to cap VRAM
    try:
        chunk = max(1, int(os.getenv("JOINTDIT_VID_DECODE_CHUNK", "8")))
    except Exception:
        chunk = 8

    frames = []

    # Temporal path
    if isinstance(v_vae, AutoencoderKLTemporalDecoder):
        if chunk >= T:  # all at once
            z_btchw = z.reshape(T, C, H, W)  # (T,C,H,W)
            x = v_vae.decode(z_btchw, num_frames=T).sample  # (1,3,T,H,W)
            if x.dim() == 4:  # rare: T==1 → (1,3,H,W)
                x = x.unsqueeze(2)  # (1,3,1,H,W)
            x = (x.clamp(-1, 1) + 1) * 0.5
            frames = x[0].permute(1, 0, 2, 3).contiguous().cpu()  # (T,3,H,W)
        else:
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                z_chunk = z[s:e].reshape(e - s, C, H, W)  # (chunk,C,H,W)
                x = v_vae.decode(z_chunk, num_frames=(e - s)).sample
                # If num_frames==1, diffusers returns (1,3,H,W); otherwise (1,3,chunk,H,W)
                if x.dim() == 4:  # (1,3,H,W)
                    x = x.unsqueeze(2)  # (1,3,1,H,W)
                x = (x.clamp(-1, 1) + 1) * 0.5
                frames.append(x[0].permute(1, 0, 2, 3).contiguous().cpu())  # (chunk,3,H,W)
            frames = torch.cat(frames, dim=0)

    # Plain image VAE path
    else:
        if chunk >= T:
            x = v_vae.decode(z).sample  # (T,3,H,W)
            x = (x.clamp(-1, 1) + 1) * 0.5
            frames = x.contiguous().cpu()
        else:
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                z_chunk = z[s:e]  # (chunk,C,H,W)
                x = v_vae.decode(z_chunk).sample  # (chunk,3,H,W)
                x = (x.clamp(-1, 1) + 1) * 0.5
                frames.append(x.contiguous().cpu())
            frames = torch.cat(frames, dim=0)

    return frames  # (T,3,H,W)


@torch.no_grad()
def decode_audio_latents_to_mel(a_vae, a_latents: torch.Tensor, scaling: float) -> torch.Tensor:
    """
    a_latents: (1, Ca, Ha, Wa)
    Returns mel image: (Ha, Wa) in [0,1]
    """
    device = next(a_vae.parameters()).device
    weight_dtype = next(a_vae.parameters()).dtype
    za = (a_latents / scaling).to(device=device, dtype=weight_dtype)
    x = a_vae.decode(za).sample  # (1,1,Ha,Wa) or (1,C,Ha,Wa) but we expect C=1
    x = (x.clamp(-1, 1) + 1) * 0.5
    return x[0, 0].detach().cpu()


def mel_to_waveform(
    mel_img: torch.Tensor, sr: int, n_mels: int, hop: int, win: int, fmin: float, fmax: float
):
    if not HAVE_TA:
        print("[audio][warn] torchaudio not available; skipping WAV decode.")
        return None
    mel_db_range = float(os.environ.get("JOINTDIT_MEL_DB_RANGE", "60"))
    mel_db_floor = -mel_db_range
    mel_db = mel_img * mel_db_range + mel_db_floor
    mel_power = (10.0 ** (mel_db / 10.0)).clamp(min=1e-10)
    invmel = torchaudio.transforms.InverseMelScale(
        n_stft=win // 2 + 1, n_mels=n_mels, sample_rate=sr, f_min=fmin, f_max=fmax
    )
    spec_power = invmel(mel_power)
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=win, hop_length=hop, win_length=win, power=1.0, n_iter=32
    )
    wav = griffin(spec_power)
    peak = float(wav.abs().max())
    if peak > 0:
        wav = 0.99 * (wav / peak)
    gain_db = float(os.environ.get("JOINTDIT_AUDIO_GAIN_DB", "0"))
    if gain_db != 0:
        wav = wav * (10.0 ** (gain_db / 20.0))
        wav = wav.clamp(-1.0, 1.0)
    return wav


def write_video_mp4(frames: torch.Tensor, out_path: Path, fps: float):
    if not HAVE_TVIO:
        raise RuntimeError("torchvision is required to write MP4.")
    frames_u8 = (frames.clamp(0, 1) * 255).to(torch.uint8)  # (T,3,H,W)
    frames_thwc = frames_u8.permute(0, 2, 3, 1).contiguous().cpu()  # (T,H,W,3)
    tvio.write_video(str(out_path), frames_thwc, fps=fps, video_codec="h264", options={"crf": "18"})


def write_wav(wav: torch.Tensor, sr: int, out_path: Path):
    if wav is None:
        return
    import soundfile as sf

    sf.write(str(out_path), wav.numpy(), sr)


def mux_audio_into_mp4(mp4_path: Path, wav_path: Path, out_path: Path) -> bool:
    if not wav_path.exists() or shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4_path),
                "-i",
                str(wav_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except Exception as e:
        print("[mux][warn] could not mux:", e)
        return False


# -------------------- Env overrides --------------------
def _env_override(cfg: dict):
    # top-level basics
    if os.getenv("JOINTDIT_CKPT"):
        cfg["ckpt"] = os.getenv("JOINTDIT_CKPT")
    if os.getenv("JOINTDIT_REF_META"):
        cfg["ref_meta"] = os.getenv("JOINTDIT_REF_META")

    if os.getenv("JOINTDIT_PROMPT") is not None:
        cfg["prompt"] = os.getenv("JOINTDIT_PROMPT")
    if os.getenv("JOINTDIT_NEG_PROMPT") is not None:
        cfg["negative_prompt"] = os.getenv("JOINTDIT_NEG_PROMPT")

    # guidance
    g = cfg.setdefault("guidance", {})

    def _setf(env, key):
        v = os.getenv(env)
        if v:
            g[key] = float(v)

    _setf("JOINTDIT_WV", "weight_v")
    _setf("JOINTDIT_WA", "weight_a")
    _setf("JOINTDIT_WT", "weight_txt")
    _setf("JOINTDIT_WNT", "weight_neg_txt")
    _setf("JOINTDIT_WI", "weight_img")

    # step/seed
    v = os.getenv("JOINTDIT_STEPS")
    cfg["steps"] = int(v) if v else cfg.get("steps", 30)
    v = os.getenv("JOINTDIT_SEED")
    cfg["seeds"] = [int(v)] if v else cfg.get("seeds", [0])

    # CLIP / image controls
    clip = cfg.setdefault("clip", {})

    # explicit image path
    img_override = os.getenv("JOINTDIT_IMAGE")
    if img_override:
        clip["image_path"] = img_override
        clip["use_image_firstframe"] = False
        clip["enabled"] = True

    # use meta first-frame flag
    v = os.getenv("JOINTDIT_USE_IMG")
    if v is not None:
        clip["use_image_firstframe"] = str(v).lower() not in ("0", "false", "no")
        if clip["use_image_firstframe"]:
            clip["enabled"] = True

    # override CLIP backbone/tag
    v = os.getenv("JOINTDIT_CLIP_MODEL")
    if v:
        clip["model"] = v
    v = os.getenv("JOINTDIT_CLIP_PRETRAINED")
    if v:
        clip["pretrained"] = v

    # auto-enable clip if any condition is present
    if (
        cfg.get("prompt")
        or cfg.get("negative_prompt")
        or clip.get("image_path")
        or clip.get("use_image_firstframe")
    ):
        clip["enabled"] = True


# Manual shapes if no ref_meta is supplied
def _shapes_from_manual(cfg: dict, v_vae_scale: float, a_vae_scale: float):
    m = cfg.get("manual_shapes", {})
    if not m:
        return None
    T = int(m.get("T", 12))
    v = m.get("video", {})
    a = m.get("audio", {})
    vc, vh, vw = int(v.get("c", 4)), int(v.get("h", 32)), int(v.get("w", 32))
    ac, ah, aw = int(a.get("c", 8)), int(a.get("h", 16)), int(a.get("w", 32))
    fps = int(m.get("fps", 12))
    sr = int(m.get("sr", 16000))
    audio_cfg = {
        "sr": sr,
        "n_mels": 80,
        "hop_length": 256,
        "win_length": 1024,
        "fmin": 0.0,
        "fmax": 8000.0,
    }
    return (T, vc, vh, vw), (ac, ah, aw), fps, audio_cfg, float(v_vae_scale), float(a_vae_scale)


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    _env_override(cfg)

    device = torch.device(cfg["runtime"]["device"])
    out_dir = Path(cfg.get("out_dir", cfg.get("out", {}).get("dir", "./outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # model config from ckpt (preferred) or day05 config fallback
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    if "cfg" in ckpt and "model" in ckpt:
        mcfg = ckpt["cfg"]["model"]
    elif "cfg" in ckpt and "cfg" in ckpt["cfg"] and "model" in ckpt["cfg"]["cfg"]:
        # Just-in-case some trainers pack differently
        mcfg = ckpt["cfg"]["cfg"]["model"]
    else:
        mcfg = load_yaml(str(Path(__file__).resolve().parents[2] / "configs" / "day05_train.yaml"))[
            "model"
        ]

    rope_cfg = {
        "video": {"enable": bool(mcfg.get("rope_video", mcfg.get("rope", True)))},
        "audio": {"enable": bool(mcfg.get("rope_audio", mcfg.get("rope", True)))},
    }
    model = JointDiT(
        d_model=mcfg["d_model"],
        heads=mcfg.get("heads", mcfg.get("n_heads", 8)),
        ff_mult=mcfg.get("ff_mult", 2),
        dropout=mcfg.get("attn_dropout", mcfg.get("ff_dropout", 0.0)),
        rope_cfg=rope_cfg,
        joint_blocks=mcfg.get("joint_blocks", mcfg.get("n_blocks", 2)),
        video_in_ch=mcfg.get("video_in_ch", 4),
        audio_in_ch=mcfg.get("audio_in_ch", 8),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # VAEs first (needed for scaling when running without meta)
    v_vae = load_video_vae(cfg["video_vae_path"], device)
    a_vae = load_audio_vae(cfg["audio_vae_path"], device)
    v_scale_default = getattr(v_vae.config, "scaling_factor", 0.18215)
    a_scale_default = getattr(a_vae.config, "scaling_factor", 0.18215)

    # Shapes: prefer meta, else manual
    meta_path = cfg.get("ref_meta")
    if meta_path and Path(meta_path).exists():
        (T, vc, vh, vw), (ac, ah, aw), fps_meta, aparams, v_scale, a_scale = (
            prepare_shapes_from_meta(load_meta(meta_path))
        )
        fps = float(cfg.get("mp4_fps", fps_meta))
        sr = int(cfg.get("audio_sr", aparams["sr"]))
    else:
        over = _shapes_from_manual(cfg, v_scale_default, a_scale_default)
        if over is None:
            raise RuntimeError(
                "No ref_meta provided and manual_shapes missing. Provide one of them."
            )
        (T, vc, vh, vw), (ac, ah, aw), fps, aparams, v_scale, a_scale = over
        fps = float(cfg.get("mp4_fps", fps))
        sr = int(cfg.get("audio_sr", aparams["sr"]))
        aparams["sr"] = sr

    seeds = cfg.get("seeds", [0])
    steps = int(cfg["steps"])

    # log-space sigmas (simple Karras/DDPM-ish schedule)
    sig_v = torch.exp(torch.linspace(math.log(1.0), math.log(1e-3), steps + 1, device=device))
    sig_a = torch.exp(torch.linspace(math.log(1.0), math.log(1e-3), steps + 1, device=device))

    # guidance weights
    gcfg = cfg.get("guidance", {})
    wv = float(gcfg.get("weight_v", 1.5))
    wa = float(gcfg.get("weight_a", 1.5))
    wt = float(gcfg.get("weight_txt", 1.5))
    wnt = float(gcfg.get("weight_neg_txt", 0.0))
    wi = float(gcfg.get("weight_img", 0.0))

    # CLIP config
    clip_cfg = cfg.get("clip", {})
    clip_on = bool(clip_cfg.get("enabled", False))
    clip_variant = clip_cfg.get("variant") or clip_cfg.get("model") or "ViT-B-16"
    clip_tag = clip_cfg.get("pretrained", "openai")

    prompt = str(cfg.get("prompt", "") or "")
    neg_prompt = str(cfg.get("negative_prompt", "") or "")

    # choose image_path: explicit override > meta first-frame (if allowed)
    img_path = clip_cfg.get("image_path", None)
    if not img_path and clip_cfg.get("use_image_firstframe", False) and meta_path:
        meta = load_meta(meta_path)
        img_path = meta.get("img_firstframe")

    # If user provided any condition (text/image), try to enable CLIP.
    wants_clip = clip_on or bool(prompt) or bool(neg_prompt) or bool(img_path)
    enc = None
    t_pos = t_neg = i_pos = None
    if wants_clip:
        try:
            enc = CLIPCondEncoder(
                d_model=mcfg["d_model"],
                clip_name=clip_variant,
                pretrained=clip_tag,
                device=device,
            )
            print(f"[clip] loaded {clip_variant} · pretrained='{clip_tag}'")
            if prompt:
                t_pos, _ = enc([prompt], [None], device=device)
            if neg_prompt:
                t_neg, _ = enc([neg_prompt], [None], device=device)
            if img_path:
                _, i_pos = enc(None, [img_path], device=device)
            used = []
            if t_pos is not None:
                used.append("text+")
            if t_neg is not None:
                used.append("text-")
            if i_pos is not None:
                used.append("image")
            print("[clip] using:", ", ".join(used) if used else "none")
        except Exception as e:
            print(f"[clip][warn] disabling CLIP due to load error: {e}")
            enc = None
            t_pos = t_neg = i_pos = None
            wt = wnt = wi = 0.0
    else:
        # no prompt/image and not explicitly enabled
        wt = wnt = wi = 0.0
        print("[clip] disabled (no prompt/image and clip.enabled=False)")

    # -------------------- Sampler loop --------------------
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(int(seed))

        v_t = sig_v[0] * torch.randn(
            1, T, vc, vh, vw, device=device, dtype=torch.float32, generator=gen
        )
        a_t = sig_a[0] * torch.randn(
            1, ac, ah, aw, device=device, dtype=torch.float32, generator=gen
        )

        with torch.no_grad():
            for i in range(steps):
                if (i % max(1, steps // 10) == 0) or (i == steps - 1):
                    print(f"[sample] step {i+1}/{steps}", flush=True)

                s_v, s_v_next = sig_v[i], sig_v[i + 1]
                s_a, s_a_next = sig_a[i], sig_a[i + 1]

                v0_hat, a0_hat = joint_guided_pred(
                    model,
                    v_t,
                    a_t,
                    w_v=wv,
                    w_a=wa,
                    w_txt=wt,
                    w_neg_txt=wnt,
                    w_img=wi,
                    t_pos=t_pos,
                    t_neg=t_neg,
                    i_pos=i_pos,
                )

                v_t = x0_to_next_xt(v_t, v0_hat, s_v, s_v_next)
                a_t = x0_to_next_xt(a_t, a0_hat, s_a, s_a_next)

        # final x0 (remove batch dim for video; keep 1x for audio latent)
        v0 = v0_hat[0].detach().cpu()  # (T,C,H,W)
        a0 = a0_hat[0:1].detach().cpu()  # (1,C,H,W)

        print(f"[decode] video {tuple(v0.shape)}  audio {tuple(a0.shape)}")
        v_frames = decode_video_latents(v_vae, v0, v_scale)
        mel_img = decode_audio_latents_to_mel(a_vae, a0, a_scale)
        wav = mel_to_waveform(
            mel_img,
            sr,
            aparams["n_mels"],
            aparams["hop_length"],
            aparams["win_length"],
            aparams["fmin"],
            aparams["fmax"],
        )

        stem = f"seed{int(seed):03d}_steps{steps}"
        mp4_path = out_dir / f"{stem}.mp4"
        wav_path = out_dir / f"{stem}.wav"

        if HAVE_TVIO:
            write_video_mp4(v_frames, mp4_path, fps=fps)
            print(f"[out] {mp4_path}")
        if wav is not None:
            write_wav(wav, sr, wav_path)
            print(f"[out] {wav_path}")

        if HAVE_TVIO and wav is not None:
            av_path = out_dir / f"{stem}_av.mp4"
            if mux_audio_into_mp4(mp4_path, wav_path, av_path):
                print(f"[out] {av_path}")


if __name__ == "__main__":
    main()
