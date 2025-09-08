import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# add repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_loader.collate import collate_jointdit
from data_loader.jointdit_dataset import JointDiTDataset
from models.cond.clip_cond import CLIPCondEncoder
from models.jointdit import JointDiT
from models.noise_schedules import (
    add_gaussian_noise,
    ddpm_like_sigma_from_tuni,
    edm_sigma_from_tuni,
    has_nan_or_inf,
    mse_x0,
)


# ---------------- utils ----------------
def load_cfg(p):
    return yaml.safe_load(Path(p).read_text())


def latest_ckpt(ckpt_dir: Path) -> Path | None:
    files = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    return files[-1] if files else None


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _get_any(d: dict, names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def _rope_from_cfg(mcfg):
    rv = bool(mcfg.get("rope_video", mcfg.get("rope", True)))
    ra = bool(mcfg.get("rope_audio", mcfg.get("rope", True)))
    return {"video": {"enable": rv}, "audio": {"enable": ra}}


def build_model(cfg, device):
    mcfg = cfg["model"]
    d_model = int(mcfg["d_model"])
    heads = int(mcfg.get("heads", mcfg.get("n_heads", 8)))
    ff_mult = int(mcfg.get("ff_mult", 2))
    dropout = float(mcfg.get("attn_dropout", mcfg.get("ff_dropout", 0.0)))
    jb = int(mcfg.get("joint_blocks", mcfg.get("n_blocks", 2)))
    rope_cfg = _rope_from_cfg(mcfg)

    model = JointDiT(
        d_model=d_model,
        heads=heads,
        ff_mult=ff_mult,
        dropout=dropout,
        rope_cfg=rope_cfg,
        joint_blocks=jb,
        video_in_ch=int(mcfg.get("video_in_ch", 4)),
        audio_in_ch=int(mcfg.get("audio_in_ch", 8)),
    ).to(
        device=device, dtype=torch.float32
    )  # keep params fp32
    return model


def _extract_meta_list(batch):
    # best-effort to recover per-sample dicts
    if "meta_list" in batch and isinstance(batch["meta_list"], (list, tuple)):
        return list(batch["meta_list"])
    if "meta" in batch and isinstance(batch["meta"], (list, tuple)):
        return list(batch["meta"])
    if "meta" in batch and isinstance(batch["meta"], dict):
        return [batch["meta"]]
    if "ref_meta" in batch and isinstance(batch["ref_meta"], (list, tuple, str)):
        paths = (
            batch["ref_meta"]
            if isinstance(batch["ref_meta"], (list, tuple))
            else [batch["ref_meta"]]
        )
        out = []
        for p in paths:
            try:
                out.append(
                    yaml.safe_load(Path(p).read_text())
                    if str(p).endswith((".yml", ".yaml"))
                    else __import__("json").load(open(p, "r"))
                )
            except Exception:
                out.append({})
        return out
    return [{}] * (batch["video_latents"].shape[0] if "video_latents" in batch else 1)


# --------------- main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-suffix", default="")
    ap.add_argument("--ckpt-suffix", default="")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = cfg["runtime"]["device"]
    set_seed(int(cfg["runtime"]["seed"]))

    # dirs
    out_cfg = cfg["out"]
    ckpt_dir = Path(out_cfg["ckpt_dir"] + (f"_{args.ckpt_suffix}" if args.ckpt_suffix else ""))
    log_dir = Path(out_cfg["log_dir"] + (f"_{args.log_suffix}" if args.log_suffix else ""))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    # data
    ds = JointDiTDataset(cfg["data"]["cache_dir"], cfg["data"]["train_split"])
    dl = DataLoader(
        ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=True,
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=cfg["loader"]["pin_memory"],
        drop_last=True,
        collate_fn=collate_jointdit,
    )

    # model
    model = build_model(cfg, device)

    # CLIP cond (optional)
    clip_cfg = cfg.get("clip", {})
    clip_on = bool(clip_cfg.get("enabled", False))
    drop_txt = float(clip_cfg.get("dropout_prob", 0.15))
    use_img = bool(clip_cfg.get("use_image_firstframe", False))
    clip_enc = None
    if clip_on:
        clip_enc = CLIPCondEncoder(
            d_model=cfg["model"]["d_model"],
            clip_name=clip_cfg.get("model", "ViT-B-16"),
            pretrained=clip_cfg.get("pretrained", "laion2b_s34b_b79k"),
            device=device,
        )

    # optimizer setup
    main_params = [p for p in model.parameters() if p.requires_grad]
    optim_cfg = cfg["optim"]
    optim = torch.optim.AdamW(
        main_params,
        lr=float(optim_cfg["lr_main"]),
        betas=tuple(optim_cfg["betas"]),
        weight_decay=float(optim_cfg["weight_decay"]),
    )
    amp_enabled = bool(optim_cfg.get("amp", str(cfg["runtime"]["dtype"]).lower() == "fp16"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # resume
    start_step = 0
    if args.resume:
        ckpt_path = latest_ckpt(ckpt_dir) if args.resume == "auto" else Path(args.resume)
        if ckpt_path and ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ck["model"], strict=False)
            optim.load_state_dict(ck["optim"])
            if "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            start_step = int(ck.get("step", 0))
            print(f"[resume] {ckpt_path} @ step {start_step}")

    # knobs
    max_steps = args.max_steps if args.max_steps is not None else int(optim_cfg["max_steps"])
    grad_accum = int(optim_cfg["grad_accum"])
    grad_clip = float(optim_cfg.get("grad_clip", 0.0))
    log_every = int(optim_cfg["log_every"])
    ckpt_every = int(optim_cfg["ckpt_every"])

    sched = cfg["schedule"]
    lam_v = float(cfg["loss"]["lambda_v"])
    lam_a = float(cfg["loss"]["lambda_a"])

    step = start_step
    model.train()
    it = iter(dl)
    t0 = time.time()
    tgt_dtype = torch.float16 if str(cfg["runtime"]["dtype"]).lower() == "fp16" else torch.float32

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        v0 = _get_any(batch, ["video_latents", "v_lat", "v_latents", "video"])
        a0 = _get_any(batch, ["audio_latents", "a_lat", "a_latents", "audio"])
        if v0 is None or a0 is None:
            raise KeyError(f"Missing latents; keys={list(batch.keys())}")

        if v0.dim() == 4:
            v0 = v0.unsqueeze(0)
        if a0.dim() == 5 and a0.shape[1] == 1:
            a0 = a0[:, 0]
        if a0.dim() == 4 and v0.shape[0] > 1 and a0.shape[0] == 1:
            a0 = a0.expand(v0.shape[0], *a0.shape[1:])

        max_T = int(os.environ.get("JOINTDIT_MAX_T", "0"))
        if max_T > 0 and v0.shape[1] > max_T:
            v0 = v0[:, :max_T]

        v0 = v0.to(device=device, dtype=tgt_dtype, non_blocking=True)
        a0 = a0.to(device=device, dtype=tgt_dtype, non_blocking=True)

        t_uni = torch.rand(v0.shape[0], device=device, dtype=tgt_dtype)
        sigma_v = edm_sigma_from_tuni(
            t_uni,
            sched["edm_P_mean"],
            sched["edm_P_std"],
            sched["sigma_video_min"],
            sched["sigma_video_max"],
        )
        sigma_a = ddpm_like_sigma_from_tuni(
            t_uni, sched["sigma_audio_min"], sched["sigma_audio_max"]
        )

        v_t, _ = add_gaussian_noise(v0, sigma_v)
        a_t, _ = add_gaussian_noise(a0, sigma_a)

        # CLIP tokens
        t_ctx = i_ctx = None
        if clip_on:
            metas = _extract_meta_list(batch)
            prompts = [(m.get("caption") or "") for m in metas]
            images = [(m.get("img_firstframe") if use_img else None) for m in metas]
            t_ctx, i_ctx = clip_enc(prompts, images, device=device)
            # text CFG dropout
            if t_ctx is not None and drop_txt > 0:
                mask = (torch.rand(t_ctx.shape[0], device=t_ctx.device) < drop_txt).view(-1, 1, 1)
                t_ctx = torch.where(mask, torch.zeros_like(t_ctx), t_ctx)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            v_hat, a_hat = model(v_t, a_t, mode="full", t_ctx=t_ctx, i_ctx=i_ctx)
            loss_v = mse_x0(v_hat, v0)
            loss_a = mse_x0(a_hat, a0)
            loss = lam_v * loss_v + lam_a * loss_a

        if has_nan_or_inf(loss):
            print("[warn] NaN/Inf; skip")
            optim.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            if grad_clip > 0:
                if amp_enabled:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if amp_enabled:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            dt = time.time() - t0
            writer.add_scalar("loss/total", float(loss.detach().cpu()), step)
            writer.add_scalar("loss/video", float(loss_v.detach().cpu()), step)
            writer.add_scalar("loss/audio", float(loss_a.detach().cpu()), step)
            print(
                f"[step {step:05d}] loss={loss.item():.4f} v={loss_v.item():.4f} a={loss_a.item():.4f} ({dt:.1f}s)"
            )

        if step > 0 and step % ckpt_every == 0:
            ck = {
                "step": step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else {},
                "cfg": cfg,
            }
            outp = ckpt_dir / f"ckpt_step_{step:06d}.pt"
            torch.save(ck, outp)
            print(f"[ckpt] {outp}")

        step += 1

    ck = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if amp_enabled else {},
        "cfg": cfg,
    }
    outp = ckpt_dir / f"ckpt_step_{step:06d}.pt"
    torch.save(ck, outp)
    print(f"[done] {outp}")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
