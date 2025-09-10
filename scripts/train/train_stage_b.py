#!/usr/bin/env python
# scripts/train/train_stage_b.py
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add repo root
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

# ==========================
# Utilities
# ==========================


def load_cfg(p: str):
    return yaml.safe_load(Path(p).read_text())


def latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
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


def _get_any(d, names):
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


def freeze_all_unfreeze_experts(model: JointDiT, blocks=(2, 3), unfreeze_io=True):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze selected joint blocks
    if hasattr(model, "blocks"):
        for i in blocks:
            if 0 <= i < len(model.blocks):
                for p in model.blocks[i].parameters():
                    p.requires_grad = True
    # optionally unfreeze in/out projections
    if unfreeze_io:
        for name in ("v_inproj", "a_inproj", "v_outproj", "a_outproj"):
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad = True


def group_params_for_stageB(model: JointDiT, lr_expert, lr_io, lr_fallback):
    groups, expert, io, other = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".blocks." in n:
            expert.append(p)
        elif any(k in n for k in ("_inproj", "_outproj")):
            io.append(p)
        else:
            other.append(p)
    if expert:
        groups.append({"params": expert, "lr": float(lr_expert)})
    if io:
        groups.append({"params": io, "lr": float(lr_io)})
    if other:
        groups.append({"params": other, "lr": float(lr_fallback)})
    return groups


def _extract_meta_list(batch):
    if "meta_list" in batch and isinstance(batch["meta_list"], (list, tuple)):
        return list(batch["meta_list"])
    if "meta" in batch and isinstance(batch["meta"], (list, tuple)):
        return list(batch["meta"])
    if "meta" in batch and isinstance(batch["meta"], dict):
        return [batch["meta"]]
    if "ref_meta" in batch:
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
    return [{}]


# ==========================
# Stage-A init helpers
# ==========================


def _resolve_init_from(args, cfg) -> Optional[Path]:
    """
    Returns a Stage-A checkpoint file (.pt), or None if scratch is allowed.
    Search order:
      1) --init-from (file or dir). "auto" means search common dirs.
      2) Env JOINTDIT_INIT_FROM
      3) cfg['stageB']['init_from'] if present
      4) common defaults: checkpoints/day05_stage_a* (and /workspace/... variants)
    """
    allow_scratch = bool(args.allow_scratch or os.getenv("ALLOW_STAGEB_SCRATCH", "0") == "1")
    choice = (args.init_from or "").strip()

    if not choice:
        choice = os.getenv("JOINTDIT_INIT_FROM", "").strip()
    if not choice:
        choice = str(cfg.get("stageB", {}).get("init_from", "")).strip()

    candidates: list[Path] = []
    if choice and choice.lower() != "auto" and choice.lower() != "none":
        candidates.append(Path(choice))
    elif (not choice) or choice.lower() == "auto":
        # AUTO mode: check a few likely locations
        candidates.extend(
            [
                Path("checkpoints/day05_stage_a_finalA"),
                Path("checkpoints/day05_stage_a"),
                Path("/workspace/jointdit/checkpoints/day05_stage_a_finalA"),
                Path("/workspace/jointdit/checkpoints/day05_stage_a"),
            ]
        )

    # Expand: if any candidate is a dir, pick latest ckpt in it
    for c in candidates:
        if c.suffix == ".pt" and c.exists():
            return c
        if c.exists() and c.is_dir():
            ck = latest_ckpt(c)
            if ck:
                return ck

    if allow_scratch:
        print(
            "[stageB][warn] No Stage-A checkpoint found; proceeding from scratch because allow_scratch=True"
        )
        return None

    raise FileNotFoundError(
        "[stageB][err] Could not locate a Stage-A checkpoint.\n"
        "Try one of:\n"
        "  • pass --init-from /path/to/ckpt_step_xxxxxx.pt\n"
        "  • pass --init-from /path/to/day05_stage_a (dir with ckpt_step_*.pt)\n"
        "  • export JOINTDIT_INIT_FROM=/path/to/ckpt_or_dir\n"
        "  • set --allow-scratch if you really want to train Stage-B from scratch (not recommended)"
    )


def _infer_stageA_dims_from_ckpt(sd: dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Best-effort inference of (d_model, ff_mult) from checkpoint weights.
    d_model inferred from v_inproj.weight shape [d_model, video_in_ch] (if present) or layernorm sizes.
    ff_mult inferred from feed-forward shapes if present.
    """
    d_model = None
    ff_mult = None

    # Try projection layers first
    win = sd.get("v_inproj.weight", None)
    if win is None:
        # sometimes modules are prefixed
        for k in sd.keys():
            if k.endswith("v_inproj.weight"):
                win = sd[k]
                break
    if win is not None:
        d_model = win.shape[0]

    # Try layernorm size as fallback
    if d_model is None:
        for k, v in sd.items():
            if k.endswith("ln_v1.weight") or k.endswith("ln_a1.weight"):
                d_model = int(v.shape[0])
                break

    # Infer ff_mult from first block FFN shapes: ff.net.0.proj (hidden, d_model)
    ff0 = None
    for k, v in sd.items():
        if ".ff_v.net.0.proj.weight" in k:
            ff0 = v
            break
    if ff0 is not None and d_model is not None:
        # ff0 shape is [hidden, d_model]
        hidden = int(ff0.shape[0])
        ff_mult = hidden // d_model if d_model > 0 else None

    return d_model, ff_mult


def _load_stageA_weights(model: JointDiT, ckpt_path: Path, stageB_cfg: dict):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck

    # Sanity: check width & ff_mult compatibility and warn/fail clearly
    want_d_model = int(stageB_cfg["model"]["d_model"])
    want_ff_mult = int(stageB_cfg["model"].get("ff_mult", 4))
    got_d_model, got_ff_mult = _infer_stageA_dims_from_ckpt(sd)

    if got_d_model is not None and got_d_model != want_d_model:
        raise RuntimeError(
            f"[stageB][shape-mismatch] Stage-A d_model={got_d_model} but Stage-B config d_model={want_d_model}. "
            f"Set configs/day07_trainB.yaml model.d_model={got_d_model} to match Stage-A."
        )
    if got_ff_mult is not None and got_ff_mult != want_ff_mult:
        raise RuntimeError(
            f"[stageB][shape-mismatch] Stage-A ff_mult={got_ff_mult} but Stage-B config ff_mult={want_ff_mult}. "
            f"Set configs/day07_trainB.yaml model.ff_mult={got_ff_mult} to match Stage-A."
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    m = (
        len(getattr(missing, "missing_keys", missing))
        if isinstance(missing, (list, tuple))
        else len(missing)
    )
    u = (
        len(getattr(unexpected, "unexpected_keys", unexpected))
        if isinstance(unexpected, (list, tuple))
        else len(unexpected)
    )
    print(f"[stageB] init-from: {ckpt_path}")
    print(f"[stageB] state_dict load: missing={m} unexpected={u}")
    return ck


# ==========================
# Main
# ==========================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--resume", default="")  # resume Stage-B run (file or "auto")
    ap.add_argument("--init-from", default="auto")  # Stage-A warm start (file/dir/auto/none)
    ap.add_argument(
        "--allow-scratch", action="store_true", help="Allow Stage-B to start from random init"
    )
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-suffix", default="")
    ap.add_argument("--ckpt-suffix", default="")
    args = ap.parse_args()

    # Normalize string args that might include accidental spaces
    args.init_from = (args.init_from or "").strip()
    args.resume = (args.resume or "").strip()

    cfg = load_cfg(args.cfg)
    device = torch.device(cfg["runtime"]["device"])
    set_seed(int(cfg["runtime"]["seed"]))

    # Dirs
    out_cfg = cfg["out"]
    ckpt_dir = Path(out_cfg["ckpt_dir"] + (f"_{args.ckpt_suffix}" if args.ckpt_suffix else ""))
    log_dir = Path(out_cfg["log_dir"] + (f"_{args.log_suffix}" if args.log_suffix else ""))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    # Data
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

    # Model
    model = build_model(cfg, device)

    # ===== Resume or Stage-A init =====
    resume_ck = None
    start_step = 0

    if args.resume:
        ckpt_path = latest_ckpt(ckpt_dir) if args.resume == "auto" else Path(args.resume)
        if not ckpt_path or not ckpt_path.exists():
            raise FileNotFoundError(f"[resume] requested but not found: {ckpt_path}")
        resume_ck = torch.load(ckpt_path, map_location="cpu")
        # Load model weights from Stage-B checkpoint
        model.load_state_dict(resume_ck["model"], strict=False)
        start_step = int(resume_ck.get("step", 0))
        print(f"[resume] loaded Stage-B checkpoint: {ckpt_path} @ step={start_step}")
    else:
        if args.init_from.lower() != "none":
            init_ck = _resolve_init_from(args, cfg)
            if init_ck is not None:
                _load_stageA_weights(model, init_ck, cfg)
        elif not args.allow_scratch:
            raise RuntimeError("Stage-B requires --init-from (or --allow-scratch to bypass).")

    # Stage-B freezing
    sb = cfg.get("stageB", {})
    freeze_all_unfreeze_experts(
        model,
        blocks=tuple(sb.get("unfreeze_blocks", [2, 3])),
        unfreeze_io=bool(sb.get("unfreeze_io", True)),
    )

    # Optimizer + AMP
    optim_cfg = cfg["optim"]
    param_groups = group_params_for_stageB(
        model,
        lr_expert=sb.get("lr_expert", optim_cfg["lr_main"]),
        lr_io=sb.get("lr_io", optim_cfg["lr_main"]),
        lr_fallback=optim_cfg["lr_main"],
    )
    optim = torch.optim.AdamW(
        param_groups,
        betas=tuple(optim_cfg["betas"]),
        weight_decay=float(optim_cfg["weight_decay"]),
    )
    amp_enabled = bool(optim_cfg.get("amp", str(cfg["runtime"]["dtype"]).lower() == "fp16"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # If resuming, now that optimizer/scaler exist, load them
    if resume_ck is not None:
        if "optim" in resume_ck:
            try:
                optim.load_state_dict(resume_ck["optim"])
            except Exception as e:
                print(f"[resume][warn] could not load optimizer state: {e}")
        if "scaler" in resume_ck and resume_ck["scaler"]:
            try:
                scaler.load_state_dict(resume_ck["scaler"])
            except Exception as e:
                print(f"[resume][warn] could not load scaler state: {e}")

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

    # Training knobs
    max_steps = args.max_steps if args.max_steps is not None else int(optim_cfg["max_steps"])
    grad_accum = int(optim_cfg["grad_accum"])
    grad_clip = float(optim_cfg.get("grad_clip", 0.0))
    log_every = int(optim_cfg["log_every"])
    ckpt_every = int(optim_cfg["ckpt_every"])

    sched = cfg["schedule"]
    lam_v = float(cfg["loss"]["lambda_v"])
    lam_a = float(cfg["loss"]["lambda_a"])
    disable_joint = bool(cfg.get("ablation", {}).get("disable_joint", False))
    max_T_env = int(os.environ.get("JOINTDIT_MAX_T", "0"))

    # Loop
    model.train()
    it = iter(dl)
    t0 = time.time()
    tgt_dtype = torch.float16 if str(cfg["runtime"]["dtype"]).lower() == "fp16" else torch.float32

    step = start_step  # <-- continue from resume step

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        v0 = _get_any(batch, ["video_latents", "v_latents", "v_lat", "video"])
        a0 = _get_any(batch, ["audio_latents", "a_latents", "a_lat", "audio"])
        if v0 is None or a0 is None:
            raise KeyError(f"Missing latents; keys={list(batch.keys())}")

        if v0.dim() == 4:
            v0 = v0.unsqueeze(0)  # (1,T,C,H,W)
        if a0.dim() == 5 and a0.shape[1] == 1:
            a0 = a0[:, 0]
        if a0.dim() == 4 and v0.shape[0] > 1 and a0.shape[0] == 1:
            a0 = a0.expand(v0.shape[0], *a0.shape[1:])
        if max_T_env > 0 and v0.shape[1] > max_T_env:
            v0 = v0[:, :max_T_env]

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
            if t_ctx is not None and drop_txt > 0:
                mask = (torch.rand(t_ctx.shape[0], device=t_ctx.device) < drop_txt).view(-1, 1, 1)
                t_ctx = torch.where(mask, torch.zeros_like(t_ctx), t_ctx)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if disable_joint:
                v_hat, _ = model(v_t, a_t, mode="iso_v", t_ctx=t_ctx, i_ctx=i_ctx)
                _, a_hat = model(v_t, a_t, mode="iso_a", t_ctx=t_ctx, i_ctx=i_ctx)
            else:
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
