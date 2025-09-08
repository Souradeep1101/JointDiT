#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data_loader.jointdit_dataset import JointDiTDataset
from models.jointdit import JointDiT
from models.noise_schedules import mse_x0


def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def latest_ckpt(ckpt_dir: Path) -> Path | None:
    files = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", default="latest", help='"latest" or path')
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = cfg["runtime"]["device"]
    dtype = cfg["runtime"]["dtype"]
    cache_dir = cfg["data"]["cache_dir"]
    val_split = cfg["data"]["val_split"]

    # data
    ds = JointDiTDataset(cache_dir, val_split)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # model
    mcfg = cfg["model"]
    model = JointDiT(
        d_model=mcfg["d_model"],
        heads=mcfg.get("heads", mcfg.get("n_heads", 8)),
        n_blocks=mcfg["n_blocks"],
        ff_mult=mcfg["ff_mult"],
        attn_dropout=mcfg["attn_dropout"],
        ff_dropout=mcfg["ff_dropout"],
        rope_video=mcfg.get("rope_video", True),
        rope_audio=mcfg.get("rope_audio", True),
    ).to(device=device, dtype=(torch.float16 if str(dtype).lower() == "fp16" else torch.float32))

    # load ckpt
    ckpt_dir = Path(cfg["out"]["ckpt_dir"])
    ckpt_path = latest_ckpt(ckpt_dir) if args.ckpt == "latest" else Path(args.ckpt)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found for {args.ckpt}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[val] loaded {ckpt_path}")

    model.eval()
    tot_v = tot_a = n = 0.0
    with torch.no_grad():
        for batch in dl:
            v0 = batch["video_latents"]
            a0 = batch["audio_latents"]
            if v0.ndim == 4:
                v0 = v0.unsqueeze(0)
            if a0.ndim == 4:
                a0 = a0.unsqueeze(0)
            target_dtype = torch.float16 if str(dtype).lower() == "fp16" else torch.float32
            v0 = v0.to(device=device, dtype=target_dtype)
            a0 = a0.to(device=device, dtype=target_dtype)

            v_hat, a_hat = model(v0, a0, mode="full")  # identity-ish baseline
            lv = mse_x0(v_hat, v0).item()
            la = mse_x0(a_hat, a0).item()
            tot_v += lv
            tot_a += la
            n += 1

    print(
        f"[val] avg_loss_video={tot_v/n:.6f}  avg_loss_audio={tot_a/n:.6f} over {int(n)} samples."
    )


if __name__ == "__main__":
    main()
