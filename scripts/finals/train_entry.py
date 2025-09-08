#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["A", "B"], default="A")
    p.add_argument("--cfg", required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--ckpt-suffix", default="")
    p.add_argument("--log-suffix", default="")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    entry = repo / (
        "scripts/train/train_stage_a.py" if args.stage == "A" else "scripts/train/train_stage_b.py"
    )
    cmd = [sys.executable, "-u", str(entry), "--cfg", args.cfg]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.ckpt_suffix:
        cmd += ["--ckpt-suffix", args.ckpt_suffix]
    if args.log_suffix:
        cmd += ["--log-suffix", args.log_suffix]
    if args.resume:
        cmd += ["--resume", args.resume]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("JOINTDIT_Q_CHUNK_V", "128")
    os.environ.setdefault("JOINTDIT_Q_CHUNK_A", "0")
    os.environ.setdefault("JOINTDIT_KV_DOWNSAMPLE", "4")

    print("[train_entry] running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
