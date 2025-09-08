#!/usr/bin/env python
import argparse

from data_loader.jointdit_dataset import JointDiTDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="/workspace/jointdit/data/cache")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    ds = JointDiTDataset(args.cache_root, args.split)
    print(f"[dataset] {len(ds)} items in split={args.split}")

    for i in range(min(args.n, len(ds))):
        ex = ds[i]
        v = ex["v_latents"]
        a = ex["a_latents"]
        ce = ex["clip_emb"]
        print(
            f"- {i}: frames={v.shape[0]} v_lat={tuple(v.shape)}  a_lat={tuple(a.shape)}  clip={'yes' if ce is not None else 'no'}"
        )
        print(f"  img_firstframe={ex['img_firstframe_path']}")


if __name__ == "__main__":
    main()
