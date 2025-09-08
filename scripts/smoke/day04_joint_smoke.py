import glob
import json
import os

import torch
import yaml

from models.jointdit import JointDiT

# (Optional) if you want to try plugging slicers later:
# from models.svd_unet_slicer import SVDUNetSlicer
# from models.aldm_unet_slicer import ALDM2UNetSlicer


def pick_first_meta(cache_root, split):
    metas = sorted(glob.glob(os.path.join(cache_root, "meta", split, "*.json")))
    assert metas, f"No meta json under {cache_root}/meta/{split}"
    return metas[0]


def load_latents(meta_path):
    meta = json.load(open(meta_path, "r"))
    v = torch.load(meta["video_latents"], map_location="cpu")["latents"]  # (T,Cv,Hv,Wv)
    a = torch.load(meta["audio_latents"], map_location="cpu")["latents"]  # (1,Ca,Ha,Wa)
    # add batch
    v = v.unsqueeze(0)  # (B=1,T,Cv,Hv,Wv)
    a = a  # already (1,Ca,Ha,Wa)
    return v, a, meta


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    cfg = yaml.safe_load(open("configs/day04_joint.yaml", "r"))
    device = cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    use_fp16 = str(cfg["runtime"]["dtype"]).lower() == "fp16" and device.startswith("cuda")

    meta_path = pick_first_meta(cfg["data"]["cache_root"], cfg["data"]["split"])
    v, a, meta = load_latents(meta_path)
    B, T, Cv, Hv, Wv = v.shape
    B2, Ca, Ha, Wa = a.shape
    print(f"[shapes] v={tuple(v.shape)} a={tuple(a.shape)}")

    d_model = int(cfg["model"]["d_model"])
    heads = int(cfg["model"]["heads"])
    ff_mult = int(cfg["model"]["ff_mult"])
    dropout = float(cfg["model"]["dropout"])
    rope_cfg = cfg["model"]["rope"]
    n_blks = int(cfg["model"]["joint_blocks"])

    # Build the model (no slicers for smoke; shape-preserving skeleton)
    joint = JointDiT(
        d_model=d_model,
        heads=heads,
        ff_mult=ff_mult,
        dropout=dropout,
        rope_cfg=rope_cfg,
        video_in_ch=Cv,
        audio_in_ch=Ca,
        joint_blocks=n_blks,
        svd_slicer=None,
        aldm_slicer=None,
    ).to(device)
    # match model dtype to inputs when fp16 is requested
    if use_fp16 and device.startswith("cuda"):
        joint = joint.half()
    joint.eval()
    print(
        f"[params] JointDiT={count_params(joint)/1e6:.2f}M  blocks={n_blks} d_model={d_model} heads={heads}"
    )

    v = v.to(device)
    a = a.to(device)
    if use_fp16:
        v = v.half()
        a = a.half()

    modes = cfg.get("modes_for_smoke", ["full", "iso_v", "iso_a"])
    with torch.no_grad():
        for m in modes:
            v_out, a_out = joint(v, a, mode=m)
            assert (
                v_out.shape == v.shape
            ), f"video shape mismatch in mode={m}: {v_out.shape} vs {v.shape}"
            assert (
                a_out.shape == a.shape
            ), f"audio shape mismatch in mode={m}: {a_out.shape} vs {a.shape}"
            bad = (
                torch.isnan(v_out).any()
                or torch.isinf(v_out).any()
                or torch.isnan(a_out).any()
                or torch.isinf(a_out).any()
            )
            print(f"[mode={m}] v_out={tuple(v_out.shape)} a_out={tuple(a_out.shape)} nan/inf={bad}")

    print("\n[OK] Day-4 JointDiT skeleton is healthy (modes passed, shapes preserved).")


if __name__ == "__main__":
    main()
