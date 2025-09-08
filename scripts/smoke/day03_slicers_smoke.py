import glob
import json

import torch
import yaml

from models.aldm_unet_slicer import ALDM2UNetSlicer
from models.svd_unet_slicer import SVDUNetSlicer


def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def pick_shapes_from_cache(glob_pat):
    metas = sorted(glob.glob(glob_pat))
    if not metas:
        # default tiny shapes if no cache present
        return (12, 4, 40, 53), (1, 8, 20, 15)
    with open(metas[0], "r") as f:
        m = json.load(f)
    v = torch.load(m["video_latents"], map_location="cpu", weights_only=False)
    a = torch.load(m["audio_latents"], map_location="cpu", weights_only=False)
    return tuple(v["latents"].shape), tuple(a["latents"].shape)


def describe(name, counts):
    ks = ["adaln1", "adaln2", "adaln3", "adaln4", "adaln5", "adaln6", "unet_total"]
    msg = f"[{name}] params:"
    for k in ks:
        if k in counts:
            msg += f" {k}={counts[k]/1e6:.2f}M"
    return msg


def main():
    cfg = load_cfg("configs/day03_models.yaml")
    device = cfg["runtime"]["device"]
    dtype = torch.float16 if str(cfg["runtime"]["dtype"]).lower() == "fp16" else torch.float32

    v_shape, a_shape = pick_shapes_from_cache(cfg.get("cache_meta_glob", ""))
    print(f"[shapes] video={v_shape}  audio={a_shape}")

    # dummy latents
    v_lat = torch.randn(*v_shape, device=device, dtype=dtype)
    a_lat = torch.randn(*a_shape, device=device, dtype=dtype)

    # dummy timesteps
    t_v = torch.tensor([250], device=device, dtype=torch.long)
    t_a = torch.tensor([250], device=device, dtype=torch.long)

    # load slicers
    svd = SVDUNetSlicer(cfg["svd"]["unet_path"], device=device, dtype=dtype)
    aldm = ALDM2UNetSlicer(cfg["audio"]["unet_path"], device=device, dtype=dtype)

    print(describe("SVD", svd.param_counts()))
    print(describe("ALDM2", aldm.param_counts()))

    # ----- SVD path -----
    x = v_lat
    x1 = svd.forward_input(x, t_v, cond=None)
    x2 = svd.forward_expert(x1, t_v, cond=None, idx=2)
    x3 = svd.forward_expert(x2, t_v, cond=None, idx=3)
    xo = svd.forward_output(x3, t_v, cond=None)
    print(f"[SVD] input→{tuple(x1.shape)} →{tuple(x2.shape)} →{tuple(x3.shape)} →{tuple(xo.shape)}")
    assert xo.shape == v_lat.shape, "SVD shapes changed in scaffold pass"

    # ----- ALDM2 path -----
    y = a_lat
    y1 = aldm.forward_input(y, t_a, cond=None)
    y2 = aldm.forward_expert(y1, t_a, cond=None, idx=2)
    y3 = aldm.forward_expert(y2, t_a, cond=None, idx=3)
    yo = aldm.forward_output(y3, t_a, cond=None)
    print(
        f"[ALDM2] input→{tuple(y1.shape)} →{tuple(y2.shape)} →{tuple(y3.shape)} →{tuple(yo.shape)}"
    )
    assert yo.shape == a_lat.shape, "ALDM2 shapes changed in scaffold pass"

    print("\n[OK] Day-3 slicer scaffold is healthy (shapes preserved, params present).")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
