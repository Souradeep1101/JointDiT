# /workspace/jointdit/models/aldm_unet_slicer.py
import collections
import os
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

try:
    from diffusers import UNet2DConditionModel as _UNET_2D
except Exception:
    _UNET_2D = None

from safetensors.torch import load_file as safe_load

from .adapt_layers import LazyAdaLN


def _is_dir_with_weights(p: str) -> bool:
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, "config.json"))
        and os.path.exists(os.path.join(p, "diffusion_pytorch_model.safetensors"))
    )


def _flatten_ints(x) -> Iterable[int]:
    if isinstance(x, int):
        yield x
    elif isinstance(x, (list, tuple, set)):
        for y in x:
            yield from _flatten_ints(y)


def _choose_single_dim(vals: Iterable[int]) -> int:
    vals = list(vals)
    if not vals:
        return 1024
    freq = collections.Counter(vals)
    [(mode_val, _)] = freq.most_common(1)
    return int(mode_val)


def _load_aldm_unet(
    model_id_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16
):
    if _UNET_2D is None:
        raise RuntimeError("diffusers UNet2DConditionModel not available.")

    # --- Preferred: if we have a local folder with weights, do partial real-weights load
    if _is_dir_with_weights(model_id_or_path):
        cfg = _UNET_2D.load_config(model_id_or_path)
        xad = cfg.get("cross_attention_dim", 1024)
        if isinstance(xad, (list, tuple, set)):
            chosen = _choose_single_dim(_flatten_ints(xad))
            print(f"[patch][ALDM2] normalized cross_attention_dim -> {chosen}")
            cfg["cross_attention_dim"] = int(chosen)

        model = _UNET_2D.from_config(cfg)
        model.to(device, dtype=dtype)

        # load safetensors and filter by matching shape
        wt_path = os.path.join(model_id_or_path, "diffusion_pytorch_model.safetensors")
        sd_src = safe_load(wt_path, device="cpu")
        sd_dst = model.state_dict()
        loadable, mismatched = {}, []
        for k, v in sd_src.items():
            if k in sd_dst and sd_dst[k].shape == v.shape:
                loadable[k] = v
            else:
                mismatched.append(k)
        missing = [k for k in sd_dst.keys() if k not in loadable]

        model.load_state_dict(loadable, strict=False)
        print(
            f"[ALDM2][realweights] loaded {len(loadable)}/{len(sd_dst)} tensors | "
            f"mismatched: {len(mismatched)} | missing: {len(missing)}"
        )
        return model

    # --- Fallbacks: try from_pretrained (remote or local) with ignore_mismatched_sizes
    try:
        model = _UNET_2D.from_pretrained(
            model_id_or_path,
            subfolder="unet" if not os.path.isdir(model_id_or_path) else None,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
        )
        return model.to(device)
    except TypeError as e:
        print(f"[warn][ALDM2] from_pretrained failed (likely cross_attention_dim list): {e}")
        # sanitize config and retry
        cfg = _UNET_2D.load_config(
            model_id_or_path,
            subfolder="unet" if not os.path.isdir(model_id_or_path) else None,
        )
        xad = cfg.get("cross_attention_dim", 1024)
        if isinstance(xad, (list, tuple, set)):
            chosen = _choose_single_dim(_flatten_ints(xad))
            print(f"[patch][ALDM2] normalized cross_attention_dim -> {chosen}")
            cfg["cross_attention_dim"] = int(chosen)
        model = _UNET_2D.from_config(cfg).to(device, dtype=dtype)
        print(
            "[warn][ALDM2] built from sanitized config; weights will be random unless you point to a local folder with safetensors."
        )
        return model


class ALDM2UNetSlicer(nn.Module):
    def __init__(self, unet_id_or_path: str, device="cuda", dtype=torch.float16):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.unet = _load_aldm_unet(unet_id_or_path, device=device, dtype=dtype)
        self.adaln1 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln2 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln3 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln4 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln5 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln6 = LazyAdaLN(dtype=dtype, device=device)

    def forward_input(
        self, x: torch.Tensor, t: torch.Tensor, cond: Optional[Dict[str, Any]] = None
    ):
        return self.adaln1(x)

    def forward_expert(
        self, x: torch.Tensor, t: torch.Tensor, cond: Optional[Dict[str, Any]] = None, idx: int = 2
    ):
        if idx == 2:
            return self.adaln2(x)
        if idx == 3:
            return self.adaln3(x)
        raise ValueError("expert idx must be 2 or 3")

    def forward_output(
        self, x: torch.Tensor, t: torch.Tensor, cond: Optional[Dict[str, Any]] = None
    ):
        x = self.adaln4(x)
        x = self.adaln5(x)
        x = self.adaln6(x)
        return x

    def param_counts(self):
        def count(p):
            return sum(i.numel() for i in p if i.requires_grad)

        return {
            "adaln1": count(self.adaln1.parameters()),
            "adaln2": count(self.adaln2.parameters()),
            "adaln3": count(self.adaln3.parameters()),
            "adaln4": count(self.adaln4.parameters()),
            "adaln5": count(self.adaln5.parameters()),
            "adaln6": count(self.adaln6.parameters()),
            "unet_total": count(self.unet.parameters()),
        }
