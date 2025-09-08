from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Prefer the spatio-temporal UNet (SVD); fall back to 2D if unavailable (offline setups)
_UNET_ST = None
try:
    from diffusers import UNetSpatioTemporalConditionModel as _UNET_ST
except Exception:
    _UNET_ST = None

_UNET_2D = None
try:
    from diffusers import UNet2DConditionModel as _UNET_2D
except Exception:
    _UNET_2D = None

from .adapt_layers import LazyAdaLN


def _is_dir_with_config(p: str) -> bool:
    import os

    return os.path.isdir(p) and (
        os.path.exists(f"{p}/config.json")
        or os.path.exists(f"{p}/diffusion_pytorch_model.safetensors")
    )


def _load_svd_unet(model_id_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    kwargs = dict(torch_dtype=dtype)
    if _UNET_ST is not None:
        try:
            # HF repo root -> use subfolder, else a local folder that directly contains config.json
            if _is_dir_with_config(model_id_or_path):
                model = _UNET_ST.from_pretrained(model_id_or_path, **kwargs)
            else:
                model = _UNET_ST.from_pretrained(model_id_or_path, subfolder="unet", **kwargs)
            return model.to(device)
        except Exception as e:
            print(f"[warn][SVD slicer] SpatioTemporal UNet load failed: {e}")
    # Fallback (rare) to 2D UNet just to keep the scaffolding alive
    if _UNET_2D is None:
        raise RuntimeError("Could not import any UNet class suitable for SVD.")
    if _is_dir_with_config(model_id_or_path):
        model = _UNET_2D.from_pretrained(model_id_or_path, **kwargs)
    else:
        model = _UNET_2D.from_pretrained(model_id_or_path, subfolder="unet", **kwargs)
    return model.to(device)


class SVDUNetSlicer(nn.Module):
    """
    Day-3 scaffold around SVD UNet.
    We *load* the real UNet, but our per-block forwards are no-ops except AdaLN.
    Tomorrow (Day 4) we'll thread true per-block execution and Joint Blocks.
    """

    def __init__(self, unet_id_or_path: str, device="cuda", dtype=torch.float16):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.unet = _load_svd_unet(unet_id_or_path, device=device, dtype=dtype)

        # 6 logical groups (1 | 2,3 | 4,5,6)
        self.adaln1 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln2 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln3 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln4 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln5 = LazyAdaLN(dtype=dtype, device=device)
        self.adaln6 = LazyAdaLN(dtype=dtype, device=device)

    # ----- no-op “sliced” forwards (shapes preserved) -----
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

    # ----- helpers -----
    def param_counts(self):
        def count(p):
            return sum(i.numel() for i in p if i.requires_grad)

        groups = {
            "adaln1": count(self.adaln1.parameters()),
            "adaln2": count(self.adaln2.parameters()),
            "adaln3": count(self.adaln3.parameters()),
            "adaln4": count(self.adaln4.parameters()),
            "adaln5": count(self.adaln5.parameters()),
            "adaln6": count(self.adaln6.parameters()),
            "unet_total": count(self.unet.parameters()),
        }
        return groups
