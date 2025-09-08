import json
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path("/workspace/jointdit/assets/models/audioldm2/unet")
CFG = ROOT / "config.json"
ST = ROOT / "diffusion_pytorch_model.safetensors"
BAK = ROOT / "diffusion_pytorch_model.safetensors.bak"


def pad_or_truncate_cols(W: torch.Tensor, target_in: int):
    out, in_ = W.shape
    if in_ == target_in:
        return W
    if in_ > target_in:
        return W[:, :target_in].contiguous()
    # pad zeros on the right
    return torch.cat([W, torch.zeros(out, target_in - in_, dtype=W.dtype)], dim=1).contiguous()


def main():
    cfg = json.load(open(CFG))
    # Build an uninitialized model on CPU from config (no weights)
    model = UNet2DConditionModel.from_config(cfg)

    # Map expected in_features for every attn2.{to_q,to_k,to_v}.weight
    expected_in = {}
    for name, mod in model.named_modules():
        if (
            name.endswith("attn2.to_q")
            or name.endswith("attn2.to_k")
            or name.endswith("attn2.to_v")
        ):
            if hasattr(mod, "in_features"):
                expected_in[f"{name}.weight"] = int(mod.in_features)

    if not expected_in:
        raise RuntimeError("Could not find any attn2.{to_q,to_k,to_v} linear layers in the UNet.")

    # Load checkpoint tensors and patch where needed
    new_tensors = {}
    changes = defaultdict(int)
    with safe_open(str(ST), framework="pt", device="cpu") as f:
        for k in f.keys():
            T = f.get_tensor(k)
            if k in expected_in and T.ndim == 2:
                tgt = expected_in[k]
                T2 = pad_or_truncate_cols(T, tgt)
                if T2.shape != T.shape:
                    changes[k.split(".")[-2]] += 1  # counts by to_q/to_k/to_v
                new_tensors[k] = T2
            else:
                new_tensors[k] = T

    if changes:
        print("[reshape] changed columns for:", dict(changes))
    else:
        print("[reshape] no q/k/v column changes were required (already matched).")

    # Backup then write
    if not BAK.exists():
        ST.replace(BAK)
        print(f"[backup] wrote {BAK}")
    else:
        ST.unlink(missing_ok=True)
    save_file(new_tensors, str(ST))
    print(f"[save] wrote reshaped weights -> {ST}")


if __name__ == "__main__":
    main()
