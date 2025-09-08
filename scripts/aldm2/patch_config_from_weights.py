import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open

unet_dir = Path("/workspace/jointdit/assets/models/audioldm2/unet")
cfg_path = unet_dir / "config.json"
st_path = unet_dir / "diffusion_pytorch_model.safetensors"

cfg = json.load(open(cfg_path))

cands = []
with safe_open(str(st_path), framework="pt", device="cpu") as f:
    for k in f.keys():
        # target cross-attn projection keys
        if k.endswith("attn2.to_k.weight"):
            shape = f.get_tensor(k).shape  # (out_features, in_features)
            if len(shape) == 2:
                cands.append(int(shape[1]))

if not cands:
    raise RuntimeError("Could not find any attn2.to_k.weight tensors to infer cross_attention_dim")

# pick the most common in_features across blocks
mode_val = Counter(cands).most_common(1)[0][0]
old = cfg.get("cross_attention_dim", None)
cfg["cross_attention_dim"] = int(mode_val)

print(f"[patch] cross_attention_dim: {old} -> {mode_val} (from weights)")

json.dump(cfg, open(cfg_path, "w"), indent=2)
print("[patch] wrote", cfg_path)
