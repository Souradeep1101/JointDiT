from collections import Counter
from pathlib import Path

from safetensors import safe_open

st_path = Path(
    "/workspace/jointdit/assets/models/audioldm2/unet/diffusion_pytorch_model.safetensors"
)
vals = []
with safe_open(str(st_path), framework="pt", device="cpu") as f:
    for k in f.keys():
        # cross-attention projections use attn2.*
        if (
            k.endswith("attn2.to_k.weight")
            or k.endswith("attn2.to_q.weight")
            or k.endswith("attn2.to_v.weight")
        ):
            w = f.get_tensor(k)
            if w.ndim == 2:
                vals.append(int(w.shape[1]))  # in_features
from pprint import pprint

cnt = Counter(vals)
print("cross_attention_dim in_features histogram:")
pprint(cnt)
if cnt:
    print("distinct values:", sorted(cnt.keys()))
else:
    print("No attn2.to_* weights found")
