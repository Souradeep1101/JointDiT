from pathlib import Path

from safetensors import safe_open

st = Path("/workspace/jointdit/assets/models/audioldm2/unet/diffusion_pytorch_model.safetensors")
bad = []
with safe_open(str(st), framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.endswith(("attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight")):
            w = f.get_tensor(k)
            if w.ndim == 2 and w.shape[1] != 1024:
                bad.append((k, tuple(w.shape)))
print("BAD" if bad else "OK", bad)
