from diffusers import UNet2DConditionModel
from safetensors import safe_open

ROOT = "/workspace/jointdit/assets/models/audioldm2/unet"
ST = f"{ROOT}/diffusion_pytorch_model.safetensors"

model = UNet2DConditionModel.from_config(f"{ROOT}/config.json")
model_keys = set(k for k, _ in model.state_dict().items())

ckpt_keys = set()
with safe_open(ST, framework="pt", device="cpu") as f:
    for k in f.keys():
        ckpt_keys.add(k)

used = sorted(model_keys & ckpt_keys)
missing = sorted(model_keys - ckpt_keys)
unused = sorted(ckpt_keys - model_keys)


def pct(a, b):
    return 0 if b == 0 else round(100 * len(a) / len(b), 2)


print(f"ckpt tensors: {len(ckpt_keys)}")
print(f"model tensors: {len(model_keys)}")
print(f"used: {len(used)} ({pct(used, model_keys)}% of model)")
print(f"missing (randomly init): {len(missing)}")
print(f"unused in ckpt: {len(unused)}")
