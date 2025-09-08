import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
ROOT = "/workspace/jointdit/assets/models/audioldm2"

unet = (
    UNet2DConditionModel.from_pretrained(
        ROOT, subfolder="unet", torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    .to("cuda")
    .eval()
)
unet.set_attn_processor(AttnProcessor())  # prefer classic attention on your nightly

vae = (
    AutoencoderKL.from_pretrained(
        ROOT, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    .to("cuda")
    .eval()
)

print("unet.in_channels:", unet.config.in_channels)
print("unet.cross_attention_dim:", unet.config.cross_attention_dim)

# infer VAE spatial scale (×2, ×4, ...)
factor = 1
dbts = getattr(vae.config, "down_block_types", None)
bocs = getattr(vae.config, "block_out_channels", None)
if dbts:
    for name in dbts:
        if "Down" in name:
            factor *= 2
elif bocs:
    factor = 2 ** max(0, len(bocs) - 1)

C_lat = int(getattr(vae.config, "latent_channels", 4))
print("vae.latent_channels:", C_lat, "vae.scale:", factor)

B, H, W = 1, 80, 192
x = torch.randn(B, unet.config.in_channels, H, W, device="cuda", dtype=torch.float16)
t = torch.tensor([500], device="cuda", dtype=torch.float32)
E = int(unet.config.cross_attention_dim)
eh = torch.randn(B, 77, E, device="cuda", dtype=torch.float16)

with torch.no_grad():
    y = unet(x, t, encoder_hidden_states=eh).sample
print("unet_out:", tuple(y.shape))

hL, wL = max(1, H // factor), max(1, W // factor)
latent = torch.randn(B, C_lat, hL, wL, device="cuda", dtype=torch.float16)
with torch.no_grad():
    rec = vae.decode(latent).sample
print("vae_decode:", tuple(rec.shape))
