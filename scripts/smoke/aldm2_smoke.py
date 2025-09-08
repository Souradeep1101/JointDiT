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
unet.set_attn_processor(AttnProcessor())

vae = (
    AutoencoderKL.from_pretrained(
        ROOT, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    .to("cuda")
    .eval()
)

# Probe true scale
B, C_lat = 1, int(vae.config.latent_channels)
hL, wL = 10, 24
z = torch.randn(B, C_lat, hL, wL, device="cuda", dtype=torch.float16)
with torch.no_grad():
    x = vae.decode(z).sample
sf = x.shape[-1] // wL
print(f"VAE scale = {sf} (decoded {x.shape} from latent {z.shape})")

# Choose latent H/W that keep memory sane
H, W = 80, 192  # latent spatial resolution (not decoded!)
E = int(unet.config.cross_attention_dim)

x_lat = torch.randn(B, unet.config.in_channels, H, W, device="cuda", dtype=torch.float16)
t = torch.tensor([500], device="cuda", dtype=torch.float32)
eh = torch.randn(B, 77, E, device="cuda", dtype=torch.float16)

with torch.no_grad():
    y_lat = unet(x_lat, t, encoder_hidden_states=eh).sample
print("unet_out (latent):", tuple(y_lat.shape))

# Decode the UNet latent directly
with torch.no_grad():
    x_rec = vae.decode(y_lat).sample
print("decoded:", tuple(x_rec.shape), "(expected ~", (B, 1, H * sf, W * sf), ")")
