import torch
from diffusers import AutoencoderKL

ROOT = "/workspace/jointdit/assets/models/audioldm2"
vae = (
    AutoencoderKL.from_pretrained(ROOT, subfolder="vae", torch_dtype=torch.float16)
    .eval()
    .to("cuda")
)

B, C_lat, hL, wL = 1, int(vae.config.latent_channels), 7, 11  # odd sizes to be sure
z = torch.randn(B, C_lat, hL, wL, device="cuda", dtype=torch.float16)
with torch.no_grad():
    x = vae.decode(z).sample
print("latent:", tuple(z.shape), "-> decode:", tuple(x.shape))
sf_h = x.shape[-2] // hL
sf_w = x.shape[-1] // wL
print(f"scale_h={sf_h}, scale_w={sf_w}, unified_scale={sf_h if sf_h==sf_w else (sf_h, sf_w)}")
