# /workspace/jointdit/scripts/smoke/svd_smoke.py
import os

import torch
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = "/workspace/jointdit/assets/models/svd"
unet = (
    UNetSpatioTemporalConditionModel.from_pretrained(
        ROOT, subfolder="unet", torch_dtype=torch.float16
    )
    .to(DEV)
    .eval()
)
vae = (
    AutoencoderKLTemporalDecoder.from_pretrained(ROOT, subfolder="vae", torch_dtype=torch.float16)
    .to(DEV)
    .eval()
)

B, F = 1, 6
H_lat = W_lat = unet.config.sample_size
C_lat = unet.config.in_channels // 2  # UNet expects 2*latent_channels
E = 1024  # text encoder dim in SVD
print(f"UNet in_channels: {unet.config.in_channels}")
print(f"UNet sample_size (latent H/W): {unet.config.sample_size}")
print(f"VAE latent_channels: {C_lat}")

# fake inputs
x = torch.randn(
    B, F, unet.config.in_channels, H_lat, W_lat, device=DEV, dtype=torch.float16
)  # BFCHW for 3D UNet (diffusers expects BFCHW with time folded)
x = x.view(B * F, unet.config.in_channels, H_lat, W_lat)  # diffusers UNet3D takes (B*F, C, H, W)
t = torch.tensor([500] * (B * F), device=DEV, dtype=torch.float32)
eh = torch.randn(B * F, 77, E, device=DEV, dtype=torch.float16)

# SVD requires added_time_ids (fps, motion_bucket_id, noise_aug_strength)
added_time_ids = torch.tensor([[6.0, 127, 0.02]], device=DEV, dtype=torch.float16).repeat(B * F, 1)

with (
    torch.no_grad(),
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True),
):
    y = unet(
        x, t, encoder_hidden_states=eh, added_time_ids=added_time_ids
    ).sample  # (B*F, 2*C_lat, H, W) -> but cfg’d as 4 when C_lat=2
    y = y.view(B, F, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4).contiguous()  # (B, C, F, H, W)
    print("unet_out (B,F,C,H,W) or similar:", tuple(y.permute(0, 2, 1, 3, 4).shape))

    # VAE decode expects (B, C, F, H, W); returns (B, F, 3, H*8, W*8)
    dec = vae.decode(y, num_frames=F, return_dict=True)
    vid = dec.sample  # (B, F, 3, H*8, W*8)
    print("vae_decode (B,F,3,H*8,W*8):", tuple(vid.shape))
