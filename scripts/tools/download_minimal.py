import os
from pathlib import Path

from huggingface_hub import hf_hub_download

os.environ["HF_HOME"] = "/workspace/hf_home"


def fetch(repo, files, dst_root):
    Path(dst_root).mkdir(parents=True, exist_ok=True)
    for f in files:
        local = hf_hub_download(
            repo_id=repo,
            filename=f,
            local_dir=dst_root,
            local_dir_use_symlinks=True,
            resume_download=True,
        )
        print("ok:", local)


# SVD: UNet + VAE
svd_root = "/workspace/jointdit/assets/models/svd"
fetch(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    [
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    svd_root,
)

# AudioLDM2: UNet + VAE
aldm_root = "/workspace/jointdit/assets/models/audioldm2"
fetch(
    "cvssp/audioldm2",
    [
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    aldm_root,
)

print("done")
