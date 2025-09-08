import os
import pathlib

from huggingface_hub import snapshot_download


def fetch(repo_id, allow, dst):
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[hf] {repo_id} -> {dst}  allow={allow}")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow,
        local_dir=str(dst),
        local_dir_use_symlinks=False,
        token=os.getenv("HF_TOKEN", None),
    )


# SVD VAE (video)
fetch(os.getenv("SVD_VAE_REPO", "$(SVD_VAE_REPO)"), ["vae/*", "*.json"], "assets/models/svd")

# AudioLDM2 VAE (audio)
fetch(
    os.getenv("ALDM2_VAE_REPO", "$(ALDM2_VAE_REPO)"), ["vae/*", "*.json"], "assets/models/audioldm2"
)

print("[ok] model assets fetched")
