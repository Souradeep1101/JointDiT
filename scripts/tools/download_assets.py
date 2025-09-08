# scripts/download_assets.py
import os
from pathlib import Path

from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster downloads

BASE = Path("/workspace/jointdit/assets/models")
TARGETS = {
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt",
    "audioldm2": "cvssp/audioldm2",  # <-- fixed
    "clip": "openai/clip-vit-large-patch14",
}

for name, repo in TARGETS.items():
    out = BASE / name
    out.mkdir(parents=True, exist_ok=True)
    p = snapshot_download(
        repo_id=repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack.index"],
    )
    print(name, "->", p)
print("done")
