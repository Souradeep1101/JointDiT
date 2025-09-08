import json
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path("/workspace/jointdit/assets/models")
TARGET = ROOT / "audioldm2"  # final "active" ALDM2 dir
CANDIDATES = [
    ("audioldm2-large", "cvssp/audioldm2-large"),
    ("audioldm2-music", "cvssp/audioldm2-music"),
    ("audioldm2", "cvssp/audioldm2"),
]


def dl_min(repo, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for f in [
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]:
        hf_hub_download(repo_id=repo, filename=f, local_dir=str(outdir), resume_download=True)


def mode_cross_in(st_path: Path) -> int | None:
    vals = []
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(("attn2.to_k.weight", "attn2.to_q.weight", "attn2.to_v.weight")):
                w = f.get_tensor(k)
                if w.ndim == 2:
                    vals.append(int(w.shape[1]))
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def patch_cross(cfg_path: Path, dim: int):
    cfg = json.load(open(cfg_path))
    old = cfg.get("cross_attention_dim")
    cfg["cross_attention_dim"] = int(dim)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"[patch] cross_attention_dim: {old} -> {dim}")


def try_strict_load(model_dir: Path) -> bool:
    import torch
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor

    unet = UNet2DConditionModel.from_pretrained(
        str(model_dir), subfolder="unet", torch_dtype=torch.float16
    )
    unet.set_attn_processor(AttnProcessor())
    return True


ROOT.mkdir(parents=True, exist_ok=True)

picked = None
for short, repo in CANDIDATES:
    cand = ROOT / short
    print(f"\n=== Checking {repo} ===")
    try:
        dl_min(repo, cand)
        st = cand / "unet/diffusion_pytorch_model.safetensors"
        cfg = cand / "unet/config.json"
        mode = mode_cross_in(st)
        if mode is None:
            print("  [skip] no attn2 weights found")
            continue
        patch_cross(cfg, mode)
        try:
            if try_strict_load(cand):
                # promote this candidate to TARGET
                if TARGET.exists() or TARGET.is_symlink():
                    shutil.rmtree(TARGET, ignore_errors=True)
                try:
                    TARGET.symlink_to(cand, target_is_directory=True)
                    print(f"[pick] Symlinked {TARGET} -> {cand}")
                except Exception:
                    shutil.copytree(cand, TARGET)
                    print(f"[pick] Copied {cand} -> {TARGET}")
                picked = cand
                break
        except Exception as e:
            print("  [strict load failed]:", type(e).__name__, str(e)[:160], "…")
    except Exception as e:
        print("  [download failed]:", type(e).__name__, str(e)[:160], "…")

if not picked:
    print(
        "\n[warn] No strict-loading ALDM2 variant found. You can still load with ignore_mismatched_sizes (see Step 3)."
    )
else:
    print("[result] strict load OK:", picked)
