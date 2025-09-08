import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def scan_in_features(st_path: Path):
    vals_by_key = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(("attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight")):
                w = f.get_tensor(k)  # [out, in]
                if w.ndim == 2:
                    vals_by_key[k] = int(w.shape[1])
    return vals_by_key


def choose_target(vals_by_key, explicit=None):
    if explicit is not None:
        return int(explicit)
    cnt = Counter(vals_by_key.values())
    # prefer the most common; tie-breaker = larger dim (keeps more info)
    best = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[-1][0] if cnt else None
    return best


def pad_or_truncate(mat: torch.Tensor, target_in: int):
    out, in_ = mat.shape
    if in_ == target_in:
        return mat
    if in_ > target_in:
        return mat[:, :target_in].contiguous()
    # in_ < target_in: pad zeros on the right
    pad = torch.zeros(out, target_in - in_, dtype=mat.dtype)
    return torch.cat([mat, pad], dim=1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/workspace/jointdit/assets/models/audioldm2/unet")
    ap.add_argument(
        "--target",
        type=int,
        default=None,
        help="force a specific cross-attn dim (e.g., 768). If omitted, use mode.",
    )
    args = ap.parse_args()

    unet_dir = Path(args.model_dir)
    st_path = unet_dir / "diffusion_pytorch_model.safetensors"
    cfg_path = unet_dir / "config.json"
    bak_path = unet_dir / "diffusion_pytorch_model.safetensors.bak"

    vals_by_key = scan_in_features(st_path)
    if not vals_by_key:
        raise RuntimeError(
            "No attn2.to_{q,k,v}.weight tensors found; cannot infer cross-attn dims."
        )

    hist = Counter(vals_by_key.values())
    print("cross-attn in_features histogram:", dict(sorted(hist.items())))

    target = choose_target(vals_by_key, args.target)
    if target is None:
        raise RuntimeError("Could not determine target cross-attn dim.")
    print(f"[choose] target cross_attention_dim = {target}")

    # Load all tensors, patch those that need it
    new_tensors = {}
    patched_counts = defaultdict(int)
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if (
                k.endswith(("attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight"))
                and t.ndim == 2
            ):
                t_new = pad_or_truncate(t, target)
                if t_new.shape != t.shape:
                    patched_counts[(k.split("attn2.")[-1]).replace(".weight", "")] += 1
                new_tensors[k] = t_new
            else:
                new_tensors[k] = t

    if patched_counts:
        print("[patch] changed layers:", dict(patched_counts))
    else:
        print("[patch] no attn2 weights needed changes")

    # Backup then write
    if not bak_path.exists():
        st_path.replace(bak_path)
        print(f"[backup] wrote {bak_path}")
    else:
        # if backup already exists, keep it and overwrite current
        st_path.unlink(missing_ok=True)
    save_file(new_tensors, str(st_path))
    print(f"[save] wrote coerced weights -> {st_path}")

    # Patch config
    cfg = json.load(open(cfg_path))
    old = cfg.get("cross_attention_dim", None)
    cfg["cross_attention_dim"] = int(target)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"[config] cross_attention_dim: {old} -> {target} in {cfg_path}")


if __name__ == "__main__":
    main()
