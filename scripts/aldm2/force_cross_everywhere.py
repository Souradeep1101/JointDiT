import json
from pathlib import Path

UNET_DIR = Path("/workspace/jointdit/assets/models/audioldm2/unet")
CFG = UNET_DIR / "config.json"
TARGET = 1024


def replace_nested(v, to=TARGET):
    if isinstance(v, list):
        return [replace_nested(x, to) for x in v]
    # Treat None or any other scalar as cross-dim `to`
    return int(to)


cfg = json.load(open(CFG))

# Common fields across different ALDM2 exports
for k in [
    "cross_attention_dim",  # canonical
    "addition_attention_dim",  # sometimes used
    "encoder_hid_dim",  # rare alias
]:
    if k in cfg:
        before = cfg[k]
        after = replace_nested(before, TARGET)
        cfg[k] = after
        print(f"[patch] {k}: {before} -> {after}")

# Some exports carry nested per-block settings under these names too
for k in [
    "down_block_types",
    "up_block_types",
    "mid_block_type",
    # no changes needed here; we leave types as-is
]:
    if k in cfg:
        print(f"[info] {k}: {cfg[k]}")

json.dump(cfg, open(CFG, "w"), indent=2)
print(f"[save] wrote {CFG}")
