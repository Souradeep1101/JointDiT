import json
from pathlib import Path

cfg_path = Path("/workspace/jointdit/assets/models/audioldm2/unet/config.json")
cfg = json.load(open(cfg_path))


def flatten_ints(x):
    """Yield all integer-like dims from nested structures, skipping None."""
    if x is None:
        return
    if isinstance(x, (int, float)):
        yield int(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from flatten_ints(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from flatten_ints(y)


changed = False

# 1) cross_attention_dim → single int
cad = cfg.get("cross_attention_dim", None)
if cad is not None and not isinstance(cad, int):
    ints = list(flatten_ints(cad))
    if len(ints) == 0:
        raise ValueError(f"cross_attention_dim had no numeric entries: {cad}")
    # Pick a sensible value. ALDM2 often uses 768 or 1024; prefer the largest we see.
    val = max(ints)
    print(f"[patch] cross_attention_dim: {cad} -> {val}")
    cfg["cross_attention_dim"] = int(val)
    changed = True

# 2) (Optional hardening) Some configs have down_block/cross dims as lists; UNet2DConditionModel
# expects a *single* value. We leave block-level head dims alone (Diffusers can handle them),
# but we coerce global "encoder_hidden_states" dim fields to int if present.
for key in ("addition_embed_type_num_heads", "num_attention_heads"):
    # keep as-is; Diffusers handles these per-block lists
    pass

if changed:
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print("[patch] wrote", cfg_path)
else:
    print("[patch] no changes needed", cfg_path)
