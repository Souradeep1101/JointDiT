# JointDiT: Video+Audio Latent Diffusion (Mini-Stack)

This repo contains a compact, reproducible stack for **joint video–audio denoising** (aka *JointDiT*), following our Day 1–7 build plan:
- Day 1–3: asset mirroring & UNet config sanitation for AudioLDM2, SVD VAE handling
- Day 2: latent caching (video & audio) + CLIP gates (optional)
- Day 3–4: slicer scaffolds + JointDiT skeleton (iso/full modes)
- Day 5: Stage-A training loop (OOM-friendly, chunked attention)
- Day 6: simple sampler + decode (MP4 + WAV)
- Day 7: Stage-B fine-tuning (freeze-most, unfreeze experts + in/out)

> For a runnable, end-to-end walkthrough: open **`notebooks/master/00_end_to_end_guide.ipynb`**.

---

## TL;DR — Common commands

### Environment
```bash
cd /workspace/jointdit
python -V
. .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
````

### Cache latents (Day 2)

```bash
# Train split
PYTHONPATH=. python scripts/data/cache_latents.py --cfg configs/day02_cache.yaml --split train
# Val split
PYTHONPATH=. python scripts/data/cache_latents.py --cfg configs/day02_cache.yaml --split val
```

### Final training (Stage-A)

We ship heuristics that scale with VRAM (or let you enforce your own).

```bash
# Safe defaults for ~48 GB (override as needed)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export JOINTDIT_MAX_T=6
export JOINTDIT_Q_CHUNK_V=64
export JOINTDIT_Q_CHUNK_A=0
export JOINTDIT_KV_DOWNSAMPLE=8

# Train A (25 steps smoke here; extend max-steps for real)
make final-train-a
# Or force **your** envs (Makefile respects FORCE_KEEP_USER_ENVS=1)
FORCE_KEEP_USER_ENVS=1 make final-train-a
```

### Final training (Stage-B)

```bash
# Same env knobs by default (auto VRAM scaling).
make final-train-b
# Or keep your envs:
FORCE_KEEP_USER_ENVS=1 make final-train-b
```

### Inference (simple sampler)

```bash
# Uses configs/day06_infer.yaml by default; overwrite ckpt/paths as needed.
make final-infer
# Or with explicit seeds/steps via config edits.
```

Outputs land under:

```
checkpoints/
  day05_stage_a_<suffix>/
  day07_stage_b_<suffix>/
outputs/
  day06/seed###_stepsNN.mp4
  day06/seed###_stepsNN.wav
```

---

## VRAM knobs (env)

We read these at runtime (both training & inference):

* `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"` — helps fragmentation on PyTorch 2.6+
* `JOINTDIT_MAX_T` — clamp video time tokens (e.g., 6 or 12)
* `JOINTDIT_Q_CHUNK_V` — query chunk size for video SDPA (64, 128…)
* `JOINTDIT_Q_CHUNK_A` — query chunk size for audio SDPA (usually 0)
* `JOINTDIT_KV_DOWNSAMPLE` — subsample KV bank (1, 2, 4, 8…)
* `FORCE_KEEP_USER_ENVS=1` — Makefile will **not** override your exports

> Rule of thumb: Higher `Q_CHUNK_V` and lower `KV_DOWNSAMPLE` are **faster** but use **more** VRAM.

---

## Typical profiles

| GPU class | Example settings (safe)                              |
| --------- | ---------------------------------------------------- |
| \~24 GB   | `MAX_T=4`, `Q_CHUNK_V=64`, `KV_DOWNSAMPLE=8`         |
| \~48 GB   | `MAX_T=6`, `Q_CHUNK_V=64–128`, `KV_DOWNSAMPLE=8–4`   |
| ≥80 GB    | `MAX_T=12`, `Q_CHUNK_V=128–256`, `KV_DOWNSAMPLE=4–1` |

---

## Troubleshooting

* **OOM in SDPA**
  Lower `JOINTDIT_Q_CHUNK_V`, raise `JOINTDIT_KV_DOWNSAMPLE`, or clamp `JOINTDIT_MAX_T`.

* **`expandable_segments` crash**
  Do **not** set `expandable_segments:true` on this PyTorch build. Use only `max_split_size_mb:128`.

* **Missing CLIP weights / skip CLIP caching**
  Set `clip.enabled: false` in `configs/day02_cache.yaml`.

* **FP16 scaler error “Attempting to unscale FP16 gradients.”**
  We guard unscale with AMP enabled checks; ensure config `runtime.dtype` is `fp16` only if you intend to use AMP.

* **VAE dtype mismatch at inference**
  We keep model params in `fp32` and cast VAE inputs to the VAE’s weight dtype before decode.

---

## Repo map

```
scripts/
  data/cache_latents.py         # Day 2 cacheer
  smoke/                        # day smokes
  train/train_stage_a.py        # Stage A
  train/train_stage_b.py        # Stage B
  infer/infer_joint.py          # Day 6 sampler
  finals/train.sh               # unified Train (A/B) with VRAM heuristics
  finals/infer.sh               # unified Inference hook
models/
  jointdit.py                   # core
  joint/                        # attention blocks + RoPE
configs/
  day02_cache.yaml
  day05_train.yaml
  day06_infer.yaml
  day07_trainB.yaml
notebooks/
  master/00_end_to_end_guide.ipynb
  day_01/ ... day_07/
assets/models/
  svd/vae
  audioldm2/vae
data/cache/
  video_latents/
  audio_latents/
  meta/
outputs/
checkpoints/
```

---


## Day 02 — Final scripts & Gradio UI

**Notebook:** `notebooks/day_02/02_final_scripts_and_ui.ipynb`

**What’s new**
- Final train drivers: `scripts/finals/train.sh` (Stage-A/B)
- Final CLI infer: `scripts/finals/infer.sh` (Day-6 sampler)
- Gradio UI: `scripts/ui/app.py`
- Make targets: `make final-train-a`, `make final-train-b`, `make final-infer`, `make ui`

### VRAM profiles (export first)

```bash
# Max-Quality (>=80GB)
export JOINTDIT_MAX_T=12
export JOINTDIT_Q_CHUNK_V=128
export JOINTDIT_Q_CHUNK_A=0
export JOINTDIT_KV_DOWNSAMPLE=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Balanced (~48GB)
export JOINTDIT_MAX_T=12
export JOINTDIT_Q_CHUNK_V=128
export JOINTDIT_Q_CHUNK_A=0
export JOINTDIT_KV_DOWNSAMPLE=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Conservative (<=24GB)
export JOINTDIT_MAX_T=6
export JOINTDIT_Q_CHUNK_V=64
export JOINTDIT_Q_CHUNK_A=0
export JOINTDIT_KV_DOWNSAMPLE=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
````

> The Makefile includes `FORCE_KEEP_USER_ENVS=1` so your exports override defaults.

### Train

```bash
# Stage-A
make final-train-a
# or
PYTHONPATH=. scripts/finals/train.sh --stage A --cfg configs/day05_train.yaml --max-steps 25 --ckpt-suffix finalA --log-suffix finalA

# Stage-B
make final-train-b
# or
PYTHONPATH=. scripts/finals/train.sh --stage B --cfg configs/day07_trainB.yaml --max-steps 1000 --ckpt-suffix finalB --log-suffix finalB
```

### Inference (CLI)

```bash
make final-infer
# or
PYTHONPATH=. scripts/finals/infer.sh \
  --cfg configs/day06_infer.yaml \
  --ckpt checkpoints/day07_stage_b_finalB/ckpt_step_001000.pt \
  --steps 30 --seed 0
```

### Inference (UI)

```bash
make ui
# then open http://127.0.0.1:7860
```

Pick a config (`configs/ui_infer.yaml`), checkpoint (`*.pt`), and ref meta (`data/cache/meta/val/*.json`), choose a VRAM profile, set steps/seed/CFG, then click **Run**.

### UI troubleshooting

If you get **“No API found”** or a schema error, pin dependencies:

**Option A (stable)**

```bash
pip install -U "altair<6" "numpy==1.26.4"
pip install --no-deps --force-reinstall "gradio==4.19.2" "gradio_client==0.10.1"
pip install "fastapi<0.112" "starlette<0.37" "uvicorn<0.30"
```

**Option B (newer gradio)**

```bash
pip install -U "gradio==4.44.0" "gradio_client==1.4.2"
pip check
```

The app includes a small JSON-schema **safety patch** to avoid the crash.

# Text + Image Conditioning (CLIP) — Integration Notes

This addendum explains how **text** and **image** (first-frame) conditions are used during caching, training, and inference after today’s updates.

---

## 1) Caching with optional CLIP image embeddings

Edit `configs/day02_cache.yaml`:

```yaml
clip:
  enabled: true
  # Use either a local checkpoint folder/file OR a tag that open_clip knows.
  # For tags, prefer: model_path: openai
  model_path: openai
  # Valid variants include: ViT-B-16, ViT-L-14, ViT-H-14  (no slashes)
  variant: ViT-B-16
````

Then run:

```bash
make cache-train
make cache-val
```

This will produce per-clip metadata under `data/cache/meta/{split}/*.json`. When CLIP is enabled, the script also writes `data/cache/img_clip/{split}/*.pt` containing a normalized image embedding (from the first frame) and sets `meta.clip_file` to that path.

> **Note on names:** If you previously used `ViT-L/14` (with a slash), switch to `ViT-L-14`. open\_clip expects dashes, not slashes.

---

## 2) Training with conditions

* **Dataset & collate** already carry:

  * `clip_emb` (if present)
  * `v_latents` (T, C, H, W)
  * `a_latents` (1, C, H, W)
  * `meta` including `img_firstframe` and optional `captions`

* **Stage A** (`scripts/train/train_stage_a.py`): learns base reconstruction with your existing schedules.

* **Stage B** (`scripts/train/train_stage_b.py`): unfreezes joint blocks and I/O projections (configurable) for modest finetuning.

* If you enable text/image conditioning in the model forward (e.g., via `t_ctx` / `i_ctx`), the current setup *guides at inference-time*. You can later extend Stage B to inject and train adapters with `clip_emb` as an auxiliary target if desired.

Quick-smoke examples:

```bash
# Small smoke
PYTHONPATH=. python scripts/train/train_stage_a.py --cfg configs/day05_train.yaml --max-steps 25 --log-suffix smoke --ckpt-suffix smoke

# Small Stage-B smoke
PYTHONPATH=. python scripts/train/train_stage_b.py --cfg configs/day07_trainB.yaml --max-steps 100 --log-suffix smoke --ckpt-suffix smoke
```

---

## 3) Inference (CLI) with text + image

`infer_joint.py` now accepts:

* **Text** prompt (`prompt`) and **negative** prompt (`negative_prompt`)
* **Image**: either explicit `clip.image_path` in the YAML or `clip.use_image_firstframe: true` to pull the first frame path from `ref_meta`.

### Minimal YAML (`configs/ui_infer.yaml`)

```yaml
runtime:
  device: "cuda"
  dtype: "fp32"

out_dir: "outputs/ui_infer"

# Provide either a ref_meta OR (see § manual shapes) a 'manual_shapes' block
ref_meta: "data/cache/meta/val/your_clip.json"

video_vae_path: "assets/models/svd/vae"
audio_vae_path: "assets/models/audioldm2/vae"

ckpt: "checkpoints/.../ckpt_step_000100.pt"

mp4_fps: 12
audio_sr: 16000
steps: 30
seeds: [0]

guidance:
  type: "jointcfg"
  weight_v: 1.2
  weight_a: 1.2
  weight_txt: 1.5
  weight_neg_txt: 0.0
  weight_img: 1.0

clip:
  enabled: true
  # Either (image_path) OR (use_image_firstframe: true)
  use_image_firstframe: true
  # For open_clip:
  model: ViT-B-16     # alias: 'variant'
  pretrained: openai
```

### Env overrides (no YAML edits)

```bash
JOINTDIT_PROMPT="a baby laughing" \
JOINTDIT_NEG_PROMPT="" \
JOINTDIT_CKPT="checkpoints/.../ckpt_step_000100.pt" \
JOINTDIT_REF_META="data/cache/meta/val/clip_val_0001.json" \
JOINTDIT_STEPS=30 \
JOINTDIT_SEED=0 \
JOINTDIT_WV=1.2 \
JOINTDIT_WA=1.2 \
JOINTDIT_WT=1.5 \
JOINTDIT_WNT=0.0 \
JOINTDIT_WI=1.0 \
JOINTDIT_USE_IMG=1 \
PYTHONPATH=. python scripts/infer/infer_joint.py --cfg configs/ui_infer.yaml
```

---

## 4) Inference without a `ref_meta`

You can specify shapes manually in the YAML (useful for quick tests):

```yaml
manual_shapes:
  T: 12
  fps: 12
  sr: 16000
  video: { c: 4, h: 40, w: 72 }
  audio: { c: 8, h: 20, w: 16 }
```

`infer_joint.py` will use VAE scaling factors from your loaded VAEs when `ref_meta` is absent.

---

## 5) Gradio UI

The UI reads the same config and exposes:

* steps, seed, FPS
* guidance weights (video/audio/text/neg-text/image)
* Prompt / Negative prompt fields
* Toggle to use the **ref image** (first-frame) if `clip.enabled` is set

To run:

```bash
PYTHONPATH=. python scripts/ui/app.py
```

Outputs will appear under the configured `out_dir` (by default `outputs/ui_infer`).

---

## 6) Troubleshooting

* **CLIP cannot load / wrong variant name**
  Use dash-not-slash variants. Examples: `ViT-B-16`, `ViT-L-14`, `ViT-H-14`.
  Set `pretrained: openai` for the simplest path.

* **“object of type 'NoneType' has no len()” during CLIP load**
  Usually indicates a mismatched `variant`/`pretrained` pair or a failed download. Try `model: ViT-B-16` with `pretrained: openai`.

* **Device mismatch (cpu vs cuda) in CLIP encode**
  Ensure the encoder and inputs are moved to the **same** `device` (the provided `CLIPCondEncoder` handles this; avoid manual `.to('cpu')` on tokens).

* **No outputs in UI panel**
  The app now selects the newest files in `out_dir` **after** each run. If multiple files already exist, that’s fine—newest wins. Clear the folder if you want a clean view.

---

That’s it! You now have a repeatable path to run **text + image** conditioned JointDiT through cache → train → infer → UI.

### Outputs

* UI → `outputs/ui/`
* CLI → `outputs/day06/`
* Checkpoints → `checkpoints/day05_stage_a_*`, `checkpoints/day07_stage_b_*`

## Notes

* **Determinism:** we seed RNG per run, but cudnn benchmark is enabled for throughput.
* **Licensing:** Check licensing of model weights (SVD, AudioLDM2) before publishing.
* **Next:** Gradio UI, Docker, eval datasets (benchmarks), CI smoke on PRs.

Happy training! 🎬🔊