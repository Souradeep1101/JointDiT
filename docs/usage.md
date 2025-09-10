# Usage

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make fetch-models
````

## Data & latents

* **Dataset**: AVSync15 (or your subset).
* **Cache latents** (VAE-encoded video/audio) for speed:

```bash
make cache-train   # uses configs/day02_cache.yaml
make cache-val
```

## Training

### One-GPU (48 GB) defaults

The final wrappers auto-scale memory knobs. To keep your own overrides set `FORCE_KEEP_USER_ENVS=1`.

```bash
# Stage-A (quick)
make final-train-a

# Stage-B (long)
make final-train-b
# or keep your custom env:
make final-train-b-keepenv JOINTDIT_MAX_T=6 JOINTDIT_Q_CHUNK_V=32 \
  JOINTDIT_KV_DOWNSAMPLE=8 JOINTDIT_Q_CHUNK_A=0 \
  PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True"
```

### Memory levers (env)

* `JOINTDIT_Q_CHUNK_V` – process video queries in chunks (e.g., 32).
* `JOINTDIT_KV_DOWNSAMPLE` – downsample keys/values by factor (e.g., 8).
* `JOINTDIT_MAX_T` – max frames.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` – mitigate fragmentation.

> These directly address attention quadratic cost and avoid OOM on 48 GB.&#x20;

## Inference (CLI)

```bash
# Headless
make infer CKPT=checkpoints/...pt REF=data/cache/meta/val/clip_val_0001.json \
  PROMPT="a crow on a fence" NEG_PROMPT="" STEPS=30 SEED=7 WV=1.5 WA=1.5 WI=0.5

# Gradio
make ui
```

### Video decoding

Temporal VAE (SVD) requires `num_frames`; the helper now supplies it and supports **chunked decode** via `JOINTDIT_VID_DECODE_CHUNK` to limit VRAM spikes.

## Troubleshooting

* **OOM during Stage-B**: increase `JOINTDIT_KV_DOWNSAMPLE` to 8 or 12; raise `JOINTDIT_Q_CHUNK_V` (32→64).
* **UI crash at decode**: ensure `decode_video_latents` passes `num_frames=T` to temporal VAE; use chunked decode.
* **No CLIP available**: set `clip.enabled=false` or omit prompts.

## Repro script (48 GB, \~1-sec latents)

```bash
FORCE_KEEP_USER_ENVS=1 \
JOINTDIT_MAX_T=6 \
JOINTDIT_Q_CHUNK_V=32 \
JOINTDIT_KV_DOWNSAMPLE=8 \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" \
make final-train-b
```
