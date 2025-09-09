
---

# JointDiT — Image → Sounding Video (I2SV)

From-scratch implementation of a **Joint Diffusion Transformer** that generates synchronized **video + audio** from an image and text conditions. This repo includes smoke tests, training pipelines (Stage A/B), a headless inference CLI, and a Gradio UI.

> **Paper:** This project implements the key ideas from the referenced JointDiT research paper (Input/Joint/Output blocks + guidance). See `docs/` for an overview and notebook walkthroughs.

---

## Features

* **End-to-end pipeline**: data → latent caching → training (A/B) → inference → UI
* **Joint interaction** between audio & video experts (ablation: iso-v / iso-a)
* **Repro-first**: Makefile runners, small smoke configs, notebooks, and an auto-built docs site
* **Conditioning**: text prompt (+ negative), optional **CLIP** image guidance (first frame or custom image)

---

## Architecture and Design Notes

JointDiT follows the **Input–Joint–Output** block design from the paper. We integrate a pre-trained video diffusion UNet and a pre-trained audio diffusion UNet, inserting **JointBlocks** (transformer-based cross-modality blocks) between their layers. Each JointBlock performs a full cross-modal attention and separate feed-forward on each modality’s tokens.

- **Input Block:** The first layer of each UNet is used as an input encoder for that modality. In our implementation, we bypass the original conv layer and instead **flatten each latent** and apply a linear projection (video 4→256 dims, audio 8→256 dims). This produces token sequences for each modality (preserving spatial/temporal positions) that enter the JointBlocks. *(This is a simplification: a linear projection on channels acts similar to a 1x1 conv. It avoids heavy computation from the original model’s first conv.)*

- **Joint Blocks:** We have `N` JointBlocks (configurable, e.g. 2). Each JointBlock performs:
  1. **Modality-specific pre-norm & self-attention** (within the JointBlock’s attention, we effectively allow each modality’s queries to attend to a combined set of keys).
  2. **Full cross-modal attention** – the core of JointDiT. Using a Perceiver-style mechanism, video and audio tokens attend jointly to each other. This enables fine-grained sync: e.g., an audio token at time *t* can attend to video frame tokens at time *t* (and all others), and vice versa.
  3. **Extra context conditioning (optional):** If CLIP text/image embeddings are provided, they are added as additional keys/values in the attention. This guides the generation with semantic context.
  4. **Modality-specific feed-forward**: After attention, each modality’s tokens pass through its own feed-forward network (MLP) to process intra-modal information, then residual connections complete the transformer block structure.
  
  <small>*Each sub-layer (attention or MLP) is preceded by a LayerNorm (one per modality path). In the paper, an adaptive LayerNorm (AdaLN) is used:contentReference[oaicite:10]{index=10}; in our code we use standard LN here, with adaptability handled by the next point.*</small>

- **Output Block:** The last layers of each UNet act as decoders for their modality. In JointDiT, after the final JointBlock, we map tokens back to the original latent shapes. We apply a series of lightweight adaptive normalization layers (`LazyAdaLN`) corresponding to the original UNet’s remaining layers, then a linear projection back to the UNet’s channel count (video 256→4, audio 256→8). This yields the denoised latents $\hat{v}_0$ and $\hat{a}_0$.

**Pre-trained Weights Usage:** We load the original video and audio diffusion models (e.g., Stable Video Diffusion and AudioLDM2). To preserve their knowledge:
- In **Stage A (initial joint training)**, we **freeze** the pre-trained model parameters. We do *not* modify the original conv weights; instead, we learn the JointBlocks and adaptive norms on top. Notably, our current code does not execute the heavy UNet convolutions for intermediate layers during Stage A – it relies on the AdaLN layers to adjust the activations as placeholders. This keeps Stage A fast and memory-light, but it means the original model’s transformations beyond the first layer are initially bypassed.
- In **Stage B (fine-tuning)**, we **unfreeze or incorporate** parts of the original models. You can configure which layers to fine-tune. By default, we enable training for the JointBlocks (the cross-modal transformer) and can optionally unfreeze the Input/Output projections (`JOINTDIT_UNFREEZE_IO` flag) or specific UNet blocks. The idea is to gradually re-introduce the original model’s capacity. For example, one might unfreeze the “Expert” layers (the layers around where JointBlocks plug in) so the model can jointly adjust those features.

**Known Limitations:** Currently, the implementation doesn’t feed activations through the actual conv layers of the Expert/Output blocks during joint training. This is an area for improvement. In theory, incorporating those layers (at least in inference or Stage B) should yield higher fidelity results, since the UNet’s later layers are trained for decoding details. Our minimalist approach of using multiple AdaLNs as placeholders means the model must learn more within the transformer to compensate. If you find the generated video or audio quality is lacking fine details, this could be a reason. Future updates may integrate the UNet’s convolutional forward passes fully, trading off compute for quality.

---

## Quickstart

### 1) Environment (no Docker)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -c requirements.lock.txt
pre-commit install  # optional: formatting/lint on commit
```

System deps (Ubuntu): see `system_deps.txt` (e.g., `ffmpeg`, `libsndfile1`).
**CUDA/Torch**: repo targets CUDA 12.1 wheels (adjust if your platform differs).

Sanity checks:

```bash
python scripts/dev/check_env.py
make test
```

### 2) Models & Data (Hugging Face + quick dataset stubs)

```bash
# Required VAEs (uses HF snapshot_download; set HF_TOKEN if gated)
make fetch-models
# AVSync15 1-second subset (small, for quick experiments; uses gdown + ffmpeg)
make deps-media
make fetch-avsync15-1s
# Optional: VATEX mini
make fetch-vatex-mini
```

### 3) Cache latents (small splits)

```bash
make cache-train
make cache-val
```

### 4) Training (smokes → real)

```bash
# Smoke tests
make smoke            # video/audio expert smokes
make day7-smoke       # tiny Stage-B run (~100 steps)

# Stage-B / ablation
make day7-trainB
make day7-ablate-nojoint
```

### 5) Inference (headless) + UI

```bash
# Generic CLI (set your ckpt + meta)
make infer CKPT=checkpoints/...pt REF=data/cache/meta/val/clip_val_0001.json \
  PROMPT="a baby laughing" STEPS=40 USE_IMG=1

# Latest Stage-B checkpoint helper
make day7-infer

# Gradio UI
make ui
```

---

## CLIP & Image Guidance

Enable in `configs/ui_infer.yaml`:

```yaml
clip:
  enabled: true
  model: ViT-B-16
  pretrained: openai
  use_image_firstframe: true   # or set image_path manually
```

Or override per-run via env (the Makefile passes these through):

* `JOINTDIT_PROMPT`, `JOINTDIT_NEG_PROMPT`
* `JOINTDIT_USE_IMG=1`
* `JOINTDIT_IMAGE=path/to/cond.png`
* `JOINTDIT_CLIP_MODEL=ViT-B-16`, `JOINTDIT_CLIP_PRETRAINED=openai`
* Guidance/weights: `JOINTDIT_WV`, `JOINTDIT_WA`, `JOINTDIT_WT`, `JOINTDIT_WNT`, `JOINTDIT_WI`

Example:

```bash
make infer CKPT=checkpoints/.../ckpt_step_1000.pt REF=data/cache/meta/val/clip_val_0001.json \
  PROMPT="rainy street, distant thunder" NEG_PROMPT="people" \
  STEPS=40 SEED=123 USE_IMG=1 CLIP_MODEL=ViT-B-16 CLIP_PRETRAINED=openai
```

---

## Makefile Cheat-Sheet

| Target                                                                | What it does                                                                                     |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `smoke`, `smoke-svd`, `smoke-aldm2`                                   | Minimal expert smoke tests                                                                       |
| `cache-train`, `cache-val`, `smoke-day02`                             | Cache latents / cache smoke                                                                      |
| `smoke-day03`, `smoke-day04`, `smoke-day05`, `smoke-day06`            | Day-wise scaffolds & checks                                                                      |
| `day7-trainB`, `day7-ablate-nojoint`, `day7-smoke`, `day7-infer`      | Stage-B training/ablation/smoke/infer                                                            |
| `final-train-a`, `final-train-b` (+ `-keepenv`)                       | Wrapped runs via `scripts/finals/train.sh` (set `FORCE_KEEP_USER_ENVS=1` to keep your env knobs) |
| `infer`                                                               | Headless inference helper (uses `CFG_INFER`, `JOINTDIT_*` vars)                                  |
| `ui`                                                                  | Launch Gradio app                                                                                |
| `fetch-models`, `fetch-vatex-mini`, `fetch-avsync15-1s`, `deps-media` | Assets & toy datasets                                                                            |
| `sys-deps`                                                            | Print/install system deps (Debian/Ubuntu)                                                        |
| `docs-install`, `docs-build`, `docs-serve`, `docs-deploy`             | Build and publish docs site                                                                      |

---

## Project Docs (MkDocs + Material)

This repo ships a live docs site with Overview, Architecture, Timeline, and your **notebooks rendered as pages**.

Local preview:

```bash
make docs-install
make docs-serve DOCS_HOST=0.0.0.0 DOCS_PORT=8000
```

Publish (GitHub Pages via Actions):

1. Push to `main`.
2. In GitHub → Settings → Pages → Source: **GitHub Actions**.
3. Each push auto-deploys.

> Notebooks are copied to `docs/notebooks/` and normalized on every run. The site uses Material theme, sticky ToC, and light/dark toggle. You can tweak colors/fonts in `mkdocs.yml`.

---

## Repo Layout (high-level)

```
configs/           # experiment configs
scripts/           # training / inference / tools / ui
docs/              # mkdocs site (index, usage, architecture, timeline, notebooks/)
assets/            # model assets (not all tracked in git)
tools/             # docs generator / notebook sanitizer
Makefile           # entrypoints for everything above
```

---

## Troubleshooting

* **HF gated repos**: set `HF_TOKEN` before `make fetch-models`.
* **Missing system deps**: run `make sys-deps` (and/or see `system_deps.txt`).
* **Docs convert errors** (nbconvert/mistune): pin `mistune==2.0.5` in `requirements-docs.txt`, then `make docs-install`.
* **Notebook schema warnings** (missing cell IDs / execution metadata): notebooks are sanitized during docs build; re-run `make docs-serve`.
* **API docs disabled**: If you later want API pages, add back `docs/api.md` and `mkdocstrings` plugin in `mkdocs.yml`.

---

## Repro Tips

* Start with small **smoke** runs to verify wired configs, then scale steps/bs.
* Use `FORCE_KEEP_USER_ENVS=1` with `final-train-*` if you want your CUDA/JOINTDIT\_\* env kept intact.
* Keep notebooks focused: short cells, save after each major step; the Timeline uses file mtimes.

---

## Acknowledgements

* Research ideas from the JointDiT paper (see included PDF).
* Uses Hugging Face assets for pretrained VAEs.
* CLIP via OpenCLIP/OpenAI variants.
* Many thanks to the open-source ecosystem (PyTorch, Diffusers-style tooling, MkDocs Material).

---

## License

Add your license of choice (e.g., Apache-2.0/MIT). Document dataset/model license constraints in `assets/` if applicable.

---

### Changelog

See `docs/changelog.md` (auto-generated from git when available).
