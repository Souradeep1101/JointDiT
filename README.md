
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
