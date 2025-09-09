.PHONY: smoke-svd smoke-aldm2 smoke \
        cache-train cache-val smoke-day02 \
        smoke-day03 smoke-day04 smoke-day05 smoke-day06 \
        day7-trainB day7-ablate-nojoint day7-infer day7-smoke \
        final-train-a final-train-a-keepenv final-train-b final-train-b-keepenv final-train-b-safe\
        ui infer \
        deps-media fetch-vatex-mini fetch-models \
		docs-install docs-autodoc docs-build docs-serve docs-deploy \
        sys-deps

# =========================================================
# Common knobs (used by inference/UI)
# =========================================================
CFG_INFER          ?= configs/ui_infer.yaml
PROMPT             ?=
NEG_PROMPT         ?=
CKPT               ?=
REF                ?=
STEPS              ?=
SEED               ?=
WV                 ?=
WA                 ?=
WT                 ?=
WNT                ?=
WI                 ?=            # guidance weight (image)
USE_IMG            ?= 1          # 1=true (use first frame image from meta), 0=false
IMAGE              ?=            # optional path to an override image for CLIP
CLIP_MODEL         ?= ViT-B-16   # open_clip model name/variant
CLIP_PRETRAINED    ?= openai     # open_clip pretrained tag

# =========================================================
# Model fetching (kept out of git)
# - Uses Hugging Face snapshot_download to populate:
#     assets/models/svd/vae
#     assets/models/audioldm2/vae
# - Set HF_TOKEN if repositories are gated.
# - Override repo ids by exporting SVD_VAE_REPO / ALDM2_VAE_REPO.
# =========================================================
SVD_VAE_REPO       ?= stabilityai/stable-video-diffusion-img2vid-xt
ALDM2_VAE_REPO     ?= cvssp/audioldm2

fetch-models:
	@echo "==> Fetching required model assets into assets/models/ (not tracked in git)"
	. .venv/bin/activate && python scripts/tools/fetch_models.py

# =========================================================
# Day 1 — Smoke tests
# =========================================================
smoke-svd:
	. .venv/bin/activate && python scripts/smoke/svd_smoke.py

smoke-aldm2:
	. .venv/bin/activate && python scripts/smoke/aldm2_smoke.py

smoke: smoke-svd smoke-aldm2
	@echo "Both smoke tests finished."

# =========================================================
# Day 2 — Cache helpers
# =========================================================
cache-train:
	. .venv/bin/activate && PYTHONPATH=. python scripts/data/cache_latents.py --cfg configs/day02_cache.yaml --split train

cache-val:
	. .venv/bin/activate && PYTHONPATH=. python scripts/data/cache_latents.py --cfg configs/day02_cache.yaml --split val

smoke-day02:
	. .venv/bin/activate && PYTHONPATH=. python scripts/data/smoke_day02_cache.py --split train --n 3

# =========================================================
# Day 3 — Slicer scaffold
# =========================================================
smoke-day03:
	. .venv/bin/activate && PYTHONPATH=. python scripts/smoke/day03_slicers_smoke.py

# =========================================================
# Day 4 — Joint DiT
# =========================================================
smoke-day04:
	. .venv/bin/activate && PYTHONPATH=. python scripts/smoke/day04_joint_smoke.py

# =========================================================
# Day 5 — Training (Stage A)
# =========================================================
smoke-day05:
	. .venv/bin/activate && PYTHONPATH=. python scripts/train/train_stage_a.py --cfg configs/day05_train.yaml --max-steps 25 --log-suffix smoke --ckpt-suffix smoke

# =========================================================
# Day 6 — Inference + guidance smoke
# Uses ui_infer.yaml and supports CLIP (text/neg/image) via env overrides
# =========================================================
smoke-day06:
	. .venv/bin/activate && \
	JOINTDIT_PROMPT="$(PROMPT)" \
	JOINTDIT_NEG_PROMPT="$(NEG_PROMPT)" \
	JOINTDIT_CKPT="$(CKPT)" \
	JOINTDIT_REF_META="$(REF)" \
	JOINTDIT_STEPS="$(STEPS)" \
	JOINTDIT_SEED="$(SEED)" \
	JOINTDIT_WV="$(WV)" \
	JOINTDIT_WA="$(WA)" \
	JOINTDIT_WT="$(WT)" \
	JOINTDIT_WNT="$(WNT)" \
	JOINTDIT_WI="$(WI)" \
	JOINTDIT_USE_IMG="$(USE_IMG)" \
	JOINTDIT_IMAGE="$(IMAGE)" \
	JOINTDIT_CLIP_MODEL="$(CLIP_MODEL)" \
	JOINTDIT_CLIP_PRETRAINED="$(CLIP_PRETRAINED)" \
	PYTHONPATH=. python scripts/infer/infer_joint.py --cfg $(CFG_INFER)

# =========================================================
# Day 7 — Stage-B + Ablation + Packaging
# =========================================================
day7-trainB:
	. .venv/bin/activate && PYTHONPATH=. python scripts/train/train_stage_b.py --cfg configs/day07_trainB.yaml --log-suffix d7 --ckpt-suffix d7

day7-ablate-nojoint:
	. .venv/bin/activate && PYTHONPATH=. python scripts/train/train_stage_b.py --cfg configs/day07_ablate_nojoint.yaml --log-suffix d7nojoint --ckpt-suffix d7nojoint

# Small sanity (100 steps) before the long run
day7-smoke:
	. .venv/bin/activate && PYTHONPATH=. python scripts/train/train_stage_b.py --cfg configs/day07_trainB.yaml --max-steps 100 --log-suffix smoke --ckpt-suffix smoke

# Inference using latest Stage-B ckpt via unified infer script
day7-infer:
	@CKPT_SB=$$(ls -1t checkpoints/day07_stage_b/ckpt_step_*.pt 2>/dev/null | head -n1); \
	if [ -z "$$CKPT_SB" ]; then echo "No Stage-B ckpt found"; exit 1; fi; \
	REF_META=$${REF:-data/cache/meta/val/clip_val_0001.json}; \
	echo "[infer] Using $$CKPT_SB (ref: $$REF_META)"; \
	. .venv/bin/activate && \
	JOINTDIT_CKPT="$$CKPT_SB" \
	JOINTDIT_REF_META="$$REF_META" \
	PYTHONPATH=. python scripts/infer/infer_joint.py --cfg $(CFG_INFER)

# =========================================================
# Final wrappers (scripts/finals/train.sh)
# - If FORCE_KEEP_USER_ENVS=1, keep user JOINTDIT_* & CUDA envs.
# - Else train.sh auto-picks VRAM-safe defaults.
# =========================================================
FORCE_KEEP_USER_ENVS ?= 0

final-train-a:
	. .venv/bin/activate && \
	FORCE_KEEP_USER_ENVS=$(FORCE_KEEP_USER_ENVS) PYTHONPATH=. scripts/finals/train.sh \
	  --stage A --cfg configs/day05_train.yaml --max-steps 25 \
	  --ckpt-suffix finalA --log-suffix finalA

final-train-a-keepenv:
	. .venv/bin/activate && \
	FORCE_KEEP_USER_ENVS=1 PYTHONPATH=. scripts/finals/train.sh \
	  --stage A --cfg configs/day05_train.yaml --max-steps 25 \
	  --ckpt-suffix finalA --log-suffix finalA

final-train-b:
	. .venv/bin/activate && \
	FORCE_KEEP_USER_ENVS=$(FORCE_KEEP_USER_ENVS) PYTHONPATH=. scripts/finals/train.sh \
	  --stage B --cfg configs/day07_trainB.yaml --max-steps 1000 \
	  --ckpt-suffix finalB --log-suffix finalB

final-train-b-keepenv:
	. .venv/bin/activate && \
	FORCE_KEEP_USER_ENVS=1 PYTHONPATH=. scripts/finals/train.sh \
	  --stage B --cfg configs/day07_trainB.yaml --max-steps 1000 \
	  --ckpt-suffix finalB --log-suffix finalB

# A memory-safe Stage-B run that preserves these envs (KEEP=1)
final-train-b-safe:
	. .venv/bin/activate && \
	FORCE_KEEP_USER_ENVS=1 \
	JOINTDIT_MAX_T=$(SAFE_MAX_T) \
	JOINTDIT_Q_CHUNK_V=$(SAFE_Q_CHUNK_V) \
	JOINTDIT_Q_CHUNK_A=$(SAFE_Q_CHUNK_A) \
	JOINTDIT_KV_DOWNSAMPLE=$(SAFE_KV_DOWNSAMPLE) \
	PYTORCH_CUDA_ALLOC_CONF="$(SAFE_ALLOC_CONF)" \
	PYTHONPATH=. scripts/finals/train.sh \
	  --stage B --cfg configs/day07_trainB.yaml --max-steps 1000 \
	  --ckpt-suffix finalB --log-suffix finalB

# =========================================================
# UI (Gradio)
# =========================================================
ui:
	. .venv/bin/activate && PYTHONPATH=. python scripts/ui/app.py

# =========================================================
# Generic inference helper (headless).
# Example:
# make infer CKPT=checkpoints/...pt REF=data/cache/meta/val/foo.json \
#      PROMPT="rainy street" NEG_PROMPT="people" \
#      STEPS=40 SEED=123 WV=1.5 WA=1.2 WT=1.5 WNT=0.8 WI=0.5 USE_IMG=1
# Or override image:
#      IMAGE=assets/samples/cond.png USE_IMG=0
# =========================================================
infer:
	. .venv/bin/activate && \
	JOINTDIT_PROMPT="$(PROMPT)" \
	JOINTDIT_NEG_PROMPT="$(NEG_PROMPT)" \
	JOINTDIT_CKPT="$(CKPT)" \
	JOINTDIT_REF_META="$(REF)" \
	JOINTDIT_STEPS="$(STEPS)" \
	JOINTDIT_SEED="$(SEED)" \
	JOINTDIT_WV="$(WV)" \
	JOINTDIT_WA="$(WA)" \
	JOINTDIT_WT="$(WT)" \
	JOINTDIT_WNT="$(WNT)" \
	JOINTDIT_WI="$(WI)" \
	JOINTDIT_USE_IMG="$(USE_IMG)" \
	JOINTDIT_IMAGE="$(IMAGE)" \
	JOINTDIT_CLIP_MODEL="$(CLIP_MODEL)" \
	JOINTDIT_CLIP_PRETRAINED="$(CLIP_PRETRAINED)" \
	PYTHONPATH=. python scripts/infer/infer_joint.py --cfg $(CFG_INFER)

# =========================================================
# Dataset fetch helpers
# =========================================================
fetch-vatex-mini:
	. .venv/bin/activate && PYTHONPATH=. python scripts/data/fetch_vatex_mini.py

# Fetch 1-second clips from AVSync15 for quick experiments
FOLDER_URL ?= https://drive.google.com/drive/folders/1onvx5y6QOceDrHZy8-ajFJ4RUuGWwT5V

deps-media:
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "Installing ffmpeg (requires sudo/apt)"; \
		apt-get update && apt-get install -y ffmpeg; \
	fi
	. .venv/bin/activate && pip install -q gdown

fetch-avsync15-1s: deps-media
	. .venv/bin/activate && PYTHONPATH=. python scripts/data/fetch_avsync15_1s.py --folder-url "$(FOLDER_URL)" --limit-train 3 --limit-val 1

# =========================================================
# System deps (Ubuntu/Debian)
# - Exports a list and installs (optional, requires sudo).
# =========================================================
sys-deps:
	. .venv/bin/activate && PYTHONPATH=. python scripts/tools/export_system_deps.py
	bash scripts/tools/install_system_deps.sh

# =========================================================
# Docs (MkDocs)
# =========================================================
DOCS_PORT ?= 8000
DOCS_HOST ?= 0.0.0.0

docs-install:
	. .venv/bin/activate && pip install -r requirements-docs.txt

docs-autodoc:
	. .venv/bin/activate && python tools/autodoc.py

docs-build: docs-autodoc
	. .venv/bin/activate && mkdocs build

docs-serve: docs-autodoc
	. .venv/bin/activate && mkdocs serve -a $(DOCS_HOST):$(DOCS_PORT)

docs-deploy: docs-autodoc
	. .venv/bin/activate && mkdocs gh-deploy --force
