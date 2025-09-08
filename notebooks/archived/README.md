
# Notebooks: Daily Runs & Repro Steps

This folder contains lightweight, **runnable notebooks** that document what we did on each day, plus a bootstrap notebook to quickly rehydrate a new machine (Runpod or local). Each notebook is designed so you can **trace + re-run** commands end-to-end.

---

## Files here

- `00_bootstrap_on_start.ipynb` — one-time (or every new session) setup: create/activate venv, validate GPU, set env vars, verify repo layout, and run both SVD & ALDM2 smoke tests.
- `01_day1_walkthrough.ipynb` — Day 1 log: ALDM2 config patching (cross-attention dim), strict loads, repo cleanup, and smoke runs.
- `02_day2_cache_latents.ipynb` — Day 2: create folder layout, dummy/raw mini-dataset, cache **video/audio latents**, and a smoke to inspect shapes.
- `03_day3_unet_slicers.ipynb` — Day 3: SVD/ALDM2 **UNet slicers** (param counts, shape-preserving forward), handling ALDM2 cross-attn config quirks, and optionally mirroring real weights.
- `04_day4_joint_skeleton.ipynb` — Day 4: **JointDiT skeleton** with iso/full modes, dtype hygiene fixes, and shape checks.
- `05_day5_minitrain.ipynb` — Day 5: tiny **Stage-A** training loop w/ memory savers (chunking, KV downsample, max T) and checkpointing.
- `06_day6_infer.ipynb` — Day 6: **joint inference** (video+audio) with VAEs; write MP4+WAV.
- `07_day7_finetune_and_infer.ipynb` — Day 7: **Stage-B expert fine-tune** (+ I/O), then joint inference from the new Stage-B checkpoint.

> Naming: `NN_dayN_title.ipynb` where `NN` is zero-padded order.

---

## Usage

### Open & run interactively
- **VS Code**: open the `.ipynb`, select the Python kernel from `.venv`, run all cells.
- **JupyterLab/Notebook**:
  ```bash
  source .venv/bin/activate
  jupyter lab  # or: jupyter notebook
  ```

### Run non-interactively (headless)

```bash
source .venv/bin/activate
python -m pip install nbconvert  # once

# examples
jupyter nbconvert --to notebook --execute --inplace notebooks/day_07/07_day7_finetune_and_infer.ipynb
```

---

## Environment expectations

* Project root: `/workspace/jointdit`
* Python venv: `.venv/` at repo root
* CUDA visible:

  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
* Helpful allocator setting(s):

  ```bash
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
  ```
* Common memory levers read by the model:

  ```bash
  export JOINTDIT_MAX_T=6
  export JOINTDIT_Q_CHUNK_V=128
  export JOINTDIT_Q_CHUNK_A=0
  export JOINTDIT_KV_DOWNSAMPLE=4
  ```

---

## Make targets (quick reference)

```Makefile
smoke-svd:
	. .venv/bin/activate && python scripts/smoke/svd_smoke.py

smoke-aldm2:
	. .venv/bin/activate && python scripts/smoke/aldm2_smoke.py

smoke-day02:
	. .venv/bin/activate && PYTHONPATH=. python scripts/data/smoke_day02_cache.py --split train --n 3

smoke-day03:
	. .venv/bin/activate && PYTHONPATH=. python scripts/smoke/day03_slicers_smoke.py

smoke-day04:
	. .venv/bin/activate && PYTHONPATH=. python scripts/smoke/day04_joint_smoke.py

smoke-day05:
	. .venv/bin/activate && PYTHONPATH=. python scripts/train/train_stage_a.py --cfg configs/day05_train.yaml --max-steps 25 --log-suffix smoke --ckpt-suffix smoke

smoke-day06:
	. .venv/bin/activate && PYTHONPATH=. python scripts/infer/infer_joint.py --cfg configs/day06_infer.yaml

day7-smoke:
	. .venv/bin/activate && PYTHONPATH=. python scripts/train/train_stage_b.py --cfg configs/day07_trainB.yaml --max-steps 100 --log-suffix smoke --ckpt-suffix smoke

day7-infer:
	. .venv/bin/activate && PYTHONPATH=. python scripts/infer/infer_joint.py --cfg configs/day07_infer.yaml
```

---

## Day 7 quick recipe (CLI)

```bash
# memory knobs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export JOINTDIT_Q_CHUNK_V=128
export JOINTDIT_Q_CHUNK_A=0
export JOINTDIT_KV_DOWNSAMPLE=4
export JOINTDIT_MAX_T=6

# train Stage-B (experts)
make day7-smoke            # or raise --max-steps via manual python call

# point inference config at the Stage-B ckpt you want
cp configs/day06_infer.yaml configs/day07_infer.yaml
sed -i 's#^ckpt:.*#ckpt: /workspace/jointdit/checkpoints/day07_stage_b_d7/ckpt_step_001000.pt#' configs/day07_infer.yaml
sed -i 's#^out_dir:.*#out_dir: /workspace/jointdit/outputs/day07#' configs/day07_infer.yaml

# run inference
make day7-infer
```

---

## Troubleshooting quickies

* **“No Stage-B ckpt found”**: your Make rule probably checks a fixed folder. Either update `configs/day07_infer.yaml: ckpt:` to your actual file or symlink your ckpt into the expected path:

  ```bash
  mkdir -p checkpoints/day07_stage_b
  ln -sfn /workspace/jointdit/checkpoints/day07_stage_b_d7/ckpt_step_001000.pt checkpoints/day07_stage_b/ckpt_step_001000.pt
  ```
* **GradScaler: “Attempting to unscale FP16 gradients.”**
  We only use AMP if `runtime.dtype=fp16` in the config. Otherwise we train in fp32 and skip unscale.
* **OOM**: lower `JOINTDIT_MAX_T` (e.g., 4), increase `JOINTDIT_Q_CHUNK_V`, or raise `JOINTDIT_KV_DOWNSAMPLE`. Keep `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`.

---

Happy shipping 🚀

