#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python}"
: "${PYTHONPATH:=.}"
: "${PYTORCH_CUDA_ALLOC_CONF:=max_split_size_mb:128}"
: "${JOINTDIT_Q_CHUNK_V:=128}"
: "${JOINTDIT_Q_CHUNK_A:=0}"
: "${JOINTDIT_KV_DOWNSAMPLE:=4}"

cfg="configs/day06_infer.yaml"
ckpt=""
out_dir="outputs/final"
seeds="0"
steps="30"
mp4_fps=""
audio_sr=""

usage() {
  cat <<EOF
Usage: scripts/finals/infer.sh [--cfg cfg.yaml] --ckpt /path/to/ckpt.pt [--out-dir DIR] [--seeds "0,1,2"] [--steps N] [--mp4-fps N] [--audio-sr N]

Env knobs:
  PYTORCH_CUDA_ALLOC_CONF   ${PYTORCH_CUDA_ALLOC_CONF}
  JOINTDIT_Q_CHUNK_V        ${JOINTDIT_Q_CHUNK_V}
  JOINTDIT_Q_CHUNK_A        ${JOINTDIT_Q_CHUNK_A}
  JOINTDIT_KV_DOWNSAMPLE    ${JOINTDIT_KV_DOWNSAMPLE}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) cfg="${2}"; shift 2;;
    --ckpt) ckpt="${2}"; shift 2;;
    --out-dir) out_dir="${2}"; shift 2;;
    --seeds) seeds="${2}"; shift 2;;
    --steps) steps="${2}"; shift 2;;
    --mp4-fps) mp4_fps="${2}"; shift 2;;
    --audio-sr) audio_sr="${2}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

[[ -n "${ckpt}" ]] || { echo "[err] --ckpt is required"; usage; exit 2; }
mkdir -p "${out_dir}"

if [[ -d ".venv" ]]; then source .venv/bin/activate; fi
export PYTHONPATH="${PYTHONPATH}"

# Build a temp cfg that overrides ckpt/out_dir/seeds/steps/fps/sr safely via PyYAML
tmp_cfg="$(mktemp -t infer_cfg_XXXX.yaml)"
python - <<PY > "${tmp_cfg}"
import sys, yaml, pathlib, os
base = "${cfg}"
ckpt = "${ckpt}"
out_dir = "${out_dir}"
seeds = [int(s) for s in "${seeds}".split(",") if s.strip()!=""]
steps = int("${steps}")
mp4_fps = "${mp4_fps}".strip()
audio_sr = "${audio_sr}".strip()

with open(base,"r") as f:
    cfg = yaml.safe_load(f)

cfg["ckpt"] = ckpt
cfg["out_dir"] = out_dir
cfg["seeds"] = seeds
cfg["steps"] = steps
if mp4_fps:
    cfg["mp4_fps"] = int(mp4_fps)
if audio_sr:
    cfg["audio_sr"] = int(audio_sr)

yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
PY

echo "[infer] cfg=${cfg} → ${tmp_cfg}"
echo "[infer] ckpt=${ckpt} out_dir=${out_dir} seeds=${seeds} steps=${steps}"
echo "[env]   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}  JOINTDIT_Q_CHUNK_V=${JOINTDIT_Q_CHUNK_V}  JOINTDIT_Q_CHUNK_A=${JOINTDIT_Q_CHUNK_A}  JOINTDIT_KV_DOWNSAMPLE=${JOINTDIT_KV_DOWNSAMPLE}"

set -x
"${PYTHON}" -u scripts/infer/infer_joint.py --cfg "${tmp_cfg}"
