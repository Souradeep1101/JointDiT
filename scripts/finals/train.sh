#!/usr/bin/env bash
# scripts/finals/train.sh
set -euo pipefail

: "${PYTHON:=python}"
: "${PYTHONPATH:=.}"

# --- Parse args ---
stage="A"; cfg=""; max_steps=""; ckpt_suffix=""; log_suffix=""; resume=""
usage() {
  cat <<EOF
Usage: scripts/finals/train.sh --cfg <path.yaml> [--stage A|B] [--max-steps N] [--ckpt-suffix S] [--log-suffix S] [--resume PATH|auto]
Env knobs (auto-set per VRAM; override by exporting FORCE_KEEP_USER_ENVS=1):
  JOINTDIT_MAX_T
  JOINTDIT_Q_CHUNK_V
  JOINTDIT_Q_CHUNK_A
  JOINTDIT_KV_DOWNSAMPLE
  PYTORCH_CUDA_ALLOC_CONF
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2}"; shift 2;;
    --cfg) cfg="${2}"; shift 2;;
    --max-steps) max_steps="${2}"; shift 2;;
    --ckpt-suffix) ckpt_suffix="${2}"; shift 2;;
    --log-suffix) log_suffix="${2}"; shift 2;;
    --resume) resume="${2}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done
[[ -n "${cfg}" ]] || { echo "[err] --cfg is required"; usage; exit 2; }

# --- Python env ---
if [[ -d ".venv" ]]; then source .venv/bin/activate; fi
export PYTHONPATH="${PYTHONPATH}"

# --- Pick entrypoint by stage ---
if [[ "${stage^^}" == "A" ]]; then entry="scripts/train/train_stage_a.py"
elif [[ "${stage^^}" == "B" ]]; then entry="scripts/train/train_stage_b.py"
else echo "[err] --stage must be A or B"; exit 2; fi

# --- Detect VRAM (fallback=48) ---
vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 48)"
# buckets: <=56 => 48/40GB; 57-79 => 64GB-ish; >=80 => 80GB+
bucket="48"
if   (( vram >= 80 )); then bucket="80"
elif (( vram >= 56 )); then bucket="64"
fi

# honor user envs only if explicitly requested
KEEP="${FORCE_KEEP_USER_ENVS:-0}"

# --- Recommended defaults per bucket ---
# We always force the 48GB-safe combo unless KEEP=1
if [[ "${bucket}" == "48" && "${KEEP}" != "1" ]]; then
  export JOINTDIT_MAX_T=6
  export JOINTDIT_Q_CHUNK_V=64
  export JOINTDIT_Q_CHUNK_A=0
  export JOINTDIT_KV_DOWNSAMPLE=8
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
else
  # Otherwise set sensible defaults only if not provided
  if [[ "${bucket}" == "64" ]]; then
    : "${JOINTDIT_MAX_T:=8}"
    : "${JOINTDIT_Q_CHUNK_V:=96}"
    : "${JOINTDIT_Q_CHUNK_A:=0}"
    : "${JOINTDIT_KV_DOWNSAMPLE:=6}"
  else # 80+
    : "${JOINTDIT_MAX_T:=12}"
    : "${JOINTDIT_Q_CHUNK_V:=128}"
    : "${JOINTDIT_Q_CHUNK_A:=0}"
    : "${JOINTDIT_KV_DOWNSAMPLE:=4}"
  fi
  : "${PYTORCH_CUDA_ALLOC_CONF:=max_split_size_mb:128}"
  export JOINTDIT_MAX_T JOINTDIT_Q_CHUNK_V JOINTDIT_Q_CHUNK_A JOINTDIT_KV_DOWNSAMPLE PYTORCH_CUDA_ALLOC_CONF
fi

# Safety clamp: never allow T above bucket
max_t_cap=6; [[ "${bucket}" == "64" ]] && max_t_cap=8; [[ "${bucket}" == "80" ]] && max_t_cap=12
if (( JOINTDIT_MAX_T > max_t_cap )); then
  echo "[warn] JOINTDIT_MAX_T=${JOINTDIT_MAX_T} > cap=${max_t_cap} for ${vram}GB; clamping."
  export JOINTDIT_MAX_T="${max_t_cap}"
fi

# --- Build arg list ---
args=( --cfg "${cfg}" )
[[ -n "${max_steps}" ]]  && args+=( --max-steps "${max_steps}" )
[[ -n "${ckpt_suffix}" ]]&& args+=( --ckpt-suffix "${ckpt_suffix}" )
[[ -n "${log_suffix}" ]] && args+=( --log-suffix "${log_suffix}" )
[[ -n "${resume}" ]]     && args+=( --resume "${resume}" )

echo "[train] stage=${stage} cfg=${cfg}"
echo "[env]   VRAM=${vram}GB bucket=${bucket}  KEEP=${KEEP}"
echo "[env]   JOINTDIT_MAX_T=${JOINTDIT_MAX_T}  JOINTDIT_Q_CHUNK_V=${JOINTDIT_Q_CHUNK_V}  JOINTDIT_Q_CHUNK_A=${JOINTDIT_Q_CHUNK_A}  JOINTDIT_KV_DOWNSAMPLE=${JOINTDIT_KV_DOWNSAMPLE}"
echo "[env]   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

set -x
"${PYTHON}" -u "${entry}" "${args[@]}"
