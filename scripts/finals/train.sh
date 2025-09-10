#!/usr/bin/env bash
# scripts/finals/train.sh
set -euo pipefail

: "${PYTHON:=python}"
: "${PYTHONPATH:=.}"

usage() {
  cat <<'EOF'
Usage:
  scripts/finals/train.sh --cfg <path.yaml>
                          [--stage A|B]
                          [--max-steps N]
                          [--ckpt-suffix S]
                          [--log-suffix S]
                          [--resume PATH|auto]
                          [--init-from PATH|DIR|auto|none]   # Stage-B init source (defaults to 'auto' for B)
                          [--allow-scratch]                  # Allow Stage-B to start without Stage-A ckpt

Env knobs (auto-set per VRAM; override by exporting FORCE_KEEP_USER_ENVS=1):
  JOINTDIT_MAX_T
  JOINTDIT_Q_CHUNK_V
  JOINTDIT_Q_CHUNK_A
  JOINTDIT_KV_DOWNSAMPLE
  PYTORCH_CUDA_ALLOC_CONF
EOF
}

# ---------------- Parse args ----------------
stage="A"
cfg=""
max_steps=""
ckpt_suffix=""
log_suffix=""
resume=""
init_from=""
allow_scratch="0"
declare -a passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)        stage="${2}"; shift 2;;
    --cfg)          cfg="${2}"; shift 2;;
    --max-steps)    max_steps="${2}"; shift 2;;
    --ckpt-suffix)  ckpt_suffix="${2}"; shift 2;;
    --log-suffix)   log_suffix="${2}"; shift 2;;
    --resume)       resume="${2}"; shift 2;;
    --init-from)    init_from="${2}"; shift 2;;
    --allow-scratch)allow_scratch="1"; shift 1;;
    -h|--help)      usage; exit 0;;
    *)              # forward unknown flags/args to python entrypoint
                    passthrough+=("$1"); shift 1;;
  esac
done

[[ -n "${cfg}" ]] || { echo "[err] --cfg is required"; usage; exit 2; }

# ---------------- Python env ----------------
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="${PYTHONPATH}"

# ---------------- Entrypoint by stage ----------------
case "${stage^^}" in
  A) entry="scripts/train/train_stage_a.py" ;;
  B) entry="scripts/train/train_stage_b.py" ;;
  *) echo "[err] --stage must be A or B"; exit 2 ;;
esac

# ---------------- Detect VRAM + choose bucket ----------------
vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 48)"
bucket="48"
if   (( vram >= 80 )); then bucket="80"
elif (( vram >= 56 )); then bucket="64"
fi

KEEP="${FORCE_KEEP_USER_ENVS:-0}"

# ---------------- Recommended defaults per bucket ----------------
# If KEEP!=1, force VRAM-safe defaults for reliability.
if [[ "${KEEP}" != "1" ]]; then
  if [[ "${bucket}" == "48" ]]; then
    export JOINTDIT_MAX_T=6
    export JOINTDIT_Q_CHUNK_V=64
    export JOINTDIT_Q_CHUNK_A=0
    export JOINTDIT_KV_DOWNSAMPLE=8
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
  elif [[ "${bucket}" == "64" ]]; then
    export JOINTDIT_MAX_T=8
    export JOINTDIT_Q_CHUNK_V=96
    export JOINTDIT_Q_CHUNK_A=0
    export JOINTDIT_KV_DOWNSAMPLE=6
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
  else # 80+
    export JOINTDIT_MAX_T=12
    export JOINTDIT_Q_CHUNK_V=128
    export JOINTDIT_Q_CHUNK_A=0
    export JOINTDIT_KV_DOWNSAMPLE=4
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
  fi
else
  # KEEP=1 => only set sensible defaults if not provided by user
  if [[ "${bucket}" == "64" ]]; then
    : "${JOINTDIT_MAX_T:=8}"
    : "${JOINTDIT_Q_CHUNK_V:=96}"
    : "${JOINTDIT_Q_CHUNK_A:=0}"
    : "${JOINTDIT_KV_DOWNSAMPLE:=6}"
  elif [[ "${bucket}" == "80" ]]; then
    : "${JOINTDIT_MAX_T:=12}"
    : "${JOINTDIT_Q_CHUNK_V:=128}"
    : "${JOINTDIT_Q_CHUNK_A:=0}"
    : "${JOINTDIT_KV_DOWNSAMPLE:=4}"
  else
    : "${JOINTDIT_MAX_T:=6}"
    : "${JOINTDIT_Q_CHUNK_V:=64}"
    : "${JOINTDIT_Q_CHUNK_A:=0}"
    : "${JOINTDIT_KV_DOWNSAMPLE:=8}"
  fi
  : "${PYTORCH_CUDA_ALLOC_CONF:=max_split_size_mb:128}"
  export JOINTDIT_MAX_T JOINTDIT_Q_CHUNK_V JOINTDIT_Q_CHUNK_A JOINTDIT_KV_DOWNSAMPLE PYTORCH_CUDA_ALLOC_CONF
fi

# Safety clamp: never allow T above bucket cap
max_t_cap=6
[[ "${bucket}" == "64" ]] && max_t_cap=8
[[ "${bucket}" == "80" ]] && max_t_cap=12
if (( JOINTDIT_MAX_T > max_t_cap )); then
  echo "[warn] JOINTDIT_MAX_T=${JOINTDIT_MAX_T} > cap=${max_t_cap} for ${vram}GB; clamping."
  export JOINTDIT_MAX_T="${max_t_cap}"
fi

# ---------------- Build arg list ----------------
args=( --cfg "${cfg}" )
[[ -n "${max_steps}"    ]] && args+=( --max-steps "${max_steps}" )
[[ -n "${ckpt_suffix}"  ]] && args+=( --ckpt-suffix "${ckpt_suffix}" )
[[ -n "${log_suffix}"   ]] && args+=( --log-suffix "${log_suffix}" )
[[ -n "${resume}"       ]] && args+=( --resume "${resume}" )

# Stage-B: default to --init-from auto unless user provided something
if [[ "${stage^^}" == "B" ]]; then
  if [[ -z "${init_from}" ]]; then init_from="auto"; fi
  args+=( --init-from "${init_from}" )
  if [[ "${allow_scratch}" == "1" ]]; then
    args+=( --allow-scratch )
  fi
fi

# Forward any extra flags (from Makefile $(EXTRA) or your CLI)
if (( ${#passthrough[@]} )); then
  args+=( "${passthrough[@]}" )
fi

echo "[train] stage=${stage} cfg=${cfg}"
echo "[env]   VRAM=${vram}GB bucket=${bucket}  KEEP=${KEEP}"
echo "[env]   JOINTDIT_MAX_T=${JOINTDIT_MAX_T}  JOINTDIT_Q_CHUNK_V=${JOINTDIT_Q_CHUNK_V}  JOINTDIT_Q_CHUNK_A=${JOINTDIT_Q_CHUNK_A}  JOINTDIT_KV_DOWNSAMPLE=${JOINTDIT_KV_DOWNSAMPLE}"
echo "[env]   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

set -x
"${PYTHON}" -u "${entry}" "${args[@]}"
