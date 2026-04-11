#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
RUN_ROOT="${RUN_ROOT:-/scratch-shared/$USER/continuumbench-runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/${SLURM_JOB_ID:-manual}}"
RESULTS_COPY_DIR="${RESULTS_COPY_DIR:-$REPO_ROOT/outputs/snellius/${SLURM_JOB_ID:-manual}}"
TASK_SOURCE="${TASK_SOURCE:-dataset}"
DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
MODELS="${MODELS:-tabpfn}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-7}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-0}"

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

download_artifacts_spec="$(lowercase "${DOWNLOAD_ARTIFACTS}")"
case "${download_artifacts_spec}" in
  1|true|yes)
    download_artifacts_args=(--download-artifacts)
    ;;
  0|false|no)
    download_artifacts_args=(--no-download-artifacts)
    ;;
  *)
    echo "DOWNLOAD_ARTIFACTS must be one of: true, false, 1, 0, yes, no" >&2
    exit 1
    ;;
esac

if command -v module >/dev/null 2>&1; then
  if [[ -n "${SNELLIUS_STACK_MODULE}" ]]; then
    module load "${SNELLIUS_STACK_MODULE}"
  fi
  if [[ -n "${SNELLIUS_PYTHON_MODULE}" ]]; then
    module load "${SNELLIUS_PYTHON_MODULE}"
  fi
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}" >&2
  echo "Run scripts/setup_snellius_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-$CACHE_ROOT/relbench}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$CACHE_ROOT/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${RELBENCH_CACHE_DIR}" \
  "${TABPFN_MODEL_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${OUTPUT_ROOT}" \
  "${RESULTS_COPY_DIR}"

python - <<'PY'
import os

import torch

print(f"Python executable: {os.sys.executable}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count visible to job: {torch.cuda.device_count()}")
PY

cmd=(
  python
  continuumbench_tabfm_run.py
  --task-source "${TASK_SOURCE}"
  --models "${MODELS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_ROOT}"
)

if [[ "${TASK_SOURCE}" == "dataset" ]]; then
  cmd+=(
    --dataset-name "${DATASET_NAME}"
    --task-name "${TASK_NAME}"
    "${download_artifacts_args[@]}"
  )
fi

echo "+ ${cmd[*]}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks=1 "${cmd[@]}"
else
  "${cmd[@]}"
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${OUTPUT_ROOT}/" "${RESULTS_COPY_DIR}/"
else
  cp -R "${OUTPUT_ROOT}/." "${RESULTS_COPY_DIR}/"
fi

echo "Scratch outputs: ${OUTPUT_ROOT}"
echo "Copied results to: ${RESULTS_COPY_DIR}"
