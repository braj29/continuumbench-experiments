#!/usr/bin/env bash
# Official Relational Transformer only (no joined-table or graph tracks)
# Usage:
#   sbatch --export=ALL,DATASET_NAME=rel-f1,TASK_NAME=driver-top3,SEED=7 \
#     scripts/snellius_rt_only_a100.sbatch

# Set job requirements
#SBATCH -J cb-rt-only
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH -p gpu_a100
#SBATCH -t 08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Job configuration
DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-0}"
MODELS="${MODELS:-tabpfn}" # Ignored by --relational-only except as folder key.
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RT_REPO_PATH="${RT_REPO_PATH:-${REPO_ROOT}/relational-transformer}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/continuumbench_rt_only"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

# Loading modules
module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

cd "${REPO_ROOT}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}" >&2
  echo "Run ./scripts/setup_snellius_env.sh first." >&2
  exit 1
fi

if [[ ! -f "${RT_REPO_PATH}/rt/main.py" ]]; then
  echo "Missing official RT checkout at ${RT_REPO_PATH}" >&2
  echo "Set RT_REPO_PATH=/path/to/relational-transformer (must contain rt/main.py)." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-${CACHE_ROOT}/relbench}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${CACHE_ROOT}/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1
export RT_REPO_PATH

ensure_python_module() {
  local module_name="$1"
  local package_name="$2"
  if ! python -c "import ${module_name}" >/dev/null 2>&1; then
    echo "Installing missing RT dependency: ${package_name} (import ${module_name})"
    python -m pip install --upgrade "${package_name}"
  fi
}

# Core RT runtime deps imported by rt/main.py and rt/data.py.
ensure_python_module "wandb" "wandb"
ensure_python_module "strictfire" "strictfire"
ensure_python_module "maturin_import_hook" "maturin-import-hook"
ensure_python_module "maturin" "maturin[patchelf]"
ensure_python_module "ml_dtypes" "ml-dtypes"
ensure_python_module "orjson" "orjson"
ensure_python_module "einops" "einops"
ensure_python_module "sentence_transformers" "sentence-transformers"
ensure_python_module "polars" "polars"
ensure_python_module "uv" "uv"

if ! command -v rustc >/dev/null 2>&1; then
  echo "Missing rustc in PATH. Official RT requires Rust to build rustler." >&2
  echo "Load a Rust toolchain module before submitting (e.g. check 'module avail Rust')." >&2
  exit 1
fi

# Fail fast if RT import path is still broken.
python - <<'PY'
import os
import sys

repo = os.environ["RT_REPO_PATH"]
if repo not in sys.path:
    sys.path.insert(0, repo)

import rt.main  # noqa: F401
import rt.data  # noqa: F401

print("RT import preflight passed.")
PY

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Create output directory on scratch
mkdir -p "${JOB_OUTPUT_DIR}" "${RESULTS_DIR}"

download_artifacts_flag="--no-download-artifacts"
case "$(lowercase "${DOWNLOAD_ARTIFACTS}")" in
  1|true|yes)
    download_artifacts_flag="--download-artifacts"
    ;;
  0|false|no)
    ;;
  *)
    echo "DOWNLOAD_ARTIFACTS must be one of: true, false, 1, 0, yes, no" >&2
    exit 1
    ;;
esac

# Execute benchmark: official RT on relational view only (Protocol A only)
python "${REPO_ROOT}/continuumbench_tabfm_run.py" \
  --task-source dataset \
  --dataset-name "${DATASET_NAME}" \
  --task-name "${TASK_NAME}" \
  --models "${MODELS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${download_artifacts_flag}" \
  --use-official-rt-relational \
  --rt-repo-path "${RT_REPO_PATH}" \
  --relational-only \
  --no-run-protocol-b \
  --output-dir "${JOB_OUTPUT_DIR}"

# Copy output directory from scratch back to the repo
cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "Results copied to ${RESULTS_DIR}"
