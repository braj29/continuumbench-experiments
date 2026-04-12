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
SNELLIUS_RUST_MODULE="${SNELLIUS_RUST_MODULE:-}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RT_REPO_PATH="${RT_REPO_PATH:-${REPO_ROOT}/relational-transformer}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/continuumbench_rt_only"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
RT_PREPROCESSED_ROOT="${RT_PREPROCESSED_ROOT:-${HOME}/scratch/pre}"

detect_latest_rust_module() {
  module -t avail Rust 2>&1 \
    | awk '/^Rust\/[0-9]/{print}' \
    | sort -t'/' -k2,2V \
    | tail -n1
}

# Loading modules
module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

if ! command -v rustc >/dev/null 2>&1; then
  if [[ -n "${SNELLIUS_RUST_MODULE}" ]]; then
    module load "${SNELLIUS_RUST_MODULE}"
  else
    autodetected_rust_module="$(detect_latest_rust_module)"
    if [[ -n "${autodetected_rust_module}" ]]; then
      echo "Autoloading Rust module: ${autodetected_rust_module}"
      module load "${autodetected_rust_module}" || true
    fi
  fi
fi

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

if [[ ! -f "${RT_PREPROCESSED_ROOT}/${DATASET_NAME}/table_info.json" ]] || [[ ! -f "${RT_PREPROCESSED_ROOT}/${DATASET_NAME}/column_index.json" ]]; then
  echo "Missing RT preprocessed artifacts for dataset '${DATASET_NAME}'." >&2
  echo "Expected files:" >&2
  echo "  ${RT_PREPROCESSED_ROOT}/${DATASET_NAME}/table_info.json" >&2
  echo "  ${RT_PREPROCESSED_ROOT}/${DATASET_NAME}/column_index.json" >&2
  echo "Prepare/download RT preprocessed data (see relational-transformer README: Data Preparation / Download Preprocessed Data)." >&2
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
  echo "Set SNELLIUS_RUST_MODULE=<Rust/...> when submitting, e.g. after checking:" >&2
  echo "  module -t avail Rust" >&2
  module -t avail Rust 2>&1 | sed -n '1,40p' >&2 || true
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Missing cargo in PATH. Official RT requires Rust tooling (cargo)." >&2
  exit 1
fi

echo "Rust toolchain:"
rustc --version
cargo --version

rustler_has_sampler() {
  python - <<'PY'
import importlib
import sys

try:
    mod = importlib.import_module("rustler")
except Exception as exc:
    print(f"rustler import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

mod_file = getattr(mod, "__file__", None)
has_sampler = hasattr(mod, "Sampler")
print(f"rustler module: {mod_file}; has Sampler: {has_sampler}")
raise SystemExit(0 if has_sampler else 1)
PY
}

build_rustler_extension() {
  local manifest_path="${RT_REPO_PATH}/rustler/Cargo.toml"
  echo "Building RT rustler extension with maturin (release mode, --uv)..."
  if (
    cd "${RT_REPO_PATH}/rustler" &&
    python -m maturin develop --release --uv --manifest-path "${manifest_path}"
  ); then
    return 0
  fi

  echo "maturin --uv build failed; retrying without --uv..."
  if (
    cd "${RT_REPO_PATH}/rustler" &&
    python -m maturin develop --release --manifest-path "${manifest_path}"
  ); then
    return 0
  fi

  return 1
}

remove_conflicting_rustler_package() {
  python - <<'PY'
import importlib
import os
import sys

try:
    mod = importlib.import_module("rustler")
except Exception:
    raise SystemExit(0)

if hasattr(mod, "Sampler"):
    raise SystemExit(0)

mod_file = getattr(mod, "__file__", "") or ""
if mod_file.endswith(".py") or mod_file.endswith(".pyc") or "__init__.py" in mod_file:
    print(f"Detected conflicting python rustler package at: {mod_file}")
    raise SystemExit(1)

raise SystemExit(0)
PY
}

if ! rustler_has_sampler; then
  if ! remove_conflicting_rustler_package; then
    echo "Removing conflicting non-extension rustler package from venv..."
    python -m pip uninstall -y rustler || true
  fi
  if ! build_rustler_extension; then
    echo "Failed to build rustler extension with maturin." >&2
    echo "Diagnostics:" >&2
    python -m maturin --version >&2 || true
    rustc --version >&2 || true
    cargo --version >&2 || true
    echo "Hint: this often means the loaded Rust module is too old for edition 2024." >&2
    echo "Try a newer module via SNELLIUS_RUST_MODULE (check: module -t avail Rust)." >&2
    exit 1
  fi
fi

if ! rustler_has_sampler; then
  echo "rustler extension is still unavailable (Sampler missing)." >&2
  echo "Check build logs above for Rust/maturin errors." >&2
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

# Avoid interactive wandb prompts on cluster jobs.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

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
