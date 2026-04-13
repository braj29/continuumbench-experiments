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

# Pretrained checkpoint options (both default to 0 / disabled).
#   RT_USE_PRETRAINED=1  → download pretrain_<dataset>_<task>.pt from HF Hub
#                          and pass it as --rt-load-ckpt-path.
#   RT_ZERO_SHOT=1       → combine with RT_USE_PRETRAINED=1 to skip fine-tuning
#                          entirely (zero-shot evaluation of the foundation model).
#   RT_CKPT_PATH         → supply an explicit local checkpoint path instead of
#                          auto-downloading (overrides RT_USE_PRETRAINED).
RT_USE_PRETRAINED="${RT_USE_PRETRAINED:-0}"
RT_ZERO_SHOT="${RT_ZERO_SHOT:-0}"
RT_CKPT_PATH="${RT_CKPT_PATH:-}"
RT_CKPT_DIR="${RT_CKPT_DIR:-${HOME}/scratch/rt_ckpts}"
# HF repo that hosts the pretrained checkpoints.
RT_HF_REPO="${RT_HF_REPO:-rishabh-ranjan/relational-transformer}"

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
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Missing cargo in PATH. Official RT requires Rust tooling (cargo)." >&2
fi

# ---------------------------------------------------------------------------
# Rust version check: rkyv >= 0.8 requires rustc >= 1.81.
# Snellius 2024 module stack caps at Rust 1.78.  Fall back to the rustup-
# managed stable toolchain cached in the repo by setup_snellius_env.sh.
# ---------------------------------------------------------------------------
_RUSTUP_CACHE_HOME="${REPO_ROOT}/.rustup"
_CARGO_CACHE_HOME="${REPO_ROOT}/.cargo"
_NEED_RUSTC_MINOR=81

_rustc_minor() {
  rustc --version 2>/dev/null | awk '{print $2}' | cut -d. -f2
}

_ensure_rustc_min_version() {
  local cur_minor
  cur_minor="$(_rustc_minor 2>/dev/null || echo '0')"

  if [[ "${cur_minor}" -ge "${_NEED_RUSTC_MINOR}" ]]; then
    echo "rustc $(rustc --version | awk '{print $2}') >= 1.${_NEED_RUSTC_MINOR} — OK"
    return 0
  fi

  local cur_ver
  cur_ver="$(rustc --version 2>/dev/null || echo 'none')"
  echo "rustc ${cur_ver} < 1.${_NEED_RUSTC_MINOR}; falling back to rustup cache..."

  if [[ -x "${_CARGO_CACHE_HOME}/bin/rustc" ]]; then
    export RUSTUP_HOME="${_RUSTUP_CACHE_HOME}"
    export CARGO_HOME="${_CARGO_CACHE_HOME}"
    export PATH="${_CARGO_CACHE_HOME}/bin:${PATH}"
    local new_minor
    new_minor="$(_rustc_minor)"
    if [[ "${new_minor}" -ge "${_NEED_RUSTC_MINOR}" ]]; then
      echo "Using rustup rustc: $(rustc --version)"
      return 0
    fi
    echo "Cached rustup toolchain is also too old: $(rustc --version)" >&2
  fi

  cat >&2 <<RUSTUP_ERR

ERROR: rustc >= 1.${_NEED_RUSTC_MINOR} is required to build the RT rustler extension
       (rkyv >= 0.8 in rustler/Cargo.lock).

The Snellius module '${SNELLIUS_RUST_MODULE}' provides rustc ${cur_ver}, which is too old,
and no compatible cached toolchain was found at ${_CARGO_CACHE_HOME}/bin.

Fix: run setup_snellius_env.sh once on a login node (internet required):
  cd ${REPO_ROOT}
  ./scripts/setup_snellius_env.sh

This installs a stable rustup toolchain into .rustup / .cargo inside the repo.
Compute nodes will use it automatically on the next submission.

RUSTUP_ERR
  return 1
}

_ensure_rustc_min_version

echo "Rust toolchain:"
rustc --version
cargo --version

edition_fix_if_needed() {
  local cargo_toml="${RT_REPO_PATH}/rustler/Cargo.toml"
  local cargo_ver
  cargo_ver="$(cargo --version | awk '{print $2}')"
  local cargo_major cargo_minor
  cargo_major="$(printf '%s' "${cargo_ver}" | cut -d. -f1)"
  cargo_minor="$(printf '%s' "${cargo_ver}" | cut -d. -f2)"

  # edition2024 is stabilized in newer cargo/rust toolchains.
  if [[ "${cargo_major}" -eq 1 && "${cargo_minor}" -lt 85 ]]; then
    if grep -q 'edition = "2024"' "${cargo_toml}"; then
      echo "Detected cargo ${cargo_ver} (<1.85): patching rustler edition 2024 -> 2021 for compatibility."
      sed -i 's/edition = "2024"/edition = "2021"/' "${cargo_toml}"
    fi
  fi
}

edition_fix_if_needed

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

# ---------------------------------------------------------------------------
# RT preprocessing: nodes.rkyv / offsets.rkyv / p2f_adj.rkyv / text_emb_*.bin
# ---------------------------------------------------------------------------
# pre.rs hardcodes ~/scratch/relbench/{db} for input and
# ~/scratch/pre/{db} for output.  We make ~/scratch/relbench a symlink to
# the relbench cache used by the rest of the pipeline.
RT_PRE_DIR="${RT_PREPROCESSED_ROOT}/${DATASET_NAME}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L12-v2}"

# 1. Ensure ~/scratch/relbench -> $RELBENCH_CACHE_DIR symlink exists.
mkdir -p "${HOME}/scratch"
if [[ ! -e "${HOME}/scratch/relbench" ]]; then
  echo "Creating symlink ${HOME}/scratch/relbench -> ${RELBENCH_CACHE_DIR}"
  ln -s "${RELBENCH_CACHE_DIR}" "${HOME}/scratch/relbench"
elif [[ ! -L "${HOME}/scratch/relbench" ]]; then
  echo "WARNING: ${HOME}/scratch/relbench exists but is not a symlink." >&2
  echo "Expected it to point to ${RELBENCH_CACHE_DIR}." >&2
fi

# 2. Download RelBench parquet files if not present (pre.rs reads them).
python - <<PY
import os, sys
sys.path.insert(0, "${RT_REPO_PATH}")
os.environ.setdefault("RELBENCH_CACHE_DIR", "${RELBENCH_CACHE_DIR}")
from relbench.datasets import get_dataset
from relbench.tasks import get_task
print("Ensuring RelBench parquet data for ${DATASET_NAME}/${TASK_NAME}...")
get_dataset("${DATASET_NAME}", download=True)
get_task("${DATASET_NAME}", "${TASK_NAME}", download=True)
print("RelBench data ready.")
PY

# 3. Build and run the rustler pre binary if rkyv files are missing.
RUSTLER_BIN="${RT_REPO_PATH}/rustler/target/release/rustler"

rt_rkyv_files_exist() {
  [[ -f "${RT_PRE_DIR}/nodes.rkyv"      ]] \
  && [[ -f "${RT_PRE_DIR}/offsets.rkyv"    ]] \
  && [[ -f "${RT_PRE_DIR}/p2f_adj.rkyv"   ]] \
  && [[ -f "${RT_PRE_DIR}/table_info.json" ]] \
  && [[ -f "${RT_PRE_DIR}/column_index.json" ]]
}

if rt_rkyv_files_exist; then
  echo "RT rkyv files already exist for ${DATASET_NAME}; skipping rustler pre."
else
  echo "Running rustler pre ${DATASET_NAME}..."
  if [[ ! -x "${RUSTLER_BIN}" ]]; then
    echo "Building rustler binary (cargo build --release)..."
    (
      cd "${RT_REPO_PATH}/rustler"
      cargo build --release --bin rustler
    )
  fi
  mkdir -p "${RT_PRE_DIR}"
  "${RUSTLER_BIN}" pre "${DATASET_NAME}"
fi

# 4. Generate text embeddings if missing.
TEXT_EMB_PATH="${RT_PRE_DIR}/text_emb_${EMBEDDING_MODEL}.bin"
if [[ -f "${TEXT_EMB_PATH}" ]]; then
  echo "Text embeddings already exist (${TEXT_EMB_PATH}); skipping embed step."
else
  echo "Generating text embeddings (${EMBEDDING_MODEL}) for ${DATASET_NAME}..."
  # Run embed.py as a real file so that multiprocessing.spawn can re-open it.
  # (sentence_transformers uses multiprocessing; python - <<heredoc has no path.)
  PYTHONPATH="${RT_REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${RT_REPO_PATH}/rt/embed.py" \
      "${DATASET_NAME}" \
      --embedding_model "${EMBEDDING_MODEL}"
  echo "Text embeddings written to ${TEXT_EMB_PATH}"
fi

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

# ---------------------------------------------------------------------------
# Pretrained checkpoint: download from HuggingFace Hub if requested.
# ---------------------------------------------------------------------------
rt_ckpt_flags=()

if [[ -n "${RT_CKPT_PATH}" ]]; then
  # Caller supplied an explicit checkpoint path — use it directly.
  echo "Using explicit checkpoint: ${RT_CKPT_PATH}"
  rt_ckpt_flags+=("--rt-load-ckpt-path" "${RT_CKPT_PATH}")
elif [[ "${RT_USE_PRETRAINED}" == "1" ]]; then
  # Auto-download: pretrain_<dataset>_<task>.pt (trained with this task held out).
  RT_CKPT_NAME="pretrain_${DATASET_NAME}_${TASK_NAME}.pt"
  RT_CKPT_PATH="${RT_CKPT_DIR}/${RT_CKPT_NAME}"
  mkdir -p "${RT_CKPT_DIR}"
  if [[ ! -f "${RT_CKPT_PATH}" ]]; then
    echo "Downloading pretrained RT checkpoint ${RT_CKPT_NAME} from HF Hub..."
    python - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="${RT_HF_REPO}",
    filename="${RT_CKPT_NAME}",
    local_dir="${RT_CKPT_DIR}",
)
print(f"Downloaded: {path}")
PY
  else
    echo "Pretrained checkpoint already cached: ${RT_CKPT_PATH}"
  fi
  rt_ckpt_flags+=("--rt-load-ckpt-path" "${RT_CKPT_PATH}")
fi

if [[ "${RT_ZERO_SHOT}" == "1" ]]; then
  if [[ ${#rt_ckpt_flags[@]} -eq 0 ]]; then
    echo "ERROR: RT_ZERO_SHOT=1 requires RT_USE_PRETRAINED=1 or RT_CKPT_PATH." >&2
    exit 1
  fi
  echo "Zero-shot mode: skipping fine-tuning (max_steps=0)."
  rt_ckpt_flags+=("--rt-zero-shot")
fi

# ---------------------------------------------------------------------------
# View scope: RT_ALL_VIEWS=1 runs the full Protocol A (JT-Entity,
# JT-TemporalAgg, relational with RT, structural-count proxy graph track)
# using MODELS= as the joined-table/graph model.
# Default (RT_ALL_VIEWS=0) runs the relational view only.
# ---------------------------------------------------------------------------
RT_ALL_VIEWS="${RT_ALL_VIEWS:-0}"

view_scope_flags=()
if [[ "${RT_ALL_VIEWS}" == "1" ]]; then
  echo "Full Protocol A: JT views (${MODELS}) + relational (RT) + graph track."
else
  view_scope_flags+=("--relational-only")
fi

# Execute benchmark
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
  "${view_scope_flags[@]}" \
  --no-run-protocol-b \
  --output-dir "${JOB_OUTPUT_DIR}" \
  "${rt_ckpt_flags[@]}"

# Copy output directory from scratch back to the repo
cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "Results copied to ${RESULTS_DIR}"
