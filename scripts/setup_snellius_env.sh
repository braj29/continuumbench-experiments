#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
WARMUP_DATASET="${WARMUP_DATASET:-1}"
WARMUP_TABPFN="${WARMUP_TABPFN:-1}"
RECREATE_VENV="${RECREATE_VENV:-0}"

export DATASET_NAME TASK_NAME WARMUP_DATASET WARMUP_TABPFN

if command -v module >/dev/null 2>&1; then
  if [[ -n "${SNELLIUS_STACK_MODULE}" ]]; then
    module load "${SNELLIUS_STACK_MODULE}"
  fi
  if [[ -n "${SNELLIUS_PYTHON_MODULE}" ]]; then
    module load "${SNELLIUS_PYTHON_MODULE}"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN} on PATH." >&2
  echo "Load a Snellius Python module first, e.g. set SNELLIUS_PYTHON_MODULE after 'module avail Python'." >&2
  exit 1
fi

export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-$CACHE_ROOT/relbench}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$CACHE_ROOT/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"

mkdir -p \
  "${CACHE_ROOT}" \
  "${RELBENCH_CACHE_DIR}" \
  "${TABPFN_MODEL_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}"

if [[ -e "${VENV_DIR}" && ! -d "${VENV_DIR}" ]]; then
  echo "VENV_DIR exists but is not a directory: ${VENV_DIR}" >&2
  exit 1
fi

if [[ "${RECREATE_VENV}" == "1" && -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  "${PYTHON_BIN}" -m venv --copies "${VENV_DIR}"
else
  echo "Reusing virtual environment at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[task-source]"

python - <<'PY'
import os

from continuumbench_experiments.models import build_tabular_estimator
from continuumbench_experiments.continuumbench.sources import load_dataset_entity_problem


def enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no"}


print(f"Python executable: {os.sys.executable}")
print(f"RelBench cache: {os.environ['RELBENCH_CACHE_DIR']}")
print(f"TabPFN cache: {os.environ['TABPFN_MODEL_CACHE_DIR']}")

if enabled("WARMUP_DATASET"):
    dataset_name = os.environ["DATASET_NAME"]
    task_name = os.environ["TASK_NAME"]
    print(f"Warming RelBench dataset/task cache for {dataset_name}/{task_name}...")
    load_dataset_entity_problem(dataset_name, task_name, download=True)

if enabled("WARMUP_TABPFN"):
    print("Warming TabPFN classifier cache...")
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 1, 0, 1]
    clf = build_tabular_estimator("tabpfn", "classification", device="cpu")
    clf.fit(X, y)
PY

# ---------------------------------------------------------------------------
# Rust toolchain for the RT rustler extension
# ---------------------------------------------------------------------------
# rkyv >= 0.8 (required by rustler/Cargo.lock) requires rustc >= 1.81.
# Snellius 2024 module stack caps at Rust 1.78, so we cache a rustup-managed
# stable toolchain inside the repo directory.  Compute nodes have no internet
# access, so the toolchain must be pre-installed here on the login node.
RUSTUP_HOME="${RUSTUP_HOME:-${REPO_ROOT}/.rustup}"
CARGO_HOME="${CARGO_HOME:-${REPO_ROOT}/.cargo}"

_rustup_min_minor=81   # rustc >= 1.81 required for rkyv 0.8

_cached_rustc_ok() {
  local ver minor
  ver="$("${CARGO_HOME}/bin/rustc" --version 2>/dev/null | awk '{print $2}' || echo '0.0.0')"
  minor="$(printf '%s' "${ver}" | cut -d. -f2)"
  [[ "${minor}" -ge "${_rustup_min_minor}" ]]
}

if _cached_rustc_ok; then
  echo "Rust toolchain already cached at ${CARGO_HOME}/bin: $("${CARGO_HOME}/bin/rustc" --version)"
else
  echo "Installing stable Rust via rustup (cached at ${CARGO_HOME})..."
  echo "This requires internet access — run on a Snellius login node."
  export RUSTUP_HOME CARGO_HOME
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable
  echo "Installed: $("${CARGO_HOME}/bin/rustc" --version)"
fi

echo "Snellius environment is ready in ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
