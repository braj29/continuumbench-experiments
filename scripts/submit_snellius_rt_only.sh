#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
SEED="${SEED:-7}"
RT_REPO_PATH="${RT_REPO_PATH:-${REPO_ROOT}/relational-transformer}"
RT_PREPROCESSED_ROOT="${RT_PREPROCESSED_ROOT:-${HOME}/scratch/pre}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
SNELLIUS_RUST_MODULE="${SNELLIUS_RUST_MODULE:-}"

WATCH_LOG="${WATCH_LOG:-1}"

if ! command -v module >/dev/null 2>&1; then
  echo "Environment module command is unavailable. Run this on a Snellius login node." >&2
  exit 1
fi

module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}" >&2
  echo "Run ./scripts/setup_snellius_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install --upgrade huggingface_hub
fi

mkdir -p "${RT_PREPROCESSED_ROOT}"

table_info_path="${RT_PREPROCESSED_ROOT}/${DATASET_NAME}/table_info.json"
column_index_path="${RT_PREPROCESSED_ROOT}/${DATASET_NAME}/column_index.json"

if [[ ! -f "${table_info_path}" || ! -f "${column_index_path}" ]]; then
  echo "Downloading missing RT preprocessed artifacts for ${DATASET_NAME}..."
  hf download --repo-type dataset --local-dir "${RT_PREPROCESSED_ROOT}" \
    hvag976/relational-transformer \
    "${DATASET_NAME}/table_info.json" \
    "${DATASET_NAME}/column_index.json"
fi

if [[ ! -f "${table_info_path}" || ! -f "${column_index_path}" ]]; then
  echo "RT preprocessed artifacts are still missing after download attempt." >&2
  echo "Expected files:" >&2
  echo "  ${table_info_path}" >&2
  echo "  ${column_index_path}" >&2
  exit 1
fi

if [[ -z "${SNELLIUS_RUST_MODULE}" ]]; then
  SNELLIUS_RUST_MODULE="$(
    module -t avail Rust 2>&1 | awk '/^Rust\/[^ ]+$/ {print; exit}'
  )"
fi

if [[ -z "${SNELLIUS_RUST_MODULE}" ]]; then
  echo "Could not auto-detect a Rust module." >&2
  echo "Set SNELLIUS_RUST_MODULE explicitly (check with: module -t avail Rust)." >&2
  exit 1
fi

JOBID="$(
  sbatch --parsable \
    --export=ALL,DATASET_NAME="${DATASET_NAME}",TASK_NAME="${TASK_NAME}",SEED="${SEED}",RT_REPO_PATH="${RT_REPO_PATH}",RT_PREPROCESSED_ROOT="${RT_PREPROCESSED_ROOT}",SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE}",SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE}",SNELLIUS_RUST_MODULE="${SNELLIUS_RUST_MODULE}" \
    scripts/snellius_rt_only_a100.sbatch
)"
JOBID="${JOBID%%;*}"

LOG_PRIMARY="${REPO_ROOT}/slurm-cb-rt-only-${JOBID}.out"
LOG_FALLBACK="${REPO_ROOT}/slurm-${JOBID}.out"

echo "Submitted job ${JOBID}"
echo "Rust module: ${SNELLIUS_RUST_MODULE}"
echo "Primary log: ${LOG_PRIMARY}"

if [[ "${WATCH_LOG}" != "1" ]]; then
  exit 0
fi

for _ in $(seq 1 180); do
  if [[ -f "${LOG_PRIMARY}" ]]; then
    tail -F "${LOG_PRIMARY}"
    exit 0
  fi
  if [[ -f "${LOG_FALLBACK}" ]]; then
    tail -F "${LOG_FALLBACK}"
    exit 0
  fi

  state="$(sacct -n -X -j "${JOBID}" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL)
      break
      ;;
  esac
  sleep 2
done

if [[ -f "${LOG_PRIMARY}" ]]; then
  tail -F "${LOG_PRIMARY}"
  exit 0
fi
if [[ -f "${LOG_FALLBACK}" ]]; then
  tail -F "${LOG_FALLBACK}"
  exit 0
fi

echo "No log file found yet for job ${JOBID}." >&2
sacct -j "${JOBID}" --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,NodeList,Reason
echo "If state is terminal, inspect scheduler details:"
echo "  scontrol show job ${JOBID}"
