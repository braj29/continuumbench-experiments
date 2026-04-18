#!/usr/bin/env bash
# Submit TabReD HomeCredit + TabPFN benchmark job on Snellius.
#
# Example:
#   TABRED_DATA_DIR=/scratch-shared/$USER/tabred/data/homecredit-default \
#   ./scripts/submit_snellius_tabred_homecredit_tabpfn.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TABRED_DATA_DIR="${TABRED_DATA_DIR:-}"
SPLIT="${SPLIT:-default}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_META="${INCLUDE_META:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
WATCH_LOG="${WATCH_LOG:-1}"

[[ -z "${TABRED_DATA_DIR}" ]] && {
  echo "ERROR: TABRED_DATA_DIR is required." >&2
  exit 1
}

command -v module >/dev/null 2>&1 || {
  echo "Run on a Snellius login node." >&2
  exit 1
}

module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

[[ ! -f "${VENV_DIR}/bin/activate" ]] && {
  echo "Run setup_snellius_env.sh first." >&2
  exit 1
}

export TABRED_DATA_DIR SPLIT SEED DEVICE INCLUDE_META
export TABPFN_IGNORE_PRETRAINING_LIMITS MAX_TRAIN_ROWS
export SNELLIUS_STACK_MODULE SNELLIUS_PYTHON_MODULE VENV_DIR CACHE_ROOT REPO_ROOT

JOBID="$(sbatch --parsable --export=ALL scripts/snellius_tabred_homecredit_tabpfn_a100.sbatch)"
JOBID="${JOBID%%;*}"
LOG="${REPO_ROOT}/slurm-cb-tabred-hc-${JOBID}.out"

echo "Submitted job ${JOBID} (TabReD HomeCredit + TabPFN)"
echo "Log: ${LOG}"

[[ "${WATCH_LOG}" != "1" ]] && exit 0
for _ in $(seq 1 180); do
  [[ -f "${LOG}" ]] && { tail -F "${LOG}"; exit 0; }
  state="$(sacct -n -X -j "${JOBID}" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|*FAIL) break ;; esac
  sleep 2
done
[[ -f "${LOG}" ]] && tail -F "${LOG}"
