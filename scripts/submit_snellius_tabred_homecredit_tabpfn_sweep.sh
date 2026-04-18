#!/usr/bin/env bash
# Submit a TabReD HomeCredit + TabPFN sweep across multiple splits and seeds.
#
# Each (split, seed) pair becomes one SLURM array task.  Results land in:
#   outputs/snellius/<array_job_id>/<split>_seed<seed>/result.json
#
# Examples:
#   # Single default split, 5 seeds:
#   TABRED_DATA_DIR=/scratch-shared/$USER/tabred/data/homecredit-default \
#   SEED_LIST="7 42 123 0 1" \
#   ./scripts/submit_snellius_tabred_homecredit_tabpfn_sweep.sh
#
#   # Two splits × 5 seeds (10 array tasks):
#   TABRED_DATA_DIR=/scratch-shared/$USER/tabred/data/homecredit-default \
#   SPLIT_LIST="default temporal" \
#   SEED_LIST="7 42 123 0 1" \
#   ./scripts/submit_snellius_tabred_homecredit_tabpfn_sweep.sh
#
#   # Limit to 4 concurrent tasks and 10 000 train rows per task:
#   MAX_CONCURRENT=4 MAX_TRAIN_ROWS=10000 \
#   TABRED_DATA_DIR=... ./scripts/submit_snellius_tabred_homecredit_tabpfn_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── parameters ────────────────────────────────────────────────────────────────
TABRED_DATA_DIR="${TABRED_DATA_DIR:-}"
SPLIT_LIST="${SPLIT_LIST:-default}"
SEED_LIST="${SEED_LIST:-7 42 123 0 1}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_META="${INCLUDE_META:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
WATCH_LOG="${WATCH_LOG:-0}"   # off by default for array jobs

# ── validation ────────────────────────────────────────────────────────────────
[[ -z "${TABRED_DATA_DIR}" ]] && {
  echo "ERROR: TABRED_DATA_DIR is required." >&2
  echo "  e.g. TABRED_DATA_DIR=/scratch-shared/\$USER/tabred/data/homecredit-default" >&2
  exit 1
}

command -v module >/dev/null 2>&1 || {
  echo "ERROR: 'module' not found — run this script on a Snellius login node." >&2
  exit 1
}

[[ ! -f "${VENV_DIR}/bin/activate" ]] && {
  echo "ERROR: venv missing at ${VENV_DIR}." >&2
  echo "  Run ./scripts/setup_snellius_env.sh first." >&2
  exit 1
}

module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

# ── compute array size ────────────────────────────────────────────────────────
read -r -a _splits <<< "${SPLIT_LIST}"
read -r -a _seeds  <<< "${SEED_LIST}"
n_splits="${#_splits[@]}"
n_seeds="${#_seeds[@]}"
n_total=$(( n_splits * n_seeds ))
array_last=$(( n_total - 1 ))

echo "=== TabReD HomeCredit + TabPFN sweep ==="
echo "  splits : ${SPLIT_LIST}"
echo "  seeds  : ${SEED_LIST}"
echo "  total  : ${n_total} tasks (${n_splits} × ${n_seeds})"
echo "  max concurrent : ${MAX_CONCURRENT}"
[[ -n "${MAX_TRAIN_ROWS}" ]] && echo "  max_train_rows : ${MAX_TRAIN_ROWS}"

# ── export everything the sbatch script needs ─────────────────────────────────
export TABRED_DATA_DIR SPLIT_LIST SEED_LIST DEVICE INCLUDE_META
export TABPFN_IGNORE_PRETRAINING_LIMITS MAX_TRAIN_ROWS
export SNELLIUS_STACK_MODULE SNELLIUS_PYTHON_MODULE VENV_DIR CACHE_ROOT
export REPO_ROOT

JOBID="$(sbatch --parsable \
  --array="0-${array_last}%${MAX_CONCURRENT}" \
  --export=ALL \
  scripts/snellius_tabred_homecredit_tabpfn_array.sbatch)"
JOBID="${JOBID%%;*}"

echo ""
echo "Submitted array job ${JOBID} (tasks 0–${array_last}, max ${MAX_CONCURRENT} concurrent)"
echo "Results dir : ${REPO_ROOT}/outputs/snellius/${JOBID}/"
echo "Log pattern : ${REPO_ROOT}/slurm-cb-tabred-hc-arr-${JOBID}_*.out"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOBID}"
echo "  sacct  -j ${JOBID} --format=JobID,State,Elapsed,MaxRSS"
echo ""
echo "Collect results once complete:"
echo "  python scripts/collect_sweep_results.py --results-dir outputs/snellius/${JOBID}"

[[ "${WATCH_LOG}" != "1" ]] && exit 0

# Optional: tail the first task's log when it appears.
LOG_PATTERN="${REPO_ROOT}/slurm-cb-tabred-hc-arr-${JOBID}_0.out"
echo "Watching first task log: ${LOG_PATTERN}"
for _ in $(seq 1 120); do
  [[ -f "${LOG_PATTERN}" ]] && { tail -F "${LOG_PATTERN}"; exit 0; }
  state="$(sacct -n -X -j "${JOBID}_0" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|*FAIL) break ;; esac
  sleep 5
done
[[ -f "${LOG_PATTERN}" ]] && tail -30 "${LOG_PATTERN}"
