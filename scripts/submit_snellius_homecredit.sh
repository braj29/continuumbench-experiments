#!/usr/bin/env bash
# Submit the HomeCredit Protocol A benchmark to Snellius.
#
# Prerequisites:
#   1. Run setup_snellius_env.sh once to create .venv-snellius.
#   2. Upload the 7 Kaggle CSVs to scratch, e.g.:
#        rsync -av /local/homecredit/ snellius:/scratch-shared/$USER/homecredit/
#   3. Set HC_DATA_DIR to that path, then run this script.
#
# Example:
#   HC_DATA_DIR=/scratch-shared/$USER/homecredit \
#   GRAPH_MODEL=rgcn \
#   ./scripts/submit_snellius_homecredit.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HC_DATA_DIR="${HC_DATA_DIR:-}"
GRAPH_MODEL="${GRAPH_MODEL:-rgcn}"
MODELS="${MODELS:-tabpfn}"
GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM:-128}"
GNN_NUM_LAYERS="${GNN_NUM_LAYERS:-2}"
GNN_MAX_EPOCHS="${GNN_MAX_EPOCHS:-200}"
GNN_PATIENCE="${GNN_PATIENCE:-20}"
GNN_LR="${GNN_LR:-1e-3}"
GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-64}"
RUN_PROTOCOL_B="${RUN_PROTOCOL_B:-0}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"

SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
WATCH_LOG="${WATCH_LOG:-1}"

if [[ -z "${HC_DATA_DIR}" ]]; then
  echo "ERROR: HC_DATA_DIR is not set." >&2
  echo "Set it to the directory containing the 7 HomeCredit Kaggle CSVs:" >&2
  echo "  HC_DATA_DIR=/scratch-shared/\$USER/homecredit \\" >&2
  echo "  ./scripts/submit_snellius_homecredit.sh" >&2
  exit 1
fi

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

export HC_DATA_DIR GRAPH_MODEL MODELS SEED DEVICE
export GNN_HIDDEN_DIM GNN_NUM_LAYERS GNN_MAX_EPOCHS GNN_PATIENCE GNN_LR GNN_BATCH_SIZE
export RUN_PROTOCOL_B
export SNELLIUS_STACK_MODULE SNELLIUS_PYTHON_MODULE
export VENV_DIR CACHE_ROOT REPO_ROOT

JOBID="$(sbatch --parsable --export=ALL scripts/snellius_homecredit_a100.sbatch)"
JOBID="${JOBID%%;*}"

LOG="${REPO_ROOT}/slurm-cb-homecredit-${JOBID}.out"
LOG_FALLBACK="${REPO_ROOT}/slurm-${JOBID}.out"

echo "Submitted job ${JOBID}"
echo "  hc_data_dir : ${HC_DATA_DIR}"
echo "  models      : ${MODELS}"
echo "  graph_model : ${GRAPH_MODEL}"
echo "  protocol_b  : ${RUN_PROTOCOL_B}"
echo "  Log file    : ${LOG}"

if [[ "${WATCH_LOG}" != "1" ]]; then
  exit 0
fi

for _ in $(seq 1 180); do
  if [[ -f "${LOG}" ]]; then
    tail -F "${LOG}"
    exit 0
  fi
  if [[ -f "${LOG_FALLBACK}" ]]; then
    tail -F "${LOG_FALLBACK}"
    exit 0
  fi

  state="$(sacct -n -X -j "${JOBID}" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL)
      break ;;
  esac
  sleep 2
done

for logpath in "${LOG}" "${LOG_FALLBACK}"; do
  if [[ -f "${logpath}" ]]; then
    tail -F "${logpath}"
    exit 0
  fi
done

echo "No log file found for job ${JOBID}." >&2
sacct -j "${JOBID}" --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,NodeList,Reason
