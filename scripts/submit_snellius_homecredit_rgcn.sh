#!/usr/bin/env bash
# Submit the HomeCredit graph (RGCN) job to Snellius.
#
# Example:
#   HC_DATA_DIR=/scratch-shared/$USER/homecredit \
#   GRAPH_MODEL=rgcn \
#   ./scripts/submit_snellius_homecredit_rgcn.sh

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
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
WATCH_LOG="${WATCH_LOG:-1}"

[[ -z "${HC_DATA_DIR}" ]] && { echo "ERROR: HC_DATA_DIR is required." >&2; exit 1; }
command -v module >/dev/null 2>&1 || { echo "Run on a Snellius login node." >&2; exit 1; }

module purge; module load "${SNELLIUS_STACK_MODULE}"; module load "${SNELLIUS_PYTHON_MODULE}"
[[ ! -f "${VENV_DIR}/bin/activate" ]] && { echo "Run setup_snellius_env.sh first." >&2; exit 1; }

export HC_DATA_DIR GRAPH_MODEL MODELS SEED DEVICE
export GNN_HIDDEN_DIM GNN_NUM_LAYERS GNN_MAX_EPOCHS GNN_PATIENCE GNN_LR GNN_BATCH_SIZE
export SNELLIUS_STACK_MODULE SNELLIUS_PYTHON_MODULE VENV_DIR CACHE_ROOT REPO_ROOT

JOBID="$(sbatch --parsable --export=ALL scripts/snellius_homecredit_rgcn_a100.sbatch)"
JOBID="${JOBID%%;*}"
LOG="${REPO_ROOT}/slurm-cb-hc-rgcn-${JOBID}.out"
echo "Submitted job ${JOBID}  (track=graph, model=${GRAPH_MODEL})"
echo "Log: ${LOG}"

[[ "${WATCH_LOG}" != "1" ]] && exit 0
for _ in $(seq 1 180); do
  [[ -f "${LOG}" ]] && { tail -F "${LOG}"; exit 0; }
  state="$(sacct -n -X -j "${JOBID}" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|*FAIL) break ;; esac
  sleep 2
done
[[ -f "${LOG}" ]] && tail -F "${LOG}"
