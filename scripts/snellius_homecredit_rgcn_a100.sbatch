#!/usr/bin/env bash
# HomeCredit — graph track only (RGCN or GraphSAGE).
# Uses --track graph so only the graph view is evaluated.
# Run in parallel with the TabPFN and RT jobs.
#
# Key env vars:
#   HC_DATA_DIR        Path to directory with 7 Kaggle CSVs [required]
#   GRAPH_MODEL        rgcn | graphsage | count-proxy          [default: rgcn]
#   MODELS             Model key used as output folder name    [default: tabpfn]
#   GNN_HIDDEN_DIM     GNN hidden dim                          [default: 128]
#   GNN_NUM_LAYERS     GNN layer count                        [default: 2]
#   GNN_MAX_EPOCHS     Max GNN training epochs                 [default: 200]
#   GNN_PATIENCE       Early-stopping patience                 [default: 20]
#   SEED               random seed                             [default: 7]

#SBATCH -J cb-hc-rgcn
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH -p gpu_a100
#SBATCH -t 08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

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

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/hc_rgcn"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

[[ -z "${HC_DATA_DIR}" ]] && { echo "ERROR: HC_DATA_DIR is required." >&2; exit 1; }
[[ ! -d "${HC_DATA_DIR}" ]] && { echo "ERROR: HC_DATA_DIR not found: ${HC_DATA_DIR}" >&2; exit 1; }

module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"
cd "${REPO_ROOT}"

[[ ! -f "${VENV_DIR}/bin/activate" ]] && { echo "Run setup_snellius_env.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

mkdir -p "${JOB_OUTPUT_DIR}" "${RESULTS_DIR}"

python - <<'PY'
import os, torch
print(f"Python: {os.sys.executable}")
print(f"Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
PY

echo "=== HomeCredit — graph track (${GRAPH_MODEL}) ==="
echo "  gnn_hidden=${GNN_HIDDEN_DIM}  layers=${GNN_NUM_LAYERS}  epochs=${GNN_MAX_EPOCHS}  seed=${SEED}"

srun --ntasks=1 python "${REPO_ROOT}/continuumbench_tabfm_run.py" \
  --task-source homecredit \
  --homecredit-data-dir "${HC_DATA_DIR}" \
  --track graph \
  --models "${MODELS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --graph-model "${GRAPH_MODEL}" \
  --gnn-hidden-dim "${GNN_HIDDEN_DIM}" \
  --gnn-num-layers "${GNN_NUM_LAYERS}" \
  --gnn-max-epochs "${GNN_MAX_EPOCHS}" \
  --gnn-patience "${GNN_PATIENCE}" \
  --gnn-lr "${GNN_LR}" \
  --gnn-batch-size "${GNN_BATCH_SIZE}" \
  --no-run-protocol-b \
  --output-dir "${JOB_OUTPUT_DIR}"

rsync -a "${JOB_OUTPUT_DIR}/" "${RESULTS_DIR}/" 2>/dev/null || cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "=== Done. Results at ${RESULTS_DIR} ==="
