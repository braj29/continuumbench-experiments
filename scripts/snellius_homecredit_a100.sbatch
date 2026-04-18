#!/usr/bin/env bash
# ContinuumBench — Home Credit Default Risk
# Runs Protocol A (+ optional Protocol B) with:
#   joined-table track : TabPFN  (pre-trained foundation model, GPU accelerated)
#   graph track        : RGCN    (in-process GNN — no RelBench needed)
#   relational track   : rt-stub (RT not applicable: HomeCredit is not in RelBench)
#
# NOTE: RelGT (--graph-model relgt) does NOT work for HomeCredit because
# RelGT's official codebase fetches data from RelBench by dataset_name/task_name.
# Use --graph-model rgcn or graphsage for the graph track instead.
# To use RelGT, you would need to either:
#   (a) Register HomeCredit in RelBench and update the RelGT data-loading wrapper, or
#   (b) Implement a standalone RelGT adapter that accepts our DatabaseSpec directly.
#
# Usage (submit from repo root on a Snellius login node):
#   export HC_DATA_DIR=/scratch-shared/$USER/homecredit
#   sbatch --export=ALL scripts/snellius_homecredit_a100.sbatch
#
# Key env vars:
#   HC_DATA_DIR          Path to directory holding the 7 Kaggle CSVs [required]
#   GRAPH_MODEL          Graph track model: rgcn | graphsage | count-proxy [default: rgcn]
#   MODELS               Tabular model(s): tabpfn | xgboost | tabicl  [default: tabpfn]
#   GNN_HIDDEN_DIM       RGCN/GraphSAGE hidden dim                    [default: 128]
#   GNN_NUM_LAYERS       RGCN/GraphSAGE layer count                   [default: 2]
#   GNN_MAX_EPOCHS       Max GNN training epochs                      [default: 200]
#   GNN_PATIENCE         Early stopping patience                      [default: 20]
#   RUN_PROTOCOL_B       1 to run Protocol B ablations                [default: 0]
#   SEED                 Random seed                                  [default: 7]

#SBATCH -J cb-homecredit
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=240G
#SBATCH -p gpu_a100
#SBATCH -t 10:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment)
# ---------------------------------------------------------------------------
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

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/continuumbench_homecredit"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${HC_DATA_DIR}" ]]; then
  echo "ERROR: HC_DATA_DIR is required." >&2
  echo "Set it to the directory containing the 7 HomeCredit Kaggle CSVs, e.g.:" >&2
  echo "  export HC_DATA_DIR=/scratch-shared/\$USER/homecredit" >&2
  echo "  sbatch --export=ALL scripts/snellius_homecredit_a100.sbatch" >&2
  exit 1
fi

if [[ ! -d "${HC_DATA_DIR}" ]]; then
  echo "ERROR: HC_DATA_DIR does not exist: ${HC_DATA_DIR}" >&2
  exit 1
fi

for csv_file in application_train.csv bureau.csv bureau_balance.csv \
    previous_application.csv POS_CASH_balance.csv credit_card_balance.csv \
    installments_payments.csv; do
  if [[ ! -f "${HC_DATA_DIR}/${csv_file}" ]]; then
    echo "ERROR: Missing CSV: ${HC_DATA_DIR}/${csv_file}" >&2
    echo "Download from https://www.kaggle.com/c/home-credit-default-risk/data" >&2
    exit 1
  fi
done

if [[ "${GRAPH_MODEL}" == "relgt" ]]; then
  echo "ERROR: --graph-model relgt is not supported for HomeCredit." >&2
  echo "RelGT's official codebase loads data from RelBench by dataset/task name;" >&2
  echo "HomeCredit is not registered in RelBench." >&2
  echo "Use GRAPH_MODEL=rgcn or GRAPH_MODEL=graphsage instead." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

cd "${REPO_ROOT}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}" >&2
  echo "Run ./scripts/setup_snellius_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${CACHE_ROOT}/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

mkdir -p "${JOB_OUTPUT_DIR}" "${RESULTS_DIR}"

# GPU probe
python - <<'PY'
import os, torch
print(f"Python: {os.sys.executable}")
print(f"Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPUs: {torch.cuda.device_count()}  Device: {torch.cuda.get_device_name(0)}")
PY

echo "=== ContinuumBench — Home Credit Default Risk ==="
echo "  job_id        : ${SLURM_JOB_ID:-local}"
echo "  hc_data_dir   : ${HC_DATA_DIR}"
echo "  tabular_model : ${MODELS}"
echo "  graph_model   : ${GRAPH_MODEL}"
echo "  seed          : ${SEED}"
echo "  device        : ${DEVICE}"
echo "  protocol_b    : ${RUN_PROTOCOL_B}"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
lowercase() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

protocol_b_flag="--no-run-protocol-b"
case "$(lowercase "${RUN_PROTOCOL_B}")" in
  1|true|yes) protocol_b_flag="--run-protocol-b" ;;
esac

cmd=(
  python "${REPO_ROOT}/continuumbench_tabfm_run.py"
  --task-source homecredit
  --homecredit-data-dir "${HC_DATA_DIR}"
  --models "${MODELS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --tabpfn-ignore-pretraining-limits
  --graph-model "${GRAPH_MODEL}"
  --gnn-hidden-dim "${GNN_HIDDEN_DIM}"
  --gnn-num-layers "${GNN_NUM_LAYERS}"
  --gnn-max-epochs "${GNN_MAX_EPOCHS}"
  --gnn-patience "${GNN_PATIENCE}"
  --gnn-lr "${GNN_LR}"
  --gnn-batch-size "${GNN_BATCH_SIZE}"
  "${protocol_b_flag}"
  --output-dir "${JOB_OUTPUT_DIR}"
)

echo "+ ${cmd[*]}"
srun --ntasks=1 "${cmd[@]}"

# Copy scratch outputs back to repo
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${JOB_OUTPUT_DIR}/" "${RESULTS_DIR}/"
else
  cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
fi

echo "=== Done. Results at ${RESULTS_DIR} ==="
