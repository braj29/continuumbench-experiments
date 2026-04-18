#!/usr/bin/env bash
# HomeCredit — joined-table track only (JT-Entity + JT-TemporalAgg) with TabPFN.
# Uses --track joined so only the two JT views are evaluated; relational and
# graph tracks are skipped.  Run this job in parallel with the RGCN and RT jobs.
#
# Key env vars:
#   HC_DATA_DIR        Path to directory with 7 Kaggle CSVs [required]
#   MODELS             tabpfn | xgboost | tabicl                [default: tabpfn]
#   RUN_PROTOCOL_B     1 to also run Protocol B lookback sweep  [default: 0]
#   SEED               random seed                              [default: 7]

#SBATCH -J cb-hc-tabpfn
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH -p gpu_a100
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

HC_DATA_DIR="${HC_DATA_DIR:-}"
MODELS="${MODELS:-tabpfn}"
RUN_PROTOCOL_B="${RUN_PROTOCOL_B:-0}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/hc_tabpfn"
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

export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${CACHE_ROOT}/tabpfn}"
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

lowercase() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
protocol_b_flag="--no-run-protocol-b"
case "$(lowercase "${RUN_PROTOCOL_B}")" in 1|true|yes) protocol_b_flag="--run-protocol-b" ;; esac

echo "=== HomeCredit — joined-table track (TabPFN) ==="
echo "  model=${MODELS}  seed=${SEED}  protocol_b=${RUN_PROTOCOL_B}"

srun --ntasks=1 python "${REPO_ROOT}/continuumbench_tabfm_run.py" \
  --task-source homecredit \
  --homecredit-data-dir "${HC_DATA_DIR}" \
  --track joined \
  --models "${MODELS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --tabpfn-ignore-pretraining-limits \
  "${protocol_b_flag}" \
  --output-dir "${JOB_OUTPUT_DIR}"

rsync -a "${JOB_OUTPUT_DIR}/" "${RESULTS_DIR}/" 2>/dev/null || cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "=== Done. Results at ${RESULTS_DIR} ==="
