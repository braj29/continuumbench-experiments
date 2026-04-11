#!/usr/bin/env bash
# Set job requirements
#SBATCH -J cb-tabpfn
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH -p gpu_a100
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Job configuration
DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
MODELS="${MODELS:-tabpfn}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/continuumbench_tabpfn"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

# Loading modules
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

export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-${CACHE_ROOT}/relbench}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${CACHE_ROOT}/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

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

tabpfn_limits_flag=""
case "$(lowercase "${TABPFN_IGNORE_PRETRAINING_LIMITS}")" in
  1|true|yes)
    tabpfn_limits_flag="--tabpfn-ignore-pretraining-limits"
    ;;
  0|false|no)
    ;;
  *)
    echo "TABPFN_IGNORE_PRETRAINING_LIMITS must be one of: true, false, 1, 0, yes, no" >&2
    exit 1
    ;;
esac

# Execute the benchmark from the repo and write outputs to scratch
python "${REPO_ROOT}/continuumbench_tabfm_run.py" \
  --task-source dataset \
  --dataset-name "${DATASET_NAME}" \
  --task-name "${TASK_NAME}" \
  --models "${MODELS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${download_artifacts_flag}" \
  ${tabpfn_limits_flag} \
  --output-dir "${JOB_OUTPUT_DIR}"

# Copy output directory from scratch back to the repo
cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "Results copied to ${RESULTS_DIR}"
