#!/usr/bin/env bash
# TabReD HomeCredit task with TabPFN on Snellius (A100).
# Expects a preprocessed TabReD dataset directory containing:
#   X_num.npy X_bin.npy X_cat.npy Y.npy split-<name>/{train,val,test}_idx.npy

#SBATCH -J cb-tabred-hc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH -p gpu_a100
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

TABRED_DATA_DIR="${TABRED_DATA_DIR:-}"
SPLIT="${SPLIT:-default}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_META="${INCLUDE_META:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/tabred_homecredit_tabpfn"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

as_bool() {
  case "$(lowercase "$1")" in
    1|true|yes) return 0 ;;
    0|false|no|'') return 1 ;;
    *)
      echo "Invalid boolean value: '$1'" >&2
      exit 1
      ;;
  esac
}

validate_tabred_data_dir() {
  [[ -z "${TABRED_DATA_DIR}" ]] && {
    echo "TABRED_DATA_DIR is required." >&2
    exit 1
  }
  [[ ! -d "${TABRED_DATA_DIR}" ]] && {
    echo "TABRED_DATA_DIR does not exist: ${TABRED_DATA_DIR}" >&2
    exit 1
  }

  local required
  for required in X_num.npy X_bin.npy X_cat.npy Y.npy; do
    if [[ ! -f "${TABRED_DATA_DIR}/${required}" ]]; then
      echo "Missing required TabReD file: ${TABRED_DATA_DIR}/${required}" >&2
      exit 1
    fi
  done

  local split_dir="${TABRED_DATA_DIR}/split-${SPLIT}"
  if [[ ! -d "${split_dir}" ]]; then
    echo "Missing split directory: ${split_dir}" >&2
    echo "Available split dirs:" >&2
    ls -1 "${TABRED_DATA_DIR}" | grep '^split-' || true
    exit 1
  fi

  local idx
  for idx in train_idx.npy val_idx.npy test_idx.npy; do
    if [[ ! -f "${split_dir}/${idx}" ]]; then
      echo "Missing split index file: ${split_dir}/${idx}" >&2
      exit 1
    fi
  done
}

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
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

validate_tabred_data_dir
mkdir -p "${JOB_OUTPUT_DIR}" "${RESULTS_DIR}"

OUTPUT_JSON="${JOB_OUTPUT_DIR}/result.json"
cmd=(
  python -m continuumbench_experiments.tabred_homecredit_tabpfn
  --tabred-data-dir "${TABRED_DATA_DIR}"
  --split "${SPLIT}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-json "${OUTPUT_JSON}"
)

if as_bool "${INCLUDE_META}"; then
  cmd+=(--include-meta)
else
  cmd+=(--no-include-meta)
fi

if as_bool "${TABPFN_IGNORE_PRETRAINING_LIMITS}"; then
  cmd+=(--tabpfn-ignore-pretraining-limits)
else
  cmd+=(--no-tabpfn-ignore-pretraining-limits)
fi

if [[ -n "${MAX_TRAIN_ROWS}" ]]; then
  cmd+=(--max-train-rows "${MAX_TRAIN_ROWS}")
fi

echo "=== TabReD HomeCredit + TabPFN ==="
echo "job_id=${SLURM_JOB_ID:-unknown} split=${SPLIT} seed=${SEED} device=${DEVICE}"
echo "tabred_data_dir=${TABRED_DATA_DIR}"

echo "+ ${cmd[*]}"
"${cmd[@]}"

rsync -a "${JOB_OUTPUT_DIR}/" "${RESULTS_DIR}/" 2>/dev/null || cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "Results copied to ${RESULTS_DIR}"
