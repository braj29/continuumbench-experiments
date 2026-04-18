#!/usr/bin/env bash
# TabReD HomeCredit + TabPFN — SLURM array job (multi-seed × multi-split).
#
# Each array task maps to one (SPLIT, SEED) pair from the Cartesian product of
# SPLIT_LIST and SEED_LIST.  Results land in:
#   outputs/snellius/<job_id>/<split>_seed<seed>/result.json
#
# Usage (from submit_snellius_tabred_homecredit_tabpfn_sweep.sh):
#   sbatch --array=0-N%<max_concurrent> \
#          --export=ALL \
#          scripts/snellius_tabred_homecredit_tabpfn_array.sbatch
#
# Key env vars:
#   TABRED_DATA_DIR   (required) path to homecredit-default directory
#   SPLIT_LIST        space-separated splits, e.g. "default temporal"
#   SEED_LIST         space-separated seeds,  e.g. "7 42 123 0 1"
#   MAX_TRAIN_ROWS    optional row cap for TabPFN (empty = no cap)
#   INCLUDE_META      0/1 (default 0)
#   TABPFN_IGNORE_PRETRAINING_LIMITS  0/1 (default 1)

#SBATCH -J cb-tabred-hc-arr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH -p gpu_a100
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail

# ── env defaults ──────────────────────────────────────────────────────────────
TABRED_DATA_DIR="${TABRED_DATA_DIR:-}"
SPLIT_LIST="${SPLIT_LIST:-default}"
SEED_LIST="${SEED_LIST:-7}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_META="${INCLUDE_META:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"

# ── helpers ───────────────────────────────────────────────────────────────────
lowercase() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

as_bool() {
  case "$(lowercase "$1")" in
    1|true|yes)    return 0 ;;
    0|false|no|'') return 1 ;;
    *) echo "Invalid boolean: '$1'" >&2; exit 1 ;;
  esac
}

# ── map array index → (split, seed) ──────────────────────────────────────────
read -r -a _splits <<< "${SPLIT_LIST}"
read -r -a _seeds  <<< "${SEED_LIST}"
n_splits="${#_splits[@]}"
n_seeds="${#_seeds[@]}"
n_total=$(( n_splits * n_seeds ))

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id >= n_total )); then
  echo "Array task ${task_id} >= total combos ${n_total}; nothing to do." >&2
  exit 0
fi

SPLIT="${_splits[$(( task_id / n_seeds ))]}"
SEED="${_seeds[$(( task_id % n_seeds ))]}"

# ── validate ──────────────────────────────────────────────────────────────────
[[ -z "${TABRED_DATA_DIR}" ]] && { echo "TABRED_DATA_DIR is required." >&2; exit 1; }
[[ ! -d "${TABRED_DATA_DIR}" ]] && { echo "TABRED_DATA_DIR not found: ${TABRED_DATA_DIR}" >&2; exit 1; }

for f in X_num.npy X_bin.npy X_cat.npy Y.npy; do
  [[ ! -f "${TABRED_DATA_DIR}/${f}" ]] && {
    echo "Missing required file: ${TABRED_DATA_DIR}/${f}" >&2; exit 1; }
done

split_dir="${TABRED_DATA_DIR}/split-${SPLIT}"
[[ ! -d "${split_dir}" ]] && {
  echo "Missing split directory: ${split_dir}" >&2
  echo "Available splits:" >&2
  ls -1 "${TABRED_DATA_DIR}" | grep '^split-' || true
  exit 1
}
for idx in train_idx.npy val_idx.npy test_idx.npy; do
  [[ ! -f "${split_dir}/${idx}" ]] && {
    echo "Missing split file: ${split_dir}/${idx}" >&2; exit 1; }
done

# ── environment ───────────────────────────────────────────────────────────────
module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"
cd "${REPO_ROOT}"

[[ ! -f "${VENV_DIR}/bin/activate" ]] && {
  echo "Missing venv at ${VENV_DIR}. Run ./scripts/setup_snellius_env.sh first." >&2
  exit 1
}
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${CACHE_ROOT}/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ── output paths ──────────────────────────────────────────────────────────────
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
TASK_TAG="${SPLIT}_seed${SEED}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/${TASK_TAG}"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_ARRAY_JOB_ID}/${TASK_TAG}"

mkdir -p "${JOB_OUTPUT_DIR}" "${RESULTS_DIR}"
OUTPUT_JSON="${JOB_OUTPUT_DIR}/result.json"

# ── build command ─────────────────────────────────────────────────────────────
cmd=(
  python -m continuumbench_experiments.tabred_homecredit_tabpfn
  --tabred-data-dir "${TABRED_DATA_DIR}"
  --split           "${SPLIT}"
  --seed            "${SEED}"
  --device          "${DEVICE}"
  --output-json     "${OUTPUT_JSON}"
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

[[ -n "${MAX_TRAIN_ROWS}" ]] && cmd+=(--max-train-rows "${MAX_TRAIN_ROWS}")

# ── run ───────────────────────────────────────────────────────────────────────
echo "=== TabReD HomeCredit + TabPFN (array) ==="
echo "job=${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]  split=${SPLIT}  seed=${SEED}  device=${DEVICE}"
echo "tabred_data_dir=${TABRED_DATA_DIR}"
echo "n_combos=${n_total} (${n_splits} splits × ${n_seeds} seeds)"
echo "+ ${cmd[*]}"

"${cmd[@]}"

rsync -a "${JOB_OUTPUT_DIR}/" "${RESULTS_DIR}/" 2>/dev/null \
  || cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "Results → ${RESULTS_DIR}"
