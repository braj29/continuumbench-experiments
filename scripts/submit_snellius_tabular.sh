#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODELS="${MODELS:-tabpfn}"
TASK_SOURCE="${TASK_SOURCE:-dataset}"
DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
HC_DATA_DIR="${HC_DATA_DIR:-}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"

SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
WATCH_LOG="${WATCH_LOG:-1}"

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

TASK_SOURCE="$(printf '%s' "${TASK_SOURCE}" | tr '[:upper:]' '[:lower:]')"
case "${TASK_SOURCE}" in
  dataset|homecredit|synthetic)
    ;;
  *)
    echo "TASK_SOURCE must be one of: dataset, homecredit, synthetic" >&2
    exit 1
    ;;
esac

if [[ "${TASK_SOURCE}" == "homecredit" ]]; then
  if [[ -z "${HC_DATA_DIR}" ]]; then
    echo "WARN: TASK_SOURCE=homecredit but HC_DATA_DIR is not set; falling back to TASK_SOURCE=dataset."
    TASK_SOURCE="dataset"
  elif [[ ! -d "${HC_DATA_DIR}" ]]; then
    echo "HC_DATA_DIR not found: ${HC_DATA_DIR}" >&2
    exit 1
  fi
fi

export MODELS
export TASK_SOURCE
export DATASET_NAME
export TASK_NAME
export HC_DATA_DIR
export SEED
export DEVICE
export DOWNLOAD_ARTIFACTS
export TABPFN_IGNORE_PRETRAINING_LIMITS
export SNELLIUS_STACK_MODULE
export SNELLIUS_PYTHON_MODULE
export VENV_DIR
export CACHE_ROOT
export REPO_ROOT

JOBID="$(sbatch --parsable --export=ALL scripts/snellius_tabpfn_a100.sbatch)"
JOBID="${JOBID%%;*}"

LOG_PRIMARY="${REPO_ROOT}/slurm-cb-tabfm-${JOBID}.out"
LOG_LEGACY="${REPO_ROOT}/slurm-cb-tabpfn-${JOBID}.out"
LOG_FALLBACK="${REPO_ROOT}/slurm-${JOBID}.out"

echo "Submitted job ${JOBID}"
echo "Task source: ${TASK_SOURCE}"
echo "Models: ${MODELS}"
if [[ "${TASK_SOURCE}" == "dataset" ]]; then
  echo "Dataset/task: ${DATASET_NAME}/${TASK_NAME}"
elif [[ "${TASK_SOURCE}" == "homecredit" ]]; then
  echo "HomeCredit CSV dir: ${HC_DATA_DIR}"
fi
echo "Primary log: ${LOG_PRIMARY}"

if [[ "${WATCH_LOG}" != "1" ]]; then
  exit 0
fi

for _ in $(seq 1 180); do
  if [[ -f "${LOG_PRIMARY}" ]]; then
    tail -F "${LOG_PRIMARY}"
    exit 0
  fi
  if [[ -f "${LOG_LEGACY}" ]]; then
    tail -F "${LOG_LEGACY}"
    exit 0
  fi
  if [[ -f "${LOG_FALLBACK}" ]]; then
    tail -F "${LOG_FALLBACK}"
    exit 0
  fi

  state="$(sacct -n -X -j "${JOBID}" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL)
      break
      ;;
  esac
  sleep 2
done

if [[ -f "${LOG_PRIMARY}" ]]; then
  tail -F "${LOG_PRIMARY}"
  exit 0
fi
if [[ -f "${LOG_LEGACY}" ]]; then
  tail -F "${LOG_LEGACY}"
  exit 0
fi
if [[ -f "${LOG_FALLBACK}" ]]; then
  tail -F "${LOG_FALLBACK}"
  exit 0
fi

echo "No log file found yet for job ${JOBID}." >&2
sacct -j "${JOBID}" --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,NodeList,Reason
echo "If state is terminal, inspect scheduler details:"
echo "  scontrol show job ${JOBID}"
