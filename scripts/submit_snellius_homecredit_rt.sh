#!/usr/bin/env bash
# Submit the HomeCredit relational (RT) job to Snellius.
#
# Example — fine-tune from scratch:
#   HC_DATA_DIR=/scratch-shared/$USER/homecredit \
#   ./scripts/submit_snellius_homecredit_rt.sh
#
# Example — zero-shot with pretrained checkpoint:
#   HC_DATA_DIR=/scratch-shared/$USER/homecredit \
#   RT_USE_PRETRAINED=1 RT_ZERO_SHOT=1 \
#   ./scripts/submit_snellius_homecredit_rt.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HC_DATA_DIR="${HC_DATA_DIR:-}"
MODELS="${MODELS:-tabpfn}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
RT_REPO_PATH="${RT_REPO_PATH:-${REPO_ROOT}/relational-transformer}"
RT_USE_PRETRAINED="${RT_USE_PRETRAINED:-0}"
RT_ZERO_SHOT="${RT_ZERO_SHOT:-0}"
RT_CKPT_PATH="${RT_CKPT_PATH:-}"
RT_MAX_STEPS="${RT_MAX_STEPS:-4097}"
RT_BATCH_SIZE="${RT_BATCH_SIZE:-32}"
RT_NUM_WORKERS="${RT_NUM_WORKERS:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L12-v2}"
SNELLIUS_STACK_MODULE="${SNELLIUS_STACK_MODULE:-2024}"
SNELLIUS_PYTHON_MODULE="${SNELLIUS_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
SNELLIUS_RUST_MODULE="${SNELLIUS_RUST_MODULE:-}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
RT_PREPROCESSED_ROOT="${RT_PREPROCESSED_ROOT:-${HOME}/scratch/pre}"
RT_CKPT_DIR="${RT_CKPT_DIR:-${HOME}/scratch/rt_ckpts}"
RT_HF_REPO="${RT_HF_REPO:-rishabh-ranjan/relational-transformer}"
WATCH_LOG="${WATCH_LOG:-1}"

[[ -z "${HC_DATA_DIR}" ]] && { echo "ERROR: HC_DATA_DIR is required." >&2; exit 1; }
command -v module >/dev/null 2>&1 || { echo "Run on a Snellius login node." >&2; exit 1; }

module purge; module load "${SNELLIUS_STACK_MODULE}"; module load "${SNELLIUS_PYTHON_MODULE}"
[[ ! -f "${VENV_DIR}/bin/activate" ]] && { echo "Run setup_snellius_env.sh first." >&2; exit 1; }

export HC_DATA_DIR MODELS SEED DEVICE
export RT_REPO_PATH RT_USE_PRETRAINED RT_ZERO_SHOT RT_CKPT_PATH
export RT_MAX_STEPS RT_BATCH_SIZE RT_NUM_WORKERS EMBEDDING_MODEL
export SNELLIUS_STACK_MODULE SNELLIUS_PYTHON_MODULE SNELLIUS_RUST_MODULE
export VENV_DIR CACHE_ROOT REPO_ROOT
export RT_PREPROCESSED_ROOT RT_CKPT_DIR RT_HF_REPO

JOBID="$(sbatch --parsable --export=ALL scripts/snellius_homecredit_rt_a100.sbatch)"
JOBID="${JOBID%%;*}"
LOG="${REPO_ROOT}/slurm-cb-hc-rt-${JOBID}.out"
echo "Submitted job ${JOBID}  (track=relational, zero_shot=${RT_ZERO_SHOT}, pretrained=${RT_USE_PRETRAINED})"
echo "Log: ${LOG}"

[[ "${WATCH_LOG}" != "1" ]] && exit 0
for _ in $(seq 1 180); do
  [[ -f "${LOG}" ]] && { tail -F "${LOG}"; exit 0; }
  state="$(sacct -n -X -j "${JOBID}" --format=State 2>/dev/null | head -n1 | xargs || true)"
  case "${state}" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|*FAIL) break ;; esac
  sleep 2
done
[[ -f "${LOG}" ]] && tail -F "${LOG}"
