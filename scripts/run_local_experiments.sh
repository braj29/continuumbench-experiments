#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
UV_BIN="${UV_BIN:-uv}"

DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_runs}"
RT_REPO_PATH="${RT_REPO_PATH:-$REPO_ROOT/relational-transformer}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <target>

Targets:
  setup                 Install project deps with dev extras.
  continuumbench-smoke  Synthetic ContinuumBench smoke run (TabICL + TabPFN).
  continuumbench        Dataset-backed ContinuumBench run (TabICL + TabPFN).
  continuumbench-rt     ContinuumBench run with official relational-transformer.
  test                  Run the focused unit test suite.

Environment overrides:
  DATASET_NAME       default: ${DATASET_NAME}
  TASK_NAME          default: ${TASK_NAME}
  DEVICE             default: ${DEVICE}
  SEED               default: ${SEED}
  OUTPUT_ROOT        default: ${OUTPUT_ROOT}
  RT_REPO_PATH       default: ${RT_REPO_PATH}
EOF
}

run() {
  echo "+ $*"
  "$@"
}

require_rt_repo() {
  if [[ ! -f "${RT_REPO_PATH}/rt/main.py" ]]; then
    echo "Missing official relational-transformer checkout at: ${RT_REPO_PATH}" >&2
    echo "Set RT_REPO_PATH=/path/to/relational-transformer and retry." >&2
    exit 1
  fi
}

target="${1:-}"

case "${target}" in
  setup)
    run "${UV_BIN}" sync --extra dev --extra task-source
    ;;

  test)
    run "${UV_BIN}" run "${PYTHON_BIN}" -m unittest -q \
      tests.test_continuumbench_harness
    ;;

  continuumbench-smoke)
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --task-source synthetic \
      --models tabicl,tabpfn \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench_smoke"
    ;;

  continuumbench)
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --task-source dataset \
      --dataset-name "${DATASET_NAME}" \
      --task-name "${TASK_NAME}" \
      --models tabicl,tabpfn \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench"
    ;;

  continuumbench-rt)
    require_rt_repo
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --task-source dataset \
      --dataset-name "${DATASET_NAME}" \
      --task-name "${TASK_NAME}" \
      --models tabicl,tabpfn \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --use-official-rt-relational \
      --rt-repo-path "${RT_REPO_PATH}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench_rt"
    ;;

  *)
    usage
    if [[ -n "${target}" ]]; then
      exit 1
    fi
    ;;
esac
