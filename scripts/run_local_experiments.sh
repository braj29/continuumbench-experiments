#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
UV_BIN="${UV_BIN:-uv}"

RELBENCH_DATASET="${RELBENCH_DATASET:-rel-f1}"
RELBENCH_TASK="${RELBENCH_TASK:-driver-top3}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_runs}"
RT_REPO_PATH="${RT_REPO_PATH:-$REPO_ROOT/relational-transformer}"

GRAPHBENCH_DATASET="${GRAPHBENCH_DATASET:-socialnetwork}"
GRAPHBENCH_TASK="${GRAPHBENCH_TASK:-node}"
GRAPHBENCH_ROOT="${GRAPHBENCH_ROOT:-$REPO_ROOT/graphbench_data}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <target>

Targets:
  setup                 Install project deps with dev and GraphBench extras.
  continuumbench-smoke  Synthetic ContinuumBench smoke run (TabICL + TabPFN).
  continuumbench        RelBench-backed ContinuumBench run (TabICL + TabPFN).
  continuumbench-rt     ContinuumBench run with official relational-transformer.
  relbench-tabicl       Direct RelBench run with TabICL.
  relbench-tabpfn       Direct RelBench run with TabPFN.
  graphbench-tabicl     Direct GraphBench run with TabICL.
  graphbench-tabpfn     Direct GraphBench run with TabPFN.
  test                  Run the focused unit test suite.

Environment overrides:
  RELBENCH_DATASET   default: ${RELBENCH_DATASET}
  RELBENCH_TASK      default: ${RELBENCH_TASK}
  DEVICE             default: ${DEVICE}
  SEED               default: ${SEED}
  OUTPUT_ROOT        default: ${OUTPUT_ROOT}
  RT_REPO_PATH       default: ${RT_REPO_PATH}
  GRAPHBENCH_DATASET default: ${GRAPHBENCH_DATASET}
  GRAPHBENCH_TASK    default: ${GRAPHBENCH_TASK}
  GRAPHBENCH_ROOT    default: ${GRAPHBENCH_ROOT}
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
    run "${UV_BIN}" sync --extra dev --extra graphbench
    ;;

  test)
    run "${UV_BIN}" run "${PYTHON_BIN}" -m unittest -q \
      tests.test_continuumbench_relbench \
      tests.test_relbench_experiment \
      tests.test_graphbench_experiment \
      tests.test_graphbench_adapter
    ;;

  continuumbench-smoke)
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --benchmark-source synthetic \
      --models tabicl,tabpfn \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench_smoke"
    ;;

  continuumbench)
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --benchmark-source relbench \
      --relbench-dataset "${RELBENCH_DATASET}" \
      --relbench-task "${RELBENCH_TASK}" \
      --models tabicl,tabpfn \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench"
    ;;

  continuumbench-rt)
    require_rt_repo
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --benchmark-source relbench \
      --relbench-dataset "${RELBENCH_DATASET}" \
      --relbench-task "${RELBENCH_TASK}" \
      --models tabicl,tabpfn \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --use-official-rt-relational \
      --rt-repo-path "${RT_REPO_PATH}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench_rt"
    ;;

  relbench-tabicl)
    run "${UV_BIN}" run "${PYTHON_BIN}" relbench_run.py \
      --model tabicl \
      --dataset "${RELBENCH_DATASET}" \
      --task "${RELBENCH_TASK}" \
      --seed "${SEED}" \
      --output "${OUTPUT_ROOT}/relbench_tabicl.json"
    ;;

  relbench-tabpfn)
    run "${UV_BIN}" run "${PYTHON_BIN}" relbench_run.py \
      --model tabpfn \
      --dataset "${RELBENCH_DATASET}" \
      --task "${RELBENCH_TASK}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --output "${OUTPUT_ROOT}/relbench_tabpfn.json"
    ;;

  graphbench-tabicl)
    run "${UV_BIN}" run "${PYTHON_BIN}" graphbench_run.py \
      --model tabicl \
      --task "${GRAPHBENCH_TASK}" \
      --dataset "${GRAPHBENCH_DATASET}" \
      --root "${GRAPHBENCH_ROOT}" \
      --seed "${SEED}" \
      --output "${OUTPUT_ROOT}/graphbench_tabicl.json"
    ;;

  graphbench-tabpfn)
    run "${UV_BIN}" run "${PYTHON_BIN}" graphbench_run.py \
      --model tabpfn \
      --task "${GRAPHBENCH_TASK}" \
      --dataset "${GRAPHBENCH_DATASET}" \
      --root "${GRAPHBENCH_ROOT}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --output "${OUTPUT_ROOT}/graphbench_tabpfn.json"
    ;;

  *)
    usage
    if [[ -n "${target}" ]]; then
      exit 1
    fi
    ;;
esac
