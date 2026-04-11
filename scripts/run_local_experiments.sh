#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
UV_BIN="${UV_BIN:-uv}"

DATASET_NAME="${DATASET_NAME:-rel-f1}"
TASK_NAME="${TASK_NAME:-driver-top3}"
MODELS="${MODELS:-tabicl,tabpfn}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_runs}"
RT_REPO_PATH="${RT_REPO_PATH:-$REPO_ROOT/relational-transformer}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-true}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-true}"

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

download_artifacts_spec="$(lowercase "${DOWNLOAD_ARTIFACTS}")"
case "${download_artifacts_spec}" in
  1|true|yes)
    download_artifacts_args=(--download-artifacts)
    ;;
  0|false|no)
    download_artifacts_args=(--no-download-artifacts)
    ;;
  *)
    echo "DOWNLOAD_ARTIFACTS must be one of: true, false, 1, 0, yes, no" >&2
    exit 1
    ;;
esac

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
  MODELS             default: ${MODELS}
  DEVICE             default: ${DEVICE}
  SEED               default: ${SEED}
  OUTPUT_ROOT        default: ${OUTPUT_ROOT}
  RT_REPO_PATH       default: ${RT_REPO_PATH}
  DOWNLOAD_ARTIFACTS default: ${DOWNLOAD_ARTIFACTS}
  TABPFN_IGNORE_PRETRAINING_LIMITS default: ${TABPFN_IGNORE_PRETRAINING_LIMITS}
EOF
}

run() {
  echo "+ $*"
  "$@"
}

tabpfn_limits_spec="$(lowercase "${TABPFN_IGNORE_PRETRAINING_LIMITS}")"
case "${tabpfn_limits_spec}" in
  1|true|yes)
    tabpfn_limits_args=(--tabpfn-ignore-pretraining-limits)
    ;;
  0|false|no)
    tabpfn_limits_args=()
    ;;
  *)
    echo "TABPFN_IGNORE_PRETRAINING_LIMITS must be one of: true, false, 1, 0, yes, no" >&2
    exit 1
    ;;
esac

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
      tests.test_models_adapters \
      tests.test_models_device_resolution \
      tests.test_models_tabular \
      tests.test_continuumbench_harness \
      tests.test_continuumbench_cli
    ;;

  continuumbench-smoke)
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --task-source synthetic \
      --models "${MODELS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      "${tabpfn_limits_args[@]}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench_smoke"
    ;;

  continuumbench)
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --task-source dataset \
      --dataset-name "${DATASET_NAME}" \
      --task-name "${TASK_NAME}" \
      "${download_artifacts_args[@]}" \
      --models "${MODELS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      "${tabpfn_limits_args[@]}" \
      --output-dir "${OUTPUT_ROOT}/continuumbench"
    ;;

  continuumbench-rt)
    require_rt_repo
    run "${UV_BIN}" run "${PYTHON_BIN}" continuumbench_tabfm_run.py \
      --task-source dataset \
      --dataset-name "${DATASET_NAME}" \
      --task-name "${TASK_NAME}" \
      "${download_artifacts_args[@]}" \
      --models "${MODELS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      "${tabpfn_limits_args[@]}" \
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
