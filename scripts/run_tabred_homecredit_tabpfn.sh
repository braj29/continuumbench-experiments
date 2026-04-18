#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_PYTHON_BIN="python"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

# If the caller left the generic default "python", prefer this repo's venv.
if [[ "${PYTHON_BIN}" == "python" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
TABRED_DATA_DIR="${TABRED_DATA_DIR:-${REPO_ROOT}/data/tabred/homecredit-default}"
SPLIT="${SPLIT:-default}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-7}"
INCLUDE_META="${INCLUDE_META:-0}"
TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/outputs/tabred_homecredit_tabpfn/result.json}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find '${PYTHON_BIN}' (or python3) on PATH." >&2
    exit 1
  fi
fi

bool_flag() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes) return 0 ;;
    0|false|no|'') return 1 ;;
    *)
      echo "Invalid boolean value: '$1' (expected true/false/1/0/yes/no)" >&2
      exit 1
      ;;
  esac
}

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  "${PYTHON_BIN}" -m continuumbench_experiments.tabred_homecredit_tabpfn
  --tabred-data-dir "${TABRED_DATA_DIR}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --output-json "${OUTPUT_JSON}"
)

if bool_flag "${INCLUDE_META}"; then
  cmd+=(--include-meta)
else
  cmd+=(--no-include-meta)
fi

if bool_flag "${TABPFN_IGNORE_PRETRAINING_LIMITS}"; then
  cmd+=(--tabpfn-ignore-pretraining-limits)
else
  cmd+=(--no-tabpfn-ignore-pretraining-limits)
fi

if [[ -n "${MAX_TRAIN_ROWS}" ]]; then
  cmd+=(--max-train-rows "${MAX_TRAIN_ROWS}")
fi

if bool_flag "${DRY_RUN}"; then
  cmd+=(--dry-run)
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"
