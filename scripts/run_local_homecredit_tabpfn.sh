#!/usr/bin/env bash
#
# One-command local HomeCredit + TabPFN runner (macOS/Linux).
# - Auto-detects HomeCredit CSV source in ~/Downloads by default.
# - Stages CSVs via scripts/stage_homecredit_csvs.sh.
# - Runs ContinuumBench joined-track TabPFN on --task-source homecredit.
#
# Usage:
#   ./scripts/run_local_homecredit_tabpfn.sh
#   ./scripts/run_local_homecredit_tabpfn.sh /path/to/homecredit_csv_dir
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SRC_DIR="${1:-}"
STAGED_DIR="${STAGED_DIR:-${REPO_ROOT}/data/homecredit}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/local_homecredit_tabpfn}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cpu}"

if [[ -z "${SRC_DIR}" ]]; then
  csv_path="$(find "${HOME}/Downloads" -maxdepth 6 -type f -name application_train.csv 2>/dev/null | head -n1 || true)"
  if [[ -n "${csv_path}" ]]; then
    SRC_DIR="$(dirname "${csv_path}")"
  fi
fi

if [[ -z "${SRC_DIR}" ]]; then
  cat >&2 <<EOF
ERROR: Could not auto-detect HomeCredit CSV folder in ~/Downloads.

Pass the folder explicitly:
  ./scripts/run_local_homecredit_tabpfn.sh "/path/to/home-credit-default-risk"
EOF
  exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "ERROR: source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

echo "Staging HomeCredit CSVs..."
./scripts/stage_homecredit_csvs.sh "${SRC_DIR}" "${STAGED_DIR}"

echo "Running local HomeCredit + TabPFN..."
uv run python continuumbench_tabfm_run.py \
  --task-source homecredit \
  --homecredit-data-dir "${STAGED_DIR}" \
  --track joined \
  --models tabpfn \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --tabpfn-ignore-pretraining-limits \
  --no-run-protocol-b \
  --output-dir "${OUTPUT_DIR}"

echo "Done."
echo "Results:"
echo "  ${OUTPUT_DIR}"
echo "  ${OUTPUT_DIR}/manifest.json"
