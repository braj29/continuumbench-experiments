#!/usr/bin/env bash
#
# Copy/validate the 7 HomeCredit Kaggle CSV files into a target directory.
#
# Usage:
#   ./scripts/stage_homecredit_csvs.sh /path/to/source_dir
#   ./scripts/stage_homecredit_csvs.sh /path/to/source_dir /path/to/target_dir
#
# Defaults:
#   target_dir = <repo>/data/homecredit
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-}"
DST_DIR="${2:-${REPO_ROOT}/data/homecredit}"

if [[ -z "${SRC_DIR}" ]]; then
  echo "Usage: $0 <source_dir> [target_dir]" >&2
  echo "Example: $0 ~/Downloads/home-credit-default-risk /scratch-shared/\$USER/homecredit" >&2
  exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "ERROR: source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

required_csvs=(
  application_train.csv
  bureau.csv
  bureau_balance.csv
  previous_application.csv
  POS_CASH_balance.csv
  credit_card_balance.csv
  installments_payments.csv
)

for csv in "${required_csvs[@]}"; do
  if [[ ! -f "${SRC_DIR}/${csv}" ]]; then
    echo "ERROR: missing required file in source: ${SRC_DIR}/${csv}" >&2
    exit 1
  fi
done

mkdir -p "${DST_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -av --ignore-existing \
    "${SRC_DIR}/application_train.csv" \
    "${SRC_DIR}/bureau.csv" \
    "${SRC_DIR}/bureau_balance.csv" \
    "${SRC_DIR}/previous_application.csv" \
    "${SRC_DIR}/POS_CASH_balance.csv" \
    "${SRC_DIR}/credit_card_balance.csv" \
    "${SRC_DIR}/installments_payments.csv" \
    "${DST_DIR}/"
else
  cp -n \
    "${SRC_DIR}/application_train.csv" \
    "${SRC_DIR}/bureau.csv" \
    "${SRC_DIR}/bureau_balance.csv" \
    "${SRC_DIR}/previous_application.csv" \
    "${SRC_DIR}/POS_CASH_balance.csv" \
    "${SRC_DIR}/credit_card_balance.csv" \
    "${SRC_DIR}/installments_payments.csv" \
    "${DST_DIR}/"
fi

echo "Staged HomeCredit CSVs in: ${DST_DIR}"
for csv in "${required_csvs[@]}"; do
  ls -lh "${DST_DIR}/${csv}"
done

cat <<EOF

Next steps:
  export HC_DATA_DIR="${DST_DIR}"
  sbatch --export=ALL,TASK_SOURCE=homecredit,HC_DATA_DIR=\$HC_DATA_DIR,MODELS=tabpfn,SEED=7 \\
    scripts/snellius_tabpfn_a100.sbatch

Pretrained relational model for HomeCredit is RT (supported), not RelGT:
  HC_DATA_DIR=\$HC_DATA_DIR RT_USE_PRETRAINED=1 RT_ZERO_SHOT=1 \\
    ./scripts/submit_snellius_homecredit_rt.sh
EOF
