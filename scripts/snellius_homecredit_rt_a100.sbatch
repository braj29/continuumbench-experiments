#!/usr/bin/env bash
# HomeCredit — relational track with the official Relational Transformer.
# Pipeline:
#   1. scripts/rt_homecredit_prep.py   — CSV → RelBench parquet + PK reindex
#   2. rustler pre homecredit-default  — parquet → .rkyv binary files
#   3. rt/embed.py homecredit-default  — text embeddings
#   4. continuumbench --track relational --use-official-rt-relational
#
# Pretrained checkpoint (optional):
#   RT_USE_PRETRAINED=1  downloads pretrain_<dataset>_<task>.pt from HF Hub.
#   RT_ZERO_SHOT=1       evaluates without fine-tuning (requires RT_USE_PRETRAINED=1).
#   RT_CKPT_PATH         supply an explicit local checkpoint path instead.
#
# Key env vars:
#   HC_DATA_DIR          Path to directory with 7 Kaggle CSVs [required]
#   RT_REPO_PATH         Path to relational-transformer checkout [default: ./relational-transformer]
#   RT_USE_PRETRAINED    1 to auto-download pretrained checkpoint   [default: 0]
#   RT_ZERO_SHOT         1 to skip fine-tuning (needs RT_USE_PRETRAINED=1) [default: 0]
#   RT_CKPT_PATH         Explicit checkpoint path (overrides RT_USE_PRETRAINED)
#   RT_MAX_STEPS         Fine-tuning steps                          [default: 4097]
#   MODELS               Output folder key (tabular model name)     [default: tabpfn]
#   SEED                 Random seed                                [default: 7]

#SBATCH -J cb-hc-rt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=240G
#SBATCH -p gpu_a100
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

HC_DATA_DIR="${HC_DATA_DIR:-}"
MODELS="${MODELS:-tabpfn}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
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

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RT_REPO_PATH="${RT_REPO_PATH:-${REPO_ROOT}/relational-transformer}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/continuumbench-cache}"
RT_PREPROCESSED_ROOT="${RT_PREPROCESSED_ROOT:-${HOME}/scratch/pre}"
RT_CKPT_DIR="${RT_CKPT_DIR:-${HOME}/scratch/rt_ckpts}"
RT_HF_REPO="${RT_HF_REPO:-rishabh-ranjan/relational-transformer}"
JOB_TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
JOB_OUTPUT_DIR="${JOB_TMPDIR}/hc_rt"
RESULTS_DIR="${REPO_ROOT}/outputs/snellius/${SLURM_JOB_ID}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-snellius}"
RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-${CACHE_ROOT}/relbench}"

[[ -z "${HC_DATA_DIR}" ]] && { echo "ERROR: HC_DATA_DIR is required." >&2; exit 1; }
[[ ! -d "${HC_DATA_DIR}" ]] && { echo "ERROR: HC_DATA_DIR not found: ${HC_DATA_DIR}" >&2; exit 1; }
[[ ! -f "${RT_REPO_PATH}/rt/main.py" ]] && {
  echo "ERROR: RT repo not found at ${RT_REPO_PATH}" >&2
  echo "Set RT_REPO_PATH or clone to ./relational-transformer" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
module purge
module load "${SNELLIUS_STACK_MODULE}"
module load "${SNELLIUS_PYTHON_MODULE}"

detect_latest_rust_module() {
  module -t avail Rust 2>&1 | awk '/^Rust\/[0-9]/{print}' | sort -t'/' -k2,2V | tail -n1
}
if ! command -v rustc >/dev/null 2>&1; then
  if [[ -n "${SNELLIUS_RUST_MODULE}" ]]; then
    module load "${SNELLIUS_RUST_MODULE}"
  else
    _auto_rust="$(detect_latest_rust_module)"
    [[ -n "${_auto_rust}" ]] && module load "${_auto_rust}" || true
  fi
fi

cd "${REPO_ROOT}"
[[ ! -f "${VENV_DIR}/bin/activate" ]] && { echo "Run setup_snellius_env.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

export RELBENCH_CACHE_DIR
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${CACHE_ROOT}/tabpfn}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

mkdir -p "${JOB_OUTPUT_DIR}" "${RESULTS_DIR}" "${RT_PREPROCESSED_ROOT}" "${RT_CKPT_DIR}"

python - <<'PY'
import os, torch
print(f"Python: {os.sys.executable}")
print(f"Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
PY

# ---------------------------------------------------------------------------
# Rust version check (same logic as snellius_rt_only_a100.sbatch)
# ---------------------------------------------------------------------------
_RUSTUP_CACHE_HOME="${REPO_ROOT}/.rustup"
_CARGO_CACHE_HOME="${REPO_ROOT}/.cargo"
_NEED_RUSTC_MINOR=81

_rustc_minor() { rustc --version 2>/dev/null | awk '{print $2}' | cut -d. -f2; }
cur_minor="$(_rustc_minor 2>/dev/null || echo '0')"
if [[ "${cur_minor}" -lt "${_NEED_RUSTC_MINOR}" ]]; then
  if [[ -x "${_CARGO_CACHE_HOME}/bin/rustc" ]]; then
    export RUSTUP_HOME="${_RUSTUP_CACHE_HOME}"
    export CARGO_HOME="${_CARGO_CACHE_HOME}"
    export PATH="${_CARGO_CACHE_HOME}/bin:${PATH}"
    cur_minor="$(_rustc_minor)"
  fi
  if [[ "${cur_minor}" -lt "${_NEED_RUSTC_MINOR}" ]]; then
    echo "ERROR: rustc >= 1.${_NEED_RUSTC_MINOR} required; run setup_snellius_env.sh." >&2
    exit 1
  fi
fi
echo "Rust: $(rustc --version)"

# ---------------------------------------------------------------------------
# Install RT runtime deps (idempotent)
# ---------------------------------------------------------------------------
for pkg_spec in "wandb:wandb" "strictfire:strictfire" \
    "maturin_import_hook:maturin-import-hook" "maturin:maturin[patchelf]" \
    "ml_dtypes:ml-dtypes" "orjson:orjson" "einops:einops" \
    "sentence_transformers:sentence-transformers" "polars:polars" "uv:uv"; do
  mod="${pkg_spec%%:*}"; pkg="${pkg_spec##*:}"
  python -c "import ${mod}" >/dev/null 2>&1 || python -m pip install --quiet "${pkg}"
done

# ---------------------------------------------------------------------------
# Build rustler extension if needed
# ---------------------------------------------------------------------------
cargo_toml="${RT_REPO_PATH}/rustler/Cargo.toml"
cargo_minor="$(cargo --version | awk '{print $2}' | cut -d. -f2)"
if [[ "${cargo_minor}" -lt 85 ]] && grep -q 'edition = "2024"' "${cargo_toml}" 2>/dev/null; then
  echo "Patching rustler Cargo.toml: edition 2024 → 2021 for cargo < 1.85"
  sed -i 's/edition = "2024"/edition = "2021"/' "${cargo_toml}"
fi

rustler_ok() { python -c "from rustler import Sampler" >/dev/null 2>&1; }
if ! rustler_ok; then
  python -m pip uninstall -y rustler 2>/dev/null || true
  (cd "${RT_REPO_PATH}/rustler" && python -m maturin develop --release --uv) \
    || (cd "${RT_REPO_PATH}/rustler" && python -m maturin develop --release)
fi
rustler_ok || { echo "ERROR: rustler build failed." >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: Convert HomeCredit CSVs → RelBench parquet
# ---------------------------------------------------------------------------
HC_PARQUET_DIR="${RELBENCH_CACHE_DIR}/homecredit-default"
if [[ ! -f "${HC_PARQUET_DIR}/db/application.parquet" ]]; then
  echo "=== Step 1: rt_homecredit_prep.py ==="
  python "${REPO_ROOT}/scripts/rt_homecredit_prep.py" \
    --hc-data-dir "${HC_DATA_DIR}" \
    --relbench-dir "${RELBENCH_CACHE_DIR}"
else
  echo "=== Step 1: RelBench parquet already present at ${HC_PARQUET_DIR} — skipping ==="
fi

# Ensure ~/scratch/relbench → RELBENCH_CACHE_DIR symlink (rustler pre hardcodes this)
mkdir -p "${HOME}/scratch"
if [[ ! -e "${HOME}/scratch/relbench" ]]; then
  ln -s "${RELBENCH_CACHE_DIR}" "${HOME}/scratch/relbench"
elif [[ ! -L "${HOME}/scratch/relbench" ]]; then
  echo "WARNING: ${HOME}/scratch/relbench exists but is not a symlink." >&2
fi

# ---------------------------------------------------------------------------
# Step 2: rustler pre homecredit-default → .rkyv binary files
# ---------------------------------------------------------------------------
RT_PRE_DIR="${RT_PREPROCESSED_ROOT}/homecredit-default"
rt_rkyv_ok() {
  [[ -f "${RT_PRE_DIR}/nodes.rkyv" ]] \
  && [[ -f "${RT_PRE_DIR}/offsets.rkyv" ]] \
  && [[ -f "${RT_PRE_DIR}/p2f_adj.rkyv" ]] \
  && [[ -f "${RT_PRE_DIR}/table_info.json" ]] \
  && [[ -f "${RT_PRE_DIR}/column_index.json" ]]
}

if rt_rkyv_ok; then
  echo "=== Step 2: .rkyv files already present — skipping rustler pre ==="
else
  echo "=== Step 2: Running rustler pre homecredit-default ==="
  RUSTLER_BIN="${RT_REPO_PATH}/rustler/target/release/rustler"
  if [[ ! -x "${RUSTLER_BIN}" ]]; then
    echo "Building rustler binary …"
    (cd "${RT_REPO_PATH}/rustler" && cargo build --release --bin rustler)
  fi
  mkdir -p "${RT_PRE_DIR}"
  "${RUSTLER_BIN}" pre homecredit-default
fi

# ---------------------------------------------------------------------------
# Step 3: Text embeddings
# ---------------------------------------------------------------------------
TEXT_EMB_PATH="${RT_PRE_DIR}/text_emb_${EMBEDDING_MODEL}.bin"
if [[ -f "${TEXT_EMB_PATH}" ]]; then
  echo "=== Step 3: Text embeddings already present — skipping ==="
else
  echo "=== Step 3: Generating text embeddings (${EMBEDDING_MODEL}) ==="
  PYTHONPATH="${RT_REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${RT_REPO_PATH}/rt/embed.py" homecredit-default \
      --embedding_model "${EMBEDDING_MODEL}"
fi

# ---------------------------------------------------------------------------
# Step 4: Pretrained checkpoint (optional)
# ---------------------------------------------------------------------------
rt_ckpt_flags=()
if [[ -n "${RT_CKPT_PATH}" ]]; then
  echo "Using explicit checkpoint: ${RT_CKPT_PATH}"
  rt_ckpt_flags+=(--rt-load-ckpt-path "${RT_CKPT_PATH}")
elif [[ "${RT_USE_PRETRAINED}" == "1" ]]; then
  RT_CKPT_NAME="pretrain_homecredit-default_loan-default.pt"
  RT_CKPT_FULL="${RT_CKPT_DIR}/${RT_CKPT_NAME}"
  if [[ ! -f "${RT_CKPT_FULL}" ]]; then
    echo "Downloading pretrained checkpoint from HF Hub (${RT_HF_REPO}) …"
    python - <<PY
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="${RT_HF_REPO}",
    filename="${RT_CKPT_NAME}",
    local_dir="${RT_CKPT_DIR}",
)
print(f"Downloaded: {path}")
PY
  fi
  rt_ckpt_flags+=(--rt-load-ckpt-path "${RT_CKPT_FULL}")
fi
if [[ "${RT_ZERO_SHOT}" == "1" ]]; then
  [[ ${#rt_ckpt_flags[@]} -eq 0 ]] && { echo "ERROR: RT_ZERO_SHOT=1 needs RT_USE_PRETRAINED=1 or RT_CKPT_PATH." >&2; exit 1; }
  rt_ckpt_flags+=(--rt-zero-shot)
fi

# ---------------------------------------------------------------------------
# Step 4: Run ContinuumBench relational track
# ---------------------------------------------------------------------------
echo "=== Step 4: ContinuumBench — relational track (RT) ==="
echo "  seed=${SEED}  max_steps=${RT_MAX_STEPS}  zero_shot=${RT_ZERO_SHOT}"

srun --ntasks=1 python "${REPO_ROOT}/continuumbench_tabfm_run.py" \
  --task-source homecredit \
  --homecredit-data-dir "${HC_DATA_DIR}" \
  --track relational \
  --models "${MODELS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --use-official-rt-relational \
  --rt-repo-path "${RT_REPO_PATH}" \
  --rt-max-steps "${RT_MAX_STEPS}" \
  --rt-batch-size "${RT_BATCH_SIZE}" \
  --rt-num-workers "${RT_NUM_WORKERS}" \
  --no-run-protocol-b \
  --output-dir "${JOB_OUTPUT_DIR}" \
  "${rt_ckpt_flags[@]}"

rsync -a "${JOB_OUTPUT_DIR}/" "${RESULTS_DIR}/" 2>/dev/null || cp -r "${JOB_OUTPUT_DIR}/." "${RESULTS_DIR}/"
echo "=== Done. Results at ${RESULTS_DIR} ==="
