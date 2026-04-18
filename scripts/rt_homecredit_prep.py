"""
rt_homecredit_prep.py — Prepare HomeCredit data for the Relational Transformer.

The RT pipeline reads from a RelBench-format parquet directory that rustler
pre then converts to .rkyv binary files.  HomeCredit is not in RelBench, so
this script materialises it in the expected layout:

    {relbench_dir}/homecredit-default/
        db/
            application.parquet
            bureau.parquet
            bureau_balance.parquet
            previous_application.parquet
            pos_cash_balance.parquet
            credit_card_balance.parquet
            installments_payments.parquet
        tasks/
            loan-default/
                train.parquet
                val.parquet
                test.parquet

Critical requirement enforced here:
    RelBench (and rustler) require all primary-key columns to be consecutive
    0-based integer indices.  HomeCredit PKs are arbitrary (SK_ID_CURR starts
    at 100001, etc.), so every PK is remapped and every FK updated accordingly.

Usage (called automatically by snellius_homecredit_rt_a100.sbatch):

    python scripts/rt_homecredit_prep.py \\
        --hc-data-dir  /scratch-shared/$USER/homecredit \\
        --relbench-dir /scratch-shared/$USER/continuumbench-cache/relbench \\
        [--force-rewrite]

After this script finishes, run:
    rustler pre homecredit-default
    python relational-transformer/rt/embed.py homecredit-default
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the project src is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from continuumbench_experiments.continuumbench.sources_homecredit import (
    ENTITY_KEY,
    SEED_TIME_COL,
    TARGET_COL,
    _TRAIN_FRAC,
    _VAL_FRAC,
    _assign_application_dates,
    _add_relative_day_date,
    _add_relative_month_date,
    _check_files,
)
from relbench.base import Database, Table

# Fixed identifiers used by rt/main.py train_tasks / eval_tasks tuples.
HC_DB_NAME = "homecredit-default"
HC_TASK_NAME = "loan-default"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hc-data-dir",
        required=True,
        help="Directory containing the 7 Kaggle HomeCredit CSV files.",
    )
    parser.add_argument(
        "--relbench-dir",
        required=True,
        help=(
            "RelBench cache root.  Data is written to "
            "{relbench-dir}/homecredit-default/.  "
            "On Snellius this is typically $RELBENCH_CACHE_DIR, "
            "which ~/scratch/relbench symlinks to."
        ),
    )
    parser.add_argument(
        "--force-rewrite",
        action="store_true",
        help="Overwrite existing parquet files even if already present.",
    )
    args = parser.parse_args()

    hc_dir = Path(args.hc_data_dir)
    out_root = Path(args.relbench_dir) / HC_DB_NAME
    db_dir = out_root / "db"
    task_dir = out_root / "tasks" / HC_TASK_NAME

    _check_files(hc_dir)

    if not args.force_rewrite and _already_done(db_dir, task_dir):
        print(f"[rt_homecredit_prep] Parquet files already exist at {out_root}. "
              "Pass --force-rewrite to regenerate.")
        return

    print(f"[rt_homecredit_prep] Loading HomeCredit CSVs from {hc_dir} …")
    tic = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1. Load + date-assign
    # -----------------------------------------------------------------------
    app_df = pd.read_csv(hc_dir / "application_train.csv")
    app_df = app_df[app_df[TARGET_COL].notna()].copy()
    app_dates = _assign_application_dates(app_df)
    app_df[SEED_TIME_COL] = app_df[ENTITY_KEY].map(app_dates)

    bureau_df = pd.read_csv(hc_dir / "bureau.csv")
    bureau_df = _add_relative_day_date(bureau_df, "DAYS_CREDIT", "BUREAU_DATE", app_dates)

    bb_df = pd.read_csv(hc_dir / "bureau_balance.csv")

    prev_df = pd.read_csv(hc_dir / "previous_application.csv")
    prev_df = _add_relative_day_date(prev_df, "DAYS_DECISION", "PREV_APP_DATE", app_dates)

    pos_df = pd.read_csv(hc_dir / "POS_CASH_balance.csv")
    pos_df = _add_relative_month_date(pos_df, "MONTHS_BALANCE", "POS_CASH_DATE", app_dates)

    cc_df = pd.read_csv(hc_dir / "credit_card_balance.csv")
    cc_df = _add_relative_month_date(cc_df, "MONTHS_BALANCE", "CC_DATE", app_dates)

    inst_df = pd.read_csv(hc_dir / "installments_payments.csv")
    inst_df = _add_relative_day_date(inst_df, "DAYS_INSTALMENT", "INSTALL_DATE", app_dates)

    print(f"  CSVs loaded in {time.perf_counter() - tic:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Reindex PKs to consecutive 0-based integers (RelBench requirement)
    # -----------------------------------------------------------------------
    print("[rt_homecredit_prep] Reindexing primary keys …")

    # application: SK_ID_CURR → 0-based index
    app_id_map = _build_id_map(app_df[ENTITY_KEY])
    app_df[ENTITY_KEY] = app_df[ENTITY_KEY].map(app_id_map)

    # bureau: SK_ID_BUREAU → 0-based; remap FK SK_ID_CURR
    bureau_id_map = _build_id_map(bureau_df["SK_ID_BUREAU"])
    bureau_df["SK_ID_BUREAU"] = bureau_df["SK_ID_BUREAU"].map(bureau_id_map)
    bureau_df[ENTITY_KEY] = _remap_fk(bureau_df[ENTITY_KEY], app_id_map)

    # bureau_balance: row-index PK; remap FK SK_ID_BUREAU
    bb_df["_bb_row_id"] = np.arange(len(bb_df), dtype=np.int64)
    bb_df["SK_ID_BUREAU"] = _remap_fk(bb_df["SK_ID_BUREAU"], bureau_id_map)

    # previous_application: SK_ID_PREV → 0-based; remap FK SK_ID_CURR
    prev_id_map = _build_id_map(prev_df["SK_ID_PREV"])
    prev_df["SK_ID_PREV"] = prev_df["SK_ID_PREV"].map(prev_id_map)
    prev_df[ENTITY_KEY] = _remap_fk(prev_df[ENTITY_KEY], app_id_map)

    # pos_cash_balance: row-index PK; remap both FKs
    pos_df["_pos_row_id"] = np.arange(len(pos_df), dtype=np.int64)
    pos_df[ENTITY_KEY] = _remap_fk(pos_df[ENTITY_KEY], app_id_map)
    pos_df["SK_ID_PREV"] = _remap_fk(pos_df["SK_ID_PREV"], prev_id_map)

    # credit_card_balance: row-index PK; remap both FKs
    cc_df["_cc_row_id"] = np.arange(len(cc_df), dtype=np.int64)
    cc_df[ENTITY_KEY] = _remap_fk(cc_df[ENTITY_KEY], app_id_map)
    cc_df["SK_ID_PREV"] = _remap_fk(cc_df["SK_ID_PREV"], prev_id_map)

    # installments_payments: row-index PK; remap both FKs
    inst_df["_inst_row_id"] = np.arange(len(inst_df), dtype=np.int64)
    inst_df[ENTITY_KEY] = _remap_fk(inst_df[ENTITY_KEY], app_id_map)
    inst_df["SK_ID_PREV"] = _remap_fk(inst_df["SK_ID_PREV"], prev_id_map)

    # -----------------------------------------------------------------------
    # 3. Build RelBench Table objects
    # -----------------------------------------------------------------------
    print("[rt_homecredit_prep] Building RelBench Table objects …")

    # application — entity features (no TARGET column; it lives in the task table)
    entity_df = app_df.drop(columns=[TARGET_COL]).copy()
    tables = {
        "application": Table(
            df=entity_df,
            fkey_col_to_pkey_table={},
            pkey_col=ENTITY_KEY,
            time_col=SEED_TIME_COL,
        ),
        "bureau": Table(
            df=bureau_df,
            fkey_col_to_pkey_table={ENTITY_KEY: "application"},
            pkey_col="SK_ID_BUREAU",
            time_col="BUREAU_DATE",
        ),
        "bureau_balance": Table(
            df=bb_df,
            fkey_col_to_pkey_table={"SK_ID_BUREAU": "bureau"},
            pkey_col="_bb_row_id",
            time_col=None,
        ),
        "previous_application": Table(
            df=prev_df,
            fkey_col_to_pkey_table={ENTITY_KEY: "application"},
            pkey_col="SK_ID_PREV",
            time_col="PREV_APP_DATE",
        ),
        "pos_cash_balance": Table(
            df=pos_df,
            fkey_col_to_pkey_table={
                ENTITY_KEY: "application",
                "SK_ID_PREV": "previous_application",
            },
            pkey_col="_pos_row_id",
            time_col="POS_CASH_DATE",
        ),
        "credit_card_balance": Table(
            df=cc_df,
            fkey_col_to_pkey_table={
                ENTITY_KEY: "application",
                "SK_ID_PREV": "previous_application",
            },
            pkey_col="_cc_row_id",
            time_col="CC_DATE",
        ),
        "installments_payments": Table(
            df=inst_df,
            fkey_col_to_pkey_table={
                ENTITY_KEY: "application",
                "SK_ID_PREV": "previous_application",
            },
            pkey_col="_inst_row_id",
            time_col="INSTALL_DATE",
        ),
    }

    # -----------------------------------------------------------------------
    # 4. Build task split tables (train / val / test)
    # -----------------------------------------------------------------------
    print("[rt_homecredit_prep] Building task split tables …")

    task_df = app_df[[ENTITY_KEY, SEED_TIME_COL, TARGET_COL]].copy()
    task_df[TARGET_COL] = task_df[TARGET_COL].astype(int)

    sorted_dates = task_df[SEED_TIME_COL].sort_values().reset_index(drop=True)
    n = len(sorted_dates)
    val_cutoff = sorted_dates.iloc[int(_TRAIN_FRAC * n)]
    test_cutoff = sorted_dates.iloc[int((_TRAIN_FRAC + _VAL_FRAC) * n)]

    seed_times = pd.to_datetime(task_df[SEED_TIME_COL], utc=False)
    train_mask = seed_times < val_cutoff
    val_mask = (seed_times >= val_cutoff) & (seed_times < test_cutoff)
    test_mask = seed_times >= test_cutoff

    # Task tables are FK-linked to the entity table.  No PK column (they are
    # prediction instances, not entities).
    task_fkeys = {ENTITY_KEY: "application"}
    split_tables = {
        "train": Table(df=task_df[train_mask].reset_index(drop=True),
                       fkey_col_to_pkey_table=task_fkeys,
                       pkey_col=None, time_col=SEED_TIME_COL),
        "val":   Table(df=task_df[val_mask].reset_index(drop=True),
                       fkey_col_to_pkey_table=task_fkeys,
                       pkey_col=None, time_col=SEED_TIME_COL),
        "test":  Table(df=task_df[test_mask].reset_index(drop=True),
                       fkey_col_to_pkey_table=task_fkeys,
                       pkey_col=None, time_col=SEED_TIME_COL),
    }

    print(f"  train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")

    # -----------------------------------------------------------------------
    # 5. Write to disk
    # -----------------------------------------------------------------------
    print(f"[rt_homecredit_prep] Writing parquet files to {out_root} …")

    db = Database(table_dict=tables)
    db_dir.mkdir(parents=True, exist_ok=True)
    db.save(str(db_dir))

    task_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_tbl in split_tables.items():
        split_tbl.save(str(task_dir / f"{split_name}.parquet"))

    elapsed = time.perf_counter() - tic
    print(f"[rt_homecredit_prep] Done in {elapsed:.1f}s.")
    print(f"  DB tables  : {db_dir}")
    print(f"  Task splits: {task_dir}")
    print()
    print("Next steps:")
    print(f"  1. rustler pre {HC_DB_NAME}")
    print(f"  2. python relational-transformer/rt/embed.py {HC_DB_NAME}")
    print(f"  3. Run continuumbench --task-source homecredit --use-official-rt-relational ...")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_id_map(series: pd.Series) -> dict:
    """Return {old_id: new_0based_index} for unique sorted values."""
    unique_ids = sorted(series.dropna().unique())
    return {old: new for new, old in enumerate(unique_ids)}


def _remap_fk(series: pd.Series, id_map: dict) -> pd.Series:
    """Remap FK values using id_map; unmapped values become NaN (dangling FKs)."""
    return series.map(id_map)


def _already_done(db_dir: Path, task_dir: Path) -> bool:
    expected_db = {
        "application", "bureau", "bureau_balance",
        "previous_application", "pos_cash_balance",
        "credit_card_balance", "installments_payments",
    }
    expected_splits = {"train", "val", "test"}
    db_ok = all((db_dir / f"{t}.parquet").exists() for t in expected_db)
    task_ok = all((task_dir / f"{s}.parquet").exists() for s in expected_splits)
    return db_ok and task_ok


if __name__ == "__main__":
    main()
