"""
Home Credit Default Risk — ContinuumBench multi-table integration.

Loads the original Kaggle multi-table dataset and exposes it as the standard
(DatabaseSpec, TaskSpec, TemporalSplit) triple consumed by the harness.

Schema (7 tables):
    application        PK=SK_ID_CURR, time=APPLICATION_DATE (synthetic)
    bureau             PK=SK_ID_BUREAU, FK→application, time=BUREAU_DATE
    bureau_balance     PK=_bb_row_id,  FK→bureau,       time=None (static)
    previous_application PK=SK_ID_PREV, FK→application, time=PREV_APP_DATE
    pos_cash_balance   PK=_pos_row_id, FK→application+previous_application, time=POS_CASH_DATE
    credit_card_balance PK=_cc_row_id, FK→application+previous_application, time=CC_DATE
    installments_payments PK=_inst_row_id, FK→application+previous_application, time=INSTALL_DATE

Temporal design:
    HomeCredit has no real application timestamps.  We assign synthetic dates
    by linearly spacing SK_ID_CURR sort-rank over a fixed reference window
    (REF_START … REF_END), turning ordinal application order into wall-clock
    time.  Incident-table dates are then computed from relative DAYS_*/
    MONTHS_BALANCE columns using each row's parent APPLICATION_DATE.

    Train / val / test split: 70 / 15 / 15 percent by APPLICATION_DATE order.

Expected files in data_dir:
    application_train.csv
    bureau.csv
    bureau_balance.csv
    previous_application.csv
    POS_CASH_balance.csv
    credit_card_balance.csv
    installments_payments.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .metrics import make_temporal_split
from .specs import DatabaseSpec, TableSpec, TaskSpec, TemporalSplit

# Synthetic date window that spans the approximate competition data collection period.
_REF_START = pd.Timestamp("2015-06-01")
_REF_END = pd.Timestamp("2016-09-30")
_TRAIN_FRAC = 0.70
_VAL_FRAC = 0.15  # test gets the remainder

TASK_NAME = "homecredit-default__loan-default"
ENTITY_TABLE = "application"
ENTITY_KEY = "SK_ID_CURR"
SEED_TIME_COL = "APPLICATION_DATE"
TARGET_COL = "TARGET"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_homecredit_default(
    data_dir: Union[str, Path],
) -> tuple[DatabaseSpec, TaskSpec, TemporalSplit]:
    """
    Build (DatabaseSpec, TaskSpec, TemporalSplit) from raw Kaggle HomeCredit CSVs.

    Parameters
    ----------
    data_dir:
        Directory containing the seven Kaggle CSV files listed in the module
        docstring.  Pass the path where you extracted the competition archive.
    """
    data_dir = Path(data_dir)
    _check_files(data_dir)

    # --- application (entity table) ---
    app_df = pd.read_csv(data_dir / "application_train.csv")
    app_df = app_df[app_df[TARGET_COL].notna()].copy()
    app_dates = _assign_application_dates(app_df)
    app_df[SEED_TIME_COL] = app_df[ENTITY_KEY].map(app_dates)

    # Task table: (entity_key, seed_time, target) — one row per labelled application.
    task_table = app_df[[ENTITY_KEY, SEED_TIME_COL, TARGET_COL]].copy()
    task_table[TARGET_COL] = task_table[TARGET_COL].astype(int)

    # Entity feature table: all columns except the label.
    entity_df = app_df.drop(columns=[TARGET_COL]).copy()

    # --- bureau ---
    bureau_df = pd.read_csv(data_dir / "bureau.csv")
    bureau_df = _add_relative_day_date(bureau_df, "DAYS_CREDIT", "BUREAU_DATE", app_dates)

    # --- bureau_balance ---
    bb_df = pd.read_csv(data_dir / "bureau_balance.csv")
    bb_df["_bb_row_id"] = np.arange(len(bb_df), dtype=np.int64)

    # --- previous_application ---
    prev_df = pd.read_csv(data_dir / "previous_application.csv")
    prev_df = _add_relative_day_date(prev_df, "DAYS_DECISION", "PREV_APP_DATE", app_dates)

    # --- pos_cash_balance ---
    pos_df = pd.read_csv(data_dir / "POS_CASH_balance.csv")
    pos_df = _add_relative_month_date(pos_df, "MONTHS_BALANCE", "POS_CASH_DATE", app_dates)
    pos_df["_pos_row_id"] = np.arange(len(pos_df), dtype=np.int64)

    # --- credit_card_balance ---
    cc_df = pd.read_csv(data_dir / "credit_card_balance.csv")
    cc_df = _add_relative_month_date(cc_df, "MONTHS_BALANCE", "CC_DATE", app_dates)
    cc_df["_cc_row_id"] = np.arange(len(cc_df), dtype=np.int64)

    # --- installments_payments ---
    inst_df = pd.read_csv(data_dir / "installments_payments.csv")
    inst_df = _add_relative_day_date(inst_df, "DAYS_INSTALMENT", "INSTALL_DATE", app_dates)
    inst_df["_inst_row_id"] = np.arange(len(inst_df), dtype=np.int64)

    db = DatabaseSpec(
        tables={
            "application": TableSpec(
                name="application",
                df=entity_df,
                primary_key=ENTITY_KEY,
                time_col=SEED_TIME_COL,
                foreign_keys={},
            ),
            "bureau": TableSpec(
                name="bureau",
                df=bureau_df,
                primary_key="SK_ID_BUREAU",
                time_col="BUREAU_DATE",
                foreign_keys={ENTITY_KEY: "application"},
            ),
            "bureau_balance": TableSpec(
                name="bureau_balance",
                df=bb_df,
                primary_key="_bb_row_id",
                time_col=None,
                foreign_keys={"SK_ID_BUREAU": "bureau"},
            ),
            "previous_application": TableSpec(
                name="previous_application",
                df=prev_df,
                primary_key="SK_ID_PREV",
                time_col="PREV_APP_DATE",
                foreign_keys={ENTITY_KEY: "application"},
            ),
            "pos_cash_balance": TableSpec(
                name="pos_cash_balance",
                df=pos_df,
                primary_key="_pos_row_id",
                time_col="POS_CASH_DATE",
                foreign_keys={
                    ENTITY_KEY: "application",
                    "SK_ID_PREV": "previous_application",
                },
            ),
            "credit_card_balance": TableSpec(
                name="credit_card_balance",
                df=cc_df,
                primary_key="_cc_row_id",
                time_col="CC_DATE",
                foreign_keys={
                    ENTITY_KEY: "application",
                    "SK_ID_PREV": "previous_application",
                },
            ),
            "installments_payments": TableSpec(
                name="installments_payments",
                df=inst_df,
                primary_key="_inst_row_id",
                time_col="INSTALL_DATE",
                foreign_keys={
                    ENTITY_KEY: "application",
                    "SK_ID_PREV": "previous_application",
                },
            ),
        }
    )

    task = TaskSpec(
        name=TASK_NAME,
        task_type="classification",
        target_col=TARGET_COL,
        metric_name="auroc",
        entity_table=ENTITY_TABLE,
        entity_key=ENTITY_KEY,
        seed_time_col=SEED_TIME_COL,
        task_table=task_table,
    )

    sorted_dates = task_table[SEED_TIME_COL].sort_values().reset_index(drop=True)
    n = len(sorted_dates)
    val_cutoff = sorted_dates.iloc[int(_TRAIN_FRAC * n)]
    test_cutoff = sorted_dates.iloc[int((_TRAIN_FRAC + _VAL_FRAC) * n)]
    split = make_temporal_split(
        task_df=task_table,
        seed_time_col=SEED_TIME_COL,
        val_cutoff=val_cutoff,
        test_cutoff=test_cutoff,
    )

    return db, task, split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_files(data_dir: Path) -> None:
    required = [
        "application_train.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing HomeCredit CSV files in {data_dir}:\n  "
            + "\n  ".join(missing)
            + "\nDownload from https://www.kaggle.com/c/home-credit-default-risk/data"
        )


def _assign_application_dates(app_df: pd.DataFrame) -> pd.Series:
    """
    Map each SK_ID_CURR to a synthetic wall-clock timestamp.

    Applications are sorted by SK_ID_CURR (a monotone application ID, so rank
    is a reasonable chronological proxy) and linearly spread over the reference
    window [REF_START, REF_END].  Returns a Series indexed by SK_ID_CURR.
    """
    total_days = (_REF_END - _REF_START).days  # 487
    sorted_ids = app_df[ENTITY_KEY].sort_values().reset_index(drop=True)
    n = len(sorted_ids)
    day_offsets = (sorted_ids.index / max(n - 1, 1) * total_days).astype(int)
    dates = _REF_START + pd.to_timedelta(day_offsets, unit="D")
    return pd.Series(dates.values, index=sorted_ids.values)


def _add_relative_day_date(
    df: pd.DataFrame,
    days_col: str,
    date_col: str,
    app_dates: pd.Series,
) -> pd.DataFrame:
    """
    Compute absolute date for child-table rows using a relative DAYS_* column.

    For each row, looks up the parent application date via SK_ID_CURR and adds
    the (typically negative) day offset.  Rows where DAYS_* or SK_ID_CURR is
    missing get NaT (the view builder treats NaT rows as always within window).
    """
    df = df.copy()
    parent_dates = df[ENTITY_KEY].map(app_dates)
    valid = parent_dates.notna() & df[days_col].notna()
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    result[valid] = parent_dates[valid] + pd.to_timedelta(
        df.loc[valid, days_col].astype(float).astype(int), unit="D"
    )
    df[date_col] = result
    return df


def _add_relative_month_date(
    df: pd.DataFrame,
    months_col: str,
    date_col: str,
    app_dates: pd.Series,
) -> pd.DataFrame:
    """
    Compute absolute date using a MONTHS_BALANCE column (0 = application month,
    -k = k months before application).  Uses 30-day month approximation.
    """
    df = df.copy()
    parent_dates = df[ENTITY_KEY].map(app_dates)
    valid = parent_dates.notna() & df[months_col].notna()
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    result[valid] = parent_dates[valid] + pd.to_timedelta(
        (df.loc[valid, months_col].astype(float) * 30).astype(int), unit="D"
    )
    df[date_col] = result
    return df
