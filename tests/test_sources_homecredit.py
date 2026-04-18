"""
Unit tests for the HomeCredit multi-table loader.

Generates minimal in-memory CSV files that mirror the Kaggle schema and
verifies that load_homecredit_default() produces correct structures.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from continuumbench_experiments.continuumbench.sources_homecredit import (
    ENTITY_KEY,
    SEED_TIME_COL,
    TARGET_COL,
    TASK_NAME,
    load_homecredit_default,
)


def _write_minimal_csvs(data_dir: Path) -> None:
    """Write the smallest valid set of Kaggle-style CSVs for testing."""
    n_app = 20  # enough for a 70/15/15 split

    ids = list(range(100001, 100001 + n_app))

    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": ([0] * 14 + [1] * 6),
            "AMT_CREDIT": np.random.rand(n_app) * 100_000,
            "AMT_INCOME_TOTAL": np.random.rand(n_app) * 50_000,
            "DAYS_BIRTH": np.random.randint(-20000, -6000, n_app),
            "DAYS_EMPLOYED": np.random.randint(-5000, 0, n_app),
            "NAME_CONTRACT_TYPE": ["Cash loans"] * n_app,
        }
    )
    app.to_csv(data_dir / "application_train.csv", index=False)

    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": ids[:10] * 2,  # two bureau records per first 10 apps
            "SK_ID_BUREAU": range(200001, 200021),
            "DAYS_CREDIT": np.random.randint(-1500, -30, 20),
            "CREDIT_ACTIVE": ["Active"] * 20,
            "AMT_CREDIT_SUM": np.random.rand(20) * 50_000,
        }
    )
    bureau.to_csv(data_dir / "bureau.csv", index=False)

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": bureau["SK_ID_BUREAU"].tolist() * 3,
            "MONTHS_BALANCE": list(range(-2, 1)) * 20,
            "STATUS": ["C"] * 60,
        }
    )
    bureau_balance.to_csv(data_dir / "bureau_balance.csv", index=False)

    prev_app = pd.DataFrame(
        {
            "SK_ID_CURR": ids[:8] * 2,
            "SK_ID_PREV": range(300001, 300017),
            "DAYS_DECISION": np.random.randint(-1000, -10, 16),
            "NAME_CONTRACT_TYPE": ["Consumer loans"] * 16,
            "AMT_APPLICATION": np.random.rand(16) * 30_000,
        }
    )
    prev_app.to_csv(data_dir / "previous_application.csv", index=False)

    pos_cash = pd.DataFrame(
        {
            "SK_ID_CURR": prev_app["SK_ID_CURR"].tolist() * 2,
            "SK_ID_PREV": prev_app["SK_ID_PREV"].tolist() * 2,
            "MONTHS_BALANCE": list(range(-1, 1)) * 16,
            "CNT_INSTALMENT": np.random.rand(32) * 24,
            "NAME_CONTRACT_STATUS": ["Active"] * 32,
            "SK_DPD": [0] * 32,
        }
    )
    pos_cash.to_csv(data_dir / "POS_CASH_balance.csv", index=False)

    cc = pd.DataFrame(
        {
            "SK_ID_CURR": prev_app["SK_ID_CURR"].tolist(),
            "SK_ID_PREV": prev_app["SK_ID_PREV"].tolist(),
            "MONTHS_BALANCE": [0] * 16,
            "AMT_BALANCE": np.random.rand(16) * 10_000,
            "SK_DPD": [0] * 16,
        }
    )
    cc.to_csv(data_dir / "credit_card_balance.csv", index=False)

    inst = pd.DataFrame(
        {
            "SK_ID_CURR": prev_app["SK_ID_CURR"].tolist() * 3,
            "SK_ID_PREV": prev_app["SK_ID_PREV"].tolist() * 3,
            "NUM_INSTALMENT_VERSION": [1] * 48,
            "NUM_INSTALMENT_NUMBER": list(range(1, 4)) * 16,
            "DAYS_INSTALMENT": np.random.randint(-900, -10, 48),
            "DAYS_ENTRY_PAYMENT": np.random.randint(-900, -10, 48),
            "AMT_INSTALMENT": np.random.rand(48) * 5_000,
            "AMT_PAYMENT": np.random.rand(48) * 5_000,
        }
    )
    inst.to_csv(data_dir / "installments_payments.csv", index=False)


class HomeCreditLoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        _write_minimal_csvs(self.data_dir)
        self.db, self.task, self.split = load_homecredit_default(self.data_dir)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_task_name_and_type(self):
        self.assertEqual(self.task.name, TASK_NAME)
        self.assertEqual(self.task.task_type, "classification")
        self.assertEqual(self.task.metric_name, "auroc")

    def test_entity_key_and_seed_time_col(self):
        self.assertEqual(self.task.entity_table, "application")
        self.assertEqual(self.task.entity_key, ENTITY_KEY)
        self.assertEqual(self.task.seed_time_col, SEED_TIME_COL)

    def test_task_table_columns(self):
        cols = set(self.task.task_table.columns)
        self.assertIn(ENTITY_KEY, cols)
        self.assertIn(SEED_TIME_COL, cols)
        self.assertIn(TARGET_COL, cols)

    def test_target_not_in_entity_features(self):
        entity_df = self.db.tables["application"].df
        self.assertNotIn(TARGET_COL, entity_df.columns)

    def test_database_has_all_seven_tables(self):
        expected = {
            "application",
            "bureau",
            "bureau_balance",
            "previous_application",
            "pos_cash_balance",
            "credit_card_balance",
            "installments_payments",
        }
        self.assertEqual(set(self.db.tables.keys()), expected)

    def test_bureau_has_time_col(self):
        bureau = self.db.tables["bureau"]
        self.assertEqual(bureau.time_col, "BUREAU_DATE")
        self.assertIn("BUREAU_DATE", bureau.df.columns)
        valid = bureau.df["BUREAU_DATE"].dropna()
        self.assertGreater(len(valid), 0)

    def test_previous_application_has_time_col(self):
        prev = self.db.tables["previous_application"]
        self.assertEqual(prev.time_col, "PREV_APP_DATE")
        self.assertIn("PREV_APP_DATE", prev.df.columns)

    def test_bureau_balance_is_static(self):
        self.assertIsNone(self.db.tables["bureau_balance"].time_col)

    def test_pos_cash_balance_has_time_col(self):
        pos = self.db.tables["pos_cash_balance"]
        self.assertEqual(pos.time_col, "POS_CASH_DATE")
        self.assertIn("POS_CASH_DATE", pos.df.columns)

    def test_foreign_keys(self):
        tables = self.db.tables
        self.assertEqual(tables["bureau"].foreign_keys, {ENTITY_KEY: "application"})
        self.assertEqual(tables["bureau_balance"].foreign_keys, {"SK_ID_BUREAU": "bureau"})
        self.assertEqual(
            tables["previous_application"].foreign_keys, {ENTITY_KEY: "application"}
        )
        self.assertIn(ENTITY_KEY, tables["pos_cash_balance"].foreign_keys)
        self.assertIn("SK_ID_PREV", tables["pos_cash_balance"].foreign_keys)

    def test_split_covers_all_rows(self):
        n = len(self.task.task_table)
        total = (
            self.split.train_mask.sum()
            + self.split.val_mask.sum()
            + self.split.test_mask.sum()
        )
        self.assertEqual(total, n)

    def test_split_partitions_are_disjoint(self):
        overlap = (
            (self.split.train_mask & self.split.val_mask).any()
            or (self.split.train_mask & self.split.test_mask).any()
            or (self.split.val_mask & self.split.test_mask).any()
        )
        self.assertFalse(overlap)

    def test_synthetic_dates_are_ordered_with_sk_id_curr(self):
        task_df = self.task.task_table.copy()
        task_df = task_df.sort_values(ENTITY_KEY)
        dates = pd.to_datetime(task_df[SEED_TIME_COL])
        self.assertTrue(dates.is_monotonic_increasing)

    def test_bureau_dates_before_application_dates(self):
        bureau_df = self.db.tables["bureau"].df
        app_df = self.db.tables["application"].df
        merged = bureau_df.merge(
            app_df[[ENTITY_KEY, SEED_TIME_COL]], on=ENTITY_KEY, how="left"
        )
        valid = merged["BUREAU_DATE"].notna() & merged[SEED_TIME_COL].notna()
        bureau_dates = pd.to_datetime(merged.loc[valid, "BUREAU_DATE"])
        app_dates = pd.to_datetime(merged.loc[valid, SEED_TIME_COL])
        self.assertTrue((bureau_dates <= app_dates).all())

    def test_missing_files_raises_file_not_found(self):
        import os
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                load_homecredit_default(empty_dir)


if __name__ == "__main__":
    unittest.main()
