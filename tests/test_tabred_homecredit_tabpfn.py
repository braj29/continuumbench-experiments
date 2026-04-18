from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuumbench_experiments.tabred_homecredit_tabpfn import (
    _fit_and_eval,
    _list_available_splits,
)


def _write_mock_tabred_dataset(root: Path) -> None:
    n = 12
    np.save(root / "X_num.npy", np.arange(n * 2, dtype=np.float32).reshape(n, 2))
    np.save(root / "X_bin.npy", (np.arange(n) % 2).astype(np.float32).reshape(n, 1))
    np.save(root / "X_cat.npy", (np.arange(n) % 3).astype(np.int64).reshape(n, 1))
    np.save(root / "X_meta.npy", np.column_stack([np.arange(n), np.arange(n) + 100]).astype(np.int64))
    # Keep both classes in train/val/test to make AUROC defined.
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    np.save(root / "Y.npy", y)

    split_dir = root / "split-default"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "train_idx.npy", np.array([0, 1, 2, 3, 4, 5], dtype=np.int64))
    np.save(split_dir / "val_idx.npy", np.array([6, 7, 8], dtype=np.int64))
    np.save(split_dir / "test_idx.npy", np.array([9, 10, 11], dtype=np.int64))


class TabRedHomeCreditTabPFNTests(unittest.TestCase):
    def test_list_splits_and_dry_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_mock_tabred_dataset(root)

            self.assertEqual(_list_available_splits(root), ["default"])

            result = _fit_and_eval(
                data_dir=root,
                split="default",
                include_meta=False,
                device="cpu",
                ignore_pretraining_limits=True,
                max_train_rows=None,
                seed=7,
                dry_run=True,
            )
            self.assertTrue(result["dry_run"])
            # 2 num + 1 bin + 1 cat
            self.assertEqual(result["n_features"], 4)
            self.assertEqual(result["n_rows"]["train"], 6)
            self.assertEqual(result["n_rows"]["val"], 3)
            self.assertEqual(result["n_rows"]["test"], 3)

    def test_include_meta_adds_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_mock_tabred_dataset(root)

            result_no_meta = _fit_and_eval(
                data_dir=root,
                split="default",
                include_meta=False,
                device="cpu",
                ignore_pretraining_limits=True,
                max_train_rows=None,
                seed=7,
                dry_run=True,
            )
            result_with_meta = _fit_and_eval(
                data_dir=root,
                split="default",
                include_meta=True,
                device="cpu",
                ignore_pretraining_limits=True,
                max_train_rows=None,
                seed=7,
                dry_run=True,
            )
            self.assertEqual(result_no_meta["n_features"], 4)
            self.assertEqual(result_with_meta["n_features"], 6)


if __name__ == "__main__":
    unittest.main()
