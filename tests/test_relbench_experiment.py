import unittest

import numpy as np
import pandas as pd

from relbench_experiment import align_to_columns, build_link_training_pairs, sample_candidate_ids


class RelBenchExperimentTests(unittest.TestCase):
    def test_build_link_training_pairs_samples_valid_negatives(self):
        query_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "user_id": [1, 2],
                "item_id": [np.array([10, 11]), np.array([12])],
            }
        )
        X, y = build_link_training_pairs(
            query_df=query_df,
            dst_col="item_id",
            candidate_pool=[10, 11, 12, 13, 14],
            n_neg_per_pos=2,
            seed=7,
        )

        positives = 3
        self.assertEqual(len(X), positives * 3)
        self.assertEqual(len(y), positives * 3)
        self.assertEqual(int((y == 1).sum()), positives)
        self.assertEqual(int((y == 0).sum()), positives * 2)

        pos_lookup = {
            (pd.Timestamp("2020-01-01"), 1): {10, 11},
            (pd.Timestamp("2020-01-02"), 2): {12},
        }
        for row, label in zip(X.itertuples(index=False), y.to_numpy()):
            if label != 0:
                continue
            key = (row.date, row.user_id)
            self.assertNotIn(int(row.item_id), pos_lookup[key])

    def test_sample_candidate_ids_keeps_positives(self):
        rng = np.random.default_rng(42)
        cand = sample_candidate_ids(
            base_pool=[1, 2, 3, 4, 5, 6],
            positive_ids=[2, 5],
            max_candidates=4,
            eval_k=3,
            rng=rng,
        )
        self.assertGreaterEqual(len(cand), 3)
        self.assertIn(2, set(cand.tolist()))
        self.assertIn(5, set(cand.tolist()))

    def test_align_to_columns_adds_missing(self):
        df = pd.DataFrame({"b": [1, 2], "c": [3, 4]})
        aligned = align_to_columns(df, ["a", "b"])
        self.assertListEqual(aligned.columns.tolist(), ["a", "b"])
        self.assertTrue(aligned["a"].isna().all())
        self.assertListEqual(aligned["b"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
