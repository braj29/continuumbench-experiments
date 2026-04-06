import argparse
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from graphbench_experiment import _classification_metrics, _validate_graph_feature_policy


class GraphBenchExperimentTests(unittest.TestCase):
    def test_graph_feature_guard_blocks_stats(self):
        args = argparse.Namespace(
            allow_graph_features=False,
            tabular_mode="stats",
            add_degree_feature=False,
        )
        with self.assertRaises(ValueError):
            _validate_graph_feature_policy(args)

    def test_graph_feature_guard_blocks_degree(self):
        args = argparse.Namespace(
            allow_graph_features=False,
            tabular_mode="raw",
            add_degree_feature=True,
        )
        with self.assertRaises(ValueError):
            _validate_graph_feature_policy(args)

    def test_graph_feature_guard_allows_raw(self):
        args = argparse.Namespace(
            allow_graph_features=False,
            tabular_mode="raw",
            add_degree_feature=False,
        )
        _validate_graph_feature_policy(args)

    def test_classification_metrics_use_stable_positive_class(self):
        class DummyModel:
            classes_ = np.array([0, 1])

            def predict(self, X):
                return np.array([1, 1, 0, 1])

            def predict_proba(self, X):
                return np.array(
                    [
                        [0.1, 0.9],
                        [0.2, 0.8],
                        [0.8, 0.2],
                        [0.3, 0.7],
                    ]
                )

        y = pd.Series([1, 1, 1, 0])
        X = pd.DataFrame({"x": [0, 1, 2, 3]})

        metrics = _classification_metrics(DummyModel(), X, y)

        expected_pr_auc = average_precision_score(
            (y == 1).astype(int),
            np.array([0.9, 0.8, 0.2, 0.7]),
        )
        self.assertAlmostEqual(metrics["pr_auc"], expected_pr_auc, places=8)


if __name__ == "__main__":
    unittest.main()
