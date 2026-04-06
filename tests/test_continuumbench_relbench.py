import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

from continuumbench_relbench import (
    DatabaseSpec,
    ExperimentConfig,
    ExperimentRunner,
    ExternalGraphAdapter,
    ExternalRelationalAdapter,
    JoinedTableBuilder,
    JoinedTableConfig,
    MeanDummyAdapter,
    TableSpec,
    TaskSpec,
    _dense_feature_matrix,
    build_graph_degree_feature_table,
    load_relbench_entity_problem,
    make_synthetic_relational_problem,
    stub_graph_fit_fn,
    stub_graph_predict_fn,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)


class ContinuumBenchRelBenchTests(unittest.TestCase):
    def test_dense_feature_matrix_from_sparse(self):
        X = sp_sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0]])
        d = _dense_feature_matrix(X)
        self.assertIsInstance(d, np.ndarray)
        self.assertEqual(d.shape, (2, 2))
        np.testing.assert_allclose(d, [[0.0, 1.0], [2.0, 0.0]])

    def test_build_graph_degree_feature_table_synthetic(self):
        db, task, split = make_synthetic_relational_problem()
        train = task.task_table.loc[split.train_mask].reset_index(drop=True)
        frame = build_graph_degree_feature_table(db, task, train)
        graph_cols = [c for c in frame.columns if c.startswith("graphdeg__")]
        self.assertTrue(graph_cols)
        self.assertIn("graphdeg__transactions__count_leq_seed", graph_cols)

    def test_temporal_agg_zero_history_counts_are_zero(self):
        users = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "updated_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            }
        )
        transactions = pd.DataFrame(
            {
                "tx_id": [1, 2],
                "user_id": [1, 3],
                "amount": [10.0, 99.0],
                "event_time": pd.to_datetime(["2024-01-15", "2024-03-05"]),
            }
        )
        task_table = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "seed_time": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-01"]),
                "label": [0, 1, 0],
            }
        )
        db = DatabaseSpec(
            tables={
                "users": TableSpec(
                    name="users",
                    df=users,
                    primary_key="user_id",
                    time_col="updated_at",
                    foreign_keys={},
                ),
                "transactions": TableSpec(
                    name="transactions",
                    df=transactions,
                    primary_key="tx_id",
                    time_col="event_time",
                    foreign_keys={"user_id": "users"},
                ),
            }
        )
        task = TaskSpec(
            name="dummy",
            task_type="classification",
            target_col="label",
            metric_name="auroc",
            entity_table="users",
            entity_key="user_id",
            seed_time_col="seed_time",
            task_table=task_table,
        )

        frame = JoinedTableBuilder(db=db, task=task).build(
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30)
        )

        self.assertListEqual(frame["transactions__row_count"].tolist(), [1, 0, 0])
        self.assertFalse(pd.isna(frame["transactions__recency_days"].iloc[0]))
        self.assertTrue(pd.isna(frame["transactions__recency_days"].iloc[1]))
        self.assertTrue(pd.isna(frame["transactions__recency_days"].iloc[2]))

    def test_load_relbench_entity_problem_smoke(self):
        try:
            db, task, split = load_relbench_entity_problem(
                "rel-f1",
                "driver-top3",
                download=True,
            )
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"RelBench unavailable in this environment: {exc}")
        self.assertEqual(task.target_col, "qualifying")
        self.assertTrue(split.train_mask.any())
        self.assertTrue(split.test_mask.any())
        self.assertIn("drivers", db.tables)

    def test_latest_rows_as_of_handles_interleaved_rows_and_preserves_order(self):
        left_df = pd.DataFrame(
            {
                "user_id": [1, 2, 1, 2],
                "seed_time": pd.to_datetime(["2024-03-20", "2024-02-10", "2024-02-10", "2024-03-20"]),
                "label": [1, 0, 0, 1],
            }
        )
        right_df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "updated_at": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-01-15", "2024-02-20"]),
                "age": [10, 11, 20, 21],
            }
        )

        task = TaskSpec(
            name="dummy",
            task_type="classification",
            target_col="label",
            metric_name="auroc",
            entity_table="users",
            entity_key="user_id",
            seed_time_col="seed_time",
            task_table=left_df,
        )
        db = DatabaseSpec(
            tables={
                "users": TableSpec(
                    name="users",
                    df=right_df,
                    primary_key="user_id",
                    time_col="updated_at",
                    foreign_keys={},
                )
            }
        )
        builder = JoinedTableBuilder(db=db, task=task)

        out = builder._latest_rows_as_of(
            left_df=left_df,
            right_df=right_df,
            left_on="user_id",
            right_on="user_id",
            seed_time_col="seed_time",
            right_time_col="updated_at",
            prefix="users__",
        )

        self.assertEqual(len(out), len(left_df))
        self.assertListEqual(out["users__age"].tolist(), [11, 20, 10, 21])

    def test_protocol_a_and_b_smoke_with_dummy_adapters(self):
        db, task, split = make_synthetic_relational_problem()
        joined_models = [MeanDummyAdapter(name="dummy-jt", view_name="joined-table")]
        relational_models = [
            ExternalRelationalAdapter(
                name="rt-stub",
                fit_fn=stub_relational_fit_fn,
                predict_fn=stub_relational_predict_fn,
            )
        ]
        graph_models = [
            ExternalGraphAdapter(
                name="relgt-stub",
                fit_fn=stub_graph_fit_fn,
                predict_fn=stub_graph_predict_fn,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                ExperimentConfig(
                    benchmark_name="continuumbench-v1",
                    output_dir=tmpdir,
                )
            )
            summary_a = runner.run_protocol_a(
                db=db,
                task=task,
                split=split,
                joined_table_cfgs=[
                    JoinedTableConfig(view_name="jt_entity", max_path_hops=1),
                    JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
                ],
                joined_models=joined_models,
                relational_models=relational_models,
                graph_models=graph_models,
            )
            summary_b = runner.run_protocol_b(
                db=db,
                task=task,
                split=split,
                joined_ablation_cfgs=[
                    JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=7),
                ],
                joined_models=joined_models,
                graph_k_values=[100],
                graph_models_factory=lambda k: [
                    ExternalGraphAdapter(
                        name="relgt-stub",
                        fit_fn=stub_graph_fit_fn,
                        predict_fn=stub_graph_predict_fn,
                        neighborhood_size_k=k,
                    )
                ],
            )

            self.assertEqual(len(summary_a.per_run), 4)
            self.assertEqual(len(summary_b.per_run), 2)
            self.assertEqual(
                len(summary_a.per_task_ris["synthetic-user-churn"]["representation_scores"]),
                4,
            )
            self.assertEqual(
                len(summary_b.per_task_ris["synthetic-user-churn"]["representation_scores"]),
                2,
            )

            expected = [
                "synthetic-user-churn__protocol_a__runs.parquet",
                "synthetic-user-churn__protocol_a__ris.json",
                "synthetic-user-churn__protocol_b__runs.parquet",
                "synthetic-user-churn__protocol_b__ris.json",
            ]
            for rel_path in expected:
                self.assertTrue((Path(tmpdir) / rel_path).is_file(), rel_path)

    def test_protocol_a_accepts_precomputed_metric_override(self):
        class MetricOnlyRelationalModel:
            @property
            def name(self):
                return "rt-official"

            @property
            def view_name(self):
                return "relational"

            def fit(self, train_data, val_data, task):
                return {"precomputed_test_metric": 0.77, "backend": "test"}

            def predict(self, data, task):
                raise AssertionError("predict() must not be called when metric override is present")

        db, task, split = make_synthetic_relational_problem()
        joined_models = [MeanDummyAdapter(name="dummy-jt", view_name="joined-table")]
        relational_models = [MetricOnlyRelationalModel()]
        graph_models = [
            ExternalGraphAdapter(
                name="relgt-stub",
                fit_fn=stub_graph_fit_fn,
                predict_fn=stub_graph_predict_fn,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                ExperimentConfig(
                    benchmark_name="continuumbench-v1",
                    output_dir=tmpdir,
                )
            )
            summary_a = runner.run_protocol_a(
                db=db,
                task=task,
                split=split,
                joined_table_cfgs=[JoinedTableConfig(view_name="jt_entity", max_path_hops=1)],
                joined_models=joined_models,
                relational_models=relational_models,
                graph_models=graph_models,
            )

            rows = summary_a.to_frame()
            rt_rows = rows[rows["model_name"] == "rt-official"]
            self.assertEqual(len(rt_rows), 1)
            self.assertAlmostEqual(float(rt_rows["metric_value"].iloc[0]), 0.77, places=8)


if __name__ == "__main__":
    unittest.main()
