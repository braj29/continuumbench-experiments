import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

from continuumbench_experiments.continuumbench.harness import (
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
    compute_macro_ris,
    compute_ris,
    make_synthetic_relational_problem,
    stub_graph_fit_fn,
    stub_graph_predict_fn,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)
from continuumbench_experiments.continuumbench.sources import load_dataset_entity_problem


class ContinuumBenchHarnessTests(unittest.TestCase):
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

    def test_load_dataset_entity_problem_smoke(self):
        try:
            db, task, split = load_dataset_entity_problem(
                "rel-f1",
                "driver-top3",
                download=True,
            )
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Dataset registry unavailable in this environment: {exc}")
        self.assertEqual(task.target_col, "qualifying")
        self.assertTrue(split.train_mask.any())
        self.assertTrue(split.test_mask.any())
        self.assertIn("drivers", db.tables)

    def test_latest_rows_as_of_handles_interleaved_rows_and_preserves_order(self):
        left_df = pd.DataFrame(
            {
                "user_id": [1, 2, 1, 2],
                "seed_time": pd.to_datetime(
                    ["2024-03-20", "2024-02-10", "2024-02-10", "2024-03-20"]
                ),
                "label": [1, 0, 0, 1],
            }
        )
        right_df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "updated_at": pd.to_datetime(
                    ["2024-01-01", "2024-03-01", "2024-01-15", "2024-02-20"]
                ),
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
                    JoinedTableConfig(
                        view_name="jt_temporalagg",
                        max_path_hops=1,
                        lookback_days=30,
                    ),
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


try:
    from torch_geometric.data import HeteroData as _HeteroData  # noqa: F401

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from continuumbench_experiments.models.graph_builder import (
    build_hetero_batch,
    build_hetero_instance,
    schema_edge_types,
    schema_node_types,
)
from continuumbench_experiments.models.adapter_graph_nn import (
    GraphSAGEAdapter,
    RGCNAdapter,
)


@unittest.skipUnless(_HAS_PYG, "torch-geometric not installed")
class GraphBuilderTests(unittest.TestCase):
    def setUp(self):
        self.db, self.task, self.split = make_synthetic_relational_problem()

    def test_schema_helpers_return_expected_types(self):
        node_types = schema_node_types(self.db, self.task)
        edge_types = schema_edge_types(self.db, self.task)
        self.assertIn(self.task.entity_table, node_types)
        self.assertTrue(all(isinstance(et, tuple) and len(et) == 3 for et in edge_types))

    def test_build_hetero_instance_has_entity_node(self):
        import pandas as pd
        train_df = self.task.task_table.loc[self.split.train_mask].reset_index(drop=True)
        row = train_df.iloc[0]
        data = build_hetero_instance(
            self.db,
            self.task,
            entity_id=row[self.task.entity_key],
            seed_time=pd.Timestamp(row[self.task.seed_time_col]),
        )
        # Entity table must always be present with exactly 1 node
        self.assertEqual(data[self.task.entity_table].x.shape[0], 1)

    def test_build_hetero_batch_length_matches_rows(self):
        train_df = self.task.task_table.loc[self.split.train_mask].reset_index(drop=True)
        sample = train_df.head(4)
        batch = build_hetero_batch(self.db, self.task, sample)
        # After batching 4 instances each with 1 entity node → 4 entity nodes total
        self.assertEqual(batch[self.task.entity_table].x.shape[0], 4)

    def test_neighborhood_cap_k(self):
        import pandas as pd
        train_df = self.task.task_table.loc[self.split.train_mask].reset_index(drop=True)
        row = train_df.iloc[0]
        # K=1 should cap event tables at 1 row
        data_k1 = build_hetero_instance(
            self.db, self.task,
            entity_id=row[self.task.entity_key],
            seed_time=pd.Timestamp(row[self.task.seed_time_col]),
            k=1,
        )
        for tbl_name in schema_node_types(self.db, self.task):
            if tbl_name != self.task.entity_table:
                self.assertLessEqual(data_k1[tbl_name].x.shape[0], 1)


@unittest.skipUnless(_HAS_PYG, "torch-geometric not installed")
class GNNAdapterSmokeTests(unittest.TestCase):
    def test_rgcn_fit_predict_synthetic(self):
        db, task, split = make_synthetic_relational_problem()
        payload = {"db": db, "task": task, "graph_config": {}}
        train_df = task.task_table.loc[split.train_mask].reset_index(drop=True)
        val_df = task.task_table.loc[split.val_mask].reset_index(drop=True)
        test_df = task.task_table.loc[split.test_mask].reset_index(drop=True)

        train_data = {"payload": payload, "task_df": train_df}
        val_data = {"payload": payload, "task_df": val_df}
        test_data = {"payload": payload, "task_df": test_df}

        adapter = RGCNAdapter(name="rgcn-test", hidden_dim=16, num_layers=1,
                              max_epochs=5, patience=5)
        meta = adapter.fit(train_data, val_data, task)
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["conv_type"], "gcn")

        preds = adapter.predict(test_data, task)
        self.assertEqual(preds.shape, (len(test_df),))
        # Classification probs must be in [0, 1]
        self.assertTrue((preds >= 0).all() and (preds <= 1).all())

    def test_graphsage_fit_predict_synthetic(self):
        db, task, split = make_synthetic_relational_problem()
        payload = {"db": db, "task": task, "graph_config": {}}
        train_df = task.task_table.loc[split.train_mask].reset_index(drop=True)
        test_df = task.task_table.loc[split.test_mask].reset_index(drop=True)

        adapter = GraphSAGEAdapter(name="sage-test", hidden_dim=16, num_layers=1,
                                   max_epochs=3, patience=3)
        adapter.fit({"payload": payload, "task_df": train_df}, None, task)
        preds = adapter.predict({"payload": payload, "task_df": test_df}, task)
        self.assertEqual(preds.shape, (len(test_df),))


class RisFormulaTests(unittest.TestCase):
    """Verify the paper's RIS formula: RIS = 1 - σ(normalized utilities)."""

    def test_ris_perfect_invariance_all_equal(self):
        # All views achieve the same AUROC → RIS = 1.0
        scores = {"jt_entity": 0.7, "jt_temporalagg": 0.7, "graphified": 0.7}
        result = compute_ris(scores, "auroc")
        self.assertAlmostEqual(result["ris"], 1.0, places=6)
        self.assertAlmostEqual(result["utility_std"], 0.0, places=6)

    def test_ris_paper_example(self):
        # From the paper: u = [-0.10, 0.60, 0.56], RIS ≈ 0.65
        scores = {
            "jt_entity": 0.469,
            "jt_temporalagg": 0.682,
            "graphified": 0.670,
        }
        result = compute_ris(scores, "auroc")
        # oracle = 0.682, chance = 0.5
        # u_entity = (0.469 - 0.5) / (0.682 - 0.5) ≈ -0.170
        # u_temporalagg = (0.682 - 0.5) / (0.682 - 0.5) = 1.0
        # u_graphified = (0.670 - 0.5) / (0.682 - 0.5) ≈ 0.934
        # std ≈ 0.487; RIS ≈ 0.513
        self.assertLess(result["ris"], 0.9, "Low-invariance task should have RIS < 0.9")
        self.assertGreater(result["ris"], 0.0)

    def test_ris_auroc_oracle_normalization(self):
        # For AUROC, oracle = max(scores); utility of best view should be 1.0
        scores = {"a": 0.5, "b": 0.8}
        result = compute_ris(scores, "auroc")
        # u_a = (0.5 - 0.5) / (0.8 - 0.5) = 0.0
        # u_b = (0.8 - 0.5) / (0.8 - 0.5) = 1.0
        # std = std([0.0, 1.0]) = 0.5; RIS = 0.5
        self.assertAlmostEqual(result["utility_min"], 0.0, places=6)
        self.assertAlmostEqual(result["utility_max"], 1.0, places=6)
        self.assertAlmostEqual(result["ris"], 0.5, places=6)

    def test_ris_mae_lower_is_better(self):
        # Better MAE (lower) should map to higher utility
        scores = {"a": 10.0, "b": 2.0}
        result = compute_ris(scores, "mae")
        # chance = 10.0 (worst), oracle = 2.0 (best)
        # u_a = (10 - 10) / (10 - 2) = 0.0
        # u_b = (10 - 2) / (10 - 2) = 1.0
        self.assertAlmostEqual(result["utility_min"], 0.0, places=6)
        self.assertAlmostEqual(result["utility_max"], 1.0, places=6)

    def test_ris_single_view(self):
        # Single representation → std = 0 → RIS = 1.0
        scores = {"only": 0.75}
        result = compute_ris(scores, "auroc")
        self.assertAlmostEqual(result["ris"], 1.0, places=6)

    def test_compute_macro_ris(self):
        per_task = {
            "task_a": {"ris": 0.8},
            "task_b": {"ris": 0.6},
        }
        result = compute_macro_ris(per_task)
        self.assertAlmostEqual(result["macro_ris"], 0.7, places=6)
        self.assertEqual(result["n_tasks"], 2)

    def test_compute_macro_ris_empty(self):
        result = compute_macro_ris({})
        self.assertEqual(result["n_tasks"], 0)
        self.assertTrue(np.isnan(result["macro_ris"]))


if __name__ == "__main__":
    unittest.main()
