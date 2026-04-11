import argparse
import unittest
from unittest import mock

from continuumbench_experiments.cli import continuumbench


class ContinuumBenchCliTests(unittest.TestCase):
    def test_parse_models_deduplicates(self):
        models = continuumbench._parse_models("tabicl, tabpfn, tabicl")
        self.assertEqual(models, ["tabicl", "tabpfn"])

    def test_parse_models_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            continuumbench._parse_models("tabicl,unknown")

    def test_task_source_metadata_for_synthetic_source(self):
        args = argparse.Namespace(
            task_source="synthetic",
            dataset_name="ignored",
            task_name="ignored",
        )
        self.assertEqual(
            continuumbench._task_source_metadata(args),
            {"task_source": "synthetic", "dataset_name": None, "task_name": None},
        )

    def test_default_protocol_configs_are_readable_and_stable(self):
        configs = continuumbench._default_protocol_configs()
        self.assertEqual(
            [cfg.view_name for cfg in configs.joined_table_cfgs],
            ["jt_entity", "jt_temporalagg"],
        )
        self.assertEqual(
            [cfg.lookback_days for cfg in configs.joined_ablation_cfgs],
            [7, 30, None],
        )
        self.assertEqual(configs.graph_k_values, (100, 300, 500))

    def test_tabicl_runtime_device_prefers_env_override(self):
        args = argparse.Namespace(tabicl_device="cpu")
        with mock.patch.dict("os.environ", {"TABICL_DEVICE": "cuda"}, clear=False):
            self.assertEqual(continuumbench._tabicl_runtime_device(args), "cuda")


if __name__ == "__main__":
    unittest.main()
