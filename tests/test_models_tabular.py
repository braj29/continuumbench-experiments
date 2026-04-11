import unittest

from continuumbench_experiments.models.tabular import (
    TABPFN_MODEL_VERSION,
    build_tabpfn,
    build_tabpfn_regressor,
)


class TabularModelBuilderTests(unittest.TestCase):
    def test_tabpfn_classifier_uses_explicit_v2_checkpoint(self):
        model = build_tabpfn(device="cpu")
        self.assertEqual(TABPFN_MODEL_VERSION.value, "v2")
        self.assertIn("tabpfn-v2-", str(model.model_path))
        self.assertNotIn("v2.5", str(model.model_path))
        self.assertNotIn("v2.6", str(model.model_path))

    def test_tabpfn_regressor_uses_explicit_v2_checkpoint(self):
        model = build_tabpfn_regressor(device="cpu")
        self.assertEqual(TABPFN_MODEL_VERSION.value, "v2")
        self.assertIn("tabpfn-v2-", str(model.model_path))
        self.assertNotIn("v2.5", str(model.model_path))
        self.assertNotIn("v2.6", str(model.model_path))


if __name__ == "__main__":
    unittest.main()
