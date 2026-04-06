import unittest

import numpy as np
import torch

from graphbench_adapter import extract_edge_samples, extract_node_samples


class DummyData:
    def __init__(self, x=None, y=None, edge_index=None, edge_attr=None):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = x.shape[0] if x is not None else 0


class DummyDataset:
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def get(self, idx):
        return self._items[idx]


class GraphBenchAdapterTests(unittest.TestCase):
    def test_extract_node_samples_raw_columns(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = torch.tensor([0, 1, 0])
        dataset = DummyDataset([DummyData(x=x, y=y)])
        X, y_out = extract_node_samples(dataset, max_node_features=2, tabular_mode="raw")

        self.assertListEqual(list(X.columns), ["graph_id", "node_id", "x0", "x1"])
        self.assertEqual(len(X), 3)
        self.assertEqual(len(y_out), 3)

    def test_extract_node_samples_raw_drops_mismatched_features(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([0, 1, 0])
        dataset = DummyDataset([DummyData(x=x, y=y)])
        X, _ = extract_node_samples(dataset, tabular_mode="raw")

        self.assertListEqual(list(X.columns), ["graph_id", "node_id"])
        self.assertEqual(len(X), 3)

    def test_extract_edge_samples_raw_columns(self):
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_attr = torch.tensor([[0.1], [0.2]])
        y = torch.tensor([1, 0])
        dataset = DummyDataset([DummyData(y=y, edge_index=edge_index, edge_attr=edge_attr)])
        X, y_out = extract_edge_samples(dataset, tabular_mode="raw")

        self.assertTrue(all(col in X.columns for col in ["graph_id", "src_id", "dst_id", "e0"]))
        self.assertEqual(len(X), 2)
        self.assertTrue(np.array_equal(y_out.to_numpy(), np.array([1, 0])))


if __name__ == "__main__":
    unittest.main()
