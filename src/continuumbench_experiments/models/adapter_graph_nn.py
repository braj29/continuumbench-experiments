"""GNN adapters for the ContinuumBench graphified view.

In-process models (require ``torch-geometric``):
  RGCNAdapter       — Relational GCN using GCNConv per edge type (Schlichtkrull et al., 2018)
  GraphSAGEAdapter  — Heterogeneous GraphSAGE using SAGEConv per edge type (Hamilton et al., 2017)

Subprocess wrappers (require an external repo clone):
  RelGTSubprocessAdapter  — Relational Graph Transformer (Dwivedi et al., arXiv:2505.10960, 2025)
  ULTRASubprocessAdapter  — ULTRA foundation model (Zhu et al., ICLR 2024)
                            Note: ULTRA is designed for KG link prediction and requires
                            task adaptation for entity property prediction.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..continuumbench.specs import DatabaseSpec, TaskSpec
from .adapter_common import BaseViewModel
from .graph_builder import (
    _require_pyg,
    build_hetero_batch,
    schema_edge_types,
    schema_node_types,
)

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

_GNN_VIEW_NAME = "graphified"

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch GNN model
# ─────────────────────────────────────────────────────────────────────────────


class _HeteroGNNModel(nn.Module):
    """Heterogeneous GNN: per-node-type input encoders + stacked HeteroConv + MLP head.

    The forward pass returns logits for the *target_node_type* only.
    After batching with ``build_hetero_batch``, every instance contributes exactly
    one target node, so the output shape is ``(batch_size, out_dim)``.
    """

    def __init__(
        self,
        in_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        target_node_type: str,
        conv_type: str = "sage",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.target_node_type = target_node_type

        # Per-node-type linear projection from raw features → hidden_dim
        self.encoders = nn.ModuleDict(
            {nt: nn.Linear(dim, hidden_dim) for nt, dim in in_dims.items()}
        )

        # GNN message-passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gcn":
                conv_dict = {
                    et: GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
                    for et in edge_types
                }
            else:  # "sage"
                conv_dict = {
                    et: SAGEConv(hidden_dim, hidden_dim, normalize=True)
                    for et in edge_types
                }
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        # Encode inputs
        h: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.encoders and x.shape[0] > 0:
                h[nt] = F.relu(self.encoders[nt](x))
            else:
                h[nt] = x

        # Message passing with residual connections
        for conv in self.convs:
            new_h = conv(h, edge_index_dict)
            for nt, new_val in new_h.items():
                activated = F.relu(new_val)
                # Residual if shape matches (same hidden_dim throughout)
                if nt in h and h[nt].shape == activated.shape:
                    h[nt] = activated + h[nt]
                else:
                    h[nt] = activated

        return self.head(self.dropout(h[self.target_node_type]))


# ─────────────────────────────────────────────────────────────────────────────
# Base in-process GNN adapter (shared training loop)
# ─────────────────────────────────────────────────────────────────────────────


class _HeteroGNNAdapter(BaseViewModel):
    """Train a heterogeneous GNN on temporally-safe per-instance subgraphs.

    Subclasses set ``conv_type`` ("sage" or "gcn") to select the convolution.
    Everything else — graph construction, the training loop, early stopping,
    class-balance weighting — is shared.
    """

    conv_type: str = "sage"

    def __init__(
        self,
        name: str,
        hidden_dim: int = 128,
        num_layers: int = 2,
        max_epochs: int = 200,
        patience: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        dropout: float = 0.3,
        neighborhood_k: Optional[int] = None,
        device: str = "auto",
    ):
        self._name = name
        self._view_name = _GNN_VIEW_NAME
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout = dropout
        self.neighborhood_k = neighborhood_k
        self._device_spec = device

        self.model: Optional[_HeteroGNNModel] = None
        self._device: Optional[torch.device] = None
        self._in_dims: Optional[dict[str, int]] = None
        self._edge_types: Optional[list] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        spec = self._device_spec.lower()
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    def _discover_schema(
        self, db: DatabaseSpec, task: TaskSpec, sample_rows: pd.DataFrame
    ) -> None:
        """Build a small sample batch to infer feature dims and edge type list."""
        sample = sample_rows.head(min(8, len(sample_rows)))
        batch = build_hetero_batch(db, task, sample, k=self.neighborhood_k)

        self._in_dims = {}
        for nt in schema_node_types(db, task):
            node_store = batch[nt]
            if hasattr(node_store, "x") and node_store.x is not None and node_store.x.shape[0] > 0:
                self._in_dims[nt] = int(node_store.x.shape[-1])

        self._edge_types = schema_edge_types(db, task)

    def _make_batch(
        self, rows: pd.DataFrame, db: DatabaseSpec, task: TaskSpec
    ) -> Any:
        b = build_hetero_batch(db, task, rows, k=self.neighborhood_k)
        return b.to(self._device)

    # ── Training helpers ───────────────────────────────────────────────────

    def _train_epoch(
        self,
        train_df: pd.DataFrame,
        db: DatabaseSpec,
        task: TaskSpec,
        y_train: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        pos_weight: Optional[torch.Tensor],
    ) -> float:
        self.model.train()
        total_loss, n_total = 0.0, len(train_df)
        perm = torch.randperm(n_total)

        for start in range(0, n_total, self.batch_size):
            idx = perm[start : start + self.batch_size]
            b_rows = train_df.iloc[idx.numpy()].reset_index(drop=True)
            y_b = y_train[idx].to(self._device)

            optimizer.zero_grad()
            out = self._make_batch(b_rows, db, task)
            logits = self.model(out.x_dict, out.edge_index_dict).squeeze(-1)

            if task.task_type == "classification":
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_b.float(), pos_weight=pos_weight
                )
            else:
                loss = F.mse_loss(logits, y_b.float())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(b_rows)

        return total_loss / n_total

    @torch.no_grad()
    def _infer(
        self, rows: pd.DataFrame, db: DatabaseSpec, task: TaskSpec
    ) -> torch.Tensor:
        self.model.eval()
        parts: list[torch.Tensor] = []
        for start in range(0, len(rows), self.batch_size):
            b_rows = rows.iloc[start : start + self.batch_size].reset_index(drop=True)
            out = self._make_batch(b_rows, db, task)
            parts.append(self.model(out.x_dict, out.edge_index_dict).squeeze(-1).cpu())
        return torch.cat(parts)

    # ── Public interface ───────────────────────────────────────────────────

    def fit(
        self, train_data: Any, val_data: Optional[Any], task: TaskSpec
    ) -> dict[str, Any]:
        _require_pyg()
        self._device = self._resolve_device()

        payload = train_data["payload"]
        db: DatabaseSpec = payload["db"]
        train_df: pd.DataFrame = train_data["task_df"]
        val_df: Optional[pd.DataFrame] = (
            val_data["task_df"] if val_data is not None else None
        )

        self._discover_schema(db, task, train_df)

        self.model = _HeteroGNNModel(
            in_dims=self._in_dims,
            edge_types=self._edge_types,
            hidden_dim=self.hidden_dim,
            out_dim=1,
            num_layers=self.num_layers,
            target_node_type=task.entity_table,
            conv_type=self.conv_type,
            dropout=self.dropout,
        ).to(self._device)

        y_train = torch.tensor(train_df[task.target_col].values, dtype=torch.float32)

        # Class-balance weighting for binary classification
        pos_weight: Optional[torch.Tensor] = None
        if task.task_type == "classification":
            n_pos = float(y_train.sum())
            n_neg = float(len(y_train)) - n_pos
            if n_pos > 0 and n_neg > 0:
                pos_weight = torch.tensor([n_neg / n_pos], device=self._device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, self.patience // 2), factor=0.5
        )

        best_val_loss = float("inf")
        patience_ctr = 0
        best_state: Optional[dict] = None
        epochs_done = 0

        for epoch in range(1, self.max_epochs + 1):
            epochs_done = epoch
            train_loss = self._train_epoch(
                train_df, db, task, y_train, optimizer, pos_weight
            )

            if val_df is not None and len(val_df) > 0:
                val_logits = self._infer(val_df, db, task)
                y_val = torch.tensor(val_df[task.target_col].values, dtype=torch.float32)
                if task.task_type == "classification":
                    val_loss = F.binary_cross_entropy_with_logits(
                        val_logits, y_val
                    ).item()
                else:
                    val_loss = F.mse_loss(val_logits, y_val).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        logger.info(
                            "%s: early stop at epoch %d (val_loss=%.4f)",
                            self._name,
                            epoch,
                            best_val_loss,
                        )
                        break

            if epoch % 20 == 0:
                logger.info("%s: epoch %d train_loss=%.4f", self._name, epoch, train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "status": "ok",
            "conv_type": self.conv_type,
            "epochs_trained": epochs_done,
            "best_val_loss": round(best_val_loss, 6),
            "neighborhood_k": self.neighborhood_k,
            "device": str(self._device),
        }

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self._name} must be fit before predict.")
        _require_pyg()

        db: DatabaseSpec = data["payload"]["db"]
        rows: pd.DataFrame = data["task_df"]
        logits = self._infer(rows, db, task)

        if task.task_type == "classification":
            return torch.sigmoid(logits).numpy().astype(np.float64)
        return logits.numpy().astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete in-process GNN models
# ─────────────────────────────────────────────────────────────────────────────


class RGCNAdapter(_HeteroGNNAdapter):
    """Relational GCN (RGCN) for the graphified view.

    One GCNConv per edge type inside a HeteroConv layer (Schlichtkrull et al., 2018).
    Classic baseline for heterogeneous graphs.
    """

    conv_type = "gcn"

    def __init__(
        self,
        name: str = "rgcn",
        hidden_dim: int = 128,
        num_layers: int = 2,
        **kwargs: Any,
    ):
        super().__init__(name=name, hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)


class GraphSAGEAdapter(_HeteroGNNAdapter):
    """Heterogeneous GraphSAGE for the graphified view.

    One SAGEConv per edge type inside a HeteroConv layer (Hamilton et al., 2017).
    Inductive; handles nodes with no neighbours gracefully.
    """

    conv_type = "sage"

    def __init__(
        self,
        name: str = "graphsage",
        hidden_dim: int = 128,
        num_layers: int = 2,
        **kwargs: Any,
    ):
        super().__init__(name=name, hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# RelGT subprocess adapter
# ─────────────────────────────────────────────────────────────────────────────


class RelGTSubprocessAdapter(BaseViewModel):
    """Integration adapter for the Relational Graph Transformer (Dwivedi et al., 2025).

    Spawns a subprocess running the official RelGT codebase and scrapes the
    final test metric from logs — the same pattern as
    ``OfficialRelationalTransformerAdapter`` for the relational view.

    Requirements
    ------------
    - A clone of the RelGT repository (``--relgt-repo-path`` or ``relgt_repo_path``)
    - Linux + CUDA (RelGT is untested on CPU-only or macOS)

    The adapter passes run configuration via the ``RELGT_CONFIG`` environment
    variable (JSON dict) so it is compatible with any entry-point that reads it.
    Adapt the entry-point wrapper as needed for the specific RelGT release.

    Reference: V. P. Dwivedi et al., "Relational Graph Transformer",
    arXiv:2505.10960, 2025.
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        target_col: str,
        relgt_repo_path: str,
        name: str = "relgt-official",
        python_executable: Optional[str] = None,
        seed: int = 0,
        num_neighbors: int = 512,
        num_layers: int = 3,
        hidden_channels: int = 128,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 512,
        num_workers: int = 4,
    ):
        self._name = name
        self._view_name = _GNN_VIEW_NAME
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.target_col = target_col
        self.relgt_repo_path = str(relgt_repo_path)
        self.python_executable = python_executable or sys.executable
        self.seed = seed
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def _validate_runtime(self) -> None:
        if platform.system() != "Linux":
            raise RuntimeError(
                "RelGT requires a Linux + CUDA environment. "
                "Run on Snellius or another Linux GPU machine."
            )
        probe = subprocess.run(
            [
                self.python_executable,
                "-c",
                "import torch; assert torch.cuda.is_available()",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"RelGT runtime probe failed — CUDA not available in target python.\n"
                f"{probe.stderr[-500:]}"
            )

    def _locate_entry_point(self, repo_path: Path) -> Path:
        for candidate in ("main.py", "src/main.py", "relgt/main.py", "run.py"):
            ep = repo_path / candidate
            if ep.is_file():
                return ep
        raise FileNotFoundError(
            f"RelGT entry-point not found in '{repo_path}'. "
            "Expected main.py, src/main.py, or run.py. "
            "Check that the repo is cloned and point --relgt-repo-path correctly."
        )

    def _extract_test_metric(self, logs: str) -> Optional[float]:
        numeric = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        patterns = [
            rf"test(?:_(?:auc|auroc|mae|r2))?[:\s]+{numeric}",
            rf"(?:auc|auroc|mae|r2)/test[:\s]+{numeric}",
        ]
        for pat in patterns:
            matches = re.findall(pat, logs, re.IGNORECASE)
            if matches:
                return float(matches[-1])
        return None

    def fit(
        self, train_data: Any, val_data: Optional[Any], task: TaskSpec
    ) -> dict[str, Any]:
        repo_path = Path(self.relgt_repo_path).expanduser().resolve()
        entry_point = self._locate_entry_point(repo_path)
        self._validate_runtime()

        config = {
            "dataset": self.dataset_name,
            "task": self.task_name,
            "target_col": self.target_col,
            "seed": self.seed,
            "num_neighbors": self.num_neighbors,
            "num_layers": self.num_layers,
            "hidden_channels": self.hidden_channels,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

        env = os.environ.copy()
        env["RELGT_CONFIG"] = json.dumps(config)

        result = subprocess.run(
            [self.python_executable, str(entry_point)],
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        logs = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            tail = "\n".join(logs.splitlines()[-80:])
            raise RuntimeError(
                f"RelGT run failed (rc={result.returncode}).\n--- tail ---\n{tail}"
            )

        metric = self._extract_test_metric(logs)
        if metric is None:
            tail = "\n".join(logs.splitlines()[-80:])
            raise RuntimeError(
                "RelGT run succeeded but test metric was not found in logs.\n"
                f"--- tail ---\n{tail}"
            )

        return {
            "backend": "relgt",
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "precomputed_test_metric": float(metric),
        }

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise RuntimeError(
            "RelGTSubprocessAdapter does not emit row-wise predictions. "
            "The test metric is returned via precomputed_test_metric in fit()."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ULTRA subprocess adapter
# ─────────────────────────────────────────────────────────────────────────────


class ULTRASubprocessAdapter(BaseViewModel):
    """Integration adapter for ULTRA (Zhu et al., ICLR 2024).

    ULTRA is a *foundation model for knowledge-graph link prediction* that
    generalises zero-shot across graph schemas.  For ContinuumBench tasks
    (entity property prediction, not link prediction) ULTRA requires adaptation:

    The simplest reformulation treats the binary label as a link:
      (entity, ``has_label_1``, label_node) vs (entity, ``has_label_0``, label_node)
    and frames the task as predicting which relation holds — but this changes
    the evaluation setup.  Regression tasks (MAE) are not directly supported.

    We provide this adapter as infrastructure for researchers who wish to
    carry out that adaptation.  The adapter passes run config via the
    ``ULTRA_CONFIG`` environment variable; adapt the ULTRA runner to consume it.

    Requirements
    ------------
    - Clone from https://github.com/DeepGraphLearning/ULTRA
    - Adapt the runner for entity classification (see note above)
    - Pass ``--ultra-repo-path`` pointing to the clone

    Reference: M. Zhu et al., "Towards Foundation Models for Knowledge Graph
    Reasoning", ICLR 2024.
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        target_col: str,
        ultra_repo_path: str,
        name: str = "ultra",
        python_executable: Optional[str] = None,
        config_path: Optional[str] = None,
        seed: int = 0,
        gpus: int = 1,
        epochs: int = 10,
    ):
        self._name = name
        self._view_name = _GNN_VIEW_NAME
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.target_col = target_col
        self.ultra_repo_path = str(ultra_repo_path)
        self.python_executable = python_executable or sys.executable
        self.config_path = config_path
        self.seed = seed
        self.gpus = gpus
        self.epochs = epochs

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def _locate_entry_point(self, repo_path: Path) -> Path:
        for candidate in ("run.py", "script/run.py", "ultra/run.py", "src/run.py"):
            ep = repo_path / candidate
            if ep.is_file():
                return ep
        raise FileNotFoundError(
            f"ULTRA entry-point not found in '{repo_path}'. "
            "Expected run.py or script/run.py. "
            "Ensure you have cloned https://github.com/DeepGraphLearning/ULTRA "
            "and adapted it for entity property prediction."
        )

    def _extract_test_metric(self, logs: str) -> Optional[float]:
        numeric = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        patterns = [
            rf"test_(?:auc|auroc|mae|mrr|hits@\d+)[:\s]+{numeric}",
            rf"(?:test|Test)\s+(?:AUROC|AUC|MAE|MRR)[:\s]+{numeric}",
        ]
        for pat in patterns:
            matches = re.findall(pat, logs, re.IGNORECASE)
            if matches:
                return float(matches[-1])
        return None

    def fit(
        self, train_data: Any, val_data: Optional[Any], task: TaskSpec
    ) -> dict[str, Any]:
        repo_path = Path(self.ultra_repo_path).expanduser().resolve()
        entry_point = self._locate_entry_point(repo_path)

        config: dict[str, Any] = {
            "dataset": self.dataset_name,
            "task": self.task_name,
            "target_col": self.target_col,
            "seed": self.seed,
            "gpus": self.gpus,
            "epochs": self.epochs,
        }
        if self.config_path:
            config["config"] = self.config_path

        env = os.environ.copy()
        env["ULTRA_CONFIG"] = json.dumps(config)

        result = subprocess.run(
            [self.python_executable, str(entry_point)],
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        logs = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            tail = "\n".join(logs.splitlines()[-80:])
            raise RuntimeError(
                f"ULTRA run failed (rc={result.returncode}).\n--- tail ---\n{tail}"
            )

        metric = self._extract_test_metric(logs)
        if metric is None:
            tail = "\n".join(logs.splitlines()[-80:])
            raise RuntimeError(
                "ULTRA run succeeded but test metric was not found in logs.\n"
                f"--- tail ---\n{tail}"
            )

        return {
            "backend": "ultra",
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "precomputed_test_metric": float(metric),
        }

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise RuntimeError(
            "ULTRASubprocessAdapter does not emit row-wise predictions. "
            "The test metric is returned via precomputed_test_metric in fit()."
        )
