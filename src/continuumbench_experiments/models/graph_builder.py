"""Build PyTorch Geometric HeteroData subgraphs from a relational DatabaseSpec.

Each prediction instance (entity_id, seed_time) maps to one HeteroData graph:
  - One node for the target entity (features from entity table row)
  - Temporally-safe event nodes per directly FK-linked table (rows with time ≤ seed_time)
  - Directed edges: entity → event  and  event → entity (reverse)

All features are numeric (object/datetime columns are dropped; NaN → 0).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from ..continuumbench.specs import DatabaseSpec, TaskSpec

try:
    from torch_geometric.data import Batch, HeteroData

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def _require_pyg() -> None:
    if not _HAS_PYG:
        raise ImportError(
            "torch-geometric is required for GNN-based graph adapters.\n"
            "Install it with:  pip install torch-geometric\n"
            "or add torch-geometric to your environment."
        )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _numeric_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Return a float32 tensor of shape (n_rows, n_numeric_cols).

    String/object/datetime columns are dropped; NaN is filled with 0.
    If no numeric columns exist, returns a single zero-column tensor so every
    node always has at least one feature.
    """
    num = df.select_dtypes(include=[np.number, bool]).fillna(0.0)
    if num.shape[1] == 0:
        return torch.zeros(len(df), 1, dtype=torch.float32)
    return torch.tensor(num.values.astype(np.float32))


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _event_tables(db: DatabaseSpec, entity_table: str) -> list[tuple[str, str]]:
    """Return [(table_name, fk_col)] for tables with a FK directly to entity_table."""
    results = []
    for name, tbl in db.tables.items():
        if name == entity_table:
            continue
        for fk_col, parent in tbl.foreign_keys.items():
            if parent == entity_table:
                results.append((name, fk_col))
    return results


def schema_node_types(db: DatabaseSpec, task: TaskSpec) -> list[str]:
    """All node types that can appear in a subgraph (entity + FK-linked event tables)."""
    return [task.entity_table] + [n for n, _ in _event_tables(db, task.entity_table)]


def schema_edge_types(db: DatabaseSpec, task: TaskSpec) -> list[tuple[str, str, str]]:
    """All edge types (src_type, relation, dst_type) including reverse edges."""
    types: list[tuple[str, str, str]] = []
    for tbl_name, _ in _event_tables(db, task.entity_table):
        types.append((task.entity_table, f"has_{tbl_name}", tbl_name))
        types.append((tbl_name, f"in_{task.entity_table}", task.entity_table))
    return types


# ---------------------------------------------------------------------------
# Per-instance subgraph builder
# ---------------------------------------------------------------------------


def build_hetero_instance(
    db: DatabaseSpec,
    task: TaskSpec,
    entity_id: Any,
    seed_time: pd.Timestamp,
    k: Optional[int] = None,
) -> "HeteroData":
    """Build one temporally-safe HeteroData for a single prediction instance.

    Parameters
    ----------
    db : DatabaseSpec
    task : TaskSpec
    entity_id : value of ``task.entity_key`` for this instance
    seed_time : only rows with timestamp ≤ seed_time are included
    k : if given, keep at most *k* most-recent events per event table
    """
    _require_pyg()
    data = HeteroData()

    # ── Target entity node (always exactly 1) ─────────────────────────────
    ent_tbl = db.tables[task.entity_table]
    ent_rows = ent_tbl.df[ent_tbl.df[task.entity_key] == entity_id].head(1)
    if len(ent_rows) == 0:
        # Entity not found — produce a zero-feature placeholder node
        ent_rows = pd.DataFrame(
            np.zeros((1, len(ent_tbl.df.columns))), columns=ent_tbl.df.columns
        )
    data[task.entity_table].x = _numeric_tensor(ent_rows)  # (1, d_ent)

    # ── Event nodes + entity↔event edges ───────────────────────────────────
    for tbl_name, fk_col in _event_tables(db, task.entity_table):
        tbl = db.tables[tbl_name]
        rows = tbl.df[tbl.df[fk_col] == entity_id].copy()

        # Temporal safety: drop rows after seed_time
        if tbl.time_col and len(rows):
            t_series = pd.to_datetime(rows[tbl.time_col], utc=False)
            rows = rows[t_series <= seed_time]

        # Neighbourhood cap: keep at most k most-recent events
        if k is not None and len(rows) > k:
            rows = rows.tail(k) if tbl.time_col else rows.iloc[:k]

        n = len(rows)
        # Always register the node type so the batch schema is consistent
        data[tbl_name].x = (
            _numeric_tensor(rows) if n > 0 else torch.zeros(0, 1, dtype=torch.float32)
        )

        if n > 0:
            src = torch.zeros(n, dtype=torch.long)  # entity is always node index 0
            dst = torch.arange(n, dtype=torch.long)
            data[task.entity_table, f"has_{tbl_name}", tbl_name].edge_index = torch.stack(
                [src, dst]
            )
            data[tbl_name, f"in_{task.entity_table}", task.entity_table].edge_index = torch.stack(
                [dst, src]
            )

    return data


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def build_hetero_batch(
    db: DatabaseSpec,
    task: TaskSpec,
    rows: pd.DataFrame,
    k: Optional[int] = None,
) -> "Batch":
    """Build a PyG Batch of HeteroData, one subgraph per prediction instance in *rows*.

    The batch can be passed directly to a HeteroConv model.  Each mini-graph
    has exactly one entity node (index 0 within its graph), so the GNN output
    for the entity node type has shape (len(rows), hidden_dim).
    """
    _require_pyg()
    graphs = [
        build_hetero_instance(
            db,
            task,
            entity_id=row[task.entity_key],
            seed_time=pd.Timestamp(row[task.seed_time_col]),
            k=k,
        )
        for _, row in rows.iterrows()
    ]
    return Batch.from_data_list(graphs)
