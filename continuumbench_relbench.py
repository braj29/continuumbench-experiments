from __future__ import annotations

"""
ContinuumBench v1 experiment harness for RelBench-style temporal relational tasks.

What this file gives you:
- A leakage-safe temporal feature builder for joined-table views:
  * JT-Entity
  * JT-TemporalAgg
- A unified experiment contract for three views:
  * joined-table
  * relational
  * graphified
- A runner for Protocol A (head-to-head tri-view benchmark)
- A runner for Protocol B (controlled invariance ablations)
- Compute logging utilities
- Representation Invariance Score (RIS) computation
- Pluggable model adapters for LightGBM / tabular models / RT / RelGT

What this file does NOT do by itself:
- It does not bundle RelBench, RT, or RelGT code.
- The relational and graphified adapters are intentionally thin interfaces that you can
  wire to the official implementations you choose to use.

Extras in this repo:
- ``load_relbench_entity_problem`` maps a RelBench **entity** task into this harness so
  joined / graph-degree / optional official RT refer to the same prediction problem.
- ``GraphifiedSklearnAdapter`` + ``build_graph_degree_feature_table`` provide a
  dependency-light graph view (time-respecting incident counts on FK-linked tables).

Design goals:
- Keep the task-table + seed-time semantics fixed.
- Ensure all joined-table features are computed "as of" each seed time.
- Make representation differences explicit and reproducible.

This module is written to be usable as a starting point for a real benchmark artifact.
"""

import abc
import argparse
import dataclasses
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

LOGGER = logging.getLogger("continuumbench")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)


@contextmanager
def compute_tracker() -> Iterable[MutableMapping[str, float]]:
    """
    Lightweight wall-clock + CPU memory tracker.

    GPU metrics are left as optional hooks because the exact implementation
    depends on your runtime stack (PyTorch/CUDA, nvidia-smi, etc.).
    """
    tracemalloc.start()
    start = time.perf_counter()
    payload: MutableMapping[str, float] = {}
    try:
        yield payload
    finally:
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        payload["wall_clock_sec"] = end - start
        payload["peak_cpu_ram_mb"] = peak / (1024 * 1024)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------


@dataclass
class TableSpec:
    name: str
    df: pd.DataFrame
    primary_key: str
    time_col: Optional[str] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    # Example: {"user_id": "users", "item_id": "items"}


@dataclass
class TaskSpec:
    name: str
    task_type: str  # "classification" or "regression"
    target_col: str
    metric_name: str  # "auroc" or "mae"
    entity_table: str
    entity_key: str
    seed_time_col: str
    task_table: pd.DataFrame


@dataclass
class DatabaseSpec:
    tables: Dict[str, TableSpec]

    def table(self, name: str) -> TableSpec:
        return self.tables[name]


@dataclass
class TemporalSplit:
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series


@dataclass
class JoinedTableConfig:
    view_name: str  # "jt_entity" or "jt_temporalagg"
    lookback_days: Optional[int] = None
    max_path_hops: int = 1
    include_parent_lookups: bool = True
    aggregation_functions: Tuple[str, ...] = ("count", "nunique", "mean", "sum", "min", "max")
    recency_features: bool = True
    drop_high_cardinality_threshold: int = 500


@dataclass
class ExperimentConfig:
    benchmark_name: str
    seed: int = 7
    small_medium_track: bool = False
    output_dir: str = "outputs"
    ris_baseline_utility: float = 0.0


# -----------------------------------------------------------------------------
# Metric and RIS utilities
# -----------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
) -> float:
    if metric_name == "auroc":
        return float(roc_auc_score(y_true, y_pred))
    if metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric_name}")


def metric_to_utility(score: float, metric_name: str, eps: float = 1e-12) -> float:
    """
    Convert task metrics to a higher-is-better utility.

    AUROC: identity
    MAE: -log(MAE)

    You can swap this for a relative-to-baseline mapping later,
    but keep it fixed once the benchmark spec is published.
    """
    if metric_name == "auroc":
        return score
    if metric_name == "mae":
        return -math.log(max(score, eps))
    raise ValueError(f"Unsupported metric: {metric_name}")


def compute_ris(metric_scores: Mapping[str, float], metric_name: str) -> Dict[str, float]:
    """
    Representation Invariance Score.

    Uses 1 / (1 + stddev(utility)) so that the range is (0, 1],
    where larger means more invariant across representations.
    """
    utilities = np.array(
        [metric_to_utility(v, metric_name) for v in metric_scores.values()], dtype=float
    )
    std = float(np.std(utilities))
    ris = 1.0 / (1.0 + std)
    return {
        "ris": ris,
        "utility_std": std,
        "utility_min": float(np.min(utilities)),
        "utility_max": float(np.max(utilities)),
        "utility_gap": float(np.max(utilities) - np.min(utilities)),
    }


# -----------------------------------------------------------------------------
# Split utilities
# -----------------------------------------------------------------------------


def make_temporal_split(
    task_df: pd.DataFrame,
    seed_time_col: str,
    val_cutoff: pd.Timestamp,
    test_cutoff: pd.Timestamp,
) -> TemporalSplit:
    seed_times = pd.to_datetime(task_df[seed_time_col], utc=False)
    train_mask = seed_times < val_cutoff
    val_mask = (seed_times >= val_cutoff) & (seed_times < test_cutoff)
    test_mask = seed_times >= test_cutoff
    return TemporalSplit(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


# -----------------------------------------------------------------------------
# Leakage-safe joined-table builder
# -----------------------------------------------------------------------------


class JoinedTableBuilder:
    """
    Reference implementation for leakage-safe joined-table construction.

    Assumptions:
    - task_table has one row per prediction instance.
    - each instance is keyed by (entity_key, seed_time).
    - all time-aware joins/aggregates only use rows with timestamp <= seed_time.
    """

    def __init__(self, db: DatabaseSpec, task: TaskSpec):
        self.db = db
        self.task = task

    def build(self, config: JoinedTableConfig) -> pd.DataFrame:
        if config.view_name == "jt_entity":
            return self._build_jt_entity(config)
        if config.view_name == "jt_temporalagg":
            return self._build_jt_temporalagg(config)
        raise ValueError(f"Unknown joined-table view: {config.view_name}")

    def _base_instances(self) -> pd.DataFrame:
        df = self.task.task_table.copy()
        df[self.task.seed_time_col] = pd.to_datetime(df[self.task.seed_time_col], utc=False)
        return df

    def _build_jt_entity(self, config: JoinedTableConfig) -> pd.DataFrame:
        """
        Conservative entity-only denormalization.

        Includes:
        - central entity row as-of seed time (latest row <= seed time if temporal)
        - parent/lookup tables reachable by many-to-one / one-to-one foreign keys
        """
        base = self._base_instances()
        entity_table = self.db.table(self.task.entity_table)

        entity_features = self._latest_rows_as_of(
            left_df=base,
            right_df=entity_table.df,
            left_on=self.task.entity_key,
            right_on=entity_table.primary_key,
            seed_time_col=self.task.seed_time_col,
            right_time_col=entity_table.time_col,
            prefix=f"{entity_table.name}__",
        )

        out = pd.concat(
            [base.reset_index(drop=True), entity_features.reset_index(drop=True)], axis=1
        )

        if config.include_parent_lookups:
            out = self._join_parent_lookups(out, entity_table, self.task.seed_time_col)

        out = self._drop_duplicate_columns(out)
        return out

    def _build_jt_temporalagg(self, config: JoinedTableConfig) -> pd.DataFrame:
        """
        Time-aware aggregated joins.

        Strategy:
        1) Start from JT-Entity.
        2) For every table linked to the entity table within max_path_hops,
           compute leakage-safe aggregation features as-of seed time.
        """
        out = self._build_jt_entity(config)
        reachable = self._reachable_tables(self.task.entity_table, max_hops=config.max_path_hops)

        for table_name, join_path in reachable.items():
            if table_name == self.task.entity_table:
                continue
            table = self.db.table(table_name)
            agg_df = self._aggregate_table_along_path(
                instances=out,
                table=table,
                join_path=join_path,
                seed_time_col=self.task.seed_time_col,
                lookback_days=config.lookback_days,
                agg_funcs=config.aggregation_functions,
                recency_features=config.recency_features,
            )
            if agg_df is not None and len(agg_df.columns) > 0:
                out = pd.concat([out.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)

        out = self._drop_duplicate_columns(out)
        return out

    def _drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _join_parent_lookups(
        self, instances: pd.DataFrame, entity_table: TableSpec, seed_time_col: str
    ) -> pd.DataFrame:
        out = instances.copy()
        for fk_col, parent_table_name in entity_table.foreign_keys.items():
            if fk_col not in out.columns:
                continue
            parent_table = self.db.table(parent_table_name)
            parent_features = self._latest_rows_as_of(
                left_df=out,
                right_df=parent_table.df,
                left_on=fk_col,
                right_on=parent_table.primary_key,
                seed_time_col=seed_time_col,
                right_time_col=parent_table.time_col,
                prefix=f"{parent_table.name}__",
            )
            out = pd.concat(
                [out.reset_index(drop=True), parent_features.reset_index(drop=True)], axis=1
            )
        return out

    def _latest_rows_as_of(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_on: str,
        right_on: str,
        seed_time_col: str,
        right_time_col: Optional[str],
        prefix: str,
    ) -> pd.DataFrame:
        """
        Leakage-safe as-of join.

        For each left row, find the latest matching right row with timestamp <= seed_time.
        If right_time_col is None, treat the right table as static and do a normal join.
        """
        left = left_df[[left_on, seed_time_col]].copy()
        left["__row_id__"] = np.arange(len(left_df))
        left[seed_time_col] = pd.to_datetime(left[seed_time_col], utc=False)
        right = right_df.copy()

        if right_time_col is None:
            merged = left.merge(right, how="left", left_on=left_on, right_on=right_on)
            merged = merged.sort_values("__row_id__").reset_index(drop=True)
            merged = merged[
                [
                    c
                    for c in merged.columns
                    if c not in {left_on, seed_time_col, right_on, "__row_id__"}
                ]
            ]
            merged.columns = [f"{prefix}{c}" for c in merged.columns]
            return merged

        right[right_time_col] = pd.to_datetime(right[right_time_col], utc=False)
        # pandas.merge_asof requires frames to be sorted by the as-of key first.
        left_sorted = left.sort_values([seed_time_col, left_on, "__row_id__"]).reset_index(
            drop=True
        )
        right_sorted = right.sort_values([right_time_col, right_on]).reset_index(drop=True)

        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on=seed_time_col,
            right_on=right_time_col,
            by=None,
            left_by=left_on,
            right_by=right_on,
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.sort_values("__row_id__").reset_index(drop=True)
        merged = merged[
            [
                c
                for c in merged.columns
                if c not in {left_on, seed_time_col, right_on, right_time_col, "__row_id__"}
            ]
        ]
        merged.columns = [f"{prefix}{c}" for c in merged.columns]
        return merged

    def _reachable_tables(
        self, root_table_name: str, max_hops: int
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Build simple forward/backward schema paths from the root entity table.

        Each path element is a tuple:
            (src_table, src_col, dst_table)

        Notes:
        - This is a pragmatic schema traversal, not a full query optimizer.
        - It supports both tables referencing the root and root referencing parent lookups.
        """
        adjacency: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for t in self.db.tables.values():
            for fk_col, parent in t.foreign_keys.items():
                adjacency[t.name].append((t.name, fk_col, parent))
                adjacency[parent].append((parent, self.db.table(parent).primary_key, t.name))

        visited = {root_table_name}
        frontier: List[Tuple[str, List[Tuple[str, str, str]]]] = [(root_table_name, [])]
        paths: Dict[str, List[Tuple[str, str, str]]] = {root_table_name: []}

        for _ in range(max_hops):
            next_frontier = []
            for node, path in frontier:
                for edge in adjacency.get(node, []):
                    _, _, dst = edge
                    if dst in visited:
                        continue
                    visited.add(dst)
                    new_path = path + [edge]
                    paths[dst] = new_path
                    next_frontier.append((dst, new_path))
            frontier = next_frontier
        return paths

    def _aggregate_table_along_path(
        self,
        instances: pd.DataFrame,
        table: TableSpec,
        join_path: List[Tuple[str, str, str]],
        seed_time_col: str,
        lookback_days: Optional[int],
        agg_funcs: Sequence[str],
        recency_features: bool,
    ) -> Optional[pd.DataFrame]:
        """
        Aggregate rows from `table` per prediction instance.

        For v1, this implementation supports the most common and practical case:
        the table directly references the central entity via a foreign key, or is a
        parent lookup reachable in one hop. That already covers many RelBench-style
        temporal event tables.

        Multi-hop aggregation beyond this can be added, but it is intentionally not
        silently faked here.
        """
        entity_key = self.task.entity_key
        table_df = table.df.copy()
        table_df_columns = set(table_df.columns)

        # Identify direct relationship to central entity.
        direct_fk_to_entity = None
        for fk_col, parent in table.foreign_keys.items():
            if parent == self.task.entity_table:
                direct_fk_to_entity = fk_col
                break

        if direct_fk_to_entity is None:
            # Skip unsupported path shapes instead of pretending to aggregate them.
            LOGGER.info(
                "Skipping table '%s' for temporal aggregation: no direct FK to entity table.",
                table.name,
            )
            return None

        # Avoid pandas merge suffixes when task seed_time_col matches table.time_col (e.g. RelBench
        # "date" on both task rows and event tables).
        seed_alias = "__instance_seed_time__"
        base = instances[[entity_key, seed_time_col]].copy()
        base = base.rename(columns={seed_time_col: seed_alias})
        base[seed_alias] = pd.to_datetime(base[seed_alias], utc=False)

        t_df = table_df.copy()
        match_col = "__matched_child_row__"
        while match_col in t_df.columns or match_col in base.columns:
            match_col = f"_{match_col}"
        t_df[match_col] = 1
        if table.time_col and table.time_col in table_df_columns:
            t_df[table.time_col] = pd.to_datetime(t_df[table.time_col], utc=False)

        joined = (
            base.reset_index()
            .rename(columns={"index": "__instance_id__"})
            .merge(
                t_df,
                how="left",
                left_on=entity_key,
                right_on=direct_fk_to_entity,
            )
        )

        if table.time_col and table.time_col in joined.columns:
            joined = joined[
                joined[table.time_col].le(joined[seed_alias]) | joined[table.time_col].isna()
            ].copy()
            if lookback_days is not None:
                lower = joined[seed_alias] - pd.to_timedelta(lookback_days, unit="D")
                joined = joined[
                    joined[table.time_col].ge(lower) | joined[table.time_col].isna()
                ].copy()

        exclude_cols = {
            "__instance_id__",
            entity_key,
            seed_alias,
            seed_time_col,
            direct_fk_to_entity,
            match_col,
        }
        numeric_cols = [
            c
            for c in joined.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(joined[c])
        ]
        categorical_cols = [
            c
            for c in joined.columns
            if c not in exclude_cols
            and not pd.api.types.is_numeric_dtype(joined[c])
            and c != table.time_col
        ]

        parts = []
        zero_fill_cols: List[str] = []

        group = joined.groupby("__instance_id__", dropna=False)
        if "count" in agg_funcs:
            col_name = f"{table.name}__row_count"
            parts.append(group[match_col].count().rename(col_name).to_frame())
            zero_fill_cols.append(col_name)

        if table.time_col and recency_features:
            last_seen = group[table.time_col].max()
            seed_per_instance = joined.groupby("__instance_id__")[seed_alias].first()
            recency_days = (seed_per_instance - last_seen).dt.total_seconds() / 86400.0
            parts.append(recency_days.rename(f"{table.name}__recency_days").to_frame())

        for col in numeric_cols:
            series_group = group[col]
            agg_map = {}
            if "mean" in agg_funcs:
                agg_map[f"{table.name}__{col}__mean"] = series_group.mean()
            if "sum" in agg_funcs:
                agg_map[f"{table.name}__{col}__sum"] = series_group.sum(min_count=1)
            if "min" in agg_funcs:
                agg_map[f"{table.name}__{col}__min"] = series_group.min()
            if "max" in agg_funcs:
                agg_map[f"{table.name}__{col}__max"] = series_group.max()
            for k, v in agg_map.items():
                parts.append(v.rename(k).to_frame())

        if "nunique" in agg_funcs:
            for col in categorical_cols:
                col_name = f"{table.name}__{col}__nunique"
                parts.append(group[col].nunique(dropna=True).rename(col_name).to_frame())
                zero_fill_cols.append(col_name)

        if not parts:
            return None

        agg_df = pd.concat(parts, axis=1).sort_index()
        agg_df = agg_df.reindex(range(len(instances)))
        for col in zero_fill_cols:
            if col in agg_df.columns:
                agg_df[col] = agg_df[col].fillna(0).astype(np.int64)
        agg_df = agg_df.reset_index(drop=True)
        return agg_df


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------


class TabularPreprocessor:
    def __init__(self, target_col: str, drop_cols: Optional[Sequence[str]] = None):
        self.target_col = target_col
        self.drop_cols = set(drop_cols or [])
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_columns_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        feature_df = self._feature_df(df)
        numeric_cols = [
            c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])
        ]
        categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",
        )
        self.pipeline.fit(feature_df)
        self.feature_columns_ = list(feature_df.columns)
        return self

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        return self.pipeline.transform(self._feature_df(df))

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def _feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = {self.target_col} | self.drop_cols
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


def _maybe_subsample_xy(
    X: Any,
    y: np.ndarray,
    max_rows: Optional[int],
    seed: int,
    label: str,
) -> Tuple[Any, np.ndarray]:
    if max_rows is None:
        return X, y
    n = int(getattr(X, "shape", (0,))[0])
    if n <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    LOGGER.info("Subsample train rows for %s: %d -> %d", label, n, max_rows)
    return X[idx], y[idx]


def _dense_feature_matrix(X: Any) -> np.ndarray:
    """
    TabICL (and some TabPFN paths) require dense ``numpy`` arrays; ``ColumnTransformer``
    with ``OneHotEncoder`` returns a sparse matrix.
    """
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _positive_class_proba_maybe_chunked(
    estimator: BaseEstimator,
    X: np.ndarray,
    chunk_size: Optional[int],
) -> np.ndarray:
    """
    TabICL ``predict_proba`` is very slow on full test matrices; optional row chunks +
    logging avoid a silent multi-minute hang.
    """
    if (
        chunk_size is not None
        and chunk_size > 0
        and len(X) > chunk_size
        and type(estimator).__name__ == "TabICLClassifier"
    ):
        parts: List[np.ndarray] = []
        n = len(X)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            LOGGER.info(
                "TabICL predict_proba: rows %d-%d of %d (chunk_size=%d)",
                start,
                stop,
                n,
                chunk_size,
            )
            parts.append(estimator.predict_proba(X[start:stop])[:, 1])
        return np.concatenate(parts, axis=0)
    if type(estimator).__name__ == "TabICLClassifier" and len(X) > 0:
        LOGGER.info("TabICL predict_proba: %d test rows (single batch)", len(X))
    return estimator.predict_proba(X)[:, 1]


# -----------------------------------------------------------------------------
# Model adapters
# -----------------------------------------------------------------------------


class BaseViewModel(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def view_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise NotImplementedError


class LightGBMTabularAdapter(BaseViewModel):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model: Optional[Any] = None
        self.preprocessor: Optional[TabularPreprocessor] = None
        self._name = "lightgbm"
        self._view_name = "joined-table"

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame], task: TaskSpec
    ) -> Dict[str, Any]:
        if lgb is None:
            raise ImportError("lightgbm is not installed. Install it or swap this adapter.")

        self.preprocessor = TabularPreprocessor(
            target_col=task.target_col,
            drop_cols=[task.seed_time_col],
        )
        X_train = self.preprocessor.fit_transform(train_data)
        y_train = train_data[task.target_col].to_numpy()

        X_val, y_val = None, None
        if val_data is not None:
            X_val = self.preprocessor.transform(val_data)
            y_val = val_data[task.target_col].to_numpy()

        if task.task_type == "classification":
            model = lgb.LGBMClassifier(
                random_state=7,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                **self.params,
            )
        else:
            model = lgb.LGBMRegressor(
                random_state=7,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                **self.params,
            )

        fit_kwargs = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        model.fit(X_train, y_train, **fit_kwargs)
        self.model = model
        return {"status": "ok"}

    def predict(self, data: pd.DataFrame, task: TaskSpec) -> np.ndarray:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fit before predict.")
        X = self.preprocessor.transform(data)
        if task.task_type == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


class SklearnTabularAdapter(BaseViewModel):
    """Generic adapter for already-constructed sklearn estimators."""

    def __init__(
        self,
        estimator: BaseEstimator,
        name: str,
        max_train_rows: Optional[int] = None,
        subsample_seed: int = 0,
        predict_proba_chunk_size: Optional[int] = None,
    ):
        self.estimator = estimator
        self._name = name
        self._view_name = "joined-table"
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.max_train_rows = max_train_rows
        self.subsample_seed = int(subsample_seed)
        self.predict_proba_chunk_size = predict_proba_chunk_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame], task: TaskSpec
    ) -> Dict[str, Any]:
        self.preprocessor = TabularPreprocessor(
            target_col=task.target_col,
            drop_cols=[task.seed_time_col],
        )
        X_train = self.preprocessor.fit_transform(train_data)
        y_train = train_data[task.target_col].to_numpy()
        X_train, y_train = _maybe_subsample_xy(
            X_train,
            y_train,
            max_rows=self.max_train_rows,
            seed=self.subsample_seed,
            label=self._name,
        )
        X_train = _dense_feature_matrix(X_train)
        self.estimator.fit(X_train, y_train)
        return {"status": "ok"}

    def predict(self, data: pd.DataFrame, task: TaskSpec) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Model must be fit before predict.")
        X = _dense_feature_matrix(self.preprocessor.transform(data))
        if task.task_type == "classification" and hasattr(self.estimator, "predict_proba"):
            return _positive_class_proba_maybe_chunked(
                self.estimator,
                X,
                self.predict_proba_chunk_size,
            )
        return self.estimator.predict(X)


def _count_incident_tables(db: DatabaseSpec, task: TaskSpec) -> int:
    n = 0
    for tname, spec in db.tables.items():
        if tname == task.entity_table:
            continue
        for parent in spec.foreign_keys.values():
            if parent == task.entity_table:
                n += 1
                break
    return n


def _static_incident_counts(
    table_df: pd.DataFrame, fk_col: str, queries: pd.DataFrame, entity_col: str
) -> np.ndarray:
    counts = table_df.groupby(fk_col).size()
    mapped = queries[entity_col].map(counts)
    return mapped.fillna(0).astype(np.int64).to_numpy()


def _temporal_incident_counts(
    events: pd.DataFrame,
    fk_col: str,
    time_col: str,
    queries: pd.DataFrame,
    entity_col: str,
    seed_col: str,
) -> np.ndarray:
    events = events[[fk_col, time_col]].dropna().copy()
    events[time_col] = pd.to_datetime(events[time_col], utc=False)
    events = events.sort_values([fk_col, time_col])
    q_times = pd.to_datetime(queries[seed_col], utc=False)
    out = np.zeros(len(queries), dtype=np.int64)
    event_groups: Dict[Any, np.ndarray] = {}
    for ent, grp in events.groupby(fk_col, sort=False):
        event_groups[ent] = grp[time_col].to_numpy(dtype="datetime64[ns]")
    for i in range(len(queries)):
        ent = queries[entity_col].iloc[i]
        t = q_times.iloc[i]
        arr = event_groups.get(ent)
        if arr is None or len(arr) == 0:
            continue
        ts = np.datetime64(t.to_datetime64())
        out[i] = int(np.searchsorted(arr, ts, side="right"))
    return out


def build_graph_degree_feature_table(
    db: DatabaseSpec,
    task: TaskSpec,
    rows: pd.DataFrame,
    max_incident_tables: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a tabular frame: entity id, seed time, target, plus incident edge-count features.
    """
    cols = [task.entity_key, task.seed_time_col, task.target_col]
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise KeyError(f"rows is missing columns required for graph features: {missing}")
    base = rows[cols].copy()
    base[task.seed_time_col] = pd.to_datetime(base[task.seed_time_col], utc=False)

    candidates: List[Tuple[str, TableSpec]] = []
    for tname, spec in sorted(db.tables.items(), key=lambda x: x[0]):
        if tname == task.entity_table:
            continue
        fk_to_entity: Optional[str] = None
        for fk_col, parent in spec.foreign_keys.items():
            if parent == task.entity_table:
                fk_to_entity = fk_col
                break
        if fk_to_entity is None:
            continue
        candidates.append((tname, spec))

    if max_incident_tables is not None:
        candidates = candidates[: int(max_incident_tables)]

    feats: Dict[str, np.ndarray] = {}
    for tname, spec in candidates:
        fk = None
        for fk_col, parent in spec.foreign_keys.items():
            if parent == task.entity_table:
                fk = fk_col
                break
        if fk is None:
            continue
        if spec.time_col and spec.time_col in spec.df.columns:
            col = f"graphdeg__{tname}__count_leq_seed"
            feats[col] = _temporal_incident_counts(
                spec.df,
                fk_col=fk,
                time_col=spec.time_col,
                queries=base,
                entity_col=task.entity_key,
                seed_col=task.seed_time_col,
            )
        else:
            col = f"graphdeg__{tname}__static_count"
            feats[col] = _static_incident_counts(spec.df, fk, base, task.entity_key)

    feat_df = pd.DataFrame(feats, index=base.index)
    return pd.concat([base.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


class GraphifiedSklearnAdapter(BaseViewModel):
    """
    Graph-structured view without external GNN deps: per-row, time-respecting counts of
    incident rows in tables that reference the central entity (plus static link counts).

    Optional Protocol B hook: ``neighborhood_size_k`` in the graph payload caps how many
    incident tables contribute features (sorted by table name): ``max_tables = min(N, K // 100)``.
    """

    _K_TABLE_CAP_DIVISOR = 100

    def __init__(
        self,
        estimator: BaseEstimator,
        name: str,
        max_train_rows: Optional[int] = None,
        subsample_seed: int = 0,
        predict_proba_chunk_size: Optional[int] = None,
    ):
        self.estimator = estimator
        self._name = name
        self._view_name = "graphified"
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.max_train_rows = max_train_rows
        self.subsample_seed = int(subsample_seed)
        self.predict_proba_chunk_size = predict_proba_chunk_size
        self._max_incident_tables: Optional[int] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    @staticmethod
    def _incident_table_cap(payload: Mapping[str, Any], n_candidates: int) -> Optional[int]:
        gc = payload.get("graph_config") or {}
        k = gc.get("neighborhood_size_k")
        if k is None:
            return None
        k_int = int(k)
        return max(1, min(n_candidates, k_int // GraphifiedSklearnAdapter._K_TABLE_CAP_DIVISOR))

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> Dict[str, Any]:
        payload = train_data["payload"]
        db: DatabaseSpec = payload["db"]
        task_df: pd.DataFrame = train_data["task_df"]
        n_cand = _count_incident_tables(db, task)
        self._max_incident_tables = self._incident_table_cap(payload, n_cand)
        frame = build_graph_degree_feature_table(
            db=db,
            task=task,
            rows=task_df,
            max_incident_tables=self._max_incident_tables,
        )
        self.preprocessor = TabularPreprocessor(
            target_col=task.target_col,
            drop_cols=[task.seed_time_col],
        )
        X_train = self.preprocessor.fit_transform(frame)
        y_train = frame[task.target_col].to_numpy()
        X_train, y_train = _maybe_subsample_xy(
            X_train,
            y_train,
            max_rows=self.max_train_rows,
            seed=self.subsample_seed,
            label=self._name,
        )
        X_train = _dense_feature_matrix(X_train)
        self.estimator.fit(X_train, y_train)
        meta: Dict[str, Any] = {"status": "ok", "graphified": "incident_degree_counts"}
        if self._max_incident_tables is not None:
            meta["max_incident_tables"] = self._max_incident_tables
        return meta

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Model must be fit before predict.")
        payload = data["payload"]
        db: DatabaseSpec = payload["db"]
        task_df: pd.DataFrame = data["task_df"]
        frame = build_graph_degree_feature_table(
            db=db,
            task=task,
            rows=task_df,
            max_incident_tables=self._max_incident_tables,
        )
        X = _dense_feature_matrix(self.preprocessor.transform(frame))
        if task.task_type == "classification" and hasattr(self.estimator, "predict_proba"):
            return _positive_class_proba_maybe_chunked(
                self.estimator,
                X,
                self.predict_proba_chunk_size,
            )
        return self.estimator.predict(X)


class ExternalRelationalAdapter(BaseViewModel):
    """
    Thin wrapper for RT or another schema-native model.

    Pass two callables:
    - fit_fn(db, task, split_dict, **kwargs) -> model_obj, fit_metadata
    - predict_fn(model_obj, db, task, task_split_df, **kwargs) -> np.ndarray
    """

    def __init__(
        self,
        name: str,
        fit_fn: Callable[..., Tuple[Any, Dict[str, Any]]],
        predict_fn: Callable[..., np.ndarray],
        **kwargs: Any,
    ):
        self._name = name
        self._view_name = "relational"
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.model_obj: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> Dict[str, Any]:
        self.model_obj, meta = self.fit_fn(
            train_data=train_data, val_data=val_data, task=task, **self.kwargs
        )
        return meta

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        return self.predict_fn(model_obj=self.model_obj, data=data, task=task, **self.kwargs)


class OfficialRelationalTransformerAdapter(BaseViewModel):
    """
    Adapter for the official Relational Transformer repository:
    https://github.com/snap-stanford/relational-transformer

    This adapter runs RT as a subprocess and consumes its reported test metric.
    The official RT codepath does not expose a stable per-row prediction API,
    so this adapter returns `precomputed_test_metric` from `fit()`.
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        target_col: str,
        rt_repo_path: str,
        name: str = "rt-official",
        python_executable: Optional[str] = None,
        columns_to_drop: Optional[Sequence[str]] = None,
        project: str = "continuumbench-rt",
        eval_splits: Sequence[str] = ("val", "test"),
        eval_freq: int = 1000,
        eval_pow2: bool = False,
        max_eval_steps: int = 40,
        load_ckpt_path: Optional[str] = None,
        save_ckpt_dir: Optional[str] = None,
        compile_: bool = False,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 8,
        max_bfs_width: int = 256,
        lr: float = 1e-4,
        wd: float = 0.0,
        lr_schedule: bool = False,
        max_grad_norm: float = 1.0,
        max_steps: int = 2**12 + 1,
        embedding_model: str = "all-MiniLM-L12-v2",
        d_text: int = 384,
        seq_len: int = 1024,
        num_blocks: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
    ):
        self._name = name
        self._view_name = "relational"
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.target_col = target_col
        self.rt_repo_path = str(rt_repo_path)
        self.python_executable = python_executable or sys.executable
        self.columns_to_drop = list(columns_to_drop or [])
        self.project = project
        self.eval_splits = list(eval_splits)
        self.eval_freq = int(eval_freq)
        self.eval_pow2 = bool(eval_pow2)
        self.max_eval_steps = int(max_eval_steps)
        self.load_ckpt_path = load_ckpt_path
        self.save_ckpt_dir = save_ckpt_dir
        self.compile_ = bool(compile_)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_bfs_width = int(max_bfs_width)
        self.lr = float(lr)
        self.wd = float(wd)
        self.lr_schedule = bool(lr_schedule)
        self.max_grad_norm = float(max_grad_norm)
        self.max_steps = int(max_steps)
        self.embedding_model = embedding_model
        self.d_text = int(d_text)
        self.seq_len = int(seq_len)
        self.num_blocks = int(num_blocks)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def _rt_main_config(self) -> Dict[str, Any]:
        task_tuple = (self.dataset_name, self.task_name, self.target_col, self.columns_to_drop)
        return {
            "project": self.project,
            "eval_splits": self.eval_splits,
            "eval_freq": self.eval_freq,
            "eval_pow2": self.eval_pow2,
            "max_eval_steps": self.max_eval_steps,
            "load_ckpt_path": self.load_ckpt_path,
            "save_ckpt_dir": self.save_ckpt_dir,
            "compile_": self.compile_,
            "seed": self.seed,
            "train_tasks": [task_tuple],
            "eval_tasks": [task_tuple],
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_bfs_width": self.max_bfs_width,
            "lr": self.lr,
            "wd": self.wd,
            "lr_schedule": self.lr_schedule,
            "max_grad_norm": self.max_grad_norm,
            "max_steps": self.max_steps,
            "embedding_model": self.embedding_model,
            "d_text": self.d_text,
            "seq_len": self.seq_len,
            "num_blocks": self.num_blocks,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
        }

    def _extract_test_metric(self, logs: str) -> Optional[float]:
        numeric = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        escaped_dataset = re.escape(self.dataset_name)
        escaped_task = re.escape(self.task_name)
        patterns = [
            rf"(?:auc|r2|mae)/{escaped_dataset}/{escaped_task}/test:\s*{numeric}",
            rf"{escaped_dataset}/{escaped_task}/test:\s*{numeric}",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, logs)
            if matches:
                return float(matches[-1])
        return None

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> Dict[str, Any]:
        repo_path = Path(self.rt_repo_path).expanduser().resolve()
        rt_main_path = repo_path / "rt" / "main.py"
        if not rt_main_path.is_file():
            raise FileNotFoundError(
                f"Official RT repo not found at '{repo_path}'. Missing rt/main.py."
            )

        cfg = self._rt_main_config()
        env = os.environ.copy()
        env["RT_MAIN_CONFIG"] = json.dumps(cfg)
        command = [
            self.python_executable,
            "-c",
            (
                "import json, os\n"
                "from rt.main import main\n"
                "main(**json.loads(os.environ['RT_MAIN_CONFIG']))\n"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        logs = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if result.returncode != 0:
            tail = "\n".join(logs.splitlines()[-120:])
            raise RuntimeError(
                f"Official RT run failed. returncode={result.returncode}\n--- tail ---\n{tail}"
            )

        metric = self._extract_test_metric(logs)
        if metric is None:
            tail = "\n".join(logs.splitlines()[-120:])
            raise RuntimeError(
                "Official RT run succeeded but test metric was not found in logs.\n"
                f"--- tail ---\n{tail}"
            )

        return {
            "backend": "official_relational_transformer",
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "target_col": self.target_col,
            "precomputed_test_metric": float(metric),
        }

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise RuntimeError(
            "OfficialRelationalTransformerAdapter does not emit row-wise predictions in this integration. "
            "Use ExperimentRunner metric override support with `precomputed_test_metric`."
        )


class ExternalGraphAdapter(BaseViewModel):
    """Thin wrapper for RelGT / GNN graph baselines."""

    def __init__(
        self,
        name: str,
        fit_fn: Callable[..., Tuple[Any, Dict[str, Any]]],
        predict_fn: Callable[..., np.ndarray],
        **kwargs: Any,
    ):
        self._name = name
        self._view_name = "graphified"
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.model_obj: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> Dict[str, Any]:
        self.model_obj, meta = self.fit_fn(
            train_data=train_data, val_data=val_data, task=task, **self.kwargs
        )
        return meta

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        return self.predict_fn(model_obj=self.model_obj, data=data, task=task, **self.kwargs)


# -----------------------------------------------------------------------------
# View materialization
# -----------------------------------------------------------------------------


@dataclass
class MaterializedViews:
    joined_table: Dict[str, pd.DataFrame]
    relational: Any
    graphified: Any


class ViewFactory:
    def __init__(self, db: DatabaseSpec, task: TaskSpec):
        self.db = db
        self.task = task

    def build_joined_table(self, cfg: JoinedTableConfig) -> pd.DataFrame:
        builder = JoinedTableBuilder(self.db, self.task)
        return builder.build(cfg)

    def build_relational_view(self) -> Dict[str, Any]:
        """
        Pass-through relational payload.

        This keeps the database and task table intact for RT-style consumers.
        """
        return {
            "db": self.db,
            "task": self.task,
        }

    def build_graphified_view(self, neighborhood_size_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Minimal graphified payload contract.

        In practice, you will replace this with RelBench graph loaders / REG builders.
        The key thing is that this object preserves seed-time semantics and the task table.
        """
        return {
            "db": self.db,
            "task": self.task,
            "graph_config": {"neighborhood_size_k": neighborhood_size_k},
        }


# -----------------------------------------------------------------------------
# Experiment result types
# -----------------------------------------------------------------------------


@dataclass
class RunResult:
    task_name: str
    model_name: str
    view_name: str
    split_name: str
    metric_name: str
    metric_value: float
    utility_value: float
    compute: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolSummary:
    per_run: List[RunResult]
    per_task_ris: Dict[str, Dict[str, float]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "task_name": r.task_name,
                    "model_name": r.model_name,
                    "view_name": r.view_name,
                    "split_name": r.split_name,
                    "metric_name": r.metric_name,
                    "metric_value": r.metric_value,
                    "utility_value": r.utility_value,
                    **{f"compute__{k}": v for k, v in r.compute.items()},
                    **{f"extra__{k}": v for k, v in r.extra.items()},
                }
                for r in self.per_run
            ]
        )


# -----------------------------------------------------------------------------
# Protocol A and B runners
# -----------------------------------------------------------------------------


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        set_random_seed(config.seed)

    def run_protocol_a(
        self,
        db: DatabaseSpec,
        task: TaskSpec,
        split: TemporalSplit,
        joined_table_cfgs: Sequence[JoinedTableConfig],
        joined_models: Sequence[BaseViewModel],
        relational_models: Sequence[BaseViewModel],
        graph_models: Sequence[BaseViewModel],
    ) -> ProtocolSummary:
        """
        Protocol A: tri-view head-to-head benchmark.
        """
        results: List[RunResult] = []
        vf = ViewFactory(db, task)

        # Joined-table track.
        for jt_cfg in joined_table_cfgs:
            LOGGER.info(
                "Building joined-table view %s | task=%s",
                jt_cfg.view_name,
                task.name,
            )
            jt_df = vf.build_joined_table(jt_cfg)
            train_df = jt_df.loc[split.train_mask].reset_index(drop=True)
            val_df = jt_df.loc[split.val_mask].reset_index(drop=True)
            test_df = jt_df.loc[split.test_mask].reset_index(drop=True)
            LOGGER.info(
                "Joined view %s shapes | train=%s val=%s test=%s cols=%d",
                jt_cfg.view_name,
                len(train_df),
                len(val_df),
                len(test_df),
                jt_df.shape[1],
            )

            for model in joined_models:
                model_label = f"{model.name}:{jt_cfg.view_name}"
                LOGGER.info("Running Protocol A | task=%s | model=%s", task.name, model_label)
                with compute_tracker() as comp:
                    LOGGER.info("Fitting | %s", model_label)
                    fit_meta = model.fit(train_df, val_df, task)
                    LOGGER.info("Scoring test | %s | n_test=%d", model_label, len(test_df))
                    score = self._score_after_fit(
                        model=model,
                        fit_meta=fit_meta,
                        data=test_df,
                        y_true=test_df[task.target_col].to_numpy(),
                        task=task,
                    )
                results.append(
                    RunResult(
                        task_name=task.name,
                        model_name=model_label,
                        view_name="joined-table",
                        split_name="test",
                        metric_name=task.metric_name,
                        metric_value=score,
                        utility_value=metric_to_utility(score, task.metric_name),
                        compute=dict(comp),
                        extra={**fit_meta, "joined_subview": jt_cfg.view_name},
                    )
                )

        # Relational track.
        relational_payload = vf.build_relational_view()
        relational_train = {
            "payload": relational_payload,
            "task_df": task.task_table.loc[split.train_mask].reset_index(drop=True),
        }
        relational_val = {
            "payload": relational_payload,
            "task_df": task.task_table.loc[split.val_mask].reset_index(drop=True),
        }
        relational_test = {
            "payload": relational_payload,
            "task_df": task.task_table.loc[split.test_mask].reset_index(drop=True),
        }

        for model in relational_models:
            LOGGER.info("Running Protocol A | task=%s | model=%s", task.name, model.name)
            with compute_tracker() as comp:
                LOGGER.info("Fitting | relational | %s", model.name)
                fit_meta = model.fit(relational_train, relational_val, task)
                LOGGER.info(
                    "Scoring test | relational | %s | n_test=%d",
                    model.name,
                    len(relational_test["task_df"]),
                )
                score = self._score_after_fit(
                    model=model,
                    fit_meta=fit_meta,
                    data=relational_test,
                    y_true=relational_test["task_df"][task.target_col].to_numpy(),
                    task=task,
                )
            results.append(
                RunResult(
                    task_name=task.name,
                    model_name=model.name,
                    view_name="relational",
                    split_name="test",
                    metric_name=task.metric_name,
                    metric_value=score,
                    utility_value=metric_to_utility(score, task.metric_name),
                    compute=dict(comp),
                    extra=fit_meta,
                )
            )

        # Graphified track.
        graph_payload = vf.build_graphified_view(neighborhood_size_k=None)
        graph_train = {
            "payload": graph_payload,
            "task_df": task.task_table.loc[split.train_mask].reset_index(drop=True),
        }
        graph_val = {
            "payload": graph_payload,
            "task_df": task.task_table.loc[split.val_mask].reset_index(drop=True),
        }
        graph_test = {
            "payload": graph_payload,
            "task_df": task.task_table.loc[split.test_mask].reset_index(drop=True),
        }

        for model in graph_models:
            LOGGER.info("Running Protocol A | task=%s | model=%s", task.name, model.name)
            with compute_tracker() as comp:
                LOGGER.info("Fitting | graphified | %s", model.name)
                fit_meta = model.fit(graph_train, graph_val, task)
                LOGGER.info(
                    "Scoring test | graphified | %s | n_test=%d",
                    model.name,
                    len(graph_test["task_df"]),
                )
                score = self._score_after_fit(
                    model=model,
                    fit_meta=fit_meta,
                    data=graph_test,
                    y_true=graph_test["task_df"][task.target_col].to_numpy(),
                    task=task,
                )
            results.append(
                RunResult(
                    task_name=task.name,
                    model_name=model.name,
                    view_name="graphified",
                    split_name="test",
                    metric_name=task.metric_name,
                    metric_value=score,
                    utility_value=metric_to_utility(score, task.metric_name),
                    compute=dict(comp),
                    extra=fit_meta,
                )
            )

        per_task_ris = self._summarize_ris(results)
        summary = ProtocolSummary(per_run=results, per_task_ris=per_task_ris)
        self._save_protocol_summary(task.name, "protocol_a", summary)
        return summary

    def run_protocol_b(
        self,
        db: DatabaseSpec,
        task: TaskSpec,
        split: TemporalSplit,
        joined_ablation_cfgs: Sequence[JoinedTableConfig],
        joined_models: Sequence[BaseViewModel],
        graph_k_values: Sequence[int],
        graph_models_factory: Callable[[int], Sequence[BaseViewModel]],
    ) -> ProtocolSummary:
        """
        Protocol B: invariance brittleness through controlled ablations.

        Included here:
        - joined-table ablations: join depth / window / aggregation choices
        - graphified ablations: neighborhood size K
        """
        results: List[RunResult] = []
        vf = ViewFactory(db, task)

        for jt_cfg in joined_ablation_cfgs:
            jt_df = vf.build_joined_table(jt_cfg)
            train_df = jt_df.loc[split.train_mask].reset_index(drop=True)
            val_df = jt_df.loc[split.val_mask].reset_index(drop=True)
            test_df = jt_df.loc[split.test_mask].reset_index(drop=True)
            for model in joined_models:
                label = f"{model.name}:{jt_cfg.view_name}:hops={jt_cfg.max_path_hops}:w={jt_cfg.lookback_days}"
                LOGGER.info(
                    "Running Protocol B joined ablation | task=%s | model=%s", task.name, label
                )
                with compute_tracker() as comp:
                    fit_meta = model.fit(train_df, val_df, task)
                    score = self._score_after_fit(
                        model=model,
                        fit_meta=fit_meta,
                        data=test_df,
                        y_true=test_df[task.target_col].to_numpy(),
                        task=task,
                    )
                results.append(
                    RunResult(
                        task_name=task.name,
                        model_name=label,
                        view_name="joined-table",
                        split_name="test",
                        metric_name=task.metric_name,
                        metric_value=score,
                        utility_value=metric_to_utility(score, task.metric_name),
                        compute=dict(comp),
                        extra={**fit_meta, **dataclasses.asdict(jt_cfg)},
                    )
                )

        for k in graph_k_values:
            payload = vf.build_graphified_view(neighborhood_size_k=k)
            train = {
                "payload": payload,
                "task_df": task.task_table.loc[split.train_mask].reset_index(drop=True),
            }
            val = {
                "payload": payload,
                "task_df": task.task_table.loc[split.val_mask].reset_index(drop=True),
            }
            test = {
                "payload": payload,
                "task_df": task.task_table.loc[split.test_mask].reset_index(drop=True),
            }
            for model in graph_models_factory(k):
                label = f"{model.name}:K={k}"
                LOGGER.info(
                    "Running Protocol B graph ablation | task=%s | model=%s", task.name, label
                )
                with compute_tracker() as comp:
                    fit_meta = model.fit(train, val, task)
                    score = self._score_after_fit(
                        model=model,
                        fit_meta=fit_meta,
                        data=test,
                        y_true=test["task_df"][task.target_col].to_numpy(),
                        task=task,
                    )
                results.append(
                    RunResult(
                        task_name=task.name,
                        model_name=label,
                        view_name="graphified",
                        split_name="test",
                        metric_name=task.metric_name,
                        metric_value=score,
                        utility_value=metric_to_utility(score, task.metric_name),
                        compute=dict(comp),
                        extra={**fit_meta, "K": k},
                    )
                )

        per_task_ris = self._summarize_ris(results)
        summary = ProtocolSummary(per_run=results, per_task_ris=per_task_ris)
        self._save_protocol_summary(task.name, "protocol_b", summary)
        return summary

    @staticmethod
    def _extract_metric_override(fit_meta: Any, split_name: str = "test") -> Optional[float]:
        if not isinstance(fit_meta, Mapping):
            return None
        keys = [f"precomputed_{split_name}_metric", "precomputed_metric", "metric_override"]
        for key in keys:
            value = fit_meta.get(key)
            if value is None:
                continue
            return float(value)
        return None

    def _score_after_fit(
        self,
        model: BaseViewModel,
        fit_meta: Any,
        data: Any,
        y_true: np.ndarray,
        task: TaskSpec,
    ) -> float:
        override = self._extract_metric_override(fit_meta, split_name="test")
        if override is not None:
            return float(override)
        pred = model.predict(data, task)
        return evaluate_predictions(y_true, pred, task.metric_name)

    def _summarize_ris(self, results: Sequence[RunResult]) -> Dict[str, Dict[str, float]]:
        task_to_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        task_to_metric: Dict[str, str] = {}

        for r in results:
            task_to_metric[r.task_name] = r.metric_name
            representation_key = self._representation_key(r)
            current = task_to_scores[r.task_name].get(representation_key)
            if current is None:
                task_to_scores[r.task_name][representation_key] = r.metric_value
            else:
                if r.metric_name == "auroc":
                    task_to_scores[r.task_name][representation_key] = max(
                        current, r.metric_value
                    )
                else:
                    task_to_scores[r.task_name][representation_key] = min(
                        current, r.metric_value
                    )

        summary = {}
        for task_name, representation_scores in task_to_scores.items():
            ris_summary = compute_ris(representation_scores, task_to_metric[task_name])
            ris_summary["representation_scores"] = {
                key: float(value) for key, value in representation_scores.items()
            }
            summary[task_name] = ris_summary
        return summary

    @staticmethod
    def _representation_key(run: RunResult) -> str:
        parts = [run.view_name]
        if run.view_name == "joined-table":
            subview = run.extra.get("joined_subview")
            if subview:
                parts.append(str(subview))
            if "max_path_hops" in run.extra:
                parts.append(f"hops={run.extra['max_path_hops']}")
            if "lookback_days" in run.extra:
                parts.append(f"window={run.extra['lookback_days']}")
        elif run.view_name == "graphified":
            graphified = run.extra.get("graphified")
            if graphified:
                parts.append(str(graphified))
            if "K" in run.extra:
                parts.append(f"K={run.extra['K']}")
        elif run.view_name == "relational":
            backend = run.extra.get("backend")
            if backend:
                parts.append(str(backend))
        return "|".join(parts)

    def _save_protocol_summary(
        self, task_name: str, protocol_name: str, summary: ProtocolSummary
    ) -> None:
        out_dir = Path(self.config.output_dir)
        df = summary.to_frame()
        df.to_parquet(out_dir / f"{task_name}__{protocol_name}__runs.parquet", index=False)
        with open(out_dir / f"{task_name}__{protocol_name}__ris.json", "w", encoding="utf-8") as f:
            json.dump(summary.per_task_ris, f, indent=2)


# -----------------------------------------------------------------------------
# Leakage test helpers
# -----------------------------------------------------------------------------


def inject_future_only_signal(
    task_df: pd.DataFrame,
    seed_time_col: str,
    target_col: str,
    signal_name: str = "future_leak_signal",
) -> pd.DataFrame:
    """
    Synthetic diagnostic helper.

    Attaches a signal that is ONLY valid if code accidentally uses post-seed rows.
    Use this in unit tests to ensure your joined-table pipeline does not exploit it.
    """
    df = task_df.copy()
    seed_times = pd.to_datetime(df[seed_time_col], utc=False)
    future_ts = seed_times + pd.to_timedelta(7, unit="D")
    df[signal_name] = np.where(df[target_col].notna(), future_ts.view("int64"), np.nan)
    return df


# -----------------------------------------------------------------------------
# Example adapters for dry runs / smoke tests
# -----------------------------------------------------------------------------


class MeanDummyAdapter(BaseViewModel):
    def __init__(self, name: str, view_name: str):
        self._name = name
        self._view_name = view_name
        self.constant_: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> Dict[str, Any]:
        if isinstance(train_data, dict):
            y = train_data["task_df"][task.target_col].to_numpy()
        else:
            y = train_data[task.target_col].to_numpy()
        self.constant_ = float(np.mean(y))
        return {"status": "dummy_fitted"}

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        if self.constant_ is None:
            raise RuntimeError("Model must be fit before predict.")
        if isinstance(data, dict):
            n = len(data["task_df"])
        else:
            n = len(data)
        return np.full(shape=(n,), fill_value=self.constant_, dtype=float)


# -----------------------------------------------------------------------------
# Minimal synthetic example
# -----------------------------------------------------------------------------


def make_synthetic_relational_problem() -> Tuple[DatabaseSpec, TaskSpec, TemporalSplit]:
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "country": ["NL", "DE", "NL", "FR", "DE"],
            "age": [28, 41, 35, 24, 50],
            "updated_at": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]
            ),
        }
    )

    transactions = pd.DataFrame(
        {
            "tx_id": range(1, 16),
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5],
            "amount": [10, 15, 12, 9, 8, 2, 30, 28, 31, 3, 5, 40, 42, 38, 39],
            "channel": [
                "web",
                "app",
                "web",
                "web",
                "app",
                "app",
                "web",
                "app",
                "app",
                "web",
                "app",
                "web",
                "app",
                "web",
                "app",
            ],
            "event_time": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-15",
                    "2024-02-01",
                    "2024-01-07",
                    "2024-03-01",
                    "2024-01-20",
                    "2024-02-11",
                    "2024-03-05",
                    "2024-03-15",
                    "2024-01-03",
                    "2024-02-07",
                    "2024-01-10",
                    "2024-02-10",
                    "2024-03-10",
                    "2024-03-20",
                ]
            ),
        }
    )

    task_table = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "seed_time": pd.to_datetime(
                [
                    "2024-02-15",
                    "2024-02-15",
                    "2024-02-15",
                    "2024-02-15",
                    "2024-02-15",
                    "2024-03-16",
                    "2024-03-16",
                    "2024-03-16",
                    "2024-03-16",
                    "2024-03-16",
                ]
            ),
            "label": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
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
        name="synthetic-user-churn",
        task_type="classification",
        target_col="label",
        metric_name="auroc",
        entity_table="users",
        entity_key="user_id",
        seed_time_col="seed_time",
        task_table=task_table,
    )

    split = make_temporal_split(
        task_df=task_table,
        seed_time_col="seed_time",
        val_cutoff=pd.Timestamp("2024-03-01"),
        test_cutoff=pd.Timestamp("2024-03-10"),
    )
    return db, task, split


def load_relbench_entity_problem(
    dataset_name: str,
    task_name: str,
    *,
    download: bool = True,
) -> Tuple[DatabaseSpec, TaskSpec, TemporalSplit]:
    """
    Build ContinuumBench-style ``(db, task, split)`` from a RelBench **entity** task.

    Train/val/test masks follow RelBench's published row order (train rows first, then val,
    then test) so metrics stay aligned with RelBench / Relational Transformer expectations.
    """
    from relbench.base import EntityTask, TaskType
    from relbench.tasks import get_task

    rb_task = get_task(dataset_name, task_name, download=download)
    if not isinstance(rb_task, EntityTask):
        raise TypeError(
            f"RelBench task {task_name!r} on {dataset_name!r} is not an EntityTask; "
            "link-prediction / recommendation tasks are not wired in this harness."
        )
    if rb_task.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        raise NotImplementedError(
            f"RelBench task type {rb_task.task_type!r} is not supported yet "
            "(supported: binary classification, regression)."
        )

    train_t = rb_task.get_table("train", mask_input_cols=False)
    val_t = rb_task.get_table("val", mask_input_cols=False)
    test_t = rb_task.get_table("test", mask_input_cols=False)
    train_df = train_t.df.reset_index(drop=True)
    val_df = val_t.df.reset_index(drop=True)
    test_df = test_t.df.reset_index(drop=True)
    task_table = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    n_tr = len(train_df)
    n_va = len(val_df)
    n_te = len(test_df)
    train_mask = pd.Series([True] * n_tr + [False] * (n_va + n_te), index=task_table.index)
    val_mask = pd.Series([False] * n_tr + [True] * n_va + [False] * n_te, index=task_table.index)
    test_mask = pd.Series([False] * (n_tr + n_va) + [True] * n_te, index=task_table.index)
    split = TemporalSplit(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    rb_db = rb_task.dataset.get_db()
    tables: Dict[str, TableSpec] = {}
    for name, tbl in rb_db.table_dict.items():
        tables[name] = TableSpec(
            name=name,
            df=tbl.df.copy(),
            primary_key=tbl.pkey_col,
            time_col=tbl.time_col,
            foreign_keys=dict(tbl.fkey_col_to_pkey_table),
        )
    db = DatabaseSpec(tables=tables)

    if rb_task.task_type == TaskType.BINARY_CLASSIFICATION:
        ttype = "classification"
        metric = "auroc"
    else:
        ttype = "regression"
        metric = "mae"

    task = TaskSpec(
        name=f"{dataset_name}__{task_name}",
        task_type=ttype,
        target_col=rb_task.target_col,
        metric_name=metric,
        entity_table=rb_task.entity_table,
        entity_key=rb_task.entity_col,
        seed_time_col=rb_task.time_col,
        task_table=task_table,
    )
    return db, task, split


# -----------------------------------------------------------------------------
# Hooks you can replace with actual RT / RelGT integrations
# -----------------------------------------------------------------------------


def stub_relational_fit_fn(
    train_data: Any, val_data: Any, task: TaskSpec, **kwargs: Any
) -> Tuple[Any, Dict[str, Any]]:
    model_obj = {"mean": float(train_data["task_df"][task.target_col].mean())}
    return model_obj, {"backend": "stub_relational"}


def stub_relational_predict_fn(
    model_obj: Any, data: Any, task: TaskSpec, **kwargs: Any
) -> np.ndarray:
    return np.full(len(data["task_df"]), model_obj["mean"], dtype=float)


def stub_graph_fit_fn(
    train_data: Any, val_data: Any, task: TaskSpec, **kwargs: Any
) -> Tuple[Any, Dict[str, Any]]:
    model_obj = {"mean": float(train_data["task_df"][task.target_col].mean())}
    return model_obj, {"backend": "stub_graph", **kwargs}


def stub_graph_predict_fn(model_obj: Any, data: Any, task: TaskSpec, **kwargs: Any) -> np.ndarray:
    return np.full(len(data["task_df"]), model_obj["mean"], dtype=float)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ContinuumBench v1 experiment harness")
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic smoke test")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    runner = ExperimentRunner(
        ExperimentConfig(benchmark_name="continuumbench-v1", output_dir=args.output_dir)
    )

    if args.smoke_test:
        db, task, split = make_synthetic_relational_problem()

        joined_cfgs = [
            JoinedTableConfig(view_name="jt_entity", max_path_hops=1),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
        ]

        joined_models: List[BaseViewModel] = []
        if lgb is not None:
            joined_models.append(LightGBMTabularAdapter())
        else:
            joined_models.append(MeanDummyAdapter(name="dummy-jt", view_name="joined-table"))

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

        summary_a = runner.run_protocol_a(
            db=db,
            task=task,
            split=split,
            joined_table_cfgs=joined_cfgs,
            joined_models=joined_models,
            relational_models=relational_models,
            graph_models=graph_models,
        )
        print("Protocol A results")
        print(summary_a.to_frame())
        print(json.dumps(summary_a.per_task_ris, indent=2))

        summary_b = runner.run_protocol_b(
            db=db,
            task=task,
            split=split,
            joined_ablation_cfgs=[
                JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=7),
                JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
                JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=None),
            ],
            joined_models=joined_models,
            graph_k_values=[100, 300, 500],
            graph_models_factory=lambda k: [
                ExternalGraphAdapter(
                    name="relgt-stub",
                    fit_fn=stub_graph_fit_fn,
                    predict_fn=stub_graph_predict_fn,
                    neighborhood_size_k=k,
                )
            ],
        )
        print("Protocol B results")
        print(summary_b.to_frame())
        print(json.dumps(summary_b.per_task_ris, indent=2))


if __name__ == "__main__":
    main()
