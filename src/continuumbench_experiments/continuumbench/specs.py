from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class TableSpec:
    name: str
    df: pd.DataFrame
    primary_key: str
    time_col: Optional[str] = None
    foreign_keys: dict[str, str] = field(default_factory=dict)


@dataclass
class TaskSpec:
    name: str
    task_type: str
    target_col: str
    metric_name: str
    entity_table: str
    entity_key: str
    seed_time_col: str
    task_table: pd.DataFrame


@dataclass
class DatabaseSpec:
    tables: dict[str, TableSpec]

    def table(self, name: str) -> TableSpec:
        return self.tables[name]


@dataclass
class TemporalSplit:
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series


@dataclass
class JoinedTableConfig:
    view_name: str
    lookback_days: Optional[int] = None
    max_path_hops: int = 1
    include_parent_lookups: bool = True
    aggregation_functions: tuple[str, ...] = (
        "count",
        "nunique",
        "mean",
        "sum",
        "min",
        "max",
    )
    recency_features: bool = True
    drop_high_cardinality_threshold: int = 500


@dataclass
class ExperimentConfig:
    benchmark_name: str
    seed: int = 7
    small_medium_track: bool = False
    output_dir: str = "outputs"
    ris_baseline_utility: float = 0.0
