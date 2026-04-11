from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .specs import DatabaseSpec, JoinedTableConfig, TableSpec, TaskSpec

LOGGER = logging.getLogger("continuumbench")


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
            [base.reset_index(drop=True), entity_features.reset_index(drop=True)],
            axis=1,
        )
        if config.include_parent_lookups:
            out = self._join_parent_lookups(out, entity_table, self.task.seed_time_col)
        return self._drop_duplicate_columns(out)

    def _build_jt_temporalagg(self, config: JoinedTableConfig) -> pd.DataFrame:
        out = self._build_jt_entity(config)
        reachable = self._reachable_tables(
            self.task.entity_table,
            max_hops=config.max_path_hops,
        )

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
                out = pd.concat(
                    [out.reset_index(drop=True), agg_df.reset_index(drop=True)],
                    axis=1,
                )
        return self._drop_duplicate_columns(out)

    def _drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _join_parent_lookups(
        self,
        instances: pd.DataFrame,
        entity_table: TableSpec,
        seed_time_col: str,
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
                [out.reset_index(drop=True), parent_features.reset_index(drop=True)],
                axis=1,
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
        left = left_df[[left_on, seed_time_col]].copy()
        left["__row_id__"] = np.arange(len(left_df))
        left[seed_time_col] = pd.to_datetime(left[seed_time_col], utc=False)
        right = right_df.copy()

        if right_time_col is None:
            merged = left.merge(right, how="left", left_on=left_on, right_on=right_on)
            merged = merged.sort_values("__row_id__").reset_index(drop=True)
            merged = merged[
                [
                    column
                    for column in merged.columns
                    if column not in {left_on, seed_time_col, right_on, "__row_id__"}
                ]
            ]
            merged.columns = [f"{prefix}{column}" for column in merged.columns]
            return merged

        right[right_time_col] = pd.to_datetime(right[right_time_col], utc=False)
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
                column
                for column in merged.columns
                if column
                not in {left_on, seed_time_col, right_on, right_time_col, "__row_id__"}
            ]
        ]
        merged.columns = [f"{prefix}{column}" for column in merged.columns]
        return merged

    def _reachable_tables(
        self,
        root_table_name: str,
        max_hops: int,
    ) -> dict[str, list[tuple[str, str, str]]]:
        adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for table in self.db.tables.values():
            for fk_col, parent in table.foreign_keys.items():
                adjacency[table.name].append((table.name, fk_col, parent))
                adjacency[parent].append(
                    (parent, self.db.table(parent).primary_key, table.name)
                )

        visited = {root_table_name}
        frontier: list[tuple[str, list[tuple[str, str, str]]]] = [(root_table_name, [])]
        paths: dict[str, list[tuple[str, str, str]]] = {root_table_name: []}

        for _ in range(max_hops):
            next_frontier: list[tuple[str, list[tuple[str, str, str]]]] = []
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
        join_path: list[tuple[str, str, str]],
        seed_time_col: str,
        lookback_days: Optional[int],
        agg_funcs: tuple[str, ...] | list[str],
        recency_features: bool,
    ) -> Optional[pd.DataFrame]:
        entity_key = self.task.entity_key
        table_df = table.df.copy()
        table_df_columns = set(table_df.columns)

        direct_fk_to_entity = None
        for fk_col, parent in table.foreign_keys.items():
            if parent == self.task.entity_table:
                direct_fk_to_entity = fk_col
                break

        if direct_fk_to_entity is None:
            LOGGER.info(
                "Skipping table '%s' for temporal aggregation: no direct FK to entity table.",
                table.name,
            )
            return None

        seed_alias = "__instance_seed_time__"
        base = instances[[entity_key, seed_time_col]].copy()
        base = base.rename(columns={seed_time_col: seed_alias})
        base[seed_alias] = pd.to_datetime(base[seed_alias], utc=False)

        matched_child_row = "__matched_child_row__"
        t_df = table_df.copy()
        while matched_child_row in t_df.columns or matched_child_row in base.columns:
            matched_child_row = f"_{matched_child_row}"
        t_df[matched_child_row] = 1
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
            matched_child_row,
        }
        numeric_cols = [
            column
            for column in joined.columns
            if column not in exclude_cols and pd.api.types.is_numeric_dtype(joined[column])
        ]
        categorical_cols = [
            column
            for column in joined.columns
            if column not in exclude_cols
            and not pd.api.types.is_numeric_dtype(joined[column])
            and column != table.time_col
        ]

        parts = []
        zero_fill_cols: list[str] = []
        group = joined.groupby("__instance_id__", dropna=False)

        if "count" in agg_funcs:
            count_name = f"{table.name}__row_count"
            parts.append(group[matched_child_row].count().rename(count_name).to_frame())
            zero_fill_cols.append(count_name)

        if table.time_col and recency_features:
            last_seen = group[table.time_col].max()
            seed_per_instance = joined.groupby("__instance_id__")[seed_alias].first()
            recency_days = (seed_per_instance - last_seen).dt.total_seconds() / 86400.0
            parts.append(recency_days.rename(f"{table.name}__recency_days").to_frame())

        for column in numeric_cols:
            series_group = group[column]
            if "mean" in agg_funcs:
                parts.append(series_group.mean().rename(f"{table.name}__{column}__mean").to_frame())
            if "sum" in agg_funcs:
                parts.append(
                    series_group.sum(min_count=1).rename(f"{table.name}__{column}__sum").to_frame()
                )
            if "min" in agg_funcs:
                parts.append(series_group.min().rename(f"{table.name}__{column}__min").to_frame())
            if "max" in agg_funcs:
                parts.append(series_group.max().rename(f"{table.name}__{column}__max").to_frame())

        if "nunique" in agg_funcs:
            for column in categorical_cols:
                nunique_name = f"{table.name}__{column}__nunique"
                parts.append(group[column].nunique(dropna=True).rename(nunique_name).to_frame())
                zero_fill_cols.append(nunique_name)

        if not parts:
            return None

        agg_df = pd.concat(parts, axis=1).sort_index()
        agg_df = agg_df.reindex(range(len(instances)))
        for column in zero_fill_cols:
            if column in agg_df.columns:
                agg_df[column] = agg_df[column].fillna(0).astype(np.int64)
        return agg_df.reset_index(drop=True)


def count_incident_tables(db: DatabaseSpec, task: TaskSpec) -> int:
    total = 0
    for table_name, table in db.tables.items():
        if table_name == task.entity_table:
            continue
        for parent in table.foreign_keys.values():
            if parent == task.entity_table:
                total += 1
                break
    return total


def _static_incident_counts(
    table_df: pd.DataFrame,
    fk_col: str,
    queries: pd.DataFrame,
    entity_col: str,
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
    query_times = pd.to_datetime(queries[seed_col], utc=False)
    out = np.zeros(len(queries), dtype=np.int64)
    event_groups: dict[object, np.ndarray] = {}

    for entity, group in events.groupby(fk_col, sort=False):
        event_groups[entity] = group[time_col].to_numpy(dtype="datetime64[ns]")

    for idx in range(len(queries)):
        entity = queries[entity_col].iloc[idx]
        query_time = query_times.iloc[idx]
        times = event_groups.get(entity)
        if times is None or len(times) == 0:
            continue
        query_time64 = np.datetime64(query_time.to_datetime64())
        out[idx] = int(np.searchsorted(times, query_time64, side="right"))
    return out


def build_graph_degree_feature_table(
    db: DatabaseSpec,
    task: TaskSpec,
    rows: pd.DataFrame,
    max_incident_tables: Optional[int] = None,
) -> pd.DataFrame:
    cols = [task.entity_key, task.seed_time_col, task.target_col]
    missing = [column for column in cols if column not in rows.columns]
    if missing:
        raise KeyError(f"rows is missing columns required for graph features: {missing}")

    base = rows[cols].copy()
    base[task.seed_time_col] = pd.to_datetime(base[task.seed_time_col], utc=False)

    candidates: list[tuple[str, TableSpec]] = []
    for table_name, table in sorted(db.tables.items(), key=lambda item: item[0]):
        if table_name == task.entity_table:
            continue
        fk_to_entity = None
        for fk_col, parent in table.foreign_keys.items():
            if parent == task.entity_table:
                fk_to_entity = fk_col
                break
        if fk_to_entity is not None:
            candidates.append((table_name, table))

    if max_incident_tables is not None:
        candidates = candidates[: int(max_incident_tables)]

    features: dict[str, np.ndarray] = {}
    for table_name, table in candidates:
        fk = None
        for fk_col, parent in table.foreign_keys.items():
            if parent == task.entity_table:
                fk = fk_col
                break
        if fk is None:
            continue
        if table.time_col and table.time_col in table.df.columns:
            features[f"graphdeg__{table_name}__count_leq_seed"] = _temporal_incident_counts(
                table.df,
                fk_col=fk,
                time_col=table.time_col,
                queries=base,
                entity_col=task.entity_key,
                seed_col=task.seed_time_col,
            )
        else:
            features[f"graphdeg__{table_name}__static_count"] = _static_incident_counts(
                table.df,
                fk,
                base,
                task.entity_key,
            )

    feature_df = pd.DataFrame(features, index=base.index)
    return pd.concat([base.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


@dataclass
class MaterializedViews:
    joined_table: dict[str, pd.DataFrame]
    relational: Any
    graphified: Any


class ViewFactory:
    def __init__(self, db: DatabaseSpec, task: TaskSpec):
        self.db = db
        self.task = task

    def build_joined_table(self, cfg: JoinedTableConfig) -> pd.DataFrame:
        return JoinedTableBuilder(self.db, self.task).build(cfg)

    def build_relational_view(self) -> dict[str, Any]:
        return {"db": self.db, "task": self.task}

    def build_graphified_view(
        self,
        neighborhood_size_k: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "db": self.db,
            "task": self.task,
            "graph_config": {"neighborhood_size_k": neighborhood_size_k},
        }
