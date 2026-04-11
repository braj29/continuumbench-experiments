from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .specs import DatabaseSpec, TableSpec, TaskSpec, TemporalSplit


def load_dataset_entity_problem(
    dataset_name: str,
    task_name: str,
    *,
    download: bool = True,
) -> Tuple[DatabaseSpec, TaskSpec, TemporalSplit]:
    """
    Build ContinuumBench-style ``(db, task, split)`` from a registered **entity** task.

    Train/val/test masks follow the source registry row order (train rows first, then val,
    then test) so metrics stay aligned with the backing task definition.
    """
    from relbench.base import EntityTask, TaskType
    from relbench.tasks import get_task

    source_task = get_task(dataset_name, task_name, download=download)
    _validate_entity_task(source_task, dataset_name, task_name, EntityTask, TaskType)

    train_df, val_df, test_df = _load_split_frames(source_task)
    task_table = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    split = _build_ordered_split(train_df, val_df, test_df, task_table.index)
    db = _build_database_spec(source_task)
    task_type, metric_name = _task_type_metadata(source_task.task_type, TaskType)

    task = TaskSpec(
        name=f"{dataset_name}__{task_name}",
        task_type=task_type,
        target_col=source_task.target_col,
        metric_name=metric_name,
        entity_table=source_task.entity_table,
        entity_key=source_task.entity_col,
        seed_time_col=source_task.time_col,
        task_table=task_table,
    )
    return db, task, split


def _validate_entity_task(
    source_task,
    dataset_name: str,
    task_name: str,
    entity_task_cls,
    task_type_cls,
) -> None:
    if not isinstance(source_task, entity_task_cls):
        raise TypeError(
            f"Task {task_name!r} on dataset {dataset_name!r} is not an EntityTask; "
            "link-prediction / recommendation tasks are not wired in this harness."
        )
    if source_task.task_type not in (
        task_type_cls.BINARY_CLASSIFICATION,
        task_type_cls.REGRESSION,
    ):
        raise NotImplementedError(
            f"Task type {source_task.task_type!r} is not supported yet "
            "(supported: binary classification, regression)."
        )


def _load_split_frames(source_task) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_t = source_task.get_table("train", mask_input_cols=False)
    val_t = source_task.get_table("val", mask_input_cols=False)
    test_t = source_task.get_table("test", mask_input_cols=False)
    return (
        train_t.df.reset_index(drop=True),
        val_t.df.reset_index(drop=True),
        test_t.df.reset_index(drop=True),
    )


def _build_ordered_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_index: pd.Index,
) -> TemporalSplit:
    n_tr = len(train_df)
    n_va = len(val_df)
    n_te = len(test_df)
    train_mask = pd.Series([True] * n_tr + [False] * (n_va + n_te), index=task_index)
    val_mask = pd.Series([False] * n_tr + [True] * n_va + [False] * n_te, index=task_index)
    test_mask = pd.Series([False] * (n_tr + n_va) + [True] * n_te, index=task_index)
    return TemporalSplit(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def _build_database_spec(source_task) -> DatabaseSpec:
    source_db = source_task.dataset.get_db()
    tables: Dict[str, TableSpec] = {}
    for name, tbl in source_db.table_dict.items():
        tables[name] = TableSpec(
            name=name,
            df=tbl.df.copy(),
            primary_key=tbl.pkey_col,
            time_col=tbl.time_col,
            foreign_keys=dict(tbl.fkey_col_to_pkey_table),
        )
    return DatabaseSpec(tables=tables)


def _task_type_metadata(task_type, task_type_cls) -> tuple[str, str]:
    if task_type == task_type_cls.BINARY_CLASSIFICATION:
        return "classification", "auroc"
    return "regression", "mae"
