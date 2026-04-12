from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuumbench_experiments.continuumbench.examples import make_synthetic_relational_problem
from continuumbench_experiments.continuumbench.sources import load_dataset_entity_problem
from continuumbench_experiments.continuumbench.specs import JoinedTableConfig
from continuumbench_experiments.continuumbench.views import (
    ViewFactory,
    build_graph_degree_feature_table,
    count_incident_tables,
)


def _parse_optional_int(value: str) -> int | None:
    spec = value.strip().lower()
    if spec in {"none", "null"}:
        return None
    return int(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ContinuumBench joined, structural-count-proxy (graph), and relational views."
        )
    )
    parser.add_argument(
        "--view",
        choices=["joined", "graph", "relational"],
        required=True,
        help="Which ContinuumBench view to inspect.",
    )
    parser.add_argument(
        "--task-source",
        choices=["dataset", "synthetic"],
        default="dataset",
    )
    parser.add_argument("--dataset-name", default="rel-f1")
    parser.add_argument("--task-name", default="driver-top3")
    parser.add_argument(
        "--download-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which task-table split to inspect.",
    )
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "view_inspect",
        help="Where to write parquet/json artifacts for inspection.",
    )

    parser.add_argument(
        "--joined-view",
        choices=["jt_entity", "jt_temporalagg"],
        default="jt_entity",
    )
    parser.add_argument("--max-path-hops", type=int, default=1)
    parser.add_argument(
        "--lookback-days",
        type=_parse_optional_int,
        default=30,
        help="Use an integer or 'none'. Only relevant for jt_temporalagg.",
    )
    parser.add_argument(
        "--include-parent-lookups",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--graph-k",
        type=int,
        default=None,
        help="Approximate neighborhood size. Used to cap incident tables like the benchmark.",
    )
    return parser


def _load_problem(args: argparse.Namespace):
    if args.task_source == "dataset":
        return load_dataset_entity_problem(
            args.dataset_name,
            args.task_name,
            download=args.download_artifacts,
        )
    return make_synthetic_relational_problem()


def _split_mask(split_name: str, split) -> pd.Series:
    if split_name == "train":
        return split.train_mask
    if split_name == "val":
        return split.val_mask
    if split_name == "test":
        return split.test_mask
    return pd.Series([True] * len(split.train_mask), index=split.train_mask.index)


def _task_rows_for_split(task, split, split_name: str) -> pd.DataFrame:
    mask = _split_mask(split_name, split)
    return task.task_table.loc[mask].reset_index(drop=True)


def _graph_max_incident_tables(db, task, graph_k: int | None) -> int | None:
    if graph_k is None:
        return None
    n_candidates = count_incident_tables(db, task)
    return max(1, min(n_candidates, int(graph_k) // 100))


def _print_frame_summary(df: pd.DataFrame, head: int) -> None:
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("columns:")
    for column in df.columns:
        print(f"  {column}")
    print()
    print(df.head(head).to_string())


def _inspect_joined(args: argparse.Namespace, db, task, split) -> Path:
    cfg = JoinedTableConfig(
        view_name=args.joined_view,
        lookback_days=args.lookback_days if args.joined_view == "jt_temporalagg" else None,
        max_path_hops=args.max_path_hops,
        include_parent_lookups=args.include_parent_lookups,
    )
    view_factory = ViewFactory(db, task)
    joined_df = view_factory.build_joined_table(cfg)
    out_df = joined_df if args.split == "all" else _task_rows_for_split(
        type("TaskWrapper", (), {"task_table": joined_df})(),
        split,
        args.split,
    )
    _print_frame_summary(out_df, args.head)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{task.name}__{cfg.view_name}__{args.split}.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_path


def _inspect_graph(args: argparse.Namespace, db, task, split) -> Path:
    rows = _task_rows_for_split(task, split, args.split) if args.split != "all" else task.task_table
    max_incident_tables = _graph_max_incident_tables(db, task, args.graph_k)
    graph_df = build_graph_degree_feature_table(
        db=db,
        task=task,
        rows=rows.reset_index(drop=True),
        max_incident_tables=max_incident_tables,
    )
    _print_frame_summary(graph_df, args.head)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"k{args.graph_k}" if args.graph_k is not None else "full"
    out_path = args.output_dir / f"{task.name}__graph__{suffix}__{args.split}.parquet"
    graph_df.to_parquet(out_path, index=False)
    return out_path


def _inspect_relational(args: argparse.Namespace, db, task, split) -> Path:
    split_rows = {
        "train": int(split.train_mask.sum()),
        "val": int(split.val_mask.sum()),
        "test": int(split.test_mask.sum()),
    }
    payload: dict[str, Any] = {
        "task_name": task.name,
        "task_type": task.task_type,
        "target_col": task.target_col,
        "metric_name": task.metric_name,
        "entity_table": task.entity_table,
        "entity_key": task.entity_key,
        "seed_time_col": task.seed_time_col,
        "task_rows_total": int(len(task.task_table)),
        "task_rows_by_split": split_rows,
        "tables": {},
    }
    for table_name, table in db.tables.items():
        payload["tables"][table_name] = {
            "rows": int(len(table.df)),
            "columns": list(table.df.columns),
            "primary_key": table.primary_key,
            "time_col": table.time_col,
            "foreign_keys": dict(table.foreign_keys),
        }

    print(json.dumps(payload, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{task.name}__relational_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    args = _build_parser().parse_args(argv)
    db, task, split = _load_problem(args)

    if args.view == "joined":
        out_path = _inspect_joined(args, db, task, split)
    elif args.view == "graph":
        out_path = _inspect_graph(args, db, task, split)
    else:
        out_path = _inspect_relational(args, db, task, split)

    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
