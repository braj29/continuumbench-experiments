from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl import TabICLClassifier  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

from continuumbench_relbench import (  # noqa: E402
    BaseViewModel,
    ExperimentConfig,
    ExperimentRunner,
    ExternalRelationalAdapter,
    GraphifiedSklearnAdapter,
    JoinedTableConfig,
    OfficialRelationalTransformerAdapter,
    SklearnTabularAdapter,
    TaskSpec,
    load_relbench_entity_problem,
    make_synthetic_relational_problem,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)
from model import build_tabpfn  # noqa: E402

TABICL_AUTO_MAX_TRAIN_ROWS = 50_000


def _resolved_rt_repo_path(args: argparse.Namespace) -> str | None:
    if args.rt_repo_path:
        return args.rt_repo_path
    default_dir = PROJECT_ROOT / "relational-transformer"
    if (default_dir / "rt" / "main.py").is_file():
        return str(default_dir)
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabICL/TabPFN on the ContinuumBench-style tri-view harness. "
            "Default data source is RelBench so the joined, graph-degree, and optional "
            "official Relational Transformer tracks target the same entity task."
        )
    )
    parser.add_argument(
        "--benchmark-source",
        type=str,
        choices=["relbench", "synthetic"],
        default="relbench",
        help="relbench: load a RelBench entity task (recommended). synthetic: toy schema only.",
    )
    parser.add_argument(
        "--relbench-dataset",
        type=str,
        default="rel-f1",
        help="RelBench dataset name when --benchmark-source=relbench.",
    )
    parser.add_argument(
        "--relbench-task",
        type=str,
        default="driver-top3",
        help="RelBench entity task name when --benchmark-source=relbench.",
    )
    parser.add_argument(
        "--relbench-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download RelBench artifacts when needed.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tabicl,tabpfn",
        help="Comma-separated list from {tabicl,tabpfn}.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/continuumbench_tabfm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--tabular-max-train-rows",
        type=int,
        default=None,
        help="Cap training rows for joined/graph TabFM fits (default: 50k for tabicl only).",
    )
    parser.add_argument(
        "--tabpfn-ignore-pretraining-limits",
        action="store_true",
        help="Pass ignore_pretraining_limits=True to TabPFN.",
    )
    parser.add_argument(
        "--tabicl-device",
        type=str,
        default="auto",
        help=(
            "TabICL torch device: 'auto' uses CPU on macOS (avoids slow/hung MPS for many setups), "
            "else TabICL default. Override with cpu, cuda, mps, ...; TABICL_DEVICE env wins."
        ),
    )
    parser.add_argument(
        "--tabicl-use-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TabICL automatic mixed precision (default off for stability, especially on CPU).",
    )
    parser.add_argument(
        "--tabicl-predict-chunk",
        type=int,
        default=64,
        help="TabICL test predict_proba row chunk size (0 = one predict_proba on full test set).",
    )
    parser.add_argument(
        "--run-protocol-b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Protocol B ablations in addition to Protocol A.",
    )
    parser.add_argument(
        "--use-official-rt-relational",
        action="store_true",
        help=(
            "Run snap-stanford/relational-transformer on the same RelBench task. "
            "Requires --benchmark-source=relbench and a repo path "
            "(see --rt-repo-path; default ./relational-transformer if present)."
        ),
    )
    parser.add_argument(
        "--rt-repo-path",
        type=str,
        default=None,
        help=(
            "Path to cloned relational-transformer repo. "
            "If omitted, uses ./relational-transformer when rt/main.py exists."
        ),
    )
    parser.add_argument(
        "--rt-dataset-name",
        type=str,
        default=None,
        help="Override RT dataset id (default: --relbench-dataset).",
    )
    parser.add_argument(
        "--rt-task-name",
        type=str,
        default=None,
        help="Override RT task id (default: --relbench-task).",
    )
    parser.add_argument(
        "--rt-target-col",
        type=str,
        default=None,
        help="Override RT target column (default: RelBench task target_col).",
    )
    parser.add_argument("--rt-max-steps", type=int, default=2**12 + 1)
    parser.add_argument("--rt-batch-size", type=int, default=32)
    parser.add_argument("--rt-num-workers", type=int, default=8)
    parser.add_argument(
        "--rt-python-executable",
        type=str,
        default=None,
        help=(
            "Python executable for the official RT subprocess "
            "(defaults to the current interpreter)."
        ),
    )
    return parser.parse_args(argv)


def _parse_models(spec: str) -> list[str]:
    allowed = {"tabicl", "tabpfn"}
    models = [m.strip().lower() for m in spec.split(",") if m.strip()]
    if not models:
        raise ValueError("No models requested. Use --models tabicl,tabpfn")
    bad = [m for m in models if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported model(s): {bad}. Allowed: {sorted(allowed)}")
    return list(dict.fromkeys(models))


def _tabular_max_train_rows(model_name: str, args: argparse.Namespace) -> int | None:
    if args.tabular_max_train_rows is not None:
        return args.tabular_max_train_rows
    if model_name == "tabicl":
        return TABICL_AUTO_MAX_TRAIN_ROWS
    return None


def _resolve_tabicl_device(spec: str) -> str | None:
    s = (spec or "auto").strip().lower()
    if s == "auto":
        return "cpu" if platform.system() == "Darwin" else None
    if s in ("none", "default"):
        return None
    return spec.strip()


def _tabicl_predict_chunk(model_name: str, args: argparse.Namespace) -> int | None:
    if model_name != "tabicl":
        return None
    if args.tabicl_predict_chunk <= 0:
        return None
    return int(args.tabicl_predict_chunk)


def _validate_models_for_task(models: list[str], task: TaskSpec) -> None:
    if task.task_type == "regression" and "tabicl" in models:
        raise ValueError(
            "TabICL does not support regression tasks in this harness. "
            "Use --models tabpfn for regression RelBench tasks (e.g. driver-position)."
        )


def _build_estimator(model_name: str, args: argparse.Namespace, task: TaskSpec):
    if task.task_type == "regression":
        if model_name != "tabpfn":
            raise ValueError("Regression tasks require tabpfn in --models.")
        return TabPFNRegressor(
            n_estimators=16,
            device=args.device,
            ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
        )
    if model_name == "tabicl":
        env_dev = os.environ.get("TABICL_DEVICE")
        dev = env_dev or _resolve_tabicl_device(args.tabicl_device)
        kw: dict = {
            "n_estimators": 16,
            "use_hierarchical": True,
            "checkpoint_version": "tabicl-classifier-v1.1-0506.ckpt",
            "n_jobs": 1,
            "use_amp": args.tabicl_use_amp,
            "verbose": False,
        }
        if dev:
            kw["device"] = dev
        return TabICLClassifier(**kw)
    if model_name == "tabpfn":
        return build_tabpfn(
            device=args.device,
            ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _build_joined_model(model_name: str, args: argparse.Namespace, task: TaskSpec) -> BaseViewModel:
    cap = _tabular_max_train_rows(model_name, args)
    estimator = _build_estimator(model_name, args, task)
    return SklearnTabularAdapter(
        estimator=estimator,
        name=model_name,
        max_train_rows=cap,
        subsample_seed=args.seed,
        predict_proba_chunk_size=_tabicl_predict_chunk(model_name, args),
    )


def _build_graph_model(model_name: str, args: argparse.Namespace, task: TaskSpec) -> BaseViewModel:
    cap = _tabular_max_train_rows(model_name, args)
    estimator = _build_estimator(model_name, args, task)
    return GraphifiedSklearnAdapter(
        estimator=estimator,
        name=model_name,
        max_train_rows=cap,
        subsample_seed=args.seed,
        predict_proba_chunk_size=_tabicl_predict_chunk(model_name, args),
    )


def _load_problem(args: argparse.Namespace) -> tuple[object, TaskSpec, object]:
    if args.benchmark_source == "relbench":
        return load_relbench_entity_problem(
            args.relbench_dataset,
            args.relbench_task,
            download=args.relbench_download,
        )
    return make_synthetic_relational_problem()


def _relational_models_for_run(args: argparse.Namespace, task: TaskSpec) -> list[BaseViewModel]:
    if args.use_official_rt_relational:
        if args.benchmark_source != "relbench":
            raise ValueError(
                "Official Relational Transformer is trained on RelBench tasks. "
                "Use --benchmark-source relbench (not synthetic) so metrics are comparable."
            )
        rt_path = _resolved_rt_repo_path(args)
        if not rt_path:
            raise ValueError(
                "Official RT needs a checkout of snap-stanford/relational-transformer. "
                "Clone it to ./relational-transformer or pass --rt-repo-path."
            )
        ds = args.rt_dataset_name or args.relbench_dataset
        tn = args.rt_task_name or args.relbench_task
        target = args.rt_target_col or task.target_col
        if ds != args.relbench_dataset or tn != args.relbench_task:
            raise ValueError(
                "RT dataset/task overrides must match --relbench-dataset/--relbench-task "
                "when benchmarking a single RelBench problem."
            )
        if target != task.target_col:
            raise ValueError(
                f"RT target {target!r} must match loaded task target {task.target_col!r}."
            )
        return [
            OfficialRelationalTransformerAdapter(
                name="rt-official",
                dataset_name=ds,
                task_name=tn,
                target_col=target,
                rt_repo_path=rt_path,
                python_executable=args.rt_python_executable,
                seed=args.seed,
                max_steps=args.rt_max_steps,
                batch_size=args.rt_batch_size,
                num_workers=args.rt_num_workers,
            )
        ]
    return [
        ExternalRelationalAdapter(
            name="rt-stub",
            fit_fn=stub_relational_fit_fn,
            predict_fn=stub_relational_predict_fn,
        )
    ]


def _relational_backend_metadata(
    args: argparse.Namespace,
    relational_models: Sequence[BaseViewModel],
) -> dict[str, object]:
    model_names = [model.name for model in relational_models]
    if args.use_official_rt_relational:
        return {
            "relational_backend": "official_relational_transformer",
            "relational_model_names": model_names,
            "official_relational_transformer": True,
            "rt_repo_path": _resolved_rt_repo_path(args),
        }
    return {
        "relational_backend": "stub_relational",
        "relational_model_names": model_names,
        "official_relational_transformer": False,
        "rt_repo_path": None,
    }


def run(args: argparse.Namespace) -> None:
    models = _parse_models(args.models)
    if "tabicl" in models:
        print(
            "continuumbench: TabICL test scoring is slow; progress logs go to stderr (INFO). "
            "On macOS, CPU is used by default (--tabicl-device auto).",
            file=sys.stderr,
            flush=True,
        )
    if args.benchmark_source == "relbench" and not args.use_official_rt_relational:
        print(
            "continuumbench: relational track is using rt-stub, not the official "
            "Relational Transformer. Pass --use-official-rt-relational to benchmark RT.",
            file=sys.stderr,
            flush=True,
        )
    db, task, split = _load_problem(args)
    _validate_models_for_task(models, task)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    joined_cfgs = [
        JoinedTableConfig(view_name="jt_entity", max_path_hops=1),
        JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
    ]
    joined_ablation_cfgs = [
        JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=7),
        JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
        JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=None),
    ]

    manifest: dict[str, dict[str, object]] = {}

    for model_name in models:
        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        runner = ExperimentRunner(
            ExperimentConfig(
                benchmark_name="continuumbench-v1",
                seed=args.seed,
                output_dir=str(model_out),
            )
        )
        joined_models = [_build_joined_model(model_name, args, task)]
        relational_models = _relational_models_for_run(args, task)
        graph_models = [_build_graph_model(model_name, args, task)]
        relational_backend_meta = _relational_backend_metadata(args, relational_models)

        print(f"=== Running Protocol A with {model_name} ===", flush=True)
        summary_a = runner.run_protocol_a(
            db=db,
            task=task,
            split=split,
            joined_table_cfgs=joined_cfgs,
            joined_models=joined_models,
            relational_models=relational_models,
            graph_models=graph_models,
        )
        print(summary_a.to_frame()[["model_name", "view_name", "metric_value"]])

        model_manifest: dict[str, object] = {
            "benchmark_source": args.benchmark_source,
            "relbench_dataset": args.relbench_dataset
            if args.benchmark_source == "relbench"
            else None,
            "relbench_task": args.relbench_task if args.benchmark_source == "relbench" else None,
            "output_dir": str(model_out),
            **relational_backend_meta,
            "protocol_a": {
                "runs": f"{task.name}__protocol_a__runs.parquet",
                "ris": f"{task.name}__protocol_a__ris.json",
            },
        }

        if args.run_protocol_b:
            print(f"=== Running Protocol B with {model_name} ===")
            summary_b = runner.run_protocol_b(
                db=db,
                task=task,
                split=split,
                joined_ablation_cfgs=joined_ablation_cfgs,
                joined_models=joined_models,
                graph_k_values=[100, 300, 500],
                graph_models_factory=lambda k: [
                    GraphifiedSklearnAdapter(
                        estimator=_build_estimator(model_name, args, task),
                        name=model_name,
                        max_train_rows=_tabular_max_train_rows(model_name, args),
                        subsample_seed=args.seed,
                        predict_proba_chunk_size=_tabicl_predict_chunk(model_name, args),
                    )
                ],
            )
            print(summary_b.to_frame()[["model_name", "view_name", "metric_value"]])
            model_manifest["protocol_b"] = {
                "runs": f"{task.name}__protocol_b__runs.parquet",
                "ris": f"{task.name}__protocol_b__ris.json",
            }

        manifest[model_name] = model_manifest

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"=== Wrote manifest: {manifest_path} ===")
    print(json.dumps(manifest, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    args = parse_args(argv)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
