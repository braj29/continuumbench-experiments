from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..continuumbench.examples import make_synthetic_relational_problem
from ..continuumbench.runner import ExperimentRunner
from ..continuumbench.sources import load_dataset_entity_problem
from ..continuumbench.specs import (
    DatabaseSpec,
    ExperimentConfig,
    JoinedTableConfig,
    TaskSpec,
    TemporalSplit,
)
from ..models import (
    SUPPORTED_MODEL_NAMES,
    BaseViewModel,
    ExternalRelationalAdapter,
    GraphifiedSklearnAdapter,
    OfficialRelationalTransformerAdapter,
    SklearnTabularAdapter,
    build_tabular_estimator,
    default_max_train_rows,
    resolve_tabicl_device,
    resolve_tabpfn_device,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_K_VALUES = (100, 300, 500)


@dataclass(frozen=True)
class LoadedProblem:
    db: DatabaseSpec
    task: TaskSpec
    split: TemporalSplit


@dataclass(frozen=True)
class ProtocolConfigs:
    joined_table_cfgs: tuple[JoinedTableConfig, ...]
    joined_ablation_cfgs: tuple[JoinedTableConfig, ...]
    graph_k_values: tuple[int, ...] = DEFAULT_GRAPH_K_VALUES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabICL/TabPFN on the ContinuumBench-style tri-view harness. "
            "Default task source uses a registered dataset/task pair so the joined, "
            "graph-degree, and optional official Relational Transformer tracks target "
            "the same entity task."
        )
    )
    _add_problem_args(parser)
    _add_model_args(parser)
    _add_relational_transformer_args(parser)
    return parser


def _add_problem_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-source",
        type=str,
        choices=["dataset", "synthetic"],
        default="dataset",
        help="dataset: load a registered entity task. synthetic: toy schema only.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rel-f1",
        help="Dataset id when --task-source=dataset.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="driver-top3",
        help="Task id when --task-source=dataset.",
    )
    parser.add_argument(
        "--download-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download task artifacts when needed.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/continuumbench_tabfm")
    parser.add_argument("--seed", type=int, default=7)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        type=str,
        default="tabicl,tabpfn",
        help="Comma-separated list from {tabicl,tabpfn}.",
    )
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
        help="TabICL test predict_proba row chunk size (0 = full test set at once).",
    )
    parser.add_argument(
        "--run-protocol-b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Protocol B ablations in addition to Protocol A.",
    )


def _add_relational_transformer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-official-rt-relational",
        action="store_true",
        help=(
            "Run snap-stanford/relational-transformer on the same dataset-backed task. "
            "Requires --task-source=dataset and a repo path "
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
        help="Override RT dataset id (default: --dataset-name).",
    )
    parser.add_argument(
        "--rt-task-name",
        type=str,
        default=None,
        help="Override RT task id (default: --task-name).",
    )
    parser.add_argument(
        "--rt-target-col",
        type=str,
        default=None,
        help="Override RT target column (default: loaded task target_col).",
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


def _resolved_rt_repo_path(args: argparse.Namespace) -> str | None:
    if args.rt_repo_path:
        return args.rt_repo_path
    default_dir = REPO_ROOT / "relational-transformer"
    if (default_dir / "rt" / "main.py").is_file():
        return str(default_dir)
    return None


def _parse_models(spec: str) -> list[str]:
    models = [model.strip().lower() for model in spec.split(",") if model.strip()]
    if not models:
        raise ValueError("No models requested. Use --models tabicl,tabpfn")
    unknown = [model for model in models if model not in SUPPORTED_MODEL_NAMES]
    if unknown:
        raise ValueError(
            f"Unsupported model(s): {unknown}. Allowed: {sorted(SUPPORTED_MODEL_NAMES)}"
        )
    return list(dict.fromkeys(models))


def _tabicl_runtime_device(args: argparse.Namespace) -> str | None:
    env_device = os.environ.get("TABICL_DEVICE")
    if env_device:
        return env_device
    return resolve_tabicl_device(
        args.tabicl_device,
        platform_name=platform.system(),
    )


def _predict_proba_chunk_size(model_name: str, args: argparse.Namespace) -> int | None:
    if model_name != "tabicl" or args.tabicl_predict_chunk <= 0:
        return None
    return int(args.tabicl_predict_chunk)


def _adapter_kwargs(model_name: str, args: argparse.Namespace) -> dict[str, object]:
    return {
        "name": model_name,
        "max_train_rows": default_max_train_rows(model_name, args.tabular_max_train_rows),
        "subsample_seed": args.seed,
        "predict_proba_chunk_size": _predict_proba_chunk_size(model_name, args),
    }


def _validate_models_for_task(models: list[str], task: TaskSpec) -> None:
    if task.task_type == "regression" and "tabicl" in models:
        raise ValueError(
            "TabICL does not support regression tasks in this harness. "
            "Use --models tabpfn for regression tasks."
        )


def _build_estimator(model_name: str, args: argparse.Namespace, task: TaskSpec):
    tabpfn_device = resolve_tabpfn_device(
        args.device,
        platform_name=platform.system(),
    )
    return build_tabular_estimator(
        model_name,
        task.task_type,
        device=tabpfn_device,
        tabicl_device=_tabicl_runtime_device(args),
        tabicl_use_amp=args.tabicl_use_amp,
        ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
    )


def _build_joined_model(model_name: str, args: argparse.Namespace, task: TaskSpec) -> BaseViewModel:
    return SklearnTabularAdapter(
        estimator=_build_estimator(model_name, args, task),
        **_adapter_kwargs(model_name, args),
    )


def _build_graph_model(model_name: str, args: argparse.Namespace, task: TaskSpec) -> BaseViewModel:
    return GraphifiedSklearnAdapter(
        estimator=_build_estimator(model_name, args, task),
        **_adapter_kwargs(model_name, args),
    )


def _build_graph_model_factory(
    model_name: str,
    args: argparse.Namespace,
    task: TaskSpec,
) -> Callable[[int], list[BaseViewModel]]:
    def factory(_graph_k: int) -> list[BaseViewModel]:
        return [_build_graph_model(model_name, args, task)]

    return factory


def _load_problem(args: argparse.Namespace) -> LoadedProblem:
    if args.task_source == "dataset":
        db, task, split = load_dataset_entity_problem(
            args.dataset_name,
            args.task_name,
            download=args.download_artifacts,
        )
    else:
        db, task, split = make_synthetic_relational_problem()
    return LoadedProblem(db=db, task=task, split=split)


def _build_relational_models(args: argparse.Namespace, task: TaskSpec) -> list[BaseViewModel]:
    if not args.use_official_rt_relational:
        return [
            ExternalRelationalAdapter(
                name="rt-stub",
                fit_fn=stub_relational_fit_fn,
                predict_fn=stub_relational_predict_fn,
            )
        ]

    if args.task_source != "dataset":
        raise ValueError(
            "Official Relational Transformer expects the dataset-backed task source. "
            "Use --task-source dataset (not synthetic) so metrics are comparable."
        )

    rt_path = _resolved_rt_repo_path(args)
    if not rt_path:
        raise ValueError(
            "Official RT needs a checkout of snap-stanford/relational-transformer. "
            "Clone it to ./relational-transformer or pass --rt-repo-path."
        )

    dataset_name = args.rt_dataset_name or args.dataset_name
    task_name = args.rt_task_name or args.task_name
    target_col = args.rt_target_col or task.target_col

    if dataset_name != args.dataset_name or task_name != args.task_name:
        raise ValueError(
            "RT dataset/task overrides must match --dataset-name/--task-name "
            "when benchmarking a single registered task."
        )
    if target_col != task.target_col:
        raise ValueError(
            f"RT target {target_col!r} must match loaded task target "
            f"{task.target_col!r}."
        )

    return [
        OfficialRelationalTransformerAdapter(
            name="rt-official",
            dataset_name=dataset_name,
            task_name=task_name,
            target_col=target_col,
            rt_repo_path=rt_path,
            python_executable=args.rt_python_executable,
            seed=args.seed,
            max_steps=args.rt_max_steps,
            batch_size=args.rt_batch_size,
            num_workers=args.rt_num_workers,
        )
    ]


def _relational_backend_metadata(
    args: argparse.Namespace,
    relational_models: Sequence[BaseViewModel],
) -> dict[str, object]:
    if args.use_official_rt_relational:
        return {
            "relational_backend": "official_relational_transformer",
            "relational_model_names": [model.name for model in relational_models],
            "official_relational_transformer": True,
            "rt_repo_path": _resolved_rt_repo_path(args),
        }
    return {
        "relational_backend": "stub_relational",
        "relational_model_names": [model.name for model in relational_models],
        "official_relational_transformer": False,
        "rt_repo_path": None,
    }


def _default_protocol_configs() -> ProtocolConfigs:
    return ProtocolConfigs(
        joined_table_cfgs=(
            JoinedTableConfig(view_name="jt_entity", max_path_hops=1),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
        ),
        joined_ablation_cfgs=(
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=7),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=None),
        ),
    )


def _emit_runtime_warnings(models: Sequence[str], args: argparse.Namespace) -> None:
    if "tabicl" in models:
        print(
            "continuumbench: TabICL test scoring is slow; progress logs go to stderr (INFO). "
            "On macOS, CPU is used by default (--tabicl-device auto).",
            file=sys.stderr,
            flush=True,
        )
    if args.task_source == "dataset" and not args.use_official_rt_relational:
        print(
            "continuumbench: relational track is using rt-stub, not the official "
            "Relational Transformer. Pass --use-official-rt-relational to benchmark RT.",
            file=sys.stderr,
            flush=True,
        )


def _make_runner(output_dir: Path, seed: int) -> ExperimentRunner:
    return ExperimentRunner(
        ExperimentConfig(
            benchmark_name="continuumbench-v1",
            seed=seed,
            output_dir=str(output_dir),
        )
    )


def _protocol_artifacts(task_name: str, protocol_name: str) -> dict[str, str]:
    return {
        "runs": f"{task_name}__{protocol_name}__runs.parquet",
        "ris": f"{task_name}__{protocol_name}__ris.json",
    }


def _task_source_metadata(args: argparse.Namespace) -> dict[str, object]:
    if args.task_source != "dataset":
        return {"task_source": args.task_source, "dataset_name": None, "task_name": None}
    return {
        "task_source": args.task_source,
        "dataset_name": args.dataset_name,
        "task_name": args.task_name,
    }


def _announce_protocol(model_name: str, protocol_name: str) -> None:
    print(f"=== Running {protocol_name} with {model_name} ===", flush=True)


def _print_protocol_summary(summary) -> None:
    print(summary.to_frame()[["model_name", "view_name", "metric_value"]])


def _run_model(
    model_name: str,
    args: argparse.Namespace,
    problem: LoadedProblem,
    protocol_configs: ProtocolConfigs,
    out_root: Path,
) -> dict[str, object]:
    model_out = out_root / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    runner = _make_runner(model_out, args.seed)
    joined_models = [_build_joined_model(model_name, args, problem.task)]
    relational_models = _build_relational_models(args, problem.task)
    graph_models = [_build_graph_model(model_name, args, problem.task)]

    _announce_protocol(model_name, "Protocol A")
    summary_a = runner.run_protocol_a(
        db=problem.db,
        task=problem.task,
        split=problem.split,
        joined_table_cfgs=list(protocol_configs.joined_table_cfgs),
        joined_models=joined_models,
        relational_models=relational_models,
        graph_models=graph_models,
    )
    _print_protocol_summary(summary_a)

    manifest = {
        **_task_source_metadata(args),
        "output_dir": str(model_out),
        **_relational_backend_metadata(args, relational_models),
        "protocol_a": _protocol_artifacts(problem.task.name, "protocol_a"),
    }

    if args.run_protocol_b:
        _announce_protocol(model_name, "Protocol B")
        summary_b = runner.run_protocol_b(
            db=problem.db,
            task=problem.task,
            split=problem.split,
            joined_ablation_cfgs=list(protocol_configs.joined_ablation_cfgs),
            joined_models=joined_models,
            graph_k_values=list(protocol_configs.graph_k_values),
            graph_models_factory=_build_graph_model_factory(model_name, args, problem.task),
        )
        _print_protocol_summary(summary_b)
        manifest["protocol_b"] = _protocol_artifacts(problem.task.name, "protocol_b")

    return manifest


def _write_manifest(out_root: Path, manifest: dict[str, dict[str, object]]) -> None:
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"=== Wrote manifest: {manifest_path} ===")
    print(json.dumps(manifest, indent=2))


def run(args: argparse.Namespace) -> None:
    models = _parse_models(args.models)
    _emit_runtime_warnings(models, args)

    problem = _load_problem(args)
    _validate_models_for_task(models, problem.task)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    protocol_configs = _default_protocol_configs()

    manifest = {
        model_name: _run_model(
            model_name=model_name,
            args=args,
            problem=problem,
            protocol_configs=protocol_configs,
            out_root=out_root,
        )
        for model_name in models
    }
    _write_manifest(out_root, manifest)


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
