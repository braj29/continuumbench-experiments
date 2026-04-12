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
    GraphSAGEAdapter,
    GraphifiedSklearnAdapter,
    OfficialRelationalTransformerAdapter,
    RGCNAdapter,
    RelGTSubprocessAdapter,
    SklearnTabularAdapter,
    ULTRASubprocessAdapter,
    build_tabular_estimator,
    default_max_train_rows,
    resolve_tabicl_device,
    resolve_tabpfn_device,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_K_VALUES = (100, 300, 500)
LEGACY_GRAPH_VIEW_NAME = "graphified"
GRAPH_PROXY_VIEW_NAME = "structural-count-proxy"

# Graph model choices exposed via --graph-model
GRAPH_MODEL_CHOICES = ("count-proxy", "rgcn", "graphsage", "relgt", "ultra")


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
            "Run tabular and graph models on the ContinuumBench tri-view harness. "
            "Joined-table tracks use tabicl/tabpfn/xgboost; graph track supports "
            "count-proxy, rgcn, graphsage, relgt, and ultra."
        )
    )
    _add_problem_args(parser)
    _add_model_args(parser)
    _add_graph_model_args(parser)
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
        help="Comma-separated list from {tabicl,tabpfn,xgboost}.",
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
    parser.add_argument(
        "--relational-only",
        action="store_true",
        help=(
            "Run only the relational view in Protocol A (skip joined-table and graph tracks). "
            "Protocol B is skipped in this mode."
        ),
    )


def _add_graph_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph-model",
        type=str,
        default="count-proxy",
        choices=GRAPH_MODEL_CHOICES,
        help=(
            "Graph-view model for Protocol A/B. "
            "'count-proxy': structural degree-count features + tabfm (default, no extra deps). "
            "'rgcn': in-process RGCN (requires torch-geometric). "
            "'graphsage': in-process GraphSAGE (requires torch-geometric). "
            "'relgt': RelGT subprocess (requires --relgt-repo-path + Linux/CUDA). "
            "'ultra': ULTRA subprocess (requires --ultra-repo-path + task adaptation)."
        ),
    )
    # GNN hyperparameters (used by rgcn / graphsage)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-max-epochs", type=int, default=200)
    parser.add_argument("--gnn-patience", type=int, default=20)
    parser.add_argument("--gnn-lr", type=float, default=1e-3)
    parser.add_argument("--gnn-batch-size", type=int, default=64)
    parser.add_argument("--gnn-neighborhood-k", type=int, default=None,
                        help="Neighbourhood cap per event table for GNN subgraphs.")
    # RelGT subprocess args
    parser.add_argument(
        "--relgt-repo-path",
        type=str,
        default=None,
        help="Path to cloned RelGT repo (arXiv:2505.10960). Required for --graph-model relgt.",
    )
    parser.add_argument("--relgt-num-neighbors", type=int, default=512)
    parser.add_argument("--relgt-epochs", type=int, default=20)
    parser.add_argument("--relgt-hidden-channels", type=int, default=128)
    parser.add_argument(
        "--relgt-python-executable",
        type=str,
        default=None,
        help="Python executable for the RelGT subprocess (defaults to current interpreter).",
    )
    # ULTRA subprocess args
    parser.add_argument(
        "--ultra-repo-path",
        type=str,
        default=None,
        help=(
            "Path to cloned ULTRA repo (https://github.com/DeepGraphLearning/ULTRA). "
            "Required for --graph-model ultra. Note: requires task adaptation for "
            "entity property prediction (ULTRA is designed for KG link prediction)."
        ),
    )
    parser.add_argument("--ultra-epochs", type=int, default=10)
    parser.add_argument("--ultra-config-path", type=str, default=None,
                        help="Path to a ULTRA YAML config file (optional override).")
    parser.add_argument(
        "--ultra-python-executable",
        type=str,
        default=None,
        help="Python executable for the ULTRA subprocess (defaults to current interpreter).",
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
            "Use --models tabpfn or --models xgboost for regression tasks."
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
        seed=args.seed,
    )


def _build_joined_model(model_name: str, args: argparse.Namespace, task: TaskSpec) -> BaseViewModel:
    return SklearnTabularAdapter(
        estimator=_build_estimator(model_name, args, task),
        **_adapter_kwargs(model_name, args),
    )


def _gnn_kwargs(args: argparse.Namespace) -> dict:
    return {
        "hidden_dim": args.gnn_hidden_dim,
        "num_layers": args.gnn_num_layers,
        "max_epochs": args.gnn_max_epochs,
        "patience": args.gnn_patience,
        "lr": args.gnn_lr,
        "batch_size": args.gnn_batch_size,
        "neighborhood_k": args.gnn_neighborhood_k,
        "device": args.device,
    }


def _build_graph_model(model_name: str, args: argparse.Namespace, task: TaskSpec) -> BaseViewModel:
    """Build the graph-track model specified by --graph-model."""
    graph_model = getattr(args, "graph_model", "count-proxy")

    if graph_model == "rgcn":
        return RGCNAdapter(name=f"rgcn", **_gnn_kwargs(args))

    if graph_model == "graphsage":
        return GraphSAGEAdapter(name="graphsage", **_gnn_kwargs(args))

    if graph_model == "relgt":
        repo = args.relgt_repo_path
        if not repo:
            raise ValueError(
                "--relgt-repo-path is required when --graph-model relgt. "
                "Clone the RelGT repo and pass its path."
            )
        return RelGTSubprocessAdapter(
            dataset_name=args.dataset_name,
            task_name=args.task_name,
            target_col=task.target_col,
            relgt_repo_path=repo,
            python_executable=args.relgt_python_executable,
            seed=args.seed,
            num_neighbors=args.relgt_num_neighbors,
            hidden_channels=args.relgt_hidden_channels,
            epochs=args.relgt_epochs,
        )

    if graph_model == "ultra":
        repo = args.ultra_repo_path
        if not repo:
            raise ValueError(
                "--ultra-repo-path is required when --graph-model ultra. "
                "Clone ULTRA from https://github.com/DeepGraphLearning/ULTRA and pass its path."
            )
        return ULTRASubprocessAdapter(
            dataset_name=args.dataset_name,
            task_name=args.task_name,
            target_col=task.target_col,
            ultra_repo_path=repo,
            python_executable=args.ultra_python_executable,
            config_path=args.ultra_config_path,
            seed=args.seed,
            epochs=args.ultra_epochs,
        )

    # Default: count-proxy (structural degree features + tabfm estimator)
    return GraphifiedSklearnAdapter(
        estimator=_build_estimator(model_name, args, task),
        **_adapter_kwargs(model_name, args),
    )


def _build_graph_model_factory(
    model_name: str,
    args: argparse.Namespace,
    task: TaskSpec,
) -> Callable[[int], list[BaseViewModel]]:
    """Factory used by Protocol B graph ablations (K-sweep).

    GNN models (rgcn/graphsage/relgt/ultra) don't use the K parameter the same
    way as count-proxy, so for those the factory returns the same model regardless
    of k; the K-sweep ablation is meaningful only for the count-proxy track.
    """
    def factory(graph_k: int) -> list[BaseViewModel]:
        graph_model = getattr(args, "graph_model", "count-proxy")
        if graph_model == "count-proxy":
            # K is already embedded in the payload by the runner; the factory
            # just produces a fresh adapter whose fit() will read it from there.
            return [
                GraphifiedSklearnAdapter(
                    estimator=_build_estimator(model_name, args, task),
                    **_adapter_kwargs(model_name, args),
                )
            ]
        # For in-process GNN models, override neighborhood_k with the ablation
        # value so the K-sweep tests different subgraph sizes.
        gnn_kw = {**_gnn_kwargs(args), "neighborhood_k": graph_k}
        if graph_model == "rgcn":
            return [RGCNAdapter(name=f"rgcn-K{graph_k}", **gnn_kw)]
        if graph_model == "graphsage":
            return [GraphSAGEAdapter(name=f"graphsage-K{graph_k}", **gnn_kw)]
        # Subprocess models (relgt, ultra) don't support per-run K override.
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
            # Protocol B: lookback window sweep for JT-TemporalAgg (W ∈ {7, 30, 90, None})
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=7),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=30),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=90),
            JoinedTableConfig(view_name="jt_temporalagg", max_path_hops=1, lookback_days=None),
            # Protocol B: join depth sweep for JT-Entity (d ∈ {1, 2})
            JoinedTableConfig(view_name="jt_entity", max_path_hops=1),
            JoinedTableConfig(view_name="jt_entity", max_path_hops=2),
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
    graph_model = getattr(args, "graph_model", "count-proxy")
    if graph_model == "count-proxy":
        print(
            "continuumbench: graph track uses a structural degree-count proxy "
            "(not a full GNN). Use --graph-model rgcn or graphsage for in-process GNNs.",
            file=sys.stderr,
            flush=True,
        )
    elif graph_model in ("rgcn", "graphsage"):
        print(
            f"continuumbench: graph track using in-process {graph_model.upper()} "
            "(requires torch-geometric).",
            file=sys.stderr,
            flush=True,
        )
    elif graph_model == "relgt":
        print(
            "continuumbench: graph track using RelGT subprocess "
            "(requires Linux+CUDA and --relgt-repo-path).",
            file=sys.stderr,
            flush=True,
        )
    elif graph_model == "ultra":
        print(
            "continuumbench: graph track using ULTRA subprocess "
            "(requires --ultra-repo-path and task adaptation for entity prediction).",
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


def _display_view_name(view_name: str) -> str:
    if view_name == LEGACY_GRAPH_VIEW_NAME:
        return GRAPH_PROXY_VIEW_NAME
    return view_name


def _display_representation_key(key: str) -> str:
    return key.replace(LEGACY_GRAPH_VIEW_NAME, GRAPH_PROXY_VIEW_NAME)


def _print_protocol_summary(summary) -> None:
    table = summary.to_frame()[["model_name", "view_name", "metric_value"]].copy()
    table["view_name"] = table["view_name"].map(_display_view_name)
    print(table)


def _print_protocol_ris(summary, protocol_name: str) -> None:
    print(f"{protocol_name} RIS:")
    for task_name, payload in summary.per_task_ris.items():
        ris = float(payload["ris"])
        utility_std = float(payload["utility_std"])
        utility_gap = float(payload["utility_gap"])
        print(
            f"  task={task_name} ris={ris:.6f} "
            f"utility_std={utility_std:.6f} utility_gap={utility_gap:.6f}"
        )
        representation_scores = payload.get("representation_scores", {})
        for key in sorted(representation_scores):
            print(
                f"    {_display_representation_key(str(key))}: "
                f"{float(representation_scores[key]):.6f}"
            )


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
    relational_only = bool(getattr(args, "relational_only", False))

    joined_models = [] if relational_only else [_build_joined_model(model_name, args, problem.task)]
    relational_models = _build_relational_models(args, problem.task)
    graph_models = [] if relational_only else [_build_graph_model(model_name, args, problem.task)]

    protocol_model_name = (
        relational_models[0].name
        if relational_only and relational_models
        else model_name
    )

    _announce_protocol(protocol_model_name, "Protocol A")
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
    _print_protocol_ris(summary_a, "Protocol A")

    manifest = {
        **_task_source_metadata(args),
        "output_dir": str(model_out),
        "relational_only": relational_only,
        **_relational_backend_metadata(args, relational_models),
        "protocol_a": _protocol_artifacts(problem.task.name, "protocol_a"),
    }

    if args.run_protocol_b and not relational_only:
        _announce_protocol(protocol_model_name, "Protocol B")
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
        _print_protocol_ris(summary_b, "Protocol B")
        manifest["protocol_b"] = _protocol_artifacts(problem.task.name, "protocol_b")
    elif args.run_protocol_b and relational_only:
        print(
            "continuumbench: skipping Protocol B in --relational-only mode "
            "(Protocol B only ablates joined-table and graph tracks).",
            file=sys.stderr,
            flush=True,
        )

    return manifest


def _write_manifest(out_root: Path, manifest: dict[str, dict[str, object]]) -> None:
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"=== Wrote manifest: {manifest_path} ===")
    print(json.dumps(manifest, indent=2))


def run(args: argparse.Namespace) -> None:
    models = _parse_models(args.models)
    if args.relational_only and not args.use_official_rt_relational:
        raise ValueError(
            "--relational-only requires --use-official-rt-relational. "
            "This mode is intended for official RT-only benchmarking."
        )
    _emit_runtime_warnings(models, args)

    problem = _load_problem(args)
    _validate_models_for_task(models, problem.task)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    protocol_configs = _default_protocol_configs()

    if args.relational_only:
        selected_model = models[0]
        if len(models) > 1:
            print(
                "continuumbench: --relational-only ignores extra --models entries; "
                f"using first model token {selected_model!r} as output folder key.",
                file=sys.stderr,
                flush=True,
            )
        manifest = {
            "rt-official": _run_model(
                model_name=selected_model,
                args=args,
                problem=problem,
                protocol_configs=protocol_configs,
                out_root=out_root,
            )
        }
    else:
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
