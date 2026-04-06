"""Run TabFM experiments on GraphBench datasets."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from graphbench_adapter import (
    extract_edge_samples,
    extract_graph_samples,
    extract_node_samples,
    iter_graphbench_data,
    load_graphbench_evaluator,
    load_graphbench_splits,
)
from model import build_tabicl, build_tabpfn

TABICL_AUTO_MAX_TRAIN_ROWS = 50_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tabicl",
        choices=["tabicl", "tabpfn", "tabdpt", "mlp", "logreg", "gcn"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="node",
        choices=["node", "edge", "graph"],
        help="GraphBench task regime to convert into tabular samples.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Override task type; auto infers from labels.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="sklearn",
        choices=["sklearn", "graphbench"],
        help="Metric backend: sklearn (default) or GraphBench Evaluator.",
    )
    parser.add_argument(
        "--graphbench-metric-name",
        type=str,
        default=None,
        help="Override GraphBench metric name (e.g., bluesky, sat_as).",
    )
    parser.add_argument("--dataset", type=str, required=True, help="GraphBench dataset name.")
    parser.add_argument(
        "--root",
        type=str,
        default="./graphbench_data",
        help="Root directory to cache GraphBench datasets.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=300,
        help="Download timeout in seconds for GraphBench assets.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=10,
        help="Download retries for GraphBench assets.",
    )
    parser.add_argument(
        "--tabular-mode",
        type=str,
        default="raw",
        choices=["raw", "stats"],
        help="Raw uses IDs + raw attributes; stats derives graph-level aggregates.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--allow-graph-features",
        action="store_true",
        help="Allow graph-derived features (stats mode or degree features).",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Save tabular train/valid/test splits as compressed CSVs.",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Disable saving predictions to artifacts.",
    )
    parser.add_argument(
        "--tabdpt-path",
        type=str,
        default=None,
        help="Path to local TabDPT inference repo (adds to PYTHONPATH).",
    )
    parser.add_argument(
        "--tabdpt-weights",
        type=str,
        default=None,
        help="Optional local path to TabDPT weights.",
    )
    parser.add_argument(
        "--tabdpt-n-ensembles",
        type=int,
        default=8,
        help="Number of TabDPT ensembles for inference.",
    )
    parser.add_argument(
        "--tabdpt-temperature",
        type=float,
        default=0.8,
        help="Softmax temperature for TabDPT.",
    )
    parser.add_argument(
        "--tabdpt-context-size",
        type=int,
        default=2048,
        help="Context size for TabDPT.",
    )
    parser.add_argument(
        "--tabdpt-permute-classes",
        action="store_true",
        help="Permute class labels across ensembles for TabDPT.",
    )
    parser.add_argument(
        "--tabpfn-ignore-pretraining-limits",
        action="store_true",
        help="Allow TabPFN to fit datasets larger than its pretraining limits.",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Optional cap on classification label count (rare labels -> __other__).",
    )
    parser.add_argument(
        "--cast-labels-to-str",
        action="store_true",
        help="Cast classification labels to strings (helps with continuous labels).",
    )
    parser.add_argument(
        "--no-filter-unseen",
        action="store_true",
        help="Keep validation/test rows with labels not seen in training.",
    )
    parser.add_argument(
        "--max-node-features",
        type=int,
        default=64,
        help="Max node features used when converting graphs to tabular rows.",
    )
    parser.add_argument(
        "--max-edge-features",
        type=int,
        default=16,
        help="Max edge features used for graph-level aggregation.",
    )
    parser.add_argument(
        "--add-degree-feature",
        action="store_true",
        help="Append node degree to node/edge features.",
    )
    parser.add_argument(
        "--no-edge-attr",
        action="store_true",
        help="Disable edge attributes when building edge features.",
    )
    parser.add_argument(
        "--auto-edge-labels",
        action="store_true",
        help="Generate edge labels from edge_index when missing.",
    )
    parser.add_argument(
        "--edge-neg-ratio",
        type=float,
        default=1.0,
        help="Negative/positive edge ratio when auto-labeling.",
    )
    parser.add_argument(
        "--edge-sample-cap",
        type=int,
        default=None,
        help="Cap positives per graph when auto-labeling edges.",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for the GCN baseline.",
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=50,
        help="Training epochs for the GCN baseline.",
    )
    parser.add_argument(
        "--gnn-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the GCN baseline.",
    )
    parser.add_argument(
        "--gnn-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the GCN baseline.",
    )
    parser.add_argument(
        "--gnn-dropout",
        type=float,
        default=0.2,
        help="Dropout for the GCN baseline.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Column index to pick for multi-target labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="graphbench_metrics.json",
        help="Path to write metrics JSON.",
    )
    return parser.parse_args(argv)


def run_graphbench(args: argparse.Namespace) -> None:
    start = time.time()
    _set_seed(args.seed)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(output_dir, args)
    print("=== Starting GraphBench experiment ===")
    print(f"Model: {args.model} | task={args.task} | dataset={args.dataset}")

    try:
        splits = load_graphbench_splits(
            root=args.root,
            dataset_names=args.dataset,
            download_timeout=args.download_timeout,
            download_retries=args.download_retries,
        )
        if args.model == "gcn":
            _run_gcn_baseline(args, splits, output_path, output_dir, start)
            return
        _validate_graph_feature_policy(args)
        results: dict[str, dict] = {}

        for split in splits:
            print(f"=== Loading dataset: {split.name} ===")
            allow_multi_target = args.metrics == "graphbench" and args.target_index is None
            X_train, y_train = _extract_split(split.train, args, allow_multi_target)
            X_valid, y_valid = _extract_split(split.valid, args, allow_multi_target)
            X_test, y_test = _extract_split(split.test, args, allow_multi_target)

            X_train, y_train = _subsample(
                X_train, y_train, args.max_train, args.max_samples, args.seed, "train"
            )
            X_valid, y_valid = _subsample(
                X_valid, y_valid, args.max_valid, args.max_samples, args.seed + 1, "valid"
            )
            X_test, y_test = _subsample(
                X_test, y_test, args.max_test, args.max_samples, args.seed + 2, "test"
            )
            X_train, y_train = _maybe_auto_cap_tabicl_train(
                X_train,
                y_train,
                model_name=args.model,
                max_train=args.max_train,
                max_samples=args.max_samples,
                seed=args.seed,
            )

            task_type = _resolve_task_type(args.task_type, y_train)
            y_train, y_valid, y_test = _prepare_labels(
                y_train,
                y_valid,
                y_test,
                task_type=task_type,
                max_classes=args.max_classes,
                cast_labels=args.cast_labels_to_str,
            )
            if task_type == "classification" and not args.no_filter_unseen:
                seen = set(y_train.unique())
                X_valid, y_valid = _filter_unseen_rows(X_valid, y_valid, seen, "valid")
                X_test, y_test = _filter_unseen_rows(X_test, y_test, seen, "test")

            metrics: dict[str, dict] = {}
            evaluator = None
            if args.metrics == "graphbench":
                metric_name = args.graphbench_metric_name or _infer_metric_name(
                    split.name, task_type
                )
                if metric_name is None:
                    raise ValueError(
                        "Unable to infer GraphBench metric name. "
                        "Pass --graphbench-metric-name explicitly."
                    )
                evaluator = load_graphbench_evaluator(metric_name)
                _validate_graphbench_targets(evaluator, y_train, args.target_index)

            if task_type == "regression" and _is_multitarget(y_train):
                print("=== Fitting multi-target regressors ===")
                pred_val, pred_test = _fit_predict_multi_target(
                    X_train,
                    y_train,
                    X_valid if not X_valid.empty else None,
                    X_test if not X_test.empty else None,
                    model_name=args.model,
                    device=args.device,
                    tabdpt_path=args.tabdpt_path,
                    tabdpt_weights=args.tabdpt_weights,
                    tabdpt_n_ensembles=args.tabdpt_n_ensembles,
                    tabdpt_context_size=args.tabdpt_context_size,
                    tabpfn_ignore_limits=args.tabpfn_ignore_pretraining_limits,
                    seed=args.seed,
                )
                if not X_valid.empty and pred_val is not None:
                    metrics["val"] = _evaluate_multitarget(
                        pred_val, y_valid, evaluator, args.metrics
                    )
                if not X_test.empty and pred_test is not None:
                    metrics["test"] = _evaluate_multitarget(
                        pred_test, y_test, evaluator, args.metrics
                    )
                if not args.no_save_predictions:
                    if pred_val is not None and not X_valid.empty:
                        _save_prediction_bundle(
                            output_dir,
                            split.name,
                            "val",
                            {"pred": np.asarray(pred_val), "y_true": _as_numpy(y_valid)},
                        )
                    if pred_test is not None and not X_test.empty:
                        _save_prediction_bundle(
                            output_dir,
                            split.name,
                            "test",
                            {"pred": np.asarray(pred_test), "y_true": _as_numpy(y_test)},
                        )
            else:
                model = _build_model(args.model, task_type, args)
                if args.model == "tabicl":
                    _fit_with_heartbeat(model, X_train, y_train, model_name=args.model)
                else:
                    print("=== Fitting ===")
                    model.fit(X_train, y_train)
                if not X_valid.empty:
                    metrics["val"] = _evaluate(
                        model, X_valid, y_valid, task_type, evaluator, args.metrics
                    )
                if not X_test.empty:
                    metrics["test"] = _evaluate(
                        model, X_test, y_test, task_type, evaluator, args.metrics
                    )
                if not args.no_save_predictions:
                    _save_model_predictions(
                        output_dir,
                        split.name,
                        "val",
                        model,
                        X_valid,
                        y_valid,
                        task_type,
                        evaluator,
                        args.metrics,
                    )
                    _save_model_predictions(
                        output_dir,
                        split.name,
                        "test",
                        model,
                        X_test,
                        y_test,
                        task_type,
                        evaluator,
                        args.metrics,
                    )

            results[split.name] = {
                "task": args.task,
                "task_type": task_type,
                "train_rows": len(X_train),
                "valid_rows": len(X_valid),
                "test_rows": len(X_test),
                "metrics_backend": args.metrics,
                "graphbench_metric": getattr(evaluator, "name", None) if evaluator else None,
                "metrics": metrics,
            }
            if args.save_splits:
                _save_split_table(output_dir, split.name, "train", X_train, y_train)
                _save_split_table(output_dir, split.name, "valid", X_valid, y_valid)
                _save_split_table(output_dir, split.name, "test", X_test, y_test)

        payload = {
            "model": args.model,
            "device": args.device,
            "dataset": args.dataset,
            "results": results,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"=== Wrote metrics to {output_path} ===")
        print(json.dumps(payload, indent=2))
    except Exception:
        print("=== GraphBench experiment failed ===")
        traceback.print_exc()
        raise


def _extract_split(
    dataset: object | None,
    args: argparse.Namespace,
    allow_multi_target: bool,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    if args.task == "node":
        return extract_node_samples(
            dataset,
            max_node_features=args.max_node_features,
            add_degree=args.add_degree_feature,
            target_index=args.target_index,
            tabular_mode=args.tabular_mode,
            allow_multi_target=allow_multi_target,
        )
    if args.task == "edge":
        return extract_edge_samples(
            dataset,
            max_node_features=args.max_node_features,
            include_edge_attr=not args.no_edge_attr,
            add_degree=args.add_degree_feature,
            target_index=args.target_index,
            tabular_mode=args.tabular_mode,
            allow_multi_target=allow_multi_target,
            auto_edge_labels=args.auto_edge_labels,
            edge_neg_ratio=args.edge_neg_ratio,
            edge_sample_cap=args.edge_sample_cap,
            edge_seed=args.seed,
        )
    return extract_graph_samples(
        dataset,
        max_node_features=args.max_node_features,
        max_edge_features=args.max_edge_features,
        target_index=args.target_index,
        tabular_mode=args.tabular_mode,
        allow_multi_target=allow_multi_target,
    )


def _subsample(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    split_cap: int | None,
    global_cap: int | None,
    seed: int,
    split_label: str,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    cap = split_cap if split_cap is not None else global_cap
    if cap is None or len(X) <= cap:
        return X, y
    print(f"=== Subsample {split_label}: {len(X)} -> {cap} ===")
    idx = np.random.default_rng(seed).choice(len(X), size=cap, replace=False)
    y_sel = y.iloc[idx].reset_index(drop=True)
    return X.iloc[idx].reset_index(drop=True), y_sel


def _maybe_auto_cap_tabicl_train(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    model_name: str,
    max_train: int | None,
    max_samples: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    if model_name != "tabicl":
        return X, y
    if max_train is not None or max_samples is not None:
        return X, y
    if len(X) <= TABICL_AUTO_MAX_TRAIN_ROWS:
        return X, y
    print(
        "=== Auto-subsample train for tabicl: "
        f"{len(X)} -> {TABICL_AUTO_MAX_TRAIN_ROWS} "
        "(override with --max-train / --max-samples) ==="
    )
    return _subsample(
        X,
        y,
        split_cap=TABICL_AUTO_MAX_TRAIN_ROWS,
        global_cap=None,
        seed=seed,
        split_label="train(tabicl_auto)",
    )


def _fit_with_heartbeat(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    model_name: str,
    heartbeat_seconds: int = 30,
) -> None:
    print(
        f"=== Fitting {model_name} on "
        f"{len(X_train)} rows x {X_train.shape[1]} features ==="
    )
    start = time.time()
    error: BaseException | None = None
    error_tb = ""
    done = threading.Event()

    def _runner() -> None:
        nonlocal error, error_tb
        try:
            model.fit(X_train, y_train)
        except BaseException as exc:  # pragma: no cover - passthrough from worker thread
            error = exc
            error_tb = traceback.format_exc()
        finally:
            done.set()

    worker = threading.Thread(target=_runner, name=f"{model_name}-fit", daemon=True)
    worker.start()

    while not done.wait(timeout=heartbeat_seconds):
        elapsed = time.time() - start
        print(f"=== Fitting {model_name} in progress (elapsed={elapsed:.1f}s) ===")

    worker.join()
    if error is not None:
        print("=== Fitting failed ===")
        if error_tb:
            print(error_tb)
        raise error
    print(f"=== Fitting finished in {time.time() - start:.2f}s ===")


def _resolve_task_type(task_type: str, y_train: pd.Series | pd.DataFrame) -> str:
    if task_type != "auto":
        return task_type
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] > 1:
            return "regression"
        y_series = y_train.iloc[:, 0]
        return _resolve_task_type("auto", y_series)
    if y_train.dtype.kind in {"i", "b", "u"}:
        return "classification"
    if y_train.dtype.kind == "f":
        unique = np.unique(y_train.dropna().to_numpy())
        if unique.size <= 20 and np.allclose(unique, np.round(unique)):
            return "classification"
        return "regression"
    return "classification"


def _prepare_labels(
    y_train: pd.Series | pd.DataFrame,
    y_valid: pd.Series | pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    task_type: str,
    max_classes: int | None,
    cast_labels: bool = False,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    if task_type == "regression":
        return y_train, y_valid, y_test
    if isinstance(y_train, pd.DataFrame):
        raise ValueError("Classification tasks do not support multi-target labels.")

    if max_classes is not None and max_classes >= 2:
        top_labels = y_train.value_counts().nlargest(max_classes - 1).index
        y_train = y_train.where(y_train.isin(top_labels), "__other__")
        y_valid = y_valid.where(y_valid.isin(top_labels), "__other__")
        y_test = y_test.where(y_test.isin(top_labels), "__other__")

    if cast_labels:
        y_train = y_train.astype(str)
        y_valid = y_valid.astype(str)
        y_test = y_test.astype(str)

    return y_train, y_valid, y_test


def _filter_unseen_rows(
    X: pd.DataFrame,
    y: pd.Series,
    seen: set,
    split: str,
) -> tuple[pd.DataFrame, pd.Series]:
    keep = y.isin(seen)
    dropped = int((~keep).sum())
    if dropped:
        print(f"=== Filtered {dropped} unseen labels in {split} ===")
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def _build_model(model_name: str, task_type: str, args: argparse.Namespace):
    if model_name in {"mlp", "logreg"}:
        return _build_tabular_baseline(model_name, task_type)
    if task_type == "classification":
        if model_name == "tabicl":
            return build_tabicl(verbose=True)
        if model_name == "tabpfn":
            return build_tabpfn(
                device=args.device,
                ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
            )
        if model_name == "tabdpt":
            return _TabDPTModel(
                task_type=task_type,
                device=_resolve_tabdpt_device(args.device),
                tabdpt_path=args.tabdpt_path,
                tabdpt_weights=args.tabdpt_weights,
                n_ensembles=args.tabdpt_n_ensembles,
                temperature=args.tabdpt_temperature,
                context_size=args.tabdpt_context_size,
                permute_classes=args.tabdpt_permute_classes,
                seed=args.seed,
            )
    if model_name == "tabpfn":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            device=args.device,
            ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
        )
    if model_name == "tabdpt":
        return _TabDPTModel(
            task_type=task_type,
            device=_resolve_tabdpt_device(args.device),
            tabdpt_path=args.tabdpt_path,
            tabdpt_weights=args.tabdpt_weights,
            n_ensembles=args.tabdpt_n_ensembles,
            temperature=args.tabdpt_temperature,
            context_size=args.tabdpt_context_size,
            permute_classes=args.tabdpt_permute_classes,
            seed=args.seed,
        )
    raise ValueError("Regression is only supported with TabPFN or TabDPT.")


def _evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    task_type: str,
    evaluator,
    metrics_backend: str,
) -> dict:
    if metrics_backend == "graphbench" and evaluator is not None:
        return _graphbench_metrics(model, X, y, task_type, evaluator)
    if task_type == "classification":
        return _classification_metrics(model, X, y)
    return _regression_metrics(model, X, y)


def _classification_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2 and y.nunique(dropna=True) == 2:
            pos_label, pos_index = _binary_positive_label_and_index(model, y)
            pos_proba = proba[:, pos_index]
            y_binary = (y == pos_label).astype(int)
            metrics["roc_auc"] = roc_auc_score(y_binary, pos_proba)
            metrics["pr_auc"] = average_precision_score(y_binary, pos_proba)
    return metrics


def _regression_metrics(model, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict:
    pred = model.predict(X)
    y_arr = _as_numpy(y)
    pred_arr = np.asarray(pred)
    return {
        "mse": mean_squared_error(y_arr, pred_arr),
        "mae": mean_absolute_error(y_arr, pred_arr),
        "r2": r2_score(y_arr, pred_arr),
    }


def _graphbench_metrics(model, X, y, task_type: str, evaluator) -> dict:
    if task_type == "classification":
        pred = _graphbench_classification_pred(model, X, y)
    else:
        pred = model.predict(X)
    y_arr = _ensure_2d(_as_numpy(y))
    pred_arr = _ensure_2d(np.asarray(pred))
    if task_type == "regression":
        pred_arr = _apply_graphbench_pred_transform(evaluator, pred_arr)
    return _graphbench_eval(evaluator, pred_arr, y_arr)


def _graphbench_classification_pred(
    model,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            pos_index = 1
            if isinstance(y, pd.Series) and y.nunique(dropna=True) == 2:
                _, pos_index = _binary_positive_label_and_index(model, y)
            else:
                classes = list(getattr(model, "classes_", []))
                if classes:
                    if 1 in classes:
                        pos_index = classes.index(1)
                    else:
                        pos_index = classes.index(classes[-1])
            return proba[:, [pos_index]]
    pred = model.predict(X)
    return np.asarray(pred).reshape(-1, 1)


def _binary_positive_label_and_index(model, y: pd.Series) -> tuple[object, int]:
    classes = list(getattr(model, "classes_", []))
    if not classes:
        classes = list(pd.unique(y))
    if len(classes) != 2:
        raise ValueError(f"Expected binary classes, got: {classes}")
    if 1 in classes:
        pos_label = 1
    elif True in classes and False in classes:
        pos_label = True
    else:
        pos_label = classes[-1]
    pos_index = classes.index(pos_label)
    return pos_label, pos_index


def _evaluate_multitarget(
    pred: np.ndarray,
    y_true: pd.DataFrame,
    evaluator,
    metrics_backend: str,
) -> dict:
    y_arr = _ensure_2d(_as_numpy(y_true))
    pred_arr = _ensure_2d(np.asarray(pred))
    if metrics_backend == "graphbench" and evaluator is not None:
        pred_arr = _apply_graphbench_pred_transform(evaluator, pred_arr)
        return _graphbench_eval(evaluator, pred_arr, y_arr)
    return {
        "mse": mean_squared_error(y_arr, pred_arr),
        "mae": mean_absolute_error(y_arr, pred_arr),
        "r2": r2_score(y_arr, pred_arr),
    }


def _graphbench_eval(evaluator, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    metrics = evaluator.metric
    values = evaluator.evaluate(y_pred=y_pred, y_true=y_true)
    if isinstance(metrics, list):
        if not isinstance(values, (list, tuple)):
            return {metrics[0]: float(values)}
        return {metric: float(val) for metric, val in zip(metrics, values)}
    return {metrics: float(values)}


def _apply_graphbench_pred_transform(evaluator, pred_arr: np.ndarray) -> np.ndarray:
    metrics = evaluator.metric
    if not isinstance(metrics, list):
        metrics = [metrics]
    if "ClosedGap" in metrics:
        return -pred_arr
    return pred_arr


def _fit_predict_multi_target(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame | None,
    X_test: pd.DataFrame | None,
    model_name: str,
    device: str,
    tabdpt_path: str | None,
    tabdpt_weights: str | None,
    tabdpt_n_ensembles: int,
    tabdpt_context_size: int,
    tabpfn_ignore_limits: bool,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if model_name in {"mlp", "logreg"}:
        preds_valid: list[np.ndarray] = []
        preds_test: list[np.ndarray] = []
        for col in y_train.columns:
            model = _build_tabular_baseline(model_name, "regression")
            model.fit(X_train, y_train[col])
            if X_valid is not None:
                preds_valid.append(np.asarray(model.predict(X_valid)))
            if X_test is not None:
                preds_test.append(np.asarray(model.predict(X_test)))
        return _stack_multitarget(preds_valid, preds_test)

    if model_name == "tabpfn":
        from tabpfn import TabPFNRegressor

        preds_valid: list[np.ndarray] = []
        preds_test: list[np.ndarray] = []
        for col in y_train.columns:
            model = TabPFNRegressor(
                device=device,
                ignore_pretraining_limits=tabpfn_ignore_limits,
            )
            model.fit(X_train, y_train[col])
            if X_valid is not None:
                preds_valid.append(np.asarray(model.predict(X_valid)))
            if X_test is not None:
                preds_test.append(np.asarray(model.predict(X_test)))
        return _stack_multitarget(preds_valid, preds_test)

    if model_name == "tabdpt":
        _, TabDPTRegressor = _load_tabdpt_models(tabdpt_path)
        encoder = _TabDPTFeatureEncoder().fit(X_train)
        X_train_arr = encoder.transform(X_train)
        X_valid_arr = encoder.transform(X_valid) if X_valid is not None else None
        X_test_arr = encoder.transform(X_test) if X_test is not None else None

        preds_valid: list[np.ndarray] = []
        preds_test: list[np.ndarray] = []
        for idx, col in enumerate(y_train.columns):
            model = TabDPTRegressor(
                device=_resolve_tabdpt_device(device),
                model_weight_path=tabdpt_weights,
                verbose=False,
            )
            model.fit(X_train_arr, y_train[col].to_numpy(dtype=float))
            inner_seed = seed + idx
            if X_valid_arr is not None:
                preds_valid.append(
                    np.asarray(
                        model.predict(
                            X_valid_arr,
                            n_ensembles=tabdpt_n_ensembles,
                            context_size=tabdpt_context_size,
                            seed=inner_seed,
                        )
                    )
                )
            if X_test_arr is not None:
                preds_test.append(
                    np.asarray(
                        model.predict(
                            X_test_arr,
                            n_ensembles=tabdpt_n_ensembles,
                            context_size=tabdpt_context_size,
                            seed=inner_seed,
                        )
                    )
                )
        return _stack_multitarget(preds_valid, preds_test)

    raise ValueError(
        "Multi-target regression is only supported with TabPFN, TabDPT, or tabular baselines "
        "(mlp/logreg)."
    )


def _stack_multitarget(
    preds_valid: list[np.ndarray],
    preds_test: list[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pred_val = None
    pred_test = None
    if preds_valid:
        pred_val = np.stack(preds_valid, axis=1)
    if preds_test:
        pred_test = np.stack(preds_test, axis=1)
    return pred_val, pred_test


def _build_tabular_baseline(model_name: str, task_type: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = [col for col in X.columns if X[col].dtype == "object"]
        num_cols = [col for col in X.columns if col not in cat_cols]
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            [
                ("cat", cat_pipe, cat_cols),
                ("num", num_pipe, num_cols),
            ],
            remainder="drop",
        )

    class _SklearnBaseline:
        def __init__(self, estimator):
            self._estimator = estimator
            self._pre = None
            self.classes_ = None

        def fit(self, X: pd.DataFrame, y):
            self._pre = make_preprocessor(X)
            X_enc = self._pre.fit_transform(X)
            self._estimator.fit(X_enc, y)
            self.classes_ = getattr(self._estimator, "classes_", None)
            return self

        def predict(self, X: pd.DataFrame):
            if self._pre is None:
                raise ValueError("Baseline model must be fit before predict.")
            X_enc = self._pre.transform(X)
            return self._estimator.predict(X_enc)

        def predict_proba(self, X: pd.DataFrame):
            if not hasattr(self._estimator, "predict_proba"):
                raise AttributeError("predict_proba is not available for this baseline.")
            if self._pre is None:
                raise ValueError("Baseline model must be fit before predict_proba.")
            X_enc = self._pre.transform(X)
            return self._estimator.predict_proba(X_enc)

    if task_type == "classification":
        if model_name == "logreg":
            estimator = LogisticRegression(max_iter=1000)
        else:
            estimator = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200)
    else:
        if model_name == "logreg":
            estimator = Ridge(alpha=1.0)
        else:
            estimator = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200)

    return _SklearnBaseline(estimator)


class _TabDPTFeatureEncoder:
    def __init__(self) -> None:
        self._columns: list[str] = []
        self._categories: dict[str, list[str] | None] = {}

    def fit(self, X: pd.DataFrame) -> "_TabDPTFeatureEncoder":
        self._columns = list(X.columns)
        self._categories = {}
        for col in self._columns:
            series = X[col]
            if pd.api.types.is_numeric_dtype(series):
                self._categories[col] = None
            else:
                self._categories[col] = pd.Categorical(series).categories.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._columns:
            raise ValueError("TabDPT feature encoder must be fit before transform.")
        if X.empty:
            return np.zeros((0, len(self._columns)), dtype=float)
        cols: list[np.ndarray] = []
        for col in self._columns:
            series = X[col]
            categories = self._categories[col]
            if categories is None:
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            else:
                values = series.where(series.isin(categories), "__UNK__")
                cat_categories = categories
                if "__UNK__" not in cat_categories:
                    cat_categories = cat_categories + ["__UNK__"]
                codes = pd.Categorical(values, categories=cat_categories).codes.astype(float)
                values = codes
            cols.append(np.asarray(values, dtype=float))
        return np.stack(cols, axis=1)


class _TabDPTModel:
    def __init__(
        self,
        task_type: str,
        device: str | None,
        tabdpt_path: str | None,
        tabdpt_weights: str | None,
        n_ensembles: int,
        temperature: float,
        context_size: int,
        permute_classes: bool,
        seed: int,
    ) -> None:
        TabDPTClassifier, TabDPTRegressor = _load_tabdpt_models(tabdpt_path)
        self._task_type = task_type
        self._encoder = _TabDPTFeatureEncoder()
        self._n_ensembles = n_ensembles
        self._temperature = temperature
        self._context_size = context_size
        self._permute_classes = permute_classes
        self._seed = seed
        self._label_encoder: LabelEncoder | None = None
        if task_type == "classification":
            self._label_encoder = LabelEncoder()
            self._model = TabDPTClassifier(
                device=device,
                model_weight_path=tabdpt_weights,
                verbose=False,
            )
            self.classes_ = np.array([])
        else:
            self._model = TabDPTRegressor(
                device=device,
                model_weight_path=tabdpt_weights,
                verbose=False,
            )
            self.classes_ = np.array([])

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> "_TabDPTModel":
        X_arr = self._encoder.fit(X).transform(X)
        if self._task_type == "classification":
            if self._label_encoder is None:
                raise ValueError("TabDPT classification encoder was not initialized.")
            y_arr = np.asarray(y)
            y_enc = self._label_encoder.fit_transform(y_arr)
            self._model.fit(X_arr, y_enc.astype(int))
            self.classes_ = self._label_encoder.classes_
        else:
            y_arr = np.asarray(y, dtype=float)
            self._model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame):
        X_arr = self._encoder.transform(X)
        if self._task_type == "classification":
            pred = self._model.predict(
                X_arr,
                n_ensembles=self._n_ensembles,
                temperature=self._temperature,
                context_size=self._context_size,
                permute_classes=self._permute_classes,
                seed=self._seed,
            )
            if self._label_encoder is None:
                return pred
            return self._label_encoder.inverse_transform(pred)
        return self._model.predict(
            X_arr,
            n_ensembles=self._n_ensembles,
            context_size=self._context_size,
            seed=self._seed,
        )

    def predict_proba(self, X: pd.DataFrame):
        if self._task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        X_arr = self._encoder.transform(X)
        if self._n_ensembles and self._n_ensembles > 1:
            return self._model.ensemble_predict_proba(
                X_arr,
                n_ensembles=self._n_ensembles,
                temperature=self._temperature,
                context_size=self._context_size,
                permute_classes=self._permute_classes,
                seed=self._seed,
            )
        return self._model.predict_proba(
            X_arr,
            temperature=self._temperature,
            context_size=self._context_size,
            seed=self._seed,
        )


def _resolve_tabdpt_device(device: str) -> str | None:
    if device == "auto":
        return None
    return device


def _load_tabdpt_models(tabdpt_path: str | None):
    import os
    import sys
    from pathlib import Path

    if tabdpt_path is None:
        env_path = os.environ.get("TABDPT_PATH")
        repo_root = Path(__file__).resolve().parents[1] / "TabDPT-inference"
        candidates = [env_path, str(repo_root), str(repo_root / "src")]
    else:
        candidates = [tabdpt_path]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_dir():
            continue
        if (path / "tabdpt").is_dir():
            path_str = str(path)
        elif (path / "src" / "tabdpt").is_dir():
            path_str = str(path / "src")
        else:
            continue
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        break

    try:
        from tabdpt import TabDPTClassifier, TabDPTRegressor
    except ImportError as exc:
        raise ImportError(
            "TabDPT is not installed. Follow https://github.com/layer6ai-labs/TabDPT-inference."
        ) from exc

    return TabDPTClassifier, TabDPTRegressor


def _infer_metric_name(dataset_name: str, task_type: str) -> str | None:
    name = dataset_name.lower()
    if name.startswith("bluesky_"):
        return "bluesky"
    if name.startswith("weather"):
        return "weather"
    if name.startswith("chipdesign"):
        return "chipdesign"
    if name.startswith("electronic_circuits"):
        return "electroniccircuit"
    if name.startswith("sat_"):
        if name.endswith("_as"):
            return "sat_as"
        if name.endswith("_epm"):
            return "sat_epm"
    if name.startswith("algoreas_"):
        return "algoreas_classification" if task_type == "classification" else "algoreas_regression"
    if name.startswith("co_") and task_type == "regression":
        return "co_regression"
    return None


def _validate_graph_feature_policy(args: argparse.Namespace) -> None:
    if args.allow_graph_features:
        return
    if args.tabular_mode != "raw" or args.add_degree_feature:
        raise ValueError(
            "Graph-derived features are disabled. Use --allow-graph-features to enable "
            "--tabular-mode stats or --add-degree-feature."
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        return


def _write_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config["timestamp"] = datetime.utcnow().isoformat() + "Z"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    env = _collect_env_metadata()
    (output_dir / "env.json").write_text(json.dumps(env, indent=2))


def _collect_env_metadata() -> dict:
    try:
        import importlib.metadata as importlib_metadata
    except Exception:  # pragma: no cover
        importlib_metadata = None
    packages = [
        "tabpfn",
        "tabicl",
        "tabdpt",
        "graphbench-lib",
        "torchmetrics",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
    ]
    versions: dict[str, str] = {}
    if importlib_metadata is not None:
        for pkg in packages:
            try:
                versions[pkg] = importlib_metadata.version(pkg)
            except Exception as exc:
                versions[pkg] = f"unavailable ({exc.__class__.__name__})"
    git_commit = None
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        git_commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit,
        "packages": versions,
    }


def _save_split_table(
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
) -> None:
    df = X.copy()
    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            df[f"target_{col}"] = y[col].values
    else:
        df["target"] = y.values
    path = output_dir / f"{dataset_name}_{split_name}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")


def _save_model_predictions(
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    task_type: str,
    evaluator,
    metrics_backend: str,
) -> None:
    if X.empty:
        return
    if task_type == "classification":
        pred_label = model.predict(X)
        pred_proba = None
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(X)
        pred_graphbench = None
        if metrics_backend == "graphbench" and evaluator is not None:
            pred_graphbench = _graphbench_classification_pred(model, X, y)
        payload = {
            "pred_label": np.asarray(pred_label),
            "pred_proba": np.asarray(pred_proba) if pred_proba is not None else None,
            "pred_graphbench": np.asarray(pred_graphbench) if pred_graphbench is not None else None,
            "y_true": _as_numpy(y),
        }
        _save_prediction_bundle(output_dir, dataset_name, split_name, payload)
        return
    pred = model.predict(X)
    payload = {
        "pred": np.asarray(pred),
        "y_true": _as_numpy(y),
    }
    _save_prediction_bundle(output_dir, dataset_name, split_name, payload)


def _save_prediction_bundle(
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    payload: dict,
    task_type: str | None = None,
) -> None:
    if payload is None:
        return
    path = output_dir / f"{dataset_name}_{split_name}_preds.npz"
    arrays = {k: v for k, v in payload.items() if v is not None}
    if not arrays:
        return
    np.savez_compressed(path, **arrays)


def _run_gcn_baseline(
    args: argparse.Namespace,
    splits,
    output_path: Path,
    output_dir: Path,
    start: float,
) -> None:
    results: dict[str, dict] = {}
    for split in splits:
        print(f"=== Loading dataset: {split.name} ===")
        allow_multi_target = args.metrics == "graphbench" and args.target_index is None
        train_data = _collect_graphs(split.train, args.max_train, args.max_samples, args.seed)
        valid_data = _collect_graphs(split.valid, args.max_valid, args.max_samples, args.seed + 1)
        test_data = _collect_graphs(split.test, args.max_test, args.max_samples, args.seed + 2)

        task_type, out_dim, label_encoder = _infer_gcn_task(
            train_data, args.task, args.task_type, args.target_index, allow_multi_target
        )
        in_dim = _infer_input_dim(train_data)
        model = _GraphNativeModel(
            in_dim=in_dim,
            hidden_dim=args.gnn_hidden_dim,
            out_dim=out_dim,
            task=args.task,
            dropout=args.gnn_dropout,
        )
        device = _resolve_torch_device(args.device)
        model.to(device)
        optimizer = _build_optimizer(model, args.gnn_lr, args.gnn_weight_decay)
        criterion = _build_loss(task_type)
        print("=== Training GCN baseline ===")
        _train_gcn(
            model=model,
            train_data=train_data,
            task=args.task,
            task_type=task_type,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.gnn_epochs,
            label_encoder=label_encoder,
            target_index=args.target_index,
            allow_multi_target=allow_multi_target,
            seed=args.seed,
            max_samples=args.max_samples,
            max_train=args.max_train,
        )

        metrics: dict[str, dict] = {}
        evaluator = None
        if args.metrics == "graphbench":
            metric_name = args.graphbench_metric_name or _infer_metric_name(
                split.name, task_type
            )
            if metric_name is None:
                raise ValueError(
                    "Unable to infer GraphBench metric name. "
                    "Pass --graphbench-metric-name explicitly."
                )
            evaluator = load_graphbench_evaluator(metric_name)
            required = _required_output_dim(evaluator.metric)
            if out_dim < required:
                raise ValueError(
                    f"GraphBench metrics require {required} targets, but model outputs {out_dim}."
                )

        pred_val, y_val = _predict_gcn_split(
            model,
            valid_data,
            task=args.task,
            task_type=task_type,
            device=device,
            label_encoder=label_encoder,
            target_index=args.target_index,
            allow_multi_target=allow_multi_target,
            seed=args.seed + 1,
            max_samples=args.max_samples,
            max_split=args.max_valid,
        )
        pred_test, y_test = _predict_gcn_split(
            model,
            test_data,
            task=args.task,
            task_type=task_type,
            device=device,
            label_encoder=label_encoder,
            target_index=args.target_index,
            allow_multi_target=allow_multi_target,
            seed=args.seed + 2,
            max_samples=args.max_samples,
            max_split=args.max_test,
        )
        if pred_val is not None and y_val is not None:
            metrics["val"] = _evaluate_predictions(
                pred_val, y_val, task_type, evaluator, args.metrics
            )
        if pred_test is not None and y_test is not None:
            metrics["test"] = _evaluate_predictions(
                pred_test, y_test, task_type, evaluator, args.metrics
            )

        if not args.no_save_predictions:
            if pred_val is not None and y_val is not None:
                _save_prediction_bundle(
                    output_dir,
                    split.name,
                    "val",
                    {"pred": pred_val, "y_true": y_val},
                )
            if pred_test is not None and y_test is not None:
                _save_prediction_bundle(
                    output_dir,
                    split.name,
                    "test",
                    {"pred": pred_test, "y_true": y_test},
                )
        if args.save_splits:
            _save_graph_native_split(output_dir, split.name, "train", train_data, args)
            _save_graph_native_split(output_dir, split.name, "valid", valid_data, args)
            _save_graph_native_split(output_dir, split.name, "test", test_data, args)

        results[split.name] = {
            "task": args.task,
            "task_type": task_type,
            "train_graphs": len(train_data),
            "valid_graphs": len(valid_data),
            "test_graphs": len(test_data),
            "metrics_backend": args.metrics,
            "graphbench_metric": getattr(evaluator, "name", None) if evaluator else None,
            "metrics": metrics,
        }

    payload = {
        "model": "gcn",
        "device": args.device,
        "dataset": args.dataset,
        "results": results,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"=== Wrote metrics to {output_path} ===")
    print(json.dumps(payload, indent=2))


def _collect_graphs(
    dataset: object | None,
    split_cap: int | None,
    global_cap: int | None,
    seed: int,
) -> list[object]:
    if dataset is None:
        return []
    data_list = list(iter_graphbench_data(dataset))
    cap = split_cap if split_cap is not None else global_cap
    if cap is None or len(data_list) <= cap:
        return data_list
    print(f"=== Subsample graphs: {len(data_list)} -> {cap} ===")
    idx = np.random.default_rng(seed).choice(len(data_list), size=cap, replace=False)
    return [data_list[i] for i in idx]


def _infer_input_dim(graphs: list[object]) -> int:
    if not graphs:
        return 1
    dim = None
    for data in graphs:
        x = getattr(data, "x", None)
        if x is None:
            current = 1
        else:
            if not hasattr(x, "shape"):
                x = np.asarray(x)
            current = 1 if len(x.shape) == 1 else x.shape[1]
        if dim is None:
            dim = int(current)
        elif dim != int(current):
            raise ValueError("GCN baseline requires consistent feature dimensions.")
    return dim or 1


def _infer_gcn_task(
    graphs: list[object],
    task: str,
    task_type: str,
    target_index: int | None,
    allow_multi_target: bool,
) -> tuple[str, int, LabelEncoder | None]:
    labels = []
    for data in graphs:
        y = getattr(data, "y", None)
        if y is None:
            raise ValueError("GraphBench data missing y labels.")
        y_arr = _select_target_array(_to_numpy_array(y), target_index, allow_multi_target)
        labels.append(y_arr)

    if task_type != "auto":
        resolved = task_type
    else:
        resolved = _infer_task_type_from_labels(labels)

    if resolved == "classification":
        flat = np.concatenate([np.asarray(lbl).reshape(-1) for lbl in labels])
        encoder = LabelEncoder()
        encoder.fit(flat)
        return resolved, len(encoder.classes_), encoder

    out_dim = 1
    for lbl in labels:
        if lbl.ndim == 2 and lbl.shape[1] > 1:
            out_dim = lbl.shape[1]
            break
    return resolved, out_dim, None


def _infer_task_type_from_labels(labels: list[np.ndarray]) -> str:
    if any(lbl.ndim == 2 and lbl.shape[1] > 1 for lbl in labels):
        return "regression"
    flat = np.concatenate([lbl.reshape(-1) for lbl in labels])
    if flat.dtype.kind == "f":
        unique = np.unique(flat[~np.isnan(flat)])
        if unique.size <= 20 and np.allclose(unique, np.round(unique)):
            return "classification"
        return "regression"
    return "classification"


def _select_target_array(
    y: np.ndarray,
    target_index: int | None,
    allow_multi_target: bool,
) -> np.ndarray:
    if y.ndim == 0:
        return y.reshape(1)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        if y.shape[1] == 1:
            return y.reshape(-1)
        if target_index is None:
            if allow_multi_target:
                return y
            raise ValueError("Multi-target labels require --target-index.")
        return y[:, target_index]
    raise ValueError(f"Unsupported label shape: {y.shape}")


def _to_numpy_array(values: object) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _resolve_torch_device(device: str):
    if device == "auto":
        return "cuda" if _torch_available() and _torch_cuda_available() else "cpu"
    return device


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _build_optimizer(model, lr: float, weight_decay: float):
    import torch

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_loss(task_type: str):
    import torch.nn as nn

    if task_type == "classification":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


class _GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, num_nodes):
        x = _gcn_layer(x, edge_index, num_nodes, self.lin1)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = _gcn_layer(x, edge_index, num_nodes, self.lin2)
        return x


class _GraphNativeModel(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, task: str, dropout: float):
        super().__init__()
        self.encoder = _GCNEncoder(in_dim, hidden_dim, dropout)
        if task == "edge":
            self.head = torch.nn.Linear(hidden_dim * 2, out_dim)
        else:
            self.head = torch.nn.Linear(hidden_dim, out_dim)
        self.task = task

    def forward_node(self, x, edge_index, num_nodes):
        emb = self.encoder(x, edge_index, num_nodes)
        return self.head(emb)

    def forward_edge(self, x, edge_index, edge_label_index, num_nodes):
        emb = self.encoder(x, edge_index, num_nodes)
        src = emb[edge_label_index[0]]
        dst = emb[edge_label_index[1]]
        edge_feat = torch.cat([src, dst], dim=1)
        return self.head(edge_feat)

    def forward_graph(self, x, edge_index, num_nodes):
        emb = self.encoder(x, edge_index, num_nodes)
        graph_emb = emb.mean(dim=0, keepdim=True)
        return self.head(graph_emb)


def _gcn_layer(x, edge_index, num_nodes, linear):
    import torch

    edge_index = _ensure_edge_index(edge_index, num_nodes)
    row, col = edge_index
    values = torch.ones(row.size(0), device=x.device)
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    norm = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))
    agg = torch.sparse.mm(adj, x)
    return linear(agg)


def _ensure_edge_index(edge_index, num_nodes: int):
    import torch

    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        base = torch.arange(num_nodes, device=edge_index.device)
        return torch.stack([base, base], dim=0)
    rev = edge_index.flip(0)
    loop = torch.arange(num_nodes, device=edge_index.device)
    loop = torch.stack([loop, loop], dim=0)
    return torch.cat([edge_index, rev, loop], dim=1)


def _prepare_graph_inputs(data, device, in_dim: int):
    import torch

    x = getattr(data, "x", None)
    if x is None:
        num_nodes = int(getattr(data, "num_nodes", 0) or 0)
        if num_nodes == 0:
            edge_index = getattr(data, "edge_index", None)
            if edge_index is not None:
                edge_tensor = torch.as_tensor(edge_index)
                if edge_tensor.numel() > 0:
                    num_nodes = int(torch.max(edge_tensor) + 1)
        x = torch.zeros((num_nodes, in_dim), dtype=torch.float32)
    else:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(1)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    num_nodes = x.shape[0]
    if x.shape[1] < in_dim:
        pad = torch.zeros((num_nodes, in_dim - x.shape[1]), dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    if x.shape[1] > in_dim:
        x = x[:, :in_dim]
    return x.to(device), edge_index.to(device), num_nodes


def _extract_edge_labels(data, target_index, allow_multi_target):
    edge_label = getattr(data, "edge_label", None)
    edge_label_index = getattr(data, "edge_label_index", None)
    if edge_label is not None and edge_label_index is not None:
        labels = _select_target_array(
            _to_numpy_array(edge_label), target_index, allow_multi_target
        )
        return edge_label_index, labels
    edge_index = getattr(data, "edge_index", None)
    y = getattr(data, "y", None)
    if edge_index is None or y is None:
        raise ValueError("Edge-level task requires edge_index and edge labels.")
    y_arr = _select_target_array(_to_numpy_array(y), target_index, allow_multi_target)
    if y_arr.shape[0] != edge_index.shape[1]:
        raise ValueError("Edge labels do not match edge count.")
    return edge_index, y_arr


def _train_gcn(
    model: _GraphNativeModel,
    train_data: list[object],
    task: str,
    task_type: str,
    optimizer,
    criterion,
    device: str,
    epochs: int,
    label_encoder: LabelEncoder | None,
    target_index: int | None,
    allow_multi_target: bool,
    seed: int,
    max_samples: int | None,
    max_train: int | None,
) -> None:
    import torch

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for graph_idx, data in enumerate(train_data):
            x, edge_index, num_nodes = _prepare_graph_inputs(data, device, model.encoder.lin1.in_features)
            if task == "node":
                y = _select_target_array(
                    _to_numpy_array(getattr(data, "y")),
                    target_index,
                    allow_multi_target,
                )
                y = torch.as_tensor(y)
                idx = _sample_indices(y.shape[0], max_train or max_samples, seed + graph_idx)
                if idx is not None:
                    y = y[idx]
                out = model.forward_node(x, edge_index, num_nodes)
                if idx is not None:
                    out = out[idx]
            elif task == "edge":
                edge_label_index, labels = _extract_edge_labels(
                    data, target_index, allow_multi_target
                )
                y = torch.as_tensor(labels)
                idx = _sample_indices(y.shape[0], max_train or max_samples, seed + graph_idx)
                edge_label_index = torch.as_tensor(edge_label_index)
                if idx is not None:
                    edge_label_index = edge_label_index[:, idx]
                    y = y[idx]
                out = model.forward_edge(x, edge_index, edge_label_index.to(device), num_nodes)
            else:
                y = _select_target_array(
                    _to_numpy_array(getattr(data, "y")),
                    target_index,
                    allow_multi_target,
                )
                y = torch.as_tensor(y).reshape(1, -1)
                out = model.forward_graph(x, edge_index, num_nodes).reshape(1, -1)

            if task_type == "classification":
                y = y.long()
                if label_encoder is not None:
                    y = torch.as_tensor(label_encoder.transform(y.cpu().numpy().reshape(-1)))
                y = y.to(device)
                loss = criterion(out, y)
            else:
                y = y.to(device).float()
                if y.dim() == 1 and out.dim() == 2 and out.shape[1] == 1:
                    y = y.view(-1, 1)
                loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"=== Epoch {epoch + 1}/{epochs} | loss={total:.4f} ===")


def _predict_gcn_split(
    model: _GraphNativeModel,
    data_list: list[object],
    task: str,
    task_type: str,
    device: str,
    label_encoder: LabelEncoder | None,
    target_index: int | None,
    allow_multi_target: bool,
    seed: int,
    max_samples: int | None,
    max_split: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    import torch

    if not data_list:
        return None, None
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for graph_idx, data in enumerate(data_list):
        x, edge_index, num_nodes = _prepare_graph_inputs(
            data, device, model.encoder.lin1.in_features
        )
        if task == "node":
            y = _select_target_array(
                _to_numpy_array(getattr(data, "y")),
                target_index,
                allow_multi_target,
            )
            idx = _sample_indices(y.shape[0], max_split or max_samples, seed + graph_idx)
            out = model.forward_node(x, edge_index, num_nodes)
            if idx is not None:
                out = out[idx]
                y = y[idx]
            preds.append(_postprocess_preds(out, task_type))
            labels.append(np.asarray(y))
        elif task == "edge":
            edge_label_index, y = _extract_edge_labels(data, target_index, allow_multi_target)
            idx = _sample_indices(y.shape[0], max_split or max_samples, seed + graph_idx)
            edge_label_index = torch.as_tensor(edge_label_index)
            if idx is not None:
                edge_label_index = edge_label_index[:, idx]
                y = y[idx]
            out = model.forward_edge(x, edge_index, edge_label_index.to(device), num_nodes)
            preds.append(_postprocess_preds(out, task_type))
            labels.append(np.asarray(y))
        else:
            y = _select_target_array(
                _to_numpy_array(getattr(data, "y")),
                target_index,
                allow_multi_target,
            )
            out = model.forward_graph(x, edge_index, num_nodes)
            preds.append(_postprocess_preds(out, task_type))
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(1, -1)
            labels.append(y_arr)

    pred_arr = np.concatenate(preds, axis=0)
    y_arr = np.concatenate(labels, axis=0)
    if task_type == "classification" and label_encoder is not None:
        y_arr = label_encoder.transform(y_arr.reshape(-1))
    return pred_arr, y_arr


def _postprocess_preds(out, task_type: str) -> np.ndarray:
    import torch

    if task_type == "classification":
        proba = torch.softmax(out, dim=-1)
        return proba.detach().cpu().numpy()
    return out.detach().cpu().numpy()


def _sample_indices(n: int, cap: int | None, seed: int):
    if cap is None or n <= cap:
        return None
    idx = np.random.default_rng(seed).choice(n, size=cap, replace=False)
    return idx


def _evaluate_predictions(
    pred: np.ndarray,
    y_true: np.ndarray,
    task_type: str,
    evaluator,
    metrics_backend: str,
) -> dict:
    if metrics_backend == "graphbench" and evaluator is not None:
        return _graphbench_eval(evaluator, _ensure_2d(pred), _ensure_2d(y_true))
    if task_type == "classification":
        return _classification_metrics_from_arrays(y_true, pred)
    return {
        "mse": mean_squared_error(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def _classification_metrics_from_arrays(y_true: np.ndarray, pred_proba: np.ndarray) -> dict:
    pred_labels = pred_proba.argmax(axis=-1)
    metrics = {
        "accuracy": accuracy_score(y_true, pred_labels),
        "macro_f1": f1_score(y_true, pred_labels, average="macro", zero_division=0),
    }
    if pred_proba.ndim == 2 and pred_proba.shape[1] == 2:
        y_binary = (y_true == 1).astype(int)
        metrics["roc_auc"] = roc_auc_score(y_binary, pred_proba[:, 1])
        metrics["pr_auc"] = average_precision_score(y_binary, pred_proba[:, 1])
    return metrics


def _save_graph_native_split(
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    data_list: list[object],
    args: argparse.Namespace,
) -> None:
    rows = []
    for graph_idx, data in enumerate(data_list):
        y = getattr(data, "y", None)
        if y is None:
            continue
        y_arr = _select_target_array(
            _to_numpy_array(y),
            args.target_index,
            args.metrics == "graphbench" and args.target_index is None,
        )
        rows.append(
            {
                "graph_id": graph_idx,
                "num_nodes": int(getattr(data, "num_nodes", 0) or 0),
                "num_edges": int(getattr(getattr(data, "edge_index", None), "shape", [0, 0])[1]),
                "label_shape": str(y_arr.shape),
            }
        )
    df = pd.DataFrame(rows)
    path = output_dir / f"{dataset_name}_{split_name}_graphs.csv.gz"
    df.to_csv(path, index=False, compression="gzip")


def _validate_graphbench_targets(evaluator, y_train, target_index: int | None) -> None:
    metrics = evaluator.metric
    if not isinstance(metrics, list):
        metrics = [metrics]
    if "ClosedGap" in metrics and not _is_multitarget(y_train):
        raise ValueError(
            "GraphBench ClosedGap requires multi-target runtimes per solver. "
            "Ensure the dataset provides multiple targets and do not set --target-index."
        )
    required = _required_output_dim(evaluator.metric)
    if required <= 1:
        return
    if not _is_multitarget(y_train):
        raise ValueError(
            f"GraphBench metrics require {required} targets, but only one target "
            "was provided. Remove --target-index to use all targets."
        )
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] < required:
        raise ValueError(
            "GraphBench metrics require "
            f"{required} targets, but only {y_train.shape[1]} were found."
        )


def _required_output_dim(metrics) -> int:
    if not isinstance(metrics, list):
        metrics = [metrics]
    max_idx = -1
    for metric in metrics:
        if "_" in metric:
            suffix = metric.rsplit("_", 1)[-1]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return max_idx + 1 if max_idx >= 0 else 1


def _is_multitarget(y) -> bool:
    return isinstance(y, pd.DataFrame) and y.shape[1] > 1


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


def _as_numpy(values: pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        return values.to_numpy()
    return values.to_numpy()
