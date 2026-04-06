"""Run TabICL/TabPFN experiments end-to-end on RelBench tasks."""

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
from typing import Any, Sequence

import numpy as np
import pandas as pd
from relbench.base import EntityTask, RecommendationTask, Table, TaskType
from relbench.datasets import get_dataset_names
from relbench.tasks import get_task, get_task_names

from model import build_tabicl, build_tabpfn

TABICL_AUTO_MAX_TRAIN_ROWS = 50_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tabicl",
        choices=["tabicl", "tabpfn"],
    )
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="RelBench task name (defaults to first task registered for the dataset).",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download prebuilt RelBench dataset/task artifacts when available.",
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Delete cached task split files and recompute tables.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--tabpfn-ignore-pretraining-limits",
        action="store_true",
        help="Allow TabPFN to fit rows outside pretraining limits.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional train split cap (before link-pair expansion).",
    )
    parser.add_argument(
        "--max-valid",
        type=int,
        default=None,
        help="Optional validation split cap.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional test split cap.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap applied to train/valid/test splits.",
    )
    parser.add_argument(
        "--feature-joins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Join static foreign-key table features onto task rows.",
    )
    parser.add_argument(
        "--max-join-columns",
        type=int,
        default=8,
        help="Maximum non-key columns joined per foreign-key table.",
    )
    parser.add_argument(
        "--include-time-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand datetime columns into numeric time features.",
    )
    parser.add_argument(
        "--link-n-neg-per-pos",
        type=int,
        default=2,
        help="Negatives sampled per positive edge for link-prediction training.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=str,
        choices=["train", "all"],
        default="train",
        help="Destination candidate pool for link-prediction ranking.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2000,
        help="Max candidates scored per query in link-prediction evaluation (<=0 means full pool).",
    )
    parser.add_argument(
        "--ranking-batchsize",
        type=int,
        default=1024,
        help="Batch size while scoring candidates for link-prediction ranking.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save validation/test predictions as .npz artifacts next to metrics JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/relbench/relbench_metrics.json",
        help="Path to write metrics JSON.",
    )
    return parser.parse_args(argv)


def run_relbench(args: argparse.Namespace) -> None:
    start = time.time()
    _set_seed(args.seed)
    _validate_dataset(args)
    args.task = _resolve_task_name(args.dataset, args.task)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(output_dir, args)

    print("=== Starting RelBench experiment ===")
    print(
        f"Model: {args.model} | dataset={args.dataset} | task={args.task} | "
        f"download={args.download}"
    )

    try:
        task = get_task(
            dataset_name=args.dataset,
            task_name=args.task,
            download=args.download,
        )
        print(f"=== Task type: {task.task_type.value} ===")

        if task.task_type == TaskType.LINK_PREDICTION:
            result = _run_link_prediction_task(task, args, output_dir)
        else:
            result = _run_entity_task(task, args, output_dir)

        payload = {
            "model": args.model,
            "dataset": args.dataset,
            "task": args.task,
            "task_type": task.task_type.value,
            "device": args.device,
            "result": result,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"=== Wrote metrics to {output_path} ===")
        print(json.dumps(payload, indent=2))
    except Exception:
        print("=== RelBench experiment failed ===")
        traceback.print_exc()
        raise


def _run_entity_task(
    task: EntityTask,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    train_table = _get_task_table(
        task,
        split="train",
        include_target=True,
        force_recompute=args.force_recompute_cache,
    )
    val_table = _get_task_table(
        task,
        split="val",
        include_target=True,
        force_recompute=args.force_recompute_cache,
    )
    test_table = _get_task_table(
        task,
        split="test",
        include_target=True,
        force_recompute=args.force_recompute_cache,
    )

    train_df = _subsample_df(train_table.df, args.max_train, args.max_samples, args.seed, "train")
    val_df = _subsample_df(val_table.df, args.max_valid, args.max_samples, args.seed + 1, "val")
    test_df = _subsample_df(test_table.df, args.max_test, args.max_samples, args.seed + 2, "test")

    X_train_raw, y_train = _split_entity_features_target(train_df, task.target_col)
    X_val_raw, y_val = _split_entity_features_target(val_df, task.target_col)
    X_test_raw, y_test = _split_entity_features_target(test_df, task.target_col)

    db = task.dataset.get_db()
    X_train = _prepare_features(
        X_train_raw,
        fkey_col_to_table=train_table.fkey_col_to_pkey_table,
        db=db,
        include_time_features=args.include_time_features,
        feature_joins=args.feature_joins,
        max_join_columns=args.max_join_columns,
    )
    X_val = _prepare_features(
        X_val_raw,
        fkey_col_to_table=val_table.fkey_col_to_pkey_table,
        db=db,
        include_time_features=args.include_time_features,
        feature_joins=args.feature_joins,
        max_join_columns=args.max_join_columns,
    )
    X_test = _prepare_features(
        X_test_raw,
        fkey_col_to_table=test_table.fkey_col_to_pkey_table,
        db=db,
        include_time_features=args.include_time_features,
        feature_joins=args.feature_joins,
        max_join_columns=args.max_join_columns,
    )
    X_train, X_val, X_test = _align_features(X_train, X_val, X_test)
    X_train, y_train = _maybe_auto_cap_tabicl_train(
        X_train=X_train,
        y_train=y_train,
        model_name=args.model,
        max_train=args.max_train,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}

    if task.task_type == TaskType.REGRESSION:
        model = _build_regressor(args)
        print(f"=== Fitting regressor on {len(X_train)} rows x {X_train.shape[1]} columns ===")
        model.fit(X_train, y_train)
        val_pred = np.asarray(model.predict(X_val), dtype=float)
        test_pred = np.asarray(model.predict(X_test), dtype=float)
        metrics["val"] = task.evaluate(val_pred, target_table=_table_with_df(val_table, val_df))
        metrics["test"] = task.evaluate(test_pred, target_table=_table_with_df(test_table, test_df))
        predictions["val_pred"] = val_pred
        predictions["test_pred"] = test_pred
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        val_pred, test_pred = _fit_predict_multilabel(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            X_test=X_test,
            args=args,
        )
        metrics["val"] = task.evaluate(val_pred, target_table=_table_with_df(val_table, val_df))
        metrics["test"] = task.evaluate(test_pred, target_table=_table_with_df(test_table, test_df))
        predictions["val_pred"] = val_pred
        predictions["test_pred"] = test_pred
    else:
        model = _build_classifier(args)
        _fit_with_heartbeat(model, X_train, y_train, model_name=args.model)
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            val_pred = _predict_binary_scores(model, X_val)
            test_pred = _predict_binary_scores(model, X_test)
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            val_pred = _predict_multiclass_scores(model, X_val)
            test_pred = _predict_multiclass_scores(model, X_test)
        else:
            raise ValueError(f"Unsupported entity task type: {task.task_type}")
        metrics["val"] = task.evaluate(val_pred, target_table=_table_with_df(val_table, val_df))
        metrics["test"] = task.evaluate(test_pred, target_table=_table_with_df(test_table, test_df))
        predictions["val_pred"] = np.asarray(val_pred)
        predictions["test_pred"] = np.asarray(test_pred)

    if args.save_predictions:
        _save_prediction_bundle(
            output_dir=output_dir,
            split_name="entity",
            payload={
                "val_pred": predictions.get("val_pred"),
                "test_pred": predictions.get("test_pred"),
                "val_y_true": _to_numpy_targets(y_val),
                "test_y_true": _to_numpy_targets(y_test),
            },
        )

    return {
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
        "metrics": metrics,
    }


def _run_link_prediction_task(
    task: RecommendationTask,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    train_table = _get_task_table(
        task,
        split="train",
        include_target=True,
        force_recompute=args.force_recompute_cache,
    )
    val_table = _get_task_table(
        task,
        split="val",
        include_target=True,
        force_recompute=args.force_recompute_cache,
    )
    test_table = _get_task_table(
        task,
        split="test",
        include_target=True,
        force_recompute=args.force_recompute_cache,
    )
    dst_col = task.dst_entity_col

    train_df = _subsample_df(train_table.df, args.max_train, args.max_samples, args.seed, "train")
    val_df = _subsample_df(val_table.df, args.max_valid, args.max_samples, args.seed + 1, "val")
    test_df = _subsample_df(test_table.df, args.max_test, args.max_samples, args.seed + 2, "test")

    train_pool = _extract_destination_pool(train_df, dst_col)
    if not train_pool:
        raise ValueError("No destination IDs found in training split.")

    if args.candidate_pool == "all":
        candidate_pool = list(range(task.num_dst_nodes))
    else:
        candidate_pool = train_pool

    pair_df, pair_labels = build_link_training_pairs(
        query_df=train_df,
        dst_col=dst_col,
        candidate_pool=candidate_pool,
        n_neg_per_pos=args.link_n_neg_per_pos,
        seed=args.seed,
    )
    if pair_df.empty:
        raise ValueError("Link training expansion produced no examples.")

    db = task.dataset.get_db()
    X_train = _prepare_features(
        pair_df,
        fkey_col_to_table=train_table.fkey_col_to_pkey_table,
        db=db,
        include_time_features=args.include_time_features,
        feature_joins=args.feature_joins,
        max_join_columns=args.max_join_columns,
    )
    X_train, pair_labels = _maybe_auto_cap_tabicl_train(
        X_train=X_train,
        y_train=pair_labels,
        model_name=args.model,
        max_train=args.max_train,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    train_columns = list(X_train.columns)

    model = _build_classifier(args)
    _fit_with_heartbeat(model, X_train, pair_labels, model_name=args.model)

    max_candidates = (
        args.max_candidates
        if args.max_candidates is None or args.max_candidates > 0
        else None
    )
    val_topk = _predict_link_topk(
        model=model,
        table=_table_with_df(val_table, val_df),
        fkey_col_to_table=val_table.fkey_col_to_pkey_table,
        db=db,
        dst_col=dst_col,
        eval_k=task.eval_k,
        candidate_pool=candidate_pool,
        max_candidates=max_candidates,
        ranking_batchsize=args.ranking_batchsize,
        include_time_features=args.include_time_features,
        feature_joins=args.feature_joins,
        max_join_columns=args.max_join_columns,
        train_columns=train_columns,
        seed=args.seed + 101,
    )
    test_topk = _predict_link_topk(
        model=model,
        table=_table_with_df(test_table, test_df),
        fkey_col_to_table=test_table.fkey_col_to_pkey_table,
        db=db,
        dst_col=dst_col,
        eval_k=task.eval_k,
        candidate_pool=candidate_pool,
        max_candidates=max_candidates,
        ranking_batchsize=args.ranking_batchsize,
        include_time_features=args.include_time_features,
        feature_joins=args.feature_joins,
        max_join_columns=args.max_join_columns,
        train_columns=train_columns,
        seed=args.seed + 102,
    )
    metrics = {
        "val": task.evaluate(val_topk, target_table=_table_with_df(val_table, val_df)),
        "test": task.evaluate(test_topk, target_table=_table_with_df(test_table, test_df)),
    }

    if args.save_predictions:
        _save_prediction_bundle(
            output_dir=output_dir,
            split_name="link",
            payload={
                "val_topk": val_topk,
                "test_topk": test_topk,
            },
        )

    return {
        "train_queries": int(len(train_df)),
        "train_pairs": int(len(X_train)),
        "val_queries": int(len(val_df)),
        "test_queries": int(len(test_df)),
        "num_features": int(X_train.shape[1]),
        "candidate_pool_size": int(len(candidate_pool)),
        "eval_k": int(task.eval_k),
        "metrics": metrics,
    }


def _predict_link_topk(
    model,
    table: Table,
    fkey_col_to_table: dict[str, str],
    db,
    dst_col: str,
    eval_k: int,
    candidate_pool: Sequence[int],
    max_candidates: int | None,
    ranking_batchsize: int,
    include_time_features: bool,
    feature_joins: bool,
    max_join_columns: int,
    train_columns: list[str],
    seed: int,
) -> np.ndarray:
    df = table.df.reset_index(drop=True)
    query_df = df.drop(columns=[dst_col], errors="ignore")
    topk = np.zeros((len(query_df), eval_k), dtype=np.int64)
    rng = np.random.default_rng(seed)

    for idx, row in query_df.iterrows():
        row_dict = row.to_dict()
        true_dst = _to_int_list(df.iloc[idx][dst_col]) if dst_col in df.columns else []
        candidate_ids = sample_candidate_ids(
            base_pool=candidate_pool,
            positive_ids=true_dst,
            max_candidates=max_candidates,
            eval_k=eval_k,
            rng=rng,
        )
        scores = _score_row_candidates(
            model=model,
            row_dict=row_dict,
            candidate_ids=candidate_ids,
            dst_col=dst_col,
            fkey_col_to_table=fkey_col_to_table,
            db=db,
            include_time_features=include_time_features,
            feature_joins=feature_joins,
            max_join_columns=max_join_columns,
            train_columns=train_columns,
            batch_size=ranking_batchsize,
        )
        order = np.argsort(scores)[::-1]
        picked = candidate_ids[order][:eval_k]
        if len(picked) < eval_k:
            pad_value = int(picked[0]) if len(picked) > 0 else int(candidate_ids[0])
            picked = np.pad(picked, (0, eval_k - len(picked)), constant_values=pad_value)
        topk[idx] = picked.astype(np.int64)
        if idx == 0 or (idx + 1) % 20 == 0 or idx + 1 == len(query_df):
            print(f"=== Ranking progress: {idx + 1}/{len(query_df)} ===")
    return topk


def _score_row_candidates(
    model,
    row_dict: dict[str, Any],
    candidate_ids: np.ndarray,
    dst_col: str,
    fkey_col_to_table: dict[str, str],
    db,
    include_time_features: bool,
    feature_joins: bool,
    max_join_columns: int,
    train_columns: list[str],
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(candidate_ids), batch_size):
        stop = min(start + batch_size, len(candidate_ids))
        batch_ids = candidate_ids[start:stop]
        batch_data = {col: [value] * len(batch_ids) for col, value in row_dict.items()}
        batch_data[dst_col] = batch_ids.tolist()
        batch_df = pd.DataFrame(batch_data)
        X_batch = _prepare_features(
            batch_df,
            fkey_col_to_table=fkey_col_to_table,
            db=db,
            include_time_features=include_time_features,
            feature_joins=feature_joins,
            max_join_columns=max_join_columns,
        )
        X_batch = align_to_columns(X_batch, train_columns)
        scores.append(_predict_binary_scores(model, X_batch))
    return np.concatenate(scores, axis=0)


def build_link_training_pairs(
    query_df: pd.DataFrame,
    dst_col: str,
    candidate_pool: Sequence[int],
    n_neg_per_pos: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = random.Random(seed)
    base_cols = [col for col in query_df.columns if col != dst_col]
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    candidate_pool = list(dict.fromkeys(int(x) for x in candidate_pool))

    for idx, (_, row) in enumerate(query_df.iterrows()):
        row_map = row.to_dict()
        positives = _to_int_list(row_map.get(dst_col))
        if not positives:
            continue
        positive_set = set(positives)
        base = {col: row_map[col] for col in base_cols}
        for dst_id in positives:
            rec = dict(base)
            rec[dst_col] = int(dst_id)
            records.append(rec)
            labels.append(1)
            for _ in range(max(n_neg_per_pos, 0)):
                neg = _sample_negative(
                    candidate_pool=candidate_pool,
                    positives=positive_set,
                    rng=rng,
                )
                if neg is None:
                    break
                neg_rec = dict(base)
                neg_rec[dst_col] = int(neg)
                records.append(neg_rec)
                labels.append(0)
        if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == len(query_df):
            print(f"=== Link pair expansion: {idx + 1}/{len(query_df)} queries ===")

    X = pd.DataFrame.from_records(records)
    y = pd.Series(labels, name="label", dtype="int64")
    return X, y


def sample_candidate_ids(
    base_pool: Sequence[int],
    positive_ids: Sequence[int],
    max_candidates: int | None,
    eval_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base_arr = np.array(list(dict.fromkeys(int(x) for x in base_pool)), dtype=np.int64)
    if base_arr.size == 0:
        raise ValueError("Candidate pool cannot be empty.")
    positives = np.array(list(dict.fromkeys(int(x) for x in positive_ids)), dtype=np.int64)

    if max_candidates is None or max_candidates >= base_arr.size:
        cand = base_arr.copy()
    else:
        must_keep = positives.tolist()
        remaining_cap = max(max_candidates - len(must_keep), 0)
        if remaining_cap > 0:
            mask = ~np.isin(base_arr, positives)
            candidates = base_arr[mask]
            if candidates.size > remaining_cap:
                pick_idx = rng.choice(candidates.size, size=remaining_cap, replace=False)
                sampled = candidates[pick_idx]
            else:
                sampled = candidates
            cand = np.concatenate([np.array(must_keep, dtype=np.int64), sampled])
        else:
            cand = np.array(must_keep, dtype=np.int64)
    cand = np.unique(cand)

    if cand.size < eval_k:
        extra = base_arr[~np.isin(base_arr, cand)]
        need = min(eval_k - cand.size, extra.size)
        if need > 0:
            cand = np.concatenate([cand, extra[:need]])
    if cand.size == 0:
        cand = base_arr[:1]
    if cand.size < eval_k:
        pad = np.repeat(cand[0], eval_k - cand.size)
        cand = np.concatenate([cand, pad])
    rng.shuffle(cand)
    return cand


def _sample_negative(
    candidate_pool: Sequence[int],
    positives: set[int],
    rng: random.Random,
    max_tries: int = 100,
) -> int | None:
    if not candidate_pool:
        return None
    for _ in range(max_tries):
        candidate = int(rng.choice(candidate_pool))
        if candidate not in positives:
            return candidate
    return None


def _split_entity_features_target(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from split.")
    X = df.drop(columns=[target_col]).reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)
    return X, y


def _predict_binary_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 1:
            return proba.astype(float)
        if proba.shape[1] == 1:
            return proba[:, 0].astype(float)
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.size and 1 in classes:
            pos_index = int(np.where(classes == 1)[0][0])
        else:
            pos_index = proba.shape[1] - 1
        return proba[:, pos_index].astype(float)
    pred = np.asarray(model.predict(X), dtype=float)
    return pred


def _predict_multiclass_scores(model, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict(X))
        classes = np.unique(pred)
        mapping = {cls: idx for idx, cls in enumerate(classes)}
        out = np.zeros((len(pred), len(classes)), dtype=float)
        for i, label in enumerate(pred):
            out[i, mapping[label]] = 1.0
        return out
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == proba.shape[1] and np.issubdtype(classes.dtype, np.integer):
        if classes.min() >= 0:
            num_classes = int(classes.max()) + 1
            aligned = np.zeros((len(proba), num_classes), dtype=float)
            aligned[:, classes.astype(int)] = proba
            return aligned
    return proba


def _fit_predict_multilabel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    y_matrix = _stack_multilabel_targets(y_train)
    n_labels = y_matrix.shape[1]
    models = []
    for idx in range(n_labels):
        print(f"=== Fitting multilabel classifier {idx + 1}/{n_labels} ===")
        model = _build_classifier(args)
        _fit_with_heartbeat(model, X_train, y_matrix[:, idx], model_name=f"{args.model}[{idx}]")
        models.append(model)

    val_pred = np.column_stack([_predict_binary_scores(model, X_val) for model in models])
    test_pred = np.column_stack([_predict_binary_scores(model, X_test) for model in models])
    return val_pred, test_pred


def _stack_multilabel_targets(y: pd.Series) -> np.ndarray:
    if y.empty:
        return np.zeros((0, 0), dtype=int)
    rows = [np.asarray(row, dtype=int) for row in y]
    return np.stack(rows, axis=0)


def _build_classifier(args: argparse.Namespace):
    if args.model == "tabicl":
        return build_tabicl(verbose=True)
    return build_tabpfn(
        device=args.device,
        ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
    )


def _build_regressor(args: argparse.Namespace):
    if args.model != "tabpfn":
        raise ValueError("Regression tasks are only supported for TabPFN.")
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        device=args.device,
        ignore_pretraining_limits=args.tabpfn_ignore_pretraining_limits,
    )


def _fit_with_heartbeat(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
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
        except BaseException as exc:  # pragma: no cover
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


def _prepare_features(
    df: pd.DataFrame,
    fkey_col_to_table: dict[str, str],
    db,
    include_time_features: bool,
    feature_joins: bool,
    max_join_columns: int,
) -> pd.DataFrame:
    X = df.copy()
    X = _drop_nonscalar_columns(X)
    if include_time_features:
        X = _expand_time_features(X)
    if feature_joins:
        X = _join_fk_features(
            X=X,
            fkey_col_to_table=fkey_col_to_table,
            db=db,
            max_join_columns=max_join_columns,
        )
        X = _drop_nonscalar_columns(X)
        if include_time_features:
            X = _expand_time_features(X)

    for col in list(X.columns):
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.int8)
    for fk_col in fkey_col_to_table:
        if fk_col in X.columns:
            X[fk_col] = X[fk_col].map(_format_entity_id)
    return X.reset_index(drop=True)


def _drop_nonscalar_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = []
    for col in df.columns:
        if _series_is_scalar(df[col]):
            keep_cols.append(col)
    return df[keep_cols].copy()


def _expand_time_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for col in list(X.columns):
        ser = X[col]
        if not _is_datetime_like_series(ser):
            continue
        ts = pd.to_datetime(ser, errors="coerce", utc=True)
        unix_ns = ts.astype("int64")
        X[f"{col}__unix"] = (unix_ns // 1_000_000_000).astype("float64")
        X[f"{col}__year"] = ts.dt.year.astype("float64")
        X[f"{col}__month"] = ts.dt.month.astype("float64")
        X[f"{col}__day"] = ts.dt.day.astype("float64")
        X[f"{col}__dayofweek"] = ts.dt.dayofweek.astype("float64")
        X = X.drop(columns=[col])
    return X


def _join_fk_features(
    X: pd.DataFrame,
    fkey_col_to_table: dict[str, str],
    db,
    max_join_columns: int,
) -> pd.DataFrame:
    out = X.copy()
    for fk_col, table_name in fkey_col_to_table.items():
        if fk_col not in out.columns:
            continue
        if table_name not in db.table_dict:
            continue
        table = db.table_dict[table_name]
        if table.pkey_col is None:
            continue
        if table.time_col is not None:
            print(
                f"=== Skip join for {fk_col}->{table_name}: table is temporal "
                "(would require as-of join) ==="
            )
            continue
        lookup = table.df
        if not lookup[table.pkey_col].is_unique:
            print(f"=== Skip join for {fk_col}->{table_name}: key is not unique ===")
            continue
        join_cols = []
        for col in lookup.columns:
            if col == table.pkey_col:
                continue
            if not _series_is_scalar(lookup[col]):
                continue
            join_cols.append(col)
        if max_join_columns is not None and max_join_columns >= 0:
            join_cols = join_cols[:max_join_columns]
        if not join_cols:
            continue

        join_key = f"__{fk_col}__join_key__"
        join_df = lookup[[table.pkey_col, *join_cols]].copy()
        rename_map = {table.pkey_col: join_key}
        for col in join_cols:
            rename_map[col] = f"{fk_col}__{col}"
        join_df = join_df.rename(columns=rename_map)

        out = out.merge(join_df, left_on=fk_col, right_on=join_key, how="left")
        out = out.drop(columns=[join_key])
    return out


def _series_is_scalar(ser: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser):
        return True
    if pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_datetime64tz_dtype(ser):
        return True
    sample = ser.dropna().head(8).tolist()
    if not sample:
        return True
    return all(_is_scalar_value(value) for value in sample)


def _is_scalar_value(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return False
    return True


def _is_datetime_like_series(ser: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_datetime64tz_dtype(ser):
        return True
    sample = ser.dropna().head(8).tolist()
    if not sample:
        return False
    return all(isinstance(value, (pd.Timestamp, datetime)) for value in sample)


def _format_entity_id(value: Any) -> str:
    if pd.isna(value):
        return "__NA__"
    if isinstance(value, (np.integer, int)):
        return f"id_{int(value)}"
    if isinstance(value, float) and float(value).is_integer():
        return f"id_{int(value)}"
    return f"id_{value}"


def _extract_destination_pool(df: pd.DataFrame, dst_col: str) -> list[int]:
    values: set[int] = set()
    for row in df[dst_col]:
        values.update(_to_int_list(row))
    return sorted(values)


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [int(v) for v in values.tolist()]
    if isinstance(values, (list, tuple, set)):
        return [int(v) for v in values]
    return [int(values)]


def _align_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_cols = list(X_train.columns)
    return (
        align_to_columns(X_train, train_cols),
        align_to_columns(X_val, train_cols),
        align_to_columns(X_test, train_cols),
    )


def align_to_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[list(columns)].reset_index(drop=True)


def _table_with_df(table: Table, df: pd.DataFrame) -> Table:
    return Table(
        df=df.reset_index(drop=True),
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def _get_task_table(
    task,
    split: str,
    include_target: bool,
    force_recompute: bool,
) -> Table:
    mask_input_cols = not include_target
    cache_path = None
    if getattr(task, "cache_dir", None):
        cache_path = Path(task.cache_dir) / f"{split}.parquet"
        if force_recompute and cache_path.exists():
            print(f"=== Removing cached split: {cache_path} ===")
            cache_path.unlink()

    for attempt in range(2):
        try:
            task.get_table.cache_clear()
            return task.get_table(split, mask_input_cols=mask_input_cols)
        except Exception as exc:
            if attempt == 0 and cache_path is not None and cache_path.exists():
                print(
                    f"=== Failed to load cached split ({split}): "
                    f"{exc.__class__.__name__}; deleting cache and retrying ==="
                )
                cache_path.unlink()
                continue
            raise
    raise RuntimeError(f"Failed to load split: {split}")


def _subsample_df(
    df: pd.DataFrame,
    split_cap: int | None,
    global_cap: int | None,
    seed: int,
    split_label: str,
) -> pd.DataFrame:
    cap = split_cap if split_cap is not None else global_cap
    if cap is None or len(df) <= cap:
        return df.reset_index(drop=True)
    print(f"=== Subsample {split_label}: {len(df)} -> {cap} ===")
    idx = np.random.default_rng(seed).choice(len(df), size=cap, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def _maybe_auto_cap_tabicl_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    max_train: int | None,
    max_samples: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if model_name != "tabicl":
        return X_train, y_train
    if max_train is not None or max_samples is not None:
        return X_train, y_train
    if len(X_train) <= TABICL_AUTO_MAX_TRAIN_ROWS:
        return X_train, y_train

    print(
        "=== Auto-subsample train for tabicl: "
        f"{len(X_train)} -> {TABICL_AUTO_MAX_TRAIN_ROWS} "
        "(override with --max-train / --max-samples) ==="
    )
    idx = np.random.default_rng(seed).choice(
        len(X_train),
        size=TABICL_AUTO_MAX_TRAIN_ROWS,
        replace=False,
    )
    X_sel = X_train.iloc[idx].reset_index(drop=True)
    y_sel = y_train.iloc[idx].reset_index(drop=True)
    return X_sel, y_sel


def _validate_dataset(args: argparse.Namespace) -> None:
    dataset_names = set(get_dataset_names())
    if args.dataset not in dataset_names:
        available = ", ".join(sorted(dataset_names))
        raise ValueError(f"Unknown RelBench dataset '{args.dataset}'. Available: {available}")


def _resolve_task_name(dataset: str, task_name: str | None) -> str:
    task_names = get_task_names(dataset)
    if not task_names:
        raise ValueError(f"No tasks found for dataset '{dataset}'.")
    if task_name is None:
        chosen = task_names[0]
        print(f"=== No --task provided; defaulting to '{chosen}' ===")
        return chosen
    if task_name not in task_names:
        available = ", ".join(task_names)
        raise ValueError(f"Unknown task '{task_name}' for '{dataset}'. Available: {available}")
    return task_name


def _to_numpy_targets(y: pd.Series) -> np.ndarray:
    if y.empty:
        return np.array([])
    if y.dtype == "O":
        sample = y.dropna().head(1).tolist()
        if sample and isinstance(sample[0], (list, tuple, np.ndarray)):
            return np.stack([np.asarray(row) for row in y], axis=0)
    return y.to_numpy()


def _save_prediction_bundle(
    output_dir: Path,
    split_name: str,
    payload: dict[str, np.ndarray | None],
) -> None:
    arrays = {k: v for k, v in payload.items() if v is not None}
    if not arrays:
        return
    path = output_dir / f"predictions_{split_name}.npz"
    np.savez_compressed(path, **arrays)
    print(f"=== Saved predictions: {path} ===")


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


def _collect_env_metadata() -> dict[str, Any]:
    try:
        import importlib.metadata as importlib_metadata
    except Exception:  # pragma: no cover
        importlib_metadata = None

    packages = [
        "relbench",
        "tabpfn",
        "tabicl",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "duckdb",
        "pyarrow",
    ]
    versions: dict[str, str] = {}
    if importlib_metadata is not None:
        for pkg in packages:
            try:
                versions[pkg] = importlib_metadata.version(pkg)
            except Exception as exc:
                versions[pkg] = f"unavailable ({exc.__class__.__name__})"

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit,
        "packages": versions,
    }
