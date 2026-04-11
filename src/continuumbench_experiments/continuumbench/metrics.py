from __future__ import annotations

import math
import time
import tracemalloc
from contextlib import contextmanager
from typing import Iterable, MutableMapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .specs import TemporalSplit


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)


@contextmanager
def compute_tracker() -> Iterable[MutableMapping[str, float]]:
    tracemalloc.start()
    start = time.perf_counter()
    payload: MutableMapping[str, float] = {}
    try:
        yield payload
    finally:
        end = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        payload["wall_clock_sec"] = end - start
        payload["peak_cpu_ram_mb"] = peak / (1024 * 1024)


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
    if metric_name == "auroc":
        return score
    if metric_name == "mae":
        return -math.log(max(score, eps))
    raise ValueError(f"Unsupported metric: {metric_name}")


def compute_ris(metric_scores: dict[str, float], metric_name: str) -> dict[str, float]:
    utilities = np.array(
        [metric_to_utility(value, metric_name) for value in metric_scores.values()],
        dtype=float,
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


def inject_future_only_signal(
    task_df: pd.DataFrame,
    seed_time_col: str,
    target_col: str,
    signal_name: str = "future_leak_signal",
) -> pd.DataFrame:
    df = task_df.copy()
    seed_times = pd.to_datetime(df[seed_time_col], utc=False)
    future_ts = seed_times + pd.to_timedelta(7, unit="D")
    df[signal_name] = np.where(df[target_col].notna(), future_ts.view("int64"), np.nan)
    return df
