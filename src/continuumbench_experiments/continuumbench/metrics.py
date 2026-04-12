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

# Chance-level performance for AUROC (random classifier on balanced data).
_AUROC_CHANCE: float = 0.5


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)


def _reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _get_gpu_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


@contextmanager
def compute_tracker() -> Iterable[MutableMapping[str, float]]:
    _reset_gpu_peak()
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
        payload["peak_gpu_vram_mb"] = _get_gpu_peak_mb()


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
    """Per-run unnormalized utility (higher is always better)."""
    if metric_name == "auroc":
        return score
    if metric_name == "mae":
        return -math.log(max(score, eps))
    raise ValueError(f"Unsupported metric: {metric_name}")


def _normalized_utilities(scores: np.ndarray, metric_name: str) -> np.ndarray:
    """Map raw metric scores to [0, 1]-range utilities for RIS computation.

    For AUROC:
        u_v = (score - chance) / (oracle - chance)
        where chance = 0.5 and oracle = max(scores).

    For MAE (lower is better):
        u_v = (chance - score) / (chance - oracle)
        where chance = max(scores) and oracle = min(scores).

    Both formulas match the paper's definition:
        u_v = (μ_v - μ_chance) / (μ_oracle - μ_chance)
    with the convention that "chance" and "oracle" are on the same scale
    (higher-is-better after the flip for MAE).
    """
    if metric_name == "auroc":
        chance = _AUROC_CHANCE
        oracle = float(np.max(scores))
        denom = oracle - chance
        if abs(denom) < 1e-12:
            return np.zeros_like(scores, dtype=float)
        return (scores - chance) / denom
    if metric_name == "mae":
        # chance = worst performer (highest MAE), oracle = best (lowest MAE)
        chance = float(np.max(scores))
        oracle = float(np.min(scores))
        denom = chance - oracle
        if abs(denom) < 1e-12:
            return np.ones_like(scores, dtype=float)
        return (chance - scores) / denom
    raise ValueError(f"Unsupported metric: {metric_name}")


def compute_ris(metric_scores: dict[str, float], metric_name: str) -> dict[str, float]:
    """Compute the Representation Invariance Score for a single task.

    RIS(T) = 1 - σ({u_v})  where u_v is the normalized utility of view v.

    RIS = 1 means all representations achieve equal normalized performance.
    Lower values indicate greater representation sensitivity.
    """
    scores = np.array(list(metric_scores.values()), dtype=float)
    utilities = _normalized_utilities(scores, metric_name)
    std = float(np.std(utilities))
    ris = 1.0 - std
    return {
        "ris": ris,
        "utility_std": std,
        "utility_min": float(np.min(utilities)),
        "utility_max": float(np.max(utilities)),
        "utility_gap": float(np.max(utilities) - np.min(utilities)),
    }


def compute_macro_ris(per_task_ris: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute macro-RIS as the mean RIS across a suite of tasks.

    Macro-RIS = (1/|T|) * Σ_j RIS(T_j)

    Returns the mean and standard deviation of per-task RIS values.
    """
    ris_values = [float(v["ris"]) for v in per_task_ris.values() if "ris" in v]
    if not ris_values:
        return {"macro_ris": float("nan"), "macro_ris_std": float("nan"), "n_tasks": 0}
    arr = np.array(ris_values, dtype=float)
    return {
        "macro_ris": float(np.mean(arr)),
        "macro_ris_std": float(np.std(arr)),
        "n_tasks": len(arr),
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
