"""Minimal model builders for the standalone ContinuumBench repo."""

from __future__ import annotations

from typing import Literal

from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier, TabPFNRegressor

TABICL_CHECKPOINT_VERSION = "tabicl-classifier-v1.1-0506.ckpt"
TABICL_AUTO_MAX_TRAIN_ROWS = 50_000
SUPPORTED_MODEL_NAMES = frozenset({"tabicl", "tabpfn"})
TaskType = Literal["classification", "regression"]


def build_tabicl(
    verbose: bool = False,
    n_jobs: int | None = None,
    device: str | None = None,
    use_amp: bool = False,
) -> TabICLClassifier:
    kwargs: dict[str, object] = {
        "n_estimators": 16,
        "checkpoint_version": TABICL_CHECKPOINT_VERSION,
        "n_jobs": n_jobs,
        "use_amp": use_amp,
        "verbose": verbose,
    }
    if device:
        kwargs["device"] = device
    return TabICLClassifier(**kwargs)


def build_tabpfn(
    device: Literal["auto", "cpu", "cuda"] = "auto",
    ignore_pretraining_limits: bool = False,
) -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=16,
        device=device,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


def build_tabpfn_regressor(
    device: Literal["auto", "cpu", "cuda"] = "auto",
    ignore_pretraining_limits: bool = False,
) -> TabPFNRegressor:
    return TabPFNRegressor(
        n_estimators=16,
        device=device,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


def resolve_tabicl_device(
    device_spec: str | None,
    *,
    platform_name: str | None,
) -> str | None:
    spec = (device_spec or "auto").strip().lower()
    if spec == "auto":
        return "cpu" if platform_name == "Darwin" else None
    if spec in {"none", "default"}:
        return None
    return device_spec.strip() if device_spec is not None else None


def default_max_train_rows(model_name: str, explicit_cap: int | None = None) -> int | None:
    if explicit_cap is not None:
        return explicit_cap
    if model_name == "tabicl":
        return TABICL_AUTO_MAX_TRAIN_ROWS
    return None


def build_tabular_estimator(
    model_name: str,
    task_type: TaskType,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    tabicl_device: str | None = None,
    tabicl_use_amp: bool = False,
    ignore_pretraining_limits: bool = False,
):
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model: {model_name}. Allowed: {sorted(SUPPORTED_MODEL_NAMES)}"
        )

    if task_type == "regression":
        if model_name != "tabpfn":
            raise ValueError("Regression tasks require tabpfn.")
        return build_tabpfn_regressor(
            device=device,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )

    if model_name == "tabicl":
        return build_tabicl(
            device=tabicl_device,
            n_jobs=1,
            use_amp=tabicl_use_amp,
            verbose=False,
        )

    return build_tabpfn(
        device=device,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )
