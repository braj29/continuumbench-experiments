"""Minimal model builders for the standalone ContinuumBench repo."""

from __future__ import annotations

from typing import Literal

from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier


def build_tabicl(
    verbose: bool = False,
    n_jobs: int | None = None,
    device: str | None = None,
    use_amp: bool = False,
) -> TabICLClassifier:
    kwargs: dict[str, object] = {
        "n_estimators": 16,
        "use_hierarchical": True,
        "checkpoint_version": "tabicl-classifier-v1.1-0506.ckpt",
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
