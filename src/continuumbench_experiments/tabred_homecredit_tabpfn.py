from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .models.tabular import build_tabpfn, resolve_tabpfn_device

_PARTS = ("train", "val", "test")
_BASE_BLOCKS = ("X_num", "X_bin", "X_cat")


def _list_available_splits(data_dir: Path) -> list[str]:
    return sorted(
        p.name.removeprefix("split-")
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("split-")
    )


def _load_indices(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    split_dir = data_dir / f"split-{split}"
    if not split_dir.is_dir():
        available = _list_available_splits(data_dir)
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}. Available splits: {available}"
        )
    idx: dict[str, np.ndarray] = {}
    for part in _PARTS:
        part_path = split_dir / f"{part}_idx.npy"
        if not part_path.is_file():
            raise FileNotFoundError(f"Missing split file: {part_path}")
        idx[part] = np.load(part_path, allow_pickle=False).astype(np.int64, copy=False)
    return idx


def _load_feature_blocks(data_dir: Path, include_meta: bool) -> dict[str, np.ndarray]:
    blocks = list(_BASE_BLOCKS) + (["X_meta"] if include_meta else [])
    loaded: dict[str, np.ndarray] = {}
    for block in blocks:
        path = data_dir / f"{block}.npy"
        if not path.is_file():
            if block == "X_meta":
                continue
            raise FileNotFoundError(f"Missing required feature block: {path}")
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        loaded[block] = arr
    if not loaded:
        raise RuntimeError("No feature blocks loaded.")
    return loaded


def _assemble_matrix(
    blocks: dict[str, np.ndarray],
    idx: np.ndarray,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for block_name in ("X_num", "X_bin", "X_cat", "X_meta"):
        block = blocks.get(block_name)
        if block is None:
            continue
        values = block[idx]
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        pieces.append(np.asarray(values, dtype=np.float32))
    if not pieces:
        raise RuntimeError("No feature matrices were assembled.")
    return np.concatenate(pieces, axis=1)


def _impute_like_train(
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> np.ndarray:
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32, copy=False)

    x = np.asarray(x_other, dtype=np.float32).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = medians[np.where(bad)[1]]
    return x


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _fit_and_eval(
    data_dir: Path,
    split: str,
    include_meta: bool,
    device: str,
    ignore_pretraining_limits: bool,
    max_train_rows: int | None,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    blocks = _load_feature_blocks(data_dir, include_meta=include_meta)
    idx = _load_indices(data_dir, split)

    y_all = np.load(data_dir / "Y.npy", allow_pickle=False).reshape(-1)
    y_all = np.asarray(y_all, dtype=np.int64)

    x_by_part = {part: _assemble_matrix(blocks, idx[part]) for part in _PARTS}
    y_by_part = {part: y_all[idx[part]] for part in _PARTS}

    if max_train_rows is not None and len(x_by_part["train"]) > max_train_rows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(x_by_part["train"]), size=max_train_rows, replace=False))
        x_by_part["train"] = x_by_part["train"][keep]
        y_by_part["train"] = y_by_part["train"][keep]

    # Median-impute by train statistics for all parts.
    x_train = _impute_like_train(x_by_part["train"], x_by_part["train"])
    x_val = _impute_like_train(x_by_part["train"], x_by_part["val"])
    x_test = _impute_like_train(x_by_part["train"], x_by_part["test"])

    if np.unique(y_by_part["train"]).size < 2:
        raise RuntimeError(
            "Train split has only one class; TabPFN binary classification requires 2 classes."
        )

    result: dict[str, Any] = {
        "dataset_dir": str(data_dir),
        "split": split,
        "include_meta": include_meta,
        "device": device,
        "ignore_pretraining_limits": bool(ignore_pretraining_limits),
        "n_features": int(x_train.shape[1]),
        "n_rows": {part: int(len(y_by_part[part])) for part in _PARTS},
        "positive_rate": {
            part: float(np.mean(y_by_part[part])) if len(y_by_part[part]) else float("nan")
            for part in _PARTS
        },
    }

    if dry_run:
        result["dry_run"] = True
        return result

    model = build_tabpfn(
        device=device, ignore_pretraining_limits=ignore_pretraining_limits
    )

    started = time.perf_counter()
    model.fit(x_train, y_by_part["train"])
    fit_sec = time.perf_counter() - started

    prob_val = model.predict_proba(x_val)
    prob_test = model.predict_proba(x_test)

    val_score = prob_val[:, 1] if prob_val.ndim == 2 and prob_val.shape[1] > 1 else prob_val.ravel()
    test_score = (
        prob_test[:, 1] if prob_test.ndim == 2 and prob_test.shape[1] > 1 else prob_test.ravel()
    )

    result["fit_time_sec"] = float(fit_sec)
    result["metrics"] = {
        "val_auroc": _safe_auroc(y_by_part["val"], val_score),
        "test_auroc": _safe_auroc(y_by_part["test"], test_score),
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabPFN on TabReD-formatted homecredit-default data "
            "(X_num/X_bin/X_cat/Y + split-* indices)."
        )
    )
    parser.add_argument(
        "--tabred-data-dir",
        type=Path,
        required=True,
        help=(
            "Path to TabReD dataset directory, e.g. /path/to/tabred/data/homecredit-default."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="default",
        help="Split name to use (default: default).",
    )
    parser.add_argument(
        "--list-splits",
        action="store_true",
        help="List available split-* directories and exit.",
    )
    parser.add_argument(
        "--include-meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include X_meta features in TabPFN input.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="TabPFN device: auto|cpu|cuda (auto->cpu on macOS).",
    )
    parser.add_argument(
        "--tabpfn-ignore-pretraining-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward ignore_pretraining_limits to TabPFN (default: true).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional random cap for train rows before fitting.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize data/split without fitting TabPFN.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/tabred_homecredit_tabpfn/result.json"),
        help="Where to write JSON results.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data_dir = args.tabred_data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--tabred-data-dir does not exist: {data_dir}")

    if args.list_splits:
        print(json.dumps({"available_splits": _list_available_splits(data_dir)}, indent=2))
        return

    resolved_device = resolve_tabpfn_device(
        args.device,
        platform_name=platform.system(),
    )
    result = _fit_and_eval(
        data_dir=data_dir,
        split=args.split,
        include_meta=bool(args.include_meta),
        device=resolved_device,
        ignore_pretraining_limits=bool(args.tabpfn_ignore_pretraining_limits),
        max_train_rows=args.max_train_rows,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
    )

    out_path = args.output_json.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
