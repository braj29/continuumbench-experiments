#!/usr/bin/env python3
"""Collect and summarise per-task result.json files from a sweep job.

Usage:
    python scripts/collect_sweep_results.py \\
        --results-dir outputs/snellius/<job_id> \\
        [--output-csv outputs/snellius/<job_id>/summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_results(results_dir: Path) -> list[dict]:
    rows = []
    for task_dir in sorted(results_dir.iterdir()):
        json_path = task_dir / "result.json"
        if not json_path.is_file():
            print(f"[warn] missing result: {json_path}", file=sys.stderr)
            continue
        data = json.loads(json_path.read_text())
        row = {
            "task": task_dir.name,
            "split": data.get("split", ""),
            "seed": data.get("seed", ""),
            "n_features": data.get("n_features", ""),
            "n_train": data.get("n_rows", {}).get("train", ""),
            "n_val": data.get("n_rows", {}).get("val", ""),
            "n_test": data.get("n_rows", {}).get("test", ""),
            "val_auroc": (data.get("metrics") or {}).get("val_auroc", ""),
            "test_auroc": (data.get("metrics") or {}).get("test_auroc", ""),
            "fit_time_sec": data.get("fit_time_sec", ""),
            "ignore_pretraining_limits": data.get("ignore_pretraining_limits", ""),
            "include_meta": data.get("include_meta", ""),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = _load_results(results_dir)
    if not rows:
        print("No result.json files found.", file=sys.stderr)
        sys.exit(1)

    fields = list(rows[0].keys())
    out_csv = args.output_csv or results_dir / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_csv}")

    # Print a quick summary table to stdout.
    print(f"\n{'task':<30} {'val_auroc':>10} {'test_auroc':>11} {'fit_s':>8}")
    print("-" * 62)
    for r in rows:
        val  = f"{r['val_auroc']:.4f}"  if isinstance(r["val_auroc"],  float) else "—"
        test = f"{r['test_auroc']:.4f}" if isinstance(r["test_auroc"], float) else "—"
        fit  = f"{r['fit_time_sec']:.1f}s" if isinstance(r["fit_time_sec"], float) else "—"
        print(f"{r['task']:<30} {val:>10} {test:>11} {fit:>8}")

    # Aggregate across seeds per split.
    from collections import defaultdict
    import statistics

    by_split: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if isinstance(r["test_auroc"], float):
            by_split[r["split"]].append(r["test_auroc"])

    if by_split:
        print(f"\n{'split':<20} {'n_seeds':>8} {'mean_test':>10} {'std_test':>10}")
        print("-" * 50)
        for split, vals in sorted(by_split.items()):
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{split:<20} {len(vals):>8} {mean:>10.4f} {std:>10.4f}")


if __name__ == "__main__":
    main()
