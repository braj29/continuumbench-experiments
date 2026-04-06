"""Adapters to convert GraphBench datasets into tabular samples."""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class GraphBenchSplits:
    name: str
    train: Optional[object]
    valid: Optional[object]
    test: Optional[object]


def load_graphbench_splits(
    root: str,
    dataset_names: str | Sequence[str],
    generate_fallback: bool = False,
    solver: Optional[str] = None,
    use_satzilla_features: bool = False,
    download_timeout: Optional[int] = None,
    download_retries: Optional[int] = None,
) -> Sequence[GraphBenchSplits]:
    Loader = _resolve_graphbench_loader()
    _patch_graphbench_downloads(download_timeout, download_retries)
    loader = Loader(
        root,
        dataset_names,
        generate_fallback=generate_fallback,
        solver=solver,
        use_satzilla_features=use_satzilla_features,
    )

    dataset_ids = loader._get_dataset_names()
    if not dataset_ids:
        if isinstance(dataset_names, str):
            dataset_ids = [dataset_names]
        else:
            dataset_ids = list(dataset_names)
    dataset_ids = [_normalize_graphbench_dataset_name(name) for name in dataset_ids]
    datasets = []
    for name in dataset_ids:
        if name.startswith("bluesky_"):
            cached = _load_bluesky_cached(Path(root), name)
            if cached is not None:
                datasets.append(cached)
                continue
        if name.startswith("sat_"):
            cached = _load_sat_csv(Path(root), name)
            if cached is not None:
                datasets.append(cached)
                continue
        try:
            if name.startswith("electronic_circuits"):
                datasets.append(_load_electronic_circuits(loader, root, name))
            else:
                datasets.append(loader._loader(name))
        except Exception as exc:
            fallback = _fallback_graphbench_dataset(root, name, exc)
            if fallback is None:
                raise
            datasets.append(fallback)

    splits: list[GraphBenchSplits] = []
    for name, data in zip(dataset_ids, datasets):
        splits.append(
            GraphBenchSplits(
                name=name,
                train=data.get("train"),
                valid=data.get("valid"),
                test=data.get("test"),
            )
        )
    return splits


def _load_electronic_circuits(loader, root: str, dataset_name: str) -> dict:
    """Load electronic circuits splits without deleting __MACOSX artifacts."""
    from graphbench.datasets.electroniccircuits import ECDataset

    train_dataset = ECDataset(
        root=root,
        name=dataset_name,
        pre_transform=loader.pre_transform,
        transform=loader.transform,
        split="train",
        generate=loader.generate,
        cleanup_raw=False,
    )
    valid_dataset = ECDataset(
        root=root,
        name=dataset_name,
        pre_transform=loader.pre_transform,
        transform=loader.transform,
        split="val",
        generate=loader.generate,
        cleanup_raw=False,
    )
    test_dataset = ECDataset(
        root=root,
        name=dataset_name,
        pre_transform=loader.pre_transform,
        transform=loader.transform,
        split="test",
        generate=loader.generate,
        cleanup_raw=False,
    )
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def _normalize_graphbench_dataset_name(name: str) -> str:
    lowered = name.lower()
    alias = {
        "bluesky_retweets": "bluesky_reposts",
    }
    return alias.get(lowered, lowered)


def _patch_graphbench_downloads(
    timeout_override: Optional[int],
    max_retries_override: Optional[int],
) -> None:
    try:
        import graphbench.helpers.download as gb_download
    except Exception:
        return

    if getattr(gb_download, "_tabfm_patched", False):
        return

    orig_stream = gb_download._stream_download
    orig_download_and_unpack = gb_download._download_and_unpack
    orig_unpack_xz = gb_download._unpack_xz

    def _stream_download(
        url,
        dest,
        logger,
        chunk_size=1 << 20,
        timeout=60,
        max_retries=5,
        cooldown_seconds=5,
    ):
        import requests

        use_timeout = timeout_override if timeout_override is not None else timeout
        use_retries = max_retries_override if max_retries_override is not None else max_retries

        def _remote_size(target_url: str) -> Optional[int]:
            try:
                resp = requests.head(target_url, allow_redirects=True, timeout=use_timeout)
            except Exception:
                return None
            length = resp.headers.get("Content-Length")
            if length is None:
                return None
            try:
                return int(length)
            except ValueError:
                return None

        def _download_once(
            headers: dict[str, str],
            mode: str,
        ) -> int:
            with requests.get(url, stream=True, timeout=use_timeout, headers=headers) as resp:
                if headers and resp.status_code == 200:
                    return -1
                if resp.status_code == 416:
                    return 0
                resp.raise_for_status()
                with open(dest, mode) as handle:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            return 1

        expected = _remote_size(url)

        for attempt in range(1, use_retries + 1):
            try:
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                existing = dest.stat().st_size if dest.exists() else 0
                if expected is not None and existing >= expected:
                    logger.info("Found existing download: %s", dest)
                    return
                headers: dict[str, str] = {}
                mode = "wb"
                if existing > 0:
                    headers["Range"] = f"bytes={existing}-"
                    mode = "ab"
                result = _download_once(headers, mode)
                if result == -1:
                    headers = {}
                    mode = "wb"
                    result = _download_once(headers, mode)
                if result == 0 and expected is not None and existing >= expected:
                    return
                if expected is not None and dest.exists() and dest.stat().st_size < expected:
                    raise IOError("Download incomplete.")
                return
            except requests.RequestException as exc:
                if attempt == use_retries:
                    raise
                logger.warning(
                    "Download attempt %d/%d failed (%s); retrying in %ds",
                    attempt,
                    use_retries,
                    exc,
                    cooldown_seconds,
                )
            except Exception as exc:
                if attempt == use_retries:
                    raise
                logger.warning(
                    "Download attempt %d/%d failed (%s); retrying in %ds",
                    attempt,
                    use_retries,
                    exc,
                    cooldown_seconds,
                )
            import time

            time.sleep(cooldown_seconds)

        return orig_stream(
            url,
            dest,
            logger,
            chunk_size=chunk_size,
            timeout=use_timeout,
            max_retries=use_retries,
            cooldown_seconds=cooldown_seconds,
        )

    def _download_and_unpack(source, raw_dir, processed_dir, logger) -> None:
        processed_path = Path(processed_dir)
        if processed_path.suffix and processed_path.exists() and processed_path.is_file():
            logger.info("Found existing processed file: %s", processed_path)
            return
        if processed_path.suffix:
            processed_dir = processed_path.parent
        return orig_download_and_unpack(source, raw_dir, processed_dir, logger)

    def _unpack_xz(src: Path, dest_dir: Optional[Path] = None):
        import lzma

        if dest_dir is None:
            return orig_unpack_xz(src, dest_dir)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / src.with_suffix("").name
        with lzma.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
                f_out.write(chunk)
        src.unlink(missing_ok=True)

    gb_download._stream_download = _stream_download
    gb_download._download_and_unpack = _download_and_unpack
    gb_download._unpack_xz = _unpack_xz
    gb_download._tabfm_patched = True


class _InMemoryDatasetLite:
    def __init__(self, data, slices, data_cls) -> None:
        if isinstance(data, dict) and hasattr(data_cls, "from_dict"):
            data = data_cls.from_dict(data)
        self._data = data
        self._data_cls = data_cls
        self.slices = slices
        self._data_list = None

    def __len__(self) -> int:
        if self.slices is None:
            return 1
        first_slice = _first_slice(self.slices)
        if first_slice is None:
            return 0
        return len(first_slice) - 1

    def get(self, idx: int):
        from torch_geometric.data.separate import separate

        if self.__len__() == 1:
            return copy.copy(self._data)
        if self._data_list is None:
            self._data_list = self.__len__() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        self._data_list[idx] = copy.copy(data)
        return data


class _SubsetDataset:
    def __init__(self, base, indices: Sequence[int]) -> None:
        self._base = base
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def get(self, idx: int):
        return self._base.get(self._indices[idx])


class _ListDataset:
    def __init__(self, items: Sequence[object]) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def get(self, idx: int):
        return self._items[idx]


def _first_slice(slices):
    if isinstance(slices, dict):
        for value in slices.values():
            return _first_slice(value)
    return slices


def _load_inmemory_dataset_file(path: Path):
    data, slices, data_cls = torch.load(path, weights_only=False)
    return _InMemoryDatasetLite(data, slices, data_cls)


def _split_dataset(dataset, seed: int = 42):
    n = len(dataset)
    if n == 0:
        return None, None, None
    if n == 1:
        return dataset, None, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train_end = max(1, int(0.7 * n))
    valid_end = train_end + max(1, int(0.15 * n))
    valid_end = min(valid_end, n)
    train_idx = order[:train_end]
    valid_idx = order[train_end:valid_end]
    test_idx = order[valid_end:]
    train = _SubsetDataset(dataset, train_idx)
    valid = _SubsetDataset(dataset, valid_idx) if len(valid_idx) > 0 else None
    test = _SubsetDataset(dataset, test_idx) if len(test_idx) > 0 else None
    return train, valid, test


def _load_bluesky_cached(root: Path, dataset_name: str):
    processed_dir = root / "bluesky" / dataset_name / "processed"
    splits: dict[str, object | None] = {}
    for split in ("train", "valid", "test"):
        processed_path = processed_dir / f"{dataset_name}_{split}.pt"
        if processed_path.exists():
            splits[split] = _load_inmemory_dataset_file(processed_path)
            continue
        raw = _load_bluesky_raw_xy(root, dataset_name, split)
        splits[split] = raw
    if any(value is not None for value in splits.values()):
        return {"train": splits["train"], "valid": splits["valid"], "test": splits["test"]}
    return None


def _load_bluesky_raw_xy(root: Path, dataset_name: str, split: str):
    import torch

    suffix = dataset_name.split("_")[-1]
    raw_dir = root / "bluesky" / dataset_name / "raw"
    split_name = "val" if split == "valid" else split
    x_path = raw_dir / f"x_{suffix}_{split_name}.pt"
    y_path = raw_dir / f"y_{suffix}_{split_name}.pt"
    if not x_path.exists() or not y_path.exists():
        return None
    x = torch.load(x_path, weights_only=False)
    y = torch.load(y_path, weights_only=False)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = types.SimpleNamespace(
        x=x,
        y=y,
        edge_index=edge_index,
        num_nodes=int(getattr(x, "shape", [0])[0] or 0),
    )
    return [data]


def _load_co_cached(root: Path, dataset_name: str):
    path = root / "co" / dataset_name / "processed" / "data.pt"
    if not path.exists():
        return None
    dataset = _load_inmemory_dataset_file(path)
    train, valid, test = _split_dataset(dataset)
    return {"train": train, "valid": valid, "test": test}


def _load_sat_csv(root: Path, dataset_name: str):
    if not dataset_name.startswith("sat_"):
        return None
    parts = dataset_name.split("_")
    if len(parts) < 3:
        return None
    task_type = parts[-1]
    if task_type != "as":
        return None
    csv_dir = root / "sat_csv"
    features_path = csv_dir / "features.csv"
    instances_path = csv_dir / "instances_new.csv"
    runs_path = csv_dir / "runs.csv"
    if not (features_path.exists() and instances_path.exists() and runs_path.exists()):
        return None

    instances = pd.read_csv(instances_path)
    instances = instances[(instances["n_vars"] < 3000) & (instances["n_clauses"] < 15000)]
    if instances.empty:
        return None

    features = pd.read_csv(features_path)
    merged = instances.merge(features, on="filename", how="inner")
    if merged.empty:
        return None

    import os

    max_rows = int(os.environ.get("TABFM_SAT_CSV_MAX_ROWS", "5000"))
    if max_rows > 0 and len(merged) > max_rows:
        merged = merged.sample(n=max_rows, random_state=42)

    runs = pd.read_csv(runs_path)
    runs_all = instances.merge(runs, on="filename", how="inner")
    if runs_all.empty:
        return None
    runs_all = runs_all[runs_all["filename"].isin(merged["filename"])]
    if runs_all.empty:
        return None
    runs_all.loc[runs_all["time"] < 0.05, "time"] = 0.05
    runs_all.loc[~runs_all["status"].str.contains("SAT|UNSAT"), "time"] = 50_000
    pivot = runs_all.pivot_table(index="filename", columns="solver_name", values="time")
    order = pivot.sum().sort_values().index.tolist()
    if not order:
        return None

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if "n_vars" not in numeric_cols:
        numeric_cols.insert(0, "n_vars")
    if "n_clauses" not in numeric_cols:
        numeric_cols.insert(1, "n_clauses")

    data_list: list[object] = []
    runs_by_file = {name: frame for name, frame in runs_all.groupby("filename")}
    for _, row in merged.iterrows():
        filename = row["filename"]
        run_block = runs_by_file.get(filename)
        if run_block is None:
            continue
        y_vals: list[float] = []
        for solver in order:
            match = run_block[run_block["solver_name"] == solver]
            if match.empty:
                t = 50_000.0
                status = "TIMEOUT"
            else:
                t = float(match["time"].values[0])
                status = match["status"].values[0]
            if t < 0.05:
                t = 0.05
            if status not in {"SAT", "UNSAT"}:
                t = 50_000.0
            y_vals.append(t)
        y = np.asarray(y_vals, dtype=np.float32).reshape(1, -1)
        feats = row[numeric_cols].to_numpy(dtype=float).reshape(1, -1)
        data_list.append(
            types.SimpleNamespace(
                graph_attr=feats,
                y=y,
                num_nodes=1,
            )
        )

    if not data_list:
        return None
    dataset = _ListDataset(data_list)
    train, valid, test = _split_dataset(dataset)
    return {"train": train, "valid": valid, "test": test}


def _fallback_graphbench_dataset(root: str, dataset_name: str, exc: Exception):
    root_path = Path(root)
    print(
        f"=== Warning: GraphBench loader failed for {dataset_name}: "
        f"{exc.__class__.__name__}: {exc} ==="
    )
    if dataset_name.startswith("bluesky_"):
        cached = _load_bluesky_cached(root_path, dataset_name)
        if cached is not None:
            print(f"=== Using cached Bluesky features for {dataset_name} ===")
            return cached
    if dataset_name.startswith("sat_"):
        cached = _load_sat_csv(root_path, dataset_name)
        if cached is not None:
            print(f"=== Using SAT CSV features for {dataset_name} ===")
            return cached
    if dataset_name.startswith("co_"):
        cached = _load_co_cached(root_path, dataset_name)
        if cached is not None:
            print(f"=== Using cached CO dataset for {dataset_name} ===")
            return cached
    return None


def _resolve_graphbench_loader():
    """Load GraphBench Loader without importing graphbench.__init__ (torchmetrics dependency)."""
    try:
        import graphbench
        if hasattr(graphbench, "Loader"):
            return graphbench.Loader
    except Exception:
        pass

    try:
        import importlib.metadata as importlib_metadata
        import importlib.util
        import types
        import sys
        from pathlib import Path
    except Exception as exc:
        raise ImportError(
            "GraphBench is not installed. Install with `pip install graphbench-lib`."
        ) from exc

    dist = None
    for name in ("graphbench-lib", "graphbench"):
        try:
            dist = importlib_metadata.distribution(name)
            break
        except importlib_metadata.PackageNotFoundError:
            continue

    if dist is None:
        raise ImportError(
            "GraphBench is not installed. Install with `pip install graphbench-lib`."
        )

    loader_path = None
    package_path = None
    for file in dist.files or []:
        path = file.as_posix()
        if path.endswith("graphbench/loader.py"):
            loader_path = dist.locate_file(file)
        if path.endswith("graphbench/__init__.py"):
            package_path = Path(dist.locate_file(file)).parent

    if loader_path is None or package_path is None:
        raise ImportError("Unable to locate GraphBench loader module in the install.")

    if "graphbench" in sys.modules:
        sys.modules.pop("graphbench", None)
    pkg = types.ModuleType("graphbench")
    pkg.__file__ = str(package_path / "__init__.py")
    pkg.__path__ = [str(package_path)]
    pkg.__package__ = "graphbench"
    sys.modules["graphbench"] = pkg

    spec = importlib.util.spec_from_file_location("graphbench.loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load GraphBench loader module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["graphbench.loader"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Loader"):
        raise ImportError("GraphBench loader module missing Loader class.")
    return module.Loader


def load_graphbench_evaluator(metric_name: str):
    Evaluator = _resolve_graphbench_evaluator()
    return Evaluator(metric_name)


def _resolve_graphbench_evaluator():
    """Load GraphBench Evaluator while handling torchmetrics import failures."""
    try:
        import graphbench
        if hasattr(graphbench, "Evaluator"):
            return graphbench.Evaluator
    except Exception:
        pass

    try:
        import torchmetrics  # noqa: F401
    except Exception:
        _install_torchmetrics_shim()

    try:
        import importlib.metadata as importlib_metadata
        import importlib.util
        import types
        import sys
        from pathlib import Path
    except Exception as exc:
        raise ImportError(
            "GraphBench is not installed. Install with `pip install graphbench-lib`."
        ) from exc

    dist = None
    for name in ("graphbench-lib", "graphbench"):
        try:
            dist = importlib_metadata.distribution(name)
            break
        except importlib_metadata.PackageNotFoundError:
            continue

    if dist is None:
        raise ImportError(
            "GraphBench is not installed. Install with `pip install graphbench-lib`."
        )

    evaluator_path = None
    package_path = None
    for file in dist.files or []:
        path = file.as_posix()
        if path.endswith("graphbench/evaluator.py"):
            evaluator_path = dist.locate_file(file)
        if path.endswith("graphbench/__init__.py"):
            package_path = Path(dist.locate_file(file)).parent

    if evaluator_path is None or package_path is None:
        raise ImportError("Unable to locate GraphBench evaluator module in the install.")

    if "graphbench" in sys.modules:
        sys.modules.pop("graphbench", None)
    pkg = types.ModuleType("graphbench")
    pkg.__file__ = str(package_path / "__init__.py")
    pkg.__path__ = [str(package_path)]
    pkg.__package__ = "graphbench"
    sys.modules["graphbench"] = pkg

    spec = importlib.util.spec_from_file_location("graphbench.evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load GraphBench evaluator module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["graphbench.evaluator"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Evaluator"):
        raise ImportError("GraphBench evaluator module missing Evaluator class.")
    return module.Evaluator


def _install_torchmetrics_shim() -> None:
    """Install a minimal torchmetrics shim to avoid torchvision dependency issues."""
    import sys
    import types

    if "torchmetrics" in sys.modules:
        return

    import torch

    def _to_tensor(values):
        if torch.is_tensor(values):
            return values
        return torch.as_tensor(values)

    def _flatten(values):
        values = _to_tensor(values)
        if values.dim() > 1:
            return values.view(-1)
        return values

    def _binary_preds(preds, threshold=0.5):
        preds = _to_tensor(preds)
        if preds.dim() == 2 and preds.shape[1] == 2:
            return preds.argmax(dim=1)
        preds = _flatten(preds).float()
        return (preds >= threshold).to(torch.int64)

    def _binary_target(target):
        target = _flatten(target).to(torch.int64)
        return target

    class Accuracy:
        def __init__(self, task="binary", threshold=0.5):
            self.threshold = threshold

        def __call__(self, preds, target):
            preds = _binary_preds(preds, self.threshold)
            target = _binary_target(target)
            if preds.numel() == 0:
                return torch.tensor(0.0)
            return (preds == target).float().mean()

    class F1Score:
        def __init__(self, task="binary", threshold=0.5):
            self.threshold = threshold

        def __call__(self, preds, target):
            preds = _binary_preds(preds, self.threshold)
            target = _binary_target(target)
            tp = ((preds == 1) & (target == 1)).sum().float()
            fp = ((preds == 1) & (target == 0)).sum().float()
            fn = ((preds == 0) & (target == 1)).sum().float()
            denom = (2 * tp + fp + fn)
            if denom == 0:
                return torch.tensor(0.0)
            return (2 * tp) / denom

    class SpearmanCorrCoef:
        def __call__(self, preds, target):
            preds = _flatten(preds).double()
            target = _flatten(target).double()
            if preds.numel() == 0:
                return torch.tensor(0.0)

            def _rank(x):
                sorted_vals, sorted_idx = torch.sort(x)
                unique_vals, counts = torch.unique_consecutive(sorted_vals, return_counts=True)
                counts = counts.to(torch.long)
                ends = torch.cumsum(counts, dim=0)
                starts = ends - counts + 1
                avg_ranks = (starts.double() + ends.double()) / 2.0
                ranks_sorted = torch.repeat_interleave(avg_ranks, counts)
                ranks = torch.empty_like(ranks_sorted)
                ranks[sorted_idx] = ranks_sorted
                return ranks

            r_pred = _rank(preds)
            r_true = _rank(target)
            rp = r_pred - r_pred.mean()
            rt = r_true - r_true.mean()
            num = (rp * rt).sum()
            denom = torch.sqrt((rp**2).sum() * (rt**2).sum())
            if denom == 0:
                return torch.tensor(0.0)
            return num / denom

    class R2Score:
        def __call__(self, preds, target):
            preds = _flatten(preds).double()
            target = _flatten(target).double()
            if preds.numel() == 0:
                return torch.tensor(0.0)
            ss_res = ((target - preds) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            if ss_tot == 0:
                return torch.tensor(0.0)
            return 1 - ss_res / ss_tot

    shim = types.ModuleType("torchmetrics")
    shim.Accuracy = Accuracy
    shim.F1Score = F1Score
    shim.SpearmanCorrCoef = SpearmanCorrCoef
    shim.R2Score = R2Score
    sys.modules["torchmetrics"] = shim


def iter_graphbench_data(dataset: Optional[object]) -> Iterator[object]:
    if dataset is None:
        return iter(())
    if hasattr(dataset, "get") and hasattr(dataset, "__len__"):
        return (dataset.get(idx) for idx in range(len(dataset)))
    return iter(dataset)


def extract_node_samples(
    dataset: Optional[object],
    max_node_features: Optional[int] = None,
    add_degree: bool = False,
    target_index: Optional[int] = None,
    tabular_mode: str = "stats",
    allow_multi_target: bool = False,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    if tabular_mode == "raw":
        return _extract_node_samples_raw(
            dataset, max_node_features, target_index, allow_multi_target
        )

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for data in iter_graphbench_data(dataset):
        x = getattr(data, "x", None)
        y = _resolve_node_targets(data)
        if y is None:
            raise ValueError("Node-level task requires `data.y` labels.")
        y_arr = _select_target(_to_numpy(y), target_index, allow_multi_target)
        x_arr = _node_features(data, max_node_features=max_node_features, add_degree=add_degree)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Node labels do not match features: {x_arr.shape[0]} vs {y_arr.shape[0]}"
            )
        features.append(x_arr)
        labels.append(y_arr)

    return _to_dataframe(features), _to_labels(labels)


def extract_edge_samples(
    dataset: Optional[object],
    max_node_features: Optional[int] = None,
    include_edge_attr: bool = True,
    add_degree: bool = False,
    target_index: Optional[int] = None,
    tabular_mode: str = "stats",
    allow_multi_target: bool = False,
    auto_edge_labels: bool = False,
    edge_neg_ratio: float = 1.0,
    edge_sample_cap: Optional[int] = None,
    edge_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    if tabular_mode == "raw":
        return _extract_edge_samples_raw(
            dataset,
            include_edge_attr,
            target_index,
            allow_multi_target,
            auto_edge_labels,
            edge_neg_ratio,
            edge_sample_cap,
            edge_seed,
        )

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for graph_idx, data in enumerate(iter_graphbench_data(dataset)):
        edge_index, edge_labels, edge_attr = _edge_labels(
            data,
            auto_edge_labels=auto_edge_labels,
            edge_neg_ratio=edge_neg_ratio,
            edge_sample_cap=edge_sample_cap,
            seed=edge_seed + graph_idx,
        )
        edge_labels = _select_target(edge_labels, target_index, allow_multi_target)
        x_arr = _edge_features(
            data,
            edge_index=edge_index,
            max_node_features=max_node_features,
            include_edge_attr=include_edge_attr,
            edge_attr=edge_attr,
            add_degree=add_degree,
        )
        if x_arr.shape[0] != edge_labels.shape[0]:
            raise ValueError(
                f"Edge labels do not match features: {x_arr.shape[0]} vs {edge_labels.shape[0]}"
            )
        features.append(x_arr)
        labels.append(edge_labels)

    return _to_dataframe(features), _to_labels(labels)


def extract_graph_samples(
    dataset: Optional[object],
    max_node_features: Optional[int] = None,
    max_edge_features: Optional[int] = None,
    target_index: Optional[int] = None,
    tabular_mode: str = "stats",
    allow_multi_target: bool = False,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    if tabular_mode == "raw":
        return _extract_graph_samples_raw(
            dataset, max_node_features, target_index, allow_multi_target
        )

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for data in iter_graphbench_data(dataset):
        y = _resolve_graph_targets(data)
        if y is None:
            raise ValueError("Graph-level task requires `data.y` labels.")
        y_arr = _select_target(_to_numpy(y), target_index, allow_multi_target)
        if allow_multi_target:
            if y_arr.ndim == 1 and y_arr.shape[0] != 1:
                raise ValueError("Graph-level task expects one label per graph.")
            if y_arr.ndim == 2 and y_arr.shape[0] != 1:
                raise ValueError("Graph-level task expects one label per graph.")
        elif y_arr.ndim != 1 or y_arr.shape[0] != 1:
            raise ValueError("Graph-level task expects one label per graph.")
        feat = _graph_features(
            data,
            max_node_features=max_node_features,
            max_edge_features=max_edge_features,
        )
        features.append(feat.reshape(1, -1))
        if allow_multi_target and y_arr.ndim == 2:
            labels.append(y_arr)
        else:
            labels.append(y_arr.reshape(-1))

    return _to_dataframe(features), _to_labels(labels)


def _extract_node_samples_raw(
    dataset: Optional[object],
    max_node_features: Optional[int],
    target_index: Optional[int],
    allow_multi_target: bool,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for graph_idx, data in enumerate(iter_graphbench_data(dataset)):
        y = _resolve_node_targets(data)
        if y is None:
            raise ValueError("Node-level task requires `data.y` labels.")
        y_arr = _select_target(_to_numpy(y), target_index, allow_multi_target)
        if y_arr.ndim == 0:
            y_arr = y_arr.reshape(1)
        elif y_arr.ndim not in (1, 2):
            raise ValueError(f"Unsupported node label shape: {y_arr.shape}")
        num_nodes = y_arr.shape[0]

        node_ids = np.arange(num_nodes)
        frame = pd.DataFrame(
            {
                "graph_id": np.full(num_nodes, str(graph_idx)),
                "node_id": node_ids.astype(str),
            }
        )
        x = getattr(data, "x", None)
        if x is not None:
            x_arr = _as_2d_tensor(x)
            if x_arr.shape[0] != num_nodes:
                print(
                    "=== Warning: node features do not match label count; "
                    "dropping raw node features ==="
                )
            else:
                if max_node_features is not None:
                    x_arr = x_arr[:, :max_node_features]
                x_np = _to_numpy(x_arr)
                for idx in range(x_np.shape[1]):
                    frame[f"x{idx}"] = x_np[:, idx]

        frames.append(frame)
        labels.append(y_arr)

    return _concat_frames(frames), _to_labels(labels)


def _extract_edge_samples_raw(
    dataset: Optional[object],
    include_edge_attr: bool,
    target_index: Optional[int],
    allow_multi_target: bool,
    auto_edge_labels: bool,
    edge_neg_ratio: float,
    edge_sample_cap: Optional[int],
    edge_seed: int,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for graph_idx, data in enumerate(iter_graphbench_data(dataset)):
        edge_index, edge_labels, edge_attr = _edge_labels(
            data,
            auto_edge_labels=auto_edge_labels,
            edge_neg_ratio=edge_neg_ratio,
            edge_sample_cap=edge_sample_cap,
            seed=edge_seed + graph_idx,
        )
        edge_labels = _select_target(edge_labels, target_index, allow_multi_target)
        if edge_labels.ndim == 0:
            edge_labels = edge_labels.reshape(1)
        elif edge_labels.ndim not in (1, 2):
            raise ValueError(f"Unsupported edge label shape: {edge_labels.shape}")
        if edge_index.shape[1] != edge_labels.shape[0]:
            raise ValueError("Edge labels do not match edge count.")

        src = _to_numpy(edge_index[0]).astype(int)
        dst = _to_numpy(edge_index[1]).astype(int)
        frame = pd.DataFrame(
            {
                "graph_id": np.full(edge_labels.shape[0], str(graph_idx)),
                "src_id": src.astype(str),
                "dst_id": dst.astype(str),
            }
        )
        if include_edge_attr and edge_attr is not None:
            edge_attr = _as_2d_tensor(edge_attr)
            if edge_attr.shape[0] == edge_labels.shape[0]:
                edge_np = _to_numpy(edge_attr)
                for idx in range(edge_np.shape[1]):
                    frame[f"e{idx}"] = edge_np[:, idx]

        frames.append(frame)
        labels.append(edge_labels)

    return _concat_frames(frames), _to_labels(labels)


def _extract_graph_samples_raw(
    dataset: Optional[object],
    max_graph_features: Optional[int],
    target_index: Optional[int],
    allow_multi_target: bool,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    warned_no_attr = False
    for graph_idx, data in enumerate(iter_graphbench_data(dataset)):
        y = _resolve_graph_targets(data)
        if y is None:
            raise ValueError("Graph-level task requires `data.y` labels.")
        y_arr = _select_target(_to_numpy(y), target_index, allow_multi_target)
        if y_arr.ndim == 0:
            y_arr = y_arr.reshape(1)
        if allow_multi_target:
            if y_arr.ndim == 1 and y_arr.shape[0] != 1:
                raise ValueError("Graph-level task expects one label per graph.")
            if y_arr.ndim == 2 and y_arr.shape[0] != 1:
                raise ValueError("Graph-level task expects one label per graph.")
        else:
            if y_arr.ndim == 1 and y_arr.shape[0] != 1:
                raise ValueError("Graph-level task expects one label per graph.")
            if y_arr.ndim == 2 and y_arr.shape[0] != 1:
                raise ValueError("Graph-level task expects one label per graph.")
        if y_arr.ndim not in (1, 2):
            raise ValueError(f"Unsupported graph label shape: {y_arr.shape}")

        frame = pd.DataFrame({"graph_id": [str(graph_idx)]})
        graph_attr = getattr(data, "graph_attr", None)
        if graph_attr is None:
            x = getattr(data, "x", None)
            if x is not None:
                x_arr = _as_2d_tensor(x)
                if x_arr.shape[0] == 1:
                    graph_attr = x_arr
        if graph_attr is not None:
            graph_attr = _as_2d_tensor(graph_attr)
            if max_graph_features is not None:
                graph_attr = graph_attr[:, :max_graph_features]
            graph_np = _to_numpy(graph_attr).reshape(-1)
            graph_np = np.nan_to_num(graph_np)
            for idx, val in enumerate(graph_np):
                frame[f"g{idx}"] = val
        else:
            if not warned_no_attr:
                print("=== Warning: graph-level task has no raw graph attributes ===")
                warned_no_attr = True

        frames.append(frame)
        if allow_multi_target and y_arr.ndim == 2:
            labels.append(y_arr)
        else:
            labels.append(y_arr.reshape(-1))

    return _concat_frames(frames), _to_labels(labels)


def _to_numpy(array: object) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _to_dataframe(chunks: Sequence[np.ndarray]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    data = np.concatenate(chunks, axis=0)
    data = np.nan_to_num(data)
    columns = [f"f{i}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=columns)


def _to_labels(chunks: Sequence[np.ndarray]) -> pd.Series | pd.DataFrame:
    if not chunks:
        return pd.Series(dtype=float)
    data = np.concatenate(chunks, axis=0)
    if data.ndim == 1:
        return pd.Series(data)
    if data.ndim == 2:
        if data.shape[1] == 1:
            return pd.Series(data.reshape(-1))
        columns = [f"y{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=columns)
    data = data.reshape(-1)
    return pd.Series(data)


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _resolve_node_targets(data: object):
    y = getattr(data, "y", None)
    if y is not None:
        return y
    return getattr(data, "mis_solution", None)


def _resolve_graph_targets(data: object):
    y = getattr(data, "y", None)
    if y is not None:
        return y
    return getattr(data, "num_mis", None)


def _select_target(
    y: np.ndarray,
    target_index: Optional[int],
    allow_multi_target: bool = False,
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


def _node_features(
    data: object,
    max_node_features: Optional[int],
    add_degree: bool,
) -> np.ndarray:
    x = getattr(data, "x", None)
    x_arr: Optional[torch.Tensor]
    if x is None:
        x_arr = None
    else:
        x_arr = _as_2d_tensor(x)
        if max_node_features is not None:
            x_arr = x_arr[:, :max_node_features]

    degree = None
    if add_degree or x_arr is None:
        degree = _node_degree(data, x_arr)
    if x_arr is None:
        if degree is None:
            raise ValueError("Unable to derive node features (no x or edge_index).")
        feats = degree
    elif degree is not None:
        feats = torch.cat([x_arr, degree], dim=1)
    else:
        feats = x_arr
    return _to_numpy(feats)


def _edge_labels(
    data: object,
    auto_edge_labels: bool = False,
    edge_neg_ratio: float = 1.0,
    edge_sample_cap: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, np.ndarray, Optional[torch.Tensor]]:
    edge_label = getattr(data, "edge_label", None)
    edge_label_index = getattr(data, "edge_label_index", None)
    if edge_label is not None and edge_label_index is not None:
        return edge_label_index, _to_numpy(edge_label), None

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("Edge-level task requires `edge_index`.")
    y = getattr(data, "y", None)
    if y is not None:
        y_arr = _to_numpy(y)
        if y_arr.shape[0] == edge_index.shape[1]:
            edge_attr = getattr(data, "edge_attr", None)
            return edge_index, y_arr, edge_attr
        if not auto_edge_labels:
            raise ValueError("Edge-level labels do not match edge count.")

    if not auto_edge_labels:
        raise ValueError("Edge-level task requires `edge_label` or `y`.")

    edge_attr = getattr(data, "edge_attr", None)
    return _generate_edge_labels(
        edge_index=edge_index,
        edge_attr=edge_attr,
        data=data,
        neg_ratio=edge_neg_ratio,
        max_positives=edge_sample_cap,
        seed=seed,
    )


def _generate_edge_labels(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    data: object,
    neg_ratio: float,
    max_positives: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, np.ndarray, Optional[torch.Tensor]]:
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        labels = np.zeros((0,), dtype=int)
        return edge_index, labels, edge_attr

    num_edges = edge_index.shape[1]
    rng = np.random.default_rng(seed)
    if max_positives is not None and num_edges > max_positives:
        idx = rng.choice(num_edges, size=max_positives, replace=False)
        edge_index = edge_index[:, idx]
        if edge_attr is not None:
            edge_attr = _as_2d_tensor(edge_attr)
            if edge_attr.shape[0] == num_edges:
                edge_attr = edge_attr[idx]
        num_edges = edge_index.shape[1]

    num_neg = int(num_edges * max(0.0, float(neg_ratio)))
    if num_neg == 0:
        labels = np.ones((num_edges,), dtype=int)
        return edge_index, labels, edge_attr

    num_nodes = _infer_num_nodes(data, edge_index)
    if num_nodes <= 1:
        labels = np.ones((num_edges,), dtype=int)
        return edge_index, labels, edge_attr

    pos_edges = edge_index.detach().cpu().numpy().T
    pos_set = {(int(src), int(dst)) for src, dst in pos_edges}
    neg_edges: list[tuple[int, int]] = []
    neg_set: set[tuple[int, int]] = set()
    max_attempts = max(1000, num_neg * 10)
    attempts = 0
    while len(neg_edges) < num_neg and attempts < max_attempts:
        remaining = num_neg - len(neg_edges)
        batch = max(remaining * 2, 1000)
        src = rng.integers(0, num_nodes, size=batch, endpoint=False)
        dst = rng.integers(0, num_nodes, size=batch, endpoint=False)
        for s, d in zip(src.tolist(), dst.tolist()):
            if s == d:
                continue
            edge = (int(s), int(d))
            if edge in pos_set or edge in neg_set:
                continue
            neg_set.add(edge)
            neg_edges.append(edge)
            if len(neg_edges) >= num_neg:
                break
        attempts += batch

    if len(neg_edges) < num_neg:
        print(
            f"=== Warning: sampled {len(neg_edges)} negatives for {num_edges} positives ==="
        )

    neg_arr = np.asarray(neg_edges, dtype=int)
    if neg_arr.size == 0:
        labels = np.ones((num_edges,), dtype=int)
        return edge_index, labels, edge_attr

    neg_index = torch.as_tensor(neg_arr.T, dtype=edge_index.dtype)
    edge_index = torch.cat([edge_index, neg_index], dim=1)
    labels = np.concatenate([np.ones((num_edges,), dtype=int), np.zeros((len(neg_edges),), dtype=int)])

    if edge_attr is not None:
        edge_attr = _as_2d_tensor(edge_attr)
        zeros = torch.zeros((len(neg_edges), edge_attr.shape[1]), dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, zeros], dim=0)

    return edge_index, labels, edge_attr


def _infer_num_nodes(data: object, edge_index: torch.Tensor) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    x = getattr(data, "x", None)
    if x is not None:
        return int(getattr(x, "shape", [0])[0] or 0)
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def _edge_features(
    data: object,
    edge_index: torch.Tensor,
    max_node_features: Optional[int],
    include_edge_attr: bool,
    edge_attr: Optional[torch.Tensor],
    add_degree: bool,
) -> np.ndarray:
    x = getattr(data, "x", None)
    if x is not None:
        x = _as_2d_tensor(x)
        if max_node_features is not None:
            x = x[:, :max_node_features]
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        feats = torch.cat([src, dst], dim=1)
        if add_degree:
            degree = _node_degree(data, x)
            feats = torch.cat([feats, degree[edge_index[0]], degree[edge_index[1]]], dim=1)
    else:
        degree = _node_degree(data, None)
        feats = torch.cat([degree[edge_index[0]], degree[edge_index[1]]], dim=1)

    if include_edge_attr and edge_attr is not None:
        edge_attr = _as_2d_tensor(edge_attr)
        if edge_attr.shape[0] == edge_index.shape[1]:
            feats = torch.cat([feats, edge_attr], dim=1)
    return _to_numpy(feats)


def _graph_features(
    data: object,
    max_node_features: Optional[int],
    max_edge_features: Optional[int],
) -> np.ndarray:
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    edge_index = getattr(data, "edge_index", None)
    num_edges = int(edge_index.shape[1]) if edge_index is not None else 0
    density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    degree = _node_degree(data, None)
    avg_degree = float(degree.mean().item()) if degree is not None else 0.0
    max_degree = float(degree.max().item()) if degree is not None else 0.0

    stats: list[np.ndarray] = [
        np.array([num_nodes, num_edges, density, avg_degree, max_degree], dtype=float)
    ]

    x = getattr(data, "x", None)
    if x is not None:
        x_arr = _as_2d_tensor(x)
        if max_node_features is not None:
            x_arr = x_arr[:, :max_node_features]
        stats.extend(_stat_features(x_arr))

    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_arr = _as_2d_tensor(edge_attr)
        if max_edge_features is not None:
            edge_arr = edge_arr[:, :max_edge_features]
        stats.extend(_stat_features(edge_arr))

    return np.nan_to_num(np.concatenate(stats, axis=0))


def _stat_features(values: torch.Tensor) -> list[np.ndarray]:
    if values.numel() == 0:
        return [np.zeros(0, dtype=float)]
    values = values.float()
    mean = values.mean(dim=0).detach().cpu().numpy().reshape(-1)
    std = values.std(dim=0).detach().cpu().numpy().reshape(-1)
    vmin = values.min(dim=0).values.detach().cpu().numpy().reshape(-1)
    vmax = values.max(dim=0).values.detach().cpu().numpy().reshape(-1)
    return [mean, std, vmin, vmax]


def _node_degree(data: object, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        if x is None:
            return None
        return torch.zeros((x.shape[0], 1), dtype=torch.float)
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index)
    num_nodes = int(getattr(data, "num_nodes", 0) or (x.shape[0] if x is not None else 0))
    if num_nodes == 0:
        return torch.zeros((0, 1), dtype=torch.float)
    degree = torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float().unsqueeze(1)
    return degree


def _as_2d_tensor(values: object) -> torch.Tensor:
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    if values.dim() == 1:
        return values.unsqueeze(1)
    return values
