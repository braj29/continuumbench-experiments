from __future__ import annotations

import abc
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..continuumbench.specs import DatabaseSpec, TaskSpec
from ..continuumbench.views import build_graph_degree_feature_table, count_incident_tables

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


class BaseViewModel(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def view_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise NotImplementedError


class TabularPreprocessor:
    def __init__(self, target_col: str, drop_cols: Optional[Sequence[str]] = None):
        self.target_col = target_col
        self.drop_cols = set(drop_cols or [])
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_columns_: Optional[list[str]] = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        feature_df = self._feature_df(df)
        numeric_cols = [
            column
            for column in feature_df.columns
            if pd.api.types.is_numeric_dtype(feature_df[column])
        ]
        categorical_cols = [column for column in feature_df.columns if column not in numeric_cols]

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",
        )
        self.pipeline.fit(feature_df)
        self.feature_columns_ = list(feature_df.columns)
        return self

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        return self.pipeline.transform(self._feature_df(df))

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def _feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = {self.target_col} | self.drop_cols
        columns = [column for column in cols_to_drop if column in df.columns]
        return df.drop(columns=columns, errors="ignore")


def _maybe_subsample_xy(
    X: Any,
    y: np.ndarray,
    max_rows: Optional[int],
    seed: int,
) -> tuple[Any, np.ndarray]:
    if max_rows is None:
        return X, y
    row_count = int(getattr(X, "shape", (0,))[0])
    if row_count <= max_rows:
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(row_count, size=max_rows, replace=False))
    return X[idx], y[idx]


def _dense_feature_matrix(X: Any) -> np.ndarray:
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _positive_class_proba_maybe_chunked(
    estimator: BaseEstimator,
    X: np.ndarray,
    chunk_size: Optional[int],
) -> np.ndarray:
    if (
        chunk_size is not None
        and chunk_size > 0
        and len(X) > chunk_size
        and type(estimator).__name__ == "TabICLClassifier"
    ):
        parts: list[np.ndarray] = []
        for start in range(0, len(X), chunk_size):
            stop = min(start + chunk_size, len(X))
            parts.append(estimator.predict_proba(X[start:stop])[:, 1])
        return np.concatenate(parts, axis=0)
    return estimator.predict_proba(X)[:, 1]


class LightGBMTabularAdapter(BaseViewModel):
    def __init__(self, params: Optional[dict[str, Any]] = None):
        self.params = params or {}
        self.model: Optional[Any] = None
        self.preprocessor: Optional[TabularPreprocessor] = None
        self._name = "lightgbm"
        self._view_name = "joined-table"

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        task: TaskSpec,
    ) -> dict[str, Any]:
        if lgb is None:
            raise ImportError("lightgbm is not installed. Install it or swap this adapter.")

        self.preprocessor = TabularPreprocessor(
            target_col=task.target_col,
            drop_cols=[task.seed_time_col],
        )
        X_train = self.preprocessor.fit_transform(train_data)
        y_train = train_data[task.target_col].to_numpy()

        X_val, y_val = None, None
        if val_data is not None:
            X_val = self.preprocessor.transform(val_data)
            y_val = val_data[task.target_col].to_numpy()

        if task.task_type == "classification":
            model = lgb.LGBMClassifier(
                random_state=7,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                **self.params,
            )
        else:
            model = lgb.LGBMRegressor(
                random_state=7,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                **self.params,
            )

        fit_kwargs = {"eval_set": [(X_val, y_val)]} if X_val is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        self.model = model
        return {"status": "ok"}

    def predict(self, data: pd.DataFrame, task: TaskSpec) -> np.ndarray:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fit before predict.")
        X = self.preprocessor.transform(data)
        if task.task_type == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


class SklearnTabularAdapter(BaseViewModel):
    def __init__(
        self,
        estimator: BaseEstimator,
        name: str,
        max_train_rows: Optional[int] = None,
        subsample_seed: int = 0,
        predict_proba_chunk_size: Optional[int] = None,
    ):
        self.estimator = estimator
        self._name = name
        self._view_name = "joined-table"
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.max_train_rows = max_train_rows
        self.subsample_seed = int(subsample_seed)
        self.predict_proba_chunk_size = predict_proba_chunk_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        task: TaskSpec,
    ) -> dict[str, Any]:
        self.preprocessor = TabularPreprocessor(
            target_col=task.target_col,
            drop_cols=[task.seed_time_col],
        )
        X_train = self.preprocessor.fit_transform(train_data)
        y_train = train_data[task.target_col].to_numpy()
        X_train, y_train = _maybe_subsample_xy(
            X_train,
            y_train,
            max_rows=self.max_train_rows,
            seed=self.subsample_seed,
        )
        self.estimator.fit(_dense_feature_matrix(X_train), y_train)
        return {"status": "ok"}

    def predict(self, data: pd.DataFrame, task: TaskSpec) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Model must be fit before predict.")
        X = _dense_feature_matrix(self.preprocessor.transform(data))
        if task.task_type == "classification" and hasattr(self.estimator, "predict_proba"):
            return _positive_class_proba_maybe_chunked(
                self.estimator,
                X,
                self.predict_proba_chunk_size,
            )
        return self.estimator.predict(X)


class GraphifiedSklearnAdapter(BaseViewModel):
    _K_TABLE_CAP_DIVISOR = 100

    def __init__(
        self,
        estimator: BaseEstimator,
        name: str,
        max_train_rows: Optional[int] = None,
        subsample_seed: int = 0,
        predict_proba_chunk_size: Optional[int] = None,
    ):
        self.estimator = estimator
        self._name = name
        self._view_name = "graphified"
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.max_train_rows = max_train_rows
        self.subsample_seed = int(subsample_seed)
        self.predict_proba_chunk_size = predict_proba_chunk_size
        self._max_incident_tables: Optional[int] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    @staticmethod
    def _incident_table_cap(payload: Mapping[str, Any], n_candidates: int) -> Optional[int]:
        graph_config = payload.get("graph_config") or {}
        neighborhood_size_k = graph_config.get("neighborhood_size_k")
        if neighborhood_size_k is None:
            return None
        return max(
            1,
            min(
                n_candidates,
                int(neighborhood_size_k) // GraphifiedSklearnAdapter._K_TABLE_CAP_DIVISOR,
            ),
        )

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        payload = train_data["payload"]
        db: DatabaseSpec = payload["db"]
        task_df: pd.DataFrame = train_data["task_df"]
        self._max_incident_tables = self._incident_table_cap(
            payload,
            count_incident_tables(db, task),
        )

        frame = build_graph_degree_feature_table(
            db=db,
            task=task,
            rows=task_df,
            max_incident_tables=self._max_incident_tables,
        )
        self.preprocessor = TabularPreprocessor(
            target_col=task.target_col,
            drop_cols=[task.seed_time_col],
        )
        X_train = self.preprocessor.fit_transform(frame)
        y_train = frame[task.target_col].to_numpy()
        X_train, y_train = _maybe_subsample_xy(
            X_train,
            y_train,
            max_rows=self.max_train_rows,
            seed=self.subsample_seed,
        )
        self.estimator.fit(_dense_feature_matrix(X_train), y_train)

        metadata: dict[str, Any] = {"status": "ok", "graphified": "incident_degree_counts"}
        if self._max_incident_tables is not None:
            metadata["max_incident_tables"] = self._max_incident_tables
        return metadata

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Model must be fit before predict.")
        payload = data["payload"]
        frame = build_graph_degree_feature_table(
            db=payload["db"],
            task=task,
            rows=data["task_df"],
            max_incident_tables=self._max_incident_tables,
        )
        X = _dense_feature_matrix(self.preprocessor.transform(frame))
        if task.task_type == "classification" and hasattr(self.estimator, "predict_proba"):
            return _positive_class_proba_maybe_chunked(
                self.estimator,
                X,
                self.predict_proba_chunk_size,
            )
        return self.estimator.predict(X)


class ExternalRelationalAdapter(BaseViewModel):
    def __init__(
        self,
        name: str,
        fit_fn: Callable[..., tuple[Any, dict[str, Any]]],
        predict_fn: Callable[..., np.ndarray],
        **kwargs: Any,
    ):
        self._name = name
        self._view_name = "relational"
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.model_obj: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        self.model_obj, metadata = self.fit_fn(
            train_data=train_data,
            val_data=val_data,
            task=task,
            **self.kwargs,
        )
        return metadata

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        return self.predict_fn(model_obj=self.model_obj, data=data, task=task, **self.kwargs)


class OfficialRelationalTransformerAdapter(BaseViewModel):
    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        target_col: str,
        rt_repo_path: str,
        name: str = "rt-official",
        python_executable: Optional[str] = None,
        columns_to_drop: Optional[Sequence[str]] = None,
        project: str = "continuumbench-rt",
        eval_splits: Sequence[str] = ("val", "test"),
        eval_freq: int = 1000,
        eval_pow2: bool = False,
        max_eval_steps: int = 40,
        load_ckpt_path: Optional[str] = None,
        save_ckpt_dir: Optional[str] = None,
        compile_: bool = False,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 8,
        max_bfs_width: int = 256,
        lr: float = 1e-4,
        wd: float = 0.0,
        lr_schedule: bool = False,
        max_grad_norm: float = 1.0,
        max_steps: int = 2**12 + 1,
        embedding_model: str = "all-MiniLM-L12-v2",
        d_text: int = 384,
        seq_len: int = 1024,
        num_blocks: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
    ):
        self._name = name
        self._view_name = "relational"
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.target_col = target_col
        self.rt_repo_path = str(rt_repo_path)
        self.python_executable = python_executable or sys.executable
        self.columns_to_drop = list(columns_to_drop or [])
        self.project = project
        self.eval_splits = list(eval_splits)
        self.eval_freq = int(eval_freq)
        self.eval_pow2 = bool(eval_pow2)
        self.max_eval_steps = int(max_eval_steps)
        self.load_ckpt_path = load_ckpt_path
        self.save_ckpt_dir = save_ckpt_dir
        self.compile_ = bool(compile_)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_bfs_width = int(max_bfs_width)
        self.lr = float(lr)
        self.wd = float(wd)
        self.lr_schedule = bool(lr_schedule)
        self.max_grad_norm = float(max_grad_norm)
        self.max_steps = int(max_steps)
        self.embedding_model = embedding_model
        self.d_text = int(d_text)
        self.seq_len = int(seq_len)
        self.num_blocks = int(num_blocks)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def _rt_main_config(self) -> dict[str, Any]:
        task_tuple = (
            self.dataset_name,
            self.task_name,
            self.target_col,
            self.columns_to_drop,
        )
        return {
            "project": self.project,
            "eval_splits": self.eval_splits,
            "eval_freq": self.eval_freq,
            "eval_pow2": self.eval_pow2,
            "max_eval_steps": self.max_eval_steps,
            "load_ckpt_path": self.load_ckpt_path,
            "save_ckpt_dir": self.save_ckpt_dir,
            "compile_": self.compile_,
            "seed": self.seed,
            "train_tasks": [task_tuple],
            "eval_tasks": [task_tuple],
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_bfs_width": self.max_bfs_width,
            "lr": self.lr,
            "wd": self.wd,
            "lr_schedule": self.lr_schedule,
            "max_grad_norm": self.max_grad_norm,
            "max_steps": self.max_steps,
            "embedding_model": self.embedding_model,
            "d_text": self.d_text,
            "seq_len": self.seq_len,
            "num_blocks": self.num_blocks,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
        }

    def _extract_test_metric(self, logs: str) -> Optional[float]:
        numeric = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        dataset = re.escape(self.dataset_name)
        task = re.escape(self.task_name)
        patterns = [
            rf"(?:auc|r2|mae)/{dataset}/{task}/test:\s*{numeric}",
            rf"{dataset}/{task}/test:\s*{numeric}",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, logs)
            if matches:
                return float(matches[-1])
        return None

    def _validate_runtime_environment(self, repo_path: Path) -> None:
        if platform.system() != "Linux":
            raise RuntimeError(
                "Official Relational Transformer is not runnable on this machine. "
                "The upstream runtime expects Linux + CUDA. "
                "Use Snellius or another Linux GPU box."
            )

        probe = subprocess.run(
            [
                self.python_executable,
                "-c",
                (
                    "import json, platform\n"
                    "import torch\n"
                    "print(json.dumps({'platform': platform.system(), "
                    "'cuda_available': bool(torch.cuda.is_available())}))\n"
                ),
            ],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            tail = "\n".join(((probe.stdout or "") + "\n" + (probe.stderr or "")).splitlines()[-40:])
            raise RuntimeError(
                "Official RT runtime probe failed for the configured python executable.\n"
                f"--- tail ---\n{tail}"
            )

        try:
            payload = json.loads((probe.stdout or "").strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Official RT runtime probe returned unexpected output.\n"
                f"stdout={probe.stdout!r}\nstderr={probe.stderr!r}"
            ) from exc

        runtime_platform = payload.get("platform")
        cuda_available = bool(payload.get("cuda_available"))
        if runtime_platform != "Linux" or not cuda_available:
            raise RuntimeError(
                "Official Relational Transformer requires a Linux + CUDA runtime "
                f"in the target python environment. Detected platform={runtime_platform!r}, "
                f"cuda_available={cuda_available}. "
                "Use Snellius or another Linux GPU box."
            )

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        repo_path = Path(self.rt_repo_path).expanduser().resolve()
        rt_main_path = repo_path / "rt" / "main.py"
        if not rt_main_path.is_file():
            raise FileNotFoundError(
                f"Official RT repo not found at '{repo_path}'. Missing rt/main.py."
            )
        self._validate_runtime_environment(repo_path)

        env = os.environ.copy()
        env["RT_MAIN_CONFIG"] = json.dumps(self._rt_main_config())
        command = [
            self.python_executable,
            "-c",
            (
                "import json, os\n"
                "from rt.main import main\n"
                "main(**json.loads(os.environ['RT_MAIN_CONFIG']))\n"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        logs = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if result.returncode != 0:
            tail = "\n".join(logs.splitlines()[-120:])
            raise RuntimeError(
                f"Official RT run failed. returncode={result.returncode}\n--- tail ---\n{tail}"
            )

        metric = self._extract_test_metric(logs)
        if metric is None:
            tail = "\n".join(logs.splitlines()[-120:])
            raise RuntimeError(
                "Official RT run succeeded but test metric was not found in logs.\n"
                f"--- tail ---\n{tail}"
            )

        return {
            "backend": "official_relational_transformer",
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "target_col": self.target_col,
            "precomputed_test_metric": float(metric),
        }

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise RuntimeError(
            "OfficialRelationalTransformerAdapter does not emit row-wise predictions "
            "in this integration. Use precomputed_test_metric instead."
        )


class ExternalGraphAdapter(BaseViewModel):
    def __init__(
        self,
        name: str,
        fit_fn: Callable[..., tuple[Any, dict[str, Any]]],
        predict_fn: Callable[..., np.ndarray],
        **kwargs: Any,
    ):
        self._name = name
        self._view_name = "graphified"
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.model_obj: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        self.model_obj, metadata = self.fit_fn(
            train_data=train_data,
            val_data=val_data,
            task=task,
            **self.kwargs,
        )
        return metadata

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        return self.predict_fn(model_obj=self.model_obj, data=data, task=task, **self.kwargs)


class MeanDummyAdapter(BaseViewModel):
    def __init__(self, name: str, view_name: str):
        self._name = name
        self._view_name = view_name
        self.constant_: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        if isinstance(train_data, dict):
            y = train_data["task_df"][task.target_col].to_numpy()
        else:
            y = train_data[task.target_col].to_numpy()
        self.constant_ = float(np.mean(y))
        return {"status": "dummy_fitted"}

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        if self.constant_ is None:
            raise RuntimeError("Model must be fit before predict.")
        n_rows = len(data["task_df"]) if isinstance(data, dict) else len(data)
        return np.full(shape=(n_rows,), fill_value=self.constant_, dtype=float)


def stub_relational_fit_fn(
    train_data: Any,
    val_data: Any,
    task: TaskSpec,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    model_obj = {"mean": float(train_data["task_df"][task.target_col].mean())}
    return model_obj, {"backend": "stub_relational"}


def stub_relational_predict_fn(
    model_obj: Any,
    data: Any,
    task: TaskSpec,
    **kwargs: Any,
) -> np.ndarray:
    return np.full(len(data["task_df"]), model_obj["mean"], dtype=float)


def stub_graph_fit_fn(
    train_data: Any,
    val_data: Any,
    task: TaskSpec,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    model_obj = {"mean": float(train_data["task_df"][task.target_col].mean())}
    return model_obj, {"backend": "stub_graph", **kwargs}


def stub_graph_predict_fn(
    model_obj: Any,
    data: Any,
    task: TaskSpec,
    **kwargs: Any,
) -> np.ndarray:
    return np.full(len(data["task_df"]), model_obj["mean"], dtype=float)
