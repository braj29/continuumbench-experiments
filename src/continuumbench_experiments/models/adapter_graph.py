from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..continuumbench.specs import DatabaseSpec, TaskSpec
from ..continuumbench.views import build_graph_degree_feature_table, count_incident_tables
from .adapter_common import (
    BaseViewModel,
    TabularPreprocessor,
    _dense_feature_matrix,
    _maybe_subsample_xy,
    _positive_class_proba_maybe_chunked,
)


GRAPH_PROXY_VIEW_NAME = "structural-count-proxy"


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
        self._view_name = GRAPH_PROXY_VIEW_NAME
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

        metadata: dict[str, Any] = {
            "status": "ok",
            "graph_proxy": "incident_degree_counts",
            # Backward-compatibility key used by older analysis scripts.
            "graphified": "incident_degree_counts",
        }
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


class ExternalGraphAdapter(BaseViewModel):
    def __init__(
        self,
        name: str,
        fit_fn: Callable[..., tuple[Any, dict[str, Any]]],
        predict_fn: Callable[..., np.ndarray],
        **kwargs: Any,
    ):
        self._name = name
        self._view_name = GRAPH_PROXY_VIEW_NAME
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
