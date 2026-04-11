from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator

from ..continuumbench.specs import TaskSpec
from .adapter_common import (
    BaseViewModel,
    TabularPreprocessor,
    _dense_feature_matrix,
    _maybe_subsample_xy,
    _positive_class_proba_maybe_chunked,
)

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


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

    def predict(self, data: pd.DataFrame, task: TaskSpec):
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

    def predict(self, data: pd.DataFrame, task: TaskSpec):
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
