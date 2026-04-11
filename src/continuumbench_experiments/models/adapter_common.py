from __future__ import annotations

import abc
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..continuumbench.specs import TaskSpec


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
