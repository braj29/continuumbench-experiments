from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..continuumbench.specs import TaskSpec
from .adapter_common import BaseViewModel


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
