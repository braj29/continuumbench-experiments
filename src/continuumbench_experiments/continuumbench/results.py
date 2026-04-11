from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RunResult:
    task_name: str
    model_name: str
    view_name: str
    split_name: str
    metric_name: str
    metric_value: float
    utility_value: float
    compute: dict[str, object]
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ProtocolSummary:
    per_run: list[RunResult]
    per_task_ris: dict[str, dict[str, float]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "task_name": run.task_name,
                    "model_name": run.model_name,
                    "view_name": run.view_name,
                    "split_name": run.split_name,
                    "metric_name": run.metric_name,
                    "metric_value": run.metric_value,
                    "utility_value": run.utility_value,
                    **{f"compute__{key}": value for key, value in run.compute.items()},
                    **{f"extra__{key}": value for key, value in run.extra.items()},
                }
                for run in self.per_run
            ]
        )
