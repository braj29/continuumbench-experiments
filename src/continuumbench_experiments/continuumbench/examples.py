from __future__ import annotations

import pandas as pd

from .metrics import make_temporal_split
from .specs import DatabaseSpec, TableSpec, TaskSpec, TemporalSplit


def make_synthetic_relational_problem() -> tuple[DatabaseSpec, TaskSpec, TemporalSplit]:
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "country": ["NL", "DE", "NL", "FR", "DE"],
            "age": [28, 41, 35, 24, 50],
            "updated_at": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]
            ),
        }
    )

    transactions = pd.DataFrame(
        {
            "tx_id": range(1, 16),
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5],
            "amount": [10, 15, 12, 9, 8, 2, 30, 28, 31, 3, 5, 40, 42, 38, 39],
            "channel": [
                "web",
                "app",
                "web",
                "web",
                "app",
                "app",
                "web",
                "app",
                "app",
                "web",
                "app",
                "web",
                "app",
                "web",
                "app",
            ],
            "event_time": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-15",
                    "2024-02-01",
                    "2024-01-07",
                    "2024-03-01",
                    "2024-01-20",
                    "2024-02-11",
                    "2024-03-05",
                    "2024-03-15",
                    "2024-01-03",
                    "2024-02-07",
                    "2024-01-10",
                    "2024-02-10",
                    "2024-03-10",
                    "2024-03-20",
                ]
            ),
        }
    )

    task_table = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "seed_time": pd.to_datetime(
                [
                    "2024-02-15",
                    "2024-02-15",
                    "2024-02-15",
                    "2024-02-15",
                    "2024-02-15",
                    "2024-03-16",
                    "2024-03-16",
                    "2024-03-16",
                    "2024-03-16",
                    "2024-03-16",
                ]
            ),
            "label": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        }
    )

    db = DatabaseSpec(
        tables={
            "users": TableSpec(
                name="users",
                df=users,
                primary_key="user_id",
                time_col="updated_at",
                foreign_keys={},
            ),
            "transactions": TableSpec(
                name="transactions",
                df=transactions,
                primary_key="tx_id",
                time_col="event_time",
                foreign_keys={"user_id": "users"},
            ),
        }
    )

    task = TaskSpec(
        name="synthetic-user-churn",
        task_type="classification",
        target_col="label",
        metric_name="auroc",
        entity_table="users",
        entity_key="user_id",
        seed_time_col="seed_time",
        task_table=task_table,
    )

    split = make_temporal_split(
        task_df=task_table,
        seed_time_col="seed_time",
        val_cutoff=pd.Timestamp("2024-03-01"),
        test_cutoff=pd.Timestamp("2024-03-10"),
    )
    return db, task, split
