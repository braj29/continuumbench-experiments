from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..models.adapters import BaseViewModel
from .metrics import (
    compute_ris,
    compute_tracker,
    evaluate_predictions,
    metric_to_utility,
    set_random_seed,
)
from .results import ProtocolSummary, RunResult
from .specs import DatabaseSpec, ExperimentConfig, JoinedTableConfig, TaskSpec, TemporalSplit
from .views import ViewFactory


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        set_random_seed(config.seed)

    def run_protocol_a(
        self,
        db: DatabaseSpec,
        task: TaskSpec,
        split: TemporalSplit,
        joined_table_cfgs: Sequence[JoinedTableConfig],
        joined_models: Sequence[BaseViewModel],
        relational_models: Sequence[BaseViewModel],
        graph_models: Sequence[BaseViewModel],
    ) -> ProtocolSummary:
        results: list[RunResult] = []
        view_factory = ViewFactory(db, task)

        results.extend(
            self._run_joined_table_track(
                view_factory=view_factory,
                task=task,
                split=split,
                joined_table_cfgs=joined_table_cfgs,
                joined_models=joined_models,
            )
        )
        results.extend(
            self._run_relational_track(
                view_factory=view_factory,
                task=task,
                split=split,
                relational_models=relational_models,
            )
        )
        results.extend(
            self._run_graph_track(
                view_factory=view_factory,
                task=task,
                split=split,
                graph_models=graph_models,
            )
        )
        return self._finalize_summary(task.name, "protocol_a", results)

    def run_protocol_b(
        self,
        db: DatabaseSpec,
        task: TaskSpec,
        split: TemporalSplit,
        joined_ablation_cfgs: Sequence[JoinedTableConfig],
        joined_models: Sequence[BaseViewModel],
        graph_k_values: Sequence[int],
        graph_models_factory,
    ) -> ProtocolSummary:
        results: list[RunResult] = []
        view_factory = ViewFactory(db, task)

        results.extend(
            self._run_joined_table_ablations(
                view_factory=view_factory,
                task=task,
                split=split,
                joined_ablation_cfgs=joined_ablation_cfgs,
                joined_models=joined_models,
            )
        )
        results.extend(
            self._run_graph_ablations(
                view_factory=view_factory,
                task=task,
                split=split,
                graph_k_values=graph_k_values,
                graph_models_factory=graph_models_factory,
            )
        )
        return self._finalize_summary(task.name, "protocol_b", results)

    def _run_joined_table_track(
        self,
        view_factory: ViewFactory,
        task: TaskSpec,
        split: TemporalSplit,
        joined_table_cfgs: Sequence[JoinedTableConfig],
        joined_models: Sequence[BaseViewModel],
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for joined_cfg in joined_table_cfgs:
            joined_df = view_factory.build_joined_table(joined_cfg)
            train_df, val_df, test_df = _split_frame(joined_df, split)
            for model in joined_models:
                label = f"{model.name}:{joined_cfg.view_name}"
                fit_meta, score, compute = self._fit_and_score(
                    model=model,
                    train_data=train_df,
                    val_data=val_df,
                    test_data=test_df,
                    y_true=test_df[task.target_col].to_numpy(),
                    task=task,
                )
                results.append(
                    RunResult(
                        task_name=task.name,
                        model_name=label,
                        view_name="joined-table",
                        split_name="test",
                        metric_name=task.metric_name,
                        metric_value=score,
                        utility_value=metric_to_utility(score, task.metric_name),
                        compute=compute,
                        extra={**fit_meta, "joined_subview": joined_cfg.view_name},
                    )
                )
        return results

    def _run_relational_track(
        self,
        view_factory: ViewFactory,
        task: TaskSpec,
        split: TemporalSplit,
        relational_models: Sequence[BaseViewModel],
    ) -> list[RunResult]:
        payload = view_factory.build_relational_view()
        train_data, val_data, test_data = _build_payload_splits(payload, task.task_table, split)

        results: list[RunResult] = []
        for model in relational_models:
            fit_meta, score, compute = self._fit_and_score(
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                y_true=test_data["task_df"][task.target_col].to_numpy(),
                task=task,
            )
            results.append(
                RunResult(
                    task_name=task.name,
                    model_name=model.name,
                    view_name="relational",
                    split_name="test",
                    metric_name=task.metric_name,
                    metric_value=score,
                    utility_value=metric_to_utility(score, task.metric_name),
                    compute=compute,
                    extra=fit_meta,
                )
            )
        return results

    def _run_graph_track(
        self,
        view_factory: ViewFactory,
        task: TaskSpec,
        split: TemporalSplit,
        graph_models: Sequence[BaseViewModel],
    ) -> list[RunResult]:
        payload = view_factory.build_graphified_view(neighborhood_size_k=None)
        train_data, val_data, test_data = _build_payload_splits(payload, task.task_table, split)

        results: list[RunResult] = []
        for model in graph_models:
            fit_meta, score, compute = self._fit_and_score(
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                y_true=test_data["task_df"][task.target_col].to_numpy(),
                task=task,
            )
            results.append(
                RunResult(
                    task_name=task.name,
                    model_name=model.name,
                    view_name=model.view_name,
                    split_name="test",
                    metric_name=task.metric_name,
                    metric_value=score,
                    utility_value=metric_to_utility(score, task.metric_name),
                    compute=compute,
                    extra=fit_meta,
                )
            )
        return results

    def _run_joined_table_ablations(
        self,
        view_factory: ViewFactory,
        task: TaskSpec,
        split: TemporalSplit,
        joined_ablation_cfgs: Sequence[JoinedTableConfig],
        joined_models: Sequence[BaseViewModel],
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for joined_cfg in joined_ablation_cfgs:
            joined_df = view_factory.build_joined_table(joined_cfg)
            train_df, val_df, test_df = _split_frame(joined_df, split)
            for model in joined_models:
                label = (
                    f"{model.name}:{joined_cfg.view_name}:"
                    f"hops={joined_cfg.max_path_hops}:w={joined_cfg.lookback_days}"
                )
                fit_meta, score, compute = self._fit_and_score(
                    model=model,
                    train_data=train_df,
                    val_data=val_df,
                    test_data=test_df,
                    y_true=test_df[task.target_col].to_numpy(),
                    task=task,
                )
                results.append(
                    RunResult(
                        task_name=task.name,
                        model_name=label,
                        view_name="joined-table",
                        split_name="test",
                        metric_name=task.metric_name,
                        metric_value=score,
                        utility_value=metric_to_utility(score, task.metric_name),
                        compute=compute,
                        extra={**fit_meta, **dataclasses.asdict(joined_cfg)},
                    )
                )
        return results

    def _run_graph_ablations(
        self,
        view_factory: ViewFactory,
        task: TaskSpec,
        split: TemporalSplit,
        graph_k_values: Sequence[int],
        graph_models_factory,
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for k_value in graph_k_values:
            payload = view_factory.build_graphified_view(neighborhood_size_k=k_value)
            train_data, val_data, test_data = _build_payload_splits(payload, task.task_table, split)
            for model in graph_models_factory(k_value):
                fit_meta, score, compute = self._fit_and_score(
                    model=model,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    y_true=test_data["task_df"][task.target_col].to_numpy(),
                    task=task,
                )
                results.append(
                    RunResult(
                        task_name=task.name,
                        model_name=f"{model.name}:K={k_value}",
                        view_name=model.view_name,
                        split_name="test",
                        metric_name=task.metric_name,
                        metric_value=score,
                        utility_value=metric_to_utility(score, task.metric_name),
                        compute=compute,
                        extra={**fit_meta, "K": k_value},
                    )
                )
        return results

    def _fit_and_score(
        self,
        model: BaseViewModel,
        train_data: Any,
        val_data: Any,
        test_data: Any,
        y_true: np.ndarray,
        task: TaskSpec,
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        with compute_tracker() as compute:
            fit_meta = model.fit(train_data, val_data, task)
            score = self._score_after_fit(
                model=model,
                fit_meta=fit_meta,
                data=test_data,
                y_true=y_true,
                task=task,
            )
        return fit_meta, score, dict(compute)

    @staticmethod
    def _extract_metric_override(
        fit_meta: Any,
        split_name: str = "test",
    ) -> float | None:
        if not isinstance(fit_meta, Mapping):
            return None
        for key in (
            f"precomputed_{split_name}_metric",
            "precomputed_metric",
            "metric_override",
        ):
            value = fit_meta.get(key)
            if value is not None:
                return float(value)
        return None

    def _score_after_fit(
        self,
        model: BaseViewModel,
        fit_meta: Any,
        data: Any,
        y_true: np.ndarray,
        task: TaskSpec,
    ) -> float:
        override = self._extract_metric_override(fit_meta)
        if override is not None:
            return float(override)
        pred = model.predict(data, task)
        return evaluate_predictions(y_true, pred, task.metric_name)

    def _finalize_summary(
        self,
        task_name: str,
        protocol_name: str,
        results: Sequence[RunResult],
    ) -> ProtocolSummary:
        summary = ProtocolSummary(
            per_run=list(results),
            per_task_ris=self._summarize_ris(results),
        )
        self._save_protocol_summary(task_name, protocol_name, summary)
        return summary

    def _summarize_ris(self, results: Sequence[RunResult]) -> dict[str, dict[str, float]]:
        task_to_scores: dict[str, dict[str, float]] = defaultdict(dict)
        task_to_metric: dict[str, str] = {}

        for run in results:
            task_to_metric[run.task_name] = run.metric_name
            representation_key = self._representation_key(run)
            current = task_to_scores[run.task_name].get(representation_key)
            if current is None:
                task_to_scores[run.task_name][representation_key] = run.metric_value
            elif run.metric_name == "auroc":
                task_to_scores[run.task_name][representation_key] = max(current, run.metric_value)
            else:
                task_to_scores[run.task_name][representation_key] = min(current, run.metric_value)

        summary: dict[str, dict[str, float]] = {}
        for task_name, representation_scores in task_to_scores.items():
            ris_summary = compute_ris(representation_scores, task_to_metric[task_name])
            ris_summary["representation_scores"] = {
                key: float(value) for key, value in representation_scores.items()
            }
            summary[task_name] = ris_summary
        return summary

    @staticmethod
    def _representation_key(run: RunResult) -> str:
        parts = [run.view_name]
        if run.view_name == "joined-table":
            subview = run.extra.get("joined_subview")
            if subview:
                parts.append(str(subview))
            if "max_path_hops" in run.extra:
                parts.append(f"hops={run.extra['max_path_hops']}")
            if "lookback_days" in run.extra:
                parts.append(f"window={run.extra['lookback_days']}")
        elif run.view_name in {"graphified", "structural-count-proxy"}:
            graphified = run.extra.get("graph_proxy") or run.extra.get("graphified")
            if graphified:
                parts.append(str(graphified))
            if "K" in run.extra:
                parts.append(f"K={run.extra['K']}")
        elif run.view_name == "relational":
            backend = run.extra.get("backend")
            if backend:
                parts.append(str(backend))
        return "|".join(parts)

    def _save_protocol_summary(
        self,
        task_name: str,
        protocol_name: str,
        summary: ProtocolSummary,
    ) -> None:
        out_dir = Path(self.config.output_dir)
        summary.to_frame().to_parquet(
            out_dir / f"{task_name}__{protocol_name}__runs.parquet",
            index=False,
        )
        ris_path = out_dir / f"{task_name}__{protocol_name}__ris.json"
        with open(ris_path, "w", encoding="utf-8") as handle:
            json.dump(summary.per_task_ris, handle, indent=2)


def _split_frame(frame, split: TemporalSplit):
    return (
        frame.loc[split.train_mask].reset_index(drop=True),
        frame.loc[split.val_mask].reset_index(drop=True),
        frame.loc[split.test_mask].reset_index(drop=True),
    )


def _build_payload_splits(payload: Any, task_table, split: TemporalSplit):
    return (
        {"payload": payload, "task_df": task_table.loc[split.train_mask].reset_index(drop=True)},
        {"payload": payload, "task_df": task_table.loc[split.val_mask].reset_index(drop=True)},
        {"payload": payload, "task_df": task_table.loc[split.test_mask].reset_index(drop=True)},
    )
