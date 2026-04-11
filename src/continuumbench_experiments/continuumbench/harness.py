from __future__ import annotations

"""
Compatibility facade for the ContinuumBench core.

Reviewer-friendly implementation now lives in smaller modules:
- `specs.py`: data contracts and experiment config
- `metrics.py`: scoring, RIS, temporal splits, diagnostics
- `views.py`: joined-table and graphified view construction
- `runner.py`: Protocol A / B execution
- `examples.py`: synthetic demo problem
- `models/adapters.py`: model-facing adapters and wrappers
"""

from ..models.adapters import (
    BaseViewModel,
    ExternalGraphAdapter,
    ExternalRelationalAdapter,
    GraphifiedSklearnAdapter,
    LightGBMTabularAdapter,
    MeanDummyAdapter,
    OfficialRelationalTransformerAdapter,
    SklearnTabularAdapter,
    _dense_feature_matrix,
    stub_graph_fit_fn,
    stub_graph_predict_fn,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)
from .examples import make_synthetic_relational_problem
from .metrics import (
    compute_ris,
    compute_tracker,
    evaluate_predictions,
    inject_future_only_signal,
    make_temporal_split,
    metric_to_utility,
    set_random_seed,
)
from .results import ProtocolSummary, RunResult
from .runner import ExperimentRunner
from .specs import (
    DatabaseSpec,
    ExperimentConfig,
    JoinedTableConfig,
    TableSpec,
    TaskSpec,
    TemporalSplit,
)
from .views import (
    JoinedTableBuilder,
    MaterializedViews,
    ViewFactory,
    build_graph_degree_feature_table,
    count_incident_tables,
)

__all__ = [
    "BaseViewModel",
    "DatabaseSpec",
    "ExperimentConfig",
    "ExperimentRunner",
    "ExternalGraphAdapter",
    "ExternalRelationalAdapter",
    "GraphifiedSklearnAdapter",
    "JoinedTableBuilder",
    "JoinedTableConfig",
    "LightGBMTabularAdapter",
    "MaterializedViews",
    "MeanDummyAdapter",
    "OfficialRelationalTransformerAdapter",
    "ProtocolSummary",
    "RunResult",
    "SklearnTabularAdapter",
    "TableSpec",
    "TaskSpec",
    "TemporalSplit",
    "ViewFactory",
    "_dense_feature_matrix",
    "build_graph_degree_feature_table",
    "compute_ris",
    "compute_tracker",
    "count_incident_tables",
    "evaluate_predictions",
    "inject_future_only_signal",
    "make_synthetic_relational_problem",
    "make_temporal_split",
    "metric_to_utility",
    "set_random_seed",
    "stub_graph_fit_fn",
    "stub_graph_predict_fn",
    "stub_relational_fit_fn",
    "stub_relational_predict_fn",
]
