"""Compatibility facade for adapter imports.

The concrete adapters now live in smaller modules grouped by model/view type.
This file preserves the original import path:
`continuumbench_experiments.models.adapters`.
"""

from .adapter_common import (
    BaseViewModel,
    TabularPreprocessor,
    _dense_feature_matrix,
    _maybe_subsample_xy,
    _positive_class_proba_maybe_chunked,
)
from .adapter_dummy import MeanDummyAdapter
from .adapter_graph import (
    ExternalGraphAdapter,
    GraphifiedSklearnAdapter,
    stub_graph_fit_fn,
    stub_graph_predict_fn,
)
from .adapter_relational import (
    ExternalRelationalAdapter,
    OfficialRelationalTransformerAdapter,
    stub_relational_fit_fn,
    stub_relational_predict_fn,
)
from .adapter_tabular import LightGBMTabularAdapter, SklearnTabularAdapter

__all__ = [
    "BaseViewModel",
    "TabularPreprocessor",
    "_dense_feature_matrix",
    "_maybe_subsample_xy",
    "_positive_class_proba_maybe_chunked",
    "LightGBMTabularAdapter",
    "SklearnTabularAdapter",
    "GraphifiedSklearnAdapter",
    "ExternalRelationalAdapter",
    "OfficialRelationalTransformerAdapter",
    "ExternalGraphAdapter",
    "MeanDummyAdapter",
    "stub_relational_fit_fn",
    "stub_relational_predict_fn",
    "stub_graph_fit_fn",
    "stub_graph_predict_fn",
]
