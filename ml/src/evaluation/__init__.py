"""Evaluation utilities — metrics, SOH/RUL, and the auto-routing pipeline."""

from .metrics import (
    EOL_ABS,
    EOL_THRESHOLD,
    RATED_CAPACITY_NOMINAL,
    compute_rul,
    compute_soh,
    regression_metrics,
)
from .pipeline import (
    load_bundle,
    predict_multicycle,
    predict_singlecycle,
    run_pipeline,
    select_pipeline,
)

__all__ = [
    "EOL_ABS",
    "EOL_THRESHOLD",
    "RATED_CAPACITY_NOMINAL",
    "compute_rul",
    "compute_soh",
    "load_bundle",
    "predict_multicycle",
    "predict_singlecycle",
    "regression_metrics",
    "run_pipeline",
    "select_pipeline",
]
