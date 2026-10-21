"""Backward-compat shim — canonical implementation lives in adapters/linear/ (Phase 04)."""

from app.domains.model_engine.adapters.linear.linear_regression_adapter import (
    LinearRegressionAdapter,
    _final_estimator,
)

__all__ = ["LinearRegressionAdapter", "_final_estimator"]
