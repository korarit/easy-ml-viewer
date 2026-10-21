"""Linear package marker — Phase 04 + Phase 18/19 linear-model family."""

from app.domains.model_engine.adapters.linear.linear_regression_adapter import (
    LinearRegressionAdapter,
)
from app.domains.model_engine.adapters.linear.logistic_regression_adapter import (
    LogisticRegressionAdapter,
)
from app.domains.model_engine.adapters.linear.regularized import (
    RegularizedLinearAdapter,
    RidgeClassifierAdapter,
)

__all__ = [
    "LinearRegressionAdapter",
    "LogisticRegressionAdapter",
    "RegularizedLinearAdapter",
    "RidgeClassifierAdapter",
]
