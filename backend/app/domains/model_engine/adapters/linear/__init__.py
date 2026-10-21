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
from app.domains.model_engine.adapters.linear.elasticnet import ElasticNetAdapter
from app.domains.model_engine.adapters.linear.sgd import (
    SgdClassifierAdapter,
    SgdRegressorAdapter,
)

__all__ = [
    "ElasticNetAdapter",
    "LinearRegressionAdapter",
    "LogisticRegressionAdapter",
    "RegularizedLinearAdapter",
    "RidgeClassifierAdapter",
    "SgdClassifierAdapter",
    "SgdRegressorAdapter",
]
