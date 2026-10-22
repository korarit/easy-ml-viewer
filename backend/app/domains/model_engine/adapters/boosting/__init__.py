"""Gradient-boosting family package — Phase 31 sklearn GBM, Phase 32 HistGBM, Phase 33 XGBoost, Phase 34 LightGBM, Phase 35 CatBoost."""

from app.domains.model_engine.adapters.boosting.hist_gbm import (
    HistGBMClassifierAdapter,
    HistGBMRegressorAdapter,
)
from app.domains.model_engine.adapters.boosting.sklearn_gbm import (
    SklearnGBMClassifierAdapter,
    SklearnGBMRegressorAdapter,
)
from app.domains.model_engine.adapters.boosting.xgb_adapter import (
    XGBClassifierAdapter,
    XGBRegressorAdapter,
)
from app.domains.model_engine.adapters.boosting.lightgbm_adapter import (
    LightGBMClassifierAdapter,
    LightGBMRegressorAdapter,
)
from app.domains.model_engine.adapters.boosting.catboost_adapter import (
    CatBoostClassifierAdapter,
    CatBoostRegressorAdapter,
)

__all__ = [
    "HistGBMClassifierAdapter",
    "HistGBMRegressorAdapter",
    "SklearnGBMClassifierAdapter",
    "SklearnGBMRegressorAdapter",
    "XGBClassifierAdapter",
    "XGBRegressorAdapter",
    "LightGBMClassifierAdapter",
    "LightGBMRegressorAdapter",
    "CatBoostClassifierAdapter",
    "CatBoostRegressorAdapter",
]
