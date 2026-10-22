"""Trees & bagging family package — Phase 27 single trees, Phase 28/29 forests, Phase 30 boosting."""

from app.domains.model_engine.adapters.trees.adaboost import (
    AdaBoostClassifierAdapter,
    AdaBoostRegressorAdapter,
)
from app.domains.model_engine.adapters.trees.decision_tree import (
    DecisionTreeClassifierAdapter,
    DecisionTreeRegressorAdapter,
)
from app.domains.model_engine.adapters.trees.extra_trees import (
    ExtraTreesClassifierAdapter,
    ExtraTreesRegressorAdapter,
)
from app.domains.model_engine.adapters.trees.random_forest import (
    RandomForestClassifierAdapter,
    RandomForestRegressorAdapter,
)

__all__ = [
    "AdaBoostClassifierAdapter",
    "AdaBoostRegressorAdapter",
    "DecisionTreeClassifierAdapter",
    "DecisionTreeRegressorAdapter",
    "ExtraTreesClassifierAdapter",
    "ExtraTreesRegressorAdapter",
    "RandomForestClassifierAdapter",
    "RandomForestRegressorAdapter",
]
