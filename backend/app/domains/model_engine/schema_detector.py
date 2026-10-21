"""Universal schema introspection — names, counts, pipeline unwrapping, fitted checks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.exceptions import ModelLoadError
from app.domains.model_engine.value_objects import FeatureRequirement


class ModelSchema(BaseModel):
    n_features_in: int = Field(ge=1)
    features: list[FeatureRequirement]
    classes: list[str | int | float] | None = None

    model_config = {"arbitrary_types_allowed": True}


def _safe_n_features(estimator: Any) -> int | None:
    for attr in ("n_features_in_", "n_features_to_select_"):
        value = getattr(estimator, attr, None)
        if isinstance(value, (int,)) and value >= 1:
            return int(value)
        try:
            import numpy as np  # noqa: PLC0415

            if isinstance(value, np.integer) and int(value) >= 1:
                return int(value)
        except ImportError:
            pass
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        try:
            shape = coef.shape
            if len(shape) >= 2:
                return int(shape[-1])
            if len(shape) == 1:
                return int(shape[0])
        except (AttributeError, TypeError, ValueError):
            pass
    return None


class UniversalSchemaDetector:
    """Framework-free introspection helpers shared by every adapter phase."""

    @staticmethod
    def unwrap_model_container(obj: Any) -> tuple[Any, Any, list[str]]:
        """Split search/pipeline wrappers.

        Returns ``(predict_obj, final_estimator, step_names)`` where
        ``predict_obj`` is the object inference must run through (full
        pipeline when wrapped) and ``final_estimator`` is the terminal
        learner used for adapter resolution.
        """
        # GridSearchCV / RandomizedSearchCV / Halving*Search -> best estimator.
        best = getattr(obj, "best_estimator_", None)
        if best is not None:
            predict_obj, final_est, steps = UniversalSchemaDetector.unwrap_model_container(best)
            return predict_obj, final_est, steps
        # sklearn Pipeline / FeatureUnion-style .steps containers.
        steps_attr = getattr(obj, "steps", None)
        if isinstance(steps_attr, list) and steps_attr and all(
            isinstance(s, tuple) and len(s) == 2 for s in steps_attr
        ):
            names = [type(est).__name__ for _, est in steps_attr]
            return obj, steps_attr[-1][1], names
        return obj, obj, [type(obj).__name__]

    @staticmethod
    def validate_is_fitted(estimator: Any) -> None:
        """Raise MODEL_NOT_FITTED when the estimator never saw .fit()."""
        try:
            from sklearn.exceptions import NotFittedError  # noqa: PLC0415
            from sklearn.utils.validation import check_is_fitted  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "scikit-learn is required to validate fitted state",
                code="MODEL_VALIDATION_UNAVAILABLE",
            ) from exc
        try:
            check_is_fitted(estimator)
        except NotFittedError as exc:
            raise ModelLoadError(
                f"Model '{type(estimator).__name__}' is not fitted yet — "
                "call .fit(X, y) before uploading",
                details={"estimator_class": type(estimator).__name__},
                code="MODEL_NOT_FITTED",
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot verify fitted state of '{type(estimator).__name__}': {exc}",
                details={"estimator_class": type(estimator).__name__},
                code="NOT_A_VALID_MODEL_ARTIFACT",
            ) from exc

    @staticmethod
    def extract_schema(estimator: Any) -> ModelSchema:
        """Build the ordered feature list (embedded names or placeholders)."""
        names_attr = getattr(estimator, "feature_names_in_", None)
        n = _safe_n_features(estimator)
        if names_attr is not None:
            names = [str(v) for v in list(names_attr)]
            if n is None:
                n = len(names)
        elif n is not None:
            names = [f"feature_{i}" for i in range(n)]
        else:
            raise ModelLoadError(
                f"Cannot determine input dimensionality of '{type(estimator).__name__}'",
                details={"estimator_class": type(estimator).__name__},
                code="SCHEMA_EXTRACTION_FAILED",
            )
        auto = names_attr is None
        features = [
            FeatureRequirement(
                index=i,
                name=name,
                expected_dtype="float32",
                is_auto_generated_name=auto,
            )
            for i, name in enumerate(names)
        ]
        classes: list[str | int | float] | None = None
        raw_classes = getattr(estimator, "classes_", None)
        if raw_classes is not None:
            try:
                classes = [v.item() if hasattr(v, "item") else v for v in list(raw_classes)]
            except TypeError:
                classes = None
        return ModelSchema(n_features_in=n, features=features, classes=classes)
