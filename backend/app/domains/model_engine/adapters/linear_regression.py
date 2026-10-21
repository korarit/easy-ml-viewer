"""LinearRegression adapter — first registered family (Phase 03 core, Phase 04 slice).

Resolves any ``sklearn.linear_model.LinearRegression`` estimator (including
the final step of a Pipeline / GridSearchCV unwrap) and exposes weights,
batched prediction, exact linear traces and the tilted-hyperplane surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domains.model_engine.base_adapter import BaseModelAdapter
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import (
    GridSpec3D,
    InternalFlowRowTrace,
    PredictionBatchResult,
    SurfaceMesh3DData,
    TaskType,
    Visual3DType,
)


def _is_linear_regression(estimator: Any) -> bool:
    return (
        type(estimator).__name__ == "LinearRegression"
        and type(estimator).__module__.startswith("sklearn.linear_model")
    )


@ModelRegistry.register("linear_regression")
class LinearRegressionAdapter(BaseModelAdapter):
    algorithm_key = "linear_regression"
    display_name = "Linear Regression (OLS)"
    family_name = "Linear Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return _is_linear_regression(estimator)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        coef = getattr(estimator, "coef_", None)
        if coef is None:
            return [None] * n_features
        flat = np.asarray(coef, dtype=np.float64).ravel()
        weights: list[float | None] = []
        for i in range(n_features):
            if i < flat.size and np.isfinite(flat[i]):
                weights.append(float(flat[i]))
            else:
                weights.append(None)
        return weights

    def _predict_values(self, model_obj: Any, X_float32: np.ndarray) -> np.ndarray:
        X = np.asarray(X_float32, dtype=np.float32)
        return np.asarray(model_obj.predict(X), dtype=np.float64).ravel()

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        preds = self._predict_values(model_obj, X_float32)
        safe = [float(v) if np.isfinite(v) else 0.0 for v in preds]
        return PredictionBatchResult(predictions=safe)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
    ) -> list[InternalFlowRowTrace]:
        X = np.asarray(X_float32, dtype=np.float64)
        final_est = getattr(model_obj, "steps", [None, getattr(model_obj, "_final_estimator", model_obj)])[-1]
        if isinstance(final_est, (tuple, list)):
            final_est = final_est[1]
        coef = np.asarray(getattr(final_est, "coef_", np.zeros(X.shape[1])), dtype=np.float64).ravel()
        intercept = getattr(final_est, "intercept_", 0.0)
        base = float(np.asarray(intercept).ravel()[0]) if np.ndim(intercept) else float(intercept)
        col_means = np.nanmean(X, axis=0) if X.size else np.zeros_like(coef)
        traces: list[InternalFlowRowTrace] = []
        for r in row_indices:
            if r < 0 or r >= X.shape[0]:
                continue
            contributions = {
                feature_names[j] if j < len(feature_names) else f"feature_{j}": float(
                    coef[j] * (X[r, j] - col_means[j])
                )
                for j in range(min(X.shape[1], coef.size))
            }
            final = base + float(np.sum([coef[j] * X[r, j] for j in range(min(X.shape[1], coef.size))]))
            traces.append(
                InternalFlowRowTrace(
                    row_index=r,
                    base_value=base,
                    final_prediction=final,
                    feature_contributions=contributions,
                    extra_mechanics={"intercept": base},
                )
            )
        return traces

    def compute_3d_surface(
        self, model_obj: Any, grid_spec: GridSpec3D, n_features: int
    ) -> SurfaceMesh3DData:
        final_est = model_obj
        steps = getattr(model_obj, "steps", None)
        if isinstance(steps, list) and steps:
            final_est = steps[-1][1]
        coef = np.asarray(getattr(final_est, "coef_", np.zeros(n_features)), dtype=np.float64).ravel()
        intercept = getattr(final_est, "intercept_", 0.0)
        base = float(np.asarray(intercept).ravel()[0]) if np.ndim(intercept) else float(intercept)
        res = grid_spec.resolution
        xs = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float64)
        ys = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float64)
        wx = float(coef[grid_spec.axis_x_feature_index]) if grid_spec.axis_x_feature_index < coef.size else 0.0
        wy = float(coef[grid_spec.axis_y_feature_index]) if grid_spec.axis_y_feature_index < coef.size else 0.0
        fixed = base + sum(
            (float(coef[j]) if j < coef.size else 0.0) * m
            for j, m in grid_spec.fixed_feature_medians.items()
        )
        xx, yy = np.meshgrid(xs, ys)
        zz = fixed + wx * xx + wy * yy
        return SurfaceMesh3DData(
            visual_type=Visual3DType.FLAT_HYPERPLANE,
            resolution=res,
            x_grid=[float(v) for v in xs],
            y_grid=[float(v) for v in ys],
            z_values=[float(v) for v in zz.ravel()],
        )
