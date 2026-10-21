# backend/app/domains/model_engine/adapters/linear/linear_regression_adapter.py
"""Phase 04 — LinearRegressionAdapter (Ordinary Least Squares - OLS).

End-to-end vertical slice: exact per-term equation trace
(contrib_i = w_i * x_i), residual error, and 3D flat tilted
hyperplane surface via NumPy vectorization (float32, < 10 ms).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

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

MAX_SURFACE_RESOLUTION = 100


def _final_estimator(model_obj: Any) -> Any:
    """Unwrap Pipeline to its terminal estimator (Phase 04 edge case #4)."""
    steps = getattr(model_obj, "steps", None)
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, (tuple, list)) and len(last) == 2:
            return last[1]
        return last
    return getattr(model_obj, "_final_estimator", model_obj)


@ModelRegistry.register("linear_regression")
class LinearRegressionAdapter(BaseModelAdapter):
    display_name = "Linear Regression (Ordinary Least Squares)"
    family_name = "Linear Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, LinearRegression)

    @staticmethod
    def _extract_weights_and_intercept(
        estimator: LinearRegression, target_idx: int = 0
    ) -> tuple[np.ndarray, float]:
        coef = np.asarray(estimator.coef_, dtype=np.float32)
        if coef.ndim == 2:
            # Multi-target (n_targets, n_features): pick requested target row.
            row = min(max(target_idx, 0), coef.shape[0] - 1)
            weights = np.ascontiguousarray(coef[row], dtype=np.float32)
        else:
            weights = np.ascontiguousarray(coef.ravel(), dtype=np.float32)

        intercept_raw: Any = getattr(estimator, "intercept_", 0.0)
        if np.ndim(intercept_raw) >= 1:
            flat = np.asarray(intercept_raw, dtype=np.float32).ravel()
            if flat.size == 0:
                intercept = 0.0
            else:
                intercept = float(flat[min(max(target_idx, 0), flat.size - 1)])
        else:
            try:
                intercept = float(intercept_raw)
            except (TypeError, ValueError):
                intercept = 0.0
        if not np.isfinite(intercept):
            intercept = 0.0
        return weights, intercept

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        weights, _ = self._extract_weights_and_intercept(estimator)
        out: list[float | None] = []
        for i in range(n_features):
            if i < weights.size and bool(np.isfinite(weights[i])):
                out.append(float(weights[i]))
            else:
                out.append(None)
        return out

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_clean = np.ascontiguousarray(X_float32, dtype=np.float32)
        preds = np.asarray(model_obj.predict(X_clean), dtype=np.float32)
        if preds.ndim == 2:
            preds = preds[:, 0]
        return PredictionBatchResult(
            predictions=[float(v) if np.isfinite(v) else 0.0 for v in preds.ravel()]
        )

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
    ) -> list[InternalFlowRowTrace]:
        estimator = _final_estimator(model_obj)
        weights, intercept = self._extract_weights_and_intercept(estimator, target_idx)
        # Pipeline case: transform through preprocessors before weighting
        # so that sum(w_i * x'_i) + b == pipeline.predict(X) exactly (#4).
        if hasattr(model_obj, "steps"):
            try:
                X_trans = np.asarray(
                    model_obj[:-1].transform(np.asarray(X_float32, dtype=np.float32)),
                    dtype=np.float32,
                )
            except Exception:
                X_trans = np.asarray(X_float32, dtype=np.float32)
        else:
            X_trans = np.asarray(X_float32, dtype=np.float32)
        X_trans = np.ascontiguousarray(X_trans, dtype=np.float32)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_trans.shape[0]:
                continue
            x_vec = X_trans[pos] if X_trans.shape[0] == len(row_indices) else X_trans[row_id]
            x_vec = np.asarray(x_vec, dtype=np.float32)
            contribs = (x_vec * weights[: x_vec.size]).astype(np.float32)
            pred_val = float(np.sum(contribs, dtype=np.float32) + np.float32(intercept))
            contrib_map = {
                (feature_names[i] if i < len(feature_names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(len(contribs), len(feature_names) if feature_names else len(contribs)))
            }
            if not contrib_map:
                contrib_map = {f"feature_{i}": float(contribs[i]) for i in range(len(contribs))}
            traces.append(
                InternalFlowRowTrace(
                    row_index=row_id,
                    base_value=intercept,
                    final_prediction=pred_val,
                    feature_contributions=contrib_map,
                    extra_mechanics={
                        "transformed_inputs": [float(v) for v in x_vec],
                        "weights": [float(w) for w in weights],
                        "target_idx": int(target_idx),
                    },
                )
            )
        return traces

    def compute_3d_surface(
        self, model_obj: Any, grid_spec: GridSpec3D, n_features: int
    ) -> SurfaceMesh3DData:
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")

        # Pre-allocated float32 grid matrix: res*res x n_features (~100 KB @ 10 feats).
        n_points = res * res
        grid_matrix = np.zeros((n_points, max(n_features, 1)), dtype=np.float32)
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
            if 0 <= feat_idx < grid_matrix.shape[1]:
                grid_matrix[:, feat_idx] = np.float32(med_val)

        ax = grid_spec.axis_x_feature_index
        ay = grid_spec.axis_y_feature_index
        if 0 <= ax < grid_matrix.shape[1]:
            grid_matrix[:, ax] = xx.ravel()
        # 1D model (n_features == 1): lock Y as constant-depth ribbon (#3).
        if n_features > 1 and ay != ax and 0 <= ay < grid_matrix.shape[1]:
            grid_matrix[:, ay] = yy.ravel()
        # Same-axis selection guard: yy already ignored when ay == ax (#6).

        z_preds = np.asarray(model_obj.predict(grid_matrix), dtype=np.float32)
        if z_preds.ndim == 2:
            z_preds = z_preds[:, 0]

        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=res,
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_preds, dtype=np.float32).ravel()],
        )
