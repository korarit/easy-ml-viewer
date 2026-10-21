# backend/app/domains/model_engine/adapters/linear/regularized.py
"""Phase 19 — Ridge & Lasso regularized linear adapters.

Covers ``Ridge``, ``RidgeCV``, ``Lasso``, ``LassoCV``, ``LassoLars``,
``LassoLarsCV``, ``LassoLarsIC`` (regression) plus ``RidgeClassifier`` and
``RidgeClassifierCV`` (classification via least-squares scores).

Waterfall identity per resolved target ``k``::

    b_k + sum_j (w_{k,j} * x_j) == predict/score(X)[k]

Sparsity convention (Lasso geometry): ``|w_j| <= 1e-12`` counts as pruned
(``L1 PRUNED``). Ridge weights shrink but never hit exact zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
)

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
ZERO_TOL = 1e-12

_REGRESSION_TYPES = (Ridge, RidgeCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC)
_CLASSIFIER_TYPES = (RidgeClassifier, RidgeClassifierCV)
_L1_TYPES = (Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC)


def _final_estimator(model_obj: Any) -> Any:
    steps = getattr(model_obj, "steps", None)
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, (tuple, list)) and len(last) == 2:
            return last[1]
        return last
    return getattr(model_obj, "_final_estimator", model_obj)


def _is_sparse(matrix: Any) -> bool:
    try:
        import scipy.sparse as _sp  # noqa: PLC0415

        return bool(_sp.issparse(matrix))
    except ImportError:
        return False


def _as_dense_float32(matrix: Any) -> np.ndarray:
    if _is_sparse(matrix):
        matrix = matrix.toarray()
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))


def _extract_coef_intercept(estimator: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coef_2d (K, D) float32, intercepts (K,) float32)`` finite-guarded."""
    raw_coef = np.asarray(getattr(estimator, "coef_"), dtype=np.float32)
    if raw_coef.ndim == 1:
        raw_coef = raw_coef.reshape(1, -1)
    elif raw_coef.ndim > 2:
        raw_coef = raw_coef.reshape(raw_coef.shape[0], -1)
    coef_2d = np.ascontiguousarray(raw_coef, dtype=np.float32)

    raw_intercept = getattr(estimator, "intercept_", 0.0)
    intercepts = np.atleast_1d(np.asarray(raw_intercept, dtype=np.float32)).ravel()
    if intercepts.size == 0:
        intercepts = np.zeros((coef_2d.shape[0],), dtype=np.float32)
    elif intercepts.size == 1 and coef_2d.shape[0] > 1:
        intercepts = np.repeat(intercepts, coef_2d.shape[0]).astype(np.float32)
    intercepts = np.ascontiguousarray(intercepts.ravel(), dtype=np.float32)
    if intercepts.size < coef_2d.shape[0]:
        pad = np.zeros((coef_2d.shape[0] - intercepts.size,), dtype=np.float32)
        intercepts = np.concatenate([intercepts, pad])
    else:
        intercepts = intercepts[: coef_2d.shape[0]]
    intercepts = np.where(np.isfinite(intercepts), intercepts, 0.0).astype(np.float32)
    coef_2d = np.where(np.isfinite(coef_2d), coef_2d, 0.0).astype(np.float32)
    return coef_2d, intercepts


def _resolve_alpha(estimator: Any) -> float:
    raw = getattr(estimator, "alpha_", getattr(estimator, "alpha", 1.0))
    try:
        arr = np.asarray(raw, dtype=np.float64).ravel()
        val = float(arr[0]) if arr.size else 1.0
    except (TypeError, ValueError):
        try:
            val = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            val = 1.0
    return val if np.isfinite(val) else 1.0


def _penalty_type(estimator: Any) -> str:
    return "L1 (Lasso)" if isinstance(estimator, _L1_TYPES) else "L2 (Ridge)"


def _sparsity_stats(w: np.ndarray) -> dict[str, Any]:
    w = np.asarray(w, dtype=np.float64)
    active = np.where(np.abs(w) > ZERO_TOL)[0].tolist()
    zeroed = np.where(np.abs(w) <= ZERO_TOL)[0].tolist()
    l1 = float(np.sum(np.abs(w)))
    l2 = float(np.linalg.norm(w))
    denom = max(1, int(w.size))
    return {
        "active_indices": [int(i) for i in active],
        "zeroed_indices": [int(i) for i in zeroed],
        "active_weight_count": int(len(active)),
        "l1_norm": float(l1) if np.isfinite(l1) else 0.0,
        "l2_norm": float(l2) if np.isfinite(l2) else 0.0,
        "sparsity_ratio": float(len(zeroed) / denom),
    }


def _expit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softmax(z2d: np.ndarray) -> np.ndarray:
    z = np.asarray(z2d, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    m = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - m)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.maximum(s, 1e-12)


def _classifier_scores_probas(
    model_obj: Any, X_native: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(scores_2d (N, C), probas_2d (N, C) | None)`` pseudo-probabilities."""
    try:
        raw = np.asarray(model_obj.decision_function(X_native), dtype=np.float64)
    except (AttributeError, NotImplementedError, ValueError):
        try:
            raw = np.asarray(model_obj.predict(X_native), dtype=np.float64)
        except Exception:
            raw = np.zeros((X_native.shape[0],), dtype=np.float64)
    if raw.ndim == 1:
        scores = raw.reshape(-1, 1)
        p1 = _expit(raw)
        probas = np.column_stack([1.0 - p1, p1])
    else:
        scores = raw
        probas = _softmax(raw)
    return scores, probas


def _transform_for_contrib(model_obj: Any, X_float32: np.ndarray) -> np.ndarray:
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32, dtype=np.float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _resolve_row_k(n_rows: int, target_idx: int, target_class_idx: int | None) -> int:
    if target_class_idx is not None and 0 <= int(target_class_idx) < n_rows:
        return int(target_class_idx)
    return int(min(max(int(target_idx), 0), max(n_rows - 1, 0)))


@ModelRegistry.register("regularized_linear")
class RegularizedLinearAdapter(BaseModelAdapter):
    display_name = "Ridge / Lasso (Regularized Linear)"
    family_name = "Linear Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _REGRESSION_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        coef_2d, _ = _extract_coef_intercept(estimator)
        weights = np.mean(coef_2d, axis=0).astype(np.float32) if coef_2d.shape[0] > 1 else coef_2d[0]
        out: list[float | None] = []
        for i in range(n_features):
            if i < weights.size and bool(np.isfinite(weights[i])):
                out.append(float(weights[i]))
            else:
                out.append(None)
        return out

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        preds = np.asarray(model_obj.predict(X_in)).ravel()
        if preds.ndim == 2:
            preds = np.asarray(preds)[:, 0]
        return PredictionBatchResult(
            predictions=[float(v) if np.isfinite(v) else 0.0 for v in np.asarray(preds).ravel()]
        )

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        estimator = _final_estimator(model_obj)
        coef_2d, intercepts = _extract_coef_intercept(estimator)
        X_trans = _transform_for_contrib(model_obj, X_float32)
        X_native = _as_dense_float32(X_float32)
        try:
            native_preds = np.asarray(model_obj.predict(X_native)).ravel()
        except Exception:
            native_preds = np.zeros((X_native.shape[0],), dtype=np.float64)

        k_fixed = _resolve_row_k(coef_2d.shape[0], target_idx, target_class_idx)
        w = np.asarray(coef_2d[k_fixed], dtype=np.float32)
        b = float(intercepts[k_fixed]) if k_fixed < intercepts.size else 0.0
        stats = _sparsity_stats(w)
        alpha = _resolve_alpha(estimator)
        penalty = _penalty_type(estimator)
        criterion = getattr(estimator, "criterion", None)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_native.shape[0]:
                continue
            x_vec = (
                np.asarray(X_trans[pos], dtype=np.float32)
                if X_trans.shape[0] == len(row_indices)
                else np.asarray(X_trans[row_id], dtype=np.float32)
            )
            contribs = (x_vec[: w.size] * w[: x_vec.size]).astype(np.float32)
            contribs = np.where(np.isfinite(contribs), contribs, 0.0).astype(np.float32)
            names = feature_names or [f"feature_{i}" for i in range(contribs.size)]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(len(contribs), len(names) if names else len(contribs)))
            }
            if not contrib_map:
                contrib_map = {f"feature_{i}": float(contribs[i]) for i in range(len(contribs))}
            try:
                final_pred = float(native_preds[row_id])
            except (IndexError, TypeError, ValueError):
                final_pred = float(np.sum(contribs, dtype=np.float64) + b)
            if not np.isfinite(final_pred):
                final_pred = 0.0
            extra: dict[str, Any] = {
                "weights": [float(v) for v in w],
                "intercept": b,
                "target_idx": int(k_fixed),
                "target_class_idx": int(k_fixed),
                "penalty_type": penalty,
                "alpha": float(alpha),
                "is_classifier": False,
                "is_pseudo_proba": False,
                "criterion": str(criterion) if criterion is not None else None,
                **stats,
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=row_id,
                    base_value=b,
                    final_prediction=final_pred,
                    feature_contributions=contrib_map,
                    extra_mechanics=extra,
                )
            )
        return traces

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        estimator = _final_estimator(model_obj)
        coef_2d, intercepts = _extract_coef_intercept(estimator)
        k = _resolve_row_k(coef_2d.shape[0], target_idx, target_class_idx)
        w = np.asarray(coef_2d[k], dtype=np.float64)
        b = float(intercepts[k]) if k < intercepts.size else 0.0

        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")

        ax = grid_spec.axis_x_feature_index
        ay = grid_spec.axis_y_feature_index
        off_axis = float(b)
        for j in range(max(n_features, w.size)):
            if j == ax or j == ay or j >= w.size:
                continue
            med = float(grid_spec.fixed_feature_medians.get(j, 0.0))
            if np.isfinite(med):
                off_axis += float(w[j]) * med
        wx = float(w[ax]) if 0 <= ax < w.size else 0.0
        wy = float(w[ay]) if (0 <= ay < w.size and ay != ax) else 0.0
        if n_features <= 1:
            wy = 0.0
        zz = (wx * xx.astype(np.float64) + wy * yy.astype(np.float64) + off_axis).astype(np.float32)

        stats = _sparsity_stats(w)
        alpha = _resolve_alpha(estimator)
        penalty = _penalty_type(estimator)
        is_pipeline = hasattr(model_obj, "steps")
        z_list = [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(zz, dtype=np.float32).ravel()]
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=res,
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=z_list,
            extra_overlays={
                "wx": wx,
                "wy": wy,
                "is_x_pruned": bool(abs(wx) <= ZERO_TOL),
                "is_y_pruned": bool(abs(wy) <= ZERO_TOL),
                "zero_influence_axes": bool(abs(wx) <= ZERO_TOL and abs(wy) <= ZERO_TOL),
                "intercept_offset": float(off_axis),
                "base_intercept": float(b),
                "weight_row": [float(v) for v in w],
                "penalty_type": penalty,
                "alpha": float(alpha),
                "target_idx": int(k),
                "resolved_target_idx": int(k),
                "boundary_threshold": None,
                "gpu_formula_valid": not bool(is_pipeline),
                "is_classifier": False,
                **stats,
            },
        )


@ModelRegistry.register("ridge_classifier")
class RidgeClassifierAdapter(BaseModelAdapter):
    display_name = "Ridge Classifier (Regularized Linear)"
    family_name = "Linear Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _CLASSIFIER_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        coef_2d, _ = _extract_coef_intercept(estimator)
        weights = np.mean(coef_2d, axis=0).astype(np.float32) if coef_2d.shape[0] > 1 else coef_2d[0]
        out: list[float | None] = []
        for i in range(n_features):
            if i < weights.size and bool(np.isfinite(weights[i])):
                out.append(float(weights[i]))
            else:
                out.append(None)
        return out

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        preds = np.asarray(model_obj.predict(X_in)).ravel()
        _, probas = _classifier_scores_probas(model_obj, X_in)
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if probas is not None and probas.size:
            probabilities = [
                [float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()
            ]
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
        py_preds: list[float | int | str] = []
        for v in preds.tolist():
            if isinstance(v, (bool, np.bool_)):
                py_preds.append(int(v))
            elif isinstance(v, (np.integer, int)):
                py_preds.append(int(v))
            elif isinstance(v, (np.floating, float)):
                f = float(v)
                py_preds.append(int(f) if f.is_integer() else f)
            else:
                py_preds.append(v)
        return PredictionBatchResult(
            predictions=py_preds, probabilities=probabilities, confidences=confidences
        )

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        estimator = _final_estimator(model_obj)
        coef_2d, intercepts = _extract_coef_intercept(estimator)
        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
        is_binary = len(classes) == 2 if classes else coef_2d.shape[0] <= 1
        X_trans = _transform_for_contrib(model_obj, X_float32)
        X_native = _as_dense_float32(X_float32)
        scores_2d, probas_2d = _classifier_scores_probas(model_obj, X_native)
        try:
            native_preds = list(np.asarray(model_obj.predict(X_native)).ravel().tolist())
        except Exception:
            native_preds = [None] * X_native.shape[0]

        stats_full = _sparsity_stats(coef_2d[0] if coef_2d.size else np.zeros((0,)))
        alpha = _resolve_alpha(estimator)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_native.shape[0]:
                continue
            if is_binary:
                k = 0
                resolved = 1 if len(classes) >= 2 else 0
            else:
                if target_class_idx is not None and 0 <= int(target_class_idx) < coef_2d.shape[0]:
                    k = int(target_class_idx)
                elif probas_2d is not None and probas_2d.shape[0] > row_id:
                    k = int(np.argmax(probas_2d[row_id]))
                else:
                    k = _resolve_row_k(coef_2d.shape[0], target_idx, None)
                resolved = int(k)
            w = np.asarray(coef_2d[min(k, coef_2d.shape[0] - 1)], dtype=np.float32)
            b = float(intercepts[min(k, intercepts.size - 1)]) if intercepts.size else 0.0
            x_vec = (
                np.asarray(X_trans[pos], dtype=np.float32)
                if X_trans.shape[0] == len(row_indices)
                else np.asarray(X_trans[row_id], dtype=np.float32)
            )
            contribs = (x_vec[: w.size] * w[: x_vec.size]).astype(np.float32)
            contribs = np.where(np.isfinite(contribs), contribs, 0.0).astype(np.float32)
            names = feature_names or [f"feature_{i}" for i in range(contribs.size)]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(len(contribs), len(names) if names else len(contribs)))
            }
            if not contrib_map:
                contrib_map = {f"feature_{i}": float(contribs[i]) for i in range(len(contribs))}
            row_stats = _sparsity_stats(w)
            logit = float(scores_2d[row_id, min(k, scores_2d.shape[1] - 1)]) if scores_2d.size else 0.0
            proba_row = (
                [float(v) for v in np.asarray(probas_2d[row_id]).ravel()]
                if probas_2d is not None and row_id < probas_2d.shape[0]
                else []
            )
            final_pred = native_preds[row_id] if row_id < len(native_preds) else resolved
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if float(f).is_integer() else f
            extra: dict[str, Any] = {
                "weights": [float(v) for v in w],
                "intercept": b,
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "penalty_type": "L2 (Ridge)",
                "alpha": float(alpha),
                "is_classifier": True,
                "is_pseudo_proba": True,
                "is_binary": bool(is_binary),
                "classes": [str(c) for c in classes] if classes else [],
                "class_labels": classes if classes else [],
                "decision_scores": [float(v) for v in np.asarray(scores_2d[row_id]).ravel()] if scores_2d.size else [],
                "logit": logit,
                "probabilities": proba_row,
                "predicted_class": final_pred,
                "winning_class_idx": int(np.argmax(proba_row)) if proba_row else int(resolved),
                "decision_threshold": 0.0,
                "criterion": None,
                **row_stats,
            }
            # Keep aggregate sparsity of the primary row for HUD convenience.
            extra.setdefault("primary_sparsity_ratio", stats_full["sparsity_ratio"])
            traces.append(
                InternalFlowRowTrace(
                    row_index=row_id,
                    base_value=b,
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions=contrib_map,
                    extra_mechanics=extra,
                )
            )
        return traces

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        estimator = _final_estimator(model_obj)
        coef_2d, intercepts = _extract_coef_intercept(estimator)
        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
        is_binary = len(classes) == 2 if classes else coef_2d.shape[0] <= 1
        if is_binary:
            k = 0
            resolved = 1 if len(classes) >= 2 else 0
        else:
            k = _resolve_row_k(coef_2d.shape[0], target_idx, target_class_idx)
            resolved = int(k)
        w = np.asarray(coef_2d[min(k, coef_2d.shape[0] - 1)], dtype=np.float64)
        b = float(intercepts[min(k, intercepts.size - 1)]) if intercepts.size else 0.0

        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")

        ax = grid_spec.axis_x_feature_index
        ay = grid_spec.axis_y_feature_index
        off_axis = float(b)
        for j in range(max(n_features, w.size)):
            if j == ax or j == ay or j >= w.size:
                continue
            med = float(grid_spec.fixed_feature_medians.get(j, 0.0))
            if np.isfinite(med):
                off_axis += float(w[j]) * med
        wx = float(w[ax]) if 0 <= ax < w.size else 0.0
        wy = float(w[ay]) if (0 <= ay < w.size and ay != ax) else 0.0
        if n_features <= 1:
            wy = 0.0
        zz = (wx * xx.astype(np.float64) + wy * yy.astype(np.float64) + off_axis).astype(np.float32)

        row_stats = _sparsity_stats(w)
        alpha = _resolve_alpha(estimator)
        is_pipeline = hasattr(model_obj, "steps")
        z_list = [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(zz, dtype=np.float32).ravel()]
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=res,
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=z_list,
            extra_overlays={
                "wx": wx,
                "wy": wy,
                "is_x_pruned": bool(abs(wx) <= ZERO_TOL),
                "is_y_pruned": bool(abs(wy) <= ZERO_TOL),
                "zero_influence_axes": bool(abs(wx) <= ZERO_TOL and abs(wy) <= ZERO_TOL),
                "intercept_offset": float(off_axis),
                "base_intercept": float(b),
                "weight_row": [float(v) for v in w],
                "penalty_type": "L2 (Ridge)",
                "alpha": float(alpha),
                "is_binary": bool(is_binary),
                "classes": [str(c) for c in classes] if classes else [],
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "boundary_threshold": 0.0,
                "decision_threshold": 0.0,
                "gpu_formula_valid": not bool(is_pipeline),
                "is_classifier": True,
                "is_pseudo_proba": True,
                **row_stats,
            },
        )
