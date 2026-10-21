# backend/app/domains/model_engine/adapters/linear/sgd.py
"""Phase 21 — SGD / online-linear family adapters (margin corridor + epsilon tube).

Covers ``SGDClassifier``, ``SGDRegressor``, ``Perceptron``,
``PassiveAggressiveClassifier`` and ``PassiveAggressiveRegressor``.

Classifier waterfall identity per resolved class ``k``::

    b_k + sum_j (w_{k,j} * x_j) == decision_function(X)[k]

Regressor waterfall identity::

    b + sum_j (w_j * x_j) == predict(X)

Geometric margin convention (blueprint section 2.1)::

    d_geom(x) = f(x) / ||w||_2

with the standard ``f(x) = +/-1`` corridor planes sitting at a true
geometric distance of ``+/-1 / ||w||_2`` from the decision plane.
``|f(x)| < 1.0`` marks a row as ``INSIDE MARGIN CORRIDOR``.  Regressor
corridor width is ``epsilon`` (the ``huber`` / ``epsilon_insensitive``
no-penalty half-width) instead of ``1.0``.

Probability convention for classifiers:

- ``loss in {"log_loss", "log", "modified_huber"}`` expose native
  ``predict_proba`` — delegated to sklearn (numerically stable paths).
- All other losses (``hinge``, ``squared_hinge``, ``perceptron``) have no
  native ``predict_proba`` (``@available_if`` hides it); the adapter
  reports calibrated pseudo-confidence ``sigmoid(f)`` / ``softmax(f)``
  for HUD coloring only, flagged with ``is_pseudo_proba=True``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    SGDClassifier,
    SGDRegressor,
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
W_NORM_FLOOR = 1e-12

_CLASSIFIER_TYPES = (SGDClassifier, PassiveAggressiveClassifier, Perceptron)
_REGRESSOR_TYPES = (SGDRegressor, PassiveAggressiveRegressor)

# Blueprint edge case #3: pre-1.1 pickles may carry loss == "log".
PROBA_LOSSES = {"log_loss", "log", "modified_huber"}

# Human-readable loss regime shown on the HUD badge.
LOSS_REGIME_LABELS = {
    "hinge": "Linear SVM Mode",
    "squared_hinge": "Squared-Hinge SVM Mode",
    "log_loss": "Logistic Mode",
    "log": "Logistic Mode (legacy)",
    "modified_huber": "Modified-Huber Mode",
    "perceptron": "Perceptron Mode",
    "squared_error": "OLS Mode",
    "huber": "Huber-Robust Mode",
    "epsilon_insensitive": "Epsilon-Tube Mode",
    "squared_epsilon_insensitive": "Squared-Epsilon Mode",
}


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
    """Return ``(coef_2d (K, D) float32, intercepts (K,) float32)`` finite-guarded.

    Blueprint edge case #6: ``SGDRegressor.coef_`` is 1-D ``(D,)`` while
    classifiers expose ``(1, D)`` / ``(C, D)`` — ``atleast_2d`` normalizes both.
    """
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


def _w_norms(coef_2d: np.ndarray) -> np.ndarray:
    """Per-row L2 weight norms, floored (blueprint section 6.2: never divide by 0)."""
    norms = np.linalg.norm(np.asarray(coef_2d, dtype=np.float64), axis=1)
    return np.maximum(norms, W_NORM_FLOOR).astype(np.float64)


def _resolve_loss(estimator: Any, default: str) -> str:
    """Blueprint edge case #5: Perceptron / PA have no standard ``loss`` attr."""
    raw = getattr(estimator, "loss", default)
    return str(raw) if raw is not None else default


def _has_native_proba(model_obj: Any, loss_name: str) -> bool:
    """Blueprint edge case #1: check loss set AND attribute (hinge hides it)."""
    if loss_name not in PROBA_LOSSES:
        return False
    try:
        return callable(getattr(model_obj, "predict_proba", None))
    except Exception:
        return False


def _resolve_float(estimator: Any, names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = getattr(estimator, name, None)
        if raw is None:
            continue
        try:
            val = float(np.asarray(raw, dtype=np.float64).ravel()[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(val):
            return float(val)
    return float(default)


def _resolve_int(estimator: Any, name: str, default: int = 0) -> int:
    raw = getattr(estimator, name, None)
    try:
        return int(np.asarray(raw).max())
    except (TypeError, ValueError):
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default


def _resolve_str(estimator: Any, name: str, default: str) -> str:
    raw = getattr(estimator, name, None)
    return str(raw) if raw is not None else default


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
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)


def _native_scores_probas(
    model_obj: Any, X_native: np.ndarray, loss_name: str
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return ``(scores_2d (N, C), probas_2d (N, C) | None, is_native)``."""
    try:
        raw = np.asarray(model_obj.decision_function(X_native), dtype=np.float64)
    except (AttributeError, NotImplementedError, ValueError):
        try:
            raw = np.asarray(model_obj.predict(X_native), dtype=np.float64)
        except Exception:
            raw = np.zeros((X_native.shape[0],), dtype=np.float64)
    scores = raw.reshape(-1, 1) if raw.ndim == 1 else raw
    probas: np.ndarray | None = None
    is_native = False
    if _has_native_proba(model_obj, loss_name):
        try:
            probas = np.asarray(model_obj.predict_proba(X_native), dtype=np.float64)
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
            is_native = True
        except (AttributeError, NotImplementedError, ValueError):
            probas = None
    if probas is None:
        # Calibrated pseudo-confidence for HUD coloring only (blueprint 2.1.3).
        if scores.shape[1] == 1:
            p1 = _expit(scores[:, 0])
            probas = np.column_stack([1.0 - p1, p1])
        else:
            probas = _softmax(scores)
    return scores, probas, is_native


def _common_hyperparams(estimator: Any, loss_name: str) -> dict[str, Any]:
    return {
        "loss": loss_name,
        "loss_regime": LOSS_REGIME_LABELS.get(loss_name, "Online Linear Mode"),
        "penalty": _resolve_str(estimator, "penalty", "l2"),
        "alpha": _resolve_float(estimator, ("alpha",), 0.0001),
        "l1_ratio": _resolve_float(estimator, ("l1_ratio",), 0.15),
        "epsilon": _resolve_float(estimator, ("epsilon",), 0.1),
        "learning_rate": _resolve_str(estimator, "learning_rate", "optimal"),
        "eta0": _resolve_float(estimator, ("eta0",), 0.0),
        "n_iter": _resolve_int(estimator, "n_iter_", 0),
        "total_weight_updates_t": _resolve_int(estimator, "t_", 0),
    }


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


def _numeric_surface_scores(
    model_obj: Any,
    grid_spec: GridSpec3D,
    n_features: int,
    is_classifier: bool,
    resolved: int,
    res: int,
) -> tuple[list[float], list[float], list[float]]:
    """Evaluate the full (possibly pipelined) model on the raw-space grid.

    Pipeline-fitted weights live in transformed space, so the analytic
    ``w·x + b`` plane would mix scaled weights with raw grid coordinates.
    Routing the grid through ``model_obj`` keeps raw CSV ranges valid.
    """
    x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
    y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
    n_points = res * res
    grid = np.zeros((n_points, max(n_features, 1)), dtype=np.float32)
    for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
        if 0 <= feat_idx < grid.shape[1] and np.isfinite(med_val):
            grid[:, feat_idx] = np.float32(med_val)
    ax = grid_spec.axis_x_feature_index
    ay = grid_spec.axis_y_feature_index
    if 0 <= ax < grid.shape[1]:
        grid[:, ax] = xx.ravel()
    if n_features > 1 and ay != ax and 0 <= ay < grid.shape[1]:
        grid[:, ay] = yy.ravel()
    try:
        if is_classifier:
            raw = np.asarray(model_obj.decision_function(grid), dtype=np.float64)
            col = min(resolved, raw.shape[1] - 1) if raw.ndim == 2 else 0
            flat = raw[:, col] if raw.ndim == 2 else raw.ravel()
        else:
            flat = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
    except Exception:
        flat = np.zeros((n_points,), dtype=np.float64)
    flat = np.where(np.isfinite(flat), flat, 0.0)
    z_list = [float(v) for v in flat.tolist()]
    return [float(v) for v in x_lin], [float(v) for v in y_lin], z_list


def _plane_surface(
    w: np.ndarray,
    b: float,
    grid_spec: GridSpec3D,
    n_features: int,
    res_cap: int = MAX_SURFACE_RESOLUTION,
) -> tuple[list[float], list[float], list[float], float, float, float]:
    res = int(min(max(grid_spec.resolution, 2), res_cap))
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
    z_list = [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(zz, dtype=np.float32).ravel()]
    return [float(v) for v in x_lin], [float(v) for v in y_lin], z_list, wx, wy, float(off_axis)


@ModelRegistry.register("sgd_classifier")
class SgdClassifierAdapter(BaseModelAdapter):
    display_name = "SGD Classifier (Online Linear Margin)"
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
        estimator = _final_estimator(model_obj)
        loss_name = _resolve_loss(estimator, "perceptron")
        _, probas, _ = _native_scores_probas(model_obj, X_in, loss_name)
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if probas is not None and probas.size:
            probabilities = [[float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()]
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
        return PredictionBatchResult(predictions=py_preds, probabilities=probabilities, confidences=confidences)

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
        norms = _w_norms(coef_2d)
        loss_name = _resolve_loss(estimator, "perceptron")
        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
        is_binary = len(classes) == 2 if classes else coef_2d.shape[0] <= 1
        hyper = _common_hyperparams(estimator, loss_name)
        X_trans = _transform_for_contrib(model_obj, X_float32)
        X_native = _as_dense_float32(X_float32)
        scores_2d, probas_2d = _native_scores_probas(model_obj, X_native, loss_name)[:2]
        has_native = bool(_has_native_proba(model_obj, loss_name))
        try:
            native_preds = list(np.asarray(model_obj.predict(X_native)).ravel().tolist())
        except Exception:
            native_preds = [None] * X_native.shape[0]

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
                elif probas_2d is not None and probas_2d.shape[0] > row_id and has_native:
                    k = int(np.argmax(probas_2d[row_id]))
                else:
                    k = _resolve_row_k(coef_2d.shape[0], target_idx, None)
                resolved = int(k)
            w = np.asarray(coef_2d[min(k, coef_2d.shape[0] - 1)], dtype=np.float32)
            b = float(intercepts[min(k, intercepts.size - 1)]) if intercepts.size else 0.0
            w_norm = float(norms[min(k, norms.size - 1)])
            x_vec = (
                np.asarray(X_trans[pos], dtype=np.float32)
                if X_trans.shape[0] == len(row_indices)
                else np.asarray(X_trans[row_id], dtype=np.float32)
            )
            contribs = (x_vec[: w.size] * w[: x_vec.size]).astype(np.float32)
            contribs = np.where(np.isfinite(contribs), contribs, 0.0).astype(np.float32)
            # Geometric-space contributions: phi_tilde = (w_j / ||w||) * x_j.
            geom_contribs = (contribs / max(w_norm, W_NORM_FLOOR)).astype(np.float32)
            names = feature_names or [f"feature_{i}" for i in range(contribs.size)]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(len(contribs), len(names) if names else len(contribs)))
            }
            if not contrib_map:
                contrib_map = {f"feature_{i}": float(contribs[i]) for i in range(len(contribs))}
            score_col = min(k, scores_2d.shape[1] - 1)
            raw_score = float(scores_2d[row_id, score_col]) if scores_2d.size else 0.0
            geom_dist = float(raw_score / max(w_norm, W_NORM_FLOOR))
            inside_margin = bool(abs(raw_score) < 1.0)
            if inside_margin:
                margin_status = "INSIDE MARGIN CORRIDOR"
            elif raw_score >= 0:
                margin_status = "SAFE MARGIN (> +1)"
            else:
                margin_status = "SAFE MARGIN (< -1)"
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
                "is_classifier": True,
                "is_binary": bool(is_binary),
                "classes": [str(c) for c in classes] if classes else [],
                "class_labels": classes if classes else [],
                "has_native_proba": bool(has_native),
                "is_pseudo_proba": bool(not has_native),
                "decision_scores": [float(v) for v in np.asarray(scores_2d[row_id]).ravel()] if scores_2d.size else [],
                "raw_score": raw_score,
                "logit": raw_score,
                "geometric_distance": geom_dist,
                "geometric_contributions": [float(v) for v in geom_contribs],
                "w_norm": w_norm,
                "weight_l2_norm": w_norm,
                "inside_margin": inside_margin,
                "margin_status": margin_status,
                "margin_width": 1.0,
                "probabilities": proba_row,
                "predicted_class": final_pred,
                "winning_class_idx": int(np.argmax(proba_row)) if proba_row else int(resolved),
                "decision_threshold": 0.0,
                **hyper,
            }
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
        norms = _w_norms(coef_2d)
        loss_name = _resolve_loss(estimator, "perceptron")
        hyper = _common_hyperparams(estimator, loss_name)
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
        w_norm = float(norms[min(k, norms.size - 1)])
        is_pipeline = hasattr(model_obj, "steps")
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        if is_pipeline:
            x_grid, y_grid, z_list = _numeric_surface_scores(
                model_obj, grid_spec, n_features, True, int(k), res
            )
            wx = float(w[grid_spec.axis_x_feature_index]) if 0 <= grid_spec.axis_x_feature_index < w.size else 0.0
            wy = float(w[grid_spec.axis_y_feature_index]) if 0 <= grid_spec.axis_y_feature_index < w.size else 0.0
            off_axis = float(b)
        else:
            x_grid, y_grid, z_list, wx, wy, off_axis = _plane_surface(w, b, grid_spec, n_features)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_grid)),
            x_grid=x_grid,
            y_grid=y_grid,
            z_values=z_list,
            extra_overlays={
                "wx": wx,
                "wy": wy,
                "intercept_offset": float(off_axis),
                "base_intercept": float(b),
                "weight_row": [float(v) for v in w],
                "is_binary": bool(is_binary),
                "classes": [str(c) for c in classes] if classes else [],
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "has_native_proba": bool(_has_native_proba(model_obj, loss_name)),
                "boundary_threshold": 0.0,
                "decision_threshold": 0.0,
                # Corridor planes render client-side as +/- margin_width
                # offsets of this single geometry (blueprint section 6.1).
                "margin_width": 1.0,
                "corridor_half_width_score": 1.0,
                "corridor_half_width_geometric": float(1.0 / max(w_norm, W_NORM_FLOOR)),
                "w_norm": w_norm,
                "weight_l2_norm": w_norm,
                "gpu_formula_valid": not bool(is_pipeline),
                "is_classifier": True,
                **hyper,
            },
        )


@ModelRegistry.register("sgd_regressor")
class SgdRegressorAdapter(BaseModelAdapter):
    display_name = "SGD Regressor (Online Linear Tube)"
    family_name = "Linear Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _REGRESSOR_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        coef_2d, _ = _extract_coef_intercept(estimator)
        weights = coef_2d[0]
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
        norms = _w_norms(coef_2d)
        loss_name = _resolve_loss(estimator, "squared_error")
        hyper = _common_hyperparams(estimator, loss_name)
        w = np.asarray(coef_2d[0], dtype=np.float32)
        b = float(intercepts[0]) if intercepts.size else 0.0
        w_norm = float(norms[0]) if norms.size else 1.0
        epsilon = float(hyper["epsilon"])
        X_trans = _transform_for_contrib(model_obj, X_float32)
        X_native = _as_dense_float32(X_float32)
        try:
            native_preds = np.asarray(model_obj.predict(X_native)).ravel()
        except Exception:
            native_preds = np.zeros((X_native.shape[0],), dtype=np.float64)

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
                "target_idx": 0,
                "target_class_idx": 0,
                "is_classifier": False,
                "is_pseudo_proba": False,
                "raw_score": final_pred,
                "w_norm": w_norm,
                "weight_l2_norm": w_norm,
                # Epsilon-tube telemetry (blueprint 2.1.4): per-row
                # inside-tube status needs the ground truth, which the
                # trace endpoint attaches as residual_error instead.
                "margin_width": epsilon,
                "epsilon": epsilon,
                **hyper,
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
        norms = _w_norms(coef_2d)
        loss_name = _resolve_loss(estimator, "squared_error")
        hyper = _common_hyperparams(estimator, loss_name)
        w = np.asarray(coef_2d[0], dtype=np.float64)
        b = float(intercepts[0]) if intercepts.size else 0.0
        w_norm = float(norms[0]) if norms.size else 1.0
        epsilon = float(hyper["epsilon"])
        is_pipeline = hasattr(model_obj, "steps")
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        if is_pipeline:
            x_grid, y_grid, z_list = _numeric_surface_scores(
                model_obj, grid_spec, n_features, False, 0, res
            )
            wx = float(w[grid_spec.axis_x_feature_index]) if 0 <= grid_spec.axis_x_feature_index < w.size else 0.0
            wy = float(w[grid_spec.axis_y_feature_index]) if 0 <= grid_spec.axis_y_feature_index < w.size else 0.0
            off_axis = float(b)
        else:
            x_grid, y_grid, z_list, wx, wy, off_axis = _plane_surface(w, b, grid_spec, n_features)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_grid)),
            x_grid=x_grid,
            y_grid=y_grid,
            z_values=z_list,
            extra_overlays={
                "wx": wx,
                "wy": wy,
                "intercept_offset": float(off_axis),
                "base_intercept": float(b),
                "weight_row": [float(v) for v in w],
                "target_idx": 0,
                "resolved_target_idx": 0,
                "boundary_threshold": None,
                "margin_width": epsilon,
                "epsilon": epsilon,
                "corridor_half_width_score": epsilon,
                "w_norm": w_norm,
                "weight_l2_norm": w_norm,
                "gpu_formula_valid": not bool(is_pipeline),
                "is_classifier": False,
                **hyper,
            },
        )
