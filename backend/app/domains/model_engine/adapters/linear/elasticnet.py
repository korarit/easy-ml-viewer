# backend/app/domains/model_engine/adapters/linear/elasticnet.py
"""Phase 20 — ElasticNet family adapter (L1/L2 mix + multitask joint selection).

Covers ``ElasticNet``, ``ElasticNetCV``, ``MultiTaskElasticNet`` and
``MultiTaskElasticNetCV`` (single-target + joint multi-output regression).

Waterfall identity per resolved task ``m``::

    b_m + sum_j (w_{m,j} * x_j) == predict(X)[m]

Penalty convention (matches the Phase-20 blueprint)::

    lambda_1 = alpha * l1_ratio
    lambda_2 = 0.5 * alpha * (1 - l1_ratio)

Per-feature penalty share (HUD tooltip)::

    share_j = alpha * rho * |w_j| + 0.5 * alpha * (1 - rho) * w_j**2

Sparsity convention (shared with Phase 19): ``|w_j| <= 1e-12`` counts as
pruned (``L1/L2 JOINT PRUNED`` for multitask column norms).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
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

_SINGLE_TYPES = (ElasticNet, ElasticNetCV)
_MULTITASK_TYPES = (MultiTaskElasticNet, MultiTaskElasticNetCV)
_ELASTIC_TYPES = _SINGLE_TYPES + _MULTITASK_TYPES


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


def _extract_coef_intercept(estimator: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(coef_2d (M, D) float32, intercepts (M,) float32, is_multitask)``."""
    is_multitask = isinstance(estimator, _MULTITASK_TYPES)
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
    return coef_2d, intercepts, bool(is_multitask)


def _resolve_alpha_l1_ratio(estimator: Any) -> tuple[float, float]:
    """Resolve ``(alpha, l1_ratio)`` scalars.

    ``ElasticNetCV`` accepts ``l1_ratio`` as a list/array during fitting and
    stores the winner in ``l1_ratio_`` — always prefer the ``*_`` fitted
    attribute and take the first flat element when it is array-like
    (blueprint edge case #1).
    """
    raw_alpha = getattr(estimator, "alpha_", getattr(estimator, "alpha", 1.0))
    raw_ratio = getattr(estimator, "l1_ratio_", getattr(estimator, "l1_ratio", 0.5))
    try:
        alpha = float(np.asarray(raw_alpha, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError, IndexError):
        try:
            alpha = float(raw_alpha)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            alpha = 1.0
    try:
        l1_ratio = float(np.asarray(raw_ratio, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError, IndexError):
        try:
            l1_ratio = float(raw_ratio)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            l1_ratio = 0.5
    if not np.isfinite(alpha) or alpha < 0.0:
        alpha = 1.0
    if not np.isfinite(l1_ratio):
        l1_ratio = 0.5
    return float(alpha), float(min(1.0, max(0.0, l1_ratio)))


def _resolve_n_iter(estimator: Any) -> tuple[int, int]:
    raw = getattr(estimator, "n_iter_", 0)
    try:
        n_iter = int(np.asarray(raw).max())
    except (TypeError, ValueError):
        try:
            n_iter = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            n_iter = 0
    try:
        max_iter = int(getattr(estimator, "max_iter", 1000))
    except (TypeError, ValueError):
        max_iter = 1000
    return max(0, n_iter), max(1, max_iter)


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


def _penalty_terms(w: np.ndarray, alpha: float, l1_ratio: float) -> dict[str, Any]:
    """Effective L1/L2 penalty decomposition for one task row."""
    w64 = np.asarray(w, dtype=np.float64)
    lambda_1 = float(alpha * l1_ratio)
    lambda_2 = float(0.5 * alpha * (1.0 - l1_ratio))
    l1_term = float(np.sum(np.abs(w64)))
    l2_term = float(np.sum(w64**2))
    l1_burden = float(lambda_1 * l1_term)
    l2_burden = float(lambda_2 * l2_term)
    shares = (alpha * l1_ratio * np.abs(w64) + 0.5 * alpha * (1.0 - l1_ratio) * w64**2)
    shares = np.where(np.isfinite(shares), shares, 0.0)
    total = float(l1_burden + l2_burden)
    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "l1_term": l1_term if np.isfinite(l1_term) else 0.0,
        "l2_term": l2_term if np.isfinite(l2_term) else 0.0,
        "l1_penalty_burden": l1_burden if np.isfinite(l1_burden) else 0.0,
        "l2_penalty_burden": l2_burden if np.isfinite(l2_burden) else 0.0,
        "total_regularization": total if np.isfinite(total) else 0.0,
        "penalty_shares": [float(v) for v in shares],
    }


def _transform_for_contrib(model_obj: Any, X_float32: np.ndarray) -> np.ndarray:
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32, dtype=np.float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _resolve_task_idx(n_tasks: int, target_idx: int, target_class_idx: int | None) -> int:
    if target_class_idx is not None and 0 <= int(target_class_idx) < n_tasks:
        return int(target_class_idx)
    return int(min(max(int(target_idx), 0), max(n_tasks - 1, 0)))


@ModelRegistry.register("elastic_net")
class ElasticNetAdapter(BaseModelAdapter):
    display_name = "ElasticNet (L1/L2 Mix)"
    family_name = "Linear Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _ELASTIC_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        coef_2d, _, _ = _extract_coef_intercept(estimator)
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
        preds = np.asarray(model_obj.predict(X_in))
        if preds.ndim == 1:
            flat = preds.ravel()
        else:
            flat = np.asarray(preds[:, 0]).ravel()
        return PredictionBatchResult(
            predictions=[float(v) if np.isfinite(v) else 0.0 for v in np.asarray(flat).ravel()]
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
        coef_2d, intercepts, is_multitask = _extract_coef_intercept(estimator)
        n_tasks = int(coef_2d.shape[0])
        t_idx = _resolve_task_idx(n_tasks, target_idx, target_class_idx)
        w = np.asarray(coef_2d[t_idx], dtype=np.float32)
        b = float(intercepts[t_idx]) if t_idx < intercepts.size else 0.0
        alpha, l1_ratio = _resolve_alpha_l1_ratio(estimator)
        n_iter, max_iter = _resolve_n_iter(estimator)
        stats = _sparsity_stats(w)
        penalty = _penalty_terms(w, alpha, l1_ratio)
        # Multitask joint-prune view: column L2 norms across all tasks.
        col_norms = np.linalg.norm(np.asarray(coef_2d, dtype=np.float64), axis=0)
        joint_zeroed = np.where(col_norms <= ZERO_TOL)[0].tolist()
        positive_only = bool(getattr(estimator, "positive", False))
        convergence_warning = bool(n_iter >= max_iter)
        X_trans = _transform_for_contrib(model_obj, X_float32)
        X_native = _as_dense_float32(X_float32)
        try:
            native_preds = np.asarray(model_obj.predict(X_native))
        except Exception:
            native_preds = np.zeros((X_native.shape[0],), dtype=np.float64)
        if native_preds.ndim == 2:
            native_col = np.asarray(native_preds[:, min(t_idx, native_preds.shape[1] - 1)]).ravel()
        else:
            native_col = np.asarray(native_preds).ravel()

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
                final_pred = float(native_col[row_id])
            except (IndexError, TypeError, ValueError):
                final_pred = float(np.sum(contribs, dtype=np.float64) + b)
            if not np.isfinite(final_pred):
                final_pred = 0.0
            extra: dict[str, Any] = {
                "weights": [float(v) for v in w],
                "intercept": b,
                "target_idx": int(t_idx),
                "target_class_idx": int(t_idx),
                "active_task_idx": int(t_idx),
                "n_tasks": int(n_tasks),
                "is_multitask": bool(is_multitask),
                "is_classifier": False,
                "is_pseudo_proba": False,
                "alpha": float(alpha),
                "l1_ratio": float(l1_ratio),
                "positive_only": bool(positive_only),
                "n_iter": int(n_iter),
                "max_iter": int(max_iter),
                "convergence_warning": bool(convergence_warning),
                "joint_zeroed_indices": [int(i) for i in joint_zeroed],
                "penalty_type": "ElasticNet (L1+L2)",
                **stats,
                **penalty,
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
        coef_2d, intercepts, is_multitask = _extract_coef_intercept(estimator)
        n_tasks = int(coef_2d.shape[0])
        t_idx = _resolve_task_idx(n_tasks, target_idx, target_class_idx)
        w = np.asarray(coef_2d[t_idx], dtype=np.float64)
        b = float(intercepts[t_idx]) if t_idx < intercepts.size else 0.0
        alpha, l1_ratio = _resolve_alpha_l1_ratio(estimator)
        n_iter, max_iter = _resolve_n_iter(estimator)
        penalty = _penalty_terms(w, alpha, l1_ratio)
        stats = _sparsity_stats(w)
        positive_only = bool(getattr(estimator, "positive", False))

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
                "alpha": float(alpha),
                "l1_ratio": float(l1_ratio),
                "penalty_type": "ElasticNet (L1+L2)",
                "target_idx": int(t_idx),
                "active_task_idx": int(t_idx),
                "resolved_target_idx": int(t_idx),
                "n_tasks": int(n_tasks),
                "is_multitask": bool(is_multitask),
                "positive_only": bool(positive_only),
                "n_iter": int(n_iter),
                "max_iter": int(max_iter),
                "convergence_warning": bool(n_iter >= max_iter),
                "boundary_threshold": None,
                "gpu_formula_valid": not bool(is_pipeline),
                "is_classifier": False,
                **stats,
                **penalty,
            },
        )
