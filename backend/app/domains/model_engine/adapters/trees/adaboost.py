# backend/app/domains/model_engine/adapters/trees/adaboost.py
"""Phase 30 — AdaBoost sequential weak-learner cascade adapters (3D cascade ladder).

Covers ``AdaBoostClassifier`` (``SAMME``/``SAMME.R``) and
``AdaBoostRegressor`` (``AdaBoost.R2``) with Decision-Stump, deep-tree, or
non-tree base estimators, plus ``Pipeline`` / ``GridSearchCV``-style
``best_estimator_`` / ``TransformedTargetRegressor`` wrappers — inference
runs through the full container while staged tracing unwraps to the
fitted booster.

Blueprint mapping (phase-30-model-adaboost-cascade.md):

- Staged accumulation via the native generators
  ``staged_predict_proba`` / ``staged_predict`` (``O(M)``, never
  ``O(M^2)`` re-prediction): ``S_m(x)`` is the resolved-class probability
  (classifier) or regressed value (regressor) after the first ``m``
  learners; ``S_0`` is the base value (``0.5`` binary, ``1/C``
  multiclass, ``mean(S_1)`` regression).
- Marginal deltas ``ΔS_m = S_m - S_{m-1}`` attributed per stage:
  stump (``max_depth == 1``) fast path — vectorized full credit to
  ``tree_.feature[0]``; deep trees — equal split of ``ΔS_m`` over the
  row's ``decision_path`` internal features; non-tree estimators with
  ``coef_`` — ``|coef * x|`` proportional share (uniform fallback keeps
  ``base + ΣΦ == S_M`` exact).
- Trajectory cap: ``sampled_stage_indices`` via ``linspace`` to at most
  ``MAX_TRAJECTORY_POINTS = 50``; top-8 ``|ΔS_m|`` learners exported per
  row with ``(weight, error, feature, threshold, delta, cumulative)``.
- 3D surface: plaid terraces (mean surface) plus ``stump_splits`` overlay
  (up to 50 stump cut planes) for the curtain renderer.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import BaseDecisionTree

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

MAX_TRAJECTORY_POINTS = 50
MAX_STUMP_SPLITS = 50
MAX_SURFACE_RESOLUTION = 100
TOP_LEARNERS = 8

_CLASSIFIER_TYPES = (AdaBoostClassifier,)
_REGRESSOR_TYPES = (AdaBoostRegressor,)


def _unwrap_ab_container(model_obj: Any) -> tuple[Any, Any]:
    """Return ``(predict_obj, booster_estimator)``.

    Follows ``GridSearchCV``-style ``best_estimator_`` chains, unwraps
    ``Pipeline`` ``steps``, and follows ``TransformedTargetRegressor``
    ``regressor_``.
    """
    predict_obj = model_obj
    seen = 0
    while seen < 8:
        seen += 1
        best = getattr(model_obj, "best_estimator_", None)
        if best is not None:
            model_obj = best
            predict_obj = best
            continue
        steps = getattr(model_obj, "steps", None)
        if isinstance(steps, list) and steps:
            last = steps[-1]
            model_obj = last[1] if isinstance(last, (tuple, list)) and len(last) == 2 else last
            continue
        reg = getattr(model_obj, "regressor_", None)
        if reg is not None:
            model_obj = reg
            continue
        break
    return predict_obj, model_obj


def _final_ab_estimator(model_obj: Any) -> Any:
    return _unwrap_ab_container(model_obj)[1]


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


def _transform_for_estimators(predict_obj: Any, X_f32: np.ndarray) -> np.ndarray:
    """Apply pipeline preprocessing (all steps but last) so stump splits align."""
    steps = getattr(predict_obj, "steps", None)
    if not (isinstance(steps, list) and steps):
        return X_f32
    try:
        Xt: Any = X_f32
        for _, step_est in steps[:-1]:
            if hasattr(step_est, "transform"):
                Xt = step_est.transform(Xt)
            else:
                return X_f32
        return _as_dense_float32(Xt)
    except (ValueError, AttributeError, TypeError):
        return X_f32


def _classes_of(estimator: Any) -> list[Any]:
    raw = getattr(estimator, "classes_", None)
    if raw is None:
        return [0, 1]
    return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]


def _is_ttr(predict_obj: Any, booster: Any) -> bool:
    return predict_obj is not booster and hasattr(predict_obj, "transformer_")


def _inverse_ttr(predict_obj: Any, values: np.ndarray) -> np.ndarray:
    """Map staged regressor outputs back through a TTR transformer (best-effort)."""
    try:
        transformer = getattr(predict_obj, "transformer_", None)
        if transformer is None or values.size == 0:
            return values
        flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        back = np.asarray(transformer.inverse_transform(flat), dtype=np.float64).ravel()
        return back.reshape(values.shape)
    except (ValueError, AttributeError, TypeError):
        return values


def _staged_matrix(
    predict_obj: Any,
    booster: Any,
    X_raw_f32: np.ndarray,
    resolved: int,
    is_classifier: bool,
) -> np.ndarray:
    """Staged cumulative predictions ``(n_samples, M_actual)`` (blueprint 2.1)."""
    Xt = _transform_for_estimators(predict_obj, X_raw_f32)
    cols: list[np.ndarray] = []
    m_actual = len(getattr(booster, "estimators_", []) or [])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        try:
            if is_classifier:
                try:
                    staged = booster.staged_predict_proba(Xt)
                except (AttributeError, ValueError, TypeError):
                    staged = iter(())
                for proba in staged:
                    p = np.asarray(proba, dtype=np.float64)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    idx = int(min(max(resolved, 0), p.shape[1] - 1)) if p.size else 0
                    cols.append(p[:, idx] if p.size else np.zeros((Xt.shape[0],)))
            else:
                try:
                    staged = booster.staged_predict(Xt)
                except (AttributeError, ValueError, TypeError):
                    staged = iter(())
                for pred in staged:
                    cols.append(np.asarray(pred, dtype=np.float64).ravel())
        except (ValueError, TypeError, AttributeError, ZeroDivisionError, ArithmeticError):
            # Edge: single-class booster (n_classes_ == 1) makes sklearn's own
            # staged generator raise (ZeroDivisionError). Fall back to a flat
            # cascade at the final prediction so parity still holds.
            cols = []
    if not cols and m_actual:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                if is_classifier:
                    p = np.asarray(predict_obj.predict_proba(X_raw_f32), dtype=np.float64)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    idx = int(min(max(resolved, 0), p.shape[1] - 1)) if p.size else 0
                    final_col = p[:, idx] if p.size else np.zeros((X_raw_f32.shape[0],))
                else:
                    final_col = np.asarray(predict_obj.predict(X_raw_f32), dtype=np.float64).ravel()
                    if _is_ttr(predict_obj, booster):
                        final_col = _inverse_ttr(predict_obj, final_col.reshape(-1, 1)).ravel()
            except (ValueError, AttributeError, TypeError):
                final_col = np.zeros((X_raw_f32.shape[0],))
        cols = [np.asarray(final_col, dtype=np.float64) for _ in range(m_actual)]
    if not cols:
        return np.zeros((X_raw_f32.shape[0], 0), dtype=np.float64)
    mat = np.column_stack(cols)
    if not is_classifier and _is_ttr(predict_obj, booster):
        mat = _inverse_ttr(predict_obj, mat)
    return np.where(np.isfinite(mat), mat, 0.0)


def _coef_share_matrix(est: Any, X: np.ndarray, resolved: int) -> np.ndarray | None:
    """Row-normalized ``|coef * x|`` shares ``(n, f)`` or ``None`` when unavailable."""
    try:
        coef = np.asarray(getattr(est, "coef_"), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None
    if coef.ndim == 2:
        row = int(min(max(resolved, 0), coef.shape[0] - 1))
        coef = coef[row]
    coef = np.asarray(coef, dtype=np.float64).ravel()
    if coef.size != X.shape[1] or not np.any(np.isfinite(coef)):
        return None
    with np.errstate(invalid="ignore"):
        raw = np.abs(X.astype(np.float64) * coef.reshape(1, -1))
    raw = np.where(np.isfinite(raw), raw, 0.0)
    totals = raw.sum(axis=1, keepdims=True)
    uniform = np.full_like(raw, 1.0 / max(X.shape[1], 1))
    return np.where(totals > 0, raw / np.maximum(totals, 1e-12), uniform)


class _AdaBoostBase(BaseModelAdapter):
    is_classifier: bool = True

    def _resolve_target(
        self, booster: Any, target_idx: int, target_class_idx: int | None
    ) -> tuple[int, list[Any]]:
        classes = _classes_of(booster)
        n_classes = len(classes)
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        elif self.is_classifier and n_classes == 2:
            resolved = 1
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        return resolved, classes

    def _base_value(
        self, staged: np.ndarray, n_classes: int, resolved: int
    ) -> float:
        if staged.size == 0:
            return 0.0
        if self.is_classifier:
            if n_classes == 2:
                return 0.5
            return 1.0 / max(n_classes, 1)
        with np.errstate(invalid="ignore"):
            mean = float(np.mean(staged[:, 0]))
        return mean if np.isfinite(mean) else 0.0

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        est = _final_ab_estimator(estimator)
        width = max(int(n_features), 0)
        try:
            weights = np.asarray(getattr(est, "feature_importances_"), dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            return [None for _ in range(width)]
        if weights.size != width:
            return [None for _ in range(width)]
        return [float(v) if np.isfinite(v) else 0.0 for v in weights.tolist()]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, _ = _unwrap_ab_container(model_obj)
        X_in = _as_dense_float32(X_float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                preds = np.asarray(predict_obj.predict(X_in)).ravel()
            except (ValueError, AttributeError, TypeError):
                preds = np.zeros((X_in.shape[0],))
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if self.is_classifier:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    probas = np.asarray(predict_obj.predict_proba(X_in), dtype=np.float64)
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
                probabilities = [
                    [float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()
                ]
                confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                probabilities, confidences = None, None
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

    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        predict_obj, booster = _unwrap_ab_container(model_obj)
        estimators: list[Any] = list(getattr(booster, "estimators_", []) or [])
        if not estimators:
            raise ValueError(f"Estimator '{type(booster).__name__}' exposes no fitted estimators_")
        resolved, classes = self._resolve_target(booster, target_idx, target_class_idx)
        n_classes = len(classes)
        is_binary = bool(self.is_classifier and n_classes == 2)

        X_native = _as_dense_float32(X_float32)
        Xt = _transform_for_estimators(predict_obj, X_native)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        staged = _staged_matrix(predict_obj, booster, X_native, resolved, self.is_classifier)
        m_actual = staged.shape[1]
        if m_actual == 0:
            raise ValueError(f"Estimator '{type(booster).__name__}' produced no staged predictions")
        base_val = self._base_value(staged, n_classes, resolved)
        prev = np.full((staged.shape[0], 1), base_val, dtype=np.float64)
        deltas = np.diff(np.hstack([prev, staged]), axis=1)

        try:
            weights = np.asarray(getattr(booster, "estimator_weights_"), dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            weights = np.zeros((m_actual,), dtype=np.float64)
        try:
            errors = np.asarray(getattr(booster, "estimator_errors_"), dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            errors = np.zeros((m_actual,), dtype=np.float64)

        # Per-stage attribution into (M, n, f) streaming sums (no big 3-D array).
        n_rows = X_native.shape[0]
        width = max(n_features, 0)
        sum_c = np.zeros((n_rows, width), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            for m, est in enumerate(estimators):
                if m >= m_actual:
                    break
                dcol = deltas[:, m]
                if isinstance(est, BaseDecisionTree):
                    try:
                        tree = est.tree_
                    except AttributeError:
                        tree = None
                    if tree is not None:
                        try:
                            max_depth = int(tree.max_depth)
                            node_count = int(tree.node_count)
                            f0 = int(np.asarray(tree.feature).ravel()[0])
                            t0 = float(np.asarray(tree.threshold).ravel()[0])
                        except (AttributeError, TypeError, IndexError, ValueError):
                            max_depth, node_count, f0, t0 = 0, 0, -2, 0.0
                        _ = t0
                        if node_count > 1 and max_depth == 1 and 0 <= f0 < width:
                            # Stump fast path: vectorized full credit (blueprint 6.3).
                            sum_c[:, f0] += dcol
                            continue
                        if node_count > 1:
                            try:
                                csr = est.decision_path(Xt)
                                indptr = np.asarray(csr.indptr, dtype=np.int64).ravel()
                                indices = np.asarray(csr.indices, dtype=np.int64).ravel()
                                feats = np.asarray(tree.feature, dtype=np.int64).ravel()
                            except (ValueError, AttributeError, TypeError):
                                csr = None  # type: ignore[assignment]
                            if csr is not None:
                                for i in range(n_rows):
                                    if not (i < indptr.size - 1):
                                        continue
                                    path = [int(v) for v in indices[indptr[i] : indptr[i + 1]].tolist()]
                                    internal = [int(feats[u]) for u in path[:-1] if 0 <= int(feats[u]) < width]
                                    if internal:
                                        share = float(dcol[i]) / len(internal)
                                        for f_idx in internal:
                                            sum_c[i, f_idx] += share
                                    elif width:
                                        sum_c[i, :] += float(dcol[i]) / width
                                continue
                shares = _coef_share_matrix(est, Xt, resolved)
                if shares is not None:
                    sum_c += shares * dcol.reshape(-1, 1)
                elif width:
                    # Uniform fallback keeps base + ΣΦ == S_M exact (edge #2).
                    sum_c += (dcol.reshape(-1, 1) / width) * np.ones((1, width))

        try:
            native_preds = list(np.asarray(predict_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError, TypeError):
            native_preds = [classes[0] if classes else 0 for _ in row_indices]
        native_probas: np.ndarray | None = None
        if self.is_classifier:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    native_probas = np.asarray(predict_obj.predict_proba(X_native), dtype=np.float64)
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                native_probas = None

        k_traj = min(m_actual, MAX_TRAJECTORY_POINTS)
        step_indices = np.unique(
            np.linspace(0, m_actual - 1, k_traj, dtype=int)
        ).tolist()
        k_top = min(TOP_LEARNERS, m_actual)

        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < n_rows):
                continue
            contribs = sum_c[row_id]
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if resolved < len(classes) else resolved
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            leaf_proba = (
                [float(v) for v in native_probas[row_id].tolist()]
                if native_probas is not None and row_id < native_probas.shape[0]
                else []
            )
            row_deltas = deltas[row_id]
            top_idx = np.argsort(np.abs(row_deltas))[::-1][:k_top]
            top_sorted = sorted(int(v) for v in top_idx.tolist())
            learners: list[dict[str, Any]] = []
            for mm in top_sorted:
                est = estimators[mm] if mm < len(estimators) else None
                f_idx, th = -1, 0.0
                if isinstance(est, BaseDecisionTree):
                    try:
                        feats = np.asarray(est.tree_.feature).ravel()
                        ths = np.asarray(est.tree_.threshold).ravel()
                        if int(est.tree_.node_count) > 1:
                            f_idx = int(feats[0])
                            th = round(float(ths[0]), 4)
                    except (AttributeError, TypeError, IndexError, ValueError):
                        f_idx, th = -1, 0.0
                learners.append(
                    {
                        "stage": int(mm),
                        "weight": round(float(weights[mm]) if mm < weights.size else 0.0, 4),
                        "error": round(float(errors[mm]) if mm < errors.size else 0.0, 4),
                        "feature_idx": int(f_idx),
                        "feature_name": names[f_idx] if 0 <= f_idx < len(names) else None,
                        "threshold": th,
                        "delta": round(float(row_deltas[mm]), 5),
                        "cumulative": round(float(staged[row_id, mm]), 5),
                    }
                )
            traj = [round(float(staged[row_id, s]), 4) for s in step_indices]
            crossed: int | None = None
            if is_binary:
                for mm in range(m_actual):
                    if float(staged[row_id, mm]) >= 0.5:
                        crossed = int(mm + 1)  # 1-based stage count
                        break
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(is_binary),
                "algorithm": type(booster).__name__,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "adaboost",
                "n_estimators_actual": int(m_actual),
                "n_estimators_param": getattr(booster, "n_estimators", m_actual),
                "early_terminated": bool(m_actual < int(getattr(booster, "n_estimators", m_actual) or m_actual)),
                "base_value_stage0": float(base_val),
                "sampled_stage_indices": [int(s) for s in step_indices],
                "staged_trajectory": traj,
                "top_influential_learners": learners,
                "crossed_threshold_stage": crossed,
                "learning_rate": getattr(booster, "learning_rate", None),
                "probabilities": leaf_proba,
                "predicted_class": final_pred,
                "winning_class_idx": int(resolved) if self.is_classifier else 0,
                "resolved_class_idx": int(resolved),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base_val),
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions={
                        (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                        for i in range(int(width))
                    },
                    extra_mechanics=extra,
                )
            )
        return traces

    def _surface_common(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        predict_obj, booster = _unwrap_ab_container(model_obj)
        estimators: list[Any] = list(getattr(booster, "estimators_", []) or [])
        resolved, classes = self._resolve_target(booster, target_idx, target_class_idx)
        n_classes = len(classes)
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        n_points = res * res
        grid = np.zeros((n_points, max(int(n_features), 1)), dtype=np.float32)
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
            if 0 <= feat_idx < grid.shape[1] and np.isfinite(med_val):
                grid[:, feat_idx] = np.float32(med_val)
        ax = grid_spec.axis_x_feature_index
        ay = grid_spec.axis_y_feature_index
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
        if 0 <= ax < grid.shape[1]:
            grid[:, ax] = xx.ravel()
        if int(n_features) > 1 and ay != ax and 0 <= ay < grid.shape[1]:
            grid[:, ay] = yy.ravel()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                if self.is_classifier:
                    probas = np.asarray(predict_obj.predict_proba(grid), dtype=np.float64)
                    c_idx = int(min(resolved, probas.shape[1] - 1)) if probas.size else 0
                    z_surface = (
                        np.asarray(probas[:, c_idx], dtype=np.float64).reshape(res, res)
                        if probas.size
                        else np.zeros((res, res))
                    )
                else:
                    preds = np.asarray(predict_obj.predict(grid), dtype=np.float64).ravel()
                    z_surface = preds.reshape(res, res) if preds.size == n_points else np.zeros((res, res))
            except (ValueError, AttributeError, TypeError):
                z_surface = np.zeros((res, res), dtype=np.float64)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)

        try:
            weights = np.asarray(getattr(booster, "estimator_weights_"), dtype=np.float64).ravel()
            weights_list = [round(float(v), 4) for v in weights.tolist()]
        except (AttributeError, TypeError, ValueError):
            weights_list = []
        try:
            errors = np.asarray(getattr(booster, "estimator_errors_"), dtype=np.float64).ravel()
            errors_list = [round(float(v), 4) for v in errors.tolist()]
        except (AttributeError, TypeError, ValueError):
            errors_list = []
        # Stump cut planes for the curtain renderer (tree stumps only, capped).
        splits: list[dict[str, Any]] = []
        for m, est in enumerate(estimators[:MAX_STUMP_SPLITS]):
            if not isinstance(est, BaseDecisionTree):
                continue
            try:
                if int(est.tree_.node_count) <= 1 or int(est.tree_.max_depth) != 1:
                    continue
                f_idx = int(np.asarray(est.tree_.feature).ravel()[0])
                th = float(np.asarray(est.tree_.threshold).ravel()[0])
            except (AttributeError, TypeError, IndexError, ValueError):
                continue
            if f_idx < 0 or not np.isfinite(th):
                continue
            splits.append(
                {
                    "stage": int(m),
                    "feature_idx": int(f_idx),
                    "threshold": round(th, 4),
                    "weight": round(float(weights[m]) if m < weights.size else 0.0, 4),
                    "error": round(float(errors[m]) if m < errors.size else 0.0, 4),
                }
            )
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        try:
            imp = np.asarray(getattr(booster, "feature_importances_"), dtype=np.float64).ravel()
            importances = [round(float(v), 5) for v in imp.tolist()] if imp.size else []
        except (TypeError, ValueError):
            importances = []
        m_actual = len(estimators)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel()],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes],
                "algorithm": type(booster).__name__,
                "model_family": "adaboost",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "n_estimators_actual": int(m_actual),
                "estimator_weights": weights_list,
                "estimator_errors": errors_list,
                "stump_splits": splits,
                "feature_importances": importances,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("adaboost")
class AdaBoostClassifierAdapter(_AdaBoostBase):
    display_name = "AdaBoost (Sequential Cascade)"
    family_name = "Trees & Bagging"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_ab_estimator(estimator), _CLASSIFIER_TYPES)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return self._trace_common(
            model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)


@ModelRegistry.register("adaboost_regressor")
class AdaBoostRegressorAdapter(_AdaBoostBase):
    display_name = "AdaBoost Regressor (Sequential Cascade)"
    family_name = "Trees & Bagging"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_ab_estimator(estimator), _REGRESSOR_TYPES)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return self._trace_common(
            model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)
