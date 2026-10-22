# backend/app/domains/model_engine/adapters/boosting/sklearn_gbm.py
"""Phase 31 — Scikit-learn Gradient Boosting (GBM) staged residual adapters.

Covers ``GradientBoostingClassifier`` (``log_loss`` / ``exponential``) and
``GradientBoostingRegressor`` (``squared_error`` / ``absolute_error`` /
``huber`` / ``quantile``), plus ``Pipeline`` / ``GridSearchCV``-style
``best_estimator_`` / ``TransformedTargetRegressor`` wrappers — inference
runs through the full container while staged tracing unwraps to the
fitted booster.

Blueprint mapping (phase-31-model-sklearn-gbm.md):

- Additive functional gradient descent
  ``F_m(x) = F_0(x) + Σ_{t≤m} η·h_t(x)`` where every
  ``model.estimators_[m, c]`` is a ``DecisionTreeRegressor`` on
  pseudo-residuals — even inside the classifier (blueprint edge #1).
  Leaf values are read directly (``tree_.value``), never re-predicted
  per tree (blueprint §6.1: ``model.apply`` + value lookup).
- Base value ``F_0`` comes from the booster-native
  ``model._raw_predict_init`` so ``init='zero'`` / ``Dummy`` / custom
  estimators resolve correctly (edge #4).
- Per-stage raw jumps ``Δ_m = η·leaf_m(x)`` are attributed per row by
  equal split over the row's ``decision_path`` internal features of
  ``estimators_[m, c]``; stages past ``MAX_TRACED_STAGES`` fall back to
  a uniform split so ``base + ΣΦ == F_M`` stays exact for ``M > 100``
  models. Binary classifiers resolve the single tree column to the
  positive class; a ``target 0`` view negates ``base``/``contribs``.
- Binary vs multiclass shapes (edge #2): ``estimators_`` is ``(M, 1)``
  for binary/regression and ``(M, C)`` for multiclass; ``apply`` gains
  a trailing axis accordingly. The real stage count always comes from
  ``estimators_.shape[0]`` (edge #3, early stopping).
- Trajectory cap: ``sampled_stage_indices`` via ``linspace`` to at most
  ``MAX_TRAJECTORY_STAGES`` points; top-8 residual stages per row are
  exported for the 3D spiral ladder.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
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

MAX_TRAJECTORY_STAGES = 50
MAX_TRACED_STAGES = 100
MAX_SURFACE_RESOLUTION = 100
TOP_STAGES = 8

_CLASSIFIER_TYPES = (GradientBoostingClassifier,)
_REGRESSOR_TYPES = (GradientBoostingRegressor,)


def _unwrap_gbm_container(model_obj: Any) -> tuple[Any, Any]:
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


def _final_gbm_estimator(model_obj: Any) -> Any:
    return _unwrap_gbm_container(model_obj)[1]


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
    """Apply pipeline preprocessing (all steps but last) so splits align."""
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


def _raw_base_scores(
    booster: Any, Xt_one: np.ndarray, n_cols: int
) -> np.ndarray:
    """Initial scores ``F_0`` per tree column ``(n_cols,)`` (blueprint edge #4)."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            raw = np.asarray(booster._raw_predict_init(Xt_one), dtype=np.float64)
    except (AttributeError, ValueError, TypeError, IndexError):
        return np.zeros((max(n_cols, 1),), dtype=np.float64)
    flat = np.asarray(raw, dtype=np.float64).reshape(-1)
    if flat.size >= n_cols:
        return flat[:n_cols].copy()
    out = np.zeros((max(n_cols, 1),), dtype=np.float64)
    out[: flat.size] = flat
    return out


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


def _staged_proba_matrix(
    predict_obj: Any,
    booster: Any,
    X_raw_f32: np.ndarray,
    resolved: int,
    is_classifier: bool,
    negate_binary: bool,
) -> np.ndarray:
    """Staged predicted values ``(n_samples, M_actual)`` for ``resolved``.

    Classifier: ``staged_predict_proba`` column (handles every ``loss_``
    link function, blueprint edge #6); regressor: ``staged_predict``.
    """
    cols: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        try:
            if is_classifier:
                staged = booster.staged_predict_proba(
                    _transform_for_estimators(predict_obj, X_raw_f32)
                )
                for proba in staged:
                    p = np.asarray(proba, dtype=np.float64)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    idx = int(min(max(resolved, 0), p.shape[1] - 1)) if p.size else 0
                    cols.append(p[:, idx] if p.size else np.zeros((X_raw_f32.shape[0],)))
            else:
                staged = booster.staged_predict(
                    _transform_for_estimators(predict_obj, X_raw_f32)
                )
                for pred in staged:
                    cols.append(np.asarray(pred, dtype=np.float64).ravel())
        except (AttributeError, ValueError, TypeError):
            cols = []
    if not cols:
        return np.zeros((X_raw_f32.shape[0], 0), dtype=np.float64)
    mat = np.column_stack(cols)
    if is_classifier and negate_binary:
        mat = 1.0 - mat
    if not is_classifier and _is_ttr(predict_obj, booster):
        mat = _inverse_ttr(predict_obj, mat)
    return np.where(np.isfinite(mat), mat, 0.0)


class _SklearnGBMBase(BaseModelAdapter):
    is_classifier: bool = True

    def _resolve_target(
        self, booster: Any, n_cols: int, target_idx: int, target_class_idx: int | None
    ) -> tuple[int, int, bool, list[Any]]:
        """Return ``(resolved_class, tree_col, negate_binary, classes)``.

        Binary GBM (``n_cols == 1``) always grows the positive-class
        residual tree in column 0 (edge #2); a ``target 0`` view negates
        the raw margin so the waterfall still sums exactly.
        """
        classes = _classes_of(booster)
        n_classes = len(classes)
        if self.is_classifier and n_cols == 1:
            # Binary GBM grows the positive-class residual tree in column 0
            # (edge #2). Like AdaBoost, an unset selector explains the
            # positive class; only an explicit target_class_idx == 0 flips
            # to the negated class-0 margin view.
            if target_class_idx is not None:
                negate = bool(int(target_class_idx) == 0 and n_classes == 2)
            else:
                negate = False
            return 1 if n_classes == 2 else 0, 0, negate, classes
        if self.is_classifier:
            if target_class_idx is not None and n_classes:
                resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            elif n_classes == 2:
                resolved = 1
            else:
                resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
            tree_col = int(min(max(resolved, 0), max(n_cols - 1, 0)))
            return resolved, tree_col, False, classes
        return 0, 0, False, classes

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        est = _final_gbm_estimator(estimator)
        width = max(int(n_features), 0)
        try:
            weights = np.asarray(getattr(est, "feature_importances_"), dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            return [None for _ in range(width)]
        if weights.size != width:
            return [None for _ in range(width)]
        return [float(v) if np.isfinite(v) else 0.0 for v in weights.tolist()]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, _ = _unwrap_gbm_container(model_obj)
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
        predict_obj, booster = _unwrap_gbm_container(model_obj)
        estimators = getattr(booster, "estimators_", None)
        if estimators is None or np.asarray(estimators).shape[0] == 0:
            raise ValueError(f"Estimator '{type(booster).__name__}' exposes no fitted estimators_")
        n_stages = int(np.asarray(estimators).shape[0])
        n_cols = int(np.asarray(estimators).shape[1]) if np.asarray(estimators).ndim > 1 else 1
        resolved, tree_col, negate, classes = self._resolve_target(
            booster, n_cols, target_idx, target_class_idx
        )
        n_classes = len(classes)
        is_binary = bool(self.is_classifier and n_classes == 2)

        try:
            learning_rate = float(getattr(booster, "learning_rate", 0.1))
        except (TypeError, ValueError):
            learning_rate = 0.1
        if not np.isfinite(learning_rate):
            learning_rate = 0.1

        X_native = _as_dense_float32(X_float32)
        Xt = _transform_for_estimators(predict_obj, X_native)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        ttr_reg = bool(not self.is_classifier and _is_ttr(predict_obj, booster))
        base_vec = _raw_base_scores(booster, Xt[:1], n_cols)
        base_raw = float(base_vec[min(tree_col, base_vec.size - 1)]) if base_vec.size else 0.0
        if negate:
            base_raw = -base_raw

        # Leaf index tensor (n, M, C) — float in some sklearn builds (edge).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                leaf_tensor = np.asarray(booster.apply(Xt))
            except (ValueError, AttributeError, TypeError):
                raise ValueError(f"Estimator '{type(booster).__name__}' failed apply() on input")
        if leaf_tensor.ndim == 2:
            leaf_matrix = np.asarray(leaf_tensor, dtype=np.int64)
        elif leaf_tensor.ndim == 3:
            col = int(min(tree_col, leaf_tensor.shape[2] - 1))
            leaf_matrix = np.asarray(leaf_tensor[:, :, col], dtype=np.int64)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unexpected apply() shape {leaf_tensor.shape}")
        if leaf_matrix.shape[1] != n_stages:  # early-stopping guard (edge #3)
            n_stages = int(leaf_matrix.shape[1])

        # Stage raw jumps Δ_m = ±η·leaf (vectorized, blueprint §6.1).
        stage_deltas = np.zeros((X_native.shape[0], n_stages), dtype=np.float64)
        traced = min(n_stages, MAX_TRACED_STAGES)
        path_splits: list[list[list[int]] | None] = [None] * n_stages
        csr_cache: list[Any] = [None] * n_stages
        feat_cache: list[Any] = [None] * n_stages
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            for m in range(n_stages):
                try:
                    reg_tree = estimators[m, tree_col] if np.asarray(estimators).ndim > 1 else estimators[m]
                except (IndexError, TypeError):
                    continue
                try:
                    node_vals = np.asarray(reg_tree.tree_.value[:, 0, 0], dtype=np.float64)
                except AttributeError:
                    continue
                leaves_m = np.clip(leaf_matrix[:, m], 0, node_vals.size - 1)
                dcol = learning_rate * node_vals[leaves_m]
                stage_deltas[:, m] = -dcol if negate else dcol
                if m < traced:
                    try:
                        csr = reg_tree.decision_path(Xt)
                        indptr = np.asarray(csr.indptr, dtype=np.int64).ravel()
                        indices = np.asarray(csr.indices, dtype=np.int64).ravel()
                        feats = np.asarray(reg_tree.tree_.feature, dtype=np.int64).ravel()
                        csr_cache[m] = (indptr, indices)
                        feat_cache[m] = feats
                    except (ValueError, AttributeError, TypeError):
                        csr_cache[m] = None

        staged = _staged_proba_matrix(
            predict_obj, booster, X_native, resolved, self.is_classifier, negate
        )
        if staged.shape[1] != n_stages and staged.size:
            n_stages = int(staged.shape[1])
            stage_deltas = stage_deltas[:, :n_stages]
        if ttr_reg and staged.size:
            # Transformed-target space is nonlinear: re-derive base/deltas
            # from the inverse-transformed staged curve (AdaBoost §2.1
            # analogue) so base + ΣΦ == predict() in original units.
            with np.errstate(invalid="ignore"):
                ttr_base = float(np.mean(staged[:, 0]))
            base_raw = ttr_base if np.isfinite(ttr_base) else 0.0
            prev = np.full((staged.shape[0], 1), base_raw, dtype=np.float64)
            stage_deltas = np.diff(np.hstack([prev, staged]), axis=1)

        # Attribute Δ_m per row: equal split over decision_path internal
        # features (exact telescope to η·leaf); uniform fallback past the
        # traced cap or on path failure keeps base + ΣΦ == F_M exact.
        n_rows = X_native.shape[0]
        width = max(n_features, 0)
        sum_c = np.zeros((n_rows, width), dtype=np.float64)
        x_width = int(Xt.shape[1]) if Xt.ndim == 2 else width
        for m in range(n_stages):
            dcol = stage_deltas[:, m]
            if np.all(dcol == 0.0):
                continue
            cached = csr_cache[m] if m < traced else None
            if cached is not None:
                indptr, indices = cached
                feats = feat_cache[m]
                for i in range(n_rows):
                    if i >= indptr.size - 1:
                        if width:
                            sum_c[i, :] += float(dcol[i]) / width
                        continue
                    path = [int(v) for v in indices[indptr[i] : indptr[i + 1]].tolist()]
                    if x_width == width:
                        internal = [int(feats[u]) for u in path[:-1] if 0 <= int(feats[u]) < width]
                    else:
                        internal = []
                    if internal:
                        share = float(dcol[i]) / len(internal)
                        for f_idx in internal:
                            sum_c[i, f_idx] += share
                    elif width:
                        sum_c[i, :] += float(dcol[i]) / width
            elif width:
                sum_c[:, :] += (dcol.reshape(-1, 1) / width) * np.ones((1, width))

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

        k_traj = min(n_stages, MAX_TRAJECTORY_STAGES)
        step_indices = (
            np.unique(np.linspace(0, n_stages - 1, k_traj, dtype=int)).tolist() if n_stages else []
        )
        k_top = min(TOP_STAGES, n_stages)
        k_export = min(n_stages, MAX_TRACED_STAGES)

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
            row_deltas = stage_deltas[row_id] if n_stages else np.zeros((0,))
            top_idx = np.argsort(np.abs(row_deltas))[::-1][:k_top] if n_stages else []
            top_sorted = sorted(int(v) for v in np.asarray(top_idx).tolist())
            cum_raw = float(base_raw + np.sum(row_deltas[:0]))
            stages: list[dict[str, Any]] = []
            running = float(base_raw)
            cum_by_stage: list[float] = []
            for mm in range(n_stages):
                running += float(row_deltas[mm])
                cum_by_stage.append(running)
            for mm in top_sorted:
                leaf_id = int(leaf_matrix[row_id, mm]) if mm < leaf_matrix.shape[1] else -1
                stages.append(
                    {
                        "stage": int(mm),
                        "leaf_index": int(leaf_id),
                        "delta_raw": round(float(row_deltas[mm]), 5),
                        "cumulative_raw": round(float(cum_by_stage[mm]), 5),
                        "cumulative_pred": round(float(staged[row_id, mm]), 5)
                        if staged.size and mm < staged.shape[1]
                        else None,
                    }
                )
            traj = [round(float(staged[row_id, s]), 4) for s in step_indices] if staged.size else []
            final_raw = float(base_raw + float(np.sum(row_deltas)))
            _ = cum_raw
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(is_binary),
                "algorithm": type(booster).__name__,
                "loss": str(getattr(booster, "loss_", getattr(booster, "loss", ""))),
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "sklearn_gbm",
                "learning_rate": float(learning_rate),
                "n_stages_total": int(n_stages),
                "n_stages_traced": int(traced),
                "negated_target_zero_view": bool(negate),
                "base_raw_score": float(base_raw),
                "final_raw_score": float(final_raw),
                "sampled_stage_indices": [int(s) for s in step_indices],
                "staged_predictions": traj,
                "stage_raw_deltas_sample": [
                    round(float(v), 5) for v in row_deltas[:k_export].tolist()
                ],
                "leaf_indices_per_stage": [
                    int(v) for v in leaf_matrix[row_id, :k_export].tolist()
                ]
                if n_stages
                else [],
                "top_residual_stages": stages,
                "probabilities": leaf_proba,
                "predicted_class": final_pred,
                "winning_class_idx": int(resolved) if self.is_classifier else 0,
                "resolved_class_idx": int(resolved),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base_raw),
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
        predict_obj, booster = _unwrap_gbm_container(model_obj)
        estimators = getattr(booster, "estimators_", None)
        n_stages = int(np.asarray(estimators).shape[0]) if estimators is not None else 0
        n_cols = (
            int(np.asarray(estimators).shape[1])
            if estimators is not None and np.asarray(estimators).ndim > 1
            else 1
        )
        resolved, _tree_col, negate, classes = self._resolve_target(
            booster, n_cols, target_idx, target_class_idx
        )
        n_classes = len(classes)
        try:
            learning_rate = float(getattr(booster, "learning_rate", 0.1))
        except (TypeError, ValueError):
            learning_rate = 0.1
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
                    if negate:
                        z_surface = 1.0 - z_surface
                else:
                    preds = np.asarray(predict_obj.predict(grid), dtype=np.float64).ravel()
                    z_surface = preds.reshape(res, res) if preds.size == n_points else np.zeros((res, res))
            except (ValueError, AttributeError, TypeError):
                z_surface = np.zeros((res, res), dtype=np.float64)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        try:
            imp = np.asarray(getattr(booster, "feature_importances_"), dtype=np.float64).ravel()
            importances = [round(float(v), 5) for v in imp.tolist()] if imp.size else []
        except (TypeError, ValueError):
            importances = []
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
                "loss": str(getattr(booster, "loss_", getattr(booster, "loss", ""))),
                "model_family": "sklearn_gbm",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.1,
                "n_stages_total": int(n_stages),
                "n_estimators_actual": int(n_stages),
                "feature_importances": importances,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("sklearn_gbm")
class SklearnGBMClassifierAdapter(_SklearnGBMBase):
    display_name = "Gradient Boosting (Sklearn GBM)"
    family_name = "Gradient Boosting"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_gbm_estimator(estimator), _CLASSIFIER_TYPES)

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


@ModelRegistry.register("sklearn_gbm_regressor")
class SklearnGBMRegressorAdapter(_SklearnGBMBase):
    display_name = "Gradient Boosting Regressor (Sklearn GBM)"
    family_name = "Gradient Boosting"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_gbm_estimator(estimator), _REGRESSOR_TYPES)

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
