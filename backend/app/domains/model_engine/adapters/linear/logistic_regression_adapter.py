# backend/app/domains/model_engine/adapters/linear/logistic_regression_adapter.py
"""Phase 18 — LogisticRegressionAdapter (binary + multiclass OvR/multinomial).

Log-odds waterfall identity (per explained class ``k``)::

    b_k + sum_j (w_{k,j} * x_j) == decision_function(X)[k]

Binary models (``C == 2``) expose ``coef_`` with shape ``(1, D)`` and a 1-D
``decision_function``; the single row always describes the *positive* class
(``classes_[1]``).  Multiclass models (``C > 2``) expose ``coef_`` with shape
``(C, D)`` and a 2-D ``decision_function``; each trace explains either the
caller-requested ``target_class_idx`` or the per-row winning class
(``argmax proba``).

Design notes vs. the Phase-18 blueprint draft:
- The blueprint sketch used ``(model, feature_names)`` constructor args and
  ``InternalFlowBatch`` / ``SurfaceMesh3D`` schemas; the real codebase
  contract is :class:`BaseModelAdapter` with ``PredictionBatchResult`` /
  ``InternalFlowRowTrace`` / ``SurfaceMesh3DData`` value objects, so this
  adapter follows the Phase-04 ``LinearRegressionAdapter`` shape instead.
- All probability math delegates to sklearn's own ``predict_proba`` /
  ``decision_function`` (numerically stable ``expit`` / ``softmax`` paths),
  so extreme outliers (``z << -700``) can never raise an ``exp`` overflow.
- 3D surface ``z_values`` are probabilities in ``[0, 1]`` by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

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

_LOGISTIC_TYPES = (LogisticRegression, LogisticRegressionCV)


def _final_estimator(model_obj: Any) -> Any:
    """Unwrap sklearn Pipeline to its terminal estimator."""
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
    """Convert (possibly sparse) batch to contiguous float32 ndarray."""
    if _is_sparse(matrix):
        matrix = matrix.toarray()
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))


def _extract_coef_intercept(estimator: Any) -> tuple[np.ndarray, np.ndarray, list[Any], bool]:
    """Return ``(coef_2d (C, D) float32, intercepts (C,) float32, classes, is_binary)``."""
    raw_coef = np.asarray(getattr(estimator, "coef_"), dtype=np.float32)
    if raw_coef.ndim == 1:
        raw_coef = raw_coef.reshape(1, -1)
    coef_2d = np.ascontiguousarray(raw_coef, dtype=np.float32)

    raw_intercept = getattr(estimator, "intercept_", 0.0)
    # Edge case #3: fit_intercept=False may surface as scalar 0.0 or [0.].
    intercepts = np.atleast_1d(np.asarray(raw_intercept, dtype=np.float32)).ravel()
    n_classes = int(coef_2d.shape[0]) if coef_2d.shape[0] > 1 else 0

    raw_classes = getattr(estimator, "classes_", None)
    if raw_classes is not None:
        classes: list[Any] = [
            (v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())
        ]
    else:
        classes = []

    is_binary = len(classes) == 2 if classes else coef_2d.shape[0] == 1
    n_rows = 2 if is_binary and coef_2d.shape[0] == 1 else coef_2d.shape[0]
    if is_binary:
        # Binary: single coef row describes the positive class; mirror it so
        # per-class indexing (0/1) stays simple for callers.
        if intercepts.size == 0:
            intercepts = np.zeros((1,), dtype=np.float32)
        intercepts = np.ascontiguousarray(intercepts.ravel()[:1], dtype=np.float32)
    else:
        n_rows = coef_2d.shape[0]
        if intercepts.size == 1 and n_rows > 1:
            intercepts = np.repeat(intercepts, n_rows).astype(np.float32)
        intercepts = np.ascontiguousarray(intercepts.ravel()[:n_rows], dtype=np.float32)
        if intercepts.size < n_rows:
            pad = np.zeros((n_rows - intercepts.size,), dtype=np.float32)
            intercepts = np.concatenate([intercepts, pad])
    intercepts = np.where(np.isfinite(intercepts), intercepts, 0.0).astype(np.float32)
    coef_2d = np.where(np.isfinite(coef_2d), coef_2d, 0.0).astype(np.float32)
    return coef_2d, intercepts, classes, is_binary


def _positive_class_index(classes: list[Any]) -> int:
    return 1 if len(classes) >= 2 else 0


@ModelRegistry.register("logistic_regression")
class LogisticRegressionAdapter(BaseModelAdapter):
    display_name = "Logistic Regression (Sigmoid / Softmax)"
    family_name = "Linear Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.PROBABILITY_SIGMOID_SURFACE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _LOGISTIC_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        coef_2d, _, _, is_binary = _extract_coef_intercept(estimator)
        if is_binary:
            weights = coef_2d[0]
        else:
            # Multiclass overview: mean weight across classes (finite-guarded).
            weights = np.mean(coef_2d, axis=0).astype(np.float32)
        out: list[float | None] = []
        for i in range(n_features):
            if i < weights.size and bool(np.isfinite(weights[i])):
                out.append(float(weights[i]))
            else:
                out.append(None)
        return out

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in: Any = X_float32 if _is_sparse(X_float32) else _as_dense_float32(X_float32)
        preds = np.asarray(model_obj.predict(X_in)).ravel()
        try:
            probas = np.asarray(model_obj.predict_proba(X_in), dtype=np.float64)
        except (AttributeError, NotImplementedError, ValueError):
            probas = None
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if probas is not None and probas.size:
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
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
        coef_2d, intercepts, classes, is_binary = _extract_coef_intercept(estimator)

        # Pipeline case: contributions live in transformed space so that
        # sum(w * x') + b == pipeline.decision_function(X) exactly (#4 analogue).
        if hasattr(model_obj, "steps"):
            try:
                X_trans_raw = model_obj[:-1].transform(
                    X_float32.toarray()
                    if _is_sparse(X_float32)
                    else np.asarray(X_float32, dtype=np.float32)
                )
            except Exception:
                X_trans_raw = X_float32
        else:
            X_trans_raw = X_float32
        X_trans = _as_dense_float32(X_trans_raw)
        X_native = _as_dense_float32(X_float32)

        # Native sklearn parity references (stable expit/softmax paths).
        try:
            probas_all = np.asarray(model_obj.predict_proba(X_native), dtype=np.float64)
        except (AttributeError, NotImplementedError, ValueError):
            probas_all = np.zeros((X_native.shape[0], 1), dtype=np.float64)
        if probas_all.ndim == 1:
            probas_all = probas_all.reshape(-1, 1)
        try:
            logits_raw = np.asarray(model_obj.decision_function(X_native), dtype=np.float64)
        except (AttributeError, NotImplementedError, ValueError):
            logits_raw = np.zeros((X_native.shape[0],), dtype=np.float64)

        n_classes_prob = probas_all.shape[1]
        if is_binary:
            logits_2d = logits_raw.reshape(-1, 1)
        else:
            logits_2d = logits_raw.reshape(X_native.shape[0], -1)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_native.shape[0]:
                continue
            if X_trans.shape[0] == len(row_indices):
                x_vec = np.asarray(X_trans[pos], dtype=np.float32)
                native_pos = pos
            else:
                x_vec = np.asarray(X_trans[row_id], dtype=np.float32)
                native_pos = row_id
            if native_pos < 0 or native_pos >= probas_all.shape[0]:
                continue

            proba_row = probas_all[native_pos]
            if is_binary:
                k = 0  # single coef row == positive class
                resolved_target = _positive_class_index(classes) if classes else 1
                base = float(intercepts[0]) if intercepts.size else 0.0
                logit = float(logits_2d[native_pos, 0]) if logits_2d.size else 0.0
            else:
                if target_class_idx is not None and 0 <= target_class_idx < coef_2d.shape[0]:
                    k = int(target_class_idx)
                elif 0 <= target_idx < coef_2d.shape[0] and target_class_idx is None and False:
                    # Reserved: explicit per-target explainability via target_idx.
                    k = int(target_idx)
                else:
                    k = int(np.argmax(proba_row)) if proba_row.size else 0
                resolved_target = k
                base = float(intercepts[k]) if k < intercepts.size else 0.0
                logit = (
                    float(logits_2d[native_pos, k])
                    if logits_2d.shape[1] > k
                    else 0.0
                )

            weights_k = coef_2d[k] if k < coef_2d.shape[0] else coef_2d[0]
            w = weights_k[: x_vec.size]
            contribs = (x_vec[: w.size] * w).astype(np.float32)
            contribs = np.where(np.isfinite(contribs), contribs, 0.0).astype(np.float32)

            names = feature_names or [f"feature_{i}" for i in range(contribs.size)]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(len(contribs), len(names) if names else len(contribs)))
            }
            if not contrib_map:
                contrib_map = {f"feature_{i}": float(contribs[i]) for i in range(len(contribs))}

            try:
                pred_label_raw = model_obj.predict(X_native[native_pos : native_pos + 1])[0]
                if isinstance(pred_label_raw, (np.integer, bool, np.bool_)):
                    final_pred: float | int | str = int(pred_label_raw)
                elif isinstance(pred_label_raw, (np.floating, float)):
                    f = float(pred_label_raw)
                    final_pred = int(f) if f.is_integer() else f
                else:
                    final_pred = pred_label_raw  # type: ignore[assignment]
            except Exception:
                final_pred = resolved_target

            extra = {
                "classes": [str(c) for c in classes] if classes else [],
                "class_labels": classes if classes else [],
                "is_binary": bool(is_binary),
                "decision_scores": [float(v) for v in np.asarray(logits_2d[native_pos]).ravel()],
                "logit": logit,
                "probabilities": [float(v) for v in np.asarray(proba_row).ravel()],
                "predicted_class": final_pred,
                "target_class_idx": int(resolved_target),
                "winning_class_idx": int(np.argmax(proba_row)) if proba_row.size else 0,
                "weights": [float(v) for v in w],
                "intercept": base,
                "decision_threshold": 0.5,
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=row_id,
                    base_value=base,
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
        coef_2d, intercepts, classes, is_binary = _extract_coef_intercept(estimator)

        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")

        n_points = res * res
        grid_matrix = np.zeros((n_points, max(n_features, 1)), dtype=np.float32)
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
            if 0 <= feat_idx < grid_matrix.shape[1] and np.isfinite(med_val):
                grid_matrix[:, feat_idx] = np.float32(med_val)

        ax = grid_spec.axis_x_feature_index
        ay = grid_spec.axis_y_feature_index
        if 0 <= ax < grid_matrix.shape[1]:
            grid_matrix[:, ax] = xx.ravel()
        if n_features > 1 and ay != ax and 0 <= ay < grid_matrix.shape[1]:
            grid_matrix[:, ay] = yy.ravel()

        probas = np.asarray(model_obj.predict_proba(grid_matrix), dtype=np.float64)
        if probas.ndim == 1:
            probas = probas.reshape(-1, 1)
        n_grid_classes = probas.shape[1]

        if target_class_idx is not None and 0 <= target_class_idx < n_grid_classes:
            resolved = int(target_class_idx)
        elif 0 <= target_idx < n_grid_classes and n_grid_classes > 2:
            # Explicit per-class surface request via generic target_idx.
            resolved = int(target_idx)
        elif is_binary:
            resolved = (
                _positive_class_index(classes)
                if len(classes) >= 2
                else min(1, n_grid_classes - 1)
            )
        else:
            resolved = 1 if n_grid_classes > 1 else 0
        z_grid = np.clip(
            np.asarray(probas[:, resolved], dtype=np.float64).reshape(res, res), 0.0, 1.0
        )

        try:
            logits_grid = np.asarray(model_obj.decision_function(grid_matrix), dtype=np.float64)
        except (AttributeError, NotImplementedError, ValueError):
            logits_grid = np.zeros((n_points,), dtype=np.float64)

        # GPU phase-shift uniforms: sigma(w_x * x + w_y * y + b_rest).
        k_coef = 0 if is_binary else min(resolved, coef_2d.shape[0] - 1)
        w_row = coef_2d[k_coef] if coef_2d.size else np.zeros((grid_matrix.shape[1],), dtype=np.float32)
        b_k = float(intercepts[k_coef]) if k_coef < intercepts.size else 0.0
        wx = float(w_row[ax]) if 0 <= ax < w_row.size else 0.0
        wy = float(w_row[ay]) if 0 <= ay < w_row.size and ay != ax else 0.0
        off_axis = 0.0
        for j in range(grid_matrix.shape[1]):
            if j == ax or j == ay or j >= w_row.size:
                continue
            med = grid_spec.fixed_feature_medians.get(j, 0.0)
            try:
                off_axis += float(w_row[j]) * float(med)
            except (TypeError, ValueError):
                continue
        intercept_offset = float(b_k + off_axis)
        is_pipeline = hasattr(model_obj, "steps")

        z_list = [float(v) for v in np.asarray(z_grid, dtype=np.float32).ravel()]
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=res,
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=z_list,
            extra_overlays={
                "is_binary": bool(is_binary),
                "classes": [str(c) for c in classes] if classes else [],
                "resolved_target_class_idx": resolved,
                "boundary_threshold": 0.5,
                "decision_threshold": 0.5,
                "coef_2d": [wx, wy],
                "intercept_offset": intercept_offset,
                "weight_row": [float(v) for v in w_row],
                "base_intercept": b_k,
                "gpu_formula_valid": not bool(is_pipeline),
                "zero_influence_axes": bool(wx == 0.0 and wy == 0.0),
                "logits_sample": [float(v) for v in np.asarray(logits_grid).ravel()[:8]],
            },
        )
