# backend/app/domains/model_engine/adapters/distance/svm.py
"""Phase 23 — Support Vector Machine family adapters (3D kernel saddle).

Covers ``SVC``, ``NuSVC``, ``LinearSVC`` (classification), ``SVR``,
``NuSVR``, ``LinearSVR`` (regression) and ``OneClassSVM`` (anomaly
detection) across the four standard kernels (``rbf``, ``poly``,
``sigmoid``, ``linear``).

Blueprint mapping (phase-23-model-svm-kernel-surface.md):

- Decision identity (kernel expansion)::

      f(x_t) = sum_s dual_s * K(x_s, x_t) + b == decision_function(X)[t]

  verified 100% for binary classifiers and kernel regressors
  (``predict`` for regressors). Multiclass OvO reports raw pairwise
  scores with an ``is_ovo`` flag instead of forcing a fake identity.
- Kernel influence per support vector: ``I_{t,s} = dual_s K(x_s, x_t)``;
  the HUD's Top-5 force tethers rank by ``|I_{t,s}|``.
- Waterfall: exact ``w_j x_j`` for linear models; directional
  ``-|x - weighted_sv| * sign(sum forces)`` for non-linear kernels.
- Support-vector display payload capped at 1,000 points ranked by
  ``|dual|`` importance (blueprint section 6.1); kernel matrices are
  built in 512-row chunks to keep the memory profile flat.

Edge cases handled (blueprint section 7):

1. ``SVC(probability=False)`` (default): pseudo-probability via
   ``expit``/``softmax`` with ``is_pseudo_proba=True``.
2. ``coef_`` exists only for linear kernels — never touched otherwise.
3. ``LinearSVC``/``LinearSVR`` keep no ``support_vectors_`` — they
   render as margin-corridor planes (Phase-21 style overlays).
4. Multiclass ``decision_function_shape='ovo'`` — raw pairwise scores
   are reported plus an ``is_ovo`` badge; the surface falls back to a
   ``predict == target`` indicator for OvO without native proba.
5. Platt scaling can disagree with ``predict`` near ``f(x) ~= 0`` —
   ``predict`` is always the source of truth; conflicts are flagged.
6. ``OneClassSVM`` predicts ``{-1, +1}`` — surfaced as OUTLIER/INLIER.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
    SVC,
    SVR,
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
MAX_DISPLAY_SV = 1000
TOP_FORCES = 5
CHUNK_ROWS = 512

_CLASSIFIER_TYPES = (SVC, NuSVC, LinearSVC)
_REGRESSOR_TYPES = (SVR, NuSVR, LinearSVR)
_LINEAR_TYPES = (LinearSVC, LinearSVR)
_KERNEL_NAMES = ("linear", "rbf", "poly", "sigmoid")


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


def _transform_for_query(model_obj: Any, X_float32: np.ndarray) -> np.ndarray:
    """Map raw rows into the fitted SVM feature space (pipeline-aware)."""
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _resolve_kernel(estimator: Any) -> str:
    if isinstance(estimator, _LINEAR_TYPES):
        return "linear"
    raw = getattr(estimator, "kernel", "rbf")
    if callable(raw):
        return "custom"
    try:
        name = str(raw)
    except Exception:
        return "rbf"
    return name if name in _KERNEL_NAMES else name


def _resolve_gamma(estimator: Any, n_features: int) -> float:
    raw = getattr(estimator, "_gamma", getattr(estimator, "gamma", "scale"))
    try:
        val = float(np.asarray(raw, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError, IndexError):
        val = float("nan")
    if not np.isfinite(val):
        val = 1.0 / max(int(n_features), 1)
    return float(val)


def _resolve_float(estimator: Any, names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = getattr(estimator, name, None)
        if raw is None or callable(raw):
            continue
        try:
            val = float(np.asarray(raw, dtype=np.float64).ravel()[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(val):
            return float(val)
    return float(default)


def _resolve_int(estimator: Any, name: str, default: int) -> int:
    raw = getattr(estimator, name, None)
    try:
        return int(np.asarray(raw).ravel()[0])
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _support_arrays(estimator: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(SV (S, D) float64, dual (R, S) float64, train_idx (S,))``.

    Empty arrays when the estimator keeps no support vectors
    (``LinearSVC``/``LinearSVR`` — blueprint edge case #3).
    """
    sv_raw = getattr(estimator, "support_vectors_", None)
    if sv_raw is None:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    sv = np.asarray(sv_raw.toarray() if _is_sparse(sv_raw) else sv_raw, dtype=np.float64)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    dual_raw = getattr(estimator, "dual_coef_", None)
    if dual_raw is None:
        dual = np.zeros((0, sv.shape[0]), dtype=np.float64)
    else:
        dual = np.asarray(dual_raw, dtype=np.float64)
        if dual.ndim == 1:
            dual = dual.reshape(1, -1)
    train_idx = np.asarray(getattr(estimator, "support_", np.arange(sv.shape[0])), dtype=np.int64).ravel()
    if train_idx.size != sv.shape[0]:
        train_idx = np.arange(sv.shape[0], dtype=np.int64)
    return sv, dual, train_idx


def _kernel_matrix(
    X_q: np.ndarray,
    sv: np.ndarray,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray | None:
    """Chunked ``K(X_q, SV)`` in float64; ``None`` for custom kernels."""
    if X_q.shape[0] == 0 or sv.shape[0] == 0:
        return np.zeros((int(X_q.shape[0]), int(sv.shape[0])), dtype=np.float64)
    if kernel not in _KERNEL_NAMES:
        return None
    if kernel == "linear":
        return np.ascontiguousarray(X_q @ sv.T, dtype=np.float64)
    out = np.empty((X_q.shape[0], sv.shape[0]), dtype=np.float64)
    for s in range(0, X_q.shape[0], CHUNK_ROWS):
        e = min(s + CHUNK_ROWS, X_q.shape[0])
        block = X_q[s:e]
        if kernel == "rbf":
            x2 = np.sum(block * block, axis=1, keepdims=True)
            s2 = np.sum(sv * sv, axis=1)[None, :]
            d2 = x2 + s2 - 2.0 * (block @ sv.T)
            np.maximum(d2, 0.0, out=d2)
            out[s:e] = np.exp(-gamma * d2)
        else:
            dot = block @ sv.T
            base = gamma * dot + coef0
            if kernel == "poly":
                out[s:e] = np.power(base, int(degree))
            else:  # sigmoid
                out[s:e] = np.tanh(base)
    return np.where(np.isfinite(out), out, 0.0)


def _dual_row(dual: np.ndarray, resolved: int) -> np.ndarray:
    if dual.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if dual.shape[0] == 1:
        return np.asarray(dual[0], dtype=np.float64).ravel()
    k = int(min(max(int(resolved), 0), dual.shape[0] - 1))
    return np.asarray(dual[k], dtype=np.float64).ravel()


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


def _native_proba_or_pseudo(
    model_obj: Any, estimator: Any, X_native: np.ndarray, scores_2d: np.ndarray | None, is_classifier: bool
) -> tuple[list[list[float]] | None, bool]:
    """Return ``(probabilities, is_native)`` for the batch."""
    if not is_classifier:
        return None, False
    if bool(getattr(estimator, "probability", False)) and callable(getattr(model_obj, "predict_proba", None)):
        try:
            probas = np.asarray(model_obj.predict_proba(X_native), dtype=np.float64)
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
            if probas.size:
                return [[float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()], True
        except (AttributeError, NotImplementedError, ValueError):
            pass
    if scores_2d is None or scores_2d.size == 0:
        return None, False
    if scores_2d.shape[1] == 1:
        p1 = _expit(scores_2d[:, 0])
        pseudo = np.column_stack([1.0 - p1, p1])
    else:
        pseudo = _softmax(scores_2d)
    return [[float(v) if np.isfinite(v) else 0.0 for v in row] for row in pseudo.tolist()], False


def _support_display_payload(
    estimator: Any, sv: np.ndarray, dual: np.ndarray, train_idx: np.ndarray
) -> dict[str, Any]:
    """Top-``MAX_DISPLAY_SV`` support vectors ranked by ``|dual|`` importance."""
    n_sv = int(sv.shape[0]) if sv.ndim == 2 else 0
    if n_sv == 0:
        return {
            "coordinates": [],
            "dual_weights": [],
            "support_indices": [],
            "importance": [],
            "total_support_vectors": 0,
            "displayed_count": 0,
        }
    if dual.size:
        importance = np.linalg.norm(np.asarray(dual, dtype=np.float64), axis=0)
    else:
        importance = np.zeros((n_sv,), dtype=np.float64)
    if n_sv > MAX_DISPLAY_SV:
        order = np.argsort(importance)[::-1][:MAX_DISPLAY_SV]
    else:
        order = np.arange(n_sv, dtype=np.int64)
    if dual.size:
        w = _dual_row(dual, 0)
        disp_w = [float(w[int(i)]) if 0 <= int(i) < w.size else 0.0 for i in order.tolist()]
    else:
        disp_w = [0.0 for _ in order.tolist()]
    order_list = [int(i) for i in order.tolist()]
    return {
        "coordinates": np.asarray(sv[order], dtype=np.float32).tolist(),
        "dual_weights": disp_w,
        "support_indices": [int(train_idx[int(i)]) for i in order_list],
        "display_positions": order_list,
        "importance": [float(importance[int(i)]) for i in order_list],
        "total_support_vectors": int(n_sv),
        "displayed_count": int(order.size),
    }


def _common_overlays(estimator: Any, n_features: int, kernel: str) -> dict[str, Any]:
    gamma_raw = getattr(estimator, "_gamma", getattr(estimator, "gamma", "scale"))
    try:
        gamma_val: Any = float(np.asarray(gamma_raw, dtype=np.float64).ravel()[0])
        if not np.isfinite(gamma_val):
            raise ValueError
    except (TypeError, ValueError, IndexError):
        gamma_val = str(gamma_raw) if gamma_raw is not None else "scale"
    n_support = getattr(estimator, "n_support_", None)
    try:
        n_support_list = [int(v) for v in np.asarray(n_support).ravel().tolist()] if n_support is not None else []
    except (TypeError, ValueError):
        n_support_list = []
    return {
        "kernel": kernel,
        "gamma": gamma_val,
        "C": float(getattr(estimator, "C", float("nan")))
        if getattr(estimator, "C", None) is not None
        else None,
        "nu": float(getattr(estimator, "nu", float("nan")))
        if getattr(estimator, "nu", None) is not None
        else None,
        "degree": int(_resolve_int(estimator, "degree", 3)),
        "coef0": float(_resolve_float(estimator, ("coef0",), 0.0)),
        "decision_function_shape": str(getattr(estimator, "decision_function_shape", "ovr")),
        "has_native_proba": bool(
            bool(getattr(estimator, "probability", False))
            and callable(getattr(estimator, "predict_proba", None))
        ),
        "n_support_per_class": n_support_list,
        "total_support_vectors": int(getattr(estimator, "support_vectors_", np.zeros((0, 0))).shape[0])
        if hasattr(estimator, "support_vectors_")
        else 0,
        "is_linear_model": bool(isinstance(estimator, _LINEAR_TYPES) or kernel == "linear"),
        "epsilon": float(_resolve_float(estimator, ("epsilon",), 0.1))
        if hasattr(estimator, "epsilon")
        else None,
    }


def _scores_via_model(
    model_obj: Any, X_native: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return ``(scores_2d | None, predictions)`` through the full (pipelined) model."""
    try:
        preds = np.asarray(model_obj.predict(X_native)).ravel()
    except (ValueError, AttributeError):
        preds = np.zeros((X_native.shape[0],), dtype=np.float64)
    scores: np.ndarray | None = None
    if callable(getattr(model_obj, "decision_function", None)):
        try:
            raw = np.asarray(model_obj.decision_function(X_native), dtype=np.float64)
            scores = raw.reshape(-1, 1) if raw.ndim == 1 else raw
        except (ValueError, AttributeError):
            scores = None
    return scores, preds


def _linear_weights(estimator: Any, n_classes: int, resolved: int) -> np.ndarray | None:
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    try:
        w2d = np.asarray(coef, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if w2d.ndim == 1:
        return w2d.ravel()
    if w2d.ndim == 2:
        if n_classes == 2:
            return w2d[0].ravel()
        k = int(min(max(int(resolved), 0), w2d.shape[0] - 1))
        return w2d[k].ravel()
    return None


def _intercept_for(estimator: Any, resolved: int) -> float:
    raw = np.asarray(getattr(estimator, "intercept_", [0.0]), dtype=np.float64).ravel()
    if raw.size == 0:
        return 0.0
    if raw.size == 1:
        return float(raw[0])
    k = int(min(max(int(resolved), 0), raw.size - 1))
    return float(raw[k])


@ModelRegistry.register("svm_classifier")
class SvmClassifierAdapter(BaseModelAdapter):
    display_name = "SVM Classifier (Kernel Saddle)"
    family_name = "Distance Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.SVM_MARGIN_TUBE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _CLASSIFIER_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        kernel = _resolve_kernel(estimator)
        if kernel == "linear" or isinstance(estimator, _LINEAR_TYPES):
            w = _linear_weights(estimator, 2, 0)
            if w is None:
                return [None for _ in range(max(int(n_features), 0))]
            out: list[float | None] = []
            for i in range(int(n_features)):
                out.append(float(w[i]) if i < w.size and bool(np.isfinite(w[i])) else None)
            return out
        return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        scores, preds = _scores_via_model(model_obj, X_in)
        estimator = _final_estimator(model_obj)
        probas, _ = _native_proba_or_pseudo(model_obj, estimator, X_in, scores, True)
        probabilities = probas
        confidences = [float(max(row)) if row else 0.0 for row in probas] if probas else None
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
        kernel = _resolve_kernel(estimator)
        X_native = _as_dense_float32(X_float32)
        X_q = _transform_for_query(model_obj, X_float32)
        n_features = int(X_q.shape[1]) if X_q.ndim == 2 else 0
        gamma = _resolve_gamma(estimator, n_features)
        degree = int(_resolve_int(estimator, "degree", 3))
        coef0 = float(_resolve_float(estimator, ("coef0",), 0.0))

        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
        n_classes = len(classes) if classes else 2
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        is_binary = n_classes == 2
        is_ovo = str(getattr(estimator, "decision_function_shape", "ovr")) == "ovo" and n_classes > 2

        scores_2d, native_preds = _scores_via_model(model_obj, X_native)
        probas_all, is_native = _native_proba_or_pseudo(model_obj, estimator, X_native, scores_2d, True)

        sv, dual, train_idx = _support_arrays(estimator)
        if sv.shape[0] and X_q.shape[1] != sv.shape[1]:
            width = max(sv.shape[1], int(X_q.shape[1]))
            padded = np.zeros((X_q.shape[0], width), dtype=np.float32)
            take = min(width, int(X_q.shape[1]))
            padded[:, :take] = X_q[:, :take]
            X_q = padded[:, : sv.shape[1]]
            n_features = int(sv.shape[1])
        K_mat = _kernel_matrix(
            np.asarray(X_q, dtype=np.float64), sv, kernel, gamma, degree, coef0
        )
        dual_row = _dual_row(dual, resolved)
        b = _intercept_for(estimator, 0 if is_binary else resolved)
        w_lin = _linear_weights(estimator, n_classes, resolved)

        sv_payload = _support_display_payload(estimator, sv, dual, train_idx)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_native.shape[0]:
                continue
            q_native = np.asarray(X_native[row_id], dtype=np.float64).ravel()
            q_trans = (
                np.asarray(X_q[pos], dtype=np.float64).ravel()
                if X_q.shape[0] == len(row_indices)
                else np.asarray(X_q[row_id], dtype=np.float64).ravel()
            )
            # Score column for the resolved target.
            if scores_2d is not None and row_id < scores_2d.shape[0]:
                if is_binary or scores_2d.shape[1] == 1:
                    raw_score = float(scores_2d[row_id, 0])
                elif not is_ovo and resolved < scores_2d.shape[1]:
                    raw_score = float(scores_2d[row_id, resolved])
                else:
                    raw_score = float(scores_2d[row_id, min(resolved, scores_2d.shape[1] - 1)])
                score_row = [float(v) for v in np.asarray(scores_2d[row_id]).ravel()]
            else:
                raw_score = 0.0
                score_row = []
            # Kernel forces for this row.
            forces: list[dict[str, Any]] = []
            force_sum = 0.0
            if K_mat is not None and dual_row.size and pos < K_mat.shape[0]:
                k_row = np.asarray(K_mat[pos], dtype=np.float64).ravel()
                f_row = dual_row[: k_row.size] * k_row
                force_sum = float(np.sum(f_row))
                top = np.argsort(np.abs(f_row))[::-1][:TOP_FORCES]
                for s in top.tolist():
                    forces.append(
                        {
                            "sv_position": int(s),
                            "support_index": int(train_idx[int(s)])
                            if int(s) < train_idx.size
                            else int(s),
                            "kernel_value": float(k_row[int(s)])
                            if np.isfinite(k_row[int(s)])
                            else 0.0,
                            "force": float(f_row[int(s)])
                            if np.isfinite(f_row[int(s)])
                            else 0.0,
                        }
                    )
            # Waterfall contributions.
            if w_lin is not None and w_lin.size:
                width_c = min(int(w_lin.size), int(q_native.size))
                contribs = (q_native[:width_c] * w_lin[:width_c]).astype(np.float64)
                contribs = np.where(np.isfinite(contribs), contribs, 0.0)
            elif K_mat is not None and sv.shape[0] and pos < K_mat.shape[0]:
                k_row = np.asarray(K_mat[pos], dtype=np.float64).ravel()
                w_abs = np.abs(
                    dual_row[: k_row.size] * k_row if dual_row.size else k_row
                )
                total = float(np.sum(w_abs))
                if total > 1e-12:
                    weighted_sv = np.sum(sv[: k_row.size] * w_abs[:, None], axis=0) / total
                else:
                    weighted_sv = np.mean(sv, axis=0)
                width_c = min(int(weighted_sv.size), int(q_trans.size))
                sign = 1.0 if force_sum >= 0 else -1.0
                contribs = (-np.abs(q_trans[:width_c] - weighted_sv[:width_c]) * sign).astype(np.float64)
            else:
                contribs = np.zeros((min(len(feature_names), int(q_native.size)),), dtype=np.float64)
            names = feature_names or [f"feature_{i}" for i in range(int(contribs.size))]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(int(contribs.size), len(names) if names else int(contribs.size)))
            }
            if not contrib_map and names:
                contrib_map = {names[0]: 0.0}
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if resolved < len(classes) else 0
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            proba_row: list[float] = []
            if probas_all is not None and row_id < len(probas_all):
                proba_row = [float(v) for v in probas_all[row_id]]
            platt_conflict = False
            if is_native and proba_row and len(proba_row) == len(classes) and classes:
                try:
                    if classes[int(np.argmax(proba_row))] != final_pred:
                        platt_conflict = bool(abs(raw_score) < 1.0)
                except (IndexError, ValueError, TypeError):
                    platt_conflict = False
            if scores_2d is not None and scores_2d.shape[1] > 1 and not is_ovo:
                order = np.argsort(np.asarray(scores_2d[row_id], dtype=np.float64))[::-1]
                gap = (
                    float(scores_2d[row_id, int(order[0])] - scores_2d[row_id, int(order[1])])
                    if order.size > 1
                    else 0.0
                )
                inside = bool(abs(raw_score) < 1.0) if is_binary else bool(gap < 1.0)
                status = (
                    "INSIDE MARGIN CORRIDOR"
                    if inside
                    else ("SAFE MARGIN (> +1)" if raw_score >= 0 else "SAFE MARGIN (< -1)")
                    if is_binary
                    else (f"TOP-2 GAP {gap:.3f} (TIGHT)" if gap < 1.0 else f"TOP-2 GAP {gap:.3f} (SAFE)")
                )
            else:
                inside = bool(abs(raw_score) < 1.0)
                status = (
                    "INSIDE MARGIN CORRIDOR"
                    if inside
                    else ("SAFE MARGIN (> +1)" if raw_score >= 0 else "SAFE MARGIN (< -1)")
                )
                gap = 0.0
            extra: dict[str, Any] = {
                **_common_overlays(estimator, n_features, kernel),
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": True,
                "is_binary": bool(is_binary),
                "is_ovo": bool(is_ovo),
                "classes": [str(c) for c in classes] if classes else [],
                "class_labels": classes if classes else [],
                "has_native_proba": bool(is_native),
                "is_pseudo_proba": bool(not is_native),
                "decision_scores": score_row,
                "raw_score": float(raw_score),
                "geometric_gap": float(gap),
                "inside_margin": bool(inside),
                "margin_status": status,
                "margin_width": 1.0,
                "intercept": float(b),
                "probabilities": proba_row,
                "predicted_class": final_pred,
                "platt_conflict": bool(platt_conflict),
                "kernel_forces_top": forces,
                "kernel_force_sum": float(force_sum),
                "weight_row": [float(v) for v in w_lin] if w_lin is not None else None,
                "is_linear_model": bool(w_lin is not None),
                "support_vectors_display": sv_payload,
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(b),
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions={k: float(v) for k, v in contrib_map.items()},
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
        kernel = _resolve_kernel(estimator)
        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
        n_classes = len(classes) if classes else 2
        is_ovo = str(getattr(estimator, "decision_function_shape", "ovr")) == "ovo" and n_classes > 2
        if n_classes == 2:
            resolved = 1
            single_col = True
        elif target_class_idx is not None:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            single_col = False
        else:
            resolved = int(min(max(int(target_idx), 0), n_classes - 1))
            single_col = False
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
        use_proba_col: int | None = None
        if (
            not is_ovo
            and bool(getattr(estimator, "probability", False))
            and callable(getattr(model_obj, "predict_proba", None))
        ):
            try:
                proba = np.asarray(model_obj.predict_proba(grid), dtype=np.float64)
                col = min(resolved, proba.shape[1] - 1) if proba.ndim == 2 else 0
                z_flat = proba[:, col] if proba.ndim == 2 else proba.ravel()
                use_proba_col = int(col)
            except (ValueError, AttributeError):
                z_flat = None  # type: ignore[assignment]
        else:
            z_flat = None  # type: ignore[assignment]
        if z_flat is None:
            try:
                if callable(getattr(model_obj, "decision_function", None)):
                    raw = np.asarray(model_obj.decision_function(grid), dtype=np.float64)
                    if raw.ndim == 1 or single_col:
                        z_flat = raw.ravel()
                    elif not is_ovo and resolved < raw.shape[1]:
                        z_flat = raw[:, resolved].ravel()
                    elif is_ovo:
                        # OvO pairwise scores have no single class column —
                        # fall back to a predict==target indicator surface.
                        preds = np.asarray(model_obj.predict(grid)).ravel()
                        tgt = classes[resolved] if resolved < len(classes) else resolved
                        z_flat = (preds == tgt).astype(np.float64)
                    else:
                        z_flat = raw[:, min(resolved, raw.shape[1] - 1)].ravel()
                else:
                    z_flat = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
            except (ValueError, AttributeError):
                z_flat = np.zeros((n_points,), dtype=np.float64)
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_flat).ravel().tolist()],
            extra_overlays={
                **_common_overlays(estimator, int(n_features), kernel),
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes] if classes else [],
                "surface_value": "predict_proba" if use_proba_col is not None else "decision_function",
                "surface_proba_column": use_proba_col,
                "boundary_threshold": 0.0,
                "margin_levels": [-1.0, 0.0, 1.0],
                "is_classifier": True,
                "is_binary": bool(n_classes == 2),
                "is_ovo": bool(is_ovo),
            },
        )


@ModelRegistry.register("svm_regressor")
class SvmRegressorAdapter(BaseModelAdapter):
    display_name = "SVM Regressor (Kernel Saddle + Epsilon Tube)"
    family_name = "Distance Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.SVM_MARGIN_TUBE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _REGRESSOR_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        kernel = _resolve_kernel(estimator)
        if kernel == "linear" or isinstance(estimator, _LINEAR_TYPES):
            w = _linear_weights(estimator, 0, 0)
            if w is None:
                return [None for _ in range(max(int(n_features), 0))]
            return [
                float(w[i]) if i < w.size and bool(np.isfinite(w[i])) else None
                for i in range(int(n_features))
            ]
        return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        preds = np.asarray(model_obj.predict(X_in), dtype=np.float64).ravel()
        return PredictionBatchResult(
            predictions=[float(v) if np.isfinite(v) else 0.0 for v in preds.tolist()]
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
        kernel = _resolve_kernel(estimator)
        X_native = _as_dense_float32(X_float32)
        X_q = _transform_for_query(model_obj, X_float32)
        n_features = int(X_q.shape[1]) if X_q.ndim == 2 else 0
        gamma = _resolve_gamma(estimator, n_features)
        degree = int(_resolve_int(estimator, "degree", 3))
        coef0 = float(_resolve_float(estimator, ("coef0",), 0.0))
        epsilon = float(_resolve_float(estimator, ("epsilon",), 0.1))

        try:
            native_preds = np.asarray(model_obj.predict(X_native), dtype=np.float64).ravel()
        except (ValueError, AttributeError):
            native_preds = np.zeros((X_native.shape[0],), dtype=np.float64)

        sv, dual, train_idx = _support_arrays(estimator)
        if sv.shape[0] and X_q.shape[1] != sv.shape[1]:
            width = max(sv.shape[1], int(X_q.shape[1]))
            padded = np.zeros((X_q.shape[0], width), dtype=np.float32)
            take = min(width, int(X_q.shape[1]))
            padded[:, :take] = X_q[:, :take]
            X_q = padded[:, : sv.shape[1]]
            n_features = int(sv.shape[1])
        K_mat = _kernel_matrix(
            np.asarray(X_q, dtype=np.float64), sv, kernel, gamma, degree, coef0
        )
        dual_row = _dual_row(dual, 0)
        b = _intercept_for(estimator, 0)
        w_lin = _linear_weights(estimator, 0, 0)
        sv_payload = _support_display_payload(estimator, sv, dual, train_idx)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_native.shape[0]:
                continue
            q_native = np.asarray(X_native[row_id], dtype=np.float64).ravel()
            q_trans = (
                np.asarray(X_q[pos], dtype=np.float64).ravel()
                if X_q.shape[0] == len(row_indices)
                else np.asarray(X_q[row_id], dtype=np.float64).ravel()
            )
            forces: list[dict[str, Any]] = []
            force_sum = 0.0
            if K_mat is not None and dual_row.size and pos < K_mat.shape[0]:
                k_row = np.asarray(K_mat[pos], dtype=np.float64).ravel()
                f_row = dual_row[: k_row.size] * k_row
                force_sum = float(np.sum(f_row))
                top = np.argsort(np.abs(f_row))[::-1][:TOP_FORCES]
                for s in top.tolist():
                    forces.append(
                        {
                            "sv_position": int(s),
                            "support_index": int(train_idx[int(s)])
                            if int(s) < train_idx.size
                            else int(s),
                            "kernel_value": float(k_row[int(s)])
                            if np.isfinite(k_row[int(s)])
                            else 0.0,
                            "force": float(f_row[int(s)])
                            if np.isfinite(f_row[int(s)])
                            else 0.0,
                        }
                    )
            if w_lin is not None and w_lin.size:
                width_c = min(int(w_lin.size), int(q_native.size))
                contribs = (q_native[:width_c] * w_lin[:width_c]).astype(np.float64)
                contribs = np.where(np.isfinite(contribs), contribs, 0.0)
            elif K_mat is not None and sv.shape[0] and pos < K_mat.shape[0]:
                k_row = np.asarray(K_mat[pos], dtype=np.float64).ravel()
                w_abs = np.abs(
                    dual_row[: k_row.size] * k_row if dual_row.size else k_row
                )
                total = float(np.sum(w_abs))
                if total > 1e-12:
                    weighted_sv = np.sum(sv[: k_row.size] * w_abs[:, None], axis=0) / total
                else:
                    weighted_sv = np.mean(sv, axis=0)
                width_c = min(int(weighted_sv.size), int(q_trans.size))
                sign = 1.0 if force_sum >= 0 else -1.0
                contribs = (-np.abs(q_trans[:width_c] - weighted_sv[:width_c]) * sign).astype(np.float64)
            else:
                contribs = np.zeros((min(len(feature_names), int(q_native.size)),), dtype=np.float64)
            names = feature_names or [f"feature_{i}" for i in range(int(contribs.size))]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(int(contribs.size), len(names) if names else int(contribs.size)))
            }
            if not contrib_map and names:
                contrib_map = {names[0]: 0.0}
            try:
                final_pred = float(native_preds[row_id])
            except (IndexError, TypeError, ValueError):
                final_pred = float(force_sum + b)
            if not np.isfinite(final_pred):
                final_pred = 0.0
            extra: dict[str, Any] = {
                **_common_overlays(estimator, n_features, kernel),
                "target_idx": 0,
                "target_class_idx": 0,
                "is_classifier": False,
                "is_pseudo_proba": False,
                "raw_score": float(final_pred),
                "intercept": float(b),
                "margin_width": float(epsilon),
                "epsilon": float(epsilon),
                "kernel_forces_top": forces,
                "kernel_force_sum": float(force_sum),
                "weight_row": [float(v) for v in w_lin] if w_lin is not None else None,
                "is_linear_model": bool(w_lin is not None),
                "support_vectors_display": sv_payload,
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(b),
                    final_prediction=float(final_pred),
                    feature_contributions={k: float(v) for k, v in contrib_map.items()},
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
        kernel = _resolve_kernel(estimator)
        epsilon = float(_resolve_float(estimator, ("epsilon",), 0.1))
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
        try:
            z_flat = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
        except (ValueError, AttributeError):
            z_flat = np.zeros((n_points,), dtype=np.float64)
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in z_flat.tolist()],
            extra_overlays={
                **_common_overlays(estimator, int(n_features), kernel),
                "target_idx": 0,
                "resolved_target_idx": 0,
                "boundary_threshold": None,
                "margin_levels": [-epsilon, 0.0, epsilon],
                "is_classifier": False,
            },
        )


@ModelRegistry.register("svm_oneclass")
class OneClassSvmAdapter(BaseModelAdapter):
    display_name = "One-Class SVM (Outlier Envelope)"
    family_name = "Distance Models"
    task_type = TaskType.ANOMALY_DETECTION
    visual_3d_type = Visual3DType.SVM_MARGIN_TUBE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, OneClassSVM)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        kernel = _resolve_kernel(estimator)
        if kernel == "linear":
            w = _linear_weights(estimator, 0, 0)
            if w is None:
                return [None for _ in range(max(int(n_features), 0))]
            return [
                float(w[i]) if i < w.size and bool(np.isfinite(w[i])) else None
                for i in range(int(n_features))
            ]
        return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        scores, preds = _scores_via_model(model_obj, X_in)
        py_preds: list[float | int | str] = [int(v) for v in np.asarray(preds).ravel().tolist()]
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if scores is not None and scores.size:
            p1 = _expit(scores[:, 0])
            pseudo = np.column_stack([p1, 1.0 - p1])
            probabilities = [
                [float(v) if np.isfinite(v) else 0.0 for v in row] for row in pseudo.tolist()
            ]
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
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
        kernel = _resolve_kernel(estimator)
        X_native = _as_dense_float32(X_float32)
        X_q = _transform_for_query(model_obj, X_float32)
        n_features = int(X_q.shape[1]) if X_q.ndim == 2 else 0
        gamma = _resolve_gamma(estimator, n_features)
        degree = int(_resolve_int(estimator, "degree", 3))
        coef0 = float(_resolve_float(estimator, ("coef0",), 0.0))

        scores_2d, native_preds = _scores_via_model(model_obj, X_native)
        sv, dual, train_idx = _support_arrays(estimator)
        if sv.shape[0] and X_q.shape[1] != sv.shape[1]:
            width = max(sv.shape[1], int(X_q.shape[1]))
            padded = np.zeros((X_q.shape[0], width), dtype=np.float32)
            take = min(width, int(X_q.shape[1]))
            padded[:, :take] = X_q[:, :take]
            X_q = padded[:, : sv.shape[1]]
            n_features = int(sv.shape[1])
        K_mat = _kernel_matrix(
            np.asarray(X_q, dtype=np.float64), sv, kernel, gamma, degree, coef0
        )
        dual_row = _dual_row(dual, 0)
        b = _intercept_for(estimator, 0)
        sv_payload = _support_display_payload(estimator, sv, dual, train_idx)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if row_id < 0 or row_id >= X_native.shape[0]:
                continue
            q_trans = (
                np.asarray(X_q[pos], dtype=np.float64).ravel()
                if X_q.shape[0] == len(row_indices)
                else np.asarray(X_q[row_id], dtype=np.float64).ravel()
            )
            raw_score = (
                float(scores_2d[row_id, 0])
                if scores_2d is not None and row_id < scores_2d.shape[0]
                else 0.0
            )
            forces: list[dict[str, Any]] = []
            force_sum = 0.0
            if K_mat is not None and dual_row.size and pos < K_mat.shape[0]:
                k_row = np.asarray(K_mat[pos], dtype=np.float64).ravel()
                f_row = dual_row[: k_row.size] * k_row
                force_sum = float(np.sum(f_row))
                top = np.argsort(np.abs(f_row))[::-1][:TOP_FORCES]
                for s in top.tolist():
                    forces.append(
                        {
                            "sv_position": int(s),
                            "support_index": int(train_idx[int(s)])
                            if int(s) < train_idx.size
                            else int(s),
                            "kernel_value": float(k_row[int(s)])
                            if np.isfinite(k_row[int(s)])
                            else 0.0,
                            "force": float(f_row[int(s)])
                            if np.isfinite(f_row[int(s)])
                            else 0.0,
                        }
                    )
            if K_mat is not None and sv.shape[0] and pos < K_mat.shape[0]:
                k_row = np.asarray(K_mat[pos], dtype=np.float64).ravel()
                w_abs = np.abs(
                    dual_row[: k_row.size] * k_row if dual_row.size else k_row
                )
                total = float(np.sum(w_abs))
                if total > 1e-12:
                    weighted_sv = np.sum(sv[: k_row.size] * w_abs[:, None], axis=0) / total
                else:
                    weighted_sv = np.mean(sv, axis=0)
                width_c = min(int(weighted_sv.size), int(q_trans.size))
                sign = 1.0 if force_sum >= 0 else -1.0
                contribs = (-np.abs(q_trans[:width_c] - weighted_sv[:width_c]) * sign).astype(np.float64)
            else:
                contribs = np.zeros((min(len(feature_names), int(q_trans.size)),), dtype=np.float64)
            names = feature_names or [f"feature_{i}" for i in range(int(contribs.size))]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                for i in range(min(int(contribs.size), len(names) if names else int(contribs.size)))
            }
            if not contrib_map and names:
                contrib_map = {names[0]: 0.0}
            try:
                raw_pred = int(native_preds[row_id])
            except (IndexError, TypeError, ValueError):
                raw_pred = 1 if raw_score >= 0 else -1
            label = "INLIER" if raw_pred == 1 else "OUTLIER"
            p_in = float(_expit(np.asarray([raw_score]))[0])
            extra: dict[str, Any] = {
                **_common_overlays(estimator, n_features, kernel),
                "target_idx": 0,
                "target_class_idx": 0,
                "is_classifier": False,
                "is_oneclass": True,
                "is_pseudo_proba": True,
                "decision_scores": [float(raw_score)],
                "raw_score": float(raw_score),
                "inside_margin": bool(raw_score >= 0),
                "margin_status": "INLIER (f(x) >= 0)" if raw_score >= 0 else "OUTLIER (f(x) < 0)",
                "margin_width": 0.0,
                "intercept": float(b),
                "probabilities": [float(p_in), float(1.0 - p_in)],
                "predicted_class": int(raw_pred),
                "predicted_label": label,
                "kernel_forces_top": forces,
                "kernel_force_sum": float(force_sum),
                "support_vectors_display": sv_payload,
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(b),
                    final_prediction=int(raw_pred),
                    feature_contributions={k: float(v) for k, v in contrib_map.items()},
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
        kernel = _resolve_kernel(estimator)
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
        try:
            if callable(getattr(model_obj, "decision_function", None)):
                z_flat = np.asarray(model_obj.decision_function(grid), dtype=np.float64).ravel()
            else:
                z_flat = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
        except (ValueError, AttributeError):
            z_flat = np.zeros((n_points,), dtype=np.float64)
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in z_flat.tolist()],
            extra_overlays={
                **_common_overlays(estimator, int(n_features), kernel),
                "target_idx": 0,
                "resolved_target_class_idx": 0,
                "boundary_threshold": 0.0,
                "margin_levels": [-1.0, 0.0, 1.0],
                "is_classifier": False,
                "is_oneclass": True,
            },
        )
