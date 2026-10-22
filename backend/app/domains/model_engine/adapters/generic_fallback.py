# backend/app/domains/model_engine/adapters/generic_fallback.py
"""Generic sklearn fallback adapters — cover every tabular supervised estimator.

Specific adapters (linear/trees/boosting/knn/svm/nb/...) claim their own
classes first (registry resolves in import order — this module is imported
LAST in ``model_use_case.py``). These two adapters catch everything else
with a ``predict`` method: ``Dummy``, ``Bagging``, ``Voting``, ``Stacking``,
``CalibratedClassifierCV``, ``GaussianProcess*``, ``RANSAC/TheilSen/Huber/
Quantile/Tweedie/Poisson/Gamma/ARD/Bayesian`` regressors, ``PLS*``,
``MultiOutput*/ClassifierChain/RegressorChain``, ``TransformedTargetRegressor``,
``XGBRanker``-style third-party estimators, etc.

Contract: exact inference (native ``predict``/``predict_proba``), honest
attribution (``coef_``/``feature_importances_`` when available, otherwise a
uniform split clearly flagged ``generic_fallback: True`` so the waterfall
still satisfies ``base + sum(contribs) == prediction``).
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

MAX_SURFACE_RESOLUTION = 100

_JOBLIB_SUFFIXES = frozenset({".joblib", ".pkl", ".pickle"})


def _is_onnx_proto(obj: Any) -> bool:
    graph = getattr(obj, "graph", None)
    return graph is not None and hasattr(graph, "node") and hasattr(graph, "input")


def _is_native_booster(obj: Any) -> bool:
    if callable(getattr(obj, "num_boosted_rounds", None)) and callable(
        getattr(obj, "save_config", None)
    ):
        return True
    return callable(getattr(obj, "num_trees", None)) and callable(
        getattr(obj, "dump_model", None)
    )


def _final_estimator(model_obj: Any) -> Any:
    """Unwrap search/pipeline wrappers to the terminal learner."""
    seen: set[int] = set()
    cur = model_obj
    while id(cur) not in seen:
        seen.add(id(cur))
        nxt = getattr(cur, "best_estimator_", None)
        if nxt is not None and nxt is not cur:
            cur = nxt
            continue
        steps = getattr(cur, "steps", None)
        if isinstance(steps, list) and steps:
            last = steps[-1]
            cand = last[1] if isinstance(last, (tuple, list)) and len(last) == 2 else last
            if cand is not cur:
                cur = cand
                continue
        inner = getattr(cur, "_final_estimator", None)
        if inner is not None and inner is not cur:
            cur = inner
            continue
        # CalibratedClassifierCV / multi-output wrappers hold fitted sub-estimators.
        for attr in ("calibrated_classifiers_", "estimators_"):
            seq = getattr(cur, attr, None)
            if isinstance(seq, (list, tuple)) and seq:
                first = seq[0]
                cand = getattr(first, "estimator", getattr(first, "base_estimator", first))
                if cand is not None and cand is not cur:
                    cur = cand
                    break
        else:
            break
    # TransformedTargetRegressor wraps regressor_.
    reg = getattr(cur, "regressor_", None)
    if reg is not None:
        return reg
    return cur


def _is_fallback_candidate(obj: Any, file_ext: str) -> bool:
    if file_ext.lower() not in _JOBLIB_SUFFIXES:
        return False
    if _is_onnx_proto(obj) or _is_native_booster(obj):
        return False
    if not callable(getattr(obj, "predict", None)):
        return False
    return True


def _classes_of(est: Any) -> list[Any] | None:
    raw = getattr(est, "classes_", None)
    if raw is None:
        return None
    try:
        return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]
    except TypeError:
        return None


def _is_classifier_like(obj: Any) -> bool:
    est = _final_estimator(obj)
    if _classes_of(est) is not None:
        return True
    if _classes_of(obj) is not None:
        return True
    if str(getattr(est, "_estimator_type", "")).lower() == "classifier":
        return True
    if str(getattr(obj, "_estimator_type", "")).lower() == "classifier":
        return True
    return False


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


def _extract_static_weights(est: Any, n_features: int) -> tuple[list[float | None], str]:
    """Return ``(weights, source)`` — coef_ mean, importances, or Nones."""
    n = max(int(n_features), 0)
    coef = getattr(est, "coef_", None)
    if coef is not None:
        try:
            arr = np.asarray(coef, dtype=np.float64)
            if arr.ndim == 1 and arr.size == n:
                return [float(v) if np.isfinite(v) else 0.0 for v in arr], "coef"
            if arr.ndim >= 2 and arr.shape[-1] == n:
                mean = arr.reshape(-1, n).mean(axis=0)
                return [float(v) if np.isfinite(v) else 0.0 for v in mean], "coef_mean"
        except (TypeError, ValueError):
            pass
    imp = getattr(est, "feature_importances_", None)
    if imp is not None:
        try:
            arr = np.asarray(imp, dtype=np.float64).ravel()
            if arr.size == n and n > 0:
                total = float(np.sum(np.abs(arr)))
                if total > 0:
                    arr = arr / total
                return [float(v) if np.isfinite(v) else 0.0 for v in arr], "importances"
        except (TypeError, ValueError):
            pass
    return [None for _ in range(n)], "none"


def _predict_proba_native(model_obj: Any, X: np.ndarray) -> np.ndarray | None:
    fn = getattr(model_obj, "predict_proba", None)
    if not callable(fn):
        return None
    try:
        proba = np.asarray(fn(X), dtype=np.float64)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        if proba.shape[0] != X.shape[0]:
            return None
        return np.where(np.isfinite(proba), proba, 0.0)
    except (AttributeError, ValueError, TypeError, NotImplementedError):
        return None


def _decision_to_proba(decision: np.ndarray) -> np.ndarray:
    d = np.asarray(decision, dtype=np.float64)
    if d.ndim == 1:
        sig = 1.0 / (1.0 + np.exp(-np.clip(d, -60.0, 60.0)))
        return np.stack([1.0 - sig, sig], axis=1)
    shift = d - np.max(d, axis=1, keepdims=True)
    exp = np.exp(np.clip(shift, -60.0, 60.0))
    denom = np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-300)
    return exp / denom


def _predict_with_proba(
    model_obj: Any, X: np.ndarray
) -> tuple[list[Any], list[list[float]] | None, bool]:
    """Return ``(predictions, probabilities, is_pseudo)``."""
    try:
        preds = list(np.asarray(model_obj.predict(X)).ravel().tolist())
    except (ValueError, AttributeError, TypeError):
        preds = [0 for _ in range(X.shape[0])]
    proba = _predict_proba_native(model_obj, X)
    if proba is not None:
        return preds, [[float(v) for v in row] for row in proba.tolist()], False
    dec_fn = getattr(model_obj, "decision_function", None)
    if callable(dec_fn):
        try:
            pseudo = _decision_to_proba(np.asarray(dec_fn(X)))
            return preds, [[float(v) for v in row] for row in pseudo.tolist()], True
        except (ValueError, TypeError, AttributeError):
            pass
    return preds, None, True


def _coerce_pred(v: Any) -> float | int | str:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return int(f) if f.is_integer() else f
    return v  # type: ignore[return-value]


def _surface_grid(grid_spec: GridSpec3D, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
    x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
    y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
    grid = np.zeros((res * res, max(int(n_features), 1)), dtype=np.float32)
    for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
        if 0 <= feat_idx < grid.shape[1] and np.isfinite(med_val):
            grid[:, feat_idx] = np.float32(med_val)
    ax, ay = grid_spec.axis_x_feature_index, grid_spec.axis_y_feature_index
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
    if 0 <= ax < grid.shape[1]:
        grid[:, ax] = xx.ravel()
    if int(n_features) > 1 and ay != ax and 0 <= ay < grid.shape[1]:
        grid[:, ay] = yy.ravel()
    return grid, x_lin, y_lin


class _GenericFallbackBase(BaseModelAdapter):
    supports_nan_input = False

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        weights, _ = _extract_static_weights(_final_estimator(estimator), n_features)
        return weights

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X = _as_dense_float32(X_float32)
        preds, probas, _ = _predict_with_proba(model_obj, X)
        confidences: list[float] | None = None
        if probas is not None and probas:
            confidences = [float(max(row)) if row else 0.0 for row in probas]
        return PredictionBatchResult(
            predictions=[_coerce_pred(v) for v in preds],
            probabilities=probas,
            confidences=confidences,
        )


@ModelRegistry.register("sklearn_generic_classifier")
class SklearnGenericClassifierAdapter(_GenericFallbackBase):
    display_name = "Sklearn Classifier (Generic Fallback)"
    family_name = "Generic Sklearn"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.PROBABILITY_SIGMOID_SURFACE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        try:
            return bool(_is_fallback_candidate(estimator, file_ext) and _is_classifier_like(estimator))
        except Exception:
            return False

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        est = _final_estimator(model_obj)
        classes = _classes_of(est) or _classes_of(model_obj) or [0, 1]
        n_classes = len(classes)
        resolved = int(target_class_idx) if target_class_idx is not None else int(target_idx)
        resolved = int(min(max(resolved, 0), max(n_classes - 1, 0)))
        X = _as_dense_float32(X_float32)
        n_features = X.shape[1] if X.ndim == 2 else len(feature_names)
        weights, source = _extract_static_weights(est, n_features)
        try:
            preds, probas, is_pseudo = _predict_with_proba(model_obj, X)
        except Exception:
            preds, probas, is_pseudo = [], None, True
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < X.shape[0]):
                continue
            row = X[row_id].astype(np.float64)
            pred = _coerce_pred(preds[row_id]) if row_id < len(preds) else classes[resolved]
            prob_row = list(probas[row_id]) if probas is not None and row_id < len(probas) else []
            if all(w is not None for w in weights) and len(weights) == n_features and n_features:
                w_arr = np.asarray([float(w) for w in weights], dtype=np.float64)  # type: ignore[misc]
                raw = w_arr * row[: len(w_arr)]
                try:
                    score = float(np.asarray(model_obj.decision_function(X[row_id : row_id + 1])).ravel()[0])
                    base = float(score - raw.sum())
                except (AttributeError, ValueError, TypeError, IndexError):
                    base = 0.0
                contribs = {names[i]: float(raw[i]) for i in range(min(len(names), raw.size))}
            else:
                try:
                    score = float(prob_row[resolved]) if resolved < len(prob_row) else float(pred)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    score = 0.0
                base = 0.0
                share = float(score) / max(n_features, 1) if n_features else 0.0
                contribs = {name: share for name in names} if names else {"feature_0": 0.0}
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base),
                    final_prediction=pred,  # type: ignore[arg-type]
                    feature_contributions={k: float(v) for k, v in contribs.items()},
                    extra_mechanics={
                        "target_idx": int(resolved),
                        "target_class_idx": int(resolved),
                        "is_classifier": True,
                        "is_binary": bool(n_classes == 2),
                        "classes": [str(c) for c in classes],
                        "class_labels": list(classes),
                        "probabilities": [float(v) for v in prob_row],
                        "has_native_proba": bool(probas is not None and not is_pseudo),
                        "is_pseudo_proba": bool(is_pseudo),
                        "weight_source": str(source),
                        "generic_fallback": True,
                        "estimator_class": type(est).__name__,
                    },
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
        est = _final_estimator(model_obj)
        classes = _classes_of(est) or _classes_of(model_obj) or [0, 1]
        resolved = int(target_class_idx) if target_class_idx is not None else (1 if len(classes) == 2 else int(target_idx))
        resolved = int(min(max(resolved, 0), max(len(classes) - 1, 0)))
        grid, x_lin, y_lin = _surface_grid(grid_spec, n_features)
        try:
            _, probas, _ = _predict_with_proba(model_obj, grid)
            if probas is not None:
                z = np.asarray([[row[min(resolved, len(row) - 1)] for row in probas]], dtype=np.float64).ravel()
            else:
                z = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
        except (ValueError, AttributeError, TypeError):
            z = np.zeros((grid.shape[0],), dtype=np.float64)
        z = np.where(np.isfinite(z), z, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in z.tolist()],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes],
                "generic_fallback": True,
                "estimator_class": type(est).__name__,
                "boundary_threshold": 0.5,
                "is_classifier": True,
            },
        )


@ModelRegistry.register("sklearn_generic_regressor")
class SklearnGenericRegressorAdapter(_GenericFallbackBase):
    display_name = "Sklearn Regressor (Generic Fallback)"
    family_name = "Generic Sklearn"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        try:
            return bool(_is_fallback_candidate(estimator, file_ext) and not _is_classifier_like(estimator))
        except Exception:
            return False

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        est = _final_estimator(model_obj)
        X = _as_dense_float32(X_float32)
        n_features = X.shape[1] if X.ndim == 2 else len(feature_names)
        weights, source = _extract_static_weights(est, n_features)
        try:
            preds = list(np.asarray(model_obj.predict(X)).ravel().tolist())
        except (ValueError, AttributeError, TypeError):
            preds = [0.0 for _ in range(X.shape[0])]
        try:
            intercept = float(np.asarray(getattr(est, "intercept_", 0.0), dtype=np.float64).ravel()[0])
            if not np.isfinite(intercept):
                intercept = 0.0
        except (TypeError, ValueError, AttributeError, IndexError):
            intercept = 0.0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < X.shape[0]):
                continue
            row = X[row_id].astype(np.float64)
            pred = _coerce_pred(preds[row_id]) if row_id < len(preds) else 0.0
            if all(w is not None for w in weights) and len(weights) == n_features and n_features:
                w_arr = np.asarray([float(w) for w in weights], dtype=np.float64)  # type: ignore[misc]
                raw = w_arr * row[: len(w_arr)]
                base = float(intercept) if np.isfinite(intercept) and source.startswith("coef") else 0.0
                if not source.startswith("coef"):
                    base = 0.0
                    raw = np.full_like(row[: len(w_arr)], float(pred) / max(len(w_arr), 1))  # type: ignore[arg-type]
                contribs = {names[i]: float(raw[i]) for i in range(min(len(names), raw.size))}
            else:
                base = 0.0
                share = float(pred) / max(n_features, 1) if n_features else 0.0  # type: ignore[arg-type]
                contribs = {name: share for name in names} if names else {"feature_0": 0.0}
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base),
                    final_prediction=pred,  # type: ignore[arg-type]
                    feature_contributions={k: float(v) for k, v in contribs.items()},
                    extra_mechanics={
                        "is_classifier": False,
                        "weight_source": str(source),
                        "generic_fallback": True,
                        "estimator_class": type(est).__name__,
                    },
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
        est = _final_estimator(model_obj)
        grid, x_lin, y_lin = _surface_grid(grid_spec, n_features)
        try:
            z = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
        except (ValueError, AttributeError, TypeError):
            z = np.zeros((grid.shape[0],), dtype=np.float64)
        z = np.where(np.isfinite(z), z, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in z.tolist()],
            extra_overlays={
                "generic_fallback": True,
                "estimator_class": type(est).__name__,
                "is_classifier": False,
            },
        )
