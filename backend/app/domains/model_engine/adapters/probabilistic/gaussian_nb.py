# backend/app/domains/model_engine/adapters/probabilistic/gaussian_nb.py
"""Phase 24 — Gaussian Naive Bayes adapter (3D bivariate bell hills).

Covers ``sklearn.naive_bayes.GaussianNB`` (``fit`` and ``partial_fit``
training alike — both expose the same fitted attributes).

Blueprint mapping (phase-24-model-gaussian-naive-bayes.md):

- Per-feature Gaussian log-density::

      l_{t,k,j} = -0.5 * log(2*pi*var_{k,j}) - (x_{t,j} - theta_{k,j})^2 / (2*var_{k,j})

- Joint log-likelihood ``J_{t,k} = log(pi_k) + sum_j l_{t,k,j}``
  matches ``model._joint_log_likelihood(X)`` exactly (float64).
- Waterfall: binary mode reports the log-odds ratio
  ``base = log(pi_1/pi_0)``, ``phi_j = l_{t,1,j} - l_{t,0,j}`` so that
  ``base + sum(phi) == J_{t,1} - J_{t,0}``; multiclass mode reports the
  resolved target class with ``base = log(pi_k)``.
- Surface: analytic bivariate bell hills per class on the axis pair::

      Z_k(x, y) = pi_k * exp(-((x-mu_x)^2/(2*sx^2) + (y-mu_y)^2/(2*sy^2)))

  axis-aligned footprints (conditional independence — no diagonal
  tilt). Posterior hills are derived client-side (``hill_k / sum``).

Edge cases handled (blueprint section 7):

1. ``var_`` vs legacy ``sigma_`` — ``getattr`` chain covers pickles
   from every sklearn era.
2. Log-scale computation everywhere (no underflow at D=100).
3. Near-zero variances floored at ``1e-12`` (on top of sklearn's own
   ``epsilon_`` from ``var_smoothing``).
4. Far-from-all-centers outliers flagged at ``min_z > 4`` with a
   ``LOW DENSITY OUTLIER`` badge (posterior 1.000 is misleading there).
5. Peak density can exceed 1.0 for narrow hills — the 3D scene uses the
   relative ``[0, pi_k]`` scale and reports true peak density in telemetry.
6. Custom ``priors=`` are read from ``class_prior_`` (ground truth),
   contrasted with empirical ``class_count_`` shares.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.naive_bayes import GaussianNB

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
MAX_HILL_CLASSES = 5
VAR_FLOOR = 1e-12
PRIOR_FLOOR = 1e-15
OUTLIER_Z = 4.0


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
    """Map raw rows into the fitted GNB feature space (pipeline-aware)."""
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _fitted_params(estimator: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    """Return ``(theta (C, D) float64, var floored, log_priors, classes)``."""
    theta = np.asarray(getattr(estimator, "theta_"), dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    # Blueprint edge #1: pre-1.0 pickles store ``sigma_`` instead of ``var_``.
    raw_var = getattr(estimator, "var_", getattr(estimator, "sigma_", None))
    if raw_var is None:
        raw_var = np.ones_like(theta)
    var = np.asarray(raw_var, dtype=np.float64)
    if var.ndim == 1:
        var = var.reshape(1, -1)
    var = np.maximum(np.where(np.isfinite(var), var, 1.0), VAR_FLOOR)
    raw_prior = np.asarray(getattr(estimator, "class_prior_"), dtype=np.float64).ravel()
    raw_prior = np.where(np.isfinite(raw_prior), np.maximum(raw_prior, 0.0), 0.0)
    total = float(np.sum(raw_prior))
    if total <= 0.0:
        raw_prior = np.full((theta.shape[0],), 1.0 / max(theta.shape[0], 1))
    else:
        raw_prior = raw_prior / total
    log_priors = np.log(np.maximum(raw_prior, PRIOR_FLOOR))
    raw_classes = getattr(estimator, "classes_", None)
    classes: list[Any] = (
        [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
        if raw_classes is not None
        else [int(i) for i in range(theta.shape[0])]
    )
    return theta, var, log_priors, classes


def _per_feature_ll(
    X_q: np.ndarray, theta: np.ndarray, var: np.ndarray
) -> np.ndarray:
    """Per-feature log-likelihood tensor ``(R, C, D)`` in float64 (log-scale)."""
    x64 = np.asarray(X_q, dtype=np.float64)
    if x64.ndim == 1:
        x64 = x64.reshape(1, -1)
    r, d = x64.shape
    c = theta.shape[0]
    if theta.shape[1] != d:
        width = max(theta.shape[1], d)
        x_pad = np.zeros((r, width), dtype=np.float64)
        x_pad[:, :d] = x64
        x64 = x_pad
        d = width
    n_theta_d = theta.shape[1]
    if n_theta_d != d:
        t_pad = np.zeros((c, d), dtype=np.float64)
        v_pad = np.ones((c, d), dtype=np.float64)
        take = min(n_theta_d, d)
        t_pad[:, :take] = theta[:, :take]
        v_pad[:, :take] = var[:, :take]
        theta, var = t_pad, v_pad
    diff_sq = (x64[:, None, :] - theta[None, :, :]) ** 2  # (R, C, D)
    log_2pi_var = np.log(2.0 * np.pi * var[None, :, :])
    return -0.5 * (log_2pi_var + diff_sq / var[None, :, :])


def _joint_ll(
    model_obj: Any, estimator: Any, X_q: np.ndarray, log_priors: np.ndarray, feat_ll: np.ndarray
) -> np.ndarray:
    """Joint log-likelihood ``(R, C)`` — native when available, else manual."""
    try:
        native = np.asarray(estimator._joint_log_likelihood(np.asarray(X_q)), dtype=np.float64)
        if native.ndim == 2 and native.shape[0] == feat_ll.shape[0]:
            return np.where(np.isfinite(native), native, -1e300)
    except (AttributeError, ValueError, TypeError):
        pass
    j = log_priors[None, :] + np.sum(feat_ll, axis=2)
    return np.where(np.isfinite(j), j, -1e300)


def _logsumexp_rows(j: np.ndarray) -> np.ndarray:
    m = np.max(j, axis=1, keepdims=True)
    s = np.sum(np.exp(j - m), axis=1, keepdims=True)
    return (m + np.log(np.maximum(s, 1e-300))).ravel()


def _posteriors(j: np.ndarray) -> np.ndarray:
    m = np.max(j, axis=1, keepdims=True)
    e = np.exp(j - m)
    denom = np.maximum(np.sum(e, axis=1, keepdims=True), 1e-300)
    return e / denom


@ModelRegistry.register("gaussian_nb")
class GaussianNBAdapter(BaseModelAdapter):
    display_name = "Gaussian Naive Bayes (Bell Hills)"
    family_name = "Probabilistic Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.GAUSSIAN_ELLIPSOID_DOMES

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, GaussianNB)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        # Generative model: no static per-feature weights.
        return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        try:
            preds = np.asarray(model_obj.predict(X_in)).ravel()
        except (ValueError, AttributeError):
            preds = np.zeros((X_in.shape[0],), dtype=np.int64)
        try:
            probas = np.asarray(model_obj.predict_proba(X_in), dtype=np.float64)
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
        except (AttributeError, NotImplementedError, ValueError):
            probas = None
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
        theta, var, log_priors, classes = _fitted_params(estimator)
        n_classes = int(theta.shape[0])
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        is_binary = n_classes == 2

        X_native = _as_dense_float32(X_float32)
        X_q = _transform_for_query(model_obj, X_float32)
        valid = [r for r in row_indices if 0 <= r < X_native.shape[0]]
        Q = (
            np.ascontiguousarray(np.asarray(X_q[valid], dtype=np.float32))
            if valid
            else np.zeros((0, theta.shape[1]), dtype=np.float32)
        )
        feat_ll = _per_feature_ll(Q, theta, var)  # (R, C, D)
        joint = _joint_ll(model_obj, estimator, Q, log_priors, feat_ll)  # (R, C)
        post = _posteriors(joint)
        try:
            native_preds = list(np.asarray(model_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError):
            native_preds = [classes[int(np.argmax(post[i]))] if post.shape[0] > i else classes[0] for i in range(len(valid))]
        var_smoothing = float(getattr(estimator, "var_smoothing", 1e-9) or 1e-9)
        epsilon = float(getattr(estimator, "epsilon_", 0.0) or 0.0)
        try:
            counts = [float(v) for v in np.asarray(getattr(estimator, "class_count_")).ravel().tolist()]
        except (AttributeError, TypeError, ValueError):
            counts = [float("nan") for _ in range(n_classes)]
        priors = [float(v) for v in np.exp(log_priors).tolist()]
        stds = np.sqrt(var)

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(valid):
            ll_row = feat_ll[pos]  # (C, D)
            j_row = joint[pos]  # (C,)
            p_row = post[pos]  # (C,)
            win = int(np.argmax(p_row))
            if is_binary:
                base = float(log_priors[1] - log_priors[0])
                contrib_vec = (ll_row[1] - ll_row[0]).astype(np.float64)
                z_vec = np.abs(Q[pos].astype(np.float64) - theta[resolved]) / stds[resolved]
            else:
                base = float(log_priors[resolved])
                contrib_vec = np.asarray(ll_row[resolved], dtype=np.float64).ravel()
                z_vec = np.abs(Q[pos].astype(np.float64) - theta[resolved]) / stds[resolved]
            z_vec = np.where(np.isfinite(z_vec), z_vec, 1e6)
            names = feature_names or [f"feature_{i}" for i in range(int(contrib_vec.size))]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contrib_vec[i])
                for i in range(min(int(contrib_vec.size), len(names) if names else int(contrib_vec.size)))
            }
            if not contrib_map and names:
                contrib_map = {names[0]: 0.0}
            min_z = float(np.min(z_vec)) if z_vec.size else float("inf")
            per_class_min_z: list[float] = []
            for k in range(n_classes):
                zk = np.abs(Q[pos].astype(np.float64) - theta[k]) / stds[k]
                zk = np.where(np.isfinite(zk), zk, 1e6)
                per_class_min_z.append(float(np.min(zk)) if zk.size else float("inf"))
            is_outlier = bool(min(per_class_min_z) > OUTLIER_Z) if per_class_min_z else False
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[win] if win < len(classes) else win
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": True,
                "is_binary": bool(is_binary),
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "class_priors": priors,
                "class_counts": counts,
                "log_priors": [float(v) for v in log_priors.tolist()],
                "has_native_proba": True,
                "is_pseudo_proba": False,
                "joint_log_probs": [float(v) for v in j_row.tolist()],
                "probabilities": [float(v) for v in p_row.tolist()],
                "predicted_class": final_pred,
                "winning_class_idx": int(win),
                "resolved_class_idx": int(resolved),
                "var_smoothing": float(var_smoothing),
                "epsilon": float(epsilon),
                "winning_class_z_scores": [float(v) for v in np.asarray(z_vec).ravel().tolist()],
                "per_class_min_z": [float(v) for v in per_class_min_z],
                "min_z_score": float(min_z),
                "is_low_density_outlier": bool(is_outlier),
                "outlier_threshold_sigma": float(OUTLIER_Z),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base),
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
        theta, var, log_priors, classes = _fitted_params(estimator)
        n_classes = int(theta.shape[0])
        if n_classes == 2:
            # Binary convention (matches SVM/KNN adapters): the surface
            # renders the positive-class hill.
            resolved = 1
        elif target_class_idx is not None:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), n_classes - 1))
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
        # Pipeline-aware: the analytic hills live in transformed space.
        grid_t = _transform_for_query(model_obj, grid)
        gx = np.asarray(grid_t[:, ax] if 0 <= ax < grid_t.shape[1] else grid[:, ax], dtype=np.float64).reshape(res, res)
        gy = np.asarray(grid_t[:, ay] if 0 <= ay < grid_t.shape[1] else grid[:, ay], dtype=np.float64).reshape(res, res)
        priors = np.exp(log_priors)
        stds = np.sqrt(var)

        hill_order = list(range(n_classes))
        truncated = False
        if n_classes > 8:
            # Blueprint section 6.2: render only the Top-5 prior hills.
            hill_order = list(np.argsort(priors)[::-1][:MAX_HILL_CLASSES])
            truncated = True
        hills: dict[int, list[float]] = {}
        for k in hill_order:
            mu_x = float(theta[k, ax]) if 0 <= ax < theta.shape[1] else 0.0
            mu_y = float(theta[k, ay]) if 0 <= ay < theta.shape[1] else 0.0
            sx = float(stds[k, ax]) if 0 <= ax < stds.shape[1] else 1.0
            sy = float(stds[k, ay]) if 0 <= ay < stds.shape[1] else 1.0
            zx = ((gx - mu_x) ** 2) / (2.0 * max(sx * sx, VAR_FLOOR))
            zy = ((gy - mu_y) ** 2) / (2.0 * max(sy * sy, VAR_FLOOR))
            hill = priors[k] * np.exp(-(zx + zy))
            hills[k] = [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(hill, dtype=np.float64).ravel()]
        z_values = hills.get(resolved, hills.get(hill_order[0], [0.0] * n_points))
        # True peak densities (may exceed 1.0 for narrow hills — edge #5).
        peak_density: dict[str, float] = {}
        for k in hill_order:
            dens = 1.0
            for j in range(var.shape[1]):
                dens *= 1.0 / (np.sqrt(2.0 * np.pi * max(float(var[k, j]), VAR_FLOOR)))
            peak_density[str(k)] = float(priors[k] * dens) if np.isfinite(priors[k] * dens) else 0.0
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in (z_values or [0.0] * n_points)],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes],
                "class_priors": [float(v) for v in priors.tolist()],
                "bell_hills_per_class": {str(k): v for k, v in hills.items()},
                "hill_class_order": [int(k) for k in hill_order],
                "hills_truncated_to_top5": bool(truncated),
                "peaks_2d": [
                    [
                        float(theta[k, ax]) if 0 <= ax < theta.shape[1] else 0.0,
                        float(theta[k, ay]) if 0 <= ay < theta.shape[1] else 0.0,
                    ]
                    for k in hill_order
                ],
                "stds_2d": [
                    [
                        float(stds[k, ax]) if 0 <= ax < stds.shape[1] else 1.0,
                        float(stds[k, ay]) if 0 <= ay < stds.shape[1] else 1.0,
                    ]
                    for k in hill_order
                ],
                "peak_densities": peak_density,
                "var_smoothing": float(getattr(estimator, "var_smoothing", 1e-9) or 1e-9),
                "boundary_threshold": 0.5,
                "is_classifier": True,
                "is_binary": bool(n_classes == 2),
            },
        )
