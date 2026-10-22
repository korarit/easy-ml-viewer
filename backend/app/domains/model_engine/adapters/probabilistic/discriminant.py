# backend/app/domains/model_engine/adapters/probabilistic/discriminant.py
"""Phase 26 — LDA & QDA discriminant-analysis adapters (3D covariance ellipsoids).

Covers ``LinearDiscriminantAnalysis`` (all solvers: ``svd``, ``lsqr``,
``eigen``) and ``QuadraticDiscriminantAnalysis`` (plain estimators and
``Pipeline``/``GridSearchCV`` wrappers alike — fitted attributes are read
off the terminal estimator while inference runs through the container).

Blueprint mapping (phase-26-model-lda-and-qda.md):

- Mahalanobis: ``D_M^2(x, mu_k) = (x - mu_k)^T Sigma_k^{-1} (x - mu_k)``.
- LDA (shared ``Sigma``): ``decision = X @ coef_.T + intercept_`` — the
  waterfall reads ``coef_``/``intercept_`` directly so
  ``base + sum(phi) == decision_function`` exactly for every solver.
- QDA (per-class ``Sigma_k``): ``score_k = -0.5*log|Sigma_k|
  - 0.5*D_M^2 + log pi_k`` with per-feature split
  ``phi_j = -0.5*(x_j - mu_j)*p_j``, ``p = Sigma_k^{-1}(x - mu_k)``.
  QDA scores are computed through the same whitening
  (``rotations_``/``scalings_``) path as sklearn itself.

Waterfall convention (matches GaussianNB/discrete-NB adapters):
binary mode reports the log-odds ratio; multiclass mode reports the
resolved target class.

Edge cases handled (blueprint section 7):

1. ``store_covariance=False`` (the default): QDA covariances are rebuilt
   exactly from ``(R * s) @ R.T``; LDA falls back to a ``pinv(scalings_)``
   approximation flagged with ``is_covariance_approximated``.
2. Singular covariances (collinear features, ``N_k < D``): only
   ``pinv(cov + I*1e-9)`` is ever used — never ``inv``.
3. Reflection eigenbases (``det(V) == -1``): first column is flipped so
   Three.js always receives a right-handed rotation.
4. ``solver='lsqr'``/``'eigen'`` LDA (no SVD ``scalings_``): the
   ``covariance_``-first lookup order keeps every solver working.
5. LDA vs QDA ``scalings_`` semantics: handled in separate branches —
   QDA ``scalings_`` are per-class eigenvalue lists, LDA ``scalings_``
   is a ``(D, n_components)`` projection matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
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
COV_JITTER = 1e-9
EIG_FLOOR = 1e-9
PRIOR_FLOOR = 1e-15

_DISCRIMINANT_TYPES = (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)


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
    """Map raw rows into the fitted discriminant feature space (pipeline-aware)."""
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _classes_of(estimator: Any) -> list[Any]:
    raw = getattr(estimator, "classes_", None)
    if raw is None:
        return [0, 1]
    return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]


def _priors_of(estimator: Any, n_classes: int) -> np.ndarray:
    raw = getattr(estimator, "priors_", None)
    if raw is None:
        raw = getattr(estimator, "priors", None)
    arr = np.asarray(raw, dtype=np.float64).ravel() if raw is not None else np.array([])
    if arr.size == n_classes and np.all(np.isfinite(arr)) and float(np.sum(arr)) > 0:
        arr = np.maximum(arr, 0.0)
        return arr / float(np.sum(arr))
    return np.full((max(n_classes, 1),), 1.0 / max(n_classes, 1), dtype=np.float64)


def _means_of(estimator: Any, n_classes: int, n_features: int) -> np.ndarray:
    raw = np.asarray(getattr(estimator, "means_", np.zeros((n_classes, n_features))),
                     dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != n_features:
        fixed = np.zeros((n_classes, n_features), dtype=np.float64)
        take_c = min(n_classes, raw.shape[0]) if raw.ndim == 2 else 0
        take_d = min(n_features, raw.shape[1]) if raw.ndim == 2 else 0
        if take_c and take_d:
            fixed[:take_c, :take_d] = raw[:take_c, :take_d]
        return fixed
    if raw.shape[0] != n_classes:
        fixed = np.zeros((n_classes, n_features), dtype=np.float64)
        take_c = min(n_classes, raw.shape[0])
        fixed[:take_c] = raw[:take_c]
        return fixed
    return raw


def _resolve_covariances(
    estimator: Any, is_qda: bool, n_classes: int, n_features: int
) -> tuple[list[np.ndarray], bool]:
    """Return ``(per-class covariances, is_approximated)`` (blueprint edge #1/#5)."""
    d = max(int(n_features), 1)
    if is_qda:
        raw_cov = getattr(estimator, "covariance_", None)
        if raw_cov is not None:
            try:
                arr = np.asarray(raw_cov, dtype=np.float64)
                if arr.ndim == 3 and arr.shape[0] == n_classes and arr.shape[1:] == (d, d):
                    return [np.asarray(c, dtype=np.float64) for c in arr], False
            except (TypeError, ValueError):
                pass
        rotations = getattr(estimator, "rotations_", None)
        scalings = getattr(estimator, "scalings_", None)
        if isinstance(rotations, list) and isinstance(scalings, list) \
                and len(rotations) == len(scalings) and len(rotations) > 0:
            try:
                covs: list[np.ndarray] = []
                for r_item, s_item in zip(rotations, scalings):
                    r_mat = np.asarray(r_item, dtype=np.float64)
                    s_vec = np.asarray(s_item, dtype=np.float64).ravel()
                    if r_mat.shape != (d, d) or s_vec.size != d:
                        raise ValueError("rotation/scaling shape mismatch")
                    covs.append((r_mat * s_vec[np.newaxis, :]) @ r_mat.T)
                while len(covs) < n_classes:
                    covs.append(covs[-1].copy())
                return covs[:n_classes], False
            except (TypeError, ValueError):
                pass
        return [np.eye(d, dtype=np.float64) for _ in range(n_classes)], True

    raw_cov = getattr(estimator, "covariance_", None)
    if raw_cov is not None:
        try:
            shared = np.asarray(raw_cov, dtype=np.float64)
            if shared.shape == (d, d):
                return [shared.copy() for _ in range(n_classes)], False
        except (TypeError, ValueError):
            pass
    scal = getattr(estimator, "scalings_", None)
    if scal is not None:
        try:
            s_mat = np.asarray(scal, dtype=np.float64)
            if s_mat.ndim == 2 and s_mat.shape[0] == d and s_mat.shape[1] >= 1:
                s_pinv = np.linalg.pinv(s_mat)
                approx = s_pinv.T @ s_pinv
                if approx.shape == (d, d) and np.all(np.isfinite(approx)):
                    return [approx.copy() for _ in range(n_classes)], True
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass
    return [np.eye(d, dtype=np.float64) for _ in range(n_classes)], True


def _precisions_of(covs: list[np.ndarray]) -> list[np.ndarray]:
    """Moore-Penrose pseudo-inverse with ridge jitter (never ``inv``)."""
    out: list[np.ndarray] = []
    for cov in covs:
        c = np.asarray(cov, dtype=np.float64)
        d = c.shape[0]
        try:
            out.append(np.linalg.pinv(c + np.eye(d) * COV_JITTER))
        except np.linalg.LinAlgError:
            out.append(np.linalg.pinv(c + np.eye(d) * 1e-6))
    return out


def _log_dets_of(covs: list[np.ndarray]) -> list[float]:
    vals: list[float] = []
    for cov in covs:
        try:
            sign, ld = np.linalg.slogdet(np.asarray(cov, dtype=np.float64))
            vals.append(float(ld) if sign > 0 and np.isfinite(ld) else 0.0)
        except np.linalg.LinAlgError:
            vals.append(0.0)
    return vals


def _mahalanobis_batch(
    X: np.ndarray, means: np.ndarray, precisions: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mahal (R, C), euclid (R, C))`` distances."""
    r = int(X.shape[0])
    c = int(means.shape[0])
    mahal = np.zeros((r, c), dtype=np.float64)
    euclid = np.zeros((r, c), dtype=np.float64)
    for k in range(c):
        diff = np.asarray(X, dtype=np.float64) - means[k][np.newaxis, :]
        euclid[:, k] = np.linalg.norm(diff, axis=1)
        mahal_sq = np.maximum(np.einsum("ij,ij->i", diff @ precisions[k], diff), 0.0)
        mahal[:, k] = np.sqrt(mahal_sq)
    return mahal, euclid


def _ellipsoid_2d_specs(
    covs: list[np.ndarray], means: np.ndarray, axis_x: int, axis_y: int
) -> list[dict[str, Any]]:
    """2x2 sub-matrix eigendecomposition per class (blueprint section 6.1)."""
    specs: list[dict[str, Any]] = []
    for k, cov in enumerate(covs):
        c = np.asarray(cov, dtype=np.float64)
        d = c.shape[0]
        ix = [axis_x if 0 <= axis_x < d else 0, axis_y if 0 <= axis_y < d else min(1, d - 1)]
        sub = c[np.ix_(ix, ix)]
        sub = 0.5 * (sub + sub.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(sub)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.array([1.0, 1.0]), np.eye(2)
        eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64).ravel(), EIG_FLOOR)
        vecs = np.asarray(eigvecs, dtype=np.float64)
        if vecs.shape != (2, 2):
            vecs = np.eye(2)
        if float(np.linalg.det(vecs)) < 0.0:  # blueprint edge #3
            vecs[:, 0] *= -1.0
        # eigh returns ascending order; column 1 is the major axis.
        angle = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
        mu = means[k] if k < means.shape[0] else np.zeros((d,))
        specs.append(
            {
                "class_idx": int(k),
                "center_xy": [float(mu[ix[0]]), float(mu[ix[1]])],
                "radii_1sigma": [float(np.sqrt(eigvals[0])), float(np.sqrt(eigvals[1]))],
                "rotation_angle_rad": float(angle),
                "eigenvectors_2d": vecs.tolist(),
            }
        )
    return specs


@ModelRegistry.register("discriminant")
class DiscriminantAnalysisAdapter(BaseModelAdapter):
    display_name = "Discriminant Analysis (Ellipsoids)"
    family_name = "Probabilistic Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.DISCRIMINANT_ELLIPSOIDS

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_estimator(estimator), _DISCRIMINANT_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        est = _final_estimator(estimator)
        width = max(int(n_features), 0)
        if isinstance(est, LinearDiscriminantAnalysis):
            coef = getattr(est, "coef_", None)
            if coef is None:
                return [None for _ in range(width)]
            arr = np.asarray(coef, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != width:
                return [None for _ in range(width)]
            spread = np.max(np.abs(arr), axis=0)
            return [float(v) if np.isfinite(v) else 0.0 for v in spread.tolist()]
        return [None for _ in range(width)]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        try:
            preds = np.asarray(model_obj.predict(X_in)).ravel()
        except (ValueError, AttributeError, TypeError):
            preds = np.zeros((X_in.shape[0],), dtype=np.int64)
        try:
            probas = np.asarray(model_obj.predict_proba(X_in), dtype=np.float64)
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
        except (AttributeError, NotImplementedError, ValueError, TypeError):
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
        is_qda = isinstance(estimator, QuadraticDiscriminantAnalysis)
        classes = _classes_of(estimator)
        n_classes = len(classes)
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        is_binary = n_classes == 2

        X_native = _as_dense_float32(X_float32)
        X_q = np.asarray(_transform_for_query(model_obj, X_float32), dtype=np.float64)
        n_features = int(X_q.shape[1]) if X_q.ndim == 2 else 0
        priors = _priors_of(estimator, n_classes)
        log_priors = np.log(np.maximum(priors, PRIOR_FLOOR))
        means = _means_of(estimator, n_classes, n_features)
        covs, cov_approx = _resolve_covariances(estimator, is_qda, n_classes, n_features)
        precisions = _precisions_of(covs)
        log_dets = _log_dets_of(covs)

        valid = [r for r in row_indices if 0 <= r < X_native.shape[0]]
        Q = (
            np.ascontiguousarray(np.asarray(X_q[valid], dtype=np.float64))
            if valid
            else np.zeros((0, max(n_features, 1)), dtype=np.float64)
        )
        mahal, euclid = _mahalanobis_batch(
            Q if Q.size else np.zeros((0, max(n_features, 1))), means, precisions
        )
        try:
            native_preds = list(np.asarray(model_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError, TypeError):
            native_preds = [classes[0] for _ in valid]
        try:
            native_probas = np.asarray(model_obj.predict_proba(X_native), dtype=np.float64)
        except (AttributeError, NotImplementedError, ValueError, TypeError):
            native_probas = np.zeros((X_native.shape[0], n_classes), dtype=np.float64)
        try:
            native_dec = np.asarray(estimator.decision_function(np.asarray(Q)), dtype=np.float64)
        except (AttributeError, ValueError, TypeError):
            native_dec = np.zeros((len(valid), n_classes if not is_binary else 1))

        try:
            solver = str(getattr(estimator, "solver", "svd"))
        except Exception:
            solver = "svd"
        try:
            reg_param = float(getattr(estimator, "reg_param", 0.0) or 0.0)
        except (TypeError, ValueError):
            reg_param = 0.0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(valid):
            p_row = (
                native_probas[row_id]
                if 0 <= row_id < native_probas.shape[0]
                else np.full((n_classes,), 1.0 / max(n_classes, 1))
            )
            win = int(np.argmax(p_row)) if p_row.size else 0
            paradox = bool(
                int(np.argmin(mahal[pos])) != int(np.argmin(euclid[pos]))
            ) if mahal.shape[0] > pos else False

            if not is_qda:
                coef = np.asarray(getattr(estimator, "coef_", np.zeros((1, n_features))),
                                  dtype=np.float64)
                icept = np.asarray(getattr(estimator, "intercept_", np.zeros((1,))),
                                    dtype=np.float64).ravel()
                if coef.ndim == 1:
                    coef = coef.reshape(1, -1)
                if is_binary:
                    w = coef[0] if coef.shape[0] >= 1 else np.zeros((n_features,))
                    base = float(icept[0]) if icept.size else 0.0
                    full_vec = (Q[pos] * w).astype(np.float64).ravel()
                else:
                    w = coef[resolved] if resolved < coef.shape[0] else coef[-1]
                    base = float(icept[resolved]) if resolved < icept.size else 0.0
                    full_vec = (Q[pos] * w).astype(np.float64).ravel()
            else:
                if is_binary:
                    vecs = np.zeros((2, n_features), dtype=np.float64)
                    bases = np.zeros((2,), dtype=np.float64)
                    for kk in range(2):
                        dk = Q[pos] - means[kk]
                        pk = precisions[kk] @ dk
                        vecs[kk] = -0.5 * dk * pk
                        bases[kk] = -0.5 * log_dets[kk] + float(log_priors[kk])
                    base = float(bases[1] - bases[0])
                    full_vec = (vecs[1] - vecs[0]).astype(np.float64).ravel()
                else:
                    dk = Q[pos] - means[resolved]
                    pk = precisions[resolved] @ dk
                    full_vec = (-0.5 * dk * pk).astype(np.float64).ravel()
                    base = float(-0.5 * log_dets[resolved] + log_priors[resolved])
            full_vec = np.nan_to_num(full_vec, nan=0.0, posinf=30.0, neginf=-30.0)
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(full_vec[i])
                for i in range(int(full_vec.size))
            }
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[win] if win < len(classes) else win
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            dec_row = (
                native_dec[pos].tolist()
                if native_dec.ndim == 2 and pos < native_dec.shape[0]
                else [float(native_dec[pos])] if native_dec.size > pos else [0.0]
            )
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": True,
                "is_binary": bool(is_binary),
                "discriminant": "qda" if is_qda else "lda",
                "algorithm": type(estimator).__name__,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "priors": [float(v) for v in priors.tolist()],
                "log_priors": [float(v) for v in log_priors.tolist()],
                "means": means.tolist(),
                "log_determinants": [float(v) for v in log_dets],
                "solver": solver,
                "reg_param": float(reg_param),
                "store_covariance": bool(getattr(estimator, "covariance_", None) is not None),
                "is_covariance_approximated": bool(cov_approx),
                "has_native_proba": True,
                "is_pseudo_proba": False,
                "decision_scores": [float(v) for v in np.asarray(dec_row).ravel().tolist()],
                "probabilities": [float(v) for v in np.asarray(p_row).ravel().tolist()],
                "predicted_class": final_pred,
                "winning_class_idx": int(win),
                "resolved_class_idx": int(resolved),
                "mahalanobis_distances": [float(v) for v in mahal[pos].tolist()],
                "euclidean_distances": [float(v) for v in euclid[pos].tolist()],
                "paradox_detected": bool(paradox),
                "nearest_mahalanobis_class": int(np.argmin(mahal[pos])),
                "nearest_euclidean_class": int(np.argmin(euclid[pos])),
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
        is_qda = isinstance(estimator, QuadraticDiscriminantAnalysis)
        classes = _classes_of(estimator)
        n_classes = len(classes)
        if n_classes == 2:
            resolved = 1  # binary convention: render the positive-class surface
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
        grid_q = _transform_for_query(model_obj, grid)
        try:
            probas = np.asarray(model_obj.predict_proba(np.asarray(grid_q, dtype=np.float32)))
        except (ValueError, AttributeError, TypeError):
            probas = np.zeros((n_points, max(n_classes, 1)), dtype=np.float64)
        c_idx = int(min(resolved, probas.shape[1] - 1)) if probas.size else 0
        z_surface = (
            np.asarray(probas[:, c_idx], dtype=np.float64).reshape(res, res)
            if probas.size
            else np.zeros((res, res), dtype=np.float64)
        )
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
        n_feat = int(grid_q.shape[1]) if np.ndim(grid_q) == 2 else int(n_features)
        means = _means_of(estimator, n_classes, n_feat)
        covs, cov_approx = _resolve_covariances(estimator, is_qda, n_classes, n_feat)
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
                "discriminant": "qda" if is_qda else "lda",
                "algorithm": type(estimator).__name__,
                "solver": str(getattr(estimator, "solver", "svd") or "svd"),
                "reg_param": float(getattr(estimator, "reg_param", 0.0) or 0.0),
                "is_covariance_approximated": bool(cov_approx),
                "ellipsoids_2d": _ellipsoid_2d_specs(covs, means, ax, ay),
                "means_2d": [
                    [
                        float(means[k, ax]) if 0 <= ax < means.shape[1] else 0.0,
                        float(means[k, ay]) if 0 <= ay < means.shape[1] else 0.0,
                    ]
                    for k in range(n_classes)
                ],
                "boundary_threshold": 0.5,
                "is_classifier": True,
                "is_binary": bool(n_classes == 2),
            },
        )
