# backend/app/domains/model_engine/adapters/probabilistic/discrete_nb.py
"""Phase 25 — Discrete Naive Bayes adapters (3D non-negative probability surface + towers).

Covers ``MultinomialNB``, ``BernoulliNB``, ``ComplementNB`` and
``CategoricalNB`` (plain estimators and ``Pipeline``/``GridSearchCV``
wrappers alike — fitted attributes are read off the terminal estimator
while inference runs through the full container).

Blueprint mapping (phase-25-model-multinomial-bernoulli-nb.md):

- ``MultinomialNB`` (linear in log-space):
  ``J[t,k] = log pi[k] + sum_j X[t,j] * feature_log_prob_[k,j]``.
- ``BernoulliNB`` (explicit absence penalty):
  ``J[t,k] = log pi[k] + sum_j (b*log theta + (1-b)*log(1-theta))``
  with ``b = X > binarize``; the waterfall therefore splits each feature
  into a *presence* term and an *absence* term so that
  ``base + sum(phi) == joint`` exactly.
- ``ComplementNB``: ``J[t,k] = sum_j X[t,j] * w[k,j]`` where ``w`` is the
  complement-normalized ``feature_log_prob_``. NOTE: the installed
  scikit-learn (>= 1.x) scores classes with ``+X @ w`` (no negation, no
  prior except the single-class edge) and ``predict`` takes the argmax —
  the adapter follows the *native* sign so the waterfall closes over
  ``predict_joint_log_proba`` exactly, even though the phase blueprint
  text describes the older ``-X @ w`` convention.
- ``CategoricalNB``: ``J[t,k] = log pi[k] + sum_j FLP[j][k, cat[t,j]]``
  where ``FLP[j]`` is the ``(C, K_j)`` table of feature ``j``.

Waterfall convention (matches GaussianNB/KNN adapters):
binary mode reports the log-odds ratio (``base = log(pi1/pi0)``,
``phi = vec_1 - vec_0``); multiclass mode reports the resolved target
class (``base = log pi[k]``, ``phi = vec_k``).

Edge cases handled (blueprint section 7):

1. Negative grid/input values for Multinomial/Complement/Categorical —
   clamped to ``0.0`` (``safe_x_min = max(0, x_min)``).
2. Bernoulli absence term — ``(1-b)*log(1-theta)`` is always included.
3. Categorical unseen categories — clipped to ``[0, K_j - 1]`` with an
   ``unseen_category_clamped`` telemetry flag.
4. ``BernoulliNB(binarize=None)`` — no thresholding, input used as-is.
5. ComplementNB sign — follows the installed sklearn native ``+X @ w``
   convention (argmax decision) so waterfall sums match exactly.
6. ``alpha=0`` — ``0 * -inf`` NaNs are sanitized via ``nan_to_num``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    MultinomialNB,
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
NEG_PROB_FLOOR = 1e-15
LOG_FLOOR = -30.0
PRIOR_FLOOR = 1e-15
# Blueprint section 6.1: ship Top-30 + Top-30 instead of a (N, 50000) payload.
FLOW_PRUNE_THRESHOLD = 500
FLOW_PRUNE_TOPK = 30

_DISCRETE_TYPES = (MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB)


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
    """Map raw rows into the fitted discrete-NB feature space (pipeline-aware)."""
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _variant_of(estimator: Any) -> str:
    if isinstance(estimator, BernoulliNB):
        return "bernoulli"
    if isinstance(estimator, ComplementNB):
        return "complement"
    if isinstance(estimator, CategoricalNB):
        return "categorical"
    return "multinomial"


def _classes_of(estimator: Any) -> list[Any]:
    raw = getattr(estimator, "classes_", None)
    if raw is None:
        return [0, 1]
    return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]


def _log_priors_of(estimator: Any, n_classes: int) -> np.ndarray:
    raw = getattr(estimator, "class_log_prior_", None)
    if raw is not None:
        lp = np.asarray(raw, dtype=np.float64).ravel()
        if lp.size == n_classes:
            return np.where(np.isfinite(lp), lp, np.log(PRIOR_FLOOR))
    # ComplementNB (and friends without priors): uniform base so the
    # waterfall still closes over the native joint log-likelihood.
    return np.full((n_classes,), -np.log(max(n_classes, 1)), dtype=np.float64)


def _neg_log_prob(flp: np.ndarray) -> np.ndarray:
    """``log(1 - theta)`` from ``log theta`` without underflow."""
    theta = np.exp(np.clip(flp, LOG_FLOOR, 0.0))
    return np.log(np.maximum(1.0 - theta, NEG_PROB_FLOOR))


def _clip_non_negative(X: np.ndarray, variant: str) -> np.ndarray:
    if variant == "bernoulli":
        return X
    return np.maximum(X, 0.0)


def _joint_native(
    estimator: Any, X_q: np.ndarray, manual: np.ndarray
) -> np.ndarray:
    """Native ``predict_joint_log_proba`` when available, else manual."""
    try:
        native = np.asarray(
            estimator.predict_joint_log_proba(np.asarray(X_q)), dtype=np.float64
        )
        if native.ndim == 2 and native.shape == manual.shape:
            return np.where(np.isfinite(native), native, manual)
    except (AttributeError, ValueError, TypeError, IndexError):
        pass
    try:
        native = np.asarray(
            estimator._joint_log_likelihood(np.asarray(X_q)), dtype=np.float64
        )
        if native.ndim == 2 and native.shape == manual.shape:
            return np.where(np.isfinite(native), native, manual)
    except (AttributeError, ValueError, TypeError, IndexError):
        pass
    return manual


def _manual_joint_and_vectors(
    estimator: Any, X_q: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(joint (R, C), per-class vectors (R, C, D), telemetry)``."""
    n_classes = int(np.asarray(getattr(estimator, "classes_", [0, 1])).ravel().size)
    log_priors = _log_priors_of(estimator, n_classes)
    tele: dict[str, Any] = {"unseen_category_clamped": False}
    r = int(X_q.shape[0])

    if variant == "categorical":
        tables = list(getattr(estimator, "feature_log_prob_", []))
        d = len(tables)
        X_int = np.clip(np.rint(np.asarray(X_q, dtype=np.float64)), 0, None).astype(np.int64)
        if X_int.shape[1] != d and d > 0:
            width = max(d, int(X_int.shape[1]))
            padded = np.zeros((r, width), dtype=np.int64)
            padded[:, : X_int.shape[1]] = X_int
            X_int = padded
        vecs = np.zeros((r, n_classes, max(d, 1)), dtype=np.float64)
        for j, tab in enumerate(tables):
            tab = np.asarray(tab, dtype=np.float64)
            kj = int(tab.shape[1]) if tab.ndim == 2 else 1
            col = np.clip(X_int[:, j] if j < X_int.shape[1] else 0, 0, max(kj - 1, 0))
            if np.any(
                (X_int[:, j] if j < X_int.shape[1] else 0) > max(kj - 1, 0)
            ) or np.any(X_int[:, j] < 0 if j < X_int.shape[1] else False):
                tele["unseen_category_clamped"] = True
            for k in range(n_classes):
                row = tab[k] if k < tab.shape[0] else tab[-1]
                vecs[:, k, j] = row[col]
        joint = log_priors[None, :] + np.sum(vecs, axis=2)
        return np.where(np.isfinite(joint), joint, LOG_FLOOR), vecs, tele

    flp = np.asarray(getattr(estimator, "feature_log_prob_", np.zeros((n_classes, 1))),
                     dtype=np.float64)
    if flp.ndim == 1:
        flp = flp.reshape(1, -1)
    d = int(flp.shape[1])
    if X_q.shape[1] != d:
        width = max(d, int(X_q.shape[1]))
        x_pad = np.zeros((r, width), dtype=np.float64)
        x_pad[:, : X_q.shape[1]] = np.asarray(X_q, dtype=np.float64)
        X_use = x_pad
        f_pad = np.zeros((n_classes, width), dtype=np.float64)
        f_pad[:, :d] = flp
        flp = f_pad
        d = width
    else:
        X_use = np.asarray(X_q, dtype=np.float64)

    if variant == "bernoulli":
        thresh = getattr(estimator, "binarize", 0.0)
        if thresh is not None:
            try:
                X_bin = (X_use > float(thresh)).astype(np.float64)
            except (TypeError, ValueError):
                X_bin = np.asarray(X_use, dtype=np.float64)
        else:
            X_bin = np.asarray(X_use, dtype=np.float64)
        neg = _neg_log_prob(flp)
        vecs = X_bin[:, None, :] * flp[None, :, :] + (1.0 - X_bin)[:, None, :] * neg[None, :, :]
        joint = log_priors[None, :] + np.sum(vecs, axis=2)
    elif variant == "complement":
        # Native sklearn>=1.x sign: J = +X @ w (prior only for 1-class).
        vecs = X_use[:, None, :] * flp[None, :, :]
        joint = np.sum(vecs, axis=2)
    else:  # multinomial
        vecs = X_use[:, None, :] * flp[None, :, :]
        joint = log_priors[None, :] + np.sum(vecs, axis=2)
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=LOG_FLOOR * -1.0, neginf=LOG_FLOOR)
    joint = np.nan_to_num(joint, nan=LOG_FLOOR, posinf=LOG_FLOOR * -1.0, neginf=LOG_FLOOR)
    return joint, vecs, tele


def _posteriors(joint: np.ndarray) -> np.ndarray:
    m = np.max(joint, axis=1, keepdims=True)
    e = np.exp(np.clip(joint - m, LOG_FLOOR, 0.0))
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-300)


@ModelRegistry.register("discrete_nb")
class DiscreteNBAdapter(BaseModelAdapter):
    display_name = "Discrete Naive Bayes (Towers)"
    family_name = "Probabilistic Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.DISCRETE_NB_TOWERS

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_estimator(estimator), _DISCRETE_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        # Discriminative spread per feature: max_k log p - min_k log p.
        # Drives the "Top Discriminative Features" ranking (blueprint 3.2).
        est = _final_estimator(estimator)
        width = max(int(n_features), 0)
        if isinstance(est, CategoricalNB):
            spreads: list[float | None] = []
            for tab in list(getattr(est, "feature_log_prob_", [])):
                arr = np.asarray(tab, dtype=np.float64)
                spreads.append(float(np.max(arr) - np.min(arr)) if arr.size else 0.0)
            spreads += [0.0] * max(0, width - len(spreads))
            return spreads[:width]
        flp = getattr(est, "feature_log_prob_", None)
        if flp is None:
            return [None for _ in range(width)]
        arr = np.asarray(flp, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != width:
            return [None for _ in range(width)]
        spread = np.max(arr, axis=0) - np.min(arr, axis=0)
        return [float(v) if np.isfinite(v) else 0.0 for v in spread.tolist()]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        est = _final_estimator(model_obj)
        variant = _variant_of(est)
        query = _clip_non_negative(np.asarray(X_in, dtype=np.float64), variant)
        if variant == "categorical":
            query = np.clip(np.rint(query), 0, None)
        try:
            preds = np.asarray(model_obj.predict(np.asarray(X_in))).ravel()
        except (ValueError, AttributeError, TypeError, IndexError):
            try:
                preds = np.asarray(
                    model_obj.predict(np.asarray(query, dtype=np.float32))
                ).ravel()
            except Exception:
                preds = np.zeros((X_in.shape[0],), dtype=np.int64)
        try:
            probas = np.asarray(
                model_obj.predict_proba(np.asarray(X_in)), dtype=np.float64
            )
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
        except (AttributeError, NotImplementedError, ValueError, TypeError, IndexError):
            try:
                probas = np.asarray(
                    model_obj.predict_proba(np.asarray(query, dtype=np.float32)),
                    dtype=np.float64,
                )
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
            except Exception:
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
        variant = _variant_of(estimator)
        classes = _classes_of(estimator)
        n_classes = len(classes)
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        is_binary = n_classes == 2

        X_native = _as_dense_float32(X_float32)
        X_q_raw = _transform_for_query(model_obj, X_float32)
        X_q = _clip_non_negative(np.asarray(X_q_raw, dtype=np.float64), variant)
        if variant == "categorical":
            X_q = np.clip(np.rint(X_q), 0, None)

        valid = [r for r in row_indices if 0 <= r < X_native.shape[0]]
        Q = (
            np.ascontiguousarray(np.asarray(X_q[valid], dtype=np.float64))
            if valid
            else np.zeros((0, max(int(X_q.shape[1]), 1)), dtype=np.float64)
        )
        manual_joint, vecs, tele = _manual_joint_and_vectors(estimator, Q, variant)
        joint = _joint_native(estimator, Q, manual_joint)
        post = _posteriors(joint)
        try:
            native_preds = list(np.asarray(model_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError, TypeError, IndexError):
            native_preds = [
                classes[int(np.argmax(post[i]))] if post.shape[0] > i else classes[0]
                for i in range(len(valid))
            ]
        log_priors = _log_priors_of(estimator, n_classes)
        alpha = float(getattr(estimator, "alpha", 1.0) or 0.0)
        binarize_raw = getattr(estimator, "binarize", None)
        try:
            binarize_val = None if binarize_raw is None else float(binarize_raw)
        except (TypeError, ValueError):
            binarize_val = None
        names = feature_names or [f"feature_{i}" for i in range(int(vecs.shape[2]))]
        try:
            counts = [float(v) for v in np.asarray(getattr(estimator, "class_count_")).ravel().tolist()]
        except (AttributeError, TypeError, ValueError):
            counts = [float("nan") for _ in range(n_classes)]

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(valid):
            j_row = joint[pos]
            p_row = post[pos]
            win = int(np.argmax(p_row))
            if is_binary:
                base = 0.0 if variant == "complement" else float(log_priors[1] - log_priors[0])
                full_vec = (vecs[pos, 1] - vecs[pos, 0]).astype(np.float64).ravel()
            else:
                base = 0.0 if variant == "complement" else float(log_priors[resolved])
                full_vec = np.asarray(vecs[pos, resolved], dtype=np.float64).ravel()
            full_vec = np.nan_to_num(full_vec, nan=0.0, posinf=30.0, neginf=-30.0)
            # Blueprint 6.1: high-dimensional vocab ships Top-30 +/- only,
            # the hidden tail folds into ``pruned_remainder_sum`` so the
            # waterfall identity stays exactly verifiable.
            pruned_remainder = 0.0
            pruned_count = 0
            keep = np.ones(full_vec.shape, dtype=bool)
            if full_vec.size > FLOW_PRUNE_THRESHOLD:
                order = np.argsort(full_vec)
                keep_set = set(order[:FLOW_PRUNE_TOPK].tolist()) | set(
                    order[-FLOW_PRUNE_TOPK:].tolist()
                )
                keep = np.array([i in keep_set for i in range(full_vec.size)])
                pruned_remainder = float(np.sum(full_vec[~keep]))
                pruned_count = int(np.sum(~keep))
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(full_vec[i])
                for i in range(full_vec.size)
                if keep[i] and (i < len(names) or True)
            }
            if variant == "bernoulli" and binarize_val is not None and valid:
                X_row = np.asarray(X_q[pos], dtype=np.float64).ravel()
                presence: dict[str, str] = {}
                for i in range(min(full_vec.size, len(names))):
                    if not keep[i]:
                        continue
                    try:
                        presence[names[i]] = (
                            "present" if float(X_row[i]) > binarize_val else "absent"
                        )
                    except IndexError:
                        presence[names[i]] = "unknown"
            else:
                presence = {}
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
                "nb_variant": variant,
                "algorithm": type(estimator).__name__,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "class_counts": counts,
                "log_priors": [float(v) for v in log_priors.tolist()],
                "class_log_prior": [float(v) for v in log_priors.tolist()],
                "alpha": float(alpha),
                "binarize": binarize_val,
                "requires_non_negative_input": bool(variant != "bernoulli"),
                "non_negative_clamped": bool(variant != "bernoulli"),
                "has_native_proba": True,
                "is_pseudo_proba": False,
                "joint_log_probs": [float(v) for v in j_row.tolist()],
                "probabilities": [float(v) for v in p_row.tolist()],
                "predicted_class": final_pred,
                "winning_class_idx": int(win),
                "resolved_class_idx": int(resolved),
                "presence_flags": presence,
                "unseen_category_clamped": bool(tele.get("unseen_category_clamped", False)),
                "contributions_pruned": bool(pruned_count > 0),
                "pruned_feature_count": int(pruned_count),
                "pruned_remainder_sum": float(pruned_remainder),
                "tower_heights": [float(v) for v in p_row.tolist()],
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
        variant = _variant_of(estimator)
        classes = _classes_of(estimator)
        n_classes = len(classes)
        if n_classes == 2:
            resolved = 1  # binary convention: render the positive-class surface
        elif target_class_idx is not None:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), n_classes - 1))
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        # Blueprint edge #1: the count domain never dips below zero.
        x_lo = max(0.0, grid_spec.x_min) if variant != "bernoulli" else grid_spec.x_min
        y_lo = max(0.0, grid_spec.y_min) if variant != "bernoulli" else grid_spec.y_min
        x_hi = max(x_lo + 1.0, grid_spec.x_max)
        y_hi = max(y_lo + 1.0, grid_spec.y_max)
        x_lin = np.linspace(x_lo, x_hi, res, dtype=np.float32)
        y_lin = np.linspace(y_lo, y_hi, res, dtype=np.float32)
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
        grid_use = _clip_non_negative(np.asarray(grid_q, dtype=np.float64), variant)
        if variant == "categorical":
            grid_use = np.clip(np.rint(grid_use), 0, None)
        try:
            probas = np.asarray(model_obj.predict_proba(np.asarray(grid_use, dtype=np.float32)))
        except (ValueError, AttributeError, TypeError, IndexError):
            fallback = _clip_non_negative(np.asarray(grid_use, dtype=np.float64), variant)
            probas = np.asarray(
                model_obj.predict_proba(np.asarray(fallback, dtype=np.float32))
            )
        c_idx = int(min(resolved, probas.shape[1] - 1))
        z_surface = np.asarray(probas[:, c_idx], dtype=np.float64).reshape(res, res)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
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
                "nb_variant": variant,
                "algorithm": type(estimator).__name__,
                "alpha": float(getattr(estimator, "alpha", 1.0) or 0.0),
                "binarize": (
                    None
                    if getattr(estimator, "binarize", None) is None
                    else float(getattr(estimator, "binarize"))
                ),
                "clamped_non_negative": bool(variant != "bernoulli"),
                "non_negative_domain": bool(variant != "bernoulli"),
                "boundary_threshold": 0.5,
                "is_classifier": True,
                "is_binary": bool(n_classes == 2),
            },
        )
