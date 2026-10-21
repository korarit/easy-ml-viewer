# backend/app/domains/model_engine/adapters/distance/knn.py
"""Phase 22 — K-Nearest Neighbors family adapters (3D expanding radar sphere).

Covers ``KNeighborsClassifier``, ``KNeighborsRegressor``,
``RadiusNeighborsClassifier`` and ``RadiusNeighborsRegressor``.

Blueprint mapping (phase-22-model-knn-radar-sphere.md):

- Radar radius per frame ``R_t = max neighbor distance`` (fixed-K) or
  ``model.radius`` (radius family); geometry shape follows
  ``effective_metric_``: ``sphere`` (L2), ``octahedron`` (L1/manhattan),
  ``cube`` (chebyshev).
- Vote weights: ``uniform`` -> ``1/K``; ``distance`` ->
  ``(1 / max(d, eps)) / sum`` with ``eps = 1e-12`` (edge #2).
- Local contributions (distance waterfall): weighted centroid
  ``bar{x} = sum_k w_k x_(i_k)`` then
  ``phi_j = -|x_j - bar{x}_j|`` (closer to the winning centroid -> ~0).
- Reference cloud: mandatory union of referenced ``_fit_X`` rows (100%
  spider-line endpoints) + seeded background fill capped at 5,000 points;
  rendered client-side with a single ``InstancedMesh`` draw call.

Waterfall / parity identities enforced by tests:

- Classifier ``predict_batch`` == ``model.predict``; regressor likewise.
- Regressor weighted centroid mean == ``predict`` for ``weights='uniform'``
  (and distance-weighted mean == ``predict`` for ``weights='distance'``).

Edge cases handled (blueprint section 7):

1. D-dim vs 3D-projected distance: trace reports full D-dim distances;
   the HUD renders a ``projected_3d_dist`` column client-side.
2. ``d == 0.0`` with ``weights='distance'``: floored at ``1e-12``.
3. Radius with zero neighbors: ``OUTLIER`` telemetry, no crash.
4. Sparse ``_fit_X`` (csr_matrix): only selected rows are densified.
5. Non-euclidean metrics: radar shape switches automatically.
6. Large K (e.g. 200): spider lines capped at top-15 client-side; the
   trace ships the full (K) list plus a ``display_capped`` flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
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
MAX_BACKGROUND_REF_POINTS = 5000
MAX_RADIUS_NEIGHBORS_PER_ROW = 100
SPIDER_DISPLAY_CAP = 15
DIST_EPS = 1e-12

_CLASSIFIER_TYPES = (KNeighborsClassifier, RadiusNeighborsClassifier)
_REGRESSOR_TYPES = (KNeighborsRegressor, RadiusNeighborsRegressor)
_RADIUS_TYPES = (RadiusNeighborsClassifier, RadiusNeighborsRegressor)

_MANHATTAN_ALIASES = {"manhattan", "cityblock", "l1"}


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
    """Map raw rows into the fitted KNN feature space (pipeline-aware)."""
    if hasattr(model_obj, "steps"):
        try:
            raw = X_float32.toarray() if _is_sparse(X_float32) else np.asarray(X_float32)
            return _as_dense_float32(model_obj[:-1].transform(raw))
        except Exception:
            return _as_dense_float32(X_float32)
    return _as_dense_float32(X_float32)


def _fit_row_count(fit_X: Any) -> int:
    try:
        return int(fit_X.shape[0])
    except (AttributeError, TypeError, ValueError):
        return 0


def _fit_dim(fit_X: Any) -> int:
    try:
        return int(fit_X.shape[1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return 0


def _get_fit_rows(fit_X: Any, idx: np.ndarray) -> np.ndarray:
    """Densify only the selected ``_fit_X`` rows (sparse-safe)."""
    idx_arr = np.asarray(idx, dtype=np.int64).ravel()
    if idx_arr.size == 0:
        dim = _fit_dim(fit_X)
        return np.zeros((0, max(dim, 0)), dtype=np.float32)
    if _is_sparse(fit_X):
        try:
            sel = fit_X[idx_arr]
            return np.ascontiguousarray(np.asarray(sel.toarray(), dtype=np.float32))
        except (IndexError, ValueError):
            dim = _fit_dim(fit_X)
            return np.zeros((idx_arr.size, max(dim, 0)), dtype=np.float32)
    arr = np.asarray(fit_X)
    if arr.dtype != np.float32:
        # Index first (cheap view), then cast only the selection.
        try:
            sel = arr[idx_arr]
        except (IndexError, ValueError):
            dim = _fit_dim(fit_X)
            return np.zeros((idx_arr.size, max(dim, 0)), dtype=np.float32)
        return np.ascontiguousarray(np.asarray(sel, dtype=np.float32))
    try:
        return np.ascontiguousarray(np.asarray(arr[idx_arr], dtype=np.float32))
    except (IndexError, ValueError):
        dim = _fit_dim(fit_X)
        return np.zeros((idx_arr.size, max(dim, 0)), dtype=np.float32)


def _resolve_metric(estimator: Any) -> tuple[str, float]:
    raw_metric = getattr(estimator, "effective_metric_", getattr(estimator, "metric", "euclidean"))
    if callable(raw_metric):
        metric = "custom"
    else:
        try:
            metric = str(raw_metric)
        except Exception:
            metric = "euclidean"
    try:
        p_val = float(getattr(estimator, "p", 2))
    except (TypeError, ValueError):
        p_val = 2.0
    if not np.isfinite(p_val):
        p_val = 2.0
    return metric, float(p_val)


def _radar_shape(metric: str, p_val: float) -> str:
    m = metric.lower()
    if m in _MANHATTAN_ALIASES or p_val == 1.0:
        return "octahedron"
    if m == "chebyshev":
        return "cube"
    return "sphere"


def _resolve_weights_mode(estimator: Any) -> str:
    raw = getattr(estimator, "weights", "uniform")
    if callable(raw):
        return "custom"
    s = str(raw) if raw is not None else "uniform"
    return s if s in ("uniform", "distance") else s


def _vote_weights(dists_row: np.ndarray, mode: str) -> np.ndarray:
    k = int(dists_row.size)
    if k == 0:
        return np.zeros((0,), dtype=np.float64)
    d = np.asarray(dists_row, dtype=np.float64)
    d = np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
    if mode == "distance":
        inv = 1.0 / np.maximum(d, DIST_EPS)
        total = float(np.sum(inv))
        if not np.isfinite(total) or total <= 0.0:
            return np.full((k,), 1.0 / max(k, 1), dtype=np.float64)
        return (inv / total).astype(np.float64)
    return np.full((k,), 1.0 / max(k, 1), dtype=np.float64)


def _downsample_reference_cloud(
    fit_X: Any,
    fit_targets: np.ndarray,
    mandatory_indices: np.ndarray,
    max_points: int = MAX_BACKGROUND_REF_POINTS,
) -> dict[str, Any]:
    """Mandatory union (100% spider endpoints) + seeded background fill."""
    n_total = _fit_row_count(fit_X)
    if n_total <= 0:
        return {
            "original_indices": [],
            "coordinates": [],
            "targets": [],
            "index_map": {},
            "total_reference_points": 0,
            "mandatory_count": 0,
        }
    try:
        mandatory_uniq = np.unique(np.asarray(mandatory_indices, dtype=np.int64).ravel())
    except (TypeError, ValueError):
        mandatory_uniq = np.zeros((0,), dtype=np.int64)
    # Drop out-of-range ids defensively (stale caches, capped radius lists).
    mandatory_uniq = mandatory_uniq[(mandatory_uniq >= 0) & (mandatory_uniq < n_total)]
    if n_total <= max_points:
        selected = np.arange(n_total, dtype=np.int64)
    else:
        rem_budget = max(0, int(max_points) - int(mandatory_uniq.size))
        if rem_budget <= 0:
            # Over-budget mandatory set (huge radius): keep a deterministic
            # seeded subsample so the payload never exceeds the cap.
            rng = np.random.default_rng(seed=42)
            selected = np.sort(
                rng.choice(mandatory_uniq, size=int(max_points), replace=False).astype(np.int64)
            )
        else:
            mask = np.ones((n_total,), dtype=bool)
            mask[mandatory_uniq] = False
            pool = np.flatnonzero(mask)
            rng = np.random.default_rng(seed=42)
            take = min(rem_budget, int(pool.size))
            bg = rng.choice(pool, size=take, replace=False) if take > 0 else np.zeros((0,), dtype=np.int64)
            selected = np.sort(np.concatenate([mandatory_uniq, np.asarray(bg, dtype=np.int64)]))
    coords = _get_fit_rows(fit_X, selected)
    try:
        flat_targets = np.asarray(fit_targets).ravel()
        targets = [flat_targets[int(i)] for i in selected.tolist()]
        py_targets: list[Any] = []
        for v in targets:
            if isinstance(v, (np.integer, bool, np.bool_)):
                py_targets.append(int(v))
            elif isinstance(v, (np.floating, float)):
                f = float(v)
                py_targets.append(int(f) if f.is_integer() else f)
            elif isinstance(v, np.ndarray):
                py_targets.append(v.tolist())
            else:
                try:
                    py_targets.append(v.item() if hasattr(v, "item") else v)
                except (ValueError, AttributeError):
                    py_targets.append(v)
    except (IndexError, TypeError, ValueError):
        py_targets = [None for _ in selected.tolist()]
    idx_map = {int(orig): int(loc) for loc, orig in enumerate(selected.tolist())}
    return {
        "original_indices": [int(v) for v in selected.tolist()],
        "coordinates": np.asarray(coords, dtype=np.float32).tolist(),
        "targets": py_targets,
        "index_map": idx_map,
        "total_reference_points": int(n_total),
        "mandatory_count": int(mandatory_uniq.size),
    }


def _fit_targets_for_cloud(estimator: Any) -> np.ndarray:
    raw_y = getattr(estimator, "_y", None)
    if raw_y is None:
        return np.zeros((_fit_row_count(getattr(estimator, "_fit_X", [])),), dtype=np.float32)
    arr = np.asarray(raw_y)
    if arr.ndim == 1:
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            try:
                labels = np.asarray(classes)[np.asarray(arr).ravel().astype(np.int64)]
                return labels
            except (IndexError, ValueError, TypeError):
                return arr
        return arr
    # Multi-output: cloud colors use the first target column.
    return np.asarray(arr[:, 0]).ravel()


def _coerce_pred(value: Any) -> float | int | str:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return int(f) if f.is_integer() else f
    if isinstance(value, np.ndarray):
        return _coerce_pred(value.tolist())
    if isinstance(value, (list, tuple)):
        return _coerce_pred(value[0]) if value else "OUTLIER_NO_NEIGHBORS"
    return value  # type: ignore[return-value]


def _gather_query_matrix(
    X_q: np.ndarray, X_native: np.ndarray, row_indices: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Build the (R, D) query matrix in ``row_indices`` order + valid ids."""
    n_batch = int(X_q.shape[0])
    vecs: list[np.ndarray] = []
    valid: list[int] = []
    for row_id in row_indices:
        if row_id < 0 or row_id >= n_batch:
            continue
        vecs.append(np.asarray(X_q[row_id], dtype=np.float32).ravel())
        valid.append(int(row_id))
    if not vecs:
        dim = int(X_q.shape[1]) if X_q.ndim == 2 else 0
        return np.zeros((0, max(dim, 0)), dtype=np.float32), []
    width = max(int(v.size) for v in vecs)
    mat = np.zeros((len(vecs), width), dtype=np.float32)
    for i, v in enumerate(vecs):
        mat[i, : v.size] = v
    return np.ascontiguousarray(mat), valid


def _fixed_k_neighbors(
    estimator: Any, Q: np.ndarray, k_requested: int, n_total: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-K lookup with a brute-force fallback when K > n_total."""
    if Q.shape[0] == 0 or n_total <= 0:
        dim_k = max(int(k_requested), 0)
        return np.zeros((Q.shape[0], dim_k), dtype=np.float64), np.zeros(
            (Q.shape[0], dim_k), dtype=np.int64
        )
    try:
        dists, idx = estimator.kneighbors(Q, return_distance=True)
        return np.asarray(dists, dtype=np.float64), np.asarray(idx, dtype=np.int64)
    except (ValueError, AttributeError):
        k_eff = max(min(int(k_requested), int(n_total)), 1)
        fit_X = getattr(estimator, "_fit_X", None)
        ref = _get_fit_rows(fit_X, np.arange(n_total, dtype=np.int64)).astype(np.float64)
        q64 = np.asarray(Q, dtype=np.float64)
        # Chunked brute force keeps peak RAM flat for large reference sets.
        dists_out = np.zeros((q64.shape[0], k_eff), dtype=np.float64)
        idx_out = np.zeros((q64.shape[0], k_eff), dtype=np.int64)
        chunk = 512
        for s in range(0, q64.shape[0], chunk):
            e = min(s + chunk, q64.shape[0])
            diff = q64[s:e, None, :] - ref[None, :, :]
            d = np.sqrt(np.sum(diff * diff, axis=2))
            part = np.argpartition(d, kth=min(k_eff - 1, d.shape[1] - 1), axis=1)[:, :k_eff]
            order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1)
            sel = np.take_along_axis(part, order, axis=1)
            idx_out[s:e] = sel
            dists_out[s:e] = np.take_along_axis(d, sel, axis=1)
        return dists_out, idx_out


def _radius_neighbors(
    estimator: Any, Q: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if Q.shape[0] == 0:
        return [], []
    try:
        dists_obj, idx_obj = estimator.radius_neighbors(Q, return_distance=True)
    except (ValueError, AttributeError):
        return [np.zeros((0,)) for _ in range(Q.shape[0])], [
            np.zeros((0,), dtype=np.int64) for _ in range(Q.shape[0])
        ]
    dists_list = [np.asarray(d, dtype=np.float64).ravel() for d in list(dists_obj)]
    idx_list = [np.asarray(i, dtype=np.int64).ravel() for i in list(idx_obj)]
    # radius_neighbors already returns distance-sorted rows; cap the payload.
    capped_d = [d[:MAX_RADIUS_NEIGHBORS_PER_ROW] for d in dists_list]
    capped_i = [i[:MAX_RADIUS_NEIGHBORS_PER_ROW] for i in idx_list]
    return capped_d, capped_i


def _common_overlays(estimator: Any) -> dict[str, Any]:
    metric, p_val = _resolve_metric(estimator)
    return {
        "n_neighbors": int(getattr(estimator, "n_neighbors", 0) or 0),
        "radius": float(getattr(estimator, "radius", 0.0) or 0.0)
        if isinstance(estimator, _RADIUS_TYPES)
        else None,
        "weights": _resolve_weights_mode(estimator),
        "algorithm": str(getattr(estimator, "algorithm", "auto")),
        "leaf_size": int(getattr(estimator, "leaf_size", 30) or 30),
        "effective_metric": str(metric),
        "p": float(p_val),
        "radar_geometry_shape": _radar_shape(str(metric), float(p_val)),
        "total_reference_points": int(_fit_row_count(getattr(estimator, "_fit_X", []))),
        "is_radius": bool(isinstance(estimator, _RADIUS_TYPES)),
    }


@ModelRegistry.register("knn_classifier")
class KnnClassifierAdapter(BaseModelAdapter):
    display_name = "KNN Classifier (Radar Sphere)"
    family_name = "Distance Models"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.VORONOI_KNN_PROXIMITY

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _CLASSIFIER_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        # Instance-based model: no static per-feature weights.
        return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        try:
            preds = np.asarray(model_obj.predict(X_in)).ravel()
        except ValueError:
            # Radius outlier batch: sklearn raises when a row has no
            # neighbors and no outlier_label is set.
            preds = np.asarray([0 for _ in range(X_in.shape[0])])
            try:
                single: list[Any] = []
                for r in range(X_in.shape[0]):
                    try:
                        single.append(model_obj.predict(X_in[r : r + 1])[0])
                    except ValueError:
                        single.append(-1)
                preds = np.asarray(single)
            except Exception:
                pass
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
        py_preds: list[float | int | str] = [_coerce_pred(v) for v in preds.tolist()]
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
        fit_X = getattr(estimator, "_fit_X", None)
        n_total = _fit_row_count(fit_X)
        fit_targets = _fit_targets_for_cloud(estimator)
        metric, p_val = _resolve_metric(estimator)
        weights_mode = _resolve_weights_mode(estimator)
        shape = _radar_shape(str(metric), float(p_val))
        is_radius = bool(isinstance(estimator, _RADIUS_TYPES))
        k_requested = int(getattr(estimator, "n_neighbors", 5) or 5)

        X_native = _as_dense_float32(X_float32)
        X_q = _transform_for_query(model_obj, X_float32)
        # Align query width with the fitted space (pipeline encoders).
        fit_dim = _fit_dim(fit_X)
        if fit_dim > 0 and X_q.shape[1] != fit_dim:
            width = max(fit_dim, int(X_q.shape[1]))
            padded = np.zeros((X_q.shape[0], width), dtype=np.float32)
            take = min(width, int(X_q.shape[1]))
            padded[:, :take] = X_q[:, :take]
            X_q = padded[:, :fit_dim] if width >= fit_dim else padded

        Q, valid_ids = _gather_query_matrix(X_q, X_native, list(row_indices))
        try:
            native_preds = list(np.asarray(model_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError):
            native_preds = []
            for row_id in range(X_native.shape[0]):
                try:
                    native_preds.append(model_obj.predict(X_native[row_id : row_id + 1])[0])
                except (ValueError, AttributeError):
                    native_preds.append("OUTLIER_NO_NEIGHBORS")
        try:
            native_probas = np.asarray(model_obj.predict_proba(X_native), dtype=np.float64)
        except (AttributeError, NotImplementedError, ValueError):
            native_probas = None
        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )

        if is_radius:
            dists_rows, idx_rows = _radius_neighbors(estimator, Q)
        else:
            if Q.shape[0]:
                dists_all, idx_all = _fixed_k_neighbors(estimator, Q, k_requested, n_total)
                dists_rows = [np.asarray(dists_all[i]).ravel() for i in range(dists_all.shape[0])]
                idx_rows = [np.asarray(idx_all[i], dtype=np.int64).ravel() for i in range(idx_all.shape[0])]
            else:
                dists_rows, idx_rows = [], []

        # Mandatory union guarantees every spider-line endpoint exists in 3D.
        if idx_rows:
            try:
                mandatory = np.unique(np.concatenate([np.asarray(i).ravel() for i in idx_rows if np.asarray(i).size]))
            except ValueError:
                mandatory = np.zeros((0,), dtype=np.int64)
        else:
            mandatory = np.zeros((0,), dtype=np.int64)
        ref_cloud = _downsample_reference_cloud(fit_X, fit_targets, mandatory)
        index_map: dict[str, int] = {str(k): int(v) for k, v in ref_cloud.get("index_map", {}).items()}

        try:
            outlier_label = getattr(estimator, "outlier_label_", getattr(estimator, "outlier_label", None))
        except Exception:
            outlier_label = None

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(valid_ids):
            dists_row = np.asarray(dists_rows[pos], dtype=np.float64).ravel() if pos < len(dists_rows) else np.zeros((0,))
            idx_row = np.asarray(idx_rows[pos], dtype=np.int64).ravel() if pos < len(idx_rows) else np.zeros((0,), dtype=np.int64)
            k_eff = int(idx_row.size)
            w_row = _vote_weights(dists_row, weights_mode if weights_mode in ("uniform", "distance") else "uniform")
            try:
                neigh_feat = _get_fit_rows(fit_X, idx_row).astype(np.float64) if k_eff else np.zeros((0, max(fit_dim, 0)))
            except (ValueError, TypeError):
                neigh_feat = np.zeros((0, max(fit_dim, 0)))
            q_vec = np.asarray(Q[pos], dtype=np.float64).ravel() if Q.shape[0] > pos else np.zeros((max(fit_dim, 0),))
            if k_eff and neigh_feat.shape[0] == k_eff:
                if w_row.size == k_eff and float(np.sum(w_row)) > 0:
                    centroid = np.sum(neigh_feat * w_row[:, None], axis=0)
                else:
                    centroid = np.mean(neigh_feat, axis=0)
                width_c = min(int(centroid.size), int(q_vec.size))
                contrib_vec = -np.abs(q_vec[:width_c] - centroid[:width_c])
            else:
                contrib_vec = np.zeros((min(int(q_vec.size), max(len(feature_names), 0)),), dtype=np.float64)
            names = feature_names or [f"feature_{i}" for i in range(int(contrib_vec.size))]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contrib_vec[i])
                for i in range(min(int(contrib_vec.size), len(names) if names else int(contrib_vec.size)))
            }
            if not contrib_map and names:
                contrib_map = {names[0]: 0.0}
            try:
                labels_row = [fit_targets[int(i)] for i in idx_row.tolist()] if k_eff else []
            except (IndexError, ValueError, TypeError):
                labels_row = []
            py_labels: list[Any] = []
            for v in labels_row:
                if isinstance(v, (np.integer, bool, np.bool_)):
                    py_labels.append(int(v))
                elif isinstance(v, (np.floating, float)):
                    f = float(v)
                    py_labels.append(int(f) if f.is_integer() else f)
                else:
                    try:
                        py_labels.append(v.item() if hasattr(v, "item") else v)
                    except (ValueError, AttributeError):
                        py_labels.append(v)
            # Vote tally (counts + weight shares) for the HUD table.
            tally: dict[str, dict[str, float]] = {}
            for lab, w in zip(py_labels, w_row.tolist() if w_row.size else []):
                key = str(lab)
                slot = tally.setdefault(key, {"count": 0.0, "weight": 0.0})
                slot["count"] += 1.0
                slot["weight"] += float(w)
            vote_list = [
                {"label": k, "count": int(v["count"]), "weight_share": float(v["weight"])}
                for k, v in sorted(tally.items(), key=lambda kv: (-kv[1]["weight"], str(kv[0])))
            ]
            radar_radius = float(np.max(dists_row)) if dists_row.size else float(getattr(estimator, "radius", 0.0) or 0.0)
            if not np.isfinite(radar_radius):
                radar_radius = 0.0
            is_outlier = bool(is_radius and k_eff == 0)
            try:
                final_pred: float | int | str = _coerce_pred(native_preds[row_id]) if row_id < len(native_preds) else _coerce_pred(classes[0] if classes else 0)
            except (IndexError, TypeError, ValueError):
                final_pred = outlier_label if outlier_label is not None else "OUTLIER_NO_NEIGHBORS"
            if is_outlier and (final_pred is None or (isinstance(final_pred, float) and not np.isfinite(final_pred))):
                final_pred = outlier_label if outlier_label is not None else "OUTLIER_NO_NEIGHBORS"
            proba_row: list[float] = []
            if native_probas is not None and row_id < native_probas.shape[0]:
                try:
                    proba_row = [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(native_probas[row_id]).ravel()]
                except (IndexError, TypeError, ValueError):
                    proba_row = []
            local_idx = [index_map.get(str(int(i)), -1) for i in idx_row.tolist()]
            extra: dict[str, Any] = {
                "n_neighbors": int(k_requested),
                "n_neighbors_effective": int(k_eff),
                "radius": float(getattr(estimator, "radius", 0.0) or 0.0) if is_radius else None,
                "weights": weights_mode,
                "algorithm": str(getattr(estimator, "algorithm", "auto")),
                "leaf_size": int(getattr(estimator, "leaf_size", 30) or 30),
                "effective_metric": str(metric),
                "p": float(p_val),
                "radar_geometry_shape": shape,
                "radar_radius": float(radar_radius),
                "neighbor_distances": [float(v) if np.isfinite(v) else 0.0 for v in dists_row.tolist()],
                "neighbor_original_indices": [int(v) for v in idx_row.tolist()],
                "neighbor_local_indices": [int(v) for v in local_idx],
                "neighbor_labels": py_labels,
                "neighbor_weights": [float(v) for v in (w_row.tolist() if w_row.size else [])],
                "vote_tally": vote_list,
                "display_capped": bool(k_eff > SPIDER_DISPLAY_CAP),
                "spider_display_cap": int(SPIDER_DISPLAY_CAP),
                "is_outlier": bool(is_outlier),
                "outlier_label": outlier_label if outlier_label is not None else None,
                "is_radius": bool(is_radius),
                "is_classifier": True,
                "classes": [str(c) for c in classes] if classes else [],
                "class_labels": classes if classes else [],
                "has_native_proba": bool(native_probas is not None),
                "probabilities": proba_row,
                "predicted_class": final_pred,
                "total_reference_points": int(n_total),
                "mandatory_count": int(ref_cloud.get("mandatory_count", 0)),
                "reference_cloud": ref_cloud,
                "target_idx": int(target_idx),
                "target_class_idx": int(target_class_idx) if target_class_idx is not None else int(target_idx),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=0.0,
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
        metric, p_val = _resolve_metric(estimator)
        shape = _radar_shape(str(metric), float(p_val))
        raw_classes = getattr(estimator, "classes_", None)
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
        n_classes = len(classes) if classes else 0
        if n_classes == 2:
            # Binary convention (matches SGD/logistic adapters): the surface
            # always renders the positive-class probability P(class 1).
            resolved = 1
        elif target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), max(n_classes - 1, 0)))
        elif n_classes:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        else:
            resolved = int(max(int(target_idx), 0))
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
            if hasattr(model_obj, "predict_proba"):
                proba = np.asarray(model_obj.predict_proba(grid), dtype=np.float64)
                col = min(resolved, proba.shape[1] - 1) if proba.ndim == 2 else 0
                z_flat = proba[:, col] if proba.ndim == 2 else proba.ravel()
            else:
                z_flat = np.asarray(model_obj.predict(grid), dtype=np.float64).ravel()
        except (ValueError, AttributeError):
            # Radius outliers on the grid: fall back to 0.5 (max uncertainty).
            z_flat = np.full((n_points,), 0.5, dtype=np.float64)
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in z_flat.tolist()],
            extra_overlays={
                **_common_overlays(estimator),
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes] if classes else [],
                "boundary_threshold": 0.5,
                "is_classifier": True,
            },
        )


@ModelRegistry.register("knn_regressor")
class KnnRegressorAdapter(BaseModelAdapter):
    display_name = "KNN Regressor (Radar Sphere)"
    family_name = "Distance Models"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.VORONOI_KNN_PROXIMITY

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(estimator, _REGRESSOR_TYPES)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X_in = _as_dense_float32(X_float32)
        try:
            preds = np.asarray(model_obj.predict(X_in))
        except ValueError:
            preds = np.full((X_in.shape[0],), np.nan, dtype=np.float64)
            for r in range(X_in.shape[0]):
                try:
                    preds[r] = float(np.asarray(model_obj.predict(X_in[r : r + 1])).ravel()[0])
                except (ValueError, TypeError):
                    preds[r] = float("nan")
        flat = np.asarray(preds)
        if flat.ndim == 2 and flat.shape[1] >= 1:
            flat = flat[:, 0]
        flat = np.asarray(flat).ravel()
        return PredictionBatchResult(
            predictions=[float(v) if np.isfinite(v) else 0.0 for v in flat.tolist()]
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
        fit_X = getattr(estimator, "_fit_X", None)
        n_total = _fit_row_count(fit_X)
        fit_targets = _fit_targets_for_cloud(estimator)
        raw_y = np.asarray(getattr(estimator, "_y", []))
        multioutput = bool(raw_y.ndim == 2 and raw_y.shape[1] > 1)
        n_targets = int(raw_y.shape[1]) if multioutput else 1
        if target_class_idx is not None and multioutput:
            t_idx = int(min(max(int(target_class_idx), 0), n_targets - 1))
        else:
            t_idx = int(min(max(int(target_idx), 0), n_targets - 1)) if multioutput else 0
        metric, p_val = _resolve_metric(estimator)
        weights_mode = _resolve_weights_mode(estimator)
        shape = _radar_shape(str(metric), float(p_val))
        is_radius = bool(isinstance(estimator, _RADIUS_TYPES))
        k_requested = int(getattr(estimator, "n_neighbors", 5) or 5)

        X_native = _as_dense_float32(X_float32)
        X_q = _transform_for_query(model_obj, X_float32)
        fit_dim = _fit_dim(fit_X)
        if fit_dim > 0 and X_q.shape[1] != fit_dim:
            width = max(fit_dim, int(X_q.shape[1]))
            padded = np.zeros((X_q.shape[0], width), dtype=np.float32)
            take = min(width, int(X_q.shape[1]))
            padded[:, :take] = X_q[:, :take]
            X_q = padded[:, :fit_dim] if width >= fit_dim else padded

        Q, valid_ids = _gather_query_matrix(X_q, X_native, list(row_indices))
        try:
            native_all = np.asarray(model_obj.predict(X_native))
        except (ValueError, AttributeError):
            native_all = np.full((X_native.shape[0],), np.nan, dtype=np.float64)
            for row_id in range(X_native.shape[0]):
                try:
                    native_all[row_id] = float(np.asarray(model_obj.predict(X_native[row_id : row_id + 1])).ravel()[0])
                except (ValueError, TypeError, AttributeError):
                    native_all[row_id] = float("nan")

        if is_radius:
            dists_rows, idx_rows = _radius_neighbors(estimator, Q)
        else:
            if Q.shape[0]:
                dists_all, idx_all = _fixed_k_neighbors(estimator, Q, k_requested, n_total)
                dists_rows = [np.asarray(dists_all[i]).ravel() for i in range(dists_all.shape[0])]
                idx_rows = [np.asarray(idx_all[i], dtype=np.int64).ravel() for i in range(idx_all.shape[0])]
            else:
                dists_rows, idx_rows = [], []

        if idx_rows:
            try:
                mandatory = np.unique(np.concatenate([np.asarray(i).ravel() for i in idx_rows if np.asarray(i).size]))
            except ValueError:
                mandatory = np.zeros((0,), dtype=np.int64)
        else:
            mandatory = np.zeros((0,), dtype=np.int64)
        ref_cloud = _downsample_reference_cloud(fit_X, fit_targets, mandatory)
        index_map: dict[str, int] = {str(k): int(v) for k, v in ref_cloud.get("index_map", {}).items()}

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(valid_ids):
            dists_row = np.asarray(dists_rows[pos], dtype=np.float64).ravel() if pos < len(dists_rows) else np.zeros((0,))
            idx_row = np.asarray(idx_rows[pos], dtype=np.int64).ravel() if pos < len(idx_rows) else np.zeros((0,), dtype=np.int64)
            k_eff = int(idx_row.size)
            w_row = _vote_weights(dists_row, weights_mode if weights_mode in ("uniform", "distance") else "uniform")
            try:
                neigh_feat = _get_fit_rows(fit_X, idx_row).astype(np.float64) if k_eff else np.zeros((0, max(fit_dim, 0)))
            except (ValueError, TypeError):
                neigh_feat = np.zeros((0, max(fit_dim, 0)))
            q_vec = np.asarray(Q[pos], dtype=np.float64).ravel() if Q.shape[0] > pos else np.zeros((max(fit_dim, 0),))
            if k_eff and neigh_feat.shape[0] == k_eff:
                if w_row.size == k_eff and float(np.sum(w_row)) > 0:
                    centroid = np.sum(neigh_feat * w_row[:, None], axis=0)
                else:
                    centroid = np.mean(neigh_feat, axis=0)
                width_c = min(int(centroid.size), int(q_vec.size))
                contrib_vec = -np.abs(q_vec[:width_c] - centroid[:width_c])
            else:
                contrib_vec = np.zeros((min(int(q_vec.size), max(len(feature_names), 0)),), dtype=np.float64)
            names = feature_names or [f"feature_{i}" for i in range(int(contrib_vec.size))]
            contrib_map = {
                (names[i] if i < len(names) else f"feature_{i}"): float(contrib_vec[i])
                for i in range(min(int(contrib_vec.size), len(names) if names else int(contrib_vec.size)))
            }
            if not contrib_map and names:
                contrib_map = {names[0]: 0.0}
            try:
                if multioutput:
                    neigh_targets = np.asarray(raw_y)[idx_row, t_idx].astype(np.float64).ravel() if k_eff else np.zeros((0,))
                else:
                    neigh_targets = np.asarray(fit_targets)[idx_row].astype(np.float64).ravel() if k_eff else np.zeros((0,))
                    neigh_targets = np.where(np.isfinite(neigh_targets), neigh_targets, 0.0)
            except (IndexError, ValueError, TypeError):
                neigh_targets = np.zeros((max(k_eff, 0),), dtype=np.float64)
            if neigh_targets.size and w_row.size == neigh_targets.size:
                weighted_mean = float(np.sum(neigh_targets * w_row))
            elif neigh_targets.size:
                weighted_mean = float(np.mean(neigh_targets))
            else:
                weighted_mean = 0.0
            radar_radius = float(np.max(dists_row)) if dists_row.size else float(getattr(estimator, "radius", 0.0) or 0.0)
            if not np.isfinite(radar_radius):
                radar_radius = 0.0
            is_outlier = bool(is_radius and k_eff == 0)
            try:
                raw_pred = native_all[row_id] if row_id < np.asarray(native_all).shape[0] else weighted_mean
                if isinstance(raw_pred, np.ndarray):
                    raw_pred = float(np.asarray(raw_pred).ravel()[min(t_idx, int(np.asarray(raw_pred).size) - 1)])
                final_pred = float(raw_pred) if np.isfinite(float(raw_pred)) else float(weighted_mean)
            except (IndexError, TypeError, ValueError):
                final_pred = float(weighted_mean) if np.isfinite(weighted_mean) else 0.0
            local_idx = [index_map.get(str(int(i)), -1) for i in idx_row.tolist()]
            extra: dict[str, Any] = {
                "n_neighbors": int(k_requested),
                "n_neighbors_effective": int(k_eff),
                "radius": float(getattr(estimator, "radius", 0.0) or 0.0) if is_radius else None,
                "weights": weights_mode,
                "algorithm": str(getattr(estimator, "algorithm", "auto")),
                "leaf_size": int(getattr(estimator, "leaf_size", 30) or 30),
                "effective_metric": str(metric),
                "p": float(p_val),
                "radar_geometry_shape": shape,
                "radar_radius": float(radar_radius),
                "neighbor_distances": [float(v) if np.isfinite(v) else 0.0 for v in dists_row.tolist()],
                "neighbor_original_indices": [int(v) for v in idx_row.tolist()],
                "neighbor_local_indices": [int(v) for v in local_idx],
                "neighbor_targets": [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(neigh_targets).ravel().tolist()],
                "neighbor_weights": [float(v) for v in (w_row.tolist() if w_row.size else [])],
                "weighted_neighbor_mean": float(weighted_mean) if np.isfinite(weighted_mean) else 0.0,
                "display_capped": bool(k_eff > SPIDER_DISPLAY_CAP),
                "spider_display_cap": int(SPIDER_DISPLAY_CAP),
                "is_outlier": bool(is_outlier),
                "is_radius": bool(is_radius),
                "is_classifier": False,
                "is_pseudo_proba": False,
                "target_idx": int(t_idx),
                "target_class_idx": int(t_idx),
                "active_target_idx": int(t_idx),
                "n_targets": int(n_targets),
                "is_multioutput": bool(multioutput),
                "total_reference_points": int(n_total),
                "mandatory_count": int(ref_cloud.get("mandatory_count", 0)),
                "reference_cloud": ref_cloud,
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=0.0,
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
        metric, p_val = _resolve_metric(estimator)
        shape = _radar_shape(str(metric), float(p_val))
        raw_y = np.asarray(getattr(estimator, "_y", []))
        multioutput = bool(raw_y.ndim == 2 and raw_y.shape[1] > 1)
        n_targets = int(raw_y.shape[1]) if multioutput else 1
        if target_class_idx is not None and multioutput:
            resolved = int(min(max(int(target_class_idx), 0), n_targets - 1))
        else:
            resolved = int(min(max(int(target_idx), 0), n_targets - 1)) if multioutput else 0
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
            raw = np.asarray(model_obj.predict(grid))
            if raw.ndim == 2 and raw.shape[1] > 1:
                z_flat = np.asarray(raw[:, min(resolved, raw.shape[1] - 1)], dtype=np.float64).ravel()
            else:
                z_flat = np.asarray(raw, dtype=np.float64).ravel()
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
                **_common_overlays(estimator),
                "target_idx": int(resolved),
                "resolved_target_idx": int(resolved),
                "active_target_idx": int(resolved),
                "n_targets": int(n_targets),
                "is_multioutput": bool(multioutput),
                "boundary_threshold": None,
                "is_classifier": False,
            },
        )
