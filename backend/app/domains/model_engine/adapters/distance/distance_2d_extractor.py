"""Phase 43 — 2D radar & kernel geometry for Distance & Kernel models (KNN & SVM).

Complements the universal Marching-Squares contour engine (Phase 41) with the
*exact* per-frame / per-model overlays whose data the dense grid cannot
provide:

- **KNN**: the expanding 2D radar circle
  ``R_{t,2D} = max_k ||(x_{t,1}, x_{t,2}) - (x_{(k),1}, x_{(k),2})||`` around
  the active query row, spider-line endpoints at the true ``_fit_X`` reference
  rows, vote weights (``uniform`` → ``1/K``; ``distance`` → ``1/max(d, eps)``
  with ``eps = 1e-12`` so ``d = 0`` never divides by zero), and the vote tally.
- **SVM**: support-vector halos (``support_vectors_`` + ``dual_coef_`` parity)
  with halo radii ``r = 4 + 6·|α|/C`` and bound/free status (``|α| ≈ C`` means
  the point violates the margin), plus kernel / gamma / C metadata. The curved
  ``f = 0`` boundary and ``f = ±1`` margins themselves come from the contour
  score-field path (see ``contour_2d_generator``).

Conventions (shared with Phase 22 / Phase 23 adapters):

- Coordinates are **raw feature units**; the contour orchestrator maps them to
  world coords via ``WorldCoordinateScaler``. Pure NumPy, no DataFrame
  retention (memory budget §5.1).
- Pipeline-wrapped estimators live in *transformed* space, so raw axis
  indices cannot address ``_fit_X`` columns exactly — the extractor returns
  ``None`` and the caller falls back to the sampled contour.
- Spider payloads are capped at ``MAX_SPIDER_NEIGHBORS`` (100) rows;
  SV halos at ``MAX_SV_HALOS`` (2000) ranked by ``|dual|`` — both with
  explicit truncation flags so the client never guesses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

DIST_EPS = 1e-12
MAX_SPIDER_NEIGHBORS = 100
MAX_SV_HALOS = 2000
BOUND_TOL = 1e-6

_KNN_FIXED_NAMES = {"KNeighborsClassifier", "KNeighborsRegressor"}
_KNN_RADIUS_NAMES = {"RadiusNeighborsClassifier", "RadiusNeighborsRegressor"}
_KNN_NAMES = _KNN_FIXED_NAMES | _KNN_RADIUS_NAMES
_SVM_SV_NAMES = {"SVC", "NuSVC", "SVR", "NuSVR", "OneClassSVM"}
_SVM_LINEAR_NAMES = {"LinearSVC", "LinearSVR"}
_SVM_NAMES = _SVM_SV_NAMES | _SVM_LINEAR_NAMES


class KnnNeighbor2D(BaseModel):
    """One spider-line endpoint (mirrors the frontend ``KnnNeighbor2D``)."""

    ref_index: int = Field(description="Training reference row in _fit_X")
    x: float = Field(description="Raw X-axis coordinate of the reference row")
    y: float = Field(description="Raw Y-axis coordinate of the reference row")
    distance_full: float = Field(description="Full D-dim neighbor distance")
    distance_2d: float = Field(description="Planar distance on the (X, Y) axes")
    vote_weight: float = Field(description="Normalized vote weight (sums to 1)")
    rank: int = Field(description="1-based closeness rank (unique even on overlap)")
    label: Any = Field(default=None, description="Reference target (classifier) ")


class KnnRadarState2D(BaseModel):
    model_type: str = ""
    k: int = 0
    weights_mode: str = "uniform"
    metric: str = "euclidean"
    is_radius_model: bool = False
    radius: float | None = None
    outlier: bool = Field(default=False, description="Radius model returned zero neighbors")
    query_2d: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    radar_radius_2d: float = 0.0
    neighbors_2d: list[KnnNeighbor2D] = Field(default_factory=list)
    votes: dict[str, float] = Field(default_factory=dict)
    predicted: Any = None
    display_capped: bool = False
    total_neighbors: int = 0


class SvmHalo2D(BaseModel):
    train_index: int = 0
    x: float = 0.0
    y: float = 0.0
    dual_coef: float = 0.0
    halo_scale: float = Field(default=0.0, description="|α|/C clamped to [0, 1] (0.5 when C unknown)")
    status: str = Field(default="support", description="on_margin | violates_margin | support")


class SvmState2D(BaseModel):
    model_type: str = ""
    kernel: str = "rbf"
    gamma: float | None = None
    c: float | None = None
    degree: int = 3
    coef0: float = 0.0
    is_regressor: bool = False
    is_binary: bool = True
    n_classes: int = 0
    n_support_total: int = 0
    is_linear_no_sv: bool = False
    support_vectors_2d: list[SvmHalo2D] = Field(default_factory=list)
    dual_coefs: list[float] = Field(default_factory=list, description="Resolved dual row (|α| ranking source)")
    halos_truncated: bool = False


def _is_pipeline(model_obj: Any) -> bool:
    return isinstance(getattr(model_obj, "steps", None), list) and bool(model_obj.steps)


def _unwrap(model_obj: Any) -> Any:
    steps = getattr(model_obj, "steps", None)
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, (tuple, list)) and len(last) == 2:
            return last[1]
        return last
    return getattr(model_obj, "_final_estimator", model_obj)


def _resolve_query_space(
    model_obj: Any, est: Any, query: np.ndarray
) -> tuple[np.ndarray, Any, bool]:
    """Return ``(query_in_fitted_space, neighbor_source, is_pipeline)``.

    Bare models query ``est`` directly. Scaler-style pipelines (dims
    preserved) are supported by transforming the query into fitted space;
    anything that changes dimensionality (or lacks ``inverse_transform``)
    raises so the caller can honestly return ``None``.
    """
    q = np.ascontiguousarray(np.asarray(query, dtype=np.float64).ravel())
    if not _is_pipeline(model_obj):
        return q, est, False
    pre = model_obj[:-1]
    try:
        q_trans = np.asarray(pre.transform(q.reshape(1, -1)), dtype=np.float64).ravel()
    except Exception as exc:
        raise ValueError(f"pipeline transform failed: {exc}") from exc
    if q_trans.size != q.size:
        # Dimensionality-changing preprocessing (PCA, encoders, selectors)
        # has no exact per-axis raw mapping.
        raise ValueError("pipeline changes feature dimensionality — no exact 2D mapping")
    fit_X = getattr(est, "_fit_X", None)
    try:
        dim = int(fit_X.shape[1]) if fit_X is not None else -1
    except (AttributeError, TypeError, ValueError, IndexError):
        dim = -1
    if q_trans.size != dim:
        raise ValueError("pipeline changes feature dimensionality — no exact 2D mapping")
    try:
        back = np.asarray(pre.inverse_transform(q_trans.reshape(1, -1)), dtype=np.float64).ravel()
    except Exception as exc:
        raise ValueError(f"pipeline has no inverse mapping: {exc}") from exc
    if back.size != q.size or not np.all(np.isfinite(back)):
        raise ValueError("pipeline inverse mapping is not faithful")
    # Lossy round-trips (PCA, feature selectors) have no *exact* raw mapping.
    if not np.allclose(back, q, atol=1e-6, rtol=1e-6):
        raise ValueError("pipeline round-trip is lossy — no exact raw mapping")
    return q_trans, est, True


def _to_raw_space(model_obj: Any, rows_trans: np.ndarray, is_pipeline: bool) -> np.ndarray:
    if not is_pipeline:
        return np.ascontiguousarray(np.asarray(rows_trans, dtype=np.float64))
    try:
        back = np.asarray(model_obj[:-1].inverse_transform(np.asarray(rows_trans, dtype=np.float64)), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"pipeline inverse mapping failed: {exc}") from exc
    if not np.all(np.isfinite(back)):
        raise ValueError("pipeline inverse mapping produced non-finite rows")
    return np.ascontiguousarray(back)


def _dense_rows(matrix: Any, idx: np.ndarray) -> np.ndarray:
    """Densify only the selected rows (sparse-safe, mirrors knn.py)."""
    idx_arr = np.asarray(idx, dtype=np.int64).ravel()
    try:
        import scipy.sparse as _sp  # noqa: PLC0415

        if _sp.issparse(matrix):
            return np.ascontiguousarray(np.asarray(matrix[idx_arr].toarray(), dtype=np.float64))
    except ImportError:
        pass
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr[idx_arr])


def _vote_weights(dists: np.ndarray, mode: str, weights_callable: Any = None) -> np.ndarray:
    k = int(dists.size)
    if k == 0:
        return np.zeros((0,), dtype=np.float64)
    d = np.where(np.isfinite(np.asarray(dists, dtype=np.float64)), np.asarray(dists, dtype=np.float64), 0.0)
    d = np.maximum(d, 0.0)
    if mode == "distance":
        inv = 1.0 / np.maximum(d, DIST_EPS)
        total = float(np.sum(inv))
        if not np.isfinite(total) or total <= 0.0:
            return np.full((k,), 1.0 / max(k, 1), dtype=np.float64)
        return (inv / total).astype(np.float64)
    if weights_callable is not None and callable(weights_callable):
        try:
            w = np.asarray(weights_callable(d), dtype=np.float64).ravel()
            if w.size == k and np.all(np.isfinite(w)) and float(np.sum(w)) > 0.0:
                return (w / float(np.sum(w))).astype(np.float64)
        except Exception:
            pass
    return np.full((k,), 1.0 / max(k, 1), dtype=np.float64)


class Distance2DExtractor:
    """Stateless 2D radar / halo builder (see module docstring)."""

    # -- KNN ------------------------------------------------------------
    @staticmethod
    def extract_knn_2d_state(
        model_obj: Any,
        query: np.ndarray,
        x_idx: int = 0,
        y_idx: int = 1,
        max_neighbors: int = MAX_SPIDER_NEIGHBORS,
    ) -> dict[str, Any] | None:
        """Return the per-frame radar payload, or ``None`` when not mappable.

        Bare estimators map directly; scaler-style pipelines are folded back
        to raw space via ``inverse_transform``. Dimensionality-changing (or
        non-invertible) pipelines honestly yield ``None``.
        """
        est = _unwrap(model_obj)
        est_name = type(est).__name__
        if est_name not in _KNN_NAMES:
            return None
        fit_X = getattr(est, "_fit_X", None)
        if fit_X is None:
            return None
        try:
            n_fit = int(fit_X.shape[0])
        except (AttributeError, TypeError, ValueError):
            return None
        if n_fit <= 0:
            return None

        q_raw_probe = np.ascontiguousarray(np.asarray(query, dtype=np.float64).ravel())
        try:
            dim_raw = int(np.asarray(q_raw_probe).size)
            q_fit, _, is_pipeline = _resolve_query_space(model_obj, est, q_raw_probe)
        except ValueError:
            return None
        try:
            dim = int(fit_X.shape[1])
        except (AttributeError, TypeError, ValueError, IndexError):
            return None
        if q_fit.size != dim or dim_raw <= max(x_idx, y_idx):
            return None
        if not (0 <= x_idx < dim_raw and 0 <= y_idx < dim_raw):
            return None

        is_radius = est_name in _KNN_RADIUS_NAMES
        weights_mode = str(getattr(est, "weights", "uniform") or "uniform")
        weights_callable = getattr(est, "weights", None)
        if not callable(weights_callable):
            weights_callable = None
        metric = str(getattr(est, "effective_metric_", getattr(est, "metric", "euclidean")))
        is_classifier = est_name.endswith("Classifier")

        try:
            if is_radius:
                dists_b, idx_b = est.radius_neighbors(q_fit.reshape(1, -1))
            else:
                dists_b, idx_b = est.kneighbors(q_fit.reshape(1, -1))
        except (ValueError, TypeError, AttributeError):
            return None
        dists = np.asarray(dists_b[0], dtype=np.float64).ravel()
        indices = np.asarray(idx_b[0], dtype=np.int64).ravel()
        order = np.argsort(dists, kind="stable")
        dists, indices = dists[order], indices[order]
        total = int(dists.size)

        capped = False
        if total > max_neighbors:
            dists, indices = dists[:max_neighbors], indices[:max_neighbors]
            capped = True

        weights = _vote_weights(
            dists, weights_mode if weights_mode in ("uniform", "distance") else "uniform",
            weights_callable if weights_mode not in ("uniform", "distance") else None,
        )

        try:
            ref_fit = _dense_rows(fit_X, indices) if indices.size else np.zeros((0, dim))
            ref_rows = _to_raw_space(model_obj, ref_fit, is_pipeline)
        except ValueError:
            return None
        if ref_rows.shape[1] != dim_raw:
            return None
        qx, qy = float(q_raw_probe[x_idx]), float(q_raw_probe[y_idx])
        try:
            fit_targets = np.asarray(getattr(est, "_y", [])).ravel()
        except (TypeError, ValueError):
            fit_targets = np.zeros((0,))

        neighbors: list[dict[str, Any]] = []
        for rank, (pos, (d_full, w)) in enumerate(zip(indices.tolist(), zip(dists.tolist(), weights.tolist())), start=1):
            rx = float(ref_rows[rank - 1, x_idx]) if ref_rows.shape[0] >= rank else 0.0
            ry = float(ref_rows[rank - 1, y_idx]) if ref_rows.shape[0] >= rank else 0.0
            d_2d = float(np.hypot(qx - rx, qy - ry))
            label: Any = None
            if 0 <= int(pos) < fit_targets.size:
                raw_lab = fit_targets[int(pos)]
                label = raw_lab.item() if hasattr(raw_lab, "item") else raw_lab
            neighbors.append({
                "ref_index": int(pos),
                "x": rx,
                "y": ry,
                "distance_full": float(d_full) if np.isfinite(d_full) else 0.0,
                "distance_2d": d_2d,
                "vote_weight": float(w),
                "rank": int(rank),
                "label": label,
            })

        radar_radius = max((nb["distance_2d"] for nb in neighbors), default=0.0)
        votes: dict[str, float] = {}
        predicted: Any = None
        if is_classifier:
            for nb in neighbors:
                key = str(nb["label"])
                votes[key] = votes.get(key, 0.0) + float(nb["vote_weight"])
            if votes:
                predicted = max(votes.items(), key=lambda kv: kv[1])[0]
                try:
                    predicted = int(predicted) if float(predicted).is_integer() else predicted
                except (TypeError, ValueError):
                    pass
        elif neighbors:
            try:
                predicted = float(np.sum([nb["vote_weight"] * float(nb["label"]) for nb in neighbors]))
            except (TypeError, ValueError):
                predicted = None

        try:
            k = int(getattr(est, "n_neighbors", total))
        except (TypeError, ValueError):
            k = int(total)
        radius_val: float | None = None
        if is_radius:
            try:
                radius_val = float(getattr(est, "radius", 0.0))
            except (TypeError, ValueError):
                radius_val = None

        state = KnnRadarState2D(
            model_type=est_name,
            k=k if not is_radius else total,
            weights_mode=weights_mode,
            metric=metric,
            is_radius_model=is_radius,
            radius=radius_val,
            outlier=bool(is_radius and total == 0),
            query_2d=[qx, qy],
            radar_radius_2d=float(radar_radius),
            neighbors_2d=[KnnNeighbor2D(**nb) for nb in neighbors],  # type: ignore[arg-type]
            votes={str(kk): float(vv) for kk, vv in votes.items()},
            predicted=predicted,
            display_capped=capped,
            total_neighbors=total,
        )
        return state.model_dump(mode="json")

    # -- SVM ------------------------------------------------------------
    @staticmethod
    def extract_svm_2d_state(
        model_obj: Any,
        x_idx: int = 0,
        y_idx: int = 1,
        max_halos: int = MAX_SV_HALOS,
    ) -> dict[str, Any] | None:
        """Return the per-model SV halo payload, or ``None`` when N/A."""
        if _is_pipeline(model_obj):
            return None
        est = _unwrap(model_obj)
        est_name = type(est).__name__
        if est_name not in _SVM_NAMES:
            return None

        kernel = str(getattr(est, "kernel", "linear") or "linear")
        if est_name in _SVM_LINEAR_NAMES:
            kernel = "linear"
        try:
            degree = int(getattr(est, "degree", 3))
        except (TypeError, ValueError):
            degree = 3
        try:
            coef0 = float(getattr(est, "coef0", 0.0))
        except (TypeError, ValueError):
            coef0 = 0.0
        if not np.isfinite(coef0):
            coef0 = 0.0

        raw_classes = getattr(est, "classes_", None)
        try:
            classes: list[Any] = (
                [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
                if raw_classes is not None
                else []
            )
        except (TypeError, ValueError):
            classes = []
        is_regressor = est_name in ("SVR", "NuSVR", "LinearSVR")
        is_binary = (len(classes) == 2) if classes else True

        gamma_val: float | None = None
        if kernel in ("rbf", "poly", "sigmoid"):
            try:
                n_feat = int(np.asarray(getattr(est, "support_vectors_", [])).shape[1])
            except (AttributeError, TypeError, ValueError, IndexError):
                n_feat = 0
            raw_g = getattr(est, "_gamma", getattr(est, "gamma", "scale"))
            try:
                gamma_val = float(np.asarray(raw_g, dtype=np.float64).ravel()[0])
            except (TypeError, ValueError, IndexError):
                gamma_val = float("nan")
            if not np.isfinite(gamma_val):
                gamma_val = 1.0 / max(n_feat, 1) if n_feat else None

        c_val: float | None = None
        for attr in ("C", "nu"):
            raw_c = getattr(est, attr, None)
            if raw_c is None or callable(raw_c):
                continue
            try:
                cand = float(np.asarray(raw_c, dtype=np.float64).ravel()[0])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(cand) and cand > 0:
                c_val = cand
                break

        if est_name in _SVM_LINEAR_NAMES:
            coef = np.asarray(getattr(est, "coef_", []), dtype=np.float64)
            try:
                intercept = float(np.asarray(getattr(est, "intercept_", 0.0), dtype=np.float64).ravel()[0])
            except (TypeError, ValueError, IndexError):
                intercept = 0.0
            return SvmState2D(
                model_type=est_name,
                kernel="linear",
                gamma=None,
                c=c_val,
                degree=degree,
                coef0=coef0,
                is_regressor=est_name == "LinearSVR",
                is_binary=is_binary,
                n_classes=len(classes),
                n_support_total=0,
                is_linear_no_sv=True,
            ).model_dump(mode="json")

        sv_raw = getattr(est, "support_vectors_", None)
        if sv_raw is None:
            return None
        try:
            import scipy.sparse as _sp  # noqa: PLC0415

            sv = np.asarray(sv_raw.toarray() if _sp.issparse(sv_raw) else sv_raw, dtype=np.float64)
        except ImportError:
            sv = np.asarray(sv_raw, dtype=np.float64)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        n_sv = int(sv.shape[0])
        if n_sv == 0 or not (0 <= x_idx < sv.shape[1] and 0 <= y_idx < sv.shape[1]):
            return None

        dual_raw = getattr(est, "dual_coef_", None)
        dual = np.asarray(dual_raw, dtype=np.float64) if dual_raw is not None else np.zeros((0, n_sv))
        if dual.ndim == 1:
            dual = dual.reshape(1, -1)
        if dual.shape[1] != n_sv:
            dual = np.zeros((1, n_sv), dtype=np.float64)
        # Halo ranking: strongest absolute pull across OvO rows.
        strength = np.max(np.abs(dual), axis=0) if dual.shape[0] else np.zeros((n_sv,))
        resolved = np.asarray(dual[0], dtype=np.float64).ravel() if dual.shape[0] else np.zeros((n_sv,))

        try:
            train_idx = np.asarray(getattr(est, "support_", np.arange(n_sv)), dtype=np.int64).ravel()
        except (TypeError, ValueError):
            train_idx = np.arange(n_sv, dtype=np.int64)
        if train_idx.size != n_sv:
            train_idx = np.arange(n_sv, dtype=np.int64)

        order = np.argsort(-strength, kind="stable")
        truncated = bool(n_sv > max_halos)
        kept = order[:max_halos] if truncated else order

        halos: list[SvmHalo2D] = []
        for pos in kept.tolist():
            a = float(strength[int(pos)])
            if c_val is not None and np.isfinite(c_val) and c_val > 0:
                scale = min(a / c_val, 1.0)
                status = "violates_margin" if a >= c_val * (1.0 - BOUND_TOL) else "on_margin"
            else:
                denom = float(np.max(strength)) if float(np.max(strength)) > 0 else 1.0
                scale = min(a / denom, 1.0)
                status = "support"
            halos.append(SvmHalo2D(
                train_index=int(train_idx[int(pos)]),
                x=float(sv[int(pos), x_idx]),
                y=float(sv[int(pos), y_idx]),
                dual_coef=float(resolved[int(pos)]) if resolved.size == n_sv else 0.0,
                halo_scale=float(scale),
                status=status,
            ))

        state = SvmState2D(
            model_type=est_name,
            kernel=kernel,
            gamma=gamma_val,
            c=c_val,
            degree=degree,
            coef0=coef0,
            is_regressor=is_regressor,
            is_binary=is_binary,
            n_classes=len(classes),
            n_support_total=n_sv,
            is_linear_no_sv=False,
            support_vectors_2d=halos,
            dual_coefs=[float(v) for v in resolved.tolist()] if resolved.size == n_sv else [],
            halos_truncated=truncated,
        )
        return state.model_dump(mode="json")
