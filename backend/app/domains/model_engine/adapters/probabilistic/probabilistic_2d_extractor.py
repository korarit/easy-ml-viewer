"""Phase 44 — 2D decision geometry for Probabilistic & Discriminant models.

Complements the universal Marching-Squares contour engine (Phase 41) with
the *exact* statistical overlays whose parameters the dense grid cannot
provide:

- **GaussianNB**: axis-aligned bivariate equiprobability ellipses
  ``(x1-mu1)^2/s1^2 + (x2-mu2)^2/s2^2 = k^2`` for ``k in {1, 2, 3}``
  (39.3 % / 86.5 % / 98.9 % mass in 2D) read off ``theta_`` / ``var_``
  (legacy ``sigma_`` pickles covered, floored at ``1e-12``).
- **LDA** (shared covariance): identically-shaped rotated ellipses from the
  2x2 sub-matrix eigendecomposition plus the exact binary decision line
  ``w1*x1 + w2*x2 + b_eff = 0`` (``coef_``/``intercept_`` first, pooled
  Mahalanobis fallback), clipped to the data rect.
- **QDA** (per-class covariances): per-class rotated ellipses plus the exact
  binary conic ``A*u^2 + B*u*v + C*v^2 + D*u + E*v + F = 0`` derived from
  the precision difference, with ``conic_type`` (hyperbola / ellipse /
  parabola) and vector ``conic_segments_world`` sampled from the analytic
  score difference (not from ``predict_proba``, so flat-proba regions
  still render crisply).
- **Discrete NB** (Multinomial / Bernoulli / Complement / Categorical):
  variant metadata + per-axis log-probability slices that drive the 2D
  step legend; the dense heatmap already shows the accumulation.

Conventions (shared with Phase 42 / 43 extractors):

- Coordinates are **raw feature units**; world mapping uses
  ``WorldCoordinateScaler`` (centres via ``to_world_coords``, radii via
  ``r / span * 10`` per axis — raw circles skew under anisotropic spans,
  documented like the KNN radar ellipse).
- Pipeline-wrapped estimators live in *transformed* space, so raw axis
  indices cannot address fitted attributes exactly — the extractor returns
  ``None`` and the caller falls back to the sampled contour.
- Ellipse payloads are capped at ``MAX_ELLIPSE_CLASSES`` (8) ranked by
  prior (GNB/discrete) or class order (LDA/QDA), with an explicit
  truncation flag so the client never guesses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

VAR_FLOOR = 1e-12
EIG_FLOOR = 1e-9
ZERO_TOL = 1e-12
PRIOR_FLOOR = 1e-15
MAX_ELLIPSE_CLASSES = 8
CONIC_GRID = 160
CONIC_SEGMENT_CAP = 2000

_GNB_NAMES = {"GaussianNB"}
_DISCRETE_NAMES = {"MultinomialNB", "BernoulliNB", "ComplementNB", "CategoricalNB"}
_LDA_NAMES = {"LinearDiscriminantAnalysis"}
_QDA_NAMES = {"QuadraticDiscriminantAnalysis"}


class ClassEllipse2D(BaseModel):
    """One class ellipse (raw stats + pre-projected world geometry)."""

    class_index: int = 0
    label: Any = None
    mean_raw: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    std_raw: list[float] = Field(default_factory=lambda: [1.0, 1.0])
    rotation_rad: float = 0.0
    center_world: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    radii_world_1sigma: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    prior: float | None = None


class DiscriminantLine2D(BaseModel):
    """Exact LDA binary decision line clipped to the raw data rect."""

    segment_raw: list[float] | None = None
    segment_world: list[float] | None = None
    w1: float = 0.0
    w2: float = 0.0
    b_eff: float = 0.0
    label: str = ""


class Conic2D(BaseModel):
    """Exact QDA binary conic in raw units + vector segments in world units."""

    kind: str = "qda_conic"
    conic_type: str = Field(description="hyperbola | ellipse | parabola | degenerate")
    coefficients_raw: dict[str, float] = Field(
        default_factory=dict,
        description="A*u^2 + B*u*v + C*v^2 + D*u + E*v + F = 0",
    )
    class_a: int = 0
    class_b: int = 1
    segments_world: list[list[float]] = Field(default_factory=list)
    segments_truncated: bool = False
    label: str = ""


class DiscreteNb2D(BaseModel):
    """Step-legend payload for Multinomial / Bernoulli / Complement / Categorical."""

    variant: str = "multinomial"
    algorithm: str = ""
    alpha: float | None = None
    binarize: float | None = None
    # Per-class [s_x, s_y] slices of feature_log_prob_ on the (X, Y) axes
    # (categorical: per-axis [P(cat=0|c)] exemplar slices, capped).
    axis_log_prob: list[list[float]] = Field(default_factory=list)
    classes: list[Any] = Field(default_factory=list)
    formula: str = ""


class Probabilistic2DGeometry(BaseModel):
    """Exact 2D planar geometry for one probabilistic model on one axis pair."""

    family: str = Field(description="gnb | discrete_nb | lda | qda")
    algorithm: str = ""
    x_feature: str = ""
    y_feature: str = ""
    x_feature_idx: int = 0
    y_feature_idx: int = 0
    n_classes: int = 0
    classes: list[Any] = Field(default_factory=list)
    ellipses: list[ClassEllipse2D] = Field(default_factory=list)
    ellipses_truncated: bool = False
    is_covariance_approximated: bool = False
    discriminant_line: DiscriminantLine2D | None = None
    conic: Conic2D | None = None
    discrete: DiscreteNb2D | None = None
    formula: str = ""


# ---------------------------------------------------------------------------
# Small helpers (duck-typed; pipelines resolve to None upstream)
# ---------------------------------------------------------------------------


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


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(np.asarray(value, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError, IndexError):
        try:
            out = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _classes_of(est: Any) -> list[Any]:
    raw = getattr(est, "classes_", None)
    if raw is None:
        return [0, 1]
    try:
        return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]
    except (TypeError, ValueError):
        return [0, 1]


def _pinned_vector(X: np.ndarray) -> np.ndarray:
    ref = np.asarray(X, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        base = np.nanmedian(ref, axis=0)
    base = np.asarray(base, dtype=np.float64).ravel()
    return np.where(np.isfinite(base), base, 0.0)


def _clip_line_to_rect(
    w1: float,
    w2: float,
    b_eff: float,
    level: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[float] | None:
    """Clip ``w1*x + w2*y + b_eff = level`` to the data rect (Liang-Barsky)."""
    c = float(b_eff - level)
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    if abs(w1) <= ZERO_TOL and abs(w2) <= ZERO_TOL:
        return None
    if abs(w2) <= ZERO_TOL:
        x = -c / w1
        if not (x_min - 1e-9 <= x <= x_max + 1e-9):
            return None
        xc = min(max(x, x_min), x_max)
        return [xc, y_min, xc, y_max]
    if abs(w1) <= ZERO_TOL:
        y = -c / w2
        if not (y_min - 1e-9 <= y <= y_max + 1e-9):
            return None
        yc = min(max(y, y_min), y_max)
        return [x_min, yc, x_max, yc]
    norm = math.hypot(w1, w2)
    dx, dy = -w2 / norm, w1 / norm
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    s0 = w1 * cx + w2 * cy + c
    px, py = cx - (s0 / (norm * norm)) * w1, cy - (s0 / (norm * norm)) * w2
    t_enter, t_exit = -math.inf, math.inf
    for p, d, lo, hi in ((px, dx, x_min, x_max), (py, dy, y_min, y_max)):
        if abs(d) < 1e-15:
            if p < lo - 1e-9 or p > hi + 1e-9:
                return None
            continue
        t0, t1 = (lo - p) / d, (hi - p) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
        if t_enter > t_exit:
            return None
    if not math.isfinite(t_enter) or not math.isfinite(t_exit) or t_exit - t_enter < 1e-12:
        return None
    return [px + t_enter * dx, py + t_enter * dy, px + t_exit * dx, py + t_exit * dy]


def _world_point(vx: float, vy: float, x_scale: Any, y_scale: Any) -> list[float] | None:
    try:
        from app.core.math_3d import WorldCoordinateScaler as _WCS

        wx = _WCS.to_world_coords(np.asarray([vx], dtype=np.float32), x_scale)
        wy = _WCS.to_world_coords(np.asarray([vy], dtype=np.float32), y_scale)
        return [round(float(wx[0]), 3), round(float(wy[0]), 3)]
    except (TypeError, ValueError, AttributeError):
        return None


def _world_radii(rx_raw: float, ry_raw: float, x_scale: Any, y_scale: Any) -> list[float]:
    try:
        x_span = max(float(x_scale.raw_max - x_scale.raw_min), 1e-12)
        y_span = max(float(y_scale.raw_max - y_scale.raw_min), 1e-12)
    except (AttributeError, TypeError, ValueError):
        return [round(float(rx_raw), 4), round(float(ry_raw), 4)]
    return [
        round(float(rx_raw) / x_span * 10.0, 4),
        round(float(ry_raw) / y_span * 10.0, 4),
    ]


def _ellipse_specs_from_covs(
    covs: list[np.ndarray], means: np.ndarray, x_idx: int, y_idx: int
) -> list[dict[str, Any]]:
    """2x2 sub-matrix eigendecomposition per class (mirrors discriminant.py)."""
    specs: list[dict[str, Any]] = []
    for k, cov in enumerate(covs):
        c = np.asarray(cov, dtype=np.float64)
        d = c.shape[0]
        ix = x_idx if 0 <= x_idx < d else 0
        iy = y_idx if 0 <= y_idx < d else min(1, d - 1)
        sub = np.asarray(c[np.ix_([ix, iy], [ix, iy])], dtype=np.float64)
        sub = 0.5 * (sub + sub.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(sub)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.array([1.0, 1.0]), np.eye(2)
        eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64).ravel(), EIG_FLOOR)
        vecs = np.asarray(eigvecs, dtype=np.float64)
        if vecs.shape != (2, 2):
            vecs = np.eye(2)
        if float(np.linalg.det(vecs)) < 0.0:
            vecs[:, 0] *= -1.0
        angle = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
        mu = means[k] if k < means.shape[0] else np.zeros((d,))
        specs.append(
            {
                "center": [float(mu[ix]), float(mu[iy])],
                "radii": [float(np.sqrt(eigvals[0])), float(np.sqrt(eigvals[1]))],
                "angle": float(angle),
            }
        )
    return specs


def _shared_covariance(est: Any, d: int) -> tuple[np.ndarray, bool]:
    """Shared (D, D) covariance for LDA + approximated flag."""
    raw = getattr(est, "covariance_", None)
    if raw is not None:
        try:
            shared = np.asarray(raw, dtype=np.float64)
            if shared.shape == (d, d) and np.all(np.isfinite(shared)):
                return np.ascontiguousarray(shared), False
        except (TypeError, ValueError):
            pass
    scal = getattr(est, "scalings_", None)
    if scal is not None:
        try:
            s_mat = np.asarray(scal, dtype=np.float64)
            if s_mat.ndim == 2 and s_mat.shape[0] == d and s_mat.shape[1] >= 1:
                approx = np.linalg.pinv(s_mat).T @ np.linalg.pinv(s_mat)
                if approx.shape == (d, d) and np.all(np.isfinite(approx)):
                    return np.ascontiguousarray(approx), True
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass
    return np.eye(d, dtype=np.float64), True


def _qda_covariances(est: Any, n_classes: int, d: int) -> tuple[list[np.ndarray], bool]:
    raw_cov = getattr(est, "covariance_", None)
    if raw_cov is not None:
        try:
            arr = np.asarray(raw_cov, dtype=np.float64)
            if arr.ndim == 3 and arr.shape[0] == n_classes and arr.shape[1:] == (d, d):
                return [np.asarray(c, dtype=np.float64) for c in arr], False
        except (TypeError, ValueError):
            pass
    rotations = getattr(est, "rotations_", None)
    scalings = getattr(est, "scalings_", None)
    if (
        isinstance(rotations, list)
        and isinstance(scalings, list)
        and len(rotations) == len(scalings)
        and len(rotations) > 0
    ):
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


def _marching_squares(
    xs: np.ndarray, ys: np.ndarray, field: np.ndarray, threshold: float
) -> list[list[float]]:
    """Local 16-case Marching Squares (mirrors contour_2d_generator)."""
    segments: list[list[float]] = []
    ny, nx = field.shape
    if ny < 2 or nx < 2:
        return segments
    thr = float(threshold)
    above = field >= thr
    codes = (
        above[:-1, :-1].astype(np.uint8)
        | (above[:-1, 1:].astype(np.uint8) << 1)
        | (above[1:, 1:].astype(np.uint8) << 2)
        | (above[1:, :-1].astype(np.uint8) << 3)
    )
    cross_j, cross_i = np.nonzero((codes != 0) & (codes != 15))
    if cross_j.size == 0:
        return segments
    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    fld = np.asarray(field, dtype=np.float64)

    def _interp(p_a: float, v_a: float, p_b: float, v_b: float) -> float:
        denom = v_b - v_a
        if abs(denom) < 1e-12:
            return (p_a + p_b) / 2.0
        t = min(1.0, max(0.0, (thr - v_a) / denom))
        return p_a + t * (p_b - p_a)

    for j, i in zip(cross_j.tolist(), cross_i.tolist()):
        s0, s1 = float(fld[j, i]), float(fld[j, i + 1])
        s2, s3 = float(fld[j + 1, i + 1]), float(fld[j + 1, i])
        x0, x1 = float(xs_f[i]), float(xs_f[i + 1])
        y0, y1 = float(ys_f[j]), float(ys_f[j + 1])
        top = (_interp(x0, s0, x1, s1), y0) if (s0 >= thr) != (s1 >= thr) else None
        right = (x1, _interp(y0, s1, y1, s2)) if (s1 >= thr) != (s2 >= thr) else None
        bottom = (_interp(x0, s3, x1, s2), y1) if (s3 >= thr) != (s2 >= thr) else None
        left = (x0, _interp(y0, s0, y1, s3)) if (s0 >= thr) != (s3 >= thr) else None
        pts = [p for p in (top, right, bottom, left) if p is not None]
        if len(pts) == 2:
            segments.append([round(pts[0][0], 3), round(pts[0][1], 3), round(pts[1][0], 3), round(pts[1][1], 3)])
        elif len(pts) == 4:
            segments.append([round(pts[0][0], 3), round(pts[0][1], 3), round(pts[1][0], 3), round(pts[1][1], 3)])
            segments.append([round(pts[2][0], 3), round(pts[2][1], 3), round(pts[3][0], 3), round(pts[3][1], 3)])
    return segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class Probabilistic2DExtractor:
    """Stateless 2D ellipse / discriminant builder (see module docstring)."""

    @staticmethod
    def extract_probabilistic_2d_geometry(
        *,
        model_obj: Any,
        X: np.ndarray,
        x_feature_idx: int,
        y_feature_idx: int,
        x_feature_name: str,
        y_feature_name: str,
        x_range_raw: tuple[float, float],
        y_range_raw: tuple[float, float],
        x_scale: Any = None,
        y_scale: Any = None,
    ) -> Probabilistic2DGeometry | None:
        """Build the exact 2D geometry, or ``None`` when N/A / pipelined."""
        if _is_pipeline(model_obj):
            return None
        est = _unwrap(model_obj)
        est_name = type(est).__name__
        ref = np.asarray(X, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[0] == 0 or ref.shape[1] == 0:
            return None
        d = int(ref.shape[1])
        if not (0 <= x_feature_idx < d and 0 <= y_feature_idx < d):
            return None
        if x_feature_idx == y_feature_idx and d >= 2:
            return None
        classes = _classes_of(est)
        n_classes = len(classes)

        if est_name in _GNB_NAMES:
            return Probabilistic2DExtractor._gnb(
                est, est_name, classes, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, x_range_raw, y_range_raw,
                x_scale, y_scale,
            )
        if est_name in _DISCRETE_NAMES:
            return Probabilistic2DExtractor._discrete(
                est, est_name, classes, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, x_range_raw, y_range_raw,
                x_scale, y_scale,
            )
        if est_name in _LDA_NAMES:
            return Probabilistic2DExtractor._lda(
                est, est_name, classes, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, x_range_raw, y_range_raw,
                x_scale, y_scale,
            )
        if est_name in _QDA_NAMES:
            return Probabilistic2DExtractor._qda(
                est, est_name, classes, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, x_range_raw, y_range_raw,
                x_scale, y_scale,
            )
        return None

    # -- GaussianNB ------------------------------------------------------
    @staticmethod
    def _gnb(
        est: Any, est_name: str, classes: list[Any],
        x_idx: int, y_idx: int, xn: str, yn: str,
        xr: tuple[float, float], yr: tuple[float, float],
        x_scale: Any, y_scale: Any,
    ) -> Probabilistic2DGeometry | None:
        try:
            theta = np.asarray(getattr(est, "theta_"), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        raw_var = getattr(est, "var_", getattr(est, "sigma_", None))
        if raw_var is None:
            return None
        try:
            var = np.asarray(raw_var, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if var.ndim == 1:
            var = var.reshape(1, -1)
        if theta.shape != var.shape or theta.shape[1] <= max(x_idx, y_idx):
            return None
        var = np.maximum(np.where(np.isfinite(var), var, 1.0), VAR_FLOOR)
        try:
            priors = np.asarray(getattr(est, "class_prior_"), dtype=np.float64).ravel()
            priors = np.where(np.isfinite(priors), np.maximum(priors, 0.0), 0.0)
            total = float(np.sum(priors))
            priors = priors / total if total > 0 else np.full((theta.shape[0],), 1.0 / max(theta.shape[0], 1))
        except (TypeError, ValueError, AttributeError):
            priors = np.full((theta.shape[0],), 1.0 / max(theta.shape[0], 1))
        order = list(range(theta.shape[0]))
        truncated = False
        if len(order) > MAX_ELLIPSE_CLASSES:
            order = list(np.argsort(priors)[::-1][:MAX_ELLIPSE_CLASSES])
            truncated = True
        ellipses: list[ClassEllipse2D] = []
        for k in order:
            mx, my = float(theta[k, x_idx]), float(theta[k, y_idx])
            sx = float(np.sqrt(max(float(var[k, x_idx]), VAR_FLOOR)))
            sy = float(np.sqrt(max(float(var[k, y_idx]), VAR_FLOOR)))
            center = _world_point(mx, my, x_scale, y_scale) or [0.0, 0.0]
            radii = _world_radii(sx, sy, x_scale, y_scale)
            ellipses.append(ClassEllipse2D(
                class_index=int(k),
                label=classes[k] if k < len(classes) else k,
                mean_raw=[round(mx, 6), round(my, 6)],
                std_raw=[round(sx, 6), round(sy, 6)],
                rotation_rad=0.0,
                center_world=center,
                radii_world_1sigma=radii,
                prior=float(priors[k]) if k < priors.size else None,
            ))
        return Probabilistic2DGeometry(
            family="gnb",
            algorithm=est_name,
            x_feature=str(xn),
            y_feature=str(yn),
            x_feature_idx=int(x_idx),
            y_feature_idx=int(y_idx),
            n_classes=int(theta.shape[0]),
            classes=classes,
            ellipses=ellipses,
            ellipses_truncated=bool(truncated),
            formula=f"ellipses (mu, sigma) per class on ({xn}, {yn}); k=1/2/3 sigma",
        )

    # -- Discrete NB -----------------------------------------------------
    @staticmethod
    def _discrete(
        est: Any, est_name: str, classes: list[Any],
        x_idx: int, y_idx: int, xn: str, yn: str,
        xr: tuple[float, float], yr: tuple[float, float],
        x_scale: Any, y_scale: Any,
    ) -> Probabilistic2DGeometry | None:
        variant = (
            "bernoulli" if est_name == "BernoulliNB"
            else "complement" if est_name == "ComplementNB"
            else "categorical" if est_name == "CategoricalNB"
            else "multinomial"
        )
        n_classes = len(classes)
        axis_log_prob: list[list[float]] = []
        formula = ""
        if variant == "categorical":
            tables = list(getattr(est, "feature_log_prob_", []))
            for k in range(n_classes):
                row: list[float] = []
                for j in (x_idx, y_idx):
                    if 0 <= j < len(tables):
                        tab = np.asarray(tables[j], dtype=np.float64)
                        kk = min(k, tab.shape[0] - 1) if tab.ndim == 2 else 0
                        try:
                            row.append(float(tab[kk, 0]) if tab.ndim == 2 else float(tab.ravel()[0]))
                        except (IndexError, TypeError, ValueError):
                            row.append(0.0)
                    else:
                        row.append(0.0)
                axis_log_prob.append([round(v, 6) for v in row])
            formula = f"J[c] = log pi[c] + LL_x[cat] + LL_y[cat] ({est_name})"
        else:
            try:
                flp = np.asarray(getattr(est, "feature_log_prob_"), dtype=np.float64)
            except (TypeError, ValueError):
                return None
            if flp.ndim != 2 or flp.shape[0] != n_classes:
                return None
            if flp.shape[1] <= max(x_idx, y_idx):
                return None
            flp = np.where(np.isfinite(flp), flp, -30.0)
            for k in range(n_classes):
                axis_log_prob.append([round(float(flp[k, x_idx]), 6), round(float(flp[k, y_idx]), 6)])
            if variant == "bernoulli":
                formula = "J[c] = log pi[c] + x*log t + (1-x)*log(1-t) per axis (presence/absence steps)"
            elif variant == "complement":
                formula = "J[c] = sum x * w[c] (complement weights, argmax decision)"
            else:
                formula = "J[c] = log pi[c] + x*log p[c] per axis (count steps)"
        try:
            alpha = float(getattr(est, "alpha", 1.0) or 0.0)
        except (TypeError, ValueError):
            alpha = None
        bin_raw = getattr(est, "binarize", None)
        try:
            binarize = None if bin_raw is None else float(bin_raw)
        except (TypeError, ValueError):
            binarize = None
        # Step legend anchors on the axis means (world) so badges sit on data.
        order = list(range(n_classes))
        truncated = False
        if len(order) > MAX_ELLIPSE_CLASSES:
            order = order[:MAX_ELLIPSE_CLASSES]
            truncated = True
        return Probabilistic2DGeometry(
            family="discrete_nb",
            algorithm=est_name,
            x_feature=str(xn),
            y_feature=str(yn),
            x_feature_idx=int(x_idx),
            y_feature_idx=int(y_idx),
            n_classes=int(n_classes),
            classes=classes,
            ellipses=[],
            ellipses_truncated=bool(truncated),
            discrete=DiscreteNb2D(
                variant=variant,
                algorithm=est_name,
                alpha=alpha,
                binarize=binarize,
                axis_log_prob=axis_log_prob,
                classes=classes,
                formula=formula,
            ),
            formula=formula,
        )

    # -- LDA -------------------------------------------------------------
    @staticmethod
    def _lda(
        est: Any, est_name: str, classes: list[Any], ref: np.ndarray,
        x_idx: int, y_idx: int, xn: str, yn: str,
        xr: tuple[float, float], yr: tuple[float, float],
        x_scale: Any, y_scale: Any,
    ) -> Probabilistic2DGeometry | None:
        n_classes = len(classes)
        d = int(ref.shape[1])
        try:
            means = np.asarray(getattr(est, "means_"), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if means.ndim != 2 or means.shape[1] != d or means.shape[0] != n_classes:
            return None
        shared, approx = _shared_covariance(est, d)
        try:
            prec = np.linalg.pinv(shared + np.eye(d) * 1e-9)
        except np.linalg.LinAlgError:
            return None
        specs = _ellipse_specs_from_covs([shared for _ in range(n_classes)], means, x_idx, y_idx)
        try:
            priors = np.asarray(getattr(est, "priors_", getattr(est, "priors", None)), dtype=np.float64).ravel()
            if priors.size != n_classes:
                raise ValueError("prior size mismatch")
            priors = np.where(np.isfinite(priors), np.maximum(priors, 0.0), 0.0)
            priors = priors / float(np.sum(priors)) if float(np.sum(priors)) > 0 else np.full((n_classes,), 1.0 / max(n_classes, 1))
        except (TypeError, ValueError, AttributeError):
            priors = np.full((n_classes,), 1.0 / max(n_classes, 1))
        order = list(range(n_classes))
        truncated = False
        if len(order) > MAX_ELLIPSE_CLASSES:
            order = order[:MAX_ELLIPSE_CLASSES]
            truncated = True
        ellipses: list[ClassEllipse2D] = []
        for k in order:
            spec = specs[k]
            center = _world_point(spec["center"][0], spec["center"][1], x_scale, y_scale) or [0.0, 0.0]
            radii = _world_radii(spec["radii"][0], spec["radii"][1], x_scale, y_scale)
            std_raw = [round(float(v), 6) for v in spec["radii"]]
            ellipses.append(ClassEllipse2D(
                class_index=int(k),
                label=classes[k] if k < len(classes) else k,
                mean_raw=[round(float(spec["center"][0]), 6), round(float(spec["center"][1]), 6)],
                std_raw=std_raw,
                rotation_rad=float(spec["angle"]),
                center_world=center,
                radii_world_1sigma=radii,
                prior=float(priors[k]) if k < priors.size else None,
            ))
        # Exact binary line: native coef_ first, pooled-Mahalanobis fallback.
        line: DiscriminantLine2D | None = None
        if n_classes == 2:
            pinned = _pinned_vector(ref)
            w_full: np.ndarray | None = None
            b_full = 0.0
            coef = getattr(est, "coef_", None)
            icept = getattr(est, "intercept_", None)
            try:
                c_arr = np.asarray(coef, dtype=np.float64) if coef is not None else None
                i_arr = np.asarray(icept, dtype=np.float64).ravel() if icept is not None else None
                if c_arr is not None and c_arr.size == d:
                    w_full = c_arr.reshape(-1)[:d].astype(np.float64)
                    b_full = float(i_arr[0]) if i_arr is not None and i_arr.size else 0.0
                elif c_arr is not None and c_arr.ndim == 2 and c_arr.shape[1] == d:
                    w_full = np.asarray(c_arr[0], dtype=np.float64)
                    b_full = float(i_arr[0]) if i_arr is not None and i_arr.size else 0.0
            except (TypeError, ValueError, IndexError):
                w_full = None
            if w_full is None:
                try:
                    logp = np.log(np.maximum(priors, PRIOR_FLOOR))
                    w_full = prec @ (means[1] - means[0])
                    b_full = float(
                        -0.5 * (means[1] @ prec @ means[1] - means[0] @ prec @ means[0])
                        + (logp[1] - logp[0])
                    )
                except (IndexError, ValueError):
                    w_full = None
            if w_full is not None:
                w1 = float(w_full[x_idx])
                w2 = float(w_full[y_idx])
                b_eff = float(b_full) + sum(
                    float(w_full[j]) * float(pinned[j])
                    for j in range(min(d, pinned.size))
                    if j != x_idx and j != y_idx
                )
                raw = _clip_line_to_rect(w1, w2, b_eff, 0.0, xr, yr)
                seg_raw = [round(float(v), 6) for v in raw] if raw is not None else None
                seg_world = None
                if seg_raw is not None:
                    p1 = _world_point(seg_raw[0], seg_raw[1], x_scale, y_scale)
                    p2 = _world_point(seg_raw[2], seg_raw[3], x_scale, y_scale)
                    if p1 is not None and p2 is not None:
                        seg_world = [p1[0], p1[1], p2[0], p2[1]]
                line = DiscriminantLine2D(
                    segment_raw=seg_raw,
                    segment_world=seg_world,
                    w1=float(w1),
                    w2=float(w2),
                    b_eff=float(b_eff),
                    label=f"LDA shared-covariance boundary ({classes[0]} vs {classes[1]})",
                )
        return Probabilistic2DGeometry(
            family="lda",
            algorithm=est_name,
            x_feature=str(xn),
            y_feature=str(yn),
            x_feature_idx=int(x_idx),
            y_feature_idx=int(y_idx),
            n_classes=int(n_classes),
            classes=classes,
            ellipses=ellipses,
            ellipses_truncated=bool(truncated),
            is_covariance_approximated=bool(approx),
            discriminant_line=line,
            formula=f"delta[c] = x^T S^-1 mu[c] - 0.5 mu[c]^T S^-1 mu[c] + log pi[c] (shared S, straight line)",
        )

    # -- QDA -------------------------------------------------------------
    @staticmethod
    def _qda(
        est: Any, est_name: str, classes: list[Any], ref: np.ndarray,
        x_idx: int, y_idx: int, xn: str, yn: str,
        xr: tuple[float, float], yr: tuple[float, float],
        x_scale: Any, y_scale: Any,
    ) -> Probabilistic2DGeometry | None:
        n_classes = len(classes)
        d = int(ref.shape[1])
        try:
            means = np.asarray(getattr(est, "means_"), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if means.ndim != 2 or means.shape[1] != d or means.shape[0] != n_classes:
            return None
        covs, approx = _qda_covariances(est, n_classes, d)
        specs = _ellipse_specs_from_covs(covs, means, x_idx, y_idx)
        try:
            priors = np.asarray(getattr(est, "priors_", getattr(est, "priors", None)), dtype=np.float64).ravel()
            if priors.size != n_classes:
                raise ValueError("prior size mismatch")
            priors = np.where(np.isfinite(priors), np.maximum(priors, 0.0), 0.0)
            priors = priors / float(np.sum(priors)) if float(np.sum(priors)) > 0 else np.full((n_classes,), 1.0 / max(n_classes, 1))
        except (TypeError, ValueError, AttributeError):
            priors = np.full((n_classes,), 1.0 / max(n_classes, 1))
        order = list(range(n_classes))
        truncated = False
        if len(order) > MAX_ELLIPSE_CLASSES:
            order = order[:MAX_ELLIPSE_CLASSES]
            truncated = True
        ellipses: list[ClassEllipse2D] = []
        for k in order:
            spec = specs[k]
            center = _world_point(spec["center"][0], spec["center"][1], x_scale, y_scale) or [0.0, 0.0]
            radii = _world_radii(spec["radii"][0], spec["radii"][1], x_scale, y_scale)
            ellipses.append(ClassEllipse2D(
                class_index=int(k),
                label=classes[k] if k < len(classes) else k,
                mean_raw=[round(float(spec["center"][0]), 6), round(float(spec["center"][1]), 6)],
                std_raw=[round(float(v), 6) for v in spec["radii"]],
                rotation_rad=float(spec["angle"]),
                center_world=center,
                radii_world_1sigma=radii,
                prior=float(priors[k]) if k < priors.size else None,
            ))
        conic: Conic2D | None = None
        if n_classes == 2:
            conic = Probabilistic2DExtractor._binary_conic(
                covs, means, priors, ref, x_idx, y_idx, xr, yr, x_scale, y_scale,
                classes,
            )
        return Probabilistic2DGeometry(
            family="qda",
            algorithm=est_name,
            x_feature=str(xn),
            y_feature=str(yn),
            x_feature_idx=int(x_idx),
            y_feature_idx=int(y_idx),
            n_classes=int(n_classes),
            classes=classes,
            ellipses=ellipses,
            ellipses_truncated=bool(truncated),
            is_covariance_approximated=bool(approx),
            conic=conic,
            formula="delta[c] = -0.5 log|S[c]| - 0.5 (x-mu[c])^T S[c]^-1 (x-mu[c]) + log pi[c] (conic boundary)",
        )

    @staticmethod
    def _binary_conic(
        covs: list[np.ndarray], means: np.ndarray, priors: np.ndarray,
        ref: np.ndarray, x_idx: int, y_idx: int,
        xr: tuple[float, float], yr: tuple[float, float],
        x_scale: Any, y_scale: Any, classes: list[Any],
    ) -> Conic2D | None:
        d = int(ref.shape[1])
        try:
            precs = [np.linalg.pinv(c + np.eye(d) * 1e-9) for c in covs]
            logdets = []
            for c in covs:
                sign, ld = np.linalg.slogdet(np.asarray(c, dtype=np.float64))
                logdets.append(float(ld) if sign > 0 and np.isfinite(ld) else 0.0)
        except np.linalg.LinAlgError:
            return None
        pinned = _pinned_vector(ref)
        rest = np.asarray(pinned, dtype=np.float64).ravel()
        PA, PB = precs[0], precs[1]
        muA, muB = means[0], means[1]
        try:
            logp = np.log(np.maximum(np.asarray(priors, dtype=np.float64).ravel(), PRIOR_FLOOR))
        except (TypeError, ValueError):
            logp = np.zeros((2,))

        def _q_lin(P: np.ndarray, mu: np.ndarray, axis: int) -> float:
            s = 0.0
            for k in range(d):
                if k == x_idx or k == y_idx:
                    continue
                s += float(P[axis, k]) * float(rest[k] if k < rest.size else 0.0)
            return float(s - float(P[axis, :] @ mu))

        A = -0.5 * (float(PA[x_idx, x_idx]) - float(PB[x_idx, x_idx]))
        B = -(float(PA[x_idx, y_idx]) - float(PB[x_idx, y_idx]))
        C = -0.5 * (float(PA[y_idx, y_idx]) - float(PB[y_idx, y_idx]))
        Dx = -_q_lin(PA, muA, x_idx) + _q_lin(PB, muB, x_idx)
        Ey = -_q_lin(PA, muA, y_idx) + _q_lin(PB, muB, y_idx)
        # Constant: rest-rest block + mu terms + log-det / prior offsets.
        const = 0.0
        for P, mu in ((PA, muA), (PB, muB)):
            sign = -1.0 if P is PA else 1.0
            rr = np.zeros((d,), dtype=np.float64)
            for k in range(d):
                if k != x_idx and k != y_idx and k < rest.size:
                    rr[k] = float(rest[k])
            const += sign * float(0.5 * (rr @ P @ rr) - float(mu @ P @ rr) + 0.5 * float(mu @ P @ mu))
        const += float(-0.5 * (logdets[0] - logdets[1]) + (logp[0] - logp[1]))
        # Decision D(u,v) = A u^2 + B uv + C v^2 + Dx u + Ey v + F with
        # class-A-minus-B sign; boundary D = 0.
        F = float(const)
        disc = float(B * B - 4.0 * A * C)
        if abs(disc) <= 1e-9 * max(1.0, abs(A) + abs(B) + abs(C)):
            ctype = "parabola"
        elif disc > 0:
            ctype = "hyperbola"
        elif abs(A) < 1e-12 and abs(B) < 1e-12 and abs(C) < 1e-12:
            return None  # degenerate: identical covariances behave like LDA
        else:
            ctype = "ellipse"
        coeffs = {
            "A": round(float(A), 9),
            "B": round(float(B), 9),
            "C": round(float(C), 9),
            "D": round(float(Dx), 9),
            "E": round(float(Ey), 9),
            "F": round(float(F), 9),
        }

        def _eval(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            return A * u * u + B * u * v + C * v * v + Dx * u + Ey * v + F

        n = CONIC_GRID
        x_lo, x_hi = (float(xr[0]), float(xr[1])) if xr[0] <= xr[1] else (float(xr[1]), float(xr[0]))
        y_lo, y_hi = (float(yr[0]), float(yr[1])) if yr[0] <= yr[1] else (float(yr[1]), float(yr[0]))
        if not (np.isfinite([x_lo, x_hi, y_lo, y_hi]).all() and x_hi > x_lo and y_hi > y_lo):
            return None
        uu = np.linspace(x_lo, x_hi, n, dtype=np.float64)
        vv = np.linspace(y_lo, y_hi, n, dtype=np.float64)
        UU, VV = np.meshgrid(uu, vv, indexing="xy")
        field = _eval(UU, VV)
        if not np.isfinite(field).any():
            return None
        field = np.where(np.isfinite(field), field, 0.0)
        try:
            from app.core.math_3d import WorldCoordinateScaler as _WCS

            gx_w = _WCS.to_world_coords(uu.astype(np.float32), x_scale).astype(np.float64)
            gy_w = _WCS.to_world_coords(vv.astype(np.float32), y_scale).astype(np.float64)
        except (TypeError, ValueError, AttributeError):
            return None
        segs = _marching_squares(gx_w, gy_w, field, 0.0)
        truncated = bool(len(segs) > CONIC_SEGMENT_CAP)
        if truncated:
            segs = segs[:CONIC_SEGMENT_CAP]
        return Conic2D(
            kind="qda_conic",
            conic_type=ctype,
            coefficients_raw=coeffs,
            class_a=0,
            class_b=1,
            segments_world=segs,
            segments_truncated=bool(truncated),
            label=f"QDA {ctype} boundary ({classes[0]} vs {classes[1]})",
        )

    @staticmethod
    def evaluate_conic(
        coefficients: dict[str, float], u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Evaluate ``A u^2 + B uv + C v^2 + D u + E v + F`` (test hook)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        return (
            float(coefficients.get("A", 0.0)) * u * u
            + float(coefficients.get("B", 0.0)) * u * v
            + float(coefficients.get("C", 0.0)) * v * v
            + float(coefficients.get("D", 0.0)) * u
            + float(coefficients.get("E", 0.0)) * v
            + float(coefficients.get("F", 0.0))
        )
