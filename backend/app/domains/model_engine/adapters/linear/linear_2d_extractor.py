"""Phase 42 — Exact 2D decision geometry for Linear & Generalized Linear models.

Complements the universal Marching-Squares contour engine (Phase 41) with the
*analytic* planar solution ``w1*x1 + w2*x2 + b_eff = t``.  For linear families
the decision boundary is exactly a straight line, so this module returns the
clipped segment endpoints directly (< 1e-4 deviation by construction) instead
of a sampled approximation:

- ``LinearRegression`` (OLS): regression plane ``ŷ = w1*x1 + w2*x2 + b_eff``
  plus the X-sweep slope line used by the 1D response view.
- ``LogisticRegression``: binary sigmoid boundary (``P = 0.5``) with parallel
  confidence isolines ``P in {0.7, 0.8, 0.9}`` via ``t = logit(P)``, and
  pairwise ``z_A = z_B`` lines for multinomial softmax (convex polygons).
- ``Ridge`` / ``RidgeClassifier``: shrunk line + ``z = ±1`` margin corridor.
- ``Lasso`` family: sparsity flags (``|w_j| <= 1e-12`` counts as pruned) —
  a zeroed axis yields an axis-parallel boundary; both zeroed yields a
  warning with no drawable line.
- ``ElasticNet``: ``l1_ratio`` / ``alpha`` balance payload for the dial.
- ``SGDClassifier`` / ``SGDRegressor``: ``z = ±1`` margin corridor
  (``W = 2 / ||w12||``) or the ``epsilon`` tube for regressors.

Pipeline-wrapped estimators live in *transformed* space, so raw-space pinned
values cannot fold into ``b_eff`` exactly — the extractor returns ``None``
and the caller falls back to the sampled contour.  Non-linear estimators
(no ``coef_``) likewise yield ``None``.

Pure NumPy + float-safe math; no DataFrame retention (memory budget §5.1).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

ZERO_TOL = 1e-12
CONFIDENCE_PROBAS = (0.7, 0.8, 0.9)
MARGIN_LEVEL = 1.0
MAX_PAIRWISE_CLASSES = 8


class Linear2DLine(BaseModel):
    """One exact analytic line clipped to the raw X/Y data rect."""

    kind: str = Field(
        description="decision | margin_pos | margin_neg | confidence | pairwise | slope"
    )
    level: float = Field(description="Score threshold t (or proba for confidence)")
    segment_raw: list[float] | None = Field(
        default=None,
        description="[x1, y1, x2, y2] in raw feature units, None when the line misses the rect",
    )
    segment_world: list[float] | None = Field(
        default=None,
        description="[x1, y1, x2, y2] in world coords [-5, +5], None when missing",
    )
    label: str = ""
    class_a: int | None = None
    class_b: int | None = None


class Linear2DGeometry(BaseModel):
    """Exact 2D planar geometry for one linear model on one (X, Y) axis pair."""

    family: str = Field(description="ols | logistic | ridge | lasso | elasticnet | sgd | ridge_classifier")
    task: Literal["classification", "regression"]
    x_feature: str = ""
    y_feature: str = ""
    x_feature_idx: int = 0
    y_feature_idx: int = 0
    w1: float = 0.0
    w2: float = 0.0
    b_eff: float = Field(default=0.0, description="Effective bias with off-axis features pinned")
    decision_threshold: float = Field(description="0.5 (proba) | 0.0 (scores) | NaN (regression)")
    normal_vector: list[float] = Field(default_factory=lambda: [0.0, 0.0], description="Unit ascent direction (w1, w2)/||w|| in raw space")
    corridor_width: float | None = Field(
        default=None, description="Margin width W = 2/||w12|| (raw units), None when degenerate"
    )
    margin_level: float | None = Field(default=None, description="±score half-width (1.0) or epsilon tube half-width")
    is_x_zeroed: bool = False
    is_y_zeroed: bool = False
    both_zeroed: bool = False
    sparsity_ratio: float = 0.0
    l1_ratio: float | None = None
    alpha: float | None = None
    penalty: str | None = None
    slope_w1: float | None = Field(default=None, description="Regression dŷ/dx1 at pinned background")
    x_sweep_intercept: float | None = Field(default=None, description="Regression ŷ(x1=0) at pinned background")
    formula: str = ""
    is_multiclass: bool = False
    n_classes: int = 0
    lines: list[Linear2DLine] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Weight resolution (duck-typed: works for bare estimators; pipelines → None)
# ---------------------------------------------------------------------------

_L1_NAMES = {
    "Lasso", "LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC",
}
_RIDGE_NAMES = {"Ridge", "RidgeCV"}
_RIDGE_CLF_NAMES = {"RidgeClassifier", "RidgeClassifierCV"}
_ELASTIC_NAMES = {
    "ElasticNet", "ElasticNetCV", "MultiTaskElasticNet", "MultiTaskElasticNetCV",
}
_SGD_CLF_NAMES = {
    "SGDClassifier", "Perceptron", "PassiveAggressiveClassifier",
}
_SGD_REG_NAMES = {"SGDRegressor", "PassiveAggressiveRegressor"}
_PROBA_LOSSES = {"log_loss", "log", "modified_huber"}


def _is_pipeline(model_obj: Any) -> bool:
    return isinstance(getattr(model_obj, "steps", None), list) and bool(model_obj.steps)


def _finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(np.asarray(value, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError, IndexError):
        try:
            out = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _resolve_coef_intercept(final_est: Any) -> tuple[np.ndarray, np.ndarray] | None:
    raw_coef = getattr(final_est, "coef_", None)
    if raw_coef is None:
        return None
    try:
        coef = np.asarray(raw_coef, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    elif coef.ndim > 2:
        coef = coef.reshape(coef.shape[0], -1)
    if coef.size == 0:
        return None
    raw_intercept = getattr(final_est, "intercept_", 0.0)
    try:
        intercepts = np.atleast_1d(np.asarray(raw_intercept, dtype=np.float64)).ravel()
    except (TypeError, ValueError):
        intercepts = np.zeros((coef.shape[0],), dtype=np.float64)
    if intercepts.size == 0:
        intercepts = np.zeros((coef.shape[0],), dtype=np.float64)
    elif intercepts.size == 1 and coef.shape[0] > 1:
        intercepts = np.repeat(intercepts, coef.shape[0])
    intercepts = intercepts[: coef.shape[0]]
    coef = np.where(np.isfinite(coef), coef, 0.0)
    intercepts = np.where(np.isfinite(intercepts), intercepts, 0.0)
    return np.ascontiguousarray(coef), np.ascontiguousarray(intercepts)


def _pinned_vector(
    X: np.ndarray, pinning_mode: str, current_row_values: np.ndarray | None
) -> np.ndarray:
    ref = np.asarray(X, dtype=np.float64)
    if pinning_mode == "mean":
        with np.errstate(invalid="ignore"):
            base = np.nanmean(ref, axis=0)
    elif pinning_mode == "current_row" and current_row_values is not None:
        row = np.asarray(current_row_values, dtype=np.float64).ravel()
        base = np.zeros((ref.shape[1],), dtype=np.float64)
        n = min(ref.shape[1], row.size)
        base[:n] = row[:n]
    elif pinning_mode == "zero":
        base = np.zeros((ref.shape[1],), dtype=np.float64)
    else:  # "median" (recommended) and any unknown mode
        base = np.nanmedian(ref, axis=0)
    return np.where(np.isfinite(base), base, 0.0)


def _clip_line_to_rect(
    w1: float, w2: float, b_eff: float, level: float,
    x_range: tuple[float, float], y_range: tuple[float, float],
) -> list[float] | None:
    """Clip ``w1*x + w2*y + b_eff = level`` to the data rect (Liang–Barsky).

    Returns ``[x1, y1, x2, y2]`` in raw units, or ``None`` when the line is
    degenerate (both weights ~0) or misses the rect entirely.
    """
    c = float(b_eff - level)
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    if abs(w1) <= ZERO_TOL and abs(w2) <= ZERO_TOL:
        return None
    if abs(w2) > ZERO_TOL and abs(w1) <= ZERO_TOL:
        # Horizontal line y = -c / w2.
        y = -c / w2
        if not (y_min - 1e-9 <= y <= y_max + 1e-9):
            return None
        yc = min(max(y, y_min), y_max)
        return [x_min, yc, x_max, yc]
    if abs(w1) > ZERO_TOL and abs(w2) <= ZERO_TOL:
        # Vertical line x = -c / w1.
        x = -c / w1
        if not (x_min - 1e-9 <= x <= x_max + 1e-9):
            return None
        xc = min(max(x, x_min), x_max)
        return [xc, y_min, xc, y_max]
    # General case: param march along the direction perpendicular to w.
    norm = math.hypot(w1, w2)
    dx, dy = -w2 / norm, w1 / norm
    # Particular point: closest approach to the rect centre.
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


def _short_name(name: str, limit: int = 24) -> str:
    name = str(name)
    return name if len(name) <= limit else name[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_linear_2d_geometry(
    *,
    model_obj: Any,
    final_est: Any,
    X: np.ndarray,
    x_feature_idx: int,
    y_feature_idx: int,
    x_feature_name: str,
    y_feature_name: str,
    x_range_raw: tuple[float, float],
    y_range_raw: tuple[float, float],
    x_scale: Any = None,
    y_scale: Any = None,
    pinning_mode: str = "median",
    current_row_values: np.ndarray | None = None,
) -> Linear2DGeometry | None:
    """Build the exact 2D geometry, or ``None`` for non-linear / pipelined models."""
    if _is_pipeline(model_obj):
        return None
    resolved = _resolve_coef_intercept(final_est)
    if resolved is None:
        return None
    coef_2d, intercepts = resolved
    n_rows, n_cols = coef_2d.shape
    if x_feature_idx < 0 or y_feature_idx < 0:
        return None
    if x_feature_idx >= n_cols or y_feature_idx >= n_cols:
        return None

    est_name = type(final_est).__name__
    ref = np.asarray(X, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] == 0:
        return None
    pinned = _pinned_vector(ref, pinning_mode, current_row_values)

    raw_classes = getattr(final_est, "classes_", None)
    try:
        classes: list[Any] = (
            [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw_classes).ravel())]
            if raw_classes is not None
            else []
        )
    except (TypeError, ValueError):
        classes = []

    # -- family / task dispatch -------------------------------------------
    if est_name in _ELASTIC_NAMES:
        family, task = "elasticnet", "regression"
    elif est_name in _L1_NAMES:
        family, task = "lasso", "regression"
    elif est_name in _RIDGE_NAMES:
        family, task = "ridge", "regression"
    elif est_name in _RIDGE_CLF_NAMES:
        family, task = "ridge_classifier", "classification"
    elif est_name in ("LogisticRegression", "LogisticRegressionCV"):
        family, task = "logistic", "classification"
    elif est_name in ("LinearRegression",):
        family, task = "ols", "regression"
    elif est_name in _SGD_CLF_NAMES:
        family, task = "sgd", "classification"
    elif est_name in _SGD_REG_NAMES:
        family, task = "sgd", "regression"
    elif len(classes) >= 2:
        family, task = "logistic", "classification"  # generic linear scorer
    elif n_rows >= 1 and coef_2d.shape[1] == ref.shape[1]:
        family, task = "ols", "regression"  # generic linear regressor
    else:
        return None

    is_binary = len(classes) == 2 if classes else n_rows == 1
    is_multiclass = task == "classification" and not is_binary and n_rows > 1

    def _beff(w: np.ndarray, b: float) -> float:
        total = float(b)
        for j in range(min(w.size, pinned.size)):
            if j == x_feature_idx or j == y_feature_idx:
                continue
            total += float(w[j]) * float(pinned[j])
        return total

    # Primary (binary / single-row) weight vector.
    w0 = np.asarray(coef_2d[0], dtype=np.float64)
    b0 = float(intercepts[0]) if intercepts.size else 0.0

    if task == "regression":
        w1 = float(w0[x_feature_idx]) if x_feature_idx < w0.size else 0.0
        w2 = float(w0[y_feature_idx]) if y_feature_idx < w0.size else 0.0
        if x_feature_idx == y_feature_idx:
            w2 = 0.0
        b_eff = _beff(w0, b0)
    elif is_multiclass:
        # Overview line uses the first class row; exact per-pair lines below.
        w1 = float(w0[x_feature_idx])
        w2 = float(w0[y_feature_idx])
        b_eff = _beff(w0, b0)
    else:
        w1 = float(w0[x_feature_idx]) if x_feature_idx < w0.size else 0.0
        w2 = float(w0[y_feature_idx]) if y_feature_idx < w0.size else 0.0
        if x_feature_idx == y_feature_idx:
            w2 = 0.0
        b_eff = _beff(w0, b0)

    norm12 = math.hypot(w1, w2)
    normal = [w1 / norm12, w2 / norm12] if norm12 > ZERO_TOL else [0.0, 0.0]
    is_x_zeroed = abs(w1) <= ZERO_TOL
    is_y_zeroed = abs(w2) <= ZERO_TOL
    both_zeroed = bool(is_x_zeroed and is_y_zeroed)
    full_w = w0[: min(w0.size, pinned.size)]
    sparsity_ratio = float(np.mean(np.abs(full_w) <= ZERO_TOL)) if full_w.size else 1.0
    corridor_width = (2.0 / norm12) if norm12 > ZERO_TOL else None

    # World conversion helper (lazy import keeps the module dependency-light).
    def _to_world(seg_raw: list[float] | None) -> list[float] | None:
        if seg_raw is None or x_scale is None or y_scale is None:
            return None
        try:
            from app.core.math_3d import WorldCoordinateScaler as _WCS

            wx = _WCS.to_world_coords(np.asarray([seg_raw[0], seg_raw[2]], dtype=np.float32), x_scale)
            wy = _WCS.to_world_coords(np.asarray([seg_raw[1], seg_raw[3]], dtype=np.float32), y_scale)
            return [round(float(wx[0]), 4), round(float(wy[0]), 4), round(float(wx[1]), 4), round(float(wy[1]), 4)]
        except (TypeError, ValueError, AttributeError):
            return None

    def _line(kind: str, level: float, lw1: float, lw2: float, lb: float, label: str,
              ca: int | None = None, cb: int | None = None) -> Linear2DLine:
        raw = _clip_line_to_rect(lw1, lw2, lb, level, x_range_raw, y_range_raw)
        seg_raw = [round(float(v), 6) for v in raw] if raw is not None else None
        return Linear2DLine(
            kind=kind, level=float(level), segment_raw=seg_raw,
            segment_world=_to_world(seg_raw), label=label, class_a=ca, class_b=cb,
        )

    lines: list[Linear2DLine] = []
    decision_threshold = float("nan")
    margin_level: float | None = None
    l1_ratio: float | None = None
    alpha: float | None = None
    penalty: str | None = None
    slope_w1: float | None = None
    x_sweep_intercept: float | None = None

    if task == "regression":
        decision_threshold = float("nan")
        slope_w1 = float(w1)
        # ŷ(x1) at pinned background: w1*x1 + (b_eff + w2*pinned_y).
        pinned_y = float(pinned[y_feature_idx]) if y_feature_idx < pinned.size else 0.0
        x_sweep_intercept = float(b_eff + (w2 * pinned_y if x_feature_idx != y_feature_idx else 0.0))
        if family == "elasticnet":
            l1_ratio = _finite_float(getattr(final_est, "l1_ratio_", getattr(final_est, "l1_ratio", 0.5)), 0.5)
            l1_ratio = min(1.0, max(0.0, l1_ratio))
            alpha = _finite_float(getattr(final_est, "alpha_", getattr(final_est, "alpha", 1.0)), 1.0)
            penalty = "ElasticNet (L1+L2)"
        elif family == "lasso":
            alpha = _finite_float(getattr(final_est, "alpha_", getattr(final_est, "alpha", 1.0)), 1.0)
            penalty = "L1 (Lasso)"
        elif family == "ridge":
            alpha = _finite_float(getattr(final_est, "alpha_", getattr(final_est, "alpha", 1.0)), 1.0)
            penalty = "L2 (Ridge)"
        elif family == "sgd":
            margin_level = _finite_float(getattr(final_est, "epsilon", 0.1), 0.1)
            penalty = str(getattr(final_est, "loss", "squared_error") or "squared_error")
        if not both_zeroed:
            # No drawable line lives in the (x1, x2) plane for regressors —
            # the plane ŷ(x1, x2) is rendered exactly on the client as
            # parallel elevation guides derived from (w1, w2, b_eff) at the
            # contour's elevation thresholds (see surfaces_2d/linear2dMath).
            # The 1D X-sweep ŷ = slope_w1*x1 + x_sweep_intercept is exact.
            pass
    else:
        if family in ("logistic",) or (family == "sgd" and str(getattr(final_est, "loss", "")) in _PROBA_LOSSES):
            decision_threshold = 0.5
        else:
            # Score-space families (RidgeClassifier, hinge-SGD): f(x) = 0.
            decision_threshold = 0.0

        if is_multiclass:
            rows = min(n_rows, MAX_PAIRWISE_CLASSES)
            for a in range(rows):
                for b_ in range(a + 1, rows):
                    dw = np.asarray(coef_2d[a], dtype=np.float64) - np.asarray(coef_2d[b_], dtype=np.float64)
                    db = (float(intercepts[a]) if a < intercepts.size else 0.0) - (
                        float(intercepts[b_]) if b_ < intercepts.size else 0.0)
                    dw1 = float(dw[x_feature_idx]) if x_feature_idx < dw.size else 0.0
                    dw2 = float(dw[y_feature_idx]) if y_feature_idx < dw.size else 0.0
                    if x_feature_idx == y_feature_idx:
                        dw2 = 0.0
                    db_eff = db + sum(
                        float(dw[j]) * float(pinned[j])
                        for j in range(min(dw.size, pinned.size))
                        if j != x_feature_idx and j != y_feature_idx
                    )
                    la = str(classes[a]) if a < len(classes) else str(a)
                    lb = str(classes[b_]) if b_ < len(classes) else str(b_)
                    line = _line("pairwise", 0.0, dw1, dw2, db_eff, f"z({la}) = z({lb})", ca=a, cb=b_)
                    if line.segment_raw is not None:
                        lines.append(line)
        elif not both_zeroed:
            if decision_threshold == 0.5:
                lines.append(_line("decision", 0.0, w1, w2, b_eff, "P = 0.50"))
                for p in CONFIDENCE_PROBAS:
                    # Parallel isoline w·x + b_eff = logit(P).
                    t = math.log(p / (1.0 - p))
                    raw = _clip_line_to_rect(w1, w2, b_eff, t, x_range_raw, y_range_raw)
                    if raw is not None:
                        seg_raw = [round(float(v), 6) for v in raw]
                        lines.append(Linear2DLine(
                            kind="confidence", level=float(p),
                            segment_raw=seg_raw, segment_world=_to_world(seg_raw),
                            label=f"P = {p:.2f}",
                        ))
            else:
                lines.append(_line("decision", 0.0, w1, w2, b_eff, "f(x) = 0"))

        # Margin corridor (score-space families + native hinge SGD).
        wants_margin = family in ("sgd", "ridge_classifier") and task == "classification" and not is_multiclass
        if wants_margin and not both_zeroed:
            margin_level = MARGIN_LEVEL
            pos = _line("margin_pos", MARGIN_LEVEL, w1, w2, b_eff, "z = +1")
            neg = _line("margin_neg", -MARGIN_LEVEL, w1, w2, b_eff, "z = −1")
            if pos.segment_raw is not None:
                lines.append(pos)
            if neg.segment_raw is not None:
                lines.append(neg)
        elif family == "logistic":
            margin_level = None

        if family == "elasticnet":
            l1_ratio = _finite_float(getattr(final_est, "l1_ratio_", getattr(final_est, "l1_ratio", 0.5)), 0.5)
            l1_ratio = min(1.0, max(0.0, l1_ratio))
            alpha = _finite_float(getattr(final_est, "alpha_", getattr(final_est, "alpha", 1.0)), 1.0)
            penalty = "ElasticNet (L1+L2)"
        elif family == "lasso":
            alpha = _finite_float(getattr(final_est, "alpha_", getattr(final_est, "alpha", 1.0)), 1.0)
            penalty = "L1 (Lasso)"
        elif family == "ridge" or family == "ridge_classifier":
            alpha = _finite_float(getattr(final_est, "alpha_", getattr(final_est, "alpha", 1.0)), 1.0)
            penalty = "L2 (Ridge)"
        elif family == "sgd":
            penalty = str(getattr(final_est, "loss", "perceptron") or "perceptron")

    xn, yn = _short_name(x_feature_name), _short_name(y_feature_name)
    if task == "regression":
        formula = f"ŷ = {w1:.4g}·{xn} + {w2:.4g}·{yn} + {b_eff:.4g}"
    elif is_multiclass:
        formula = f"argmax_c z_c, z_c = w_c·x + b_c ({n_rows} classes)"
    else:
        thr = "P=0.5" if decision_threshold == 0.5 else "f(x)=0"
        formula = f"{w1:.4g}·{xn} + {w2:.4g}·{yn} + {b_eff:.4g} = 0  ({thr})"

    return Linear2DGeometry(
        family=family,
        task=task,  # type: ignore[arg-type]
        x_feature=str(x_feature_name),
        y_feature=str(y_feature_name),
        x_feature_idx=int(x_feature_idx),
        y_feature_idx=int(y_feature_idx),
        w1=float(w1),
        w2=float(w2),
        b_eff=float(b_eff),
        decision_threshold=float(decision_threshold),
        normal_vector=[float(normal[0]), float(normal[1])],
        corridor_width=float(corridor_width) if corridor_width is not None else None,
        margin_level=float(margin_level) if margin_level is not None else None,
        is_x_zeroed=bool(is_x_zeroed),
        is_y_zeroed=bool(is_y_zeroed),
        both_zeroed=bool(both_zeroed),
        sparsity_ratio=float(sparsity_ratio),
        l1_ratio=float(l1_ratio) if l1_ratio is not None else None,
        alpha=float(alpha) if alpha is not None else None,
        penalty=penalty,
        slope_w1=float(slope_w1) if slope_w1 is not None else None,
        x_sweep_intercept=float(x_sweep_intercept) if x_sweep_intercept is not None else None,
        formula=formula,
        is_multiclass=bool(is_multiclass),
        n_classes=int(n_rows if task == "classification" else 0),
        lines=lines,
    )
