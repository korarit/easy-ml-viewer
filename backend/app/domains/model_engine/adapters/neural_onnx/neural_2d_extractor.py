"""Phase 47 — 2D decision geometry for Neural Networks, Pipelines & ONNX.

Complements the universal Marching-Squares contour engine (Phase 41) with
the exact overlays that make this model group recognizable in the 2D
planar viewport:

- **MLP** (``MLPClassifier`` / ``MLPRegressor``): first-hidden-layer
  switching hyperplanes ``z_k = w1*x + w2*y + b_eff = 0`` clipped to the
  slice rect, plus Dead-ReLU flags (linear min/max over the rect corners)
  and the active-neuron ratio.
- **Scikit-Learn Pipeline**: ``[Raw Space] <-> [Scaled Space]`` toggle
  metadata — per-axis ``(mu, sigma)`` / ``(min, scale)`` for
  ``StandardScaler`` / ``MinMaxScaler`` / ``RobustScaler`` prefixes that
  preserve feature order. Anything fancier (ColumnTransformer, OHE,
  polynomial expansion) returns ``has_toggle=False`` with an honest badge
  instead of a fabricated mapping.
- **ONNX Classical ML** (``ai.onnx.ml``): operator summary (first
  estimator node, chain, tree counts) — the dense contour already is the
  universal decision map for these graphs.
- **ONNX Deep Learning**: 2D latent SVD projection plane — the penultimate
  activation ``H`` of a sample of rows is centered and projected with a
  truncated SVD to ``Z in R^{N x 2}`` (float32, <=200 points). When an
  ``onnxruntime`` probed session can be built in-memory the activations
  are real; otherwise the extractor reports the operator chain with
  ``latent_available=False`` and the client keeps the contour fallback.

Conventions (shared with Phase 42–46 extractors):

- Coordinates are **raw feature units**; world mapping uses
  ``WorldCoordinateScaler``. World floats rounded to 2 decimals; raw to 6.
- Splits on off-axis features follow the pinned background (median row of
  ``X``); pipelined estimators live in *transformed* space, so the MLP
  branch returns ``None`` for pipelined models and the caller falls back
  to the sampled contour.
- Bounded output: hyperplane segments capped at ``MAX_HYPERPLANES`` with
  an explicit truncation flag; latent points capped at ``MAX_LATENT_POINTS``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

MAX_HYPERPLANES = 64
MAX_LATENT_POINTS = 200
LATENT_SAMPLE_ROWS = 200

_MLP_NAMES = {"MLPClassifier", "MLPRegressor"}
_CLASSICAL_OPS = frozenset({
    "TreeEnsembleClassifier", "TreeEnsembleRegressor", "TreeEnsemble",
    "LinearClassifier", "LinearRegressor",
    "SVMClassifier", "SVMRegressor",
})
_DEEP_HINT_OPS = frozenset({
    "Gemm", "MatMul", "Conv", "Relu", "Gelu", "Silu", "Sigmoid", "Tanh",
    "BatchNormalization", "LayerNormalization", "Attention", "Add",
    "Softmax", "LeakyRelu", "Elu",
})


class MlpHyperplane2D(BaseModel):
    """One first-layer switching line ``z_k = 0`` clipped to the slice rect."""

    neuron_idx: int = 0
    w1: float = 0.0
    w2: float = 0.0
    b_eff: float = 0.0
    segment_raw: list[float] | None = Field(
        default=None, description="[x1, y1, x2, y2] in raw units; null when off-slice/degenerate")
    segment_world: list[float] | None = Field(
        default=None, description="[x1, y1, x2, y2] in world coords [-5, +5]")
    is_dead: bool = False
    is_always_active: bool = False


class PipelineSpace2D(BaseModel):
    """Raw <-> Scaled toggle metadata for one pipeline prefix."""

    has_toggle: bool = False
    scaler_kind: str = "none"
    inner_family: str = ""
    inner_algorithm: str = ""
    mu_x: float | None = None
    sigma_x: float | None = None
    mu_y: float | None = None
    sigma_y: float | None = None
    scaled_x_range: list[float] | None = None
    scaled_y_range: list[float] | None = None
    formula: str = ""


class OnnxLatent2D(BaseModel):
    """ONNX summary + optional 2D latent SVD projection plane."""

    operator: str = ""
    chain: list[str] = Field(default_factory=list)
    n_nodes: int = 0
    latent_available: bool = False
    latent_points_2d: list[list[float]] = Field(default_factory=list)
    latent_explained_variance: list[float] = Field(default_factory=list)
    penultimate_width: int = 0
    formula: str = ""


class Neural2DGeometry(BaseModel):
    """Exact 2D planar geometry for the neural / pipeline / ONNX group."""

    family: str = Field(description="mlp | pipeline | onnx_classical | onnx_deep")
    algorithm: str = ""
    x_feature: str = ""
    y_feature: str = ""
    x_feature_idx: int = 0
    y_feature_idx: int = 0
    activation: str | None = None
    hyperplanes: list[MlpHyperplane2D] = Field(default_factory=list)
    hyperplanes_truncated: bool = False
    dead_count: int = 0
    active_count: int = 0
    active_ratio: float | None = None
    pipeline: PipelineSpace2D | None = None
    onnx: OnnxLatent2D | None = None
    no_axis_splits: bool = False
    formula: str = ""


def _is_pipeline(model_obj: Any) -> bool:
    return isinstance(getattr(model_obj, "steps", None), list) and bool(model_obj.steps)


def _unwrap(model_obj: Any) -> Any:
    seen = 0
    while seen < 8:
        seen += 1
        best = getattr(model_obj, "best_estimator_", None)
        if best is not None:
            model_obj = best
            continue
        ccs = getattr(model_obj, "calibrated_classifiers_", None)
        if isinstance(ccs, list) and ccs:
            inner = getattr(ccs[0], "estimator", getattr(ccs[0], "base_estimator", None))
            if inner is not None:
                model_obj = inner
                continue
        break
    return getattr(model_obj, "_final_estimator", model_obj)


def _pinned_vector(X: np.ndarray) -> np.ndarray:
    ref = np.asarray(X, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        base = np.nanmedian(ref, axis=0)
    base = np.asarray(base, dtype=np.float64).ravel()
    return np.where(np.isfinite(base), base, 0.0)


def _world_seg(seg_raw: list[float], x_scale: Any, y_scale: Any) -> list[float] | None:
    try:
        from app.core.math_3d import WorldCoordinateScaler as _WCS

        wx = _WCS.to_world_coords(
            np.asarray([seg_raw[0], seg_raw[2]], dtype=np.float32), x_scale)
        wy = _WCS.to_world_coords(
            np.asarray([seg_raw[1], seg_raw[3]], dtype=np.float32), y_scale)
        return [round(float(wx[0]), 2), round(float(wy[0]), 2),
                round(float(wx[1]), 2), round(float(wy[1]), 2)]
    except (TypeError, ValueError, AttributeError):
        return None


def _clip_line_to_rect(
    w1: float, w2: float, b: float,
    xr: tuple[float, float], yr: tuple[float, float],
) -> list[float] | None:
    """Clip ``w1*x + w2*y + b = 0`` to the rect; None when missing/degenerate."""
    if not (np.isfinite(w1) and np.isfinite(w2) and np.isfinite(b)):
        return None
    norm = math.hypot(w1, w2)
    if norm < 1e-12:
        return None
    xmin, xmax = xr
    ymin, ymax = yr
    pts: list[tuple[float, float]] = []
    if abs(w2) > 1e-12:
        for x in (xmin, xmax):
            y = -(w1 * x + b) / w2
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                pts.append((x, min(max(y, ymin), ymax)))
    if abs(w1) > 1e-12:
        for y in (ymin, ymax):
            x = -(w2 * y + b) / w1
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                pts.append((min(max(x, xmin), xmax), y))
    # Deduplicate corner touches.
    uniq: list[tuple[float, float]] = []
    for p in pts:
        if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    # Farthest pair spans the slice.
    best = (uniq[0], uniq[1])
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = math.hypot(uniq[i][0] - uniq[j][0], uniq[i][1] - uniq[j][1])
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])
    if best_d < 1e-12:
        return None
    return [round(best[0][0], 6), round(best[0][1], 6),
            round(best[1][0], 6), round(best[1][1], 6)]


def project_latent_svd(H: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Centered truncated-SVD projection ``(N, D) -> (N, 2)`` (pure NumPy).

    Returns ``(points_2d float32, explained_variance_ratio[2])``. Ranks are
    guarded: constant / degenerate inputs collapse to zeros instead of NaNs.
    """
    mat = np.asarray(H, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.float32), [0.0, 0.0]
    mat = np.where(np.isfinite(mat), mat, 0.0)
    centered = mat - np.mean(mat, axis=0, keepdims=True)
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((mat.shape[0], 2), dtype=np.float32), [0.0, 0.0]
    var = (s ** 2) / max(mat.shape[0] - 1, 1)
    total = float(np.sum(var)) if var.size else 0.0
    if total < 1e-12 or vt.shape[0] == 0:
        return np.zeros((mat.shape[0], 2), dtype=np.float32), [0.0, 0.0]
    k = min(2, int(vt.shape[0]))
    proj = centered @ vt[:k].T
    if k < 2:
        proj = np.pad(proj, ((0, 0), (0, 2 - k)))
    evr = [round(float(var[i] / total), 4) if i < var.size else 0.0 for i in range(2)]
    return np.ascontiguousarray(proj[:, :2], dtype=np.float32), evr


def _is_onnx_proto(obj: Any) -> bool:
    graph = getattr(obj, "graph", None)
    return graph is not None and hasattr(graph, "node") and hasattr(graph, "input")


def _onnx_chain(proto: Any) -> tuple[list[str], int]:
    try:
        chain = []
        for node in list(proto.graph.node):
            domain = str(getattr(node, "domain", "") or "ai.onnx")
            chain.append(f"{domain}::{node.op_type}")
        return chain, len(chain)
    except (AttributeError, TypeError):
        return [], 0


def _onnx_family(proto: Any) -> str:
    try:
        for node in list(proto.graph.node):
            if str(getattr(node, "domain", "")) == "ai.onnx.ml" \
                    and str(node.op_type) in _CLASSICAL_OPS:
                return "onnx_classical"
    except (AttributeError, TypeError):
        pass
    return "onnx_deep"


def _first_classical_op(proto: Any) -> str:
    try:
        for node in list(proto.graph.node):
            if str(getattr(node, "domain", "")) == "ai.onnx.ml" \
                    and str(node.op_type) in _CLASSICAL_OPS:
                return str(node.op_type)
    except (AttributeError, TypeError):
        pass
    return ""


class Neural2DExtractor:
    """Stateless 2D geometry builder for MLP / Pipeline / ONNX."""

    @staticmethod
    def extract_mlp_first_layer_hyperplanes(
        mlp_model: Any,
        x_idx: int,
        y_idx: int,
        pinned_features: np.ndarray,
    ) -> list[dict[str, float]]:
        """Spec §5.2 prototype — raw ``(w1, w2, b_eff)`` per first-layer neuron."""
        w_all = np.asarray(mlp_model.coefs_[0], dtype=np.float64)
        b_all = np.asarray(mlp_model.intercepts_[0], dtype=np.float64).ravel()
        w1 = w_all[x_idx, :]
        w2 = w_all[y_idx, :]
        mask = np.ones(w_all.shape[0], dtype=bool)
        mask[x_idx] = False
        mask[y_idx] = False
        other_contrib = np.dot(
            np.asarray(pinned_features, dtype=np.float64).ravel()[mask],
            w_all[mask, :],
        )
        b_eff = b_all + np.asarray(other_contrib, dtype=np.float64).ravel()
        return [
            {"neuron_idx": int(i), "w1": float(w1[i]),
             "w2": float(w2[i]), "b_eff": float(b_eff[i])}
            for i in range(len(b_eff))
        ]

    # -- public entry point -------------------------------------------------
    @staticmethod
    def extract_neural_2d_geometry(
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
        feature_names: list[str] | None = None,
        active_row: np.ndarray | None = None,
    ) -> Neural2DGeometry | None:
        if _is_pipeline(model_obj):
            return Neural2DExtractor._pipeline(
                model_obj, X, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, x_range_raw, y_range_raw)
        est = _unwrap(model_obj)
        if _is_onnx_proto(est) or _is_onnx_proto(model_obj):
            proto = est if _is_onnx_proto(est) else model_obj
            return Neural2DExtractor._onnx(
                proto, X, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name)
        est_name = type(est).__name__
        if est_name in _MLP_NAMES:
            return Neural2DExtractor._mlp(
                est, est_name, X, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, x_range_raw, y_range_raw,
                x_scale, y_scale)
        # Foreign model (trees, boosting, linear…): caller keeps the contour.
        return None

    # -- MLP -------------------------------------------------------------------
    @staticmethod
    def _mlp(est: Any, est_name: str, X: np.ndarray,
             x_idx: int, y_idx: int, xn: str, yn: str,
             xr: tuple[float, float], yr: tuple[float, float],
             x_scale: Any, y_scale: Any) -> Neural2DGeometry | None:
        try:
            coefs = list(getattr(est, "coefs_", []))
            intercepts = list(getattr(est, "intercepts_", []))
            if not coefs or not intercepts:
                return None
            w_all = np.asarray(coefs[0], dtype=np.float64)
            b_all = np.asarray(intercepts[0], dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        ref = np.asarray(X, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[0] == 0:
            return None
        d = int(ref.shape[1])
        if not (0 <= x_idx < d and 0 <= y_idx < d) or w_all.shape[0] != d:
            return None
        if x_idx == y_idx and d >= 2:
            return None
        n_hidden = int(w_all.shape[1])
        if n_hidden == 0 or b_all.size != n_hidden:
            return None
        hidden_name = str(getattr(est, "activation", "relu") or "relu").lower()
        if hidden_name not in ("relu", "tanh", "logistic", "identity"):
            hidden_name = "relu"
        pinned = _pinned_vector(ref)
        try:
            raw_planes = Neural2DExtractor.extract_mlp_first_layer_hyperplanes(
                est, x_idx, y_idx, pinned)
        except (TypeError, ValueError, IndexError, AttributeError):
            return None
        xr_c = (math.floor(min(xr[0], xr[1]) * 1e6) / 1e6,
                math.ceil(max(xr[0], xr[1]) * 1e6) / 1e6)
        yr_c = (math.floor(min(yr[0], yr[1]) * 1e6) / 1e6,
                math.ceil(max(yr[0], yr[1]) * 1e6) / 1e6)
        corners = [(xr_c[0], yr_c[0]), (xr_c[0], yr_c[1]),
                   (xr_c[1], yr_c[0]), (xr_c[1], yr_c[1])]
        hyperplanes: list[MlpHyperplane2D] = []
        truncated = False
        dead = 0
        for i, plane in enumerate(raw_planes):
            if i >= MAX_HYPERPLANES:
                truncated = True
                break
            try:
                w1 = round(float(plane["w1"]), 6)
                w2 = round(float(plane["w2"]), 6)
                b = round(float(plane["b_eff"]), 6)
            except (TypeError, ValueError, KeyError):
                continue
            seg_raw = _clip_line_to_rect(w1, w2, b, xr_c, yr_c)
            seg_w = _world_seg(seg_raw, x_scale, y_scale) if seg_raw else None
            zs = [w1 * cx + w2 * cy + b for cx, cy in corners]
            finite = [z for z in zs if np.isfinite(z)]
            if hidden_name == "relu" and finite:
                is_dead = bool(max(finite) < 0)
                is_active = bool(min(finite) > 0)
            else:
                is_dead, is_active = False, True
            if is_dead:
                dead += 1
            hyperplanes.append(MlpHyperplane2D(
                neuron_idx=int(plane.get("neuron_idx", i)),
                w1=w1, w2=w2, b_eff=b,
                segment_raw=seg_raw, segment_world=seg_w,
                is_dead=bool(is_dead), is_always_active=bool(is_active)))
        total = len(hyperplanes)
        active = total - dead
        no_axis = all(h.segment_raw is None for h in hyperplanes) if hyperplanes else True
        return Neural2DGeometry(
            family="mlp", algorithm=est_name,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            activation=hidden_name,
            hyperplanes=hyperplanes, hyperplanes_truncated=bool(truncated),
            dead_count=int(dead), active_count=int(active),
            active_ratio=round(active / total, 4) if total else None,
            no_axis_splits=bool(no_axis),
            formula=(f"layer-1 cuts z_k=0 ({active}/{total} active, {hidden_name}); "
                     f"non-linear boundary folds along these lines"),
        )

    # -- Pipeline ---------------------------------------------------------------
    @staticmethod
    def _pipeline(model_obj: Any, X: np.ndarray,
                  x_idx: int, y_idx: int, xn: str, yn: str,
                  xr: tuple[float, float], yr: tuple[float, float]) -> Neural2DGeometry | None:
        try:
            steps = list(getattr(model_obj, "steps", []))
        except TypeError:
            return None
        if not steps:
            return None
        final_est = steps[-1][1] if isinstance(steps[-1], (tuple, list)) else steps[-1]
        inner_family = type(final_est).__name__
        # Only order-preserving single scalers qualify for an exact toggle.
        scaler_kind = "none"
        mu_x = sigma_x = mu_y = sigma_y = None
        has_toggle = False
        pre = steps[:-1]
        if len(pre) == 1:
            trans = pre[0][1] if isinstance(pre[0], (tuple, list)) else pre[0]
            tname = type(trans).__name__
            try:
                if tname == "StandardScaler":
                    mean = np.asarray(getattr(trans, "mean_", []), dtype=np.float64).ravel()
                    scale = np.asarray(getattr(trans, "scale_", []), dtype=np.float64).ravel()
                    n = int(np.asarray(X, dtype=np.float64).shape[1])
                    if mean.size == n and scale.size == n \
                            and 0 <= x_idx < n and 0 <= y_idx < n:
                        sx = float(scale[x_idx])
                        sy = float(scale[y_idx])
                        if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
                            mu_x, sigma_x = float(mean[x_idx]), sx
                            mu_y, sigma_y = float(mean[y_idx]), sy
                            scaler_kind, has_toggle = "standard", True
                elif tname == "MinMaxScaler":
                    dmin = np.asarray(getattr(trans, "data_min_", []), dtype=np.float64).ravel()
                    dmax = np.asarray(getattr(trans, "data_max_", []), dtype=np.float64).ravel()
                    n = int(np.asarray(X, dtype=np.float64).shape[1])
                    if dmin.size == n and dmax.size == n \
                            and 0 <= x_idx < n and 0 <= y_idx < n:
                        mu_x, sigma_x = float(dmin[x_idx]), float(dmax[x_idx] - dmin[x_idx])
                        mu_y, sigma_y = float(dmin[y_idx]), float(dmax[y_idx] - dmin[y_idx])
                        if np.isfinite(sigma_x) and np.isfinite(sigma_y) \
                                and sigma_x > 0 and sigma_y > 0:
                            scaler_kind, has_toggle = "minmax", True
                elif tname == "RobustScaler":
                    center = np.asarray(getattr(trans, "center_", []), dtype=np.float64).ravel()
                    scale = np.asarray(getattr(trans, "scale_", []), dtype=np.float64).ravel()
                    n = int(np.asarray(X, dtype=np.float64).shape[1])
                    if center.size == n and scale.size == n \
                            and 0 <= x_idx < n and 0 <= y_idx < n:
                        sx = float(scale[x_idx])
                        sy = float(scale[y_idx])
                        if np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0:
                            mu_x, sigma_x = float(center[x_idx]), sx
                            mu_y, sigma_y = float(center[y_idx]), sy
                            scaler_kind, has_toggle = "robust", True
            except (TypeError, ValueError, AttributeError):
                has_toggle = False
        scaled_x = scaled_y = None
        formula: str
        if has_toggle and mu_x is not None and sigma_x \
                and mu_y is not None and sigma_y:
            scaled_x = [round((xr[0] - mu_x) / sigma_x, 4),
                        round((xr[1] - mu_x) / sigma_x, 4)]
            scaled_y = [round((yr[0] - mu_y) / sigma_y, 4),
                        round((yr[1] - mu_y) / sigma_y, 4)]
            formula = (f"x'=(x-{mu_x:.4g})/{sigma_x:.4g}, "
                       f"y'=(y-{mu_y:.4g})/{sigma_y:.4g} [{scaler_kind}] → {inner_family}")
        else:
            scaler_kind = "none"
            has_toggle = False
            formula = (f"preprocessing changes width/order "
                       f"({len(pre)} step(s)) → raw/scaled toggle unavailable; "
                       f"contour shown in raw space → {inner_family}")
        return Neural2DGeometry(
            family="pipeline", algorithm="Pipeline",
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            pipeline=PipelineSpace2D(
                has_toggle=bool(has_toggle), scaler_kind=scaler_kind,
                inner_family=str(inner_family), inner_algorithm=str(inner_family),
                mu_x=mu_x, sigma_x=sigma_x, mu_y=mu_y, sigma_y=sigma_y,
                scaled_x_range=scaled_x, scaled_y_range=scaled_y,
                formula=formula),
            no_axis_splits=not has_toggle,
            formula=formula,
        )

    # -- ONNX --------------------------------------------------------------------
    @staticmethod
    def _onnx(proto: Any, X: np.ndarray,
              x_idx: int, y_idx: int, xn: str, yn: str) -> Neural2DGeometry | None:
        chain, n_nodes = _onnx_chain(proto)
        family = _onnx_family(proto)
        if family == "onnx_classical":
            op = _first_classical_op(proto)
            return Neural2DGeometry(
                family="onnx_classical", algorithm=op or "ONNX-Classical",
                x_feature=str(xn), y_feature=str(yn),
                x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
                onnx=OnnxLatent2D(
                    operator=op, chain=chain[:12], n_nodes=int(n_nodes),
                    latent_available=False,
                    formula=(f"{op or 'classical graph'} "
                             f"({n_nodes} nodes); universal contour is the exact 2D map")),
                formula=f"ai.onnx.ml::{op} universal contour ({n_nodes} nodes)",
            )
        # Deep graph: try a real probed ORT session, else operator summary.
        latent_pts: list[list[float]] = []
        evr: list[float] = [0.0, 0.0]
        width = 0
        available = False
        try:
            from app.domains.model_engine.adapters.neural_onnx.onnx_deep import (  # noqa: PLC0415
                build_probed_session,
                normalize_intermediate,
                run_session_outputs,
            )
            resolved = build_probed_session(proto)
            if resolved.penultimate_name is not None:
                ref = np.ascontiguousarray(
                    np.asarray(X, dtype=np.float32)[:LATENT_SAMPLE_ROWS])
                by_name = dict(zip(resolved.output_names,
                                   run_session_outputs(resolved, ref)))
                pen = by_name.get(resolved.penultimate_name, None)
                if pen is not None:
                    normed = normalize_intermediate(pen, int(ref.shape[0]))
                    flat = np.asarray(normed, dtype=np.float64)
                    if flat.ndim > 2:
                        flat = flat.reshape(flat.shape[0], -1)
                    width = int(flat.shape[1]) if flat.ndim == 2 else 0
                    pts, evr = project_latent_svd(flat)
                    step = max(1, int(len(pts) // MAX_LATENT_POINTS))
                    slim = pts[::step][:MAX_LATENT_POINTS]
                    latent_pts = [[round(float(a), 4), round(float(b), 4)]
                                  for a, b in slim.tolist()]
                    available = bool(latent_pts)
        except Exception:
            available = False
            latent_pts = []
        try:
            op0 = str(chain[-1]) if chain else "ONNX-Deep"
        except (IndexError, TypeError):
            op0 = "ONNX-Deep"
        return Neural2DGeometry(
            family="onnx_deep", algorithm=op0,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            onnx=OnnxLatent2D(
                operator=op0, chain=chain[:12], n_nodes=int(n_nodes),
                latent_available=bool(available),
                latent_points_2d=latent_pts,
                latent_explained_variance=list(evr),
                penultimate_width=int(width),
                formula=(f"penultimate latent SVD plane "
                         f"({width}D → 2D, {len(latent_pts)} pts)" if available
                         else f"deep graph ({n_nodes} nodes); latent probe unavailable — contour fallback")),
            no_axis_splits=not available,
            formula=(f"ONNX deep latent SVD plane ({len(latent_pts)} pts)" if available
                     else f"ONNX deep graph ({n_nodes} nodes); contour fallback"),
        )
