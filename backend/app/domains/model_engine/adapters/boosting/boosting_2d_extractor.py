"""Phase 46 — 2D decision geometry for Gradient Boosting Heavyweights.

Complements the universal Marching-Squares contour engine (Phase 41) with
the *exact* overlays that make each boosting library recognizable in the
2D planar viewport:

- **sklearn GBM** (``GradientBoostingClassifier/Regressor``): superimposed
  cuts of the first ``MAX_GBM_TREES`` regression stumps/trees — the dense
  contour already is the additive residual field, the overlay adds the
  mosaic structure (Phase 45 forest analogue).
- **HistGradientBoosting**: exact ``_bin_mapper.bin_thresholds_`` grid
  lines on both axes (255 quantized bins) plus first-iteration cuts and
  the active bin ``(bin_x, bin_y)`` of the query row.
- **XGBoost**: first-tree leaf-partition polygons (clipped boxes + cut
  segments from the JSON dump) plus the native C++ Exact TreeSHAP push
  vector ``(phi_x, phi_y)`` for the active row.
- **LightGBM**: first-tree asymmetric leaf-wise polygons from
  ``dump_model`` plus native ``pred_contrib`` push vector.
- **CatBoost**: symmetric oblivious checkerboard of tree 0 — cartesian
  ``(Kx+1) x (Ky+1)`` cells with bitwise leaf indexes — plus native C++
  SHAP push vector (numeric-only models).

Conventions (shared with Phase 42–45 extractors):

- Coordinates are **raw feature units**; world mapping uses
  ``WorldCoordinateScaler`` (``to_world_coords``). World floats are
  rounded to 2 decimals; raw floats to 6.
- Splits on off-axis features follow the branch matching the pinned
  background (median row of ``X``); pipelined estimators live in
  *transformed* space, so the extractor returns ``None`` and the caller
  falls back to the sampled contour.
- Bounded output (memory budget §5.1 / ≤150 KB contour SLA): cut segments
  are capped at ``MAX_SEGMENTS`` with an explicit truncation flag, leaf
  boxes at ``MAX_LEAVES``, oblivious cells at ``MAX_CELLS``. Bin edges
  ship full-resolution (≤255 floats per axis ≈ 2 KB) so the DoD
  "thresholds match the model 100%" holds exactly.
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

TREE_UNDEFINED = -2
MAX_SEGMENTS = 600
MAX_LEAVES = 64
MAX_CELLS = 64
MAX_CUTS_PER_AXIS = 7
MAX_GBM_TREES = 8
MAX_GBM_SEGMENTS_PER_TREE = 100
MAX_HIST_ITERS = 5

_SKLEARN_GBM_NAMES = {"GradientBoostingClassifier", "GradientBoostingRegressor"}
_HIST_NAMES = {"HistGradientBoostingClassifier", "HistGradientBoostingRegressor"}
_XGB_NAMES = {"XGBClassifier", "XGBRegressor", "XGBRFClassifier", "XGBRFRegressor"}
_LGBM_NAMES = {"LGBMClassifier", "LGBMRegressor", "LGBMRanker"}
_CB_NAMES = {"CatBoostClassifier", "CatBoostRegressor"}

_FNAME_RE = re.compile(r"^f(\d+)$")


class BoostCutSegment2D(BaseModel):
    """One axis-aligned cut, clipped to its node's live box."""

    axis: str = Field(description="'x' (vertical, x = t) | 'y' (horizontal, y = t)")
    threshold: float = 0.0
    node_id: int = 0
    tree_index: int = 0
    depth: int = 0
    segment_raw: list[float] = Field(description="[x1, y1, x2, y2] in raw feature units")
    segment_world: list[float] = Field(description="[x1, y1, x2, y2] in world coords [-5, +5]")


class BoostLeafBox2D(BaseModel):
    """One leaf's bounding box on the (X, Y) slice (first-tree polygons)."""

    leaf_id: int = 0
    tree_index: int = 0
    box_raw: list[float] = Field(description="[xmin, xmax, ymin, ymax] in raw units")
    box_world: list[float] = Field(description="[xmin, xmax, ymin, ymax] in world coords")
    leaf_value: float = 0.0
    rule: str = ""
    is_active: bool = False


class BoostShapVector2D(BaseModel):
    """Native TreeSHAP push vector for the active row on the (X, Y) plane."""

    phi_x: float = 0.0
    phi_y: float = 0.0
    magnitude: float = 0.0
    bias: float | None = None
    method: str = Field(description="xgb_pred_contribs | lgbm_pred_contrib | catboost_shap")


class ObliviousCell2D(BaseModel):
    """One checkerboard cell of the CatBoost symmetric grid (tree 0)."""

    cell_id: int = 0
    leaf_index: int = 0
    bitmask: str = ""
    box_raw: list[float] = Field(description="[xmin, xmax, ymin, ymax] in raw units")
    box_world: list[float] = Field(description="[xmin, xmax, ymin, ymax] in world coords")
    is_active: bool = False


class Boosting2DGeometry(BaseModel):
    """Exact 2D planar geometry for one boosting model on one axis pair."""

    family: str = Field(description="sklearn_gbm | hist_gbm | xgboost | lightgbm | catboost")
    algorithm: str = ""
    x_feature: str = ""
    y_feature: str = ""
    x_feature_idx: int = 0
    y_feature_idx: int = 0
    n_estimators: int = 0
    trees_exported: int = 0
    segments: list[BoostCutSegment2D] = Field(default_factory=list)
    segments_truncated: bool = False
    leaves: list[BoostLeafBox2D] = Field(default_factory=list)
    leaves_truncated: bool = False
    # HistGBM quantized bins: full-resolution raw edges + world line positions.
    bin_edges_x_raw: list[float] = Field(default_factory=list)
    bin_edges_y_raw: list[float] = Field(default_factory=list)
    bin_lines_x_world: list[float] = Field(default_factory=list)
    bin_lines_y_world: list[float] = Field(default_factory=list)
    active_bin: list[int] | None = Field(default=None, description="[bin_x, bin_y] of active row")
    # CatBoost oblivious checkerboard (tree 0).
    oblivious_x_cuts_raw: list[float] = Field(default_factory=list)
    oblivious_y_cuts_raw: list[float] = Field(default_factory=list)
    oblivious_x_lines_world: list[float] = Field(default_factory=list)
    oblivious_y_lines_world: list[float] = Field(default_factory=list)
    oblivious_cells: list[ObliviousCell2D] = Field(default_factory=list)
    oblivious_truncated: bool = False
    oblivious_rule: str = ""
    # Native TreeSHAP push vector for the active row (None when N/A).
    shap_vector: BoostShapVector2D | None = None
    no_axis_splits: bool = Field(
        default=False,
        description="True when no exported tree splits on X/Y (honest badge)",
    )
    formula: str = ""


def _is_pipeline(model_obj: Any) -> bool:
    return isinstance(getattr(model_obj, "steps", None), list) and bool(model_obj.steps)


def _unwrap(model_obj: Any) -> Any:
    """Follow GridSearchCV / CalibratedClassifierCV chains to the fitted estimator."""
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
            np.asarray([seg_raw[0], seg_raw[2]], dtype=np.float32), x_scale
        )
        wy = _WCS.to_world_coords(
            np.asarray([seg_raw[1], seg_raw[3]], dtype=np.float32), y_scale
        )
        return [round(float(wx[0]), 2), round(float(wy[0]), 2),
                round(float(wx[1]), 2), round(float(wy[1]), 2)]
    except (TypeError, ValueError, AttributeError):
        return None


def _world_line_positions(raw_vals: list[float], scale: Any) -> list[float]:
    try:
        from app.core.math_3d import WorldCoordinateScaler as _WCS

        arr = np.asarray(raw_vals, dtype=np.float32)
        w = _WCS.to_world_coords(arr, scale)
        return [round(float(v), 2) for v in np.asarray(w, dtype=np.float64).tolist()]
    except (TypeError, ValueError, AttributeError):
        return []


def _fmt_threshold(t: float) -> str:
    a = abs(t)
    if a != 0.0 and (a >= 10000 or a < 0.001):
        return f"{t:.2e}"
    return f"{round(float(t), 4)}"


def _feature_index(token: Any, booster_names: list[str] | None, width: int) -> int:
    if isinstance(token, (int, np.integer)):
        return int(token) if 0 <= int(token) < width else -1
    s = str(token)
    if booster_names and s in booster_names:
        return booster_names.index(s)
    m = _FNAME_RE.match(s)
    if m:
        try:
            j = int(m.group(1))
            return j if 0 <= j < width else -1
        except ValueError:
            return -1
    try:
        j = int(s)
        return j if 0 <= j < width else -1
    except (TypeError, ValueError):
        return -1


class _SklearnTreeWalker:
    """Pruned BFS over one fitted sklearn ``tree_`` on the (X, Y) slice."""

    def __init__(
        self,
        tree: Any,
        x_idx: int,
        y_idx: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        pinned: np.ndarray | None,
        feature_names: list[str] | None,
        x_scale: Any = None,
        y_scale: Any = None,
        tree_index: int = 0,
        max_segments: int = MAX_SEGMENTS,
        max_leaves: int = MAX_LEAVES,
        collect_leaves: bool = True,
    ) -> None:
        self._t = tree
        self._x_idx = x_idx
        self._y_idx = y_idx
        self._rect = (x_range[0], x_range[1], y_range[0], y_range[1])
        self._pinned = pinned
        self._names = feature_names or []
        self._xs = x_scale
        self._ys = y_scale
        self._ti = tree_index
        self._seg_cap = max_segments
        self._leaf_cap = max_leaves
        self._collect_leaves = collect_leaves
        self.segments: list[BoostCutSegment2D] = []
        self.leaves: list[BoostLeafBox2D] = []
        self.segments_truncated = False
        self.leaves_truncated = False

    def _name(self, f: int) -> str:
        if 0 <= f < len(self._names) and self._names[f]:
            return str(self._names[f])
        return f"feature_{f}"

    def _off_axis_value(self, f: int) -> float | None:
        if self._pinned is not None and 0 <= f < self._pinned.size:
            v = float(self._pinned[f])
            if np.isfinite(v):
                return v
        return None

    def run(self) -> None:
        feat = np.asarray(self._t.feature, dtype=np.int64).ravel()
        thr = np.asarray(self._t.threshold, dtype=np.float64).ravel()
        left = np.asarray(self._t.children_left, dtype=np.int64).ravel()
        right = np.asarray(self._t.children_right, dtype=np.int64).ravel()
        try:
            values = np.asarray(self._t.value, dtype=np.float64)
        except (TypeError, ValueError):
            values = np.zeros((feat.size, 1, 1))
        queue: deque[tuple[int, float, float, float, float, int, list[str]]] = deque(
            [(0, self._rect[0], self._rect[1], self._rect[2], self._rect[3], 0, [])]
        )
        while queue:
            node, xmin, xmax, ymin, ymax, depth, conds = queue.popleft()
            if not (0 <= node < feat.size):
                continue
            f = int(feat[node])
            if f == TREE_UNDEFINED:
                self._emit_leaf(node, xmin, xmax, ymin, ymax, conds, values)
                continue
            t = round(float(thr[node]), 6)
            if not np.isfinite(t):
                continue
            if f == self._x_idx:
                if xmin - 1e-12 <= t <= xmax + 1e-12:
                    tc = min(max(round(t, 6), xmin), xmax)
                    self._emit_segment("x", tc, node, depth, [tc, ymin, tc, ymax])
                    queue.append((int(right[node]), tc, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(tc)}"]))
                    queue.append((int(left[node]), xmin, tc, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(tc)}"]))
                elif t < xmin:
                    queue.append((int(right[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
                else:
                    queue.append((int(left[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
            elif f == self._y_idx:
                if ymin - 1e-12 <= t <= ymax + 1e-12:
                    tc = min(max(round(t, 6), ymin), ymax)
                    self._emit_segment("y", tc, node, depth, [xmin, tc, xmax, tc])
                    queue.append((int(right[node]), xmin, xmax, tc, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(tc)}"]))
                    queue.append((int(left[node]), xmin, xmax, ymin, tc, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(tc)}"]))
                elif t < ymin:
                    queue.append((int(right[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
                else:
                    queue.append((int(left[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
            else:
                v = self._off_axis_value(f)
                if v is None:
                    queue.append((int(left[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
                elif v <= t:
                    queue.append((int(left[node]), xmin, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(t)}"]))
                else:
                    queue.append((int(right[node]), xmin, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(t)}"]))

    def _emit_segment(self, axis: str, t: float, node: int, depth: int,
                      seg_raw: list[float]) -> None:
        if len(self.segments) >= self._seg_cap:
            self.segments_truncated = True
            return
        seg_w = _world_seg([round(float(v), 6) for v in seg_raw], self._xs, self._ys)
        if seg_w is None:
            return
        self.segments.append(BoostCutSegment2D(
            axis=axis, threshold=round(float(t), 6), node_id=int(node),
            tree_index=int(self._ti), depth=int(depth),
            segment_raw=[round(float(v), 6) for v in seg_raw], segment_world=seg_w))

    def _emit_leaf(self, node: int, xmin: float, xmax: float, ymin: float, ymax: float,
                   conds: list[str], values: np.ndarray) -> None:
        if not self._collect_leaves:
            return
        if len(self.leaves) >= self._leaf_cap:
            self.leaves_truncated = True
            return
        try:
            leaf_val = round(float(np.asarray(values[node]).ravel()[0]), 5)
        except (IndexError, TypeError, ValueError):
            leaf_val = 0.0
        box_raw = [round(xmin, 6), round(xmax, 6), round(ymin, 6), round(ymax, 6)]
        box_w = _world_seg([box_raw[0], box_raw[2], box_raw[1], box_raw[3]],
                           self._xs, self._ys)
        if box_w is None:
            return
        rule = " AND ".join(conds)
        if len(rule) > 400:
            rule = rule[:397] + "…"
        self.leaves.append(BoostLeafBox2D(
            leaf_id=int(node), tree_index=int(self._ti),
            box_raw=box_raw, box_world=[box_w[0], box_w[2], box_w[1], box_w[3]],
            leaf_value=float(leaf_val) if np.isfinite(leaf_val) else 0.0, rule=rule))


class _DumpTreeWalker:
    """Pruned BFS over a nested dump tree (XGBoost JSON / LightGBM dump_model).

    ``split_of(node)`` returns ``(feature_idx, threshold_or_None)``;
    categorical splits surface ``threshold=None`` and are treated as
    off-axis (pinned branch) so geometry never fabricates a cut.
    """

    def __init__(
        self,
        x_idx: int,
        y_idx: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        pinned: np.ndarray | None,
        feature_names: list[str] | None,
        x_scale: Any = None,
        y_scale: Any = None,
        tree_index: int = 0,
        max_segments: int = MAX_SEGMENTS,
        max_leaves: int = MAX_LEAVES,
    ) -> None:
        self._x_idx = x_idx
        self._y_idx = y_idx
        self._rect = (x_range[0], x_range[1], y_range[0], y_range[1])
        self._pinned = pinned
        self._names = feature_names or []
        self._xs = x_scale
        self._ys = y_scale
        self._ti = tree_index
        self._seg_cap = max_segments
        self._leaf_cap = max_leaves
        self.segments: list[BoostCutSegment2D] = []
        self.leaves: list[BoostLeafBox2D] = []
        self.segments_truncated = False
        self.leaves_truncated = False

    def _name(self, f: int) -> str:
        if 0 <= f < len(self._names) and self._names[f]:
            return str(self._names[f])
        return f"feature_{f}"

    def _off_axis_value(self, f: int) -> float | None:
        if self._pinned is not None and 0 <= f < self._pinned.size:
            v = float(self._pinned[f])
            if np.isfinite(v):
                return v
        return None

    def run(self, root: Any, split_of: Any, children_of: Any, leaf_of: Any) -> None:
        Item = tuple[Any, float, float, float, float, int, list[str], int]
        queue: deque[Item] = deque(
            [(root, self._rect[0], self._rect[1], self._rect[2], self._rect[3], 0, [], 0)])
        while queue:
            node, xmin, xmax, ymin, ymax, depth, conds, nid = queue.popleft()
            leaf_val = leaf_of(node)
            if leaf_val is not None:
                self._emit_leaf(nid, xmin, xmax, ymin, ymax, conds, leaf_val)
                continue
            parsed = split_of(node)
            if parsed is None:
                continue
            f, t = parsed
            if t is None or not np.isfinite(t):
                # Categorical / unparseable split: follow the pinned branch.
                nxt = children_of(node)
                if nxt is None:
                    continue
                go_left = True
                v = self._off_axis_value(f)
                if v is not None and isinstance(t, float):
                    go_left = bool(v <= t)
                kids = nxt if isinstance(nxt, (list, tuple)) else (nxt, nxt)
                queue.append((kids[0] if go_left else kids[-1], xmin, xmax, ymin, ymax,
                              depth + 1, conds, nid * 2 + (0 if go_left else 1)))
                continue
            t = round(float(t), 6)
            nxt = children_of(node)
            if nxt is None:
                continue
            kids = nxt if isinstance(nxt, (list, tuple)) else (nxt, nxt)
            # Convention: kids[0] = yes/left (<=), kids[-1] = no/right (>).
            if f == self._x_idx:
                if xmin - 1e-12 <= t <= xmax + 1e-12:
                    tc = min(max(t, xmin), xmax)
                    self._emit_segment("x", tc, nid, depth, [tc, ymin, tc, ymax])
                    queue.append((kids[-1], tc, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(tc)}"], nid * 2 + 1))
                    queue.append((kids[0], xmin, tc, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(tc)}"], nid * 2))
                elif t < xmin:
                    queue.append((kids[-1], xmin, xmax, ymin, ymax, depth + 1, conds, nid * 2 + 1))
                else:
                    queue.append((kids[0], xmin, xmax, ymin, ymax, depth + 1, conds, nid * 2))
            elif f == self._y_idx:
                if ymin - 1e-12 <= t <= ymax + 1e-12:
                    tc = min(max(t, ymin), ymax)
                    self._emit_segment("y", tc, nid, depth, [xmin, tc, xmax, tc])
                    queue.append((kids[-1], xmin, xmax, tc, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(tc)}"], nid * 2 + 1))
                    queue.append((kids[0], xmin, xmax, ymin, tc, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(tc)}"], nid * 2))
                elif t < ymin:
                    queue.append((kids[-1], xmin, xmax, ymin, ymax, depth + 1, conds, nid * 2 + 1))
                else:
                    queue.append((kids[0], xmin, xmax, ymin, ymax, depth + 1, conds, nid * 2))
            else:
                v = self._off_axis_value(f)
                if v is None:
                    queue.append((kids[0], xmin, xmax, ymin, ymax, depth + 1, conds, nid * 2))
                elif v <= t:
                    queue.append((kids[0], xmin, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(t)}"], nid * 2))
                else:
                    queue.append((kids[-1], xmin, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(t)}"], nid * 2 + 1))

    def _emit_segment(self, axis: str, t: float, node: int, depth: int,
                      seg_raw: list[float]) -> None:
        if len(self.segments) >= self._seg_cap:
            self.segments_truncated = True
            return
        seg_w = _world_seg([round(float(v), 6) for v in seg_raw], self._xs, self._ys)
        if seg_w is None:
            return
        self.segments.append(BoostCutSegment2D(
            axis=axis, threshold=round(float(t), 6), node_id=int(node),
            tree_index=int(self._ti), depth=int(depth),
            segment_raw=[round(float(v), 6) for v in seg_raw], segment_world=seg_w))

    def _emit_leaf(self, node: int, xmin: float, xmax: float, ymin: float, ymax: float,
                   conds: list[str], leaf_val: float) -> None:
        if len(self.leaves) >= self._leaf_cap:
            self.leaves_truncated = True
            return
        try:
            lv = round(float(leaf_val), 5)
        except (TypeError, ValueError):
            lv = 0.0
        box_raw = [round(xmin, 6), round(xmax, 6), round(ymin, 6), round(ymax, 6)]
        box_w = _world_seg([box_raw[0], box_raw[2], box_raw[1], box_raw[3]],
                           self._xs, self._ys)
        if box_w is None:
            return
        rule = " AND ".join(conds)
        if len(rule) > 400:
            rule = rule[:397] + "…"
        self.leaves.append(BoostLeafBox2D(
            leaf_id=int(node), tree_index=int(self._ti),
            box_raw=box_raw, box_world=[box_w[0], box_w[2], box_w[1], box_w[3]],
            leaf_value=float(lv) if np.isfinite(lv) else 0.0, rule=rule))


class Boosting2DExtractor:
    """Stateless 2D geometry builder for the five boosting families."""

    # -- public entry point -------------------------------------------------
    @staticmethod
    def extract_boosting_2d_geometry(
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
    ) -> Boosting2DGeometry | None:
        """Build the exact 2D geometry, or ``None`` for foreign / pipelined models."""
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
        names = list(feature_names or [])
        pinned = _pinned_vector(ref)
        xr = (math.floor(min(float(x_range_raw[0]), float(x_range_raw[1])) * 1e6) / 1e6,
              math.ceil(max(float(x_range_raw[0]), float(x_range_raw[1])) * 1e6) / 1e6)
        yr = (math.floor(min(float(y_range_raw[0]), float(y_range_raw[1])) * 1e6) / 1e6,
              math.ceil(max(float(y_range_raw[0]), float(y_range_raw[1])) * 1e6) / 1e6)
        row = (np.asarray(active_row, dtype=np.float64).ravel()
               if active_row is not None else None)

        if est_name in _SKLEARN_GBM_NAMES:
            return Boosting2DExtractor._sklearn_gbm(
                est, est_name, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, xr, yr, x_scale, y_scale, names, pinned)
        if est_name in _HIST_NAMES:
            return Boosting2DExtractor._hist_gbm(
                est, est_name, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, xr, yr, x_scale, y_scale, names, pinned, row)
        if est_name in _XGB_NAMES or Boosting2DExtractor._is_xgb_native(est):
            return Boosting2DExtractor._xgboost(
                est, est_name, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, xr, yr, x_scale, y_scale, names, pinned, row)
        if est_name in _LGBM_NAMES or Boosting2DExtractor._is_lgbm_native(est):
            return Boosting2DExtractor._lightgbm(
                est, est_name, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, xr, yr, x_scale, y_scale, names, pinned, row)
        if est_name in _CB_NAMES:
            return Boosting2DExtractor._catboost(
                est, est_name, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, xr, yr, x_scale, y_scale, names, pinned, row)
        return None

    # -- family detectors ------------------------------------------------------
    @staticmethod
    def _is_xgb_native(est: Any) -> bool:
        try:
            import xgboost as xgb  # noqa: PLC0415

            if isinstance(est, xgb.Booster):
                return True
            return callable(getattr(est, "get_booster", None))
        except ImportError:
            return False

    @staticmethod
    def _is_lgbm_native(est: Any) -> bool:
        try:
            import lightgbm as lgb  # noqa: PLC0415

            if isinstance(est, lgb.Booster):
                return True
            return isinstance(getattr(est, "booster_", None), lgb.Booster)
        except ImportError:
            return False

    # -- sklearn GBM ------------------------------------------------------------
    @staticmethod
    def _sklearn_gbm(est: Any, est_name: str, ref: np.ndarray,
                     x_idx: int, y_idx: int, xn: str, yn: str,
                     xr: tuple[float, float], yr: tuple[float, float],
                     x_scale: Any, y_scale: Any, names: list[str],
                     pinned: np.ndarray) -> Boosting2DGeometry | None:
        try:
            estimators = np.asarray(getattr(est, "estimators_"))
        except (TypeError, ValueError):
            return None
        if estimators.size == 0:
            return None
        flat: list[Any] = []
        # Binary/regression: (M, 1); multiclass: (M, C) — first column keeps
        # the overlay bounded while staying exact for binary/regression.
        try:
            n_rounds = int(estimators.shape[0])
            for m in range(min(n_rounds, MAX_GBM_TREES)):
                sub = estimators[m, 0] if estimators.ndim >= 2 else estimators[m]
                tree = getattr(sub, "tree_", None)
                if tree is not None:
                    flat.append(tree)
        except (IndexError, TypeError, ValueError):
            return None
        if not flat:
            return None
        segments: list[BoostCutSegment2D] = []
        truncated = False
        for ti, tree in enumerate(flat):
            walker = _SklearnTreeWalker(tree, x_idx, y_idx, xr, yr, pinned, names,
                                        x_scale, y_scale, tree_index=ti,
                                        max_segments=MAX_GBM_SEGMENTS_PER_TREE,
                                        collect_leaves=False)
            walker.run()
            segments.extend(walker.segments)
            truncated = truncated or walker.segments_truncated
            if len(segments) >= MAX_SEGMENTS:
                segments = segments[:MAX_SEGMENTS]
                truncated = True
                break
        return Boosting2DGeometry(
            family="sklearn_gbm", algorithm=est_name,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            n_estimators=int(n_rounds), trees_exported=int(len(flat)),
            segments=segments, segments_truncated=bool(truncated),
            no_axis_splits=len(segments) == 0,
            formula=f"top-{len(flat)}/{n_rounds} residual-tree cuts superimposed; dense field = additive log-odds",
        )

    # -- HistGradientBoosting ----------------------------------------------------
    @staticmethod
    def _hist_gbm(est: Any, est_name: str, ref: np.ndarray,
                  x_idx: int, y_idx: int, xn: str, yn: str,
                  xr: tuple[float, float], yr: tuple[float, float],
                  x_scale: Any, y_scale: Any, names: list[str],
                  pinned: np.ndarray, row: np.ndarray | None) -> Boosting2DGeometry | None:
        try:
            predictors = list(getattr(est, "_predictors", []) or [])
        except TypeError:
            return None
        if not predictors:
            return None
        n_iter = len(predictors)
        try:
            first_cols = list(predictors[0])
        except TypeError:
            return None
        n_cols = len(first_cols)
        if n_cols == 0:
            return None
        # Exact bin edges for the two visualized axes (full resolution).
        try:
            thresholds = list(getattr(getattr(est, "_bin_mapper", None), "bin_thresholds_", []) or [])
        except TypeError:
            thresholds = []
        bin_x = Boosting2DExtractor._clean_edges(thresholds[x_idx] if x_idx < len(thresholds) else [])
        bin_y = Boosting2DExtractor._clean_edges(thresholds[y_idx] if y_idx < len(thresholds) else [])
        # First-iteration cuts (column 0 = positive-class / regression tree).
        segments: list[BoostCutSegment2D] = []
        truncated = False
        for m in range(min(n_iter, MAX_HIST_ITERS)):
            try:
                cols = list(predictors[m])
            except TypeError:
                continue
            if not cols:
                continue
            pred = cols[0]
            nodes = getattr(pred, "nodes", None)
            if nodes is None:
                continue
            walker = _SklearnTreeWalker(
                _HistTreeShim(nodes), x_idx, y_idx, xr, yr, pinned, names,
                x_scale, y_scale, tree_index=m,
                max_segments=MAX_GBM_SEGMENTS_PER_TREE, collect_leaves=False)
            walker.run()
            segments.extend(walker.segments)
            truncated = truncated or walker.segments_truncated
            if len(segments) >= MAX_SEGMENTS:
                segments = segments[:MAX_SEGMENTS]
                truncated = True
                break
        active_bin: list[int] | None = None
        if row is not None and row.size > max(x_idx, y_idx):
            try:
                bx = int(np.searchsorted(np.asarray(bin_x, dtype=np.float64),
                                         float(row[x_idx]), side="left"))
                by = int(np.searchsorted(np.asarray(bin_y, dtype=np.float64),
                                         float(row[y_idx]), side="left"))
                if bin_x and bin_y:
                    active_bin = [bx, by]
            except (TypeError, ValueError, IndexError):
                active_bin = None
        return Boosting2DGeometry(
            family="hist_gbm", algorithm=est_name,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            n_estimators=int(n_iter), trees_exported=int(min(n_iter, MAX_HIST_ITERS)),
            segments=segments, segments_truncated=bool(truncated),
            bin_edges_x_raw=bin_x, bin_edges_y_raw=bin_y,
            bin_lines_x_world=_world_line_positions(bin_x, x_scale),
            bin_lines_y_world=_world_line_positions(bin_y, y_scale),
            active_bin=active_bin,
            no_axis_splits=len(segments) == 0 and not bin_x and not bin_y,
            formula=f"quantized {len(bin_x) + 1}x{len(bin_y) + 1} bin lattice; first-{min(n_iter, MAX_HIST_ITERS)} residual cuts",
        )

    @staticmethod
    def extract_frame_shap(
        *,
        model_obj: Any,
        X: np.ndarray,
        row_index: int,
        x_feature_idx: int,
        y_feature_idx: int,
    ) -> dict[str, Any] | None:
        """Row-dependent 2D state for ``POST /projection/boost-frame-2d``.

        Returns ``{"family", "algorithm", "row_index", "query_raw",
        "shap_vector" | None}`` without parsing any tree dump, so a scrub
        tick costs one native single-row call (< 5 ms). ``None`` when the
        model is foreign / pipelined. Static geometry (segments, leaves,
        bins, checkerboard) always comes from the contour payload; the
        active leaf/cell is resolved client-side point-in-box at 60 FPS.
        """
        if _is_pipeline(model_obj):
            return None
        est = _unwrap(model_obj)
        est_name = type(est).__name__
        ref = np.asarray(X, dtype=np.float64)
        if ref.ndim != 2 or not (0 <= int(row_index) < ref.shape[0]):
            return None
        d = int(ref.shape[1])
        if not (0 <= x_feature_idx < d and 0 <= y_feature_idx < d):
            return None
        row = ref[int(row_index)]
        shap: BoostShapVector2D | None = None
        if est_name in _XGB_NAMES or Boosting2DExtractor._is_xgb_native(est):
            try:
                import xgboost as xgb  # noqa: PLC0415
            except ImportError:
                return None
            booster = est if isinstance(est, xgb.Booster) else None
            if booster is None:
                get_booster = getattr(est, "get_booster", None)
                if callable(get_booster):
                    try:
                        booster = get_booster()
                    except (AttributeError, TypeError, ValueError):
                        booster = None
            if booster is None or not isinstance(booster, xgb.Booster):
                return None
            try:
                embedded = list(getattr(booster, "feature_names", None) or [])
            except TypeError:
                embedded = []
            shap = Boosting2DExtractor._xgb_shap(booster, embedded or None, row,
                                                 x_feature_idx, y_feature_idx, d)
            family, algorithm = "xgboost", est_name
        elif est_name in _LGBM_NAMES or Boosting2DExtractor._is_lgbm_native(est):
            try:
                import lightgbm as lgb  # noqa: PLC0415
            except ImportError:
                return None
            booster = est if isinstance(est, lgb.Booster) else getattr(est, "booster_", None)
            if booster is None or not isinstance(booster, lgb.Booster):
                return None
            shap = Boosting2DExtractor._lgbm_shap(booster, row, x_feature_idx, y_feature_idx, d)
            family, algorithm = "lightgbm", est_name
        elif est_name in _CB_NAMES:
            shap = Boosting2DExtractor._catboost_shap(est, ref, row,
                                                      x_feature_idx, y_feature_idx, d)
            family, algorithm = "catboost", est_name
        elif est_name in _SKLEARN_GBM_NAMES:
            family, algorithm = "sklearn_gbm", est_name
        elif est_name in _HIST_NAMES:
            family, algorithm = "hist_gbm", est_name
        else:
            return None
        try:
            qx = round(float(row[x_feature_idx]), 6)
            qy = round(float(row[y_feature_idx]), 6)
        except (IndexError, TypeError, ValueError):
            return None
        return {
            "family": family,
            "algorithm": algorithm,
            "row_index": int(row_index),
            "query_raw": [qx, qy],
            "shap_vector": shap.model_dump(mode="json") if shap is not None else None,
        }

    @staticmethod
    def _clean_edges(raw: Any) -> list[float]:
        try:
            arr = np.asarray(raw, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return []
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return []
        return [round(float(v), 5) for v in np.unique(arr).tolist()]

    # -- XGBoost ------------------------------------------------------------------
    @staticmethod
    def _xgboost(est: Any, est_name: str, ref: np.ndarray,
                 x_idx: int, y_idx: int, xn: str, yn: str,
                 xr: tuple[float, float], yr: tuple[float, float],
                 x_scale: Any, y_scale: Any, names: list[str],
                 pinned: np.ndarray, row: np.ndarray | None) -> Boosting2DGeometry | None:
        try:
            import xgboost as xgb  # noqa: PLC0415
        except ImportError:
            return None
        booster: Any | None = None
        if isinstance(est, xgb.Booster):
            booster = est
        else:
            get_booster = getattr(est, "get_booster", None)
            if callable(get_booster):
                try:
                    booster = get_booster()
                except (AttributeError, TypeError, ValueError):
                    booster = None
        if booster is None or not isinstance(booster, xgb.Booster):
            return None
        try:
            total = int(booster.num_boosted_rounds())
        except (AttributeError, TypeError, ValueError):
            total = 0
        try:
            booster_names = list(getattr(booster, "feature_names", None) or [])
        except TypeError:
            booster_names = []
        width = int(ref.shape[1])
        try:
            import json as _json  # noqa: PLC0415

            dumps = booster.get_dump(dump_format="json", with_stats=True)
        except (AttributeError, TypeError, ValueError):
            return None
        if not dumps:
            return None
        segments: list[BoostCutSegment2D] = []
        leaves: list[BoostLeafBox2D] = []
        truncated = leaves_truncated = False
        try:
            import json as _json2  # noqa: PLC0415

            tree0 = _json2.loads(dumps[0])
        except (TypeError, ValueError):
            return None

        def _split_of(node: Any) -> tuple[int, float | None] | None:
            if not isinstance(node, dict) or "leaf" in node:
                return None
            cond = node.get("split_condition")
            if isinstance(cond, list):
                return (_feature_index(node.get("split"), booster_names or None, width), None)
            try:
                t = round(float(cond), 5) if cond is not None else None
            except (TypeError, ValueError):
                t = None
            return (_feature_index(node.get("split"), booster_names or None, width), t)

        def _children_of(node: Any) -> tuple[Any, Any] | None:
            kids = node.get("children") if isinstance(node, dict) else None
            if not isinstance(kids, list) or len(kids) < 2:
                # yes/no ids without nested children (shallow dump): no geometry.
                return None
            # Convention: children[0] = yes (<=), children[1] = no (>).
            return (kids[0], kids[1])

        def _leaf_of(node: Any) -> float | None:
            if isinstance(node, dict) and "leaf" in node:
                try:
                    return round(float(node["leaf"]), 5)
                except (TypeError, ValueError):
                    return 0.0
            return None

        walker = _DumpTreeWalker(x_idx, y_idx, xr, yr, pinned, names, x_scale, y_scale)
        walker.run(tree0, _split_of, _children_of, _leaf_of)
        segments, leaves = walker.segments, walker.leaves
        truncated, leaves_truncated = walker.segments_truncated, walker.leaves_truncated
        shap = Boosting2DExtractor._xgb_shap(booster, booster_names or None, row, x_idx, y_idx, width)
        if row is not None and row.size > max(x_idx, y_idx) and leaves:
            qx = float(row[x_idx]) if np.isfinite(row[x_idx]) else None
            qy = float(row[y_idx]) if np.isfinite(row[y_idx]) else None
            if qx is not None and qy is not None:
                for lf in leaves:
                    xmin, xmax, ymin, ymax = lf.box_raw
                    if xmin <= qx <= xmax and ymin <= qy <= ymax:
                        lf.is_active = True
                        break
        return Boosting2DGeometry(
            family="xgboost", algorithm=est_name,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            n_estimators=int(total), trees_exported=1,
            segments=segments, segments_truncated=bool(truncated),
            leaves=leaves, leaves_truncated=bool(leaves_truncated),
            shap_vector=shap,
            no_axis_splits=len(segments) == 0,
            formula=f"tree-0 leaf polygons of {total} rounds; SHAP push (phi_x, phi_y) exact C++ TreeSHAP",
        )

    @staticmethod
    def _xgb_shap(booster: Any, booster_names: list[str] | None,
                  row: np.ndarray | None, x_idx: int, y_idx: int,
                  width: int) -> BoostShapVector2D | None:
        if row is None or row.size < width or width == 0:
            return None
        try:
            import xgboost as xgb  # noqa: PLC0415
            import warnings as _w  # noqa: PLC0415

            mat = np.ascontiguousarray(np.asarray(row, dtype=np.float32).reshape(1, -1))
            embedded = list(getattr(booster, "feature_names", None) or [])
            fn = embedded if embedded and len(embedded) == mat.shape[1] else None
            with _w.catch_warnings():
                _w.filterwarnings("ignore", category=UserWarning)
                dm = xgb.DMatrix(mat, feature_names=fn)
                raw = np.asarray(booster.predict(dm, pred_contribs=True, strict_shape=True))
            if raw.ndim == 2:
                phi = np.asarray(raw[0, 0, :width] if raw.ndim == 3 else raw[0, :width],
                                 dtype=np.float64)
                bias = float(raw[0, -1])
            elif raw.ndim == 3:
                phi = np.asarray(raw[0, 0, :width], dtype=np.float64)
                bias = float(raw[0, 0, -1])
            else:
                return None
            if not (0 <= x_idx < phi.size and 0 <= y_idx < phi.size):
                return None
            px, py = float(phi[x_idx]), float(phi[y_idx])
            if not (np.isfinite(px) and np.isfinite(py)):
                return None
            return BoostShapVector2D(phi_x=round(px, 5), phi_y=round(py, 5),
                                     magnitude=round(float(np.hypot(px, py)), 5),
                                     bias=round(bias, 5) if np.isfinite(bias) else None,
                                     method="xgb_pred_contribs")
        except Exception:
            return None

    # -- LightGBM ------------------------------------------------------------------
    @staticmethod
    def _lightgbm(est: Any, est_name: str, ref: np.ndarray,
                  x_idx: int, y_idx: int, xn: str, yn: str,
                  xr: tuple[float, float], yr: tuple[float, float],
                  x_scale: Any, y_scale: Any, names: list[str],
                  pinned: np.ndarray, row: np.ndarray | None) -> Boosting2DGeometry | None:
        try:
            import lightgbm as lgb  # noqa: PLC0415
        except ImportError:
            return None
        booster: Any | None = None
        if isinstance(est, lgb.Booster):
            booster = est
        else:
            cand = getattr(est, "booster_", None)
            if isinstance(cand, lgb.Booster):
                booster = cand
        if booster is None:
            return None
        try:
            total = int(booster.num_trees())
            per_iter = max(int(booster.num_model_per_iteration()), 1)
        except (AttributeError, TypeError, ValueError):
            return None
        n_iters = max(total // per_iter, 1) if total else 0
        width = int(ref.shape[1])
        try:
            dump = booster.dump_model(num_iteration=1)
        except (AttributeError, TypeError, ValueError):
            return None
        tree_info = (dump.get("tree_info") or []) if isinstance(dump, dict) else []
        if not tree_info or not isinstance(tree_info[0], dict):
            return None
        struct = tree_info[0].get("tree_structure")
        if struct is None:
            return None

        def _split_of(node: Any) -> tuple[int, float | None] | None:
            if not isinstance(node, dict) or "split_index" not in node:
                return None
            try:
                f = int(node.get("split_feature", -1))
            except (TypeError, ValueError):
                f = -1
            if f < 0 or f >= width:
                return (f, None)
            if str(node.get("decision_type", "<=")) == "==":
                return (f, None)  # categorical bitset: no single cut
            try:
                t = round(float(node.get("threshold")), 5)
            except (TypeError, ValueError):
                return (f, None)
            return (f, t)

        def _children_of(node: Any) -> tuple[Any, Any] | None:
            l, r = node.get("left_child"), node.get("right_child")
            if l is None or r is None:
                return None
            return (l, r)

        def _leaf_of(node: Any) -> float | None:
            if isinstance(node, dict) and "split_index" not in node:
                try:
                    return round(float(node.get("leaf_value", 0.0)), 5)
                except (TypeError, ValueError):
                    return 0.0
            return None

        walker = _DumpTreeWalker(x_idx, y_idx, xr, yr, pinned, names, x_scale, y_scale)
        walker.run(struct, _split_of, _children_of, _leaf_of)
        shap = Boosting2DExtractor._lgbm_shap(booster, row, x_idx, y_idx, width)
        if row is not None and row.size > max(x_idx, y_idx) and walker.leaves:
            qx = float(row[x_idx]) if np.isfinite(row[x_idx]) else None
            qy = float(row[y_idx]) if np.isfinite(row[y_idx]) else None
            if qx is not None and qy is not None:
                for lf in walker.leaves:
                    xmin, xmax, ymin, ymax = lf.box_raw
                    if xmin <= qx <= xmax and ymin <= qy <= ymax:
                        lf.is_active = True
                        break
        return Boosting2DGeometry(
            family="lightgbm", algorithm=est_name,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            n_estimators=int(n_iters), trees_exported=1,
            segments=walker.segments, segments_truncated=bool(walker.segments_truncated),
            leaves=walker.leaves, leaves_truncated=bool(walker.leaves_truncated),
            shap_vector=shap,
            no_axis_splits=len(walker.segments) == 0,
            formula=f"tree-0 asymmetric leaf-wise polygons of {n_iters} iterations; SHAP push exact C++",
        )

    @staticmethod
    def _lgbm_shap(booster: Any, row: np.ndarray | None,
                   x_idx: int, y_idx: int, width: int) -> BoostShapVector2D | None:
        if row is None or row.size < width or width == 0:
            return None
        try:
            mat = np.ascontiguousarray(np.asarray(row, dtype=np.float32).reshape(1, -1))
            raw = np.asarray(booster.predict(mat, pred_contrib=True, num_thread=2))
            if hasattr(raw, "toarray"):
                raw = np.asarray(raw.toarray())
            raw = np.asarray(raw, dtype=np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] == width + 1:
                phi = raw[0, :width]
                bias = float(raw[0, width])
            else:
                return None  # multiclass flattened: no honest 2D slice
            if not (0 <= x_idx < phi.size and 0 <= y_idx < phi.size):
                return None
            px, py = float(phi[x_idx]), float(phi[y_idx])
            if not (np.isfinite(px) and np.isfinite(py)):
                return None
            return BoostShapVector2D(phi_x=round(px, 5), phi_y=round(py, 5),
                                     magnitude=round(float(np.hypot(px, py)), 5),
                                     bias=round(bias, 5) if np.isfinite(bias) else None,
                                     method="lgbm_pred_contrib")
        except Exception:
            return None

    # -- CatBoost -------------------------------------------------------------------
    @staticmethod
    def _catboost(est: Any, est_name: str, ref: np.ndarray,
                  x_idx: int, y_idx: int, xn: str, yn: str,
                  xr: tuple[float, float], yr: tuple[float, float],
                  x_scale: Any, y_scale: Any, names: list[str],
                  pinned: np.ndarray, row: np.ndarray | None) -> Boosting2DGeometry | None:
        try:
            n_trees = int(getattr(est, "tree_count_", 0))
        except (AttributeError, TypeError, ValueError):
            n_trees = 0
        if n_trees < 1:
            return None
        width = int(ref.shape[1])
        try:
            from catboost import Pool  # noqa: PLC0415

            sample_n = min(int(ref.shape[0]), 200)
            pool = Pool(np.ascontiguousarray(ref[:sample_n, :], dtype=np.float32))
            texts = est._get_tree_splits(0, pool)
            try:
                leaf_values = [float(v) for v in list(est._get_tree_leaf_values(0))]
            except Exception:
                leaf_values = []
            del pool
        except Exception:
            return None
        if not isinstance(texts, (list, tuple)) or not texts:
            return None
        name_to_idx = {str(n): i for i, n in enumerate(names)}
        levels: list[dict[str, Any]] = []
        for t in texts:
            parsed = Boosting2DExtractor._parse_cb_split(str(t), name_to_idx, width)
            if parsed is not None:
                levels.append(parsed)
        if not levels:
            return None
        depth = len(levels)
        # Fixed bits for off-axis levels from the pinned background.
        fixed_bits: dict[int, int] = {}
        for d, lv in enumerate(levels):
            f = int(lv["feature_idx"])
            if f == x_idx or f == y_idx or lv["kind"] != "numeric":
                continue
            pv = float(pinned[f]) if 0 <= f < pinned.size else None
            thr = lv.get("threshold")
            if pv is not None and np.isfinite(pv) and thr is not None:
                fixed_bits[d] = 1 if pv > float(thr) else 0
            else:
                fixed_bits[d] = 0
        x_cuts = sorted({float(lv["threshold"]) for lv in levels
                         if int(lv["feature_idx"]) == x_idx and lv["kind"] == "numeric"})
        y_cuts = sorted({float(lv["threshold"]) for lv in levels
                         if int(lv["feature_idx"]) == y_idx and lv["kind"] == "numeric"})
        x_cuts = x_cuts[:MAX_CUTS_PER_AXIS]
        y_cuts = y_cuts[:MAX_CUTS_PER_AXIS]
        truncated = len([lv for lv in levels
                         if int(lv["feature_idx"]) == x_idx and lv["kind"] == "numeric"]) > len(x_cuts)
        truncated = truncated or len(
            [lv for lv in levels if int(lv["feature_idx"]) == y_idx and lv["kind"] == "numeric"]) > len(y_cuts)
        # Cartesian cells; each cell center replays the full oblivious bitmask.
        x_bounds = [xr[0], *[c for c in x_cuts if xr[0] < c < xr[1]], xr[1]]
        y_bounds = [yr[0], *[c for c in y_cuts if yr[0] < c < yr[1]], yr[1]]
        cells: list[ObliviousCell2D] = []
        qx = float(row[x_idx]) if row is not None and row.size > x_idx and np.isfinite(row[x_idx]) else None
        qy = float(row[y_idx]) if row is not None and row.size > y_idx and np.isfinite(row[y_idx]) else None
        cid = 0
        for ix in range(len(x_bounds) - 1):
            for iy in range(len(y_bounds) - 1):
                if len(cells) >= MAX_CELLS:
                    truncated = True
                    break
                xmin, xmax = x_bounds[ix], x_bounds[ix + 1]
                ymin, ymax = y_bounds[iy], y_bounds[iy + 1]
                cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                bits: list[int] = []
                for d, lv in enumerate(levels):
                    f = int(lv["feature_idx"])
                    if d in fixed_bits:
                        bits.append(fixed_bits[d])
                    elif f == x_idx and lv["kind"] == "numeric":
                        bits.append(1 if cx > float(lv["threshold"]) else 0)
                    elif f == y_idx and lv["kind"] == "numeric":
                        bits.append(1 if cy > float(lv["threshold"]) else 0)
                    else:
                        bits.append(0)
                leaf = int(sum(b * (2 ** d) for d, b in enumerate(bits)))
                bitmask = "".join(str(b) for b in reversed(bits)) if bits else "0"
                box_raw = [round(xmin, 6), round(xmax, 6), round(ymin, 6), round(ymax, 6)]
                box_w = _world_seg([box_raw[0], box_raw[2], box_raw[1], box_raw[3]], x_scale, y_scale)
                if box_w is None:
                    continue
                is_active = (qx is not None and qy is not None
                             and xmin <= qx <= xmax and ymin <= qy <= ymax)
                cells.append(ObliviousCell2D(
                    cell_id=cid, leaf_index=leaf, bitmask=bitmask,
                    box_raw=box_raw, box_world=[box_w[0], box_w[2], box_w[1], box_w[3]],
                    is_active=bool(is_active)))
                cid += 1
        rule_parts: list[str] = []
        for lv in levels:
            fn = names[int(lv["feature_idx"])] if 0 <= int(lv["feature_idx"]) < len(names) else f"feature_{lv['feature_idx']}"
            if lv["kind"] == "numeric":
                rule_parts.append(f"{fn} > {_fmt_threshold(float(lv['threshold']))}")
            else:
                rule_parts.append(f"{fn} = {lv.get('category')}")
        shap = Boosting2DExtractor._catboost_shap(est, ref, row, x_idx, y_idx, width)
        return Boosting2DGeometry(
            family="catboost", algorithm=est_name,
            x_feature=str(xn), y_feature=str(yn),
            x_feature_idx=int(x_idx), y_feature_idx=int(y_idx),
            n_estimators=int(n_trees), trees_exported=1,
            oblivious_x_cuts_raw=[round(float(c), 5) for c in x_cuts],
            oblivious_y_cuts_raw=[round(float(c), 5) for c in y_cuts],
            oblivious_x_lines_world=_world_line_positions(x_cuts, x_scale),
            oblivious_y_lines_world=_world_line_positions(y_cuts, y_scale),
            oblivious_cells=cells, oblivious_truncated=bool(truncated),
            oblivious_rule=" AND ".join(rule_parts)[:400],
            shap_vector=shap,
            no_axis_splits=not x_cuts and not y_cuts,
            formula=f"tree-0 symmetric oblivious {len(x_cuts) + 1}x{len(y_cuts) + 1} checkerboard (depth {depth}); leaf = Σ I·2^d",
        )

    @staticmethod
    def _parse_cb_split(text: str, name_to_idx: dict[str, int], width: int) -> dict[str, Any] | None:
        s = text.strip()
        if "," not in s:
            return None
        feat, _, rest = s.partition(",")
        feat = feat.strip()
        rest = rest.strip()
        f_idx = name_to_idx.get(feat, -1)
        if f_idx < 0:
            try:
                f_idx = int(feat)
            except (TypeError, ValueError):
                return None
            if not (0 <= f_idx < width):
                return None
        if rest.startswith("bin="):
            try:
                thr = float(rest[4:])
            except (TypeError, ValueError):
                return None
            if not np.isfinite(thr):
                return None
            return {"feature_idx": int(f_idx), "kind": "numeric",
                    "threshold": round(thr, 5), "category": None}
        if rest.startswith("value="):
            return {"feature_idx": int(f_idx), "kind": "categorical",
                    "threshold": None, "category": rest[6:]}
        return None

    @staticmethod
    def _catboost_shap(est: Any, ref: np.ndarray, row: np.ndarray | None,
                       x_idx: int, y_idx: int, width: int) -> BoostShapVector2D | None:
        if row is None or row.size < width or width == 0:
            return None
        try:
            from catboost import Pool  # noqa: PLC0415

            try:
                cats = sorted({int(v) for v in list(est.get_cat_feature_indices() or [])})
            except (AttributeError, TypeError, ValueError):
                cats = []
            if any(0 <= c < width for c in cats):
                return None  # encoded categoricals: no honest numeric push vector
            mat = np.ascontiguousarray(np.asarray(row, dtype=np.float32).reshape(1, -1))
            pool = Pool(mat)
            raw = np.asarray(est.get_feature_importance(data=pool, type="ShapValues"),
                             dtype=np.float64)
            del pool
            if raw.ndim == 3:
                return None  # multiclass: no honest 2D slice
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < width + 1:
                return None
            phi = raw[0, :width]
            bias = float(raw[0, width])
            if not (0 <= x_idx < phi.size and 0 <= y_idx < phi.size):
                return None
            px, py = float(phi[x_idx]), float(phi[y_idx])
            if not (np.isfinite(px) and np.isfinite(py)):
                return None
            return BoostShapVector2D(phi_x=round(px, 5), phi_y=round(py, 5),
                                     magnitude=round(float(np.hypot(px, py)), 5),
                                     bias=round(bias, 5) if np.isfinite(bias) else None,
                                     method="catboost_shap")
        except Exception:
            return None


class _HistTreeShim:
    """Adapt a HistGradientBoosting ``TreePredictor.nodes`` array to the sklearn walker.

    Exposes ``feature`` / ``threshold`` (raw ``num_threshold``) /
    ``children_left`` / ``children_right`` / ``value`` so the shared
    axis-aligned walk works unchanged. Categorical nodes surface
    ``threshold = NaN`` → pinned-branch fallback (never a fabricated cut).
    """

    def __init__(self, nodes: Any) -> None:
        self._n = nodes
        n = int(np.asarray(nodes["is_leaf"]).size)
        feat = np.full((n,), TREE_UNDEFINED, dtype=np.int64)
        thr = np.full((n,), np.nan, dtype=np.float64)
        left = np.full((n,), -1, dtype=np.int64)
        right = np.full((n,), -1, dtype=np.int64)
        val = np.zeros((n, 1, 1), dtype=np.float64)
        try:
            is_leaf = np.asarray(nodes["is_leaf"]).astype(bool).ravel()
            f_all = np.asarray(nodes["feature_idx"]).ravel()
            l_all = np.asarray(nodes["left"]).ravel()
            r_all = np.asarray(nodes["right"]).ravel()
            v_all = np.asarray(nodes["value"], dtype=np.float64).ravel()
            try:
                is_cat = np.asarray(nodes["is_categorical"]).astype(bool).ravel()
            except (KeyError, TypeError, ValueError):
                is_cat = np.zeros((n,), dtype=bool)
            try:
                num_thr = np.asarray(nodes["num_threshold"], dtype=np.float64).ravel()
            except (KeyError, TypeError, ValueError):
                num_thr = np.full((n,), np.nan)
        except (KeyError, TypeError, ValueError):
            is_leaf = np.ones((n,), dtype=bool)
            f_all = np.zeros((n,), dtype=np.int64)
            l_all = r_all = np.zeros((n,), dtype=np.int64)
            v_all = np.zeros((n,))
            is_cat = np.zeros((n,), dtype=bool)
            num_thr = np.full((n,), np.nan)
        for i in range(n):
            if i < is_leaf.size and bool(is_leaf[i]):
                continue
            try:
                f = int(f_all[i])
            except (IndexError, TypeError, ValueError):
                continue
            feat[i] = f
            try:
                left[i] = int(l_all[i])
                right[i] = int(r_all[i])
            except (IndexError, TypeError, ValueError):
                pass
            if i < is_cat.size and bool(is_cat[i]):
                thr[i] = np.nan
            else:
                try:
                    t = float(num_thr[i])
                    thr[i] = t if np.isfinite(t) else np.nan
                except (IndexError, TypeError, ValueError):
                    thr[i] = np.nan
            try:
                vv = float(v_all[i])
                val[i, 0, 0] = vv if np.isfinite(vv) else 0.0
            except (IndexError, TypeError, ValueError):
                pass
        self.feature = feat
        self.threshold = thr
        self.children_left = left
        self.children_right = right
        self.value = val
