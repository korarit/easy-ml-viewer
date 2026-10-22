"""Phase 45 — 2D decision geometry for Decision Trees & Bagging Ensembles.

Complements the universal Marching-Squares contour engine (Phase 41) with
the *exact* axis-aligned overlays whose structure the dense grid cannot
provide:

- **Decision Tree / ExtraTree**: orthogonal cut segments clipped to each
  node's live bounding box (DFS pre-order, so shallow principal cuts come
  first), every leaf's bounding box with its rule string, prediction,
  sample count and impurity, plus the exact active-leaf walk for a query
  row (``<=`` goes left, per the scikit-learn contract).
- **Random Forest / ExtraTrees**: superimposed cut segments of the first
  ``MAX_FOREST_TREES`` (10) trees — the vote-fraction heatmap itself is
  already the dense contour (``predict_proba`` == mean tree vote), so the
  overlay adds the mosaic structure, not a second field.
- **AdaBoost**: per-stage weak-learner cuts ``(feature, threshold, alpha)``
  driving the staged-cascade slider; ``T = 0`` honestly shows only the
  base value (no learners).

Conventions (shared with Phase 42–44 extractors):

- Coordinates are **raw feature units**; world mapping uses
  ``WorldCoordinateScaler`` (``to_world_coords``). World floats are
  rounded to 2 decimals — plenty for a ±5 stage and keeps the DTO small.
- Splits on off-axis features cannot draw on the (X, Y) slice: the branch
  matching the pinned background (median row of ``X``) survives, the
  other is pruned. Pipeline-wrapped estimators live in *transformed*
  space, so the extractor returns ``None`` and the caller falls back to
  the sampled contour.
- Bounded output (memory budget §5.1 / ≤150 KB contour SLA): segments are
  batched client-side into 2 SVG paths (vertical / horizontal), capped at
  ``MAX_SEGMENTS`` with an explicit truncation flag; leaves are capped at
  ``MAX_LEAVES``. An ``active_row`` (full feature vector) may be passed
  so its exact leaf box is always included (``is_active=True``) even when
  the leaf list truncates.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

TREE_UNDEFINED = -2
MAX_SEGMENTS = 600
MAX_LEAVES = 200
MAX_FOREST_TREES = 10
MAX_FOREST_SEGMENTS_PER_TREE = 200
MAX_ADABOOST_STAGES = 50

_DT_NAMES = {
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
}
_RF_NAMES = {"RandomForestClassifier", "RandomForestRegressor"}
_ET_NAMES = {"ExtraTreesClassifier", "ExtraTreesRegressor"}
_AB_NAMES = {"AdaBoostClassifier", "AdaBoostRegressor"}


class TreeCutSegment2D(BaseModel):
    """One axis-aligned cut, clipped to its node's live box."""

    axis: str = Field(description="'x' (vertical, x = t) | 'y' (horizontal, y = t)")
    threshold: float = 0.0
    node_id: int = 0
    tree_index: int = 0
    depth: int = 0
    segment_raw: list[float] = Field(description="[x1, y1, x2, y2] in raw feature units")
    segment_world: list[float] = Field(description="[x1, y1, x2, y2] in world coords [-5, +5]")


class LeafBox2D(BaseModel):
    """One leaf's bounding box on the (X, Y) slice."""

    leaf_id: int = 0
    tree_index: int = 0
    box_raw: list[float] = Field(description="[xmin, xmax, ymin, ymax] in raw units")
    box_world: list[float] = Field(description="[xmin, xmax, ymin, ymax] in world coords")
    prediction: Any = None
    samples: int = 0
    impurity: float = 0.0
    rule: str = ""
    is_active: bool = False


class AdaBoostStage2D(BaseModel):
    """One weak learner for the staged-cascade slider."""

    stage: int = Field(description="1-based stage index")
    feature_index: int = -1
    axis: str = Field(description="'x' | 'y' | 'off' (split not on this slice)")
    threshold: float | None = None
    weight: float = 0.0
    tree_depth: int = 0
    segment_world: list[float] | None = None


class Trees2DGeometry(BaseModel):
    """Exact 2D planar geometry for one tree/ensemble model on one axis pair."""

    family: str = Field(description="decision_tree | random_forest | extra_trees | adaboost")
    algorithm: str = ""
    x_feature: str = ""
    y_feature: str = ""
    x_feature_idx: int = 0
    y_feature_idx: int = 0
    n_estimators: int = 1
    trees_exported: int = 1
    segments: list[TreeCutSegment2D] = Field(default_factory=list)
    segments_truncated: bool = False
    leaves: list[LeafBox2D] = Field(default_factory=list)
    leaves_truncated: bool = False
    stages: list[AdaBoostStage2D] = Field(default_factory=list)
    stages_truncated: bool = False
    base_value: float | None = None
    no_axis_splits: bool = Field(
        default=False,
        description="True when the tree never splits on X/Y (E45.5 badge)",
    )
    multi_output: bool = False
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


def _classes_of(est: Any) -> list[Any]:
    raw = getattr(est, "classes_", None)
    if raw is None:
        return []
    try:
        return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]
    except (TypeError, ValueError):
        return []


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


def _fmt_threshold(t: float) -> str:
    a = abs(t)
    if a != 0.0 and (a >= 10000 or a < 0.001):
        return f"{t:.2e}"
    return f"{round(float(t), 4)}"


class _TreeWalker:
    """Pruned DFS over one fitted ``tree_`` on the (X, Y) slice."""

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
        classes: list[Any] | None = None,
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
        self.segments: list[TreeCutSegment2D] = []
        self.leaves: list[LeafBox2D] = []
        self.segments_truncated = False
        self.leaves_truncated = False
        # Classes live on the estimator, never on the bare ``tree_`` struct.
        self._classes = list(classes or [])
        self._is_classifier = len(self._classes) > 0

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
        # Breadth-first queue — shallow principal cuts are emitted before
        # deep detail, so truncation (E45.1) drops the deepest levels first
        # and emitted depths are globally non-decreasing.
        Item = tuple[int, float, float, float, float, int, list[str]]
        queue: deque[Item] = deque(
            [(0, self._rect[0], self._rect[1], self._rect[2], self._rect[3], 0, [])]
        )
        push = queue.append
        while queue:
            node, xmin, xmax, ymin, ymax, depth, conds = queue.popleft()
            if not (0 <= node < feat.size):
                continue
            f = int(feat[node])
            if f == TREE_UNDEFINED:
                self._emit_leaf(node, xmin, xmax, ymin, ymax, conds)
                continue
            # Round-then-clip: DTO segments/boxes never poke outside the rect
            # by more than float error (the exact walk API stays unrounded).
            t = round(float(thr[node]), 6)
            if not np.isfinite(t):
                continue
            if f == self._x_idx:
                if xmin - 1e-12 <= t <= xmax + 1e-12:
                    # Clip, round, clip again: the stored cut never pokes
                    # outside the box by more than float error.
                    tc = min(max(min(max(t, xmin), xmax), xmin), xmax)
                    tc = min(max(round(tc, 6), xmin), xmax)
                    self._emit_segment("x", tc, node, depth, [tc, ymin, tc, ymax])
                    push((int(right[node]), tc, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(tc)}"]))
                    push((int(left[node]), xmin, tc, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(tc)}"]))
                elif t < xmin:
                    push((int(right[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
                else:
                    push((int(left[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
            elif f == self._y_idx:
                if ymin - 1e-12 <= t <= ymax + 1e-12:
                    tc = min(max(min(max(t, ymin), ymax), ymin), ymax)
                    tc = min(max(round(tc, 6), ymin), ymax)
                    self._emit_segment("y", tc, node, depth, [xmin, tc, xmax, tc])
                    push((int(right[node]), xmin, xmax, tc, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(tc)}"]))
                    push((int(left[node]), xmin, xmax, ymin, tc, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(tc)}"]))
                elif t < ymin:
                    push((int(right[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
                else:
                    push((int(left[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
            else:
                v = self._off_axis_value(f)
                if v is None:
                    # No background value — keep the left branch so the slice
                    # stays a valid partition (flagged upstream as ambiguous
                    # only in the point-walk API).
                    push((int(left[node]), xmin, xmax, ymin, ymax, depth + 1, conds))
                elif v <= t:
                    push((int(left[node]), xmin, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} <= {_fmt_threshold(t)}"]))
                else:
                    push((int(right[node]), xmin, xmax, ymin, ymax, depth + 1,
                                  conds + [f"{self._name(f)} > {_fmt_threshold(t)}"]))

    def _emit_segment(self, axis: str, t: float, node: int, depth: int,
                      seg_raw: list[float]) -> None:
        if len(self.segments) >= self._seg_cap:
            self.segments_truncated = True
            return
        seg_w = _world_seg([round(float(v), 6) for v in seg_raw], self._xs, self._ys)
        if seg_w is None:
            return
        self.segments.append(TreeCutSegment2D(
            axis=axis,
            threshold=round(float(t), 6),
            node_id=int(node),
            tree_index=int(self._ti),
            depth=int(depth),
            segment_raw=[round(float(v), 6) for v in seg_raw],
            segment_world=seg_w,
        ))

    def _emit_leaf(self, node: int, xmin: float, xmax: float, ymin: float, ymax: float,
                   conds: list[str]) -> None:
        if not self._collect_leaves:
            return
        if len(self.leaves) >= self._leaf_cap:
            self.leaves_truncated = True
            return
        try:
            val = np.asarray(self._t.value[node], dtype=np.float64)
        except (IndexError, TypeError, ValueError):
            val = np.zeros((1, 1))
        # Single-output row 0 (multi-output trees: output 0, flagged upstream).
        flat = val.ravel()
        width = max(int(val.shape[-1]), 1) if val.ndim else 1
        vec = flat[:width] if flat.size else np.zeros((1,))
        if self._is_classifier and vec.size:
            k = int(np.argmax(vec))
            pred: Any = self._classes[k] if 0 <= k < len(self._classes) else k
            if hasattr(pred, "item"):
                try:
                    pred = pred.item()
                except (ValueError, TypeError):
                    pass
        elif vec.size:
            pred = round(float(vec[0]), 6)
        else:
            pred = 0.0
        try:
            samples = int(np.asarray(self._t.n_node_samples, dtype=np.int64).ravel()[node])
        except (IndexError, TypeError, ValueError):
            samples = 0
        try:
            imp = round(float(np.asarray(self._t.impurity, dtype=np.float64).ravel()[node]), 6)
        except (IndexError, TypeError, ValueError):
            imp = 0.0
        box_raw = [round(xmin, 6), round(xmax, 6), round(ymin, 6), round(ymax, 6)]
        box_w = _world_seg([box_raw[0], box_raw[2], box_raw[1], box_raw[3]],
                           self._xs, self._ys)
        if box_w is None:
            return
        rule = " AND ".join(conds)
        if len(rule) > 400:
            rule = rule[:397] + "…"
        self.leaves.append(LeafBox2D(
            leaf_id=int(node),
            tree_index=int(self._ti),
            box_raw=box_raw,
            box_world=[box_w[0], box_w[2], box_w[1], box_w[3]],
            prediction=pred,
            samples=samples,
            impurity=imp,
            rule=rule,
        ))


class Tree2DExtractor:
    """Stateless 2D partition / leaf-box builder (see module docstring)."""

    # -- Dev-plan prototype API -----------------------------------------
    @staticmethod
    def extract_tree_partitions(
        tree_model: Any,
        x_idx: int,
        y_idx: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> dict[str, Any]:
        """Collect on-axis split rules + node/leaf counts (dev plan §5.2)."""
        est = _unwrap(tree_model)
        tree = getattr(est, "tree_", None)
        if tree is None:
            return {"splits": [], "node_count": 0, "leaf_count": 0}
        feat = np.asarray(tree.feature, dtype=np.int64).ravel()
        thr = np.asarray(tree.threshold, dtype=np.float64).ravel()
        splits: list[dict[str, Any]] = []
        for node in range(int(getattr(tree, "node_count", feat.size))):
            if node >= feat.size:
                break
            f = int(feat[node])
            if f == TREE_UNDEFINED:
                continue
            t = float(thr[node])
            if not np.isfinite(t):
                continue
            if f == x_idx:
                splits.append({"feature_index": int(f), "threshold": t,
                               "axis": "x", "node_id": int(node)})
            elif f == y_idx:
                splits.append({"feature_index": int(f), "threshold": t,
                               "axis": "y", "node_id": int(node)})
        return {
            "splits": splits,
            "node_count": int(getattr(tree, "node_count", 0)),
            "leaf_count": int(getattr(tree, "n_leaves", 0)),
        }

    # -- Test-plan API ----------------------------------------------------
    @staticmethod
    def compute_active_leaf_box(
        tree: Any,
        query_point: np.ndarray,
        x_idx: int = 0,
        y_idx: int = 1,
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (0.0, 1.0),
        extra_values: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """Walk root→leaf for a query point (test plan §2.2).

        On-axis splits use ``<=`` → left (scikit-learn contract, E45.4);
        off-axis splits use ``extra_values`` when given, else the left
        branch with ``off_axis_ambiguous=True``.
        """
        est = _unwrap(tree)
        tree_ = getattr(est, "tree_", None)
        if tree_ is None:
            raise ValueError("not a fitted decision tree (missing tree_)")
        q = np.asarray(query_point, dtype=np.float64).ravel()
        qx = float(q[x_idx]) if x_idx < q.size else 0.0
        qy = float(q[y_idx]) if y_idx < q.size else 0.0
        feat = np.asarray(tree_.feature, dtype=np.int64).ravel()
        thr = np.asarray(tree_.threshold, dtype=np.float64).ravel()
        left = np.asarray(tree_.children_left, dtype=np.int64).ravel()
        right = np.asarray(tree_.children_right, dtype=np.int64).ravel()
        xmin, xmax = float(x_range[0]), float(x_range[1])
        ymin, ymax = float(y_range[0]), float(y_range[1])
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        node = 0
        ambiguous = False
        guards = 0
        while 0 <= node < feat.size and guards < 100000:
            guards += 1
            f = int(feat[node])
            if f == TREE_UNDEFINED:
                break
            t = float(thr[node])
            if f == x_idx:
                if qx <= t:
                    xmax = min(xmax, t)
                    node = int(left[node])
                else:
                    xmin = max(xmin, t)
                    node = int(right[node])
            elif f == y_idx:
                if qy <= t:
                    ymax = min(ymax, t)
                    node = int(left[node])
                else:
                    ymin = max(ymin, t)
                    node = int(right[node])
            else:
                v = (extra_values or {}).get(f)
                if v is None or not np.isfinite(v):
                    ambiguous = True
                    node = int(left[node])
                elif float(v) <= t:
                    node = int(left[node])
                else:
                    node = int(right[node])
        return {
            "x_min": float(xmin),
            "x_max": float(xmax),
            "y_min": float(ymin),
            "y_max": float(ymax),
            "leaf_id": int(node),
            "off_axis_ambiguous": bool(ambiguous),
        }

    # -- Full contour-DTO entry point --------------------------------------
    @staticmethod
    def extract_trees_2d_geometry(
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
    ) -> Trees2DGeometry | None:
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
        # Outward-rounded rect: 6dp-rounded cuts/boxes always stay inside it.
        xr = (math.floor(min(float(x_range_raw[0]), float(x_range_raw[1])) * 1e6) / 1e6,
              math.ceil(max(float(x_range_raw[0]), float(x_range_raw[1])) * 1e6) / 1e6)
        yr = (math.floor(min(float(y_range_raw[0]), float(y_range_raw[1])) * 1e6) / 1e6,
              math.ceil(max(float(y_range_raw[0]), float(y_range_raw[1])) * 1e6) / 1e6)

        if est_name in _DT_NAMES:
            tree = getattr(est, "tree_", None)
            if tree is None:
                return None
            walker = _TreeWalker(tree, x_feature_idx, y_feature_idx, xr, yr,
                                 pinned, names, x_scale, y_scale,
                                 classes=_classes_of(est))
            walker.run()
            leaves = walker.leaves
            if active_row is not None:
                leaves = Tree2DExtractor._ensure_active_leaf(
                    est, tree, np.asarray(active_row, dtype=np.float64).ravel(),
                    x_feature_idx, y_feature_idx, xr, yr, pinned, names,
                    x_scale, y_scale, leaves)
            no_axis = len(walker.segments) == 0
            try:
                multi = bool(np.asarray(tree.value).shape[1] > 1)
            except (IndexError, TypeError, ValueError):
                multi = False
            return Trees2DGeometry(
                family="decision_tree",
                algorithm=est_name,
                x_feature=str(x_feature_name),
                y_feature=str(y_feature_name),
                x_feature_idx=int(x_feature_idx),
                y_feature_idx=int(y_feature_idx),
                n_estimators=1,
                trees_exported=1,
                segments=walker.segments,
                segments_truncated=walker.segments_truncated,
                leaves=leaves,
                leaves_truncated=walker.leaves_truncated,
                no_axis_splits=no_axis,
                multi_output=multi,
                formula=f"axis-aligned partitions on ({x_feature_name}, {y_feature_name}); leaf boxes exact",
            )

        if est_name in _RF_NAMES or est_name in _ET_NAMES:
            estimators = list(getattr(est, "estimators_", None) or [])
            if not estimators:
                return None
            family = "random_forest" if est_name in _RF_NAMES else "extra_trees"
            segments: list[TreeCutSegment2D] = []
            truncated = False
            exported = 0
            for ti, sub in enumerate(estimators[:MAX_FOREST_TREES]):
                sub_tree = getattr(sub, "tree_", None)
                if sub_tree is None:
                    continue
                walker = _TreeWalker(sub_tree, x_feature_idx, y_feature_idx, xr, yr,
                                     pinned, names, x_scale, y_scale,
                                     tree_index=ti,
                                     max_segments=MAX_FOREST_SEGMENTS_PER_TREE,
                                     collect_leaves=False,
                                     classes=_classes_of(est))
                walker.run()
                segments.extend(walker.segments)
                truncated = truncated or walker.segments_truncated
                exported += 1
                if len(segments) >= MAX_SEGMENTS:
                    segments = segments[:MAX_SEGMENTS]
                    truncated = True
                    break
            return Trees2DGeometry(
                family=family,
                algorithm=est_name,
                x_feature=str(x_feature_name),
                y_feature=str(y_feature_name),
                x_feature_idx=int(x_feature_idx),
                y_feature_idx=int(y_feature_idx),
                n_estimators=int(getattr(est, "n_estimators", exported)),
                trees_exported=int(exported),
                segments=segments,
                segments_truncated=bool(truncated),
                no_axis_splits=len(segments) == 0,
                formula=f"top-{exported} tree cuts superimposed; heatmap = vote fraction over {getattr(est, 'n_estimators', exported)} trees",
            )

        if est_name in _AB_NAMES:
            return Tree2DExtractor._adaboost(
                est, est_name, ref, x_feature_idx, y_feature_idx,
                x_feature_name, y_feature_name, xr, yr, x_scale, y_scale, names)
        return None

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _ensure_active_leaf(est: Any, tree: Any, row: np.ndarray,
                            x_idx: int, y_idx: int,
                            xr: tuple[float, float], yr: tuple[float, float],
                            pinned: np.ndarray, names: list[str],
                            x_scale: Any, y_scale: Any,
                            leaves: list[LeafBox2D]) -> list[LeafBox2D]:
        """Append the row's exact leaf (full vector — no pinning needed)."""
        extra = {j: float(row[j]) for j in range(min(row.size, pinned.size))
                 if j != x_idx and j != y_idx and np.isfinite(row[j])}
        try:
            box = Tree2DExtractor.compute_active_leaf_box(
                est, row, x_idx, y_idx, xr, yr, extra_values=extra)
        except ValueError:
            return leaves
        if any(l.leaf_id == box["leaf_id"] and l.tree_index == 0 for l in leaves):
            for l in leaves:
                if l.leaf_id == box["leaf_id"] and l.tree_index == 0:
                    l.is_active = True
            return leaves
        box_w = _world_seg([box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
                           x_scale, y_scale)
        if box_w is None:
            return leaves
        try:
            pred_row = np.asarray(est.predict(row.reshape(1, -1))).ravel()
            raw_pred = pred_row[0] if pred_row.size else 0
            pred: Any = raw_pred.item() if hasattr(raw_pred, "item") else raw_pred
        except (ValueError, TypeError, AttributeError):
            pred = None
        leaves = list(leaves) + [LeafBox2D(
            leaf_id=int(box["leaf_id"]),
            tree_index=0,
            box_raw=[round(box["x_min"], 6), round(box["x_max"], 6),
                     round(box["y_min"], 6), round(box["y_max"], 6)],
            box_world=[box_w[0], box_w[2], box_w[1], box_w[3]],
            prediction=pred,
            samples=0,
            impurity=0.0,
            rule="",
            is_active=True,
        )]
        return leaves

    @staticmethod
    def _adaboost(est: Any, est_name: str, ref: np.ndarray,
                  x_idx: int, y_idx: int, xn: str, yn: str,
                  xr: tuple[float, float], yr: tuple[float, float],
                  x_scale: Any, y_scale: Any,
                  names: list[str]) -> Trees2DGeometry | None:
        estimators = list(getattr(est, "estimators_", []) or [])
        try:
            weights = np.asarray(getattr(est, "estimator_weights_"), dtype=np.float64).ravel()
        except (TypeError, ValueError):
            weights = np.zeros((len(estimators),))
        n_classes = len(_classes_of(est)) or 2
        is_classifier = "Classifier" in str(est_name)
        if is_classifier:
            base: float | None = 0.5 if n_classes == 2 else (1.0 / max(n_classes, 1))
        else:
            base = None
        stages: list[AdaBoostStage2D] = []
        truncated = False
        for i, sub in enumerate(estimators):
            if i >= MAX_ADABOOST_STAGES:
                truncated = True
                break
            w = round(float(weights[i]) if i < weights.size and np.isfinite(weights[i]) else 0.0, 6)
            sub_tree = getattr(sub, "tree_", None)
            if sub_tree is None:
                stages.append(AdaBoostStage2D(stage=i + 1, feature_index=-1, axis="off",
                                              threshold=None, weight=w, tree_depth=0))
                continue
            try:
                feat = np.asarray(sub_tree.feature, dtype=np.int64).ravel()
                thr = np.asarray(sub_tree.threshold, dtype=np.float64).ravel()
                f0 = int(feat[0]) if feat.size else TREE_UNDEFINED
                t0 = float(thr[0]) if thr.size else float("nan")
                try:
                    depth = int(getattr(sub_tree, "max_depth", 1))
                except (TypeError, ValueError):
                    depth = 1
            except (TypeError, ValueError, IndexError):
                stages.append(AdaBoostStage2D(stage=i + 1, feature_index=-1, axis="off",
                                              threshold=None, weight=w, tree_depth=0))
                continue
            if f0 == TREE_UNDEFINED or not np.isfinite(t0):
                stages.append(AdaBoostStage2D(stage=i + 1, feature_index=f0, axis="off",
                                              threshold=None, weight=w, tree_depth=depth))
                continue
            if f0 == x_idx:
                axis, seg = "x", [t0, yr[0], t0, yr[1]]
            elif f0 == y_idx:
                axis, seg = "y", [xr[0], t0, xr[1], t0]
            else:
                stages.append(AdaBoostStage2D(stage=i + 1, feature_index=f0, axis="off",
                                              threshold=round(t0, 6), weight=w,
                                              tree_depth=depth))
                continue
            seg_w = _world_seg([round(float(v), 6) for v in seg], x_scale, y_scale)
            stages.append(AdaBoostStage2D(stage=i + 1, feature_index=f0, axis=axis,
                                          threshold=round(t0, 6), weight=w,
                                          tree_depth=depth, segment_world=seg_w))
        return Trees2DGeometry(
            family="adaboost",
            algorithm=est_name,
            x_feature=str(xn),
            y_feature=str(yn),
            x_feature_idx=int(x_idx),
            y_feature_idx=int(y_idx),
            n_estimators=int(len(estimators)),
            trees_exported=int(len(stages)),
            stages=stages,
            stages_truncated=bool(truncated or len(estimators) > len(stages)),
            base_value=base,
            no_axis_splits=not any(s.axis in ("x", "y") for s in stages),
            formula=f"staged sum of {len(estimators)} weak learners (alpha-weighted); T=0 shows base only",
        )
