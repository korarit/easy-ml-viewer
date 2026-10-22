# backend/app/domains/model_engine/adapters/trees/decision_tree.py
"""Phase 27 — single decision-tree adapters (3D stepped voxel terraces).

Covers ``DecisionTreeClassifier``, ``DecisionTreeRegressor``,
``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` (plain estimators and
``Pipeline``/``GridSearchCV``/``CalibratedClassifierCV`` wrappers alike —
the fitted tree is unwrapped for tracing while inference runs through the
full container).

Blueprint mapping (phase-27-model-decision-tree-voxel.md):

- Space partition: axis-aligned splits ``x[f] <= t`` (Cython ``tree_``
  struct: ``children_left/right``, ``feature`` (``-2`` = leaf),
  ``threshold``, ``impurity``, ``n_node_samples``, ``value``).
- Leaf lookup: ``model.apply(X)``; path lookup: CSR
  ``model.decision_path(X)`` sliced via ``indptr``/``indices`` only —
  never ``.toarray()`` (blueprint section 6.1).
- Exact tree-path contributions: ``phi[f_u] += E_v - E_u`` along
  root→leaf, so ``base + sum(phi) == E_leaf == prediction`` where
  ``E_m`` is the target-class probability (classifier) or node mean
  (regressor).

Waterfall convention (matches earlier adapters): binary mode reports the
resolved positive-class probability; multiclass mode reports the resolved
target class; regression reports the node mean.

Edge cases handled (blueprint section 7):

1. Deep trees: only the row's ``active_path`` (+ capped 512-node
   topology header) is exported — never the full node cloud.
2. Stump (``node_count == 1``): single-node path, zero contributions.
3. Sparse classes: target index clipped to ``value`` width, labels mapped
   through ``classes_``.
4. Multi-output trees: output 0 is traced, flagged in telemetry.
5. NaN rows: sklearn routes them natively; the breadcrumb's ``went_left``
   flag is read off the taken child (not ``<=``), which stays correct.
6. All-zero ``feature_importances_``: returned as-is; the frontend falls
   back to top-variance axes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
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
MAX_EXPORT_NODES = 512
MAX_PATH_STEPS = 32

_CLASSIFIER_TYPES = (DecisionTreeClassifier, ExtraTreeClassifier)
_REGRESSOR_TYPES = (DecisionTreeRegressor, ExtraTreeRegressor)
_TREE_TYPES = _CLASSIFIER_TYPES + _REGRESSOR_TYPES


def _unwrap_container(model_obj: Any) -> tuple[Any, Any, bool]:
    """Return ``(predict_obj, tree_estimator, calibrated)``.

    Unwraps ``Pipeline`` steps and ``CalibratedClassifierCV`` (first
    calibrated classifier's inner estimator); ``GridSearchCV``-style
    ``best_estimator_`` chains are followed too.
    """
    predict_obj = model_obj
    calibrated = False
    seen = 0
    while seen < 8:
        seen += 1
        best = getattr(model_obj, "best_estimator_", None)
        if best is not None:
            model_obj = best
            predict_obj = best
            continue
        steps = getattr(model_obj, "steps", None)
        if isinstance(steps, list) and steps:
            last = steps[-1]
            model_obj = last[1] if isinstance(last, (tuple, list)) and len(last) == 2 else last
            continue
        ccs = getattr(model_obj, "calibrated_classifiers_", None)
        if isinstance(ccs, list) and ccs:
            inner = getattr(ccs[0], "estimator", getattr(ccs[0], "base_estimator", None))
            if inner is not None:
                model_obj = inner
                calibrated = True
                continue
        break
    return predict_obj, model_obj, calibrated


def _final_estimator(model_obj: Any) -> Any:
    return _unwrap_container(model_obj)[1]


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


def _node_expectations(tree: Any, is_classifier: bool, target_idx: int) -> np.ndarray:
    """Per-node ``E_m``: target-class probability or node mean (output 0)."""
    vals = np.asarray(tree.value, dtype=np.float64)
    if vals.ndim == 3:
        vals = vals[:, 0, :]
    elif vals.ndim == 2:
        pass
    else:
        vals = vals.reshape(int(tree.node_count), -1)
    if not is_classifier:
        return np.asarray(vals[:, 0], dtype=np.float64).ravel()
    totals = np.sum(vals, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
    idx = int(min(max(int(target_idx), 0), probs.shape[1] - 1))
    return np.asarray(probs[:, idx], dtype=np.float64).ravel()


def _tree_arrays(tree: Any) -> dict[str, np.ndarray]:
    return {
        "children_left": np.asarray(tree.children_left).astype(np.int64).ravel(),
        "children_right": np.asarray(tree.children_right).astype(np.int64).ravel(),
        "feature": np.asarray(tree.feature).astype(np.int64).ravel(),
        "threshold": np.asarray(tree.threshold, dtype=np.float64).ravel(),
        "impurity": np.asarray(tree.impurity, dtype=np.float64).ravel(),
        "n_node_samples": np.asarray(tree.n_node_samples).astype(np.int64).ravel(),
    }


def _split_depths(children_left: np.ndarray, children_right: np.ndarray) -> np.ndarray:
    """BFS depth per node (root = 0)."""
    n = int(children_left.size)
    depth = np.full((n,), -1, dtype=np.int64)
    depth[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in (int(children_left[u]), int(children_right[u])):
            if v >= 0 and depth[v] < 0:
                depth[v] = depth[u] + 1
                stack.append(v)
    return depth


def _classes_of(estimator: Any) -> list[Any]:
    raw = getattr(estimator, "classes_", None)
    if raw is None:
        return [0, 1]
    return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]


def _topology(tree: Any, node_exp: np.ndarray) -> dict[str, Any]:
    arr = _tree_arrays(tree)
    n = min(int(tree.node_count), MAX_EXPORT_NODES)
    depth = _split_depths(arr["children_left"], arr["children_right"])
    return {
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
        "n_leaves": int(tree.n_leaves) if hasattr(tree, "n_leaves") else 0,
        "exported_nodes": int(n),
        "truncated": bool(int(tree.node_count) > MAX_EXPORT_NODES),
        "children_left": arr["children_left"][:n].tolist(),
        "children_right": arr["children_right"][:n].tolist(),
        "feature": arr["feature"][:n].tolist(),
        "threshold": [round(float(v), 5) for v in arr["threshold"][:n].tolist()],
        "impurity": [round(float(v), 5) for v in arr["impurity"][:n].tolist()],
        "n_node_samples": arr["n_node_samples"][:n].tolist(),
        "node_expectation": [round(float(v), 5) for v in np.asarray(node_exp[:n]).tolist()],
        "depth": depth[:n].tolist(),
    }


def _split_lines_for_axis(
    tree: Any, axis_idx: int, max_depth: int | None = None
) -> list[dict[str, Any]]:
    """Split thresholds on one axis, shallow-first (blueprint 3.1: thick top splits)."""
    arr = _tree_arrays(tree)
    depth = _split_depths(arr["children_left"], arr["children_right"])
    lines: list[dict[str, Any]] = []
    for m in range(int(tree.node_count)):
        f = int(arr["feature"][m])
        if f != int(axis_idx):
            continue
        d = int(depth[m])
        if max_depth is not None and d > max_depth:
            continue
        lines.append(
            {
                "node_id": int(m),
                "threshold": round(float(arr["threshold"][m]), 5),
                "depth": int(d),
                "impurity": round(float(arr["impurity"][m]), 5),
                "n_node_samples": int(arr["n_node_samples"][m]),
            }
        )
    lines.sort(key=lambda e: (e["depth"], e["threshold"]))
    return lines


class _TreeAdapterBase(BaseModelAdapter):
    is_classifier: bool = True

    @classmethod
    def _tree_estimator(cls, model_obj: Any) -> Any:
        return _final_estimator(model_obj)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        est = _final_estimator(estimator)
        imp = getattr(est, "feature_importances_", None)
        width = max(int(n_features), 0)
        if imp is None:
            return [None for _ in range(width)]
        arr = np.asarray(imp, dtype=np.float64).ravel()
        if arr.size != width:
            return [None for _ in range(width)]
        return [float(v) if np.isfinite(v) else 0.0 for v in arr.tolist()]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, _, _ = _unwrap_container(model_obj)
        X_in = _as_dense_float32(X_float32)
        try:
            preds = np.asarray(predict_obj.predict(X_in)).ravel()
        except (ValueError, AttributeError, TypeError):
            preds = np.zeros((X_in.shape[0],))
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if self.is_classifier:
            try:
                probas = np.asarray(predict_obj.predict_proba(X_in), dtype=np.float64)
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
                probabilities = [
                    [float(v) if np.isfinite(v) else 0.0 for v in row]
                    for row in probas.tolist()
                ]
                confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                probabilities, confidences = None, None
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

    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        predict_obj, est, calibrated = _unwrap_container(model_obj)
        classes = _classes_of(est)
        n_classes = len(classes)
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        elif self.is_classifier and n_classes == 2:
            resolved = 1  # binary convention: positive-class waterfall
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        is_binary = self.is_classifier and n_classes == 2

        X_native = _as_dense_float32(X_float32)
        tree = getattr(est, "tree_", None)
        if tree is None:
            raise ValueError(f"Estimator '{type(est).__name__}' exposes no fitted tree_")
        n_nodes = int(tree.node_count)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        node_exp = _node_expectations(tree, self.is_classifier, resolved)
        arr = _tree_arrays(tree)
        multi_output = bool(int(getattr(tree, "n_outputs", 1)) > 1)
        try:
            leaf_ids = np.asarray(est.apply(X_native), dtype=np.int64).ravel()
        except (ValueError, AttributeError, TypeError):
            leaf_ids = np.zeros((X_native.shape[0],), dtype=np.int64)
        try:
            csr = est.decision_path(X_native)
            indptr = np.asarray(csr.indptr, dtype=np.int64).ravel()
            indices = np.asarray(csr.indices, dtype=np.int64).ravel()
            have_path = True
        except (ValueError, AttributeError, TypeError):
            indptr, indices, have_path = np.zeros((X_native.shape[0] + 1,), dtype=np.int64), np.zeros(
                (0,), dtype=np.int64
            ), False
        try:
            native_preds = list(np.asarray(predict_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError, TypeError):
            native_preds = [classes[0] if classes else 0 for _ in row_indices]
        native_probas: np.ndarray | None = None
        if self.is_classifier:
            try:
                native_probas = np.asarray(predict_obj.predict_proba(X_native), dtype=np.float64)
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                native_probas = None
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        topo = _topology(tree, node_exp)
        try:
            criterion = str(getattr(est, "criterion", "gini"))
        except Exception:
            criterion = "gini"

        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < X_native.shape[0]):
                continue
            leaf = int(leaf_ids[row_id]) if row_id < leaf_ids.size else 0
            leaf = int(min(max(leaf, 0), n_nodes - 1))
            if have_path and row_id < indptr.size - 1:
                path = [int(v) for v in indices[indptr[row_id] : indptr[row_id + 1]].tolist()]
                path = [v for v in path if 0 <= v < n_nodes] or [0]
            else:
                path = [0]
            contribs = np.zeros((max(n_features, 0),), dtype=np.float64)
            steps: list[dict[str, Any]] = []
            for s_pos in range(len(path) - 1):
                u, v = path[s_pos], path[s_pos + 1]
                delta = float(node_exp[v] - node_exp[u])
                f_u = int(arr["feature"][u])
                if 0 <= f_u < contribs.size:
                    contribs[f_u] += delta
                if len(steps) < MAX_PATH_STEPS and f_u >= 0:
                    # Edge #5: direction from the taken child, NaN-safe.
                    went_left = bool(v == int(arr["children_left"][u]))
                    steps.append(
                        {
                            "node_id": int(u),
                            "next_node_id": int(v),
                            "feature_idx": int(f_u),
                            "feature_name": names[f_u] if f_u < len(names) else f"feature_{f_u}",
                            "threshold": round(float(arr["threshold"][u]), 5),
                            "went_left": bool(went_left),
                            "delta": round(float(delta), 5),
                            "impurity": round(float(arr["impurity"][u]), 5),
                        }
                    )
            base = float(node_exp[0]) if node_exp.size else 0.0
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if resolved < len(classes) else resolved
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            leaf_proba = (
                [float(v) for v in native_probas[row_id].tolist()]
                if native_probas is not None and row_id < native_probas.shape[0]
                else []
            )
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(is_binary),
                "algorithm": type(est).__name__,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "calibrated_wrapper": bool(calibrated),
                "multi_output_traced_index": 0 if multi_output else None,
                "is_multi_output": bool(multi_output),
                "criterion": criterion,
                "max_depth": int(tree.max_depth),
                "node_count": int(n_nodes),
                "n_leaves": int(tree.n_leaves) if hasattr(tree, "n_leaves") else 0,
                "leaf_id": int(leaf),
                "leaf_impurity": round(float(arr["impurity"][leaf]), 5),
                "leaf_n_samples": int(arr["n_node_samples"][leaf]),
                "leaf_expectation": float(node_exp[leaf]),
                "active_path": steps,
                "active_path_truncated": bool(len(path) - 1 > MAX_PATH_STEPS),
                "path_node_ids": [int(v) for v in path[: MAX_PATH_STEPS + 1]],
                "tree_topology": topo,
                "probabilities": leaf_proba,
                "predicted_class": final_pred,
                "winning_class_idx": int(resolved) if self.is_classifier else 0,
                "resolved_class_idx": int(resolved),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base),
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions={
                        (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                        for i in range(int(contribs.size))
                    },
                    extra_mechanics=extra,
                )
            )
        return traces

    def _surface_common(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        predict_obj, est, calibrated = _unwrap_container(model_obj)
        classes = _classes_of(est)
        n_classes = len(classes)
        if self.is_classifier:
            if n_classes == 2:
                resolved = 1
            elif target_class_idx is not None:
                resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            else:
                resolved = int(min(max(int(target_idx), 0), n_classes - 1))
        else:
            resolved = 0
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
            if self.is_classifier:
                probas = np.asarray(predict_obj.predict_proba(grid), dtype=np.float64)
                c_idx = int(min(resolved, probas.shape[1] - 1)) if probas.size else 0
                z_surface = (
                    np.asarray(probas[:, c_idx], dtype=np.float64).reshape(res, res)
                    if probas.size
                    else np.zeros((res, res))
                )
            else:
                preds = np.asarray(predict_obj.predict(grid), dtype=np.float64).ravel()
                z_surface = preds.reshape(res, res) if preds.size == n_points else np.zeros((res, res))
        except (ValueError, AttributeError, TypeError):
            z_surface = np.zeros((res, res), dtype=np.float64)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
        tree = getattr(est, "tree_", None)
        if tree is not None:
            node_exp = _node_expectations(tree, self.is_classifier, resolved)
            topo = _topology(tree, node_exp)
            split_ax = _split_lines_for_axis(tree, ax)
            split_ay = _split_lines_for_axis(tree, ay)
            depth_cap = int(tree.max_depth) if int(tree.max_depth) >= 0 else 0
        else:
            topo, split_ax, split_ay, depth_cap = {}, [], [], 0
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
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
                "algorithm": type(est).__name__,
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "calibrated_wrapper": bool(calibrated),
                "criterion": str(getattr(est, "criterion", "gini")),
                "depth_cap": int(depth_cap),
                "tree_topology": topo,
                "split_lines_x": split_ax,
                "split_lines_y": split_ay,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("decision_tree")
class DecisionTreeClassifierAdapter(_TreeAdapterBase):
    display_name = "Decision Tree (Voxel Terraces)"
    family_name = "Trees & Bagging"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.STEPPED_AXIS_BLOCKS
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_estimator(estimator), _CLASSIFIER_TYPES)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return self._trace_common(
            model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)


@ModelRegistry.register("decision_tree_regressor")
class DecisionTreeRegressorAdapter(_TreeAdapterBase):
    display_name = "Decision Tree Regressor (Voxel Terraces)"
    family_name = "Trees & Bagging"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.STEPPED_AXIS_BLOCKS
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_estimator(estimator), _REGRESSOR_TYPES)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return self._trace_common(
            model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)
