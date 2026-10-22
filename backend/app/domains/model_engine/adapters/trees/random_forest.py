# backend/app/domains/model_engine/adapters/trees/random_forest.py
"""Phase 28 — random-forest bagging ensemble adapters (3D voting arena).

Covers ``RandomForestClassifier`` and ``RandomForestRegressor`` (plain
estimators plus ``Pipeline`` / ``GridSearchCV``-style ``best_estimator_``
wrappers — inference runs through the full container while per-tree
tracing unwraps to the fitted forest).

Blueprint mapping (phase-28-model-random-forest-consensus.md):

- Bagging consensus: forest mean ``y_hat = mean_k T_k(x)`` (+ inter-tree
  ``std``); classifier soft-voting ``P = mean_k p_k(x)`` with hard-vote
  ``consensus = max(votes_pos, 1 - votes_pos)`` (regression:
  ``1 / (1 + std)``).
- Fast per-tree lookup: single ``forest.apply(X_f32)`` call
  ``(n_samples, n_estimators)`` + per-tree node-expectation lookup tables
  (``node_exp_k[leaf]``) instead of ``K`` Python ``predict`` round-trips
  (blueprint section 6.2).
- Ensemble contributions: mean of the first ``k_export`` tree-path deltas
  ``Phi_j = mean_k phi_{k,j}`` so ``base + sum(Phi) == inter_tree_mean``
  (which equals the forest prediction up to float32 rounding).
- Cap: only the first ``MAX_EXPORTED_TREES = 100`` trees are exported per
  row (``tree_predictions_sample`` / ``leaf_indices_sample``); summary
  stats always cover all ``n_estimators`` (blueprint section 6.3).

Waterfall convention (matches Phase 27): binary mode reports the resolved
positive-class probability; multiclass reports the resolved target class;
regression reports the node-mean average.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

MAX_EXPORTED_TREES = 100
MAX_SURFACE_RESOLUTION = 100
MAX_PATH_STEPS = 32

_CLASSIFIER_TYPES = (RandomForestClassifier,)
_REGRESSOR_TYPES = (RandomForestRegressor,)
_FOREST_TYPES = _CLASSIFIER_TYPES + _REGRESSOR_TYPES


def _unwrap_container(model_obj: Any) -> tuple[Any, Any]:
    """Return ``(predict_obj, forest_estimator)``.

    Follows ``GridSearchCV``-style ``best_estimator_`` chains and unwraps
    ``Pipeline`` ``steps`` to the terminal estimator. Inference runs
    through ``predict_obj`` (full pipeline); per-tree tracing runs on the
    fitted forest itself.
    """
    predict_obj = model_obj
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
        break
    return predict_obj, model_obj


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


def _transform_for_estimators(predict_obj: Any, forest: Any, X_f32: np.ndarray) -> np.ndarray:
    """Apply pipeline preprocessing (all steps but last) so tree indices align."""
    steps = getattr(predict_obj, "steps", None)
    if not (isinstance(steps, list) and steps):
        return X_f32
    try:
        Xt: Any = X_f32
        for _, step_est in steps[:-1]:
            if hasattr(step_est, "transform"):
                Xt = step_est.transform(Xt)
            else:
                return X_f32
        return _as_dense_float32(Xt)
    except (ValueError, AttributeError, TypeError):
        return X_f32


def _node_expectations(forest: Any, tree_est: Any, is_classifier: bool, target_idx: int) -> np.ndarray:
    vals = np.asarray(tree_est.tree_.value, dtype=np.float64)
    if vals.ndim == 3:
        vals = vals[:, 0, :]
    elif vals.ndim == 2:
        pass
    else:
        vals = vals.reshape(int(tree_est.tree_.node_count), -1)
    if not is_classifier:
        return np.asarray(vals[:, 0], dtype=np.float64).ravel()
    totals = np.sum(vals, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
    # Map forest-level class index onto this tree's own class axis (edge #5).
    forest_classes = list(np.asarray(getattr(forest, "classes_", [])).ravel())
    tree_classes = list(np.asarray(getattr(tree_est, "classes_", [])).ravel())
    resolved_label: Any = None
    try:
        resolved_label = forest_classes[int(target_idx)] if forest_classes else int(target_idx)
    except (IndexError, TypeError, ValueError):
        resolved_label = int(target_idx)
    col = 0
    try:
        if tree_classes:
            col = list(tree_classes).index(resolved_label)
    except ValueError:
        col = 0
    col = int(min(max(col, 0), probs.shape[1] - 1))
    return np.asarray(probs[:, col], dtype=np.float64).ravel()


def _classes_of(estimator: Any) -> list[Any]:
    raw = getattr(estimator, "classes_", None)
    if raw is None:
        return [0, 1]
    return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]


class _ForestAdapterBase(BaseModelAdapter):
    is_classifier: bool = True

    def _resolve_target(
        self, forest: Any, target_idx: int, target_class_idx: int | None
    ) -> tuple[int, list[Any]]:
        classes = _classes_of(forest)
        n_classes = len(classes)
        if target_class_idx is not None and n_classes:
            resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
        elif self.is_classifier and n_classes == 2:
            resolved = 1
        else:
            resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
        return resolved, classes

    def _per_tree_expectations(
        self, forest: Any, estimators: list[Any], resolved: int
    ) -> list[np.ndarray]:
        return [
            _node_expectations(forest, est, self.is_classifier, resolved) for est in estimators
        ]

    def _predict_all_trees(
        self,
        predict_obj: Any,
        forest: Any,
        estimators: list[Any],
        node_exp_list: list[np.ndarray],
        X_raw_f32: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single ``apply`` + fancy-index lookup (blueprint 6.2)."""
        X_t = _transform_for_estimators(predict_obj, forest, X_raw_f32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                leaf_matrix = np.asarray(forest.apply(X_t), dtype=np.int64)
            except (ValueError, AttributeError, TypeError):
                # Fallback: per-tree apply (keeps odd wrappers alive).
                cols = []
                for est in estimators:
                    try:
                        cols.append(np.asarray(est.apply(X_t), dtype=np.int64).ravel())
                    except (ValueError, AttributeError, TypeError):
                        cols.append(np.zeros((X_raw_f32.shape[0],), dtype=np.int64))
                leaf_matrix = np.stack(cols, axis=1) if cols else np.zeros((X_raw_f32.shape[0], 0), dtype=np.int64)
        if leaf_matrix.ndim == 1:
            leaf_matrix = leaf_matrix.reshape(-1, 1)
        n_samples = X_raw_f32.shape[0]
        n_est = len(estimators)
        tree_preds = np.empty((n_samples, n_est), dtype=np.float64)
        for k in range(n_est):
            exp_k = node_exp_list[k]
            col = leaf_matrix[:, k] if k < leaf_matrix.shape[1] else np.zeros((n_samples,), dtype=np.int64)
            col = np.clip(col, 0, max(int(exp_k.size) - 1, 0))
            tree_preds[:, k] = exp_k[col]
        return tree_preds, leaf_matrix

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
        predict_obj, _ = _unwrap_container(model_obj)
        X_in = _as_dense_float32(X_float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                preds = np.asarray(predict_obj.predict(X_in)).ravel()
            except (ValueError, AttributeError, TypeError):
                preds = np.zeros((X_in.shape[0],))
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if self.is_classifier:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    probas = np.asarray(predict_obj.predict_proba(X_in), dtype=np.float64)
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
                probabilities = [
                    [float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()
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
        predict_obj, forest = _unwrap_container(model_obj)
        estimators: list[Any] = list(getattr(forest, "estimators_", []) or [])
        if not estimators:
            raise ValueError(f"Estimator '{type(forest).__name__}' exposes no fitted estimators_")
        resolved, classes = self._resolve_target(forest, target_idx, target_class_idx)
        n_classes = len(classes)
        is_binary = bool(self.is_classifier and n_classes == 2)

        X_native = _as_dense_float32(X_float32)
        X_t = _transform_for_estimators(predict_obj, forest, X_native)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        node_exp_list = self._per_tree_expectations(forest, estimators, resolved)
        n_est = len(estimators)
        k_export = min(n_est, MAX_EXPORTED_TREES)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            tree_preds, leaf_matrix = self._predict_all_trees(
                predict_obj, forest, estimators, node_exp_list, X_native
            )
        inter_mean = np.mean(tree_preds, axis=1)
        inter_std = np.std(tree_preds, axis=1, ddof=0)
        if self.is_classifier:
            votes_pos = np.mean(tree_preds >= 0.5, axis=1)
            consensus = np.maximum(votes_pos, 1.0 - votes_pos)
        else:
            consensus = 1.0 / (1.0 + inter_std)

        # Native forest predictions (ground truth for parity checks).
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                native_preds = list(np.asarray(predict_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError, TypeError):
            native_preds = [classes[0] if classes else 0 for _ in row_indices]
        native_probas: np.ndarray | None = None
        if self.is_classifier:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    native_probas = np.asarray(predict_obj.predict_proba(X_native), dtype=np.float64)
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                native_probas = None

        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        base_val = float(np.mean([exp[0] for exp in node_exp_list[:k_export]])) if k_export else 0.0

        # Per-row ensemble contributions: mean of per-tree path deltas.
        # CSR decision_path per exported tree, sliced via indptr (never .toarray()).
        csr_list: list[Any] = []
        feat_list: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            for k in range(k_export):
                est = estimators[k]
                try:
                    csr_list.append(est.decision_path(X_t))
                except (ValueError, AttributeError, TypeError):
                    csr_list.append(None)
                try:
                    feat_list.append(np.asarray(est.tree_.feature, dtype=np.int64).ravel())
                except (AttributeError, TypeError):
                    feat_list.append(np.zeros((0,), dtype=np.int64))

        try:
            n_jobs = getattr(forest, "n_jobs", None)
        except Exception:
            n_jobs = None
        try:
            oob = getattr(forest, "oob_score_", None)
            oob_val = float(oob) if oob is not None and np.isfinite(float(oob)) else None
        except (TypeError, ValueError):
            oob_val = None

        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < X_native.shape[0]):
                continue
            contribs = np.zeros((max(n_features, 0),), dtype=np.float64)
            per_tree_path_len: list[int] = []
            first_path_steps: list[dict[str, Any]] = []
            for k in range(k_export):
                csr = csr_list[k]
                feats = feat_list[k]
                exp_k = node_exp_list[k]
                if csr is None:
                    per_tree_path_len.append(0)
                    continue
                try:
                    indptr = np.asarray(csr.indptr, dtype=np.int64).ravel()
                    indices = np.asarray(csr.indices, dtype=np.int64).ravel()
                except (AttributeError, TypeError, ValueError):
                    per_tree_path_len.append(0)
                    continue
                if not (row_id < indptr.size - 1):
                    per_tree_path_len.append(0)
                    continue
                path = [int(v) for v in indices[indptr[row_id] : indptr[row_id + 1]].tolist()]
                path = [v for v in path if 0 <= v < int(exp_k.size)] or [0]
                per_tree_path_len.append(len(path))
                for s_pos in range(len(path) - 1):
                    u, v = path[s_pos], path[s_pos + 1]
                    delta = float(exp_k[v] - exp_k[u])
                    f_u = int(feats[u]) if u < feats.size else -2
                    if 0 <= f_u < contribs.size:
                        contribs[f_u] += delta / k_export
                    if k == 0 and len(first_path_steps) < MAX_PATH_STEPS and f_u >= 0:
                        try:
                            tl = int(estimators[0].tree_.children_left[u])
                        except (AttributeError, TypeError, IndexError):
                            tl = -1
                        first_path_steps.append(
                            {
                                "node_id": int(u),
                                "next_node_id": int(v),
                                "feature_idx": int(f_u),
                                "feature_name": names[f_u] if f_u < len(names) else f"feature_{f_u}",
                                "threshold": round(float(estimators[0].tree_.threshold[u]), 5),
                                "went_left": bool(v == tl),
                                "delta": round(float(delta), 5),
                            }
                        )
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if resolved < len(classes) else resolved
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            leaf_row = (
                leaf_matrix[row_id, :k_export].astype(int).tolist()
                if row_id < leaf_matrix.shape[0]
                else [0] * k_export
            )
            tree_row = [round(float(v), 4) for v in tree_preds[row_id, :k_export].tolist()]
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
                "algorithm": type(forest).__name__,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "random_forest",
                "n_estimators_total": int(n_est),
                "n_estimators_exported": int(k_export),
                "consensus_ratio": round(float(consensus[row_id]), 4),
                "inter_tree_mean": round(float(inter_mean[row_id]), 5),
                "inter_tree_std": round(float(inter_std[row_id]), 5),
                "tree_predictions_sample": tree_row,
                "leaf_indices_sample": [int(v) for v in leaf_row],
                "mean_path_length": round(float(np.mean(per_tree_path_len)) if per_tree_path_len else 0.0, 3),
                "first_tree_active_path": first_path_steps,
                "first_tree_leaf_id": int(leaf_row[0]) if leaf_row else 0,
                "oob_score": oob_val,
                "n_jobs": n_jobs,
                "max_depth": getattr(forest, "max_depth", None),
                "probabilities": leaf_proba,
                "predicted_class": final_pred,
                "winning_class_idx": int(resolved) if self.is_classifier else 0,
                "resolved_class_idx": int(resolved),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base_val),
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
        predict_obj, forest = _unwrap_container(model_obj)
        estimators: list[Any] = list(getattr(forest, "estimators_", []) or [])
        resolved, classes = self._resolve_target(forest, target_idx, target_class_idx)
        n_classes = len(classes)
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
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
        # Forest summary overlays: first-tree topology as terrace reference.
        first_topo: dict[str, Any] = {}
        if estimators:
            try:
                t0 = estimators[0].tree_
                first_topo = {
                    "node_count": int(t0.node_count),
                    "max_depth": int(t0.max_depth),
                    "n_leaves": int(t0.n_leaves) if hasattr(t0, "n_leaves") else 0,
                }
            except (AttributeError, TypeError):
                first_topo = {}
        depths: list[int] = []
        for est in estimators[: min(len(estimators), 20)]:
            try:
                depths.append(int(est.tree_.max_depth))
            except (AttributeError, TypeError):
                continue
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        try:
            imp = np.asarray(getattr(forest, "feature_importances_", []), dtype=np.float64).ravel()
            importances = [round(float(v), 5) for v in imp.tolist()] if imp.size else []
        except (TypeError, ValueError):
            importances = []
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
                "algorithm": type(forest).__name__,
                "model_family": "random_forest",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "n_estimators_total": int(len(estimators)),
                "n_estimators_exported": int(min(len(estimators), MAX_EXPORTED_TREES)),
                "first_tree_topology": first_topo,
                "mean_max_depth": round(float(np.mean(depths)), 3) if depths else 0.0,
                "feature_importances": importances,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("random_forest")
class RandomForestClassifierAdapter(_ForestAdapterBase):
    display_name = "Random Forest (Voting Arena)"
    family_name = "Trees & Bagging"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
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


@ModelRegistry.register("random_forest_regressor")
class RandomForestRegressorAdapter(_ForestAdapterBase):
    display_name = "Random Forest Regressor (Voting Arena)"
    family_name = "Trees & Bagging"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
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
