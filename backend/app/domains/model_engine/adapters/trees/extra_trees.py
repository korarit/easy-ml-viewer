# backend/app/domains/model_engine/adapters/trees/extra_trees.py
"""Phase 29 — extremely-randomized-trees bagging ensemble adapters (uncertainty halo).

Covers ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` (plain
estimators plus ``Pipeline`` / ``GridSearchCV``-style ``best_estimator_``
/ ``TransformedTargetRegressor`` wrappers — inference runs through the
full container while per-tree tracing unwraps to the fitted forest).

Blueprint mapping (phase-29-model-extra-trees.md):

- Inherits the forest fast path (single ``apply`` + node-expectation
  lookup, ``k_export`` cap) from
  :class:`app.domains.model_engine.adapters.trees.random_forest._ForestAdapterBase`.
- Uncertainty decomposition (always ``ddof=0`` so ``n_estimators == 1``
  yields ``0.0``, never ``NaN`` — edge #1):
  ``uncertainty_radius = clip(2*std, 0, 1)`` for classifiers;
  ``clip(2*std / max(ptp(mean), 1e-3), 0, 1)`` for regressors (edge #2).
- Variance attribution: per-tree path contributions are accumulated
  streaming (``sum``/``sumsq`` over ``k``) so no
  ``(k_export, n_samples, n_features)`` 3-D array is ever materialised —
  ``contribution_std`` is the inter-tree ``std`` of ``phi`` per feature.
- ``split_threshold_distribution``: up to 200 random cut-points per
  feature harvested from the fitted ``tree_.threshold`` arrays for the 3D
  rug plot (blueprint section 6.2).
- 3D surface returns the mean-prediction terraces as ``z_values`` plus the
  epistemic ``uncertainty_z_values`` grid (per-grid-point inter-tree std
  mapped to halo radius) so the HUD can toggle Mean | Uncertainty.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from app.domains.model_engine.adapters.trees.random_forest import (
    MAX_EXPORTED_TREES,
    MAX_PATH_STEPS,
    MAX_SURFACE_RESOLUTION,
    _as_dense_float32,
    _classes_of,
    _ForestAdapterBase,
    _node_expectations,
    _transform_for_estimators,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import (
    GridSpec3D,
    InternalFlowRowTrace,
    SurfaceMesh3DData,
    TaskType,
    Visual3DType,
)

_CLASSIFIER_TYPES = (ExtraTreesClassifier,)
_REGRESSOR_TYPES = (ExtraTreesRegressor,)

MAX_SPLITS_PER_FEATURE = 200


def _unwrap_et_container(model_obj: Any) -> tuple[Any, Any]:
    """Return ``(predict_obj, forest_estimator)`` for Extra Trees wrappers.

    Handles ``GridSearchCV``-style ``best_estimator_`` chains, ``Pipeline``
    ``steps``, and ``TransformedTargetRegressor`` ``regressor_``.
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
        reg = getattr(model_obj, "regressor_", None)
        if reg is not None:
            model_obj = reg
            continue
        break
    return predict_obj, model_obj


def _final_et_estimator(model_obj: Any) -> Any:
    return _unwrap_et_container(model_obj)[1]


def extract_split_threshold_distribution(
    forest: Any, n_features: int, max_splits_per_feature: int = MAX_SPLITS_PER_FEATURE
) -> dict[str, list[float]]:
    """Harvest random cut-points per feature (capped) for the 3D rug plot."""
    out: dict[str, list[float]] = {str(i): [] for i in range(max(int(n_features), 0))}
    if not out:
        return out
    for est in list(getattr(forest, "estimators_", []) or [])[:MAX_EXPORTED_TREES]:
        try:
            feats = np.asarray(est.tree_.feature, dtype=np.int64).ravel()
            thresh = np.asarray(est.tree_.threshold, dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            continue
        for f_idx, th in zip(feats.tolist(), thresh.tolist()):
            key = str(int(f_idx))
            if f_idx >= 0 and key in out and len(out[key]) < max_splits_per_feature:
                if np.isfinite(th):
                    out[key].append(round(float(th), 4))
    return out


class _ExtraTreesBase(_ForestAdapterBase):
    """Shared Extra Trees tracing with uncertainty + variance attribution."""

    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        predict_obj, forest = _unwrap_et_container(model_obj)
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
        # ddof=0 always (edge #1: n_estimators == 1 -> 0.0, never NaN).
        inter_mean = np.mean(tree_preds, axis=1)
        inter_std = np.std(tree_preds, axis=1, ddof=0)
        q05 = np.percentile(tree_preds, 5, axis=1)
        q95 = np.percentile(tree_preds, 95, axis=1)
        if self.is_classifier:
            uncertainty_radius = np.clip(2.0 * inter_std, 0.0, 1.0)
            votes_pos = np.mean(tree_preds >= 0.5, axis=1)
            consensus = np.maximum(votes_pos, 1.0 - votes_pos)
        else:
            pred_span = max(float(np.ptp(inter_mean)) if inter_mean.size else 0.0, 1e-3)
            uncertainty_radius = np.clip((2.0 * inter_std) / pred_span, 0.0, 1.0)
            consensus = 1.0 / (1.0 + inter_std)

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
        predictive_entropy: np.ndarray | None = None
        if native_probas is not None and n_classes > 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                clipped = np.clip(native_probas, 1e-12, 1.0)
                predictive_entropy = -np.sum(native_probas * np.log(clipped), axis=1)

        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        base_val = float(np.mean([exp[0] for exp in node_exp_list[:k_export]])) if k_export else 0.0

        # Streaming per-tree path contributions: sum/sumsq over k (no 3-D array).
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

        n_rows = X_native.shape[0]
        width = max(n_features, 0)
        sum_c = np.zeros((n_rows, width), dtype=np.float64)
        sumsq_c = np.zeros((n_rows, width), dtype=np.float64)
        per_tree_path_len_sums = np.zeros((n_rows,), dtype=np.float64)
        first_path_steps_by_row: dict[int, list[dict[str, Any]]] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            for k in range(k_export):
                csr = csr_list[k]
                feats = feat_list[k]
                exp_k = node_exp_list[k]
                if csr is None:
                    continue
                try:
                    indptr = np.asarray(csr.indptr, dtype=np.int64).ravel()
                    indices = np.asarray(csr.indices, dtype=np.int64).ravel()
                except (AttributeError, TypeError, ValueError):
                    continue
                contrib_k = np.zeros((n_rows, width), dtype=np.float64)
                for row_pos in range(n_rows):
                    if not (row_pos < indptr.size - 1):
                        continue
                    path = [int(v) for v in indices[indptr[row_pos] : indptr[row_pos + 1]].tolist()]
                    path = [v for v in path if 0 <= v < int(exp_k.size)] or [0]
                    per_tree_path_len_sums[row_pos] += len(path)
                    for s_pos in range(len(path) - 1):
                        u, v = path[s_pos], path[s_pos + 1]
                        delta = float(exp_k[v] - exp_k[u])
                        f_u = int(feats[u]) if u < feats.size else -2
                        if 0 <= f_u < width:
                            contrib_k[row_pos, f_u] += delta
                        if (
                            k == 0
                            and row_pos in row_indices
                            and f_u >= 0
                            and len(first_path_steps_by_row.get(row_pos, [])) < MAX_PATH_STEPS
                        ):
                            try:
                                tl = int(estimators[0].tree_.children_left[u])
                            except (AttributeError, TypeError, IndexError):
                                tl = -1
                            first_path_steps_by_row.setdefault(row_pos, []).append(
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
                sum_c += contrib_k
                sumsq_c += contrib_k * contrib_k
        denom = max(k_export, 1)
        mean_c = sum_c / denom
        var_c = np.maximum(sumsq_c / denom - mean_c * mean_c, 0.0)
        std_c = np.sqrt(var_c)

        try:
            bootstrap = getattr(forest, "bootstrap", False)
        except Exception:
            bootstrap = False
        try:
            oob = getattr(forest, "oob_score_", None)
            oob_val = float(oob) if oob is not None and np.isfinite(float(oob)) else None
        except (TypeError, ValueError):
            oob_val = None

        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < n_rows):
                continue
            contribs = mean_c[row_id]
            cstd = std_c[row_id]
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
            mean_path = float(per_tree_path_len_sums[row_id] / denom) if denom else 0.0
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(is_binary),
                "algorithm": type(forest).__name__,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "extra_trees",
                "n_estimators_total": int(n_est),
                "n_estimators_exported": int(k_export),
                "consensus_ratio": round(float(consensus[row_id]), 4),
                "inter_tree_mean": round(float(inter_mean[row_id]), 5),
                "inter_tree_std": round(float(inter_std[row_id]), 5),
                "uncertainty_radius": round(float(uncertainty_radius[row_id]), 4),
                "prediction_p05": round(float(q05[row_id]), 4),
                "prediction_p95": round(float(q95[row_id]), 4),
                "tree_predictions_sample": tree_row,
                "leaf_indices_sample": [int(v) for v in leaf_row],
                "mean_path_length": round(mean_path, 3),
                "first_tree_active_path": first_path_steps_by_row.get(row_id, []),
                "first_tree_leaf_id": int(leaf_row[0]) if leaf_row else 0,
                "oob_score": oob_val,
                "bootstrap": bool(bootstrap),
                "max_depth": getattr(forest, "max_depth", None),
                "predictive_entropy": (
                    round(float(predictive_entropy[row_id]), 5)
                    if predictive_entropy is not None and row_id < predictive_entropy.size
                    else None
                ),
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
                        for i in range(int(width))
                    },
                    extra_mechanics={
                        **extra,
                        "contribution_std": {
                            (names[i] if i < len(names) else f"feature_{i}"): float(cstd[i])
                            for i in range(int(width))
                        },
                    },
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
        predict_obj, forest = _unwrap_et_container(model_obj)
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

        # Epistemic uncertainty grid: per-grid-point inter-tree std (ddof=0).
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                node_exp_list = self._per_tree_expectations(forest, estimators, resolved)
                tree_preds_grid, _ = self._predict_all_trees(
                    predict_obj, forest, estimators, node_exp_list, grid
                )
            std_grid = np.std(tree_preds_grid, axis=1, ddof=0).reshape(res, res)
            std_grid = np.where(np.isfinite(std_grid), std_grid, 0.0)
            if self.is_classifier:
                halo_grid = np.clip(2.0 * std_grid, 0.0, 1.0)
            else:
                span = max(float(np.ptp(z_surface)) if z_surface.size else 0.0, 1e-3)
                halo_grid = np.clip((2.0 * std_grid) / span, 0.0, 1.0)
        except (ValueError, AttributeError, TypeError, MemoryError):
            std_grid = np.zeros((res, res), dtype=np.float64)
            halo_grid = np.zeros((res, res), dtype=np.float64)

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
        try:
            bootstrap = bool(getattr(forest, "bootstrap", False))
        except Exception:
            bootstrap = False
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
                "model_family": "extra_trees",
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
                "bootstrap": bootstrap,
                "uncertainty_z_values": [float(v) for v in np.asarray(std_grid, dtype=np.float64).ravel()],
                "uncertainty_halo_values": [float(v) for v in np.asarray(halo_grid, dtype=np.float64).ravel()],
                "split_threshold_distribution": extract_split_threshold_distribution(
                    forest, n_features
                ),
            },
        )


@ModelRegistry.register("extra_trees")
class ExtraTreesClassifierAdapter(_ExtraTreesBase):
    display_name = "Extra Trees (Uncertainty Halo)"
    family_name = "Trees & Bagging"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_et_estimator(estimator), _CLASSIFIER_TYPES)

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


@ModelRegistry.register("extra_trees_regressor")
class ExtraTreesRegressorAdapter(_ExtraTreesBase):
    display_name = "Extra Trees Regressor (Uncertainty Halo)"
    family_name = "Trees & Bagging"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_et_estimator(estimator), _REGRESSOR_TYPES)

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
