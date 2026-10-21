"""Phase 09 — surface mesh builder (chunked batch evaluation + world normalization).

Converts a float32 feature matrix and a fitted adapter into a
:class:`SurfaceMesh3DEntity`: LOD grid heights, queue point positions and
PCA manifold data, all normalized into the ``[-5, +5]`` world cube.

Memory: model evaluation always runs in ``PREDICT_CHUNK_SIZE = 2,500``-row
mini-batches (wide-column guard §5.1); only flat float32 buffers survive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.math_3d import (
    AxisScaleMeta,
    WorldCoordinateScaler,
    build_slice_grid_matrix,
    compute_3d_pca_projection,
    safe_min_max_normalize,
)
from app.domains.spatial_projection.entities import SurfaceMesh3DEntity
from app.domains.spatial_projection.value_objects import (
    ProjectionConfigVO,
    ZMode,
)

PREDICT_CHUNK_SIZE = 2_500


def _sanitize_finite(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    fill = float(np.median(finite)) if finite.size else float(fallback)
    return np.where(np.isfinite(arr), arr, fill)


def _safe_world(values: np.ndarray) -> np.ndarray:
    """Final gate: no NaN/Inf may reach a THREE.BufferAttribute (edge #4).

    Non-finite world coords (possible with PASS_NAN mappings) collapse to
    the cube center instead of poisoning the whole draw call.
    """
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=7.0, neginf=-7.0)


class SurfaceMeshService:
    """Stateless builder for axis-explorer and PCA decision surfaces."""

    # -- batch prediction primitives -----------------------------------
    @staticmethod
    def predict_values(adapter: Any, model_obj: Any, matrix: np.ndarray) -> np.ndarray:
        """Vectorized predictions in ≤2,500-row chunks (float64, sanitized)."""
        M = np.ascontiguousarray(matrix, dtype=np.float32)
        out = np.empty((M.shape[0],), dtype=np.float64)
        out[:] = np.nan
        offset = 0
        while offset < M.shape[0]:
            chunk = M[offset : offset + PREDICT_CHUNK_SIZE]
            result = adapter.predict_batch(model_obj, np.ascontiguousarray(chunk, dtype=np.float32))
            vals = np.asarray(list(result.predictions), dtype=np.float64)
            out[offset : offset + vals.shape[0]] = vals
            del chunk, result, vals
            offset += PREDICT_CHUNK_SIZE
        return _sanitize_finite(out)

    @staticmethod
    def predict_probabilities(
        model_obj: Any, final_est: Any, matrix: np.ndarray, class_index: int
    ) -> np.ndarray | None:
        """Class-k probabilities in chunks, or ``None`` when unsupported (edge #6)."""
        est = model_obj if hasattr(model_obj, "predict_proba") else final_est
        if not hasattr(est, "predict_proba"):
            return None
        try:
            M = np.ascontiguousarray(matrix, dtype=np.float32)
            n_classes = len(getattr(est, "classes_", [0, 1]))
            col = max(0, min(int(class_index), max(0, n_classes - 1)))
            out = np.empty((M.shape[0],), dtype=np.float64)
            out[:] = np.nan
            offset = 0
            while offset < M.shape[0]:
                proba = np.asarray(est.predict_proba(M[offset : offset + PREDICT_CHUNK_SIZE]), dtype=np.float64)
                if proba.ndim != 2 or proba.shape[1] == 0:
                    return None
                take = min(col, proba.shape[1] - 1)
                out[offset : offset + proba.shape[0]] = proba[:, take]
                del proba
                offset += PREDICT_CHUNK_SIZE
            if not np.isfinite(out).any():
                return None
            return _sanitize_finite(np.clip(out, 0.0, 1.0), fallback=0.5)
        except Exception:
            return None

    # -- z-mode resolution ----------------------------------------------
    @staticmethod
    def _third_feature_index(n_features: int, x_idx: int, y_idx: int) -> int | None:
        for i in range(n_features):
            if i != x_idx and i != y_idx:
                return i
        return None

    # -- axis explorer ----------------------------------------------------
    @staticmethod
    def build_axis_surface(
        *,
        X: np.ndarray,
        feature_names: list[str],
        x_idx: int,
        y_idx: int,
        cfg: ProjectionConfigVO,
        adapter: Any,
        model_obj: Any,
        final_est: Any,
        actuals: np.ndarray | None,
        queue_rows: list[int],
        current_row_values: np.ndarray | None = None,
    ) -> SurfaceMesh3DEntity:
        res = cfg.grid_resolution
        grid, x_range, y_range = build_slice_grid_matrix(
            X, x_idx, y_idx, res, cfg.pinning_mode.value,
            current_row_values=current_row_values,
            use_robust_range=cfg.use_robust_range,
        )

        # Constant-axis badge (edge #1): raw span below epsilon on either axis.
        def _raw_span(col: np.ndarray) -> float:
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                return 0.0
            return float(np.max(finite) - np.min(finite))

        constant_warning = bool(_raw_span(X[:, x_idx]) < 1e-9 or _raw_span(X[:, y_idx]) < 1e-9)

        grid_pred = SurfaceMeshService.predict_values(adapter, model_obj, grid)

        n_features = X.shape[1]
        resolved = ZMode.PREDICTION
        if cfg.z_mode == ZMode.PROBABILITY:
            proba = SurfaceMeshService.predict_probabilities(model_obj, final_est, grid, cfg.target_class_index)
            grid_z = proba if proba is not None else grid_pred
            resolved = ZMode.PROBABILITY if proba is not None else ZMode.PREDICTION
        elif cfg.z_mode == ZMode.RESIDUAL_ERROR and actuals is not None and np.isfinite(actuals).any():
            baseline = float(np.nanmedian(actuals[np.isfinite(actuals)]))
            grid_z = np.abs(grid_pred - baseline)
            resolved = ZMode.RESIDUAL_ERROR
        elif cfg.z_mode == ZMode.FEATURE_3:
            third = SurfaceMeshService._third_feature_index(n_features, x_idx, y_idx)
            if third is not None:
                grid_z = _sanitize_finite(grid[:, third].astype(np.float64))
                resolved = ZMode.FEATURE_3
            else:
                grid_z = grid_pred
        else:
            grid_z = grid_pred

        rows = np.asarray([r for r in queue_rows if 0 <= r < X.shape[0]], dtype=np.int64)
        Q = X[rows] if rows.size else np.empty((0, n_features), dtype=np.float32)
        q_pred = SurfaceMeshService.predict_values(adapter, model_obj, Q) if rows.size else np.empty((0,))
        if resolved == ZMode.PROBABILITY and rows.size:
            q_proba = SurfaceMeshService.predict_probabilities(model_obj, final_est, Q, cfg.target_class_index)
            point_z = q_proba if q_proba is not None else q_pred
            if q_proba is None:
                resolved = ZMode.PREDICTION
        elif resolved == ZMode.RESIDUAL_ERROR and rows.size and actuals is not None:
            point_z = np.abs(q_pred - actuals[rows])
        elif resolved == ZMode.FEATURE_3 and rows.size:
            third = SurfaceMeshService._third_feature_index(n_features, x_idx, y_idx)
            point_z = _sanitize_finite(Q[:, third].astype(np.float64)) if third is not None else q_pred
        else:
            point_z = q_pred

        z_pool = [grid_z[np.isfinite(grid_z)]]
        if point_z.size and np.isfinite(point_z).any():
            z_pool.append(point_z[np.isfinite(point_z)].astype(np.float64))
        z_values = (
            np.concatenate([a.astype(np.float32) for a in z_pool if a.size])
            if any(a.size for a in z_pool)
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        z_name = {
            ZMode.PREDICTION: "Predicted value",
            ZMode.PROBABILITY: f"P(class #{cfg.target_class_index})",
            ZMode.RESIDUAL_ERROR: "Residual |y - ŷ|",
            ZMode.FEATURE_3: feature_names[SurfaceMeshService._third_feature_index(n_features, x_idx, y_idx) or 0],
        }[resolved]

        x_scale = WorldCoordinateScaler.build_axis_scale(feature_names[x_idx], X[:, x_idx], feature_idx=x_idx)
        y_scale = WorldCoordinateScaler.build_axis_scale(feature_names[y_idx], X[:, y_idx], feature_idx=y_idx)
        z_scale = WorldCoordinateScaler.build_axis_scale(z_name, z_values, feature_idx=None)

        gx = np.linspace(x_range[0], x_range[1], res, dtype=np.float32)
        gy = np.linspace(y_range[0], y_range[1], res, dtype=np.float32)
        gx_w = WorldCoordinateScaler.to_world_coords(gx, x_scale)
        gy_w = WorldCoordinateScaler.to_world_coords(gy, y_scale)
        gz_w = WorldCoordinateScaler.to_world_coords(grid_z.astype(np.float32), z_scale)

        points_world: list[list[float]] = []
        trajectory: list[dict[str, Any]] = []
        if rows.size:
            qx_raw = X[rows][:, x_idx].astype(np.float64)
            qy_raw = X[rows][:, y_idx].astype(np.float64)
            wx = _safe_world(WorldCoordinateScaler.to_world_coords(qx_raw.astype(np.float32), x_scale))
            wy = _safe_world(WorldCoordinateScaler.to_world_coords(qy_raw.astype(np.float32), y_scale))
            safe_pz = np.where(np.isfinite(point_z), point_z, (z_scale.raw_min + z_scale.raw_max) / 2.0)
            wz = _safe_world(WorldCoordinateScaler.to_world_coords(safe_pz.astype(np.float32), z_scale))
            for t, orig in enumerate(rows.tolist()):
                raw_a: float | None = None
                if actuals is not None and 0 <= int(orig) < actuals.shape[0]:
                    v = actuals[int(orig)]
                    raw_a = float(v) if np.isfinite(v) else None
                raw_p = float(point_z[t]) if np.isfinite(point_z[t]) else None
                if resolved == ZMode.RESIDUAL_ERROR:
                    wz_a: float | None = None
                    residual = raw_p
                else:
                    wz_a = (
                        float(WorldCoordinateScaler.to_world_coords(np.asarray([raw_a], dtype=np.float32), z_scale)[0])
                        if raw_a is not None else None
                    )
                    if resolved == ZMode.PREDICTION and raw_a is not None and raw_p is not None:
                        residual = raw_a - float(q_pred[t])
                    else:
                        residual = None
                points_world.append([float(wx[t]), float(wy[t]), float(wz[t])])
                trajectory.append({
                    "frame_index": t,
                    "original_row_index": int(orig),
                    "world_x": float(wx[t]),
                    "world_y": float(wy[t]),
                    "world_z_pred": float(wz[t]),
                    "world_z_actual": wz_a,
                    "raw_x": float(qx_raw[t]) if np.isfinite(qx_raw[t]) else 0.0,
                    "raw_y": float(qy_raw[t]) if np.isfinite(qy_raw[t]) else 0.0,
                    "raw_z_pred": raw_p if raw_p is not None else 0.0,
                    "raw_z_actual": raw_a,
                    "residual_error": residual,
                })

        finite_grid = grid_z[np.isfinite(grid_z)]
        z_lo = float(np.min(finite_grid)) if finite_grid.size else 0.0
        z_hi = float(np.max(finite_grid)) if finite_grid.size else 1.0
        return SurfaceMesh3DEntity(
            grid_resolution=res,
            x_feature=feature_names[x_idx],
            y_feature=feature_names[y_idx],
            x_range=(float(x_range[0]), float(x_range[1])),
            y_range=(float(y_range[0]), float(y_range[1])),
            z_range=(z_lo, z_hi),
            z_heights_world=[float(v) for v in gz_w.tolist()],
            raw_predictions=[float(v) for v in grid_z.tolist()],
            point_positions_world=points_world,
            point_row_indices=[int(r) for r in rows.tolist()],
            trajectory_points=trajectory,
            resolved_z_mode=resolved.value,
            has_constant_axis_warning=constant_warning,
            dataset_row_count=int(X.shape[0]),
        )

    # -- PCA manifold ------------------------------------------------------
    @staticmethod
    def build_pca_surface(
        *,
        X: np.ndarray,
        feature_names: list[str],
        cfg: ProjectionConfigVO,
        adapter: Any,
        model_obj: Any,
        final_est: Any,
        actuals: np.ndarray | None,
        queue_rows: list[int],
    ) -> SurfaceMesh3DEntity:
        res = cfg.grid_resolution
        coords_3d, variance, components, scaler, pca = compute_3d_pca_projection(X, return_fitted=True)

        n_comp = 0 if pca is None else int(getattr(pca, "n_components_", 0) or 0)
        rows = np.asarray([r for r in queue_rows if 0 <= r < X.shape[0]], dtype=np.int64)

        if scaler is None or pca is None or n_comp == 0:
            grid_pred = np.zeros((res * res,), dtype=np.float64)
            raw_pc = np.zeros((res * res, 3), dtype=np.float32)
            pc_means = np.zeros((max(n_comp, 1),), dtype=np.float32)
            x_range = (0.0, 1.0)
            y_range = (0.0, 1.0)
        else:
            # Raw PC scores in chunks (flat RAM profile), then grid over PC1 x PC2.
            raw_list: list[np.ndarray] = []
            offset = 0
            while offset < X.shape[0]:
                chunk = scaler.transform(X[offset : offset + PREDICT_CHUNK_SIZE])
                raw_list.append(np.asarray(pca.transform(chunk), dtype=np.float32))
                del chunk
                offset += PREDICT_CHUNK_SIZE
            raw_pc_all = np.vstack(raw_list) if raw_list else np.zeros((0, n_comp), dtype=np.float32)
            del raw_list
            pc_means = (
                np.mean(raw_pc_all, axis=0).astype(np.float32)
                if raw_pc_all.shape[0]
                else np.zeros((n_comp,), dtype=np.float32)
            )

            def _dim_range(d: int) -> tuple[float, float]:
                if d >= raw_pc_all.shape[1] or raw_pc_all.shape[0] == 0:
                    return -1.0, 1.0
                col = raw_pc_all[:, d]
                lo, hi = float(np.min(col)), float(np.max(col))
                if abs(hi - lo) < 1e-9 or not (np.isfinite(lo) and np.isfinite(hi)):
                    return float(pc_means[d]) - 1.0, float(pc_means[d]) + 1.0
                return lo, hi

            x_range = _dim_range(0)
            y_range = _dim_range(1)
            g0 = np.linspace(x_range[0], x_range[1], res, dtype=np.float32)
            g1 = np.linspace(y_range[0], y_range[1], res, dtype=np.float32)
            m0, m1 = np.meshgrid(g0, g1, indexing="xy")
            grid_pc = np.tile(pc_means, (res * res, 1))
            if n_comp > 0:
                grid_pc[:, 0] = m0.ravel()
            if n_comp > 1:
                grid_pc[:, 1] = m1.ravel()
            grid_features = np.asarray(scaler.inverse_transform(pca.inverse_transform(grid_pc)), dtype=np.float32)
            grid_features = np.nan_to_num(grid_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            grid_pred = SurfaceMeshService.predict_values(adapter, model_obj, grid_features)
            raw_pc = np.zeros((res * res, 3), dtype=np.float32)
            raw_pc[:, 0] = m0.ravel()
            if n_comp > 1:
                raw_pc[:, 1] = m1.ravel()
            del grid_pc, grid_features, raw_pc_all

        resolved = ZMode.PREDICTION
        grid_z = grid_pred
        if cfg.z_mode == ZMode.PROBABILITY:
            # Probability over the inverse-mapped grid needs features; rebuild cheaply.
            proba = None
            if scaler is not None and pca is not None and n_comp > 0:
                g0 = np.linspace(x_range[0], x_range[1], res, dtype=np.float32)
                g1 = np.linspace(y_range[0], y_range[1], res, dtype=np.float32)
                m0, m1 = np.meshgrid(g0, g1, indexing="xy")
                grid_pc = np.tile(pc_means, (res * res, 1))
                if n_comp > 0:
                    grid_pc[:, 0] = m0.ravel()
                if n_comp > 1:
                    grid_pc[:, 1] = m1.ravel()
                grid_features = np.asarray(
                    scaler.inverse_transform(pca.inverse_transform(grid_pc)), dtype=np.float32
                )
                proba = SurfaceMeshService.predict_probabilities(
                    model_obj, final_est, grid_features, cfg.target_class_index
                )
                del grid_pc, grid_features
            if proba is not None:
                grid_z = proba
                resolved = ZMode.PROBABILITY

        Q = X[rows] if rows.size else np.empty((0, X.shape[1]), dtype=np.float32)
        q_pred = SurfaceMeshService.predict_values(adapter, model_obj, Q) if rows.size else np.empty((0,))
        if resolved == ZMode.PROBABILITY and rows.size:
            q_proba = SurfaceMeshService.predict_probabilities(model_obj, final_est, Q, cfg.target_class_index)
            point_z_raw = q_proba if q_proba is not None else q_pred
        elif cfg.z_mode == ZMode.RESIDUAL_ERROR and rows.size and actuals is not None:
            point_z_raw = np.abs(q_pred - actuals[rows])
            resolved = ZMode.RESIDUAL_ERROR
        else:
            point_z_raw = q_pred

        pc_names = [
            f"PC1 ({variance[0] * 100:.1f}%)",
            f"PC2 ({variance[1] * 100:.1f}%)",
            f"PC3 ({variance[2] * 100:.1f}%)",
        ]
        z_pool = [grid_z[np.isfinite(grid_z)]]
        if point_z_raw.size and np.isfinite(point_z_raw).any():
            z_pool.append(point_z_raw[np.isfinite(point_z_raw)])
        z_values = (
            np.concatenate([a.astype(np.float32) for a in z_pool if a.size])
            if any(a.size for a in z_pool)
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        if resolved == ZMode.PROBABILITY:
            z_name = f"P(class #{cfg.target_class_index})"
        elif resolved == ZMode.RESIDUAL_ERROR:
            z_name = "Residual |y - ŷ|"
        else:
            z_name = "Predicted value"
        pz = WorldCoordinateScaler.build_axis_scale(z_name, z_values, feature_idx=None)

        # Grid world positions: raw PC ranges mapped through per-dim world scales.
        gx_w = safe_min_max_normalize(raw_pc[:, 0], x_range[0], x_range[1])
        gy_w = safe_min_max_normalize(raw_pc[:, 1], y_range[0], y_range[1])
        gz_w = WorldCoordinateScaler.to_world_coords(grid_z.astype(np.float32), pz)

        points_world: list[list[float]] = []
        trajectory: list[dict[str, Any]] = []
        if rows.size:
            wx = _safe_world(coords_3d[rows][:, 0])
            wy = _safe_world(coords_3d[rows][:, 1])
            safe_pz = np.where(np.isfinite(point_z_raw), point_z_raw, (pz.raw_min + pz.raw_max) / 2.0)
            wz = _safe_world(WorldCoordinateScaler.to_world_coords(safe_pz.astype(np.float32), pz))
            for t, orig in enumerate(rows.tolist()):
                raw_a: float | None = None
                if actuals is not None and 0 <= int(orig) < actuals.shape[0]:
                    v = actuals[int(orig)]
                    raw_a = float(v) if np.isfinite(v) else None
                raw_p = float(point_z_raw[t]) if np.isfinite(point_z_raw[t]) else None
                if resolved == ZMode.RESIDUAL_ERROR:
                    wz_a, residual = None, raw_p
                else:
                    wz_a = (
                        float(WorldCoordinateScaler.to_world_coords(np.asarray([raw_a], dtype=np.float32), pz)[0])
                        if raw_a is not None else None
                    )
                    residual = (raw_a - float(q_pred[t])) if (
                        resolved == ZMode.PREDICTION and raw_a is not None and np.isfinite(q_pred[t])
                    ) else None
                points_world.append([float(wx[t]), float(wy[t]), float(wz[t])])
                trajectory.append({
                    "frame_index": t,
                    "original_row_index": int(orig),
                    "world_x": float(wx[t]),
                    "world_y": float(wy[t]),
                    "world_z_pred": float(wz[t]),
                    "world_z_actual": wz_a,
                    "raw_x": float(coords_3d[int(orig)][0]),
                    "raw_y": float(coords_3d[int(orig)][1]),
                    "raw_z_pred": raw_p if raw_p is not None else 0.0,
                    "raw_z_actual": raw_a,
                    "residual_error": residual,
                })

        loadings: list[list[float]] | None = None
        if components is not None and components.size:
            loadings = [[float(v) for v in row] for row in np.asarray(components, dtype=np.float32).tolist()]
            while len(loadings) < 3:
                loadings.append([0.0] * len(feature_names))

        finite_grid = grid_z[np.isfinite(grid_z)]
        return SurfaceMesh3DEntity(
            grid_resolution=res,
            x_feature=pc_names[0],
            y_feature=pc_names[1],
            x_range=(float(x_range[0]), float(x_range[1])),
            y_range=(float(y_range[0]), float(y_range[1])),
            z_range=(
                float(np.min(finite_grid)) if finite_grid.size else 0.0,
                float(np.max(finite_grid)) if finite_grid.size else 1.0,
            ),
            z_heights_world=[float(v) for v in gz_w.tolist()],
            raw_predictions=[float(v) for v in grid_z.tolist()],
            point_positions_world=points_world,
            point_row_indices=[int(r) for r in rows.tolist()],
            trajectory_points=trajectory,
            pca_variance_ratio=[float(v) for v in variance],
            pca_loadings=loadings,
            resolved_z_mode=resolved.value,
            has_constant_axis_warning=False,
            dataset_row_count=int(X.shape[0]),
        )
