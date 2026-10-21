"""Phase 08 — 3D stage application use-case (initialize-scene).

Pipeline (dev spec §3.1):
1. Validate mapping belongs to the dataset/model pair.
2. Resolve X/Y axis features + load the cached parquet (mmap, chunked).
3. Batch-predict (``CHUNK_SIZE = 2_500`` rows) for the Z axis domain.
4. Build an ``R x R`` axis-explorer slice grid (other features = column
   means) and batch-predict its heights — the tilted hyperplane for the
   Phase 04 linear slice, generic for any future adapter.
5. Normalize every axis into the ``[-5, +5]`` world cube via
   :class:`WorldCoordinateScaler` (float32, clipped to ``[-7, +7]``).

Memory: peak extra RAM is one 2,500-row float32 chunk + one grid batch
(``<= 2_500 x d``); the surface vertex buffer (``R*R*3`` float32,
``<= 80*80``) is flattened once. Nothing dense is retained.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.core.math_3d import AxisScaleMeta, WorldCoordinateScaler
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

PREDICT_CHUNK_SIZE = 2_500
MIN_GRID_RESOLUTION = 10
MAX_GRID_RESOLUTION = 80
DEFAULT_GRID_RESOLUTION = 50
MAX_TRAJECTORY_POINTS = 10_000


class InitializeSceneRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    axis_x_feature_index: int = Field(ge=0)
    axis_y_feature_index: int = Field(ge=0)
    selected_row_indices: list[int] = Field(default_factory=list)
    grid_resolution: int = Field(default=DEFAULT_GRID_RESOLUTION)


class Stage3DSurface(BaseModel):
    visual_type: str
    resolution: int
    world_vertices_xyz: list[float]


class TrajectoryPoint3D(BaseModel):
    frame_index: int
    original_row_index: int
    world_x: float
    world_y: float
    world_z_pred: float
    world_z_actual: float | None = None
    raw_x: float
    raw_y: float
    raw_z_pred: float
    raw_z_actual: float | None = None
    residual_error: float | None = None


class Stage3DAxes(BaseModel):
    x: AxisScaleMeta
    y: AxisScaleMeta
    z: AxisScaleMeta


class InitializeSceneResponse(BaseModel):
    axes: Stage3DAxes
    surface: Stage3DSurface
    trajectory_points: list[TrajectoryPoint3D]
    dataset_row_count: int
    task_type: str


def _clean_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


class Stage3DUseCase:
    def __init__(
        self,
        dataset_repo: ParquetDatasetRepository | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
    ) -> None:
        self._datasets = dataset_repo or ParquetDatasetRepository()
        self._models = model_use_case or ModelIngestUseCase()
        self._mappings = mapping_repo or InMemoryMappingRepository()

    # -- public API ----------------------------------------------------
    def initialize_scene(self, req: InitializeSceneRequest) -> InitializeSceneResponse:
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={
                    "mapping_id": req.mapping_id,
                    "mapping_dataset_id": mapping.dataset_id,
                    "mapping_model_id": mapping.model_id,
                },
                http_status=422,
            )

        items = sorted(mapping.items, key=lambda x: x.feature_index)
        n_features = len(items)
        if n_features == 0:
            raise DomainException(
                code="STAGE3D_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        if req.axis_x_feature_index >= n_features or req.axis_y_feature_index >= n_features:
            raise DomainException(
                code="STAGE3D_AXIS_OUT_OF_RANGE",
                message=(
                    f"Axis feature index out of range (model has {n_features} features; "
                    f"got x={req.axis_x_feature_index}, y={req.axis_y_feature_index})"
                ),
                http_status=422,
            )
        if req.axis_x_feature_index == req.axis_y_feature_index:
            raise DomainException(
                code="STAGE3D_AXIS_DUPLICATE",
                message="X and Y axes must be different features",
                http_status=422,
            )
        for axis_idx in (req.axis_x_feature_index, req.axis_y_feature_index):
            item = items[axis_idx]
            if not item.dataset_column_name:
                raise DomainException(
                    code="STAGE3D_AXIS_UNMAPPED",
                    message=f"Feature #{axis_idx} ({item.model_feature_name}) is not mapped",
                    http_status=422,
                )

        resolution = max(MIN_GRID_RESOLUTION, min(int(req.grid_resolution), MAX_GRID_RESOLUTION))
        selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
        if len(selected) > MAX_TRAJECTORY_POINTS:
            raise DomainException(
                code="STAGE3D_TOO_MANY_POINTS",
                message=f"Select at most {MAX_TRAJECTORY_POINTS} rows per scene",
                http_status=422,
            )

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        all_columns = set(pf.schema.names)
        for col in feature_cols:
            if col not in all_columns:
                raise DomainException(
                    code="STAGE3D_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )
        for idx in selected:
            if idx >= total_rows:
                raise DomainException(
                    code="STAGE3D_ROW_OUT_OF_RANGE",
                    message=f"Row index {idx} exceeds dataset size ({total_rows} rows)",
                    http_status=422,
                )

        # -- load feature frame (mmap; single float32 matrix, freed after) --
        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df
            gc.collect()

        x_raw = X[:, req.axis_x_feature_index].astype(np.float32, copy=True)
        y_raw = X[:, req.axis_y_feature_index].astype(np.float32, copy=True)
        with np.errstate(invalid="ignore"):
            col_means = np.nanmean(X.astype(np.float64, copy=False), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)

        # -- chunked batch predictions (bounded RAM, §5.1) ----------------
        preds = np.empty((total_rows,), dtype=np.float64)
        preds[:] = np.nan
        offset = 0
        while offset < total_rows:
            chunk = np.ascontiguousarray(X[offset : offset + PREDICT_CHUNK_SIZE], dtype=np.float32)
            result = adapter.predict_batch(model_obj, chunk)
            vals = np.asarray(list(result.predictions), dtype=np.float64)
            preds[offset : offset + vals.shape[0]] = vals
            del chunk, result, vals
            gc.collect()
            offset += PREDICT_CHUNK_SIZE

        # -- ground-truth column (optional) -------------------------------
        actuals: np.ndarray | None = None
        if mapping.target_column_name and mapping.target_column_name in all_columns:
            tdf = self._datasets.load_dataframe(req.dataset_id, columns=[mapping.target_column_name])
            try:
                numeric = pd.to_numeric(tdf[mapping.target_column_name], errors="coerce").to_numpy(
                    dtype=np.float64, copy=True
                )
                if np.isfinite(numeric).any():
                    actuals = numeric
            finally:
                del tdf
                gc.collect()

        # -- axis-explorer slice grid (X x Y, others = means) -------------
        x_min, x_max = float(np.nanmin(x_raw)), float(np.nanmax(x_raw))
        y_min, y_max = float(np.nanmin(y_raw)), float(np.nanmax(y_raw))
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            x_min, x_max = 0.0, 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if abs(x_max - x_min) < 1e-7:
            x_min -= 1.0
            x_max += 1.0
        if abs(y_max - y_min) < 1e-7:
            y_min -= 1.0
            y_max += 1.0
        gx = np.linspace(x_min, x_max, resolution, dtype=np.float32)
        gy = np.linspace(y_min, y_max, resolution, dtype=np.float32)
        grid_n = resolution * resolution
        grid = np.tile(col_means.astype(np.float32, copy=False), (grid_n, 1))
        mesh_y, mesh_x = np.meshgrid(gy, gx)
        grid[:, req.axis_x_feature_index] = mesh_x.ravel().astype(np.float32)
        grid[:, req.axis_y_feature_index] = mesh_y.ravel().astype(np.float32)
        grid = np.ascontiguousarray(grid, dtype=np.float32)
        grid_pred = np.empty((grid_n,), dtype=np.float64)
        grid_pred[:] = np.nan
        goff = 0
        while goff < grid_n:
            chunk = grid[goff : goff + PREDICT_CHUNK_SIZE]
            result = adapter.predict_batch(model_obj, np.ascontiguousarray(chunk, dtype=np.float32))
            vals = np.asarray(list(result.predictions), dtype=np.float64)
            grid_pred[goff : goff + vals.shape[0]] = vals
            del chunk, result, vals
            gc.collect()
            goff += PREDICT_CHUNK_SIZE
        del grid
        gc.collect()

        # -- axis scales (ticks keep raw values, §4.2) --------------------
        x_scale = WorldCoordinateScaler.build_axis_scale(
            items[req.axis_x_feature_index].dataset_column_name or items[req.axis_x_feature_index].model_feature_name,
            x_raw,
            feature_idx=req.axis_x_feature_index,
        )
        y_scale = WorldCoordinateScaler.build_axis_scale(
            items[req.axis_y_feature_index].dataset_column_name or items[req.axis_y_feature_index].model_feature_name,
            y_raw,
            feature_idx=req.axis_y_feature_index,
        )
        z_pool = [preds[np.isfinite(preds)], grid_pred[np.isfinite(grid_pred)]]
        if actuals is not None:
            finite_actual = actuals[np.isfinite(actuals)]
            if finite_actual.size:
                z_pool.append(finite_actual.astype(np.float32))
        z_values = (
            np.concatenate([a.astype(np.float32, copy=False) for a in z_pool if a.size])
            if any(a.size for a in z_pool)
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        z_name = (
            f"Predicted {mapping.target_column_name}"
            if mapping.target_column_name
            else "Predicted value"
        )
        z_scale = WorldCoordinateScaler.build_axis_scale(z_name, z_values, feature_idx=None)

        # -- normalize to world cube --------------------------------------
        gx_world_1d = WorldCoordinateScaler.to_world_coords(gx, x_scale)
        gy_world_1d = WorldCoordinateScaler.to_world_coords(gy, y_scale)
        gz_world = WorldCoordinateScaler.to_world_coords(
            np.where(np.isfinite(grid_pred), grid_pred, (z_scale.raw_min + z_scale.raw_max) / 2.0).astype(np.float32),
            z_scale,
        )
        mesh_yw, mesh_xw = np.meshgrid(gy_world_1d, gx_world_1d)
        interleaved = np.empty((grid_n * 3,), dtype=np.float32)
        interleaved[0::3] = mesh_xw.ravel()
        interleaved[1::3] = mesh_yw.ravel()
        interleaved[2::3] = gz_world.astype(np.float32)

        sel_idx = np.asarray(selected, dtype=np.int64) if selected else np.empty((0,), dtype=np.int64)
        if sel_idx.size:
            sel_x_raw = x_raw[sel_idx].astype(np.float64)
            sel_y_raw = y_raw[sel_idx].astype(np.float64)
            sel_pred = preds[sel_idx]
            wx = WorldCoordinateScaler.to_world_coords(sel_x_raw.astype(np.float32), x_scale)
            wy = WorldCoordinateScaler.to_world_coords(sel_y_raw.astype(np.float32), y_scale)
            safe_pred = np.where(
                np.isfinite(sel_pred), sel_pred, (z_scale.raw_min + z_scale.raw_max) / 2.0
            ).astype(np.float32)
            wz_pred = WorldCoordinateScaler.to_world_coords(safe_pred, z_scale)
        else:
            sel_x_raw = np.empty((0,), dtype=np.float64)
            sel_y_raw = np.empty((0,), dtype=np.float64)
            sel_pred = np.empty((0,), dtype=np.float64)
            wx = np.empty((0,), dtype=np.float32)
            wy = np.empty((0,), dtype=np.float32)
            wz_pred = np.empty((0,), dtype=np.float32)

        points: list[TrajectoryPoint3D] = []
        for t, orig in enumerate(selected):
            raw_x = _clean_float(sel_x_raw[t])
            raw_y = _clean_float(sel_y_raw[t])
            raw_p = _clean_float(sel_pred[t])
            raw_a: float | None = None
            if actuals is not None and 0 <= orig < actuals.shape[0]:
                raw_a = _clean_float(actuals[orig])
            wz_a: float | None = None
            if raw_a is not None:
                wz_a = float(
                    WorldCoordinateScaler.to_world_coords(
                        np.asarray([raw_a], dtype=np.float32), z_scale
                    )[0]
                )
            residual = (raw_a - raw_p) if (raw_a is not None and raw_p is not None) else None
            points.append(
                TrajectoryPoint3D(
                    frame_index=t,
                    original_row_index=int(orig),
                    world_x=float(wx[t]),
                    world_y=float(wy[t]),
                    world_z_pred=float(wz_pred[t]),
                    world_z_actual=wz_a,
                    raw_x=float(raw_x) if raw_x is not None else 0.0,
                    raw_y=float(raw_y) if raw_y is not None else 0.0,
                    raw_z_pred=float(raw_p) if raw_p is not None else 0.0,
                    raw_z_actual=raw_a,
                    residual_error=residual,
                )
            )

        del X, x_raw, y_raw, preds, grid_pred
        gc.collect()

        visual = getattr(adapter, "visual_3d_type", "flat_hyperplane")
        visual_type = getattr(visual, "value", str(visual))
        task = getattr(adapter.task_type, "value", str(getattr(adapter, "task_type", "regression")))
        return InitializeSceneResponse(
            axes=Stage3DAxes(x=x_scale, y=y_scale, z=z_scale),
            surface=Stage3DSurface(
                visual_type=str(visual_type),
                resolution=resolution,
                world_vertices_xyz=[float(v) for v in interleaved.tolist()],
            ),
            trajectory_points=points,
            dataset_row_count=total_rows,
            task_type=str(task),
        )
