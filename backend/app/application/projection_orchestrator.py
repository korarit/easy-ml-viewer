"""Phase 09 — projection application orchestrator (surface-mesh use-case).

Pipeline (dev spec §3, UX flow):
1. Validate the mapping against the dataset/model pair.
2. Resolve ``x_feature`` / ``y_feature`` names to matrix columns.
3. Pre-flight RAM guard for the ``res*res x D`` grid (§5.1).
4. LRU cache lookup (hit answers in <1 ms).
5. Heavy math (grid synthesis, chunked batch predict, PCA fit) runs in
   ``asyncio.to_thread`` so the FastAPI event loop never blocks (§5.3).

Memory: one float32 feature matrix + one grid batch (≤2,500 x D) alive at
a time; everything else is flat float32 lists in the returned entity.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.spatial_projection.entities import SurfaceMesh3DEntity
from app.domains.spatial_projection.repository import SurfaceMeshCacheRepository
from app.domains.spatial_projection.services import SurfaceMeshService
from app.domains.spatial_projection.value_objects import (
    LodLevel,
    PinningMode,
    ProjectionMode,
    ZMode,
)
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

MIN_FREE_RAM_MB = 256.0
MAX_QUEUE_POINTS = 10_000


class ProjectionSurfaceRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    mode: ProjectionMode = ProjectionMode.AXIS_EXPLORER
    x_feature: str = ""
    y_feature: str = ""
    z_mode: ZMode = ZMode.PREDICTION
    target_class_index: int = Field(default=0, ge=0)
    pinning_mode: PinningMode = PinningMode.MEDIAN
    lod_level: LodLevel = LodLevel.STANDARD
    current_frame_index: int | None = None
    use_robust_range: bool = False
    selected_row_indices: list[int] = Field(default_factory=list)


def _resolve_feature_index(items: list[Any], name: str) -> int | None:
    for item in items:
        if item.dataset_column_name == name or item.model_feature_name == name:
            return item.feature_index
    lowered = name.strip().casefold()
    for item in items:
        for candidate in (item.dataset_column_name, item.model_feature_name):
            if candidate and candidate.strip().casefold() == lowered:
                return item.feature_index
    return None


class ProjectionOrchestrator:
    def __init__(
        self,
        dataset_repo: ParquetDatasetRepository | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
        cache_repo: SurfaceMeshCacheRepository | None = None,
    ) -> None:
        self._datasets = dataset_repo or ParquetDatasetRepository()
        self._models = model_use_case or ModelIngestUseCase()
        self._mappings = mapping_repo or InMemoryMappingRepository()
        self._cache = cache_repo or SurfaceMeshCacheRepository()

    async def build_surface(self, req: ProjectionSurfaceRequest) -> SurfaceMesh3DEntity:
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={"mapping_id": req.mapping_id},
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        n_features = len(items)
        if n_features == 0:
            raise DomainException(
                code="PROJECTION_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )

        x_idx = y_idx = -1
        if req.mode == ProjectionMode.AXIS_EXPLORER:
            if not req.x_feature or not req.y_feature:
                raise DomainException(
                    code="PROJECTION_AXES_REQUIRED",
                    message="Axis explorer mode requires x_feature and y_feature",
                    http_status=422,
                )
            resolved_x = _resolve_feature_index(items, req.x_feature)
            resolved_y = _resolve_feature_index(items, req.y_feature)
            if resolved_x is None or resolved_y is None:
                raise DomainException(
                    code="PROJECTION_UNKNOWN_FEATURE",
                    message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
                    details={"x_feature": req.x_feature, "y_feature": req.y_feature},
                    http_status=422,
                )
            if resolved_x == resolved_y:
                if n_features < 2:
                    resolved_y = resolved_x  # D==1: guarded 1-D curve slice
                else:
                    raise DomainException(
                        code="PROJECTION_DUPLICATE_AXES",
                        message="X and Y axes must be different features (use swapAxes on the client)",
                        http_status=422,
                    )
            x_idx, y_idx = resolved_x, resolved_y
            unmapped = [i for i in (x_idx, y_idx) if not items[i].dataset_column_name]
            if unmapped:
                raise DomainException(
                    code="PROJECTION_AXIS_UNMAPPED",
                    message=f"Feature #{unmapped[0]} is not mapped to a dataset column",
                    http_status=422,
                )

        selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
        if len(selected) > MAX_QUEUE_POINTS:
            raise DomainException(
                code="PROJECTION_TOO_MANY_POINTS",
                message=f"Select at most {MAX_QUEUE_POINTS} rows per scene",
                http_status=422,
            )

        res = req.lod_level.grid_resolution
        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        all_columns = set(pf.schema.names)
        for col in feature_cols:
            if col not in all_columns:
                raise DomainException(
                    code="PROJECTION_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )
        for idx in selected:
            if idx >= total_rows:
                raise DomainException(
                    code="PROJECTION_ROW_OUT_OF_RANGE",
                    message=f"Row index {idx} exceeds dataset size ({total_rows} rows)",
                    http_status=422,
                )

        # -- pre-flight RAM guard (§5.1) ----------------------------------
        grid_bytes = float(res * res * n_features * 4)
        available_mb = float(psutil.virtual_memory().available) / (1024 * 1024)
        if grid_bytes / (1024 * 1024) > max(0.0, available_mb - MIN_FREE_RAM_MB):
            raise DomainException(
                code="PROJECTION_WIDE_GRID_TOO_LARGE",
                message=(
                    f"Grid {res}x{res}x{n_features} needs "
                    f"{grid_bytes / (1024 * 1024):.1f} MB but only "
                    f"{available_mb:.0f} MB is free (256 MB floor kept)"
                ),
                http_status=413,
            )

        cache_key = (
            req.dataset_id, req.model_id, req.mapping_id, req.mode.value,
            req.x_feature, req.y_feature, req.z_mode.value, int(req.target_class_index),
            req.pinning_mode.value, req.lod_level.value,
            int(req.current_frame_index) if req.current_frame_index is not None else -1,
            bool(req.use_robust_range), tuple(selected),
        )
        started = time.perf_counter()
        cached = self._cache.get(cache_key)
        if cached is not None:
            hit = cached.model_copy(update={"computation_time_ms": round((time.perf_counter() - started) * 1000.0, 2)})
            return hit

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df
            # NOTE: no explicit gc.collect() on the hot path — a full collection
            # costs ~50 ms on a large heap (profiled) and NumPy/pandas temporaries
            # are freed by refcounting as soon as `del` runs. The generational
            # collector reclaims any reference cycles automatically.

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

        current_row_values: np.ndarray | None = None
        if req.pinning_mode == PinningMode.CURRENT_ROW and selected:
            frame = req.current_frame_index if req.current_frame_index is not None else 0
            frame = max(0, min(int(frame), len(selected) - 1))
            current_row_values = X[selected[frame]].astype(np.float32, copy=True)

        feature_names = [
            i.dataset_column_name or i.model_feature_name for i in items
        ]
        if req.mode == ProjectionMode.PCA_3D:
            entity = await asyncio.to_thread(
                SurfaceMeshService.build_pca_surface,
                X=X,
                feature_names=feature_names,
                cfg=_vo_from(req),
                adapter=adapter,
                model_obj=model_obj,
                final_est=final_est,
                actuals=actuals,
                queue_rows=selected,
            )
        else:
            entity = await asyncio.to_thread(
                SurfaceMeshService.build_axis_surface,
                X=X,
                feature_names=feature_names,
                x_idx=x_idx,
                y_idx=y_idx,
                cfg=_vo_from(req),
                adapter=adapter,
                model_obj=model_obj,
                final_est=final_est,
                actuals=actuals,
                queue_rows=selected,
                current_row_values=current_row_values,
            )

        del X
        entity.computation_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self._cache.put(cache_key, entity)
        return entity


def _vo_from(req: ProjectionSurfaceRequest):  # -> ProjectionConfigVO (lazy import breaks cycle)
    from app.domains.spatial_projection.value_objects import ProjectionConfigVO

    return ProjectionConfigVO(
        mode=req.mode,
        x_feature=req.x_feature or "pc1",
        y_feature=req.y_feature or "pc2",
        z_mode=req.z_mode,
        target_class_index=int(req.target_class_index),
        pinning_mode=req.pinning_mode,
        lod_level=req.lod_level,
        current_frame_index=req.current_frame_index,
        use_robust_range=bool(req.use_robust_range),
    )
