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


class ProjectionContourRequest(BaseModel):
    """Phase 41 — dense 2D contour request (120x120 decision-boundary grid)."""

    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    x_feature: str = Field(min_length=1)
    y_feature: str = Field(min_length=1)
    pinning_mode: PinningMode = PinningMode.MEDIAN
    current_frame_index: int | None = None
    use_robust_range: bool = False
    grid_resolution: int = Field(default=120, ge=16, le=200)
    selected_row_indices: list[int] = Field(default_factory=list)


class ProjectionPointIntelligenceRequest(BaseModel):
    """Phase 49 — Universal 2D point-intelligence aggregation (hexbin + density)."""

    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    x_feature: str = Field(min_length=1)
    y_feature: str = Field(min_length=1)
    hex_radius: float = Field(default=0.45, gt=0.05, le=2.0)
    residual_eps: float = Field(default=0.5, ge=0.0)
    selected_row_indices: list[int] = Field(default_factory=list)


class ProjectionKnnFrameRequest(BaseModel):
    """Phase 43 — per-frame KNN radar state for the 2D planar viewport."""

    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    x_feature: str = Field(min_length=1)
    y_feature: str = Field(min_length=1)
    row_index: int = Field(ge=0, description="Dataset row serving as the radar query")


class ProjectionBoostFrameRequest(BaseModel):
    """Phase 46 — per-frame boosting SHAP push vector for the 2D viewport."""

    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    x_feature: str = Field(min_length=1)
    y_feature: str = Field(min_length=1)
    row_index: int = Field(ge=0, description="Dataset row serving as the SHAP query")


class ProjectionDualCompare2DRequest(BaseModel):
    """Phase 48 — synchronized dual-model 2D compare on one (X, Y) plane."""

    dataset_id: str = Field(min_length=1)
    model_a_id: str = Field(min_length=1)
    model_b_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1, description="Model-A mapping defining axes + feature columns")
    x_feature: str = Field(min_length=1)
    y_feature: str = Field(min_length=1)
    pinning_mode: PinningMode = PinningMode.MEDIAN
    current_frame_index: int | None = None
    use_robust_range: bool = False
    grid_resolution: int = Field(default=120, ge=16, le=200)
    selected_row_indices: list[int] = Field(default_factory=list)


class DualContour2DResponse(BaseModel):
    """Phase 48 — dual 2D compare payload (both contours + conflict mask)."""

    x_feature: str = ""
    y_feature: str = ""
    grid_resolution: int = 120
    prediction_task: str = "classification"
    disagreement_pct: float = 0.0
    agreement_pct: float = 100.0
    n_cells: int = 0
    n_conflict_cells: int = 0
    consensus_notice: str = ""
    conflict_mask: list[int] = Field(default_factory=list)
    conflict_mask_downsampled: bool = False
    conflict_frame_indices: list[int] = Field(default_factory=list)
    model_a_name: str = ""
    model_b_name: str = ""
    contour_a: dict[str, Any] = Field(default_factory=dict)
    contour_b: dict[str, Any] = Field(default_factory=dict)
    computation_time_ms: float = 0.0


class KnnFrameNeighbor(BaseModel):
    train_index: int = 0
    x_world: float = 0.0
    y_world: float = 0.0
    distance_full: float = 0.0
    distance_2d: float = 0.0
    vote_weight: float = 0.0
    rank: int = 0
    label: Any = None


class KnnFrame2DResponse(BaseModel):
    """Per-frame radar payload (world coords; ellipse radii for anisotropy)."""

    x_feature: str = ""
    y_feature: str = ""
    row_index: int = 0
    query_world: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    radar_radius_raw: float = 0.0
    radar_radius_world: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    neighbors: list[KnnFrameNeighbor] = Field(default_factory=list)
    votes: dict[str, float] = Field(default_factory=dict)
    predicted: Any = None
    k: int = 0
    weights_mode: str = "uniform"
    metric: str = "euclidean"
    is_radius_model: bool = False
    outlier: bool = False
    display_capped: bool = False
    total_neighbors: int = 0
    computation_time_ms: float = 0.0


class BoostFrameShap(BaseModel):
    """Native TreeSHAP push vector on the (X, Y) plane (rounded DTO floats)."""

    phi_x: float = 0.0
    phi_y: float = 0.0
    magnitude: float = 0.0
    bias: float | None = None
    method: str = ""


class BoostFrame2DResponse(BaseModel):
    """Per-frame boosting payload: query point + live TreeSHAP push vector."""

    x_feature: str = ""
    y_feature: str = ""
    row_index: int = 0
    query_world: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    family: str = ""
    algorithm: str = ""
    shap: BoostFrameShap | None = None
    computation_time_ms: float = 0.0


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
                available = [i.dataset_column_name or i.model_feature_name for i in items]
                raise DomainException(
                    code="PROJECTION_UNKNOWN_FEATURE",
                    message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
                    details={"x_feature": req.x_feature, "y_feature": req.y_feature, "available_features": available},
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

    async def build_contour(self, req: ProjectionContourRequest) -> Any:
        """Phase 41 — dense 2D decision-boundary grid (delegates to the shared resolver)."""
        return await _resolve_contour_frame(req, self)

    async def build_point_intelligence(self, req: ProjectionPointIntelligenceRequest) -> Any:
        """Phase 49 — hexbin + density aggregation over the queued trajectory points.

        World coordinates reuse the exact contour-viewport scales so hexbin
        centers line up 1:1 with the ``Stage2DCanvas`` overlay. Predictions
        come from chunked ``predict_batch`` (≤2,500/chunk); actuals from the
        mapping target column when present.
        """
        from app.domains.simulation.point_intelligence_service import (
            DEFAULT_HEX_RADIUS,
            build_point_intelligence,
        )
        from app.domains.spatial_projection.point_intelligence_entities import (
            PointIntelligenceResponse,
        )

        started = time.perf_counter()
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={"mapping_id": req.mapping_id},
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        if not items:
            raise DomainException(
                code="PROJECTION_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        resolved_x = _resolve_feature_index(items, req.x_feature)
        resolved_y = _resolve_feature_index(items, req.y_feature)
        if resolved_x is None or resolved_y is None:
            raise DomainException(
                code="PROJECTION_UNKNOWN_FEATURE",
                message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
                http_status=422,
            )
        if resolved_x == resolved_y and len(items) >= 2:
            raise DomainException(
                code="PROJECTION_DUPLICATE_AXES",
                message="X and Y axes must be different features (use swapAxes on the client)",
                http_status=422,
            )
        selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
        if len(selected) > MAX_QUEUE_POINTS:
            raise DomainException(
                code="PROJECTION_TOO_MANY_POINTS",
                message=f"Select at most {MAX_QUEUE_POINTS} rows per scene",
                http_status=422,
            )

        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        all_columns = set(pf.schema.names)
        for idx in selected:
            if idx >= total_rows:
                raise DomainException(
                    code="PROJECTION_ROW_OUT_OF_RANGE",
                    message=f"Row index {idx} exceeds dataset size ({total_rows} rows)",
                    http_status=422,
                )

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df

        from app.core.math_3d import WorldCoordinateScaler
        from app.domains.simulation.contour_2d_generator import PREDICT_CHUNK_SIZE

        feature_names = [i.dataset_column_name or i.model_feature_name for i in items]
        x_scale = WorldCoordinateScaler.build_axis_scale(feature_names[resolved_x], X[:, resolved_x])
        y_scale = WorldCoordinateScaler.build_axis_scale(feature_names[resolved_y], X[:, resolved_y])

        rows = np.asarray([r for r in selected if 0 <= r < X.shape[0]], dtype=np.int64)
        if rows.size:
            qx = WorldCoordinateScaler.to_world_coords(X[rows][:, resolved_x], x_scale)
            qy = WorldCoordinateScaler.to_world_coords(X[rows][:, resolved_y], y_scale)
            world_xy = np.ascontiguousarray(
                np.stack([qx, qy], axis=1), dtype=np.float32
            )
            mat = np.ascontiguousarray(X[rows], dtype=np.float32)
            preds: list[float] = []
            off = 0
            while off < mat.shape[0]:
                chunk = np.ascontiguousarray(mat[off:off + PREDICT_CHUNK_SIZE], dtype=np.float32)
                r = adapter.predict_batch(model_obj, chunk)
                preds.extend(float(v) for v in list(r.predictions))
                del chunk, r
                off += PREDICT_CHUNK_SIZE
            y_pred = np.asarray(preds, dtype=np.float64)
            del mat
        else:
            world_xy = np.empty((0, 2), dtype=np.float32)
            y_pred = np.empty((0,), dtype=np.float64)
        del X

        y_true: np.ndarray | None = None
        if mapping.target_column_name and mapping.target_column_name in all_columns:
            tdf = self._datasets.load_dataframe(req.dataset_id, columns=[mapping.target_column_name])
            try:
                numeric = pd.to_numeric(tdf[mapping.target_column_name], errors="coerce").to_numpy(
                    dtype=np.float64, copy=True
                )
                if rows.size and np.isfinite(numeric).any():
                    y_true = np.asarray([float(numeric[r]) if np.isfinite(numeric[r]) else np.nan for r in rows.tolist()])
            finally:
                del tdf

        residuals: np.ndarray | None = None
        if y_true is not None and y_pred.size == y_true.size and y_true.size:
            with np.errstate(invalid="ignore"):
                residuals = y_true - y_pred

        entity: PointIntelligenceResponse = await asyncio.to_thread(
            build_point_intelligence,
            world_xy,
            y_true,
            y_pred if y_pred.size else None,
            residuals,
            float(req.hex_radius or DEFAULT_HEX_RADIUS),
            float(req.residual_eps),
        )
        entity.computation_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return entity

    async def build_knn_frame(self, req: ProjectionKnnFrameRequest) -> KnnFrame2DResponse:
        """Phase 43 — per-frame KNN radar state (one ``kneighbors`` call, < 5 ms)."""
        from app.core.math_3d import WorldCoordinateScaler
        from app.domains.model_engine.adapters.distance.distance_2d_extractor import (
            Distance2DExtractor,
        )

        started = time.perf_counter()
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={"mapping_id": req.mapping_id},
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        if not items:
            raise DomainException(
                code="PROJECTION_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        resolved_x = _resolve_feature_index(items, req.x_feature)
        resolved_y = _resolve_feature_index(items, req.y_feature)
        if resolved_x is None or resolved_y is None:
            raise DomainException(
                code="PROJECTION_UNKNOWN_FEATURE",
                message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
                http_status=422,
            )

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        if not hasattr(final_est if final_est is not None else model_obj, "kneighbors"):
            holder = final_est if final_est is not None else model_obj
            if not hasattr(holder, "kneighbors") and not hasattr(model_obj, "kneighbors"):
                raise DomainException(
                    code="NOT_KNN_MODEL",
                    message="knn-frame-2d requires a fitted nearest-neighbors model",
                    http_status=422,
                )

        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        if req.row_index >= total_rows:
            raise DomainException(
                code="PROJECTION_ROW_OUT_OF_RANGE",
                message=f"Row index {req.row_index} exceeds dataset size ({total_rows} rows)",
                http_status=422,
            )

        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df
        query = X[req.row_index].astype(np.float64, copy=True)
        feature_names = [i.dataset_column_name or i.model_feature_name for i in items]

        state = await asyncio.to_thread(
            Distance2DExtractor.extract_knn_2d_state,
            model_obj,
            query,
            resolved_x,
            resolved_y,
        )
        del X
        if state is None:
            raise DomainException(
                code="KNN_FRAME_UNAVAILABLE",
                message="KNN radar state unavailable (pipelined or unsupported neighbor model)",
                http_status=422,
            )

        # Scales must match the contour viewport exactly, so rebuild them
        # from the dataset column (same source as the contour path).
        df2 = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df2 = df2.reindex(columns=feature_cols)
            X2 = TypeCastingEngine.build_float32_matrix(df2, items)
        finally:
            del df2
        x_scale = WorldCoordinateScaler.build_axis_scale(feature_names[resolved_x], X2[:, resolved_x])
        y_scale = WorldCoordinateScaler.build_axis_scale(feature_names[resolved_y], X2[:, resolved_y])
        del X2

        def _w(v: float, scale: Any) -> float:
            arr = WorldCoordinateScaler.to_world_coords(np.asarray([v], dtype=np.float32), scale)
            return round(float(arr[0]), 3)

        qx, qy = state["query_2d"]
        x_span = max(float(x_scale.raw_max - x_scale.raw_min), 1e-12)
        y_span = max(float(y_scale.raw_max - y_scale.raw_min), 1e-12)
        r = float(state["radar_radius_2d"])
        neighbors = [
            KnnFrameNeighbor(
                train_index=int(n["ref_index"]),
                x_world=_w(float(n["x"]), x_scale),
                y_world=_w(float(n["y"]), y_scale),
                distance_full=float(n["distance_full"]),
                distance_2d=float(n["distance_2d"]),
                vote_weight=float(n["vote_weight"]),
                rank=int(n["rank"]),
                label=n["label"],
            )
            for n in state["neighbors_2d"]
        ]
        return KnnFrame2DResponse(
            x_feature=feature_names[resolved_x],
            y_feature=feature_names[resolved_y],
            row_index=int(req.row_index),
            query_world=[_w(float(qx), x_scale), _w(float(qy), y_scale)],
            radar_radius_raw=r,
            radar_radius_world=[round(r / x_span * 10.0, 4), round(r / y_span * 10.0, 4)],
            neighbors=neighbors,
            votes={str(k): float(v) for k, v in state["votes"].items()},
            predicted=state["predicted"],
            k=int(state["k"]),
            weights_mode=str(state["weights_mode"]),
            metric=str(state["metric"]),
            is_radius_model=bool(state["is_radius_model"]),
            outlier=bool(state["outlier"]),
            display_capped=bool(state["display_capped"]),
            total_neighbors=int(state["total_neighbors"]),
            computation_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    async def build_boost_frame(self, req: ProjectionBoostFrameRequest) -> BoostFrame2DResponse:
        """Phase 46 — per-frame boosting SHAP state (one native row call, < 5 ms)."""
        from app.core.math_3d import WorldCoordinateScaler
        from app.domains.model_engine.adapters.boosting.boosting_2d_extractor import (
            Boosting2DExtractor,
        )

        started = time.perf_counter()
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        if not items:
            raise DomainException(
                code="PROJECTION_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        resolved_x = _resolve_feature_index(items, req.x_feature)
        resolved_y = _resolve_feature_index(items, req.y_feature)
        if resolved_x is None or resolved_y is None:
            raise DomainException(
                code="PROJECTION_UNKNOWN_FEATURE",
                message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
                http_status=422,
            )

        model_obj = self._models.get_model(req.model_id)

        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        if req.row_index >= total_rows:
            raise DomainException(
                code="PROJECTION_ROW_OUT_OF_RANGE",
                message=f"Row index {req.row_index} exceeds dataset size ({total_rows} rows)",
                http_status=422,
            )

        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df
        state = await asyncio.to_thread(
            Boosting2DExtractor.extract_frame_shap,
            model_obj=model_obj,
            X=X,
            row_index=int(req.row_index),
            x_feature_idx=int(resolved_x),
            y_feature_idx=int(resolved_y),
        )
        del X
        if state is None:
            raise DomainException(
                code="BOOST_FRAME_UNAVAILABLE",
                message="boost-frame-2d requires a fitted gradient-boosting model",
                http_status=422,
            )

        # Scales must match the contour viewport exactly, so rebuild them
        # from the dataset column (same source as the contour path).
        df2 = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df2 = df2.reindex(columns=feature_cols)
            X2 = TypeCastingEngine.build_float32_matrix(df2, items)
        finally:
            del df2
        feature_names = [i.dataset_column_name or i.model_feature_name for i in items]
        x_scale = WorldCoordinateScaler.build_axis_scale(feature_names[resolved_x], X2[:, resolved_x])
        y_scale = WorldCoordinateScaler.build_axis_scale(feature_names[resolved_y], X2[:, resolved_y])
        del X2

        def _w(v: float, scale: Any) -> float:
            arr = WorldCoordinateScaler.to_world_coords(np.asarray([v], dtype=np.float32), scale)
            return round(float(arr[0]), 3)

        qx, qy = state["query_raw"]
        return BoostFrame2DResponse(
            x_feature=feature_names[resolved_x],
            y_feature=feature_names[resolved_y],
            row_index=int(req.row_index),
            query_world=[_w(float(qx), x_scale), _w(float(qy), y_scale)],
            family=str(state["family"]),
            algorithm=str(state["algorithm"]),
            shap=BoostFrameShap(**state["shap_vector"]) if state.get("shap_vector") else None,
            computation_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )


    async def build_dual_contour(self, req: ProjectionDualCompare2DRequest) -> DualContour2DResponse:
        """Phase 48 — sample both models on the same (X, Y) grid, then diff.

        Both contours share one grid (same ranges, pinning, resolution) so the
        conflict mask is scale-exact. Frame pins come from chunked
        ``predict_batch`` on the queued rows for both adapters (≤2,500/chunk).
        """
        from app.domains.simulation.contour_2d_generator import Contour2DGenerator
        from app.domains.simulation.dual_2d_compare_service import (
            compare_label_fields,
            frame_disagreement_indices,
            summary_to_response,
        )

        started = time.perf_counter()
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_a_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="mapping_id must belong to dataset + model_a (axes owner)",
                details={"mapping_id": req.mapping_id},
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        if not items:
            raise DomainException(
                code="PROJECTION_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        resolved_x = _resolve_feature_index(items, req.x_feature)
        resolved_y = _resolve_feature_index(items, req.y_feature)
        if resolved_x is None or resolved_y is None:
            raise DomainException(
                code="PROJECTION_UNKNOWN_FEATURE",
                message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
                http_status=422,
            )
        if resolved_x == resolved_y and len(items) >= 2:
            raise DomainException(
                code="PROJECTION_DUPLICATE_AXES",
                message="X and Y axes must be different features (use swapAxes on the client)",
                http_status=422,
            )
        res = max(16, min(200, int(req.grid_resolution)))

        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df

        model_a = self._models.get_model(req.model_a_id)
        model_b = self._models.get_model(req.model_b_id)
        _, final_a, _ = UniversalSchemaDetector.unwrap_model_container(model_a)
        _, final_b, _ = UniversalSchemaDetector.unwrap_model_container(model_b)
        adapter_a = ModelRegistry.resolve_adapter(final_a, ".joblib")
        try:
            adapter_b = ModelRegistry.resolve_adapter(final_b, ".joblib")
        except Exception as exc:
            raise DomainException(
                code="DUAL2D_MODEL_B_UNSUPPORTED",
                message=f"Model B is not supported for 2D compare: {exc}",
                http_status=422,
            ) from exc
        # E48.5 — both models must accept the shared plane width.
        for tag, holder in (("A", final_a if final_a is not None else model_a),
                             ("B", final_b if final_b is not None else model_b)):
            n_in = getattr(holder, "n_features_in_", None)
            if n_in is not None and int(n_in) != X.shape[1]:
                raise DomainException(
                    code="DUAL2D_FEATURE_MISMATCH",
                    message=(
                        f"Model {tag} expects {int(n_in)} features but the shared "
                        f"plane has {X.shape[1]} — align X/Y axes and mappings first"
                    ),
                    http_status=422,
                )

        current_row_values: np.ndarray | None = None
        selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
        if req.pinning_mode == PinningMode.CURRENT_ROW and selected:
            frame = req.current_frame_index if req.current_frame_index is not None else 0
            frame = max(0, min(int(frame), len(selected) - 1))
            if 0 <= selected[frame] < X.shape[0]:
                current_row_values = X[selected[frame]].astype(np.float32, copy=True)

        feature_names = [i.dataset_column_name or i.model_feature_name for i in items]
        contour_a = await asyncio.to_thread(
            Contour2DGenerator.generate_contour,
            adapter=adapter_a,
            model_obj=model_a,
            final_est=final_a,
            X=X,
            x_feature_idx=resolved_x,
            y_feature_idx=resolved_y,
            x_feature_name=feature_names[resolved_x],
            y_feature_name=feature_names[resolved_y],
            grid_resolution=res,
            pinning_mode=req.pinning_mode.value,
            current_row_values=current_row_values,
            use_robust_range=bool(req.use_robust_range),
        )
        contour_b = await asyncio.to_thread(
            Contour2DGenerator.generate_contour,
            adapter=adapter_b,
            model_obj=model_b,
            final_est=final_b,
            X=X,
            x_feature_idx=resolved_x,
            y_feature_idx=resolved_y,
            x_feature_name=feature_names[resolved_x],
            y_feature_name=feature_names[resolved_y],
            grid_resolution=res,
            pinning_mode=req.pinning_mode.value,
            current_row_values=current_row_values,
            use_robust_range=bool(req.use_robust_range),
        )

        def _decode_heatmap(c: Any) -> np.ndarray:
            q = np.asarray(c.heatmap_quantized, dtype=np.float64).ravel()
            span = float(c.heatmap_max - c.heatmap_min)
            if q.size == 0 or span < 1e-12:
                return np.zeros((res * res,), dtype=np.float32)
            norm = np.clip(q / 255.0, 0.0, 1.0)
            return (float(c.heatmap_min) + norm * span).astype(np.float32)

        task = contour_a.prediction_task if contour_a.prediction_task == contour_b.prediction_task else "classification"
        summary = compare_label_fields(
            _decode_heatmap(contour_a),
            _decode_heatmap(contour_b),
            grid_resolution=res,
            x_feature=feature_names[resolved_x],
            y_feature=feature_names[resolved_y],
            prediction_task=task,
        )
        mask_resp = summary_to_response(summary)

        conflict_frames: list[int] = []
        try:
            if selected and task == "classification":
                from app.domains.simulation.contour_2d_generator import PREDICT_CHUNK_SIZE

                def _batched(adapter: Any, holder: Any) -> list[float]:
                    out: list[float] = []
                    mat = np.ascontiguousarray(X[selected], dtype=np.float32)
                    off = 0
                    while off < mat.shape[0]:
                        chunk = np.ascontiguousarray(mat[off:off + PREDICT_CHUNK_SIZE], dtype=np.float32)
                        r = adapter.predict_batch(holder, chunk)
                        out.extend(float(v) for v in list(r.predictions))
                        del chunk, r
                        off += PREDICT_CHUNK_SIZE
                    return out

                preds_a = await asyncio.to_thread(_batched, adapter_a, model_a)
                preds_b = await asyncio.to_thread(_batched, adapter_b, model_b)
                idx = frame_disagreement_indices(preds_a, preds_b, is_classification=True)
                conflict_frames = [int(selected[i]) for i in idx if 0 <= int(i) < len(selected)][:10_000]
                del preds_a, preds_b
        except Exception:
            conflict_frames = []
        del X

        return DualContour2DResponse(
            x_feature=feature_names[resolved_x],
            y_feature=feature_names[resolved_y],
            grid_resolution=res,
            prediction_task=task,
            disagreement_pct=mask_resp["disagreement_pct"],
            agreement_pct=mask_resp["agreement_pct"],
            n_cells=mask_resp["n_cells"],
            n_conflict_cells=mask_resp["n_conflict_cells"],
            consensus_notice=mask_resp["consensus_notice"],
            conflict_mask=mask_resp["conflict_mask"],
            conflict_mask_downsampled=mask_resp["conflict_mask_downsampled"],
            conflict_frame_indices=conflict_frames,
            model_a_name=type(final_a).__name__ if final_a is not None else type(model_a).__name__,
            model_b_name=type(final_b).__name__ if final_b is not None else type(model_b).__name__,
            contour_a=contour_a.model_dump(mode="json"),
            contour_b=contour_b.model_dump(mode="json"),
            computation_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )


class _ContourCache:
    """Process-wide LRU for Phase-41 contour payloads (≤8 entries)."""

    _store: dict[tuple[Any, ...], Any] = {}
    _order: list[tuple[Any, ...]] = []

    @classmethod
    def get(cls, key: tuple[Any, ...]) -> Any | None:
        entity = cls._store.get(key)
        if entity is not None:
            cls._order.remove(key)
            cls._order.append(key)
        return entity

    @classmethod
    def put(cls, key: tuple[Any, ...], entity: Any) -> None:
        if key in cls._store:
            cls._order.remove(key)
        cls._store[key] = entity
        cls._order.append(key)
        while len(cls._order) > 8:
            oldest = cls._order.pop(0)
            cls._store.pop(oldest, None)

    @classmethod
    def clear(cls) -> None:
        cls._store.clear()
        cls._order.clear()


async def _resolve_contour_frame(
    req: ProjectionContourRequest,
    orchestrator: ProjectionOrchestrator,
) -> Any:
    """Shared validation + dataset/model loading for the contour path."""
    from app.domains.simulation.contour_2d_generator import Contour2DGenerator

    mapping = orchestrator._mappings.get(req.mapping_id)
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
    resolved_x = _resolve_feature_index(items, req.x_feature)
    resolved_y = _resolve_feature_index(items, req.y_feature)
    if resolved_x is None or resolved_y is None:
        available = [i.dataset_column_name or i.model_feature_name for i in items]
        raise DomainException(
            code="PROJECTION_UNKNOWN_FEATURE",
            message=f"Unknown feature(s): x='{req.x_feature}', y='{req.y_feature}'",
            details={"x_feature": req.x_feature, "y_feature": req.y_feature, "available_features": available},
            http_status=422,
        )
    if resolved_x == resolved_y and n_features >= 2:
        raise DomainException(
            code="PROJECTION_DUPLICATE_AXES",
            message="X and Y axes must be different features (use swapAxes on the client)",
            http_status=422,
        )

    selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
    if len(selected) > MAX_QUEUE_POINTS:
        raise DomainException(
            code="PROJECTION_TOO_MANY_POINTS",
            message=f"Select at most {MAX_QUEUE_POINTS} rows per scene",
            http_status=422,
        )

    res = max(16, min(200, int(req.grid_resolution)))
    grid_bytes = float(res * res * n_features * 4)
    available_mb = float(psutil.virtual_memory().available) / (1024 * 1024)
    if grid_bytes / (1024 * 1024) > max(0.0, available_mb - MIN_FREE_RAM_MB):
        raise DomainException(
            code="PROJECTION_WIDE_GRID_TOO_LARGE",
            message=f"Contour grid {res}x{res}x{n_features} exceeds free RAM",
            http_status=413,
        )

    feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
    parquet_path = orchestrator._datasets._require(req.dataset_id)
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

    cache_key = (
        req.dataset_id, req.model_id, req.mapping_id,
        req.x_feature, req.y_feature, req.pinning_mode.value,
        int(req.current_frame_index) if req.current_frame_index is not None else -1,
        bool(req.use_robust_range), res,
    )
    started = time.perf_counter()
    cached = _ContourCache.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"computation_time_ms": round((time.perf_counter() - started) * 1000.0, 2)})

    model_obj = orchestrator._models.get_model(req.model_id)
    _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
    adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

    df = orchestrator._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
    try:
        df = df.reindex(columns=feature_cols)
        X = TypeCastingEngine.build_float32_matrix(df, items)
    finally:
        del df

    current_row_values: np.ndarray | None = None
    if req.pinning_mode == PinningMode.CURRENT_ROW and selected:
        frame = req.current_frame_index if req.current_frame_index is not None else 0
        frame = max(0, min(int(frame), len(selected) - 1))
        if 0 <= selected[frame] < X.shape[0]:
            current_row_values = X[selected[frame]].astype(np.float32, copy=True)

    feature_names = [i.dataset_column_name or i.model_feature_name for i in items]
    entity = await asyncio.to_thread(
        Contour2DGenerator.generate_contour,
        adapter=adapter,
        model_obj=model_obj,
        final_est=final_est,
        X=X,
        x_feature_idx=resolved_x,
        y_feature_idx=resolved_y,
        x_feature_name=feature_names[resolved_x],
        y_feature_name=feature_names[resolved_y],
        grid_resolution=res,
        pinning_mode=req.pinning_mode.value,
        current_row_values=current_row_values,
        use_robust_range=bool(req.use_robust_range),
    )
    del X
    _ContourCache.put(cache_key, entity)
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
