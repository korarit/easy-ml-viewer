"""Phase 15 — `.mlv` recording orchestrator (bake + instant load).

Pipeline:
1. ``bake`` validates mapping, builds the Stage3D scene at the requested
   LOD (surface mesh + trajectory + axes in one chunked pass), then packs
   ``header`` / ``static_world`` / columnar ``frames_track`` via
   :mod:`app.domains.recording_mlv.services` and
   :mod:`app.core.binary_codec` (MsgPack + Zstd L3 + CRC32).
2. ``load`` validates Magic + CRC32, Zip-Bomb-guards the decompression
   (250 MB cap), restores typed entities and registers a read-only
   ``PrebakedMlvSession`` in a bounded in-memory LRU.
3. ``rebake_bookmarks`` lets Replay mode append notes without a model —
   only ``header.bookmarks`` is replaced before re-encoding.

Memory (§5.1): one ``(N,3)`` + ``(N,)`` float32 column set per bake;
``del`` + ``gc.collect()`` after compress; sessions LRU-capped.
"""

from __future__ import annotations

import gc
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.application.stage3d_use_case import InitializeSceneRequest, Stage3DUseCase
from app.core.binary_codec import decode_mlv_package, encode_mlv_package
from app.core.exceptions import DomainException
from app.domains.recording_mlv.entities import MlvHeaderSection, PrebakedMlvSessionEntity
from app.domains.recording_mlv.services import (
    assemble_payload,
    build_columnar_frames,
    build_static_world_payload,
    estimate_compressed_bytes,
    split_payload,
)
from app.domains.recording_mlv.value_objects import MlvAutoEventVO, MlvBookmarkVO
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)


def _mlv_error(code: str, message: str, http_status: int, details: dict[str, Any] | None = None):
    from app.core.exceptions import DomainException as _DE

    return _DE(code=code, message=message, details=details, http_status=http_status)


MAX_QUEUE_POINTS = 10_000
MAX_SESSIONS = 32

_PREBAKED: dict[str, dict[str, Any]] = {}
_PREBAKED_ORDER: list[str] = []


class MlvBakeRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    selected_row_indices: list[int] = Field(default_factory=list)
    session_id: str | None = None
    title: str = Field(default="Untitled Recording", max_length=160)
    notes: str = Field(default="", max_length=2000)
    surface_resolution: int = Field(default=60, ge=10, le=100)
    include_explainability: bool = True
    include_bookmarks: bool = True
    compression_level: int = Field(default=3, ge=1, le=19)
    # Phase 48 — optional raw Contour2DResponse JSON; baked into
    # ``contour_2d_cache`` for instant 2D (<0.1 s, no recompute on load).
    contour_2d: dict[str, Any] | None = Field(default=None)


class MlvRebakeBookmarksRequest(BaseModel):
    bookmarks: list[MlvBookmarkVO] = Field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remember_prebaked(session_id: str, blob: dict[str, Any]) -> None:
    _PREBAKED[session_id] = blob
    if session_id in _PREBAKED_ORDER:
        _PREBAKED_ORDER.remove(session_id)
    _PREBAKED_ORDER.append(session_id)
    while len(_PREBAKED_ORDER) > MAX_SESSIONS:
        old = _PREBAKED_ORDER.pop(0)
        _PREBAKED.pop(old, None)


def _confidence_from_residual(residual: float | None, span: float) -> float:
    if residual is None or not np.isfinite(residual):
        return 0.85
    scale = max(float(span) * 0.25, 1e-9)
    return float(np.clip(1.0 / (1.0 + abs(float(residual)) / scale), 0.0, 1.0))


class MlvRecordingOrchestrator:
    def __init__(
        self,
        stage3d: Stage3DUseCase | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
    ) -> None:
        self._stage3d = stage3d or Stage3DUseCase(
            model_use_case=model_use_case, mapping_repo=mapping_repo
        )
        self._mappings = mapping_repo or InMemoryMappingRepository()
        self._models = model_use_case or ModelIngestUseCase()

    # -- bake --------------------------------------------------------
    def bake(self, req: MlvBakeRequest) -> tuple[bytes, MlvHeaderSection]:
        from app.domains.model_engine.registry import ModelRegistry
        from app.domains.model_engine.schema_detector import UniversalSchemaDetector
        from app.domains.model_engine.value_objects import TaskType

        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise _mlv_error(
                "MAPPING_MISMATCH",
                "Mapping does not belong to the given dataset/model pair",
                422,
                {"mapping_id": req.mapping_id},
            )
        selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
        if len(selected) > MAX_QUEUE_POINTS:
            raise _mlv_error(
                "MLV_TOO_MANY_FRAMES",
                f"Select at most {MAX_QUEUE_POINTS} rows per .mlv recording",
                422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        feature_names = [
            i.model_feature_name or i.dataset_column_name or f"feature_{i.feature_index}"
            for i in items
        ]
        n_features = len(items)
        if n_features < 2:
            raise _mlv_error(
                "MLV_NOT_ENOUGH_FEATURES",
                "Mapping needs at least 2 mapped features for 3D recording",
                422,
            )
        x_idx = next((i for i, it in enumerate(items) if it.dataset_column_name), 0)
        y_idx = next(
            (i for i, it in enumerate(items) if it.dataset_column_name and i != x_idx),
            min(1, n_features - 1),
        )

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")
        is_classification = adapter.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

        scene = self._stage3d.initialize_scene(
            InitializeSceneRequest(
                dataset_id=req.dataset_id,
                model_id=req.model_id,
                mapping_id=req.mapping_id,
                axis_x_feature_index=x_idx,
                axis_y_feature_index=y_idx,
                selected_row_indices=selected,
                grid_resolution=int(req.surface_resolution),
            )
        )

        n = len(scene.trajectory_points)
        preds = np.asarray(
            [p.raw_z_pred if np.isfinite(p.raw_z_pred) else 0.0 for p in scene.trajectory_points],
            dtype=np.float32,
        )
        finite = preds[np.isfinite(preds)]
        span = float(np.max(finite) - np.min(finite)) if finite.size else 1.0
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        positions = np.empty((n, 3), dtype=np.float32)
        row_indices: list[int] = []
        confidences = np.empty((n,), dtype=np.float32)
        for t, p in enumerate(scene.trajectory_points):
            positions[t] = (
                float(np.clip(p.world_x, -7.0, 7.0)) if np.isfinite(p.world_x) else 0.0,
                float(np.clip(p.world_y, -7.0, 7.0)) if np.isfinite(p.world_y) else 0.0,
                float(np.clip(p.world_z_pred, -7.0, 7.0)) if np.isfinite(p.world_z_pred) else 0.0,
            )
            row_indices.append(int(p.original_row_index))
            confidences[t] = _confidence_from_residual(p.residual_error, span)

        # Optional session-linked tracks (timeline bookmarks/events + aliases).
        bookmarks: list[MlvBookmarkVO] = []
        auto_events: list[MlvAutoEventVO] = []
        class_aliases: dict = {}
        feature_aliases: dict = {}
        if req.include_bookmarks and req.session_id:
            try:
                from app.application.timeline_orchestrator import _SESSIONS as _TL

                tl = _TL.get(req.session_id)
                if tl is not None:
                    for b in tl.entity.bookmarks:
                        bookmarks.append(
                            MlvBookmarkVO(
                                bookmark_id=b.bookmark_id,
                                frame_index=int(b.frame_index),
                                dataset_row_index=int(b.dataset_row_index),
                                color=str(b.color),
                                note_text=b.note_text,
                                created_at_iso=b.created_at_iso,
                            )
                        )
                    for e in tl.entity.auto_events:
                        auto_events.append(
                            MlvAutoEventVO(
                                frame_index=int(e.frame_index),
                                kind=str(e.kind),
                                severity=float(e.severity),
                                label=str(e.label or e.kind),
                            )
                        )
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            try:
                from app.application.alias_orchestrator import _STORE as _AL

                alias_cfg = _AL.get(req.session_id)
                if alias_cfg is not None:
                    class_aliases = {
                        k: v.model_dump(mode="json") for k, v in alias_cfg.class_aliases.items()
                    }
                    feature_aliases = {
                        k: v.model_dump(mode="json") for k, v in alias_cfg.feature_aliases.items()
                    }
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        header = MlvHeaderSection(
            title=req.title.strip() or "Untitled Recording",
            notes=req.notes or "",
            created_at_iso=_utc_now_iso(),
            model_type=getattr(adapter, "algorithm_key", type(final_est).__name__),
            model_family=getattr(adapter, "model_family", ""),
            is_classification=bool(is_classification),
            feature_names=feature_names,
            total_frames=n,
            class_aliases=class_aliases,
            feature_aliases=feature_aliases,
            bookmarks=bookmarks,
            auto_events=auto_events,
        )
        static_world = build_static_world_payload(
            surface_vertices_xyz=list(scene.surface.world_vertices_xyz),
            surface_resolution=int(scene.surface.resolution),
            surface_visual_type=str(scene.surface.visual_type),
            axis_x_feature=feature_names[x_idx] if x_idx < len(feature_names) else "",
            axis_y_feature=feature_names[y_idx] if y_idx < len(feature_names) else "",
        )
        frames_columns = build_columnar_frames(row_indices, positions, preds, confidences)
        top_k = np.zeros((n, min(10, n_features)), dtype=np.float32) if req.include_explainability and n else None
        # Phase 48 — optional 2D contour block (fail-safe: never breaks 3D bake).
        contour_cache: dict[str, Any] | None = None
        if isinstance(req.contour_2d, dict) and req.contour_2d:
            try:
                from app.domains.recording_mlv.contour_2d_baker import bake_contour_2d_cache

                contour_cache = bake_contour_2d_cache(req.contour_2d)
            except Exception:
                contour_cache = None
        payload = assemble_payload(header, static_world, frames_columns, top_k, contour_cache)
        try:
            blob = encode_mlv_package(payload, compression_level=int(req.compression_level))
        finally:
            del payload, positions, preds, confidences
            gc.collect()
        return blob, header

    # -- load --------------------------------------------------------
    @staticmethod
    def load(file_bytes: bytes, session_id: str | None = None) -> PrebakedMlvSessionEntity:
        from app.core.binary_codec import MlvFormatError as _CodecError

        if not file_bytes:
            raise _mlv_error("MLV_EMPTY_FILE", "Uploaded file is empty", 400)
        try:
            payload = decode_mlv_package(file_bytes)
        except _CodecError as exc:
            msg = str(exc)
            if "Magic" in msg:
                raise _mlv_error(
                    "MLV_INVALID_MAGIC",
                    "File is not a valid .mlv v1.0 recording (Invalid Magic Header)",
                    400,
                    {"hint": "Rename-check: a .csv/.pkl renamed to .mlv will fail here"},
                ) from exc
            if "CRC32" in msg or "corrupt" in msg.lower():
                raise _mlv_error("MLV_CORRUPTED", msg, 422) from exc
            raise _mlv_error("MLV_TRUNCATED", msg, 400) from exc
        package = split_payload(payload)
        sid = session_id or f"mlv_{uuid.uuid4().hex[:12]}"
        entity = PrebakedMlvSessionEntity(
            session_id=sid,
            header=package.header,
            total_frames=package.header.total_frames,
            is_prebaked_replay=True,
        )
        ft = payload.get("frames_track", {})
        contour_block = payload.get("contour_2d_cache")
        _remember_prebaked(
            sid,
            {
                "entity": entity,
                "package": package,
                "contour_2d_cache": dict(contour_block) if isinstance(contour_block, dict) else None,
                "raw_bytes": bytes(file_bytes),
                "positions_world": np.asarray(
                    ft.get("positions_world", np.zeros((package.header.total_frames, 3), dtype=np.float32)),
                    dtype=np.float32,
                ).copy(),
                "top_k_contribs": np.asarray(
                    payload.get("top_k_contribs", np.zeros((0, 0), dtype=np.float32)),
                    dtype=np.float32,
                ).copy(),
            },
        )
        del payload
        gc.collect()
        return entity

    @staticmethod
    def get_prebaked(session_id: str) -> PrebakedMlvSessionEntity:
        try:
            return _PREBAKED[session_id]["entity"]
        except KeyError as exc:
            raise _mlv_error(
                "MLV_SESSION_NOT_FOUND",
                f"Prebaked .mlv session '{session_id}' not found — POST /api/v1/recordings/load first",
                404,
            ) from exc

    @staticmethod
    def rebake_bookmarks(session_id: str, bookmarks: list[MlvBookmarkVO]) -> bytes:
        try:
            blob = _PREBAKED[session_id]
        except KeyError as exc:
            raise _mlv_error(
                "MLV_SESSION_NOT_FOUND",
                f"Prebaked .mlv session '{session_id}' not found — POST /api/v1/recordings/load first",
                404,
            ) from exc
        package = blob["package"]
        package.header.bookmarks = list(bookmarks or [])
        stored_top_k = np.asarray(
            blob.get("top_k_contribs", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32
        )
        stored_top_k = stored_top_k if stored_top_k.size else None
        stored_contour = blob.get("contour_2d_cache")
        payload = assemble_payload(
            package.header,
            {
                "surface_resolution": package.static_world.surface_resolution,
                "surface_visual_type": package.static_world.surface_visual_type,
                "surface_world_vertices_xyz": np.asarray(
                    package.static_world.surface_world_vertices_xyz, dtype=np.float32
                ),
                "axis_x_feature": package.static_world.axis_x_feature,
                "axis_y_feature": package.static_world.axis_y_feature,
            },
            build_columnar_frames(
                list(package.frames_track.row_indices),
                np.asarray(
                    blob.get("positions_world"),
                    dtype=np.float32,
                ),
                np.asarray(package.frames_track.predictions, dtype=np.float32),
                np.asarray(package.frames_track.confidences, dtype=np.float32),
            ),
            stored_top_k,
            dict(stored_contour) if isinstance(stored_contour, dict) else None,
        )
        try:
            out = encode_mlv_package(payload)
        finally:
            del payload
            gc.collect()
        return out

    @staticmethod
    def estimate_size(n_frames: int, surface_resolution: int, include_explainability: bool) -> int:
        return estimate_compressed_bytes(n_frames, surface_resolution, include_explainability)


def _positions_from_package(package: Any) -> np.ndarray:
    n = len(package.frames_track.row_indices)
    return np.zeros((n, 3), dtype=np.float32)


def header_to_response(header: MlvHeaderSection) -> dict[str, Any]:
    return header.model_dump(mode="json")


def prebaked_to_response(entity: PrebakedMlvSessionEntity) -> dict[str, Any]:
    body = entity.model_dump(mode="json")
    # Phase 48 — instant-2D summary for the load path (legacy files → zeros).
    try:
        contour = _PREBAKED.get(entity.session_id, {}).get("contour_2d_cache")
        if isinstance(contour, dict) and contour:
            polys = contour.get("boundary_polylines", []) or []
            body["contour_2d_summary"] = {
                "has_contour_2d": True,
                "x_feature": str(contour.get("x_feature", "")),
                "y_feature": str(contour.get("y_feature", "")),
                "grid_resolution": int(contour.get("grid_resolution", 0) or 0),
                "n_boundary_polylines": len(polys) if isinstance(polys, list) else 0,
                "boundary_truncated": bool(contour.get("boundary_truncated", False)),
            }
        else:
            body["contour_2d_summary"] = {"has_contour_2d": False}
    except Exception:
        body["contour_2d_summary"] = {"has_contour_2d": False}
    return body
