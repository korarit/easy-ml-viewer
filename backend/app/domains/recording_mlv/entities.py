"""Phase 15 — `.mlv` package entities (header / static world / frames track)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.binary_codec import MLV_SCHEMA_VERSION
from app.domains.recording_mlv.value_objects import MlvAutoEventVO, MlvBookmarkVO


class MlvHeaderSection(BaseModel):
    schema_version: str = MLV_SCHEMA_VERSION
    title: str = Field(min_length=1, max_length=160)
    notes: str = Field(default="", max_length=2000)
    created_at_iso: str
    model_type: str = ""
    model_family: str = ""
    is_classification: bool = False
    feature_names: list[str] = Field(default_factory=list)
    total_frames: int = Field(ge=0)
    class_aliases: dict = Field(default_factory=dict)
    feature_aliases: dict = Field(default_factory=dict)
    bookmarks: list[MlvBookmarkVO] = Field(default_factory=list)
    auto_events: list[MlvAutoEventVO] = Field(default_factory=list)


class MlvStaticWorldSection(BaseModel):
    surface_resolution: int = Field(default=60, ge=10, le=100)
    surface_visual_type: str = ""
    surface_world_vertices_xyz: list[float] = Field(default_factory=list)
    axis_x_feature: str = ""
    axis_y_feature: str = ""
    model_config = {"arbitrary_types_allowed": True}


class MlvFramesTrackSection(BaseModel):
    row_indices: list[int] = Field(default_factory=list)
    predictions: list[float] = Field(default_factory=list)
    confidences: list[float] = Field(default_factory=list)
    model_config = {"arbitrary_types_allowed": True}


class MlvPackageEntity(BaseModel):
    header: MlvHeaderSection
    static_world: MlvStaticWorldSection
    frames_track: MlvFramesTrackSection
    # Phase 48 — optional normalized 2D contour cache (None on legacy files).
    contour_2d_cache: dict | None = None
    # Phase 56 — optional Point Intelligence 2D multi-track (None on legacy files).
    point_intelligence_2d_track: dict | None = None
    model_config = {"arbitrary_types_allowed": True}


class PrebakedMlvSessionEntity(BaseModel):
    session_id: str = Field(min_length=1)
    header: MlvHeaderSection
    total_frames: int = Field(ge=0)
    is_prebaked_replay: bool = True
    model_config = {"arbitrary_types_allowed": True}
