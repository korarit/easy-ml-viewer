"""Phase 14 — studio metadata entities (per-session alias config)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domains.studio_metadata.value_objects import ClassAliasVO, FeatureAliasVO

MAX_CLASS_ALIASES = 64
MAX_FEATURE_ALIASES = 512


class SessionAliasConfigEntity(BaseModel):
    session_id: str = Field(min_length=1)
    class_aliases: dict[str, ClassAliasVO] = Field(default_factory=dict)
    feature_aliases: dict[str, FeatureAliasVO] = Field(default_factory=dict)
