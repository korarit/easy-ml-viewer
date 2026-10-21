"""Phase 14 — alias config application orchestrator (in-memory LRU store).

Pipeline:
1. Validate alias payloads via :class:`SessionAliasConfigEntity` (pydantic
   enforces ``maxLength=64`` labels + ``^#[0-9A-Fa-f]{6}$`` colors so an
   over-long HUD label can never reach the 3D stage).
2. Store per-``session_id`` in a bounded in-memory dict (no DataFrame is
   retained — only small alias dicts, so RAM stays flat).
3. ``autosuggest`` derives human-readable labels locally (no model call).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.exceptions import DomainException
from app.domains.studio_metadata.entities import (
    MAX_CLASS_ALIASES,
    MAX_FEATURE_ALIASES,
    SessionAliasConfigEntity,
)
from app.domains.studio_metadata.services import (
    suggest_class_alias,
    suggest_feature_alias,
)
from app.domains.studio_metadata.value_objects import ClassAliasVO, FeatureAliasVO

MAX_SESSIONS = 128

_STORE: dict[str, SessionAliasConfigEntity] = {}
_ORDER: list[str] = []


class AliasUpsertRequest(BaseModel):
    session_id: str = Field(min_length=1)
    class_aliases: dict[str, ClassAliasVO] = Field(default_factory=dict)
    feature_aliases: dict[str, FeatureAliasVO] = Field(default_factory=dict)


class AliasAutosuggestRequest(BaseModel):
    feature_names: list[str] = Field(default_factory=list)
    class_values: list[str | int] = Field(default_factory=list)


def _remember(session_id: str, entity: SessionAliasConfigEntity) -> None:
    _STORE[session_id] = entity
    if session_id in _ORDER:
        _ORDER.remove(session_id)
    _ORDER.append(session_id)
    while len(_ORDER) > MAX_SESSIONS:
        old = _ORDER.pop(0)
        _STORE.pop(old, None)


class AliasOrchestrator:
    @staticmethod
    def upsert(body: AliasUpsertRequest) -> SessionAliasConfigEntity:
        if len(body.class_aliases) > MAX_CLASS_ALIASES:
            raise DomainException(
                code="ALIAS_TOO_MANY_CLASSES",
                message=f"At most {MAX_CLASS_ALIASES} class aliases per session",
                http_status=422,
            )
        if len(body.feature_aliases) > MAX_FEATURE_ALIASES:
            raise DomainException(
                code="ALIAS_TOO_MANY_FEATURES",
                message=f"At most {MAX_FEATURE_ALIASES} feature aliases per session",
                http_status=422,
            )
        entity = SessionAliasConfigEntity(
            session_id=body.session_id,
            class_aliases=dict(body.class_aliases),
            feature_aliases=dict(body.feature_aliases),
        )
        _remember(body.session_id, entity)
        return entity

    @staticmethod
    def get(session_id: str) -> SessionAliasConfigEntity:
        try:
            return _STORE[session_id]
        except KeyError as exc:
            raise DomainException(
                code="ALIAS_SESSION_NOT_FOUND",
                message=f"Alias config '{session_id}' not found — PUT it first",
                http_status=404,
            ) from exc

    @staticmethod
    def autosuggest(body: AliasAutosuggestRequest) -> SessionAliasConfigEntity:
        seen_features: set[str] = set()
        feature_aliases: dict[str, FeatureAliasVO] = {}
        for raw in body.feature_names[:MAX_FEATURE_ALIASES]:
            name = (raw or "").strip()
            if not name or name in seen_features:
                continue
            seen_features.add(name)
            feature_aliases[name] = suggest_feature_alias(name)
        class_aliases: dict[str, ClassAliasVO] = {}
        for pos, raw in enumerate(body.class_values[:MAX_CLASS_ALIASES]):
            key = str(raw)
            if key in class_aliases:
                continue
            class_aliases[key] = suggest_class_alias(raw, pos)
        return SessionAliasConfigEntity(
            session_id="autosuggest",
            class_aliases=class_aliases,
            feature_aliases=feature_aliases,
        )

    @staticmethod
    def clear() -> int:
        n = len(_STORE)
        _STORE.clear()
        _ORDER.clear()
        return n


def alias_to_response(entity: SessionAliasConfigEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")
