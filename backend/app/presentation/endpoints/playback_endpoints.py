"""Phase 10 — VCR playback REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.playback_orchestrator import (
    PlaybackManifestRequest,
    PlaybackOrchestrator,
    manifest_to_response,
)

router = APIRouter(prefix="/playback", tags=["playback"])


def get_playback_orchestrator() -> PlaybackOrchestrator:
    return PlaybackOrchestrator()


@router.post("/manifest", summary="Phase 10: build playback sequence manifest")
async def build_playback_manifest(
    body: PlaybackManifestRequest,
    orchestrator: PlaybackOrchestrator = Depends(get_playback_orchestrator),
) -> dict[str, Any]:
    entity = await orchestrator.build_manifest(body)
    return manifest_to_response(entity)


@router.get("/{session_id}/manifest", summary="Phase 10: fetch cached playback manifest")
async def get_playback_manifest(
    session_id: str,
    orchestrator: PlaybackOrchestrator = Depends(get_playback_orchestrator),
) -> dict[str, Any]:
    entity = orchestrator.get_manifest(session_id)
    return manifest_to_response(entity)
