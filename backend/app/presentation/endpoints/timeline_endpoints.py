"""Phase 11 — smart multi-track timeline REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response

from app.application.timeline_orchestrator import (
    TimelineBookmarkCreateRequest,
    TimelineOrchestrator,
    TimelineTracksBuildRequest,
    bookmark_to_response,
    entity_to_response,
)

router = APIRouter(prefix="/timeline", tags=["timeline"])


def get_timeline_orchestrator() -> TimelineOrchestrator:
    return TimelineOrchestrator()


@router.post("/tracks/build", summary="Phase 11: build multi-track timeline analysis")
async def build_timeline_tracks(
    body: TimelineTracksBuildRequest,
    orchestrator: TimelineOrchestrator = Depends(get_timeline_orchestrator),
) -> dict[str, Any]:
    entity = await orchestrator.build_tracks(body)
    return entity_to_response(entity)


@router.get("/{session_id}/tracks", summary="Phase 11: fetch cached timeline tracks")
async def get_timeline_tracks(
    session_id: str,
    orchestrator: TimelineOrchestrator = Depends(get_timeline_orchestrator),
) -> dict[str, Any]:
    entity = orchestrator.get_tracks(session_id)
    return entity_to_response(entity)


@router.post(
    "/{session_id}/bookmarks",
    status_code=201,
    summary="Phase 11: create a frame bookmark",
)
async def create_timeline_bookmark(
    session_id: str,
    body: TimelineBookmarkCreateRequest,
    response: Response,
    orchestrator: TimelineOrchestrator = Depends(get_timeline_orchestrator),
) -> dict[str, Any]:
    bookmark = orchestrator.create_bookmark(session_id, body)
    response.status_code = 201
    return bookmark_to_response(bookmark)


@router.delete(
    "/{session_id}/bookmarks/{bookmark_id}",
    summary="Phase 11: delete a frame bookmark",
)
async def delete_timeline_bookmark(
    session_id: str,
    bookmark_id: str,
    orchestrator: TimelineOrchestrator = Depends(get_timeline_orchestrator),
) -> dict[str, Any]:
    deleted = orchestrator.delete_bookmark(session_id, bookmark_id)
    return {"deleted": deleted, "session_id": session_id}
