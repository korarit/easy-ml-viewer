"""Phase 15 — `.mlv` recording REST endpoints (bake / load / replay info)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import StreamingResponse

from app.application.mlv_recording_orchestrator import (
    MlvBakeRequest,
    MlvRecordingOrchestrator,
    MlvRebakeBookmarksRequest,
    header_to_response,
    prebaked_to_response,
)

router = APIRouter(prefix="/recordings", tags=["recording-mlv"])


def get_mlv_orchestrator() -> MlvRecordingOrchestrator:
    return MlvRecordingOrchestrator()


@router.post("/{session_id}/bake", summary="Phase 15: bake session into .mlv binary")
async def bake_mlv_recording(
    session_id: str,
    body: MlvBakeRequest,
    orchestrator: MlvRecordingOrchestrator = Depends(get_mlv_orchestrator),
) -> StreamingResponse:
    if body.session_id != session_id:
        body = MlvBakeRequest(**{**body.model_dump(), "session_id": session_id})
    blob, header = orchestrator.bake(body)
    safe_title = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in header.title)[:60] or "recording"
    filename = f"{safe_title}.mlv"

    def _iter():
        yield blob

    return StreamingResponse(
        _iter(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/load", summary="Phase 15: instant-load a .mlv recording")
async def load_mlv_recording(
    file: UploadFile,
    orchestrator: MlvRecordingOrchestrator = Depends(get_mlv_orchestrator),
) -> dict[str, Any]:
    raw = await file.read()
    entity = MlvRecordingOrchestrator.load(raw)
    return prebaked_to_response(entity)


@router.get("/{session_id}/info", summary="Phase 15: prebaked replay session info")
async def get_mlv_info(
    session_id: str,
    orchestrator: MlvRecordingOrchestrator = Depends(get_mlv_orchestrator),
) -> dict[str, Any]:
    _ = orchestrator
    return prebaked_to_response(MlvRecordingOrchestrator.get_prebaked(session_id))


@router.put("/{session_id}/bookmarks", summary="Phase 15: update bookmarks + re-bake .mlv")
async def rebake_mlv_bookmarks(
    session_id: str,
    body: MlvRebakeBookmarksRequest,
    orchestrator: MlvRecordingOrchestrator = Depends(get_mlv_orchestrator),
) -> StreamingResponse:
    _ = orchestrator
    blob = MlvRecordingOrchestrator.rebake_bookmarks(session_id, list(body.bookmarks or []))

    def _iter():
        yield blob

    return StreamingResponse(
        _iter(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{session_id}_updated.mlv"'},
    )


@router.post("/estimate-size", summary="Phase 15: estimated .mlv byte size")
async def estimate_mlv_size(payload: dict[str, Any]) -> dict[str, Any]:
    n = int(payload.get("total_frames", 0) or 0)
    res = int(payload.get("surface_resolution", 60) or 60)
    include_xai = bool(payload.get("include_explainability", True))
    size = MlvRecordingOrchestrator.estimate_size(n, res, include_xai)
    return {"estimated_compressed_bytes": size, "total_frames": n, "surface_resolution": res}
