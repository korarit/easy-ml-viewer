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
from app.core.binary_codec import MlvFormatError as _CodecError
from app.core.binary_codec import decode_mlv_package
from app.domains.recording_mlv.dual_compare_service import (
    compare_decoded_payloads,
    summary_to_response,
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


@router.get("/{session_id}/contour-2d", summary="Phase 48: instant 2D contour cache from a loaded .mlv")
async def get_mlv_contour_2d(session_id: str) -> dict[str, Any]:
    """Return the baked ``contour_2d_cache`` block (404 when legacy/absent).

    The heatmap ships as ``heatmap_quantized`` (uint8) so the response stays
    under the 150 KB DTO budget; clients decode with
    ``min + (q/255) * (max-min)`` — the same rule as the live contour path.
    """
    import numpy as np

    from app.application.mlv_recording_orchestrator import _PREBAKED

    try:
        blob = _PREBAKED[session_id]
    except KeyError:
        from app.core.exceptions import DomainException as _DE

        raise _DE(
            code="MLV_SESSION_NOT_FOUND",
            message=f"Prebaked .mlv session '{session_id}' not found — POST /api/v1/recordings/load first",
            http_status=404,
        ) from None
    block = blob.get("contour_2d_cache")
    if not isinstance(block, dict) or not block:
        from app.core.exceptions import DomainException as _DE

        raise _DE(
            code="MLV_NO_CONTOUR_2D",
            message="This .mlv has no baked 2D contour (legacy v1.x) — recompute via /projection/contour-2d",
            http_status=404,
        )
    heat = np.asarray(block.get("heatmap_float32"), dtype=np.float32)
    h_min = float(block.get("heatmap_min", 0.0))
    h_max = float(block.get("heatmap_max", 1.0))
    span = float(h_max - h_min)
    if heat.size and span > 1e-12:
        norm = np.clip((heat.ravel().astype(np.float64) - h_min) / span, 0.0, 1.0)
        quantized = (np.rint(norm * 255.0).astype(np.uint8)).tolist()
    else:
        quantized = [128] * int(heat.size)
    # Decision polylines → live-contour segment shape for renderer reuse.
    segments: list[list[float]] = []
    for poly in (block.get("boundary_polylines", []) or []):
        pts = poly.get("points", []) if isinstance(poly, dict) else []
        if isinstance(pts, list) and len(pts) >= 2:
            try:
                (x1, y1), (x2, y2) = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
                segments.append([x1, y1, x2, y2])
            except (TypeError, ValueError, IndexError):
                continue
    return {
        "session_id": session_id,
        "x_feature": str(block.get("x_feature", "")),
        "y_feature": str(block.get("y_feature", "")),
        "grid_resolution": int(block.get("grid_resolution", 0) or 0),
        "prediction_task": str(block.get("prediction_task", "classification")),
        "heatmap_min": h_min,
        "heatmap_max": h_max,
        "heatmap_quantized": quantized,
        "decision_segments": segments,
        "boundary_truncated": bool(block.get("boundary_truncated", False)),
        "model_specific_2d": block.get("model_specific_2d"),
    }


@router.get("/{session_id}/point-intelligence-2d", summary="Phase 56: instant Point Intelligence 2D track from a loaded .mlv")
async def get_mlv_point_intelligence_2d(session_id: str) -> dict[str, Any]:
    """Return the baked ``point_intelligence_2d_track`` block (404 when legacy/absent).

    Frame states ship columnar (float32 ND arrays via the shared codec) so a
    1,000-row × 100-frame track stays under the 150 KB spec budget; clients
    render trails / dual-rings / error lens / hexbins with zero recompute.
    """
    import numpy as np

    from app.application.mlv_recording_orchestrator import _PREBAKED

    try:
        blob = _PREBAKED[session_id]
    except KeyError:
        from app.core.exceptions import DomainException as _DE

        raise _DE(
            code="MLV_SESSION_NOT_FOUND",
            message=f"Prebaked .mlv session '{session_id}' not found — POST /api/v1/recordings/load first",
            http_status=404,
        ) from None
    block = blob.get("point_intelligence_2d_track")
    if not isinstance(block, dict) or not block:
        from app.core.exceptions import DomainException as _DE

        raise _DE(
            code="MLV_NO_POINT_INTELLIGENCE",
            message="This .mlv has no baked Point Intelligence track (legacy file) — recompute via /projection/point-intelligence-2d",
            http_status=404,
        ) from None

    try:
        scale = float(block.get("quant_scale", 1000.0) or 1000.0)
    except (TypeError, ValueError):
        scale = 1000.0

    def _col(name: str, quantized: bool = False) -> list[float]:
        try:
            arr = np.asarray(block.get(name, [])).ravel()
            if quantized and arr.dtype == np.int16:
                arr = arr.astype(np.float32) / np.float32(scale)
            else:
                arr = arr.astype(np.float32)
            return [float(v) for v in arr.tolist()]
        except (TypeError, ValueError):
            return []

    grid = block.get("hexbin_grid", {}) or {}
    return {
        "session_id": session_id,
        "flags": dict(block.get("flags", {}) or {}),
        "hexbin_radius": float(block.get("hexbin_radius", 0.45) or 0.45),
        "hexbin_grid": {"cells": list(grid.get("cells", []) or []), "truncated": bool(grid.get("truncated", False))},
        "density_contours": list(block.get("density_contours", []) or []),
        "contours_truncated": bool(block.get("contours_truncated", False)),
        "n_frames": int(block.get("n_frames", 0) or 0),
        "quant_scale": scale,
        "point_x": _col("point_x", True),
        "point_y": _col("point_y", True),
        "point_actual": _col("point_actual"),
        "point_pred": _col("point_pred"),
        "point_residual": _col("point_residual", True),
        "point_is_error": [int(v) for v in np.asarray(block.get("point_is_error", []), dtype=np.int64).ravel().tolist()],
    }


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


@router.post("/compare", summary="Phase 40: synchronized dual .mlv comparison")
async def compare_mlv_recordings(
    file_a: UploadFile,
    file_b: UploadFile,
) -> dict[str, Any]:
    """Compare two `.mlv` files frame-by-frame (truncates to N_min).

    Returns disagreement indices, per-frame divergence, and agreement
    rate. Rejects non-`.mlv` uploads with MLV_INVALID_MAGIC.
    """
    from app.core.exceptions import DomainException as _DE

    raw_a = await file_a.read()
    raw_b = await file_b.read()
    if not raw_a or not raw_b:
        raise _DE(code="MLV_EMPTY_FILE", message="Both .mlv files must be non-empty", http_status=400)
    try:
        payload_a = decode_mlv_package(raw_a)
    except _CodecError as exc:
        raise _DE(code="MLV_INVALID_MAGIC", message=f"file_a is not a valid .mlv: {exc}", http_status=400) from exc
    try:
        payload_b = decode_mlv_package(raw_b)
    except _CodecError as exc:
        raise _DE(code="MLV_INVALID_MAGIC", message=f"file_b is not a valid .mlv: {exc}", http_status=400) from exc
    summary = compare_decoded_payloads(payload_a, payload_b)
    response = summary_to_response(summary)
    response["file_a_name"] = file_a.filename or "model_a.mlv"
    response["file_b_name"] = file_b.filename or "model_b.mlv"
    return response


@router.post("/estimate-size", summary="Phase 15: estimated .mlv byte size")
async def estimate_mlv_size(payload: dict[str, Any]) -> dict[str, Any]:
    n = int(payload.get("total_frames", 0) or 0)
    res = int(payload.get("surface_resolution", 60) or 60)
    include_xai = bool(payload.get("include_explainability", True))
    size = MlvRecordingOrchestrator.estimate_size(n, res, include_xai)
    return {"estimated_compressed_bytes": size, "total_frames": n, "surface_resolution": res}
