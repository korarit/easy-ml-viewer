"""Phase 57 — Advanced Video Export Studio endpoints (offline).

- ``POST /exports/video-plan``: validate options → deterministic compositor plan
  (layout rects, theme tokens, duration, encoder config).
- ``POST /exports/transcode-mp4``: fallback path when the browser lacks
  WebCodecs — accepts a WebM upload and re-muxes to MP4 via PyAV/FFmpeg when
  available, else HTTP 501 with an actionable offline message.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.domains.export_reporting.video_export_service import (
    VideoExportPlanError,
    av_available,
    build_video_plan,
)

router = APIRouter(prefix="/exports", tags=["export-video-studio"])


class VideoPlanRequest(BaseModel):
    dimension: str = "3d"
    layout: str = "split_studio"
    include_why_panel: bool = True
    include_what_if_panel: bool = True
    theme: str = "match_ui"
    camera_preset: str = "isometric"
    speed_multiplier: float = 1.0
    fps: int = 60
    format: str = "mp4"
    resolution: str = "1080p"
    bitrate_mbps: float = 12.0
    range_start: int = 0
    range_end: int = 0


class TranscodeResponse(BaseModel):
    download_url: str = ""
    duration_sec: float = 0.0
    output_format: str = "mp4"
    file_size_bytes: int = 0
    backend: str = "none"
    message: str = ""


@router.post("/video-plan", summary="Phase 57: validate options → compositor plan")
async def video_export_plan(body: VideoPlanRequest) -> JSONResponse:
    try:
        plan = build_video_plan(body.model_dump())
    except VideoExportPlanError as exc:
        return JSONResponse(status_code=400, content={"code": "VIDEO_PLAN_INVALID", "message": str(exc)})
    return JSONResponse(content=plan)


def _sniff_webm(raw: bytes) -> bool:
    return len(raw) >= 4 and raw[0] == 0x1A and raw[1] == 0x45 and raw[2] == 0xDF and raw[3] == 0xA3


@router.post("/transcode-mp4", summary="Phase 57: WebM → MP4 fallback transcode (PyAV)")
async def transcode_mp4(file: UploadFile = File(...)) -> Any:
    raw = await file.read()
    if not raw:
        return JSONResponse(
            status_code=400,
            content={"code": "TRANSCODE_EMPTY_FILE", "message": "Uploaded file must be non-empty"},
        )
    if len(raw) > 500 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"code": "TRANSCODE_TOO_LARGE", "message": "Upload exceeds the 500 MB offline budget"},
        )
    if not av_available():
        return JSONResponse(
            status_code=501,
            content={
                "code": "TRANSCODE_BACKEND_MISSING",
                "message": (
                    "PyAV/FFmpeg backend is not installed in this offline environment "
                    "(pip install av). Export .webm directly from the Studio instead."
                ),
                "output_format": "mp4",
                "backend": "none",
            },
        )
    try:
        import av  # type: ignore
    except Exception as exc:  # pragma: no cover — guarded above
        return JSONResponse(
            status_code=501,
            content={"code": "TRANSCODE_BACKEND_MISSING", "message": f"PyAV import failed: {exc}"},
        )
    # Decode WebM in-memory → encode H.264 MP4 to a temp file (bounded).
    try:
        with tempfile.TemporaryDirectory(prefix="mlv-transcode-") as tmpdir:
            src = Path(tmpdir) / "input.webm"
            dst = Path(tmpdir) / "output.mp4"
            src.write_bytes(raw)
            in_container = av.open(str(src))
            in_stream = next((s for s in in_container.streams if s.type == "video"), None)
            if in_stream is None:
                return JSONResponse(
                    status_code=400,
                    content={"code": "TRANSCODE_NO_VIDEO", "message": "No video stream found in upload"},
                )
            fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
            fps = min(max(fps, 5.0), 60.0)
            out_container = av.open(str(dst), mode="w")
            out_stream = out_container.add_stream("libx264", rate=int(round(fps)))
            out_stream.width = int(in_stream.width)
            out_stream.height = int(in_stream.height)
            out_stream.pix_fmt = "yuv420p"
            n_frames = 0
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)
                n_frames += 1
                if n_frames > 10_000:
                    break
            for packet in out_stream.encode():
                out_container.mux(packet)
            out_container.close()
            in_container.close()
            blob = dst.read_bytes()

        def _iter():
            yield blob

        duration = (n_frames / fps) if fps else 0.0
        return StreamingResponse(
            io.BytesIO(blob),
            media_type="video/mp4",
            headers={
                "Content-Disposition": 'attachment; filename="studio-export.mp4"',
                "X-Transcode-Frames": str(n_frames),
                "X-Transcode-Duration-Sec": f"{duration:.3f}",
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"code": "TRANSCODE_FAILED", "message": f"FFmpeg transcode failed: {exc}"},
        )
