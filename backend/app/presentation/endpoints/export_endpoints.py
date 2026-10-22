"""Phase 16 — enriched-table export REST endpoints (CSV / XLSX / Parquet)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.application.export_orchestrator import (
    ExportEnrichedTableRequest,
    ExportOrchestrator,
)
from app.domains.export_reporting.services import EnrichedDatasetExportService

router = APIRouter(prefix="/exports", tags=["export-reporting"])

_CONTENT_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "parquet": "application/octet-stream",
}


def get_export_orchestrator() -> ExportOrchestrator:
    return ExportOrchestrator()


@router.post("/enriched-table", summary="Phase 16: export enriched predictions table")
async def export_enriched_table(
    body: ExportEnrichedTableRequest,
    orchestrator: ExportOrchestrator = Depends(get_export_orchestrator),
):
    data = await orchestrator.build_export_data(body)
    base_columns: list[str] = data["base_columns"]
    raw_rows: list[dict[str, Any]] = data["raw_rows"]
    enriched_rows: list[dict[str, Any]] = data["enriched_rows"]
    row_indices: list[int] = data["row_indices"]
    n = int(data["total_frames"])
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in body.dataset_id)[:40] or "export"

    if body.format == "csv":
        chunks = EnrichedDatasetExportService.stream_enriched_csv_chunks(
            base_columns, raw_rows, enriched_rows, row_indices
        )

        def _iter():
            yield from chunks

        return StreamingResponse(
            _iter(),
            media_type=_CONTENT_TYPES["csv"],
            headers={"Content-Disposition": f'attachment; filename="enriched-{safe}-n{n}.csv"'},
        )
    if body.format == "xlsx":
        blob = EnrichedDatasetExportService.build_xlsx_bytes(
            base_columns, raw_rows, enriched_rows, row_indices
        )

        def _iter_xlsx():
            yield blob

        return StreamingResponse(
            _iter_xlsx(),
            media_type=_CONTENT_TYPES["xlsx"],
            headers={"Content-Disposition": f'attachment; filename="enriched-{safe}-n{n}.xlsx"'},
        )
    blob = EnrichedDatasetExportService.build_parquet_bytes(
        base_columns, raw_rows, enriched_rows, row_indices
    )

    def _iter_parquet():
        yield blob

    return StreamingResponse(
        _iter_parquet(),
        media_type=_CONTENT_TYPES["parquet"],
        headers={"Content-Disposition": f'attachment; filename="enriched-{safe}-n{n}.parquet"'},
    )


@router.post("/svg-2d", summary="Phase 48: publication-quality 2D decision-boundary SVG")
async def export_svg_2d(payload: dict[str, Any]):
    """Build a standalone offline SVG from 2D boundary segments + points."""
    from fastapi.responses import Response as _Response

    from app.domains.export_reporting.svg_export_service import SvgExportGenerator

    def _clean_segments(raw: Any) -> list[list[float]]:
        out: list[list[float]] = []
        if not isinstance(raw, list):
            return out
        for seg in raw[:50_000]:
            try:
                vals = [float(v) for v in list(seg)[:4]]
            except (TypeError, ValueError):
                continue
            if len(vals) == 4:
                out.append(vals)
        return out

    def _clean_points(raw: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(raw, list):
            return out
        for pt in raw[:20_000]:
            if not isinstance(pt, dict):
                continue
            try:
                x = float(pt.get("x", 0.0))
                y = float(pt.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            try:
                actual = int(pt.get("actual", 0) or 0)
            except (TypeError, ValueError):
                actual = 0
            out.append({"x": x, "y": y, "actual": actual})
        return out

    disagreement = payload.get("disagreement_pct")
    try:
        disagreement_f = float(disagreement) if disagreement is not None else None
    except (TypeError, ValueError):
        disagreement_f = None
    svg = SvgExportGenerator.generate_svg(
        x_label=str(payload.get("x_label", "X")),
        y_label=str(payload.get("y_label", "Y")),
        boundary_segments=_clean_segments(payload.get("boundary_segments")),
        data_points=_clean_points(payload.get("data_points")),
        width=int(payload.get("width", 1200) or 1200),
        height=int(payload.get("height", 900) or 900),
        theme=str(payload.get("theme", "precision-dark")),
        title=str(payload.get("title", "2D Decision Boundary")),
        disagreement_pct=disagreement_f,
    )
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(payload.get("title", "decision-2d")))[:40] or "decision-2d"
    return _Response(
        content=svg.encode("utf-8"),
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{safe}.svg"'},
    )


@router.post("/enriched-table/preview", summary="Phase 16: preview first rows of an export")
async def preview_enriched_table(
    body: ExportEnrichedTableRequest,
    orchestrator: ExportOrchestrator = Depends(get_export_orchestrator),
) -> JSONResponse:
    data = await orchestrator.build_export_data(body)
    limit = 20
    header = EnrichedDatasetExportService.header_columns(data["base_columns"])
    rows: list[list[Any]] = []
    for t in range(min(limit, len(data["raw_rows"]))):
        rows.append(
            EnrichedDatasetExportService.row_to_list(
                data["base_columns"],
                data["raw_rows"][t],
                data["enriched_rows"][t],
                t,
                int(data["row_indices"][t]),
            )
        )
    return JSONResponse(
        {
            "header": header,
            "rows": rows,
            "total_frames": data["total_frames"],
            "is_classification": data["is_classification"],
            "format": body.format,
            "scope": body.scope,
        }
    )
