"""Phase 16 acceptance: enriched-table export (CSV/XLSX/Parquet) + memory SLA."""

from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient

from app.domains.export_reporting.services import EnrichedDatasetExportService
from app.main import app

client = TestClient(app)


def _seed_housing() -> tuple[str, str, str, int]:
    ds = client.post("/api/v1/datasets/load-sample/sample_houses")
    assert ds.status_code == 200, ds.text
    dataset_id = ds.json()["dataset_id"]
    mo = client.post("/api/v1/models/load-sample/sample_house_price_linear")
    assert mo.status_code == 200, mo.text
    model_id = mo.json()["model_id"]
    sug = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": dataset_id, "model_id": model_id},
    )
    assert sug.status_code == 200, sug.text
    mapping = sug.json()["mapping"]
    saved = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "items": mapping["items"],
            "target_column_name": "selling_price",
        },
    )
    assert saved.status_code == 200, saved.text
    mapping_id = saved.json()["mapping"]["mapping_id"]
    n_rows = int(ds.json().get("row_count") or 0)
    return dataset_id, model_id, mapping_id, n_rows


def _enriched(n: int, note: str = "") -> list[dict]:
    return [
        {
            "model_prediction": 100000.0 + i,
            "model_prediction_label": "100001",
            "model_confidence": 0.9,
            "residual_error": 1.5,
            "is_misclassified": None,
            "top_positive_feature": "area_sqm",
            "top_positive_impact": 0.35,
            "top_negative_feature": "distance_bts_km",
            "top_negative_impact": -0.02,
            "auto_anomaly_flags": "",
            "user_bookmark_note": note if i == 0 else "",
        }
        for i in range(n)
    ]


def test_csv_bom_rfc4180_and_thai_notes():
    base = ["area_sqm", "selling_price"]
    raw = [
        {"area_sqm": 35.0, "selling_price": 100000.0},
        {"area_sqm": 42.5, "selling_price": 200000.0},
    ]
    enriched = _enriched(2, note="เคสต้องสงสัย, รายได้ต่ำ\nตรวจสอบเอกสารเพิ่มเติม")
    chunks = list(
        EnrichedDatasetExportService.stream_enriched_csv_chunks(
            base, raw, enriched, [10, 25], chunk_size=1
        )
    )
    # Edge #2: UTF-8 BOM is always the first emitted payload.
    assert chunks[0] == b"\xef\xbb\xbf"
    decoded = b"".join(chunks).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(decoded)))
    header = rows[0]
    assert "model_prediction_label" in header
    assert "user_bookmark_note" in header
    assert "auto_anomaly_flags" in header
    # Edge #6: comma/newline note stays in ONE column (RFC 4180 quoting).
    assert len(rows[1]) == len(header)
    assert len(rows[2]) == len(header)
    assert rows[1][header.index("user_bookmark_note")] == "เคสต้องสงสัย, รายได้ต่ำ\nตรวจสอบเอกสารเพิ่มเติม"
    assert rows[1][header.index("top_positive_feature")] == "area_sqm"


def test_xlsx_and_parquet_roundtrip():
    base = ["area_sqm"]
    raw = [{"area_sqm": 35.0}, {"area_sqm": 42.5}]
    enriched = _enriched(2)
    xlsx_blob = EnrichedDatasetExportService.build_xlsx_bytes(base, raw, enriched, [0, 1])
    assert len(xlsx_blob) > 1_000
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_blob), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    assert rows[0][0] == "area_sqm"
    assert "model_prediction" in rows[0]
    assert len(rows) == 3

    pq_blob = EnrichedDatasetExportService.build_parquet_bytes(base, raw, enriched, [0, 1])
    assert len(pq_blob) > 100
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(pq_blob))
    assert "model_prediction" in table.schema.names
    assert table.num_rows == 2


def test_export_all_formats_api():
    dataset_id, model_id, mapping_id, n_rows = _seed_housing()
    assert n_rows > 0
    queue = list(range(min(n_rows, 10)))
    for fmt, ctype in [
        ("csv", "text/csv"),
        ("xlsx", "spreadsheetml"),
        ("parquet", "octet-stream"),
    ]:
        res = client.post(
            "/api/v1/exports/enriched-table",
            json={
                "dataset_id": dataset_id,
                "model_id": model_id,
                "mapping_id": mapping_id,
                "format": fmt,
                "scope": "all_queued",
                "selected_row_indices": queue,
                "include_attribution_columns": True,
                "include_bookmarks": True,
            },
        )
        assert res.status_code == 200, f"{fmt}: {res.text}"
        assert ctype in res.headers["content-type"]
        assert len(res.content) > 100
    # CSV payload starts with the UTF-8 BOM.
    res = client.post(
        "/api/v1/exports/enriched-table",
        json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "mapping_id": mapping_id,
            "format": "csv",
            "scope": "all_queued",
            "selected_row_indices": queue,
        },
    )
    assert res.content[:3] == b"\xef\xbb\xbf"


def test_export_preview_and_sub_range():
    dataset_id, model_id, mapping_id, n_rows = _seed_housing()
    queue = list(range(min(n_rows, 10)))
    prev = client.post(
        "/api/v1/exports/enriched-table/preview",
        json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "mapping_id": mapping_id,
            "format": "csv",
            "scope": "sub_range",
            "range_start": 0,
            "range_end": 2,
            "selected_row_indices": queue,
        },
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["total_frames"] == 3
    assert "model_prediction" in body["header"]
    assert len(body["rows"]) == 3

    # Empty queue is rejected, not silently exported.
    bad = client.post(
        "/api/v1/exports/enriched-table",
        json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "mapping_id": mapping_id,
            "format": "csv",
            "scope": "all_queued",
            "selected_row_indices": [],
        },
    )
    assert bad.status_code == 422


def test_50k_rows_streaming_memory_budget():
    import numpy as np
    import psutil

    n = 50_000
    base = [f"f_{i}" for i in range(4)]
    raw = [{c: 1.0 for c in base} for _ in range(n)]
    enriched = [
        {
            "model_prediction": 1.0,
            "model_prediction_label": "1",
            "model_confidence": 0.9,
            "residual_error": 0.0,
            "is_misclassified": None,
            "top_positive_feature": "f_0",
            "top_positive_impact": 0.1,
            "top_negative_feature": "f_1",
            "top_negative_impact": -0.1,
            "auto_anomaly_flags": "",
            "user_bookmark_note": "",
        }
        for _ in range(n)
    ]
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    total_bytes = 0
    for chunk in EnrichedDatasetExportService.stream_enriched_csv_chunks(
        base, raw, enriched, list(range(n)), chunk_size=5000
    ):
        total_bytes += len(chunk)
    delta_mb = (proc.memory_info().rss - rss_before) / (1024 * 1024)
    assert total_bytes > 1_000_000
    assert delta_mb < 15.0
    del raw, enriched
