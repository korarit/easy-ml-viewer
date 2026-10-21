"""Phase 06 — API integration tests for /datasets/{id}/query + /resolve-selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.application.dataset_use_case import DatasetIngestUseCase
from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository


def _seed_dataset(client: TestClient) -> str:
    df = pd.DataFrame(
        {
            "area_sqm": np.array([45.0, 120.0, 65.0, 95.0, 80.0], dtype=np.float32),
            "bedrooms": np.array([1, 4, 2, 3, 2], dtype=np.int32),
        }
    )
    use_case = DatasetIngestUseCase(repository=ParquetDatasetRepository())
    entity = use_case._build_entity(
        type("R", (), {
            "df": df,
            "file_format": "csv",
            "detected_encoding": "utf-8",
            "detected_delimiter": ",",
            "selected_sheet": None,
            "available_sheets": [],
            "original_ram_mb": 0.01,
            "optimized_ram_mb": 0.005,
            "inf_sanitized_count": 0,
        })(),
        filename="phase06.csv",
    )
    return entity.dataset_id


def test_query_endpoint_roundtrip(client: TestClient) -> None:
    dataset_id = _seed_dataset(client)
    res = client.post(
        f"/api/v1/datasets/{dataset_id}/query",
        json={
            "filters": [{"column": "area_sqm", "operator": "gte", "value": 60.0}],
            "sort_by": "area_sqm",
            "sort_ascending": True,
            "offset": 0,
            "limit": 2,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["total_dataset_rows"] == 5
    assert body["total_filtered_rows"] == 4
    assert [r["_original_row_index"] for r in body["rows"]] == [2, 4]
    assert body["matching_row_indices_preview"] == [2, 4, 3, 1]


def test_resolve_selection_endpoint_modes(client: TestClient) -> None:
    dataset_id = _seed_dataset(client)
    res = client.post(
        f"/api/v1/datasets/{dataset_id}/resolve-selection",
        json={"mode": "first_n", "n": 2},
    )
    assert res.status_code == 200, res.text
    assert res.json()["row_indices"] == [0, 1]

    res = client.post(
        f"/api/v1/datasets/{dataset_id}/resolve-selection",
        json={"mode": "range", "range_start": 1, "range_end": 4, "range_step": 2},
    )
    assert res.status_code == 200, res.text
    assert res.json()["row_indices"] == [1, 3]


def test_query_endpoint_cache_miss_404(client: TestClient) -> None:
    res = client.post("/api/v1/datasets/ds_missing/query", json={"offset": 0, "limit": 10})
    assert res.status_code == 404
