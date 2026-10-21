"""Endpoint + repository integration: upload JSON contract, samples, LRU eviction."""

from __future__ import annotations

import io
import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository
from app.main import app


@contextmanager
def _client_and_repo(isolated_repo: ParquetDatasetRepository):
    import app.application.dataset_use_case as uc_module
    import app.presentation.endpoints.dataset_endpoints as ep_module

    repo = isolated_repo

    def _override_use_case() -> object:
        return uc_module.DatasetIngestUseCase(repository=repo)

    app.dependency_overrides[ep_module.get_dataset_use_case] = _override_use_case
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_upload_csv_endpoint_returns_json_compliant_profiles(
    isolated_repo: ParquetDatasetRepository,
) -> None:
    with _client_and_repo(isolated_repo) as client:
        csv_bytes = (
            " area_sqm , area_sqm ,bedrooms,district,price\n"
            "55.0,55.0,2,Chatuchak,3200000\n"
            "80.5,80.5,#N/A,Bangrak,inf\n"
            "120.0,120.0,4,Sathorn,9500000\n"
        ).encode("utf-8-sig")

        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test_houses.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert resp.status_code == 200

        raw_text = resp.text
        assert "NaN" not in raw_text and "Infinity" not in raw_text
        payload = json.loads(raw_text)

        assert payload["row_count"] == 3
        assert payload["detected_encoding"] == "utf-8-sig"
        assert payload["inf_sanitized_count"] == 1
        col_names = [c["name"] for c in payload["columns"]]
        assert col_names[0] == "area_sqm"
        assert col_names[1] == "area_sqm_1"
        assert len(payload["sample_rows"]) == 3
        assert "parquet_cache_path" not in payload


def test_load_sample_and_empty_rejection(
    isolated_repo: ParquetDatasetRepository,
) -> None:
    with _client_and_repo(isolated_repo) as client:
        resp = client.post("/api/v1/datasets/load-sample/house_prices")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["row_count"] == 30
        assert payload["column_count"] == 5

        resp = client.post("/api/v1/datasets/load-sample/does_not_exist")
        assert resp.status_code == 404

        empty = io.BytesIO("a,b\n".encode("utf-8"))
        resp = client.post(
            "/api/v1/datasets/upload", files={"file": ("empty.csv", empty, "text/csv")}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "EMPTY_DATASET_ERROR"


def test_dataframe_evicted_from_ram_after_parquet_cache(
    tmp_path, isolated_repo: ParquetDatasetRepository
) -> None:
    repo = isolated_repo
    for idx in range(4):
        df = pd.DataFrame(
            {
                "x": np.random.rand(50_000).astype(np.float32),
                "y": np.random.rand(50_000).astype(np.float32),
            }
        )
        repo.save_dataset_parquet(dataset_id=f"ds_{idx}", df=df)
        del df

    cached_files = list(isolated_repo._cache_dir.glob("*.parquet"))
    assert len(cached_files) == 3
    assert not (isolated_repo._cache_dir / "ds_0.parquet").exists()

    table_slice = repo.read_columns_mmap("ds_3", columns=["x"], offset=0, limit=100)
    assert table_slice.shape == (100, 1)
    assert table_slice["x"].dtype == np.float32
