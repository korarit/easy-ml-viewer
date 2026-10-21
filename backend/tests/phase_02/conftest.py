"""Shared fixtures for Phase 02 dataset tests (isolated cache dir)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository


@pytest.fixture()
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParquetDatasetRepository:
    cache = tmp_path / "cache" / "datasets"
    cache.mkdir(parents=True)
    repo = ParquetDatasetRepository(cache_dir=cache, max_cached_datasets=3)
    # Route the app-level use-case at the same isolated dir.
    import app.application.dataset_use_case as uc_module
    import app.infrastructure.storage.parquet_dataset_repository as repo_module

    monkeypatch.setattr(repo_module.settings, "DATASET_CACHE_DIR", cache)
    monkeypatch.setattr(uc_module.settings, "DATASET_CACHE_DIR", cache)
    return repo
