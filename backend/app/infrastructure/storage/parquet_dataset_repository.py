"""Parquet-backed dataset cache — disk-first, LRU-evicted, memory-mapped reads."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from app.core.config import settings
from app.core.exceptions import DomainException

_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


class ParquetDatasetRepository:
    """Stores at most ``max_cached_datasets`` .parquet files (LRU by mtime).

    Never retains DataFrames in memory: every read goes through
    ``memory_map=True`` so the OS page cache does the heavy lifting.
    """

    def __init__(self, cache_dir: Path | None = None, max_cached_datasets: int = 3) -> None:
        self._cache_dir = cache_dir or settings.DATASET_CACHE_DIR
        self._max_cached = max_cached_datasets
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------
    def _path_for(self, dataset_id: str) -> Path:
        safe = _SAFE_ID.sub("_", dataset_id).strip("_") or "dataset"
        return self._cache_dir / f"{safe}.parquet"

    def _evict_lru(self) -> None:
        files = sorted(
            self._cache_dir.glob("*.parquet"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(files) >= self._max_cached:
            oldest = files.pop(0)
            try:
                oldest.unlink()
            except OSError:
                break

    # -- port ----------------------------------------------------------
    def save_dataset_parquet(self, dataset_id: str, df: pd.DataFrame) -> Path:
        self._evict_lru()
        target = self._path_for(dataset_id)
        df.to_parquet(target, engine="pyarrow", compression="snappy", index=False)
        # Touch so mtime reflects insertion order for LRU.
        target.touch(exist_ok=True)
        return target

    def _require(self, dataset_id: str) -> Path:
        target = self._path_for(dataset_id)
        if not target.exists():
            raise DomainException(
                code="DATASET_CACHE_MISS",
                message=f"Cached dataset '{dataset_id}' not found (evicted or never saved)",
                http_status=404,
            )
        return target

    def load_dataframe(
        self, dataset_id: str, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Load the full cached parquet into pandas (memory-mapped read).

        Caller is responsible for releasing the frame promptly (Phase 06 query
        path materialises one filtered window only; the full frame is freed
        before returning the response).
        """
        table = pq.read_table(self._require(dataset_id), columns=columns, memory_map=True)
        return table.to_pandas()

    def load_head(self, dataset_id: str, limit: int = 10) -> pd.DataFrame:
        table = pq.read_table(self._require(dataset_id), memory_map=True)
        return table.slice(0, limit).to_pandas()

    def read_columns_mmap(
        self,
        dataset_id: str,
        columns: list[str] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> pd.DataFrame:
        table = pq.read_table(self._require(dataset_id), columns=columns, memory_map=True)
        return table.slice(max(0, offset), max(0, limit)).to_pandas()
