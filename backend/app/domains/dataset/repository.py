"""Dataset repository port (DDD repository interface)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class DatasetRepository(Protocol):
    def save_dataset_parquet(self, dataset_id: str, df: pd.DataFrame) -> Path:
        ...

    def load_head(self, dataset_id: str, limit: int = 10) -> pd.DataFrame:
        ...

    def read_columns_mmap(
        self, dataset_id: str, columns: list[str] | None = None, offset: int = 0, limit: int = 100
    ) -> pd.DataFrame:
        ...

    def load_dataframe(self, dataset_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        ...
