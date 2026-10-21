"""Dataset application use-cases — upload ingestion + sample loading."""

from __future__ import annotations

import gc
import tempfile
import uuid
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.core.exceptions import DomainException
from app.domains.dataset.entities import DatasetEntity
from app.domains.dataset.services import profile_dataframe
from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository
from app.infrastructure.storage.tabular_reader import TabularReader

_ALLOWED_SAMPLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}


class DatasetIngestUseCase:
    def __init__(
        self,
        reader: TabularReader | None = None,
        repository: ParquetDatasetRepository | None = None,
    ) -> None:
        self._reader = reader or TabularReader()
        self._repository = repository or ParquetDatasetRepository()

    # -- public API ----------------------------------------------------
    def ingest_upload_contents(
        self, contents: bytes, filename: str, sheet_name: str | None = None
    ) -> DatasetEntity:
        max_bytes = int(settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        if len(contents) > max_bytes:
            raise DomainException(
                code="UPLOAD_TOO_LARGE",
                message=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit",
                details={"size_bytes": len(contents)},
                http_status=413,
            )
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(
            suffix=suffix or ".csv", delete=False, dir=str(settings.DATASET_CACHE_DIR)
        ) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        try:
            return self.ingest_file(tmp_path, filename=filename, sheet_name=sheet_name)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def ingest_file(
        self, path: str | Path, filename: str | None = None, sheet_name: str | None = None
    ) -> DatasetEntity:
        # Reader already applies MemoryGuard.downcast_dataframe internally.
        result = self._reader.read_file(path, sheet_name=sheet_name)
        return self._build_entity(result, filename or Path(path).name)

    def _build_entity(self, result: object, filename: str) -> DatasetEntity:
        df: pd.DataFrame = result.df  # type: ignore[union-attr]
        columns, sample_rows = profile_dataframe(df)
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        parquet_path = self._repository.save_dataset_parquet(dataset_id, df)
        entity = DatasetEntity(
            dataset_id=dataset_id,
            filename=filename,
            file_format=result.file_format,  # type: ignore[union-attr]
            detected_encoding=result.detected_encoding,  # type: ignore[union-attr]
            detected_delimiter=result.detected_delimiter,  # type: ignore[union-attr]
            selected_sheet=result.selected_sheet,  # type: ignore[union-attr]
            available_sheets=result.available_sheets,  # type: ignore[union-attr]
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            original_ram_mb=result.original_ram_mb,  # type: ignore[union-attr]
            optimized_ram_mb=result.optimized_ram_mb,  # type: ignore[union-attr]
            inf_sanitized_count=result.inf_sanitized_count,  # type: ignore[union-attr]
            columns=columns,
            parquet_cache_path=parquet_path,
            sample_rows=sample_rows,
        )
        del df
        gc.collect()
        return entity

    def load_sample(self, sample_name: str) -> DatasetEntity:
        safe = Path(sample_name).name
        candidate: Path | None = None
        for suffix in _ALLOWED_SAMPLE_SUFFIXES:
            probe = settings.SAMPLE_DIR / f"{safe}{suffix}"
            if probe.exists():
                candidate = probe
                break
            probe2 = settings.SAMPLE_DIR / safe
            if probe2.exists() and probe2.suffix.lower() in _ALLOWED_SAMPLE_SUFFIXES:
                candidate = probe2
                break
        if candidate is None:
            available = sorted(
                p.name
                for p in settings.SAMPLE_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in _ALLOWED_SAMPLE_SUFFIXES
            ) if settings.SAMPLE_DIR.exists() else []
            raise DomainException(
                code="SAMPLE_NOT_FOUND",
                message=f"Sample dataset '{sample_name}' not found",
                details={"available_samples": available},
                http_status=404,
            )
        result = self._reader.read_file(candidate)
        return self._build_entity(result, candidate.name)
