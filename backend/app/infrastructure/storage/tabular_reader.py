"""Tabular file reader — CSV/TSV/XLSX/XLS/Parquet with Thai encoding support."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.core.exceptions import DomainException
from app.core.memory_guard import MemoryGuard
from app.domains.dataset.services import normalize_column_names

_EXCEL_ERROR_TOKENS = [
    "#DIV/0!",
    "#VALUE!",
    "#N/A",
    "#REF!",
    "#NAME?",
    "#NUM!",
    "#NULL!",
    "NULL",
    "null",
    "N/A",
    "#N/A N/A",
]
_CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620", "latin1"]
_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
_ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
_MAX_SNIFF_BYTES = 65_536


@dataclass
class TabularReadResult:
    df: pd.DataFrame
    file_format: str
    detected_encoding: str
    detected_delimiter: str | None
    available_sheets: list[str] = field(default_factory=list)
    selected_sheet: str | None = None
    # MemoryGuard stats: reader downcasts immediately so every consumer
    # (endpoints, tests, future phases) always sees float32/int32 frames.
    original_ram_mb: float = 0.0
    optimized_ram_mb: float = 0.0
    inf_sanitized_count: int = 0


def detect_encoding(raw_head: bytes) -> str:
    """Detect encoding from the first 64 KB. Thai-aware fallback chain."""
    if raw_head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "cp874", "tis-620"):
        try:
            raw_head.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin1"  # single-byte superset: always decodes


def detect_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(_CANDIDATE_DELIMITERS))
        if dialect.delimiter in _CANDIDATE_DELIMITERS:
            return dialect.delimiter
    except (csv.Error, ValueError, TypeError):
        pass
    # Frequency fallback on the header line.
    header = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    return max(_CANDIDATE_DELIMITERS, key=lambda d: header.count(d))


class TabularReader:
    """Reads tabular files into DataFrames with sanitized column names."""

    @staticmethod
    def read_file(path: str | Path, sheet_name: str | None = None) -> TabularReadResult:
        src = Path(path)
        if not src.exists():
            raise DomainException(
                code="DATASET_FILE_NOT_FOUND",
                message=f"Dataset file not found: {src.name}",
                http_status=404,
            )
        suffix = src.suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise DomainException(
                code="UNSUPPORTED_DATASET_FORMAT",
                message=f"Unsupported dataset format '{src.suffix}' (allowed: csv, tsv, xlsx, xls, parquet)",
                details={"suffix": src.suffix},
                http_status=415,
            )
        MemoryGuard.assert_can_ingest_file(src.stat().st_size)

        if suffix == ".parquet":
            return TabularReader._read_parquet(src)
        if suffix in (".xlsx", ".xls"):
            return TabularReader._read_excel(src, sheet_name)
        return TabularReader._read_delimited(src)

    # -- internals -----------------------------------------------------
    @staticmethod
    def _guard_empty(df: pd.DataFrame, filename: str) -> None:
        if len(df) == 0 or len(df.columns) == 0:
            raise DomainException(
                code="EMPTY_DATASET_ERROR",
                message=f"Dataset '{filename}' is empty (header only or 0 rows)",
                http_status=400,
            )

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = normalize_column_names([str(c) for c in df.columns])
        return df

    @staticmethod
    def _finalize(
        df: pd.DataFrame,
        file_format: str,
        encoding: str,
        delimiter: str | None,
        filename: str,
        sheets: list[str] | None = None,
        selected: str | None = None,
    ) -> TabularReadResult:
        TabularReader._guard_empty(df, filename)
        df = TabularReader._normalize(df)
        df, orig_mb, opt_mb, inf_count = MemoryGuard.downcast_dataframe(df)
        return TabularReadResult(
            df=df,
            file_format=file_format,
            detected_encoding=encoding,
            detected_delimiter=delimiter,
            available_sheets=sheets or [],
            selected_sheet=selected,
            original_ram_mb=round(orig_mb, 3),
            optimized_ram_mb=round(opt_mb, 3),
            inf_sanitized_count=inf_count,
        )

    @staticmethod
    def _read_delimited(src: Path) -> TabularReadResult:
        raw_head = src.read_bytes()[:_MAX_SNIFF_BYTES]
        encoding = detect_encoding(raw_head)
        try:
            sample_text = raw_head.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            encoding = "latin1"
            sample_text = raw_head.decode(encoding)
        delimiter = detect_delimiter(sample_text)
        df = pd.read_csv(
            src,
            encoding=encoding,
            sep=delimiter,
            na_values=_EXCEL_ERROR_TOKENS,
            keep_default_na=True,
            engine="python",
        )
        return TabularReader._finalize(
            df,
            file_format="tsv" if delimiter == "\t" else "csv",
            encoding=encoding,
            delimiter=delimiter,
            filename=src.name,
        )

    @staticmethod
    def _read_excel(src: Path, sheet_name: str | None) -> TabularReadResult:
        import openpyxl  # noqa: F401  (ensures the engine is present)

        workbook = pd.ExcelFile(src, engine="openpyxl")
        sheets = list(workbook.sheet_names)
        selected = sheet_name or sheets[0]
        if selected not in sheets:
            raise DomainException(
                code="SHEET_NOT_FOUND",
                message=f"Sheet '{selected}' not found in {src.name}",
                details={"available_sheets": sheets},
                http_status=400,
            )
        df = pd.read_excel(
            workbook,
            sheet_name=selected,
            na_values=_EXCEL_ERROR_TOKENS,
            keep_default_na=True,
            engine="openpyxl",
        )
        return TabularReader._finalize(
            df,
            file_format="xlsx",
            encoding="utf-8",
            delimiter=None,
            filename=src.name,
            sheets=sheets,
            selected=selected,
        )

    @staticmethod
    def _read_parquet(src: Path) -> TabularReadResult:
        df = pd.read_parquet(src)
        return TabularReader._finalize(
            df,
            file_format="parquet",
            encoding="utf-8",
            delimiter=None,
            filename=src.name,
        )
