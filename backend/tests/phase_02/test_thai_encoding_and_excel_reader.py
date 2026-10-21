"""Reader: Thai encodings, delimiters, Excel sheets/error-tokens, parquet, empty guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DomainException
from app.infrastructure.storage.tabular_reader import TabularReader


def test_thai_cp874_csv_reads_correctly(tmp_path: Path) -> None:
    csv_content = (
        "เขตพื้นที่,ราคาประเมิน,อัตราส่วน\nจตุจักร,3500000.50,#DIV/0!\nบางรัก,8200000.00,1.25\n"
    )
    cp874_file = tmp_path / "thai_excel_cp874.csv"
    cp874_file.write_bytes(csv_content.encode("cp874"))

    result = TabularReader.read_file(cp874_file)
    assert result.detected_encoding in ("cp874", "tis-620")
    assert list(result.df.columns) == ["เขตพื้นที่", "ราคาประเมิน", "อัตราส่วน"]
    assert result.df.loc[0, "เขตพื้นที่"] == "จตุจักร"
    assert pd.isna(result.df.loc[0, "อัตราส่วน"])
    assert result.df["อัตราส่วน"].dtype == np.float32


def test_utf8_sig_and_semicolon_delimiter(tmp_path: Path) -> None:
    p = tmp_path / "semi.csv"
    p.write_bytes("a;b\n1;2\n3;4\n".encode("utf-8-sig"))
    result = TabularReader.read_file(p)
    assert result.detected_encoding == "utf-8-sig"
    assert result.detected_delimiter == ";"
    assert list(result.df.columns) == ["a", "b"]


def test_duplicate_and_whitespace_column_names(tmp_path: Path) -> None:
    p = tmp_path / "dup.csv"
    p.write_text(" area_sqm , area_sqm ,bedrooms\n55.0,55.0,2\n", encoding="utf-8")
    result = TabularReader.read_file(p)
    assert list(result.df.columns) == ["area_sqm", "area_sqm_1", "bedrooms"]


def test_excel_error_tokens_coerced_to_nan_and_sheet_picker(tmp_path: Path) -> None:
    xlsx = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame({"price": [100.0, "#DIV/0!", 300.0], "note": ["a", "b", "c"]}).to_excel(
            writer, sheet_name="Sales", index=False
        )
        pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="Meta", index=False)

    result = TabularReader.read_file(xlsx, sheet_name="Sales")
    assert result.available_sheets == ["Sales", "Meta"]
    assert result.selected_sheet == "Sales"
    assert result.df["price"].dtype == np.float32
    assert result.df["price"].isna().sum() == 1

    with pytest.raises(DomainException) as excinfo:
        TabularReader.read_file(xlsx, sheet_name="Nope")
    assert excinfo.value.code == "SHEET_NOT_FOUND"


def test_parquet_roundtrip_and_empty_guard(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    pq_path = tmp_path / "data.parquet"
    df.to_parquet(pq_path, engine="pyarrow")
    result = TabularReader.read_file(pq_path)
    assert result.file_format == "parquet"
    assert len(result.df) == 2

    empty = tmp_path / "empty.csv"
    empty.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(DomainException) as excinfo:
        TabularReader.read_file(empty)
    assert excinfo.value.code == "EMPTY_DATASET_ERROR"
    assert excinfo.value.http_status == 400
