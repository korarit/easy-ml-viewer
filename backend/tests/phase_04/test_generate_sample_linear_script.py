"""Phase 04 — sample generator reproducibility test."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from scripts.generate_sample_linear import generate_house_price_sample


def test_sample_generator_creates_valid_csv_and_joblib(tmp_path: Path) -> None:
    generate_house_price_sample(tmp_path)
    csv_path = tmp_path / "sample_houses.csv"
    mdl_path = tmp_path / "sample_house_price_linear.joblib"

    assert csv_path.exists() and csv_path.stat().st_size > 10_000
    assert mdl_path.exists() and mdl_path.stat().st_size > 500

    df = pd.read_csv(csv_path)
    assert len(df) == 1000
    assert list(df.columns) == ["area_sqm", "bedrooms", "distance_bts_km", "selling_price"]

    model = joblib.load(mdl_path)
    assert list(model.feature_names_in_) == ["area_sqm", "bedrooms", "distance_bts_km"]
