"""Generate bundled sample models (offline, reproducible).

Trains a LinearRegression on samples/house_prices.csv and saves it with
embedded DataFrame feature names to samples/house_price_linreg.joblib.
Run:  python scripts/generate_sample_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent.parent  # backend/scripts/ -> repo root
FEATURES = ["area_sqm", "bedrooms", "distance_bts_km"]
TARGET = "selling_price"


def main() -> None:
    src = ROOT / "samples" / "house_prices.csv"
    dst = ROOT / "samples" / "house_price_linreg.joblib"
    df = pd.read_csv(src)
    model = LinearRegression()
    model.fit(df[FEATURES], df[TARGET])
    joblib.dump(model, dst)
    print(f"wrote {dst} (features={FEATURES}, coef={list(model.coef_)})")


if __name__ == "__main__":
    sys.exit(main())
