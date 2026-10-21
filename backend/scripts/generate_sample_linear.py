# backend/scripts/generate_sample_linear.py
"""Phase 04 sample bootstrapper — house-price OLS demo data + model (offline).

Generates (seed 42, reproducible):
- ``sample_houses.csv`` (1,000 rows: area_sqm, bedrooms, distance_bts_km, selling_price)
- ``sample_house_price_linear.joblib`` (LinearRegression fitted with feature names)

Run:  ``python scripts/generate_sample_linear.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_house_price_sample(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_samples = 1000

    area_sqm = rng.uniform(25.0, 250.0, size=n_samples).astype(np.float32)
    bedrooms = rng.integers(1, 6, size=n_samples).astype(np.int32)
    distance_bts_km = rng.uniform(0.2, 18.0, size=n_samples).astype(np.float32)
    noise = rng.normal(0.0, 180000.0, size=n_samples).astype(np.float32)

    selling_price = (
        150000.0
        + (42500.0 * area_sqm)
        + (350000.0 * bedrooms)
        - (185000.0 * distance_bts_km)
        + noise
    ).astype(np.float32)

    df = pd.DataFrame({
        "area_sqm": np.round(area_sqm, 2),
        "bedrooms": bedrooms,
        "distance_bts_km": np.round(distance_bts_km, 2),
        "selling_price": np.round(selling_price, 2),
    })
    df.to_csv(output_dir / "sample_houses.csv", index=False, encoding="utf-8")

    X = df[["area_sqm", "bedrooms", "distance_bts_km"]]
    y = df["selling_price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, output_dir / "sample_house_price_linear.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 04 linear-regression samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    generate_house_price_sample(Path(args.out))
    print(f"wrote sample_houses.csv + sample_house_price_linear.joblib -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
