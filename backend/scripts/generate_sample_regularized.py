# backend/scripts/generate_sample_regularized.py
"""Phase 19 sample bootstrapper — Ridge/Lasso/RidgeClassifier demo data (offline).

Generates (seeded, reproducible):
- ``sample_housing_ridge.csv`` + ``sample_housing_ridge.joblib``
- ``sample_sparse_lasso.csv`` + ``sample_sparse_lasso.joblib``
- ``sample_credit_ridge_clf.csv`` + ``sample_credit_ridge_clf.joblib``

Run:  ``python scripts/generate_sample_regularized.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, RidgeClassifier


def generate_ridge(output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    n = 500
    area = rng.uniform(25.0, 250.0, size=n).astype(np.float32)
    rooms = rng.integers(1, 6, size=n).astype(np.float32)
    age = rng.uniform(0.0, 40.0, size=n).astype(np.float32)
    noise = rng.normal(0.0, 150_000.0, size=n).astype(np.float32)
    price = (150_000.0 + 42_500.0 * area + 300_000.0 * rooms - 120_000.0 * age + noise).astype(np.float32)
    df = pd.DataFrame(
        {
            "area_sqft": np.round(area, 2),
            "rooms": rooms.astype(np.int32),
            "age_years": np.round(age, 1),
            "selling_price": np.round(price, 2),
        }
    )
    df.to_csv(output_dir / "sample_housing_ridge.csv", index=False, encoding="utf-8")
    model = Ridge(alpha=1.0)
    model.fit(df[["area_sqft", "rooms", "age_years"]], df["selling_price"])
    joblib.dump(model, output_dir / "sample_housing_ridge.joblib", compress=3)


def generate_lasso(output_dir: Path) -> None:
    rng = np.random.default_rng(19)
    n = 500
    area = rng.uniform(20.0, 200.0, size=n).astype(np.float32)
    rooms = rng.uniform(1.0, 6.0, size=n).astype(np.float32)
    noise_1 = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    noise_2 = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    y = (180.0 + 0.84 * area + 12.5 * rooms + rng.normal(0.0, 8.0, size=n)).astype(np.float32)
    df = pd.DataFrame(
        {
            "area_sqft": np.round(area, 2),
            "rooms": np.round(rooms, 2),
            "noise_1": np.round(noise_1, 3),
            "noise_2": np.round(noise_2, 3),
            "rent_price": np.round(y, 2),
        }
    )
    df.to_csv(output_dir / "sample_sparse_lasso.csv", index=False, encoding="utf-8")
    model = Lasso(alpha=2.5, max_iter=5000)
    model.fit(df[["area_sqft", "rooms", "noise_1", "noise_2"]], df["rent_price"])
    joblib.dump(model, output_dir / "sample_sparse_lasso.joblib", compress=3)


def generate_ridge_clf(output_dir: Path) -> None:
    rng = np.random.default_rng(7)
    n = 400
    income = rng.uniform(20_000.0, 150_000.0, size=n).astype(np.float32)
    age = rng.uniform(18.0, 70.0, size=n).astype(np.float32)
    debt = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
    score = rng.uniform(300.0, 850.0, size=n).astype(np.float32)
    logit = -1.0 + 0.00002 * income - 0.03 * age + 4.0 * debt - 0.004 * score
    y = (logit > 0).astype(np.int32)
    df = pd.DataFrame(
        {
            "feat_income": np.round(income, 2),
            "feat_age": np.round(age, 1),
            "feat_debt": np.round(debt, 3),
            "feat_score": np.round(score, 0).astype(np.int32),
            "default": y,
        }
    )
    df.to_csv(output_dir / "sample_credit_ridge_clf.csv", index=False, encoding="utf-8")
    model = RidgeClassifier(alpha=1.0)
    model.fit(df[["feat_income", "feat_age", "feat_debt", "feat_score"]], df["default"])
    joblib.dump(model, output_dir / "sample_credit_ridge_clf.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 19 ridge/lasso samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_ridge(out)
    generate_lasso(out)
    generate_ridge_clf(out)
    print(f"wrote ridge + lasso + ridge-clf samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
