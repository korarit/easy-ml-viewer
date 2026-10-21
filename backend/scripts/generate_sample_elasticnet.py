# backend/scripts/generate_sample_elasticnet.py
"""Phase 20 sample bootstrapper — ElasticNet / MultiTaskElasticNet demo data (offline).

Generates (seeded, reproducible):
- ``sample_housing_elasticnet.csv`` + ``sample_housing_elasticnet.joblib``
- ``sample_multitask_elasticnet.csv`` + ``sample_multitask_elasticnet.joblib``

Run:  ``python scripts/generate_sample_elasticnet.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet


def generate_elasticnet(output_dir: Path) -> None:
    rng = np.random.default_rng(20)
    n = 500
    rm = rng.uniform(3.0, 9.0, size=n).astype(np.float32)
    lstat = rng.uniform(1.0, 30.0, size=n).astype(np.float32)
    tax = rng.uniform(180.0, 720.0, size=n).astype(np.float32)
    noise_feat = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    y = (25.0 + 6.2 * rm - 1.9 * lstat - 0.02 * tax + rng.normal(0.0, 3.0, size=n)).astype(np.float32)
    df = pd.DataFrame(
        {
            "feature_rm": np.round(rm, 2),
            "feature_lstat": np.round(lstat, 2),
            "feature_tax": np.round(tax, 1),
            "feature_noise": np.round(noise_feat, 3),
            "selling_price": np.round(y, 2),
        }
    )
    df.to_csv(output_dir / "sample_housing_elasticnet.csv", index=False, encoding="utf-8")
    model = ElasticNet(alpha=0.12, l1_ratio=0.7, max_iter=5000)
    model.fit(
        df[["feature_rm", "feature_lstat", "feature_tax", "feature_noise"]],
        df["selling_price"],
    )
    joblib.dump(model, output_dir / "sample_housing_elasticnet.joblib", compress=3)


def generate_multitask(output_dir: Path) -> None:
    rng = np.random.default_rng(21)
    n = 500
    rm = rng.uniform(3.0, 9.0, size=n).astype(np.float32)
    lstat = rng.uniform(1.0, 30.0, size=n).astype(np.float32)
    tax = rng.uniform(180.0, 720.0, size=n).astype(np.float32)
    noise_feat = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    price = (25.0 + 6.2 * rm - 1.9 * lstat - 0.02 * tax + rng.normal(0.0, 3.0, size=n)).astype(np.float32)
    rent_yield = (3.0 + 0.4 * rm - 0.05 * lstat + rng.normal(0.0, 0.4, size=n)).astype(np.float32)
    df = pd.DataFrame(
        {
            "feature_rm": np.round(rm, 2),
            "feature_lstat": np.round(lstat, 2),
            "feature_tax": np.round(tax, 1),
            "feature_noise": np.round(noise_feat, 3),
            "selling_price": np.round(price, 2),
            "rent_yield": np.round(rent_yield, 2),
        }
    )
    df.to_csv(output_dir / "sample_multitask_elasticnet.csv", index=False, encoding="utf-8")
    model = MultiTaskElasticNet(alpha=0.5, l1_ratio=0.6, max_iter=5000)
    model.fit(
        df[["feature_rm", "feature_lstat", "feature_tax", "feature_noise"]],
        df[["selling_price", "rent_yield"]],
    )
    joblib.dump(model, output_dir / "sample_multitask_elasticnet.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 20 elasticnet samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_elasticnet(out)
    generate_multitask(out)
    print(f"wrote elasticnet + multitask-elasticnet samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
