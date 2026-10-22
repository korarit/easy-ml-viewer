# backend/scripts/generate_sample_sklearn_gbm.py
"""Phase 31 sample bootstrapper — sklearn GBM staged-residual demo data (offline).

Generates (seeded, reproducible):
- ``sample_binary_sklearn_gbm.*`` (GradientBoostingClassifier/log_loss on 2-D blobs)
- ``sample_multi_sklearn_gbm.*`` (GradientBoostingClassifier on 3-class data)
- ``sample_reg_sklearn_gbm.*`` (GradientBoostingRegressor on housing-like data)

Run:  ``python scripts/generate_sample_sklearn_gbm.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


def generate_binary(output_dir: Path) -> None:
    rng = np.random.default_rng(41)
    a = rng.normal(size=(250, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(250, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 250 + [1] * 250, dtype=np.int32)
    df = pd.DataFrame({"feat_income": X[:, 0], "feat_age": X[:, 1], "label": y})
    df.to_csv(output_dir / "sample_binary_sklearn_gbm.csv", index=False, encoding="utf-8")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=41
    ).fit(df[["feat_income", "feat_age"]], df["label"])
    joblib.dump(model, output_dir / "sample_binary_sklearn_gbm.joblib", compress=3)


def generate_multi(output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    centers = np.array([[5.0, 3.4, 1.5, 0.2], [6.0, 2.8, 4.3, 1.3], [6.6, 3.0, 5.5, 2.0]])
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, c in enumerate(centers):
        xs.append(rng.normal(loc=c, scale=[0.35, 0.30, 0.35, 0.15], size=(120, 4)).astype(np.float32))
        ys.extend([k] * 120)
    X = np.vstack(xs)
    df = pd.DataFrame(
        {
            "sepal_length": X[:, 0],
            "sepal_width": X[:, 1],
            "petal_length": X[:, 2],
            "petal_width": X[:, 3],
            "species": ys,
        }
    )
    df.to_csv(output_dir / "sample_multi_sklearn_gbm.csv", index=False, encoding="utf-8")
    feats = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    ).fit(df[feats], df["species"])
    joblib.dump(model, output_dir / "sample_multi_sklearn_gbm.joblib", compress=3)


def generate_reg(output_dir: Path) -> None:
    rng = np.random.default_rng(43)
    n = 400
    rooms = rng.integers(1, 7, size=n).astype(np.float32)
    area = rng.uniform(30.0, 200.0, size=n).astype(np.float32)
    age = rng.uniform(0.0, 50.0, size=n).astype(np.float32)
    price = (rooms * 28.0 + area * 1.6 - age * 0.9 + rng.normal(0, 12.0, size=n)).astype(np.float32)
    df = pd.DataFrame({"rooms": rooms, "area": area, "age": age, "price": price})
    df.to_csv(output_dir / "sample_reg_sklearn_gbm.csv", index=False, encoding="utf-8")
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=43
    ).fit(df[["rooms", "area", "age"]], df["price"])
    joblib.dump(model, output_dir / "sample_reg_sklearn_gbm.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 31 sklearn GBM samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_binary(out)
    generate_multi(out)
    generate_reg(out)
    print(f"wrote sklearn-gbm binary/multi/reg samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
