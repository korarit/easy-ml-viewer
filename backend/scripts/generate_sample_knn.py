# backend/scripts/generate_sample_knn.py
"""Phase 22 sample bootstrapper — KNN classifier / regressor demo data (offline).

Generates (seeded, reproducible):
- ``sample_iris_knn.csv`` + ``sample_iris_knn.joblib``
  (KNeighborsClassifier ``n_neighbors=5`` — radar sphere demo)
- ``sample_credit_knn_manhattan.csv`` + ``sample_credit_knn_manhattan.joblib``
  (KNeighborsClassifier ``metric="manhattan"`` — octahedron radar demo)
- ``sample_housing_knn.csv`` + ``sample_housing_knn.joblib``
  (KNeighborsRegressor ``n_neighbors=5`` — radar mean demo)

Run:  ``python scripts/generate_sample_knn.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _scaled_knn(estimator) -> Pipeline:
    # Keep raw CSV ranges valid while fitting in a normalized space.
    return Pipeline([("scaler", StandardScaler()), ("knn", estimator)])


def generate_iris_like(output_dir: Path) -> None:
    rng = np.random.default_rng(22)
    n = 450
    # Three gaussian blobs in 4-D (sepal/petal style), labels 0/1/2.
    centers = np.array(
        [[5.0, 3.4, 1.5, 0.2], [6.2, 2.9, 4.5, 1.4], [6.8, 3.0, 5.6, 2.0]],
        dtype=np.float32,
    )
    per = n // 3
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, c in enumerate(centers):
        xs.append((rng.normal(loc=c, scale=[0.35, 0.3, 0.35, 0.15], size=(per, 4))).astype(np.float32))
        ys.extend([k] * per)
    X = np.vstack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    df = pd.DataFrame(
        {
            "sepal_length": np.round(X[:, 0], 2),
            "sepal_width": np.round(X[:, 1], 2),
            "petal_length": np.round(X[:, 2], 2),
            "petal_width": np.round(X[:, 3], 2),
            "species": y,
        }
    )
    df.to_csv(output_dir / "sample_iris_knn.csv", index=False, encoding="utf-8")
    model = _scaled_knn(KNeighborsClassifier(n_neighbors=5, weights="distance"))
    model.fit(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]], df["species"])
    joblib.dump(model, output_dir / "sample_iris_knn.joblib", compress=3)


def generate_manhattan(output_dir: Path) -> None:
    rng = np.random.default_rng(23)
    n = 500
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
    df.to_csv(output_dir / "sample_credit_knn_manhattan.csv", index=False, encoding="utf-8")
    model = _scaled_knn(
        KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="manhattan")
    )
    model.fit(df[["feat_income", "feat_age", "feat_debt", "feat_score"]], df["default"])
    joblib.dump(model, output_dir / "sample_credit_knn_manhattan.joblib", compress=3)


def generate_regressor(output_dir: Path) -> None:
    rng = np.random.default_rng(24)
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
    df.to_csv(output_dir / "sample_housing_knn.csv", index=False, encoding="utf-8")
    model = _scaled_knn(KNeighborsRegressor(n_neighbors=5, weights="distance"))
    model.fit(
        df[["feature_rm", "feature_lstat", "feature_tax", "feature_noise"]],
        df["selling_price"],
    )
    joblib.dump(model, output_dir / "sample_housing_knn.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 22 KNN samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_iris_like(out)
    generate_manhattan(out)
    generate_regressor(out)
    print(f"wrote knn iris/manhattan/regressor samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
