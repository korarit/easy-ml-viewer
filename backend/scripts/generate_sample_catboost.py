# backend/scripts/generate_sample_catboost.py
"""Phase 35 sample bootstrapper — CatBoost symmetric oblivious-tree demo data (offline).

Generates (seeded, reproducible):
- ``sample_binary_catboost.joblib`` (CatBoostClassifier with native categoricals)
- ``sample_binary_catboost.cbm`` (native CatBoost binary artifact, same family)
- ``sample_multi_catboost.joblib`` (CatBoostClassifier 3-class, numeric)
- ``sample_reg_catboost.joblib`` (CatBoostRegressor, numeric)

Run:  ``python scripts/generate_sample_catboost.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


def generate_binary(output_dir: Path) -> None:
    rng = np.random.default_rng(71)
    income = rng.uniform(20.0, 80.0, size=500).astype(float)
    age = rng.uniform(18.0, 70.0, size=500).astype(float)
    city = rng.choice(["Bangkok", "Chiang Mai", "Phuket"], size=500)
    housing = rng.choice(["Rent", "Own", "Mortgage"], size=500)
    df = pd.DataFrame({"income": income, "age": age, "city": city, "housing": housing})
    y = (((df["income"] > 50).astype(int) ^ (df["city"] == "Bangkok").astype(int)) ^ (
        (df["age"] > 45).astype(int) & (df["housing"] == "Rent").astype(int))).astype(int).values
    # Inject NaNs: numeric stays NaN, categorical NaN tests the __MISSING__ path.
    # The CSV keeps raw NaNs; the training frame fills categoricals (CatBoost
    # requires str/int categories at fit time).
    df.loc[df.sample(frac=0.03, random_state=71).index, "income"] = np.nan
    df.loc[df.sample(frac=0.03, random_state=72).index, "city"] = np.nan
    df.to_csv(output_dir / "sample_binary_catboost.csv", index=False, encoding="utf-8")
    train_df = df.copy()
    for c in ("city", "housing"):
        train_df[c] = train_df[c].fillna("__MISSING__").astype(str)
    model = CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.15, verbose=False, random_seed=71
    ).fit(train_df, y, cat_features=["city", "housing"])
    joblib.dump(model, output_dir / "sample_binary_catboost.joblib", compress=3)
    model.save_model(str(output_dir / "sample_binary_catboost.cbm"), format="cbm")


def generate_multi(output_dir: Path) -> None:
    rng = np.random.default_rng(72)
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
    df.to_csv(output_dir / "sample_multi_catboost.csv", index=False, encoding="utf-8")
    feats = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    model = CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.12, verbose=False,
        random_seed=72, loss_function="MultiClass",
    ).fit(df[feats], df["species"])
    joblib.dump(model, output_dir / "sample_multi_catboost.joblib", compress=3)


def generate_reg(output_dir: Path) -> None:
    rng = np.random.default_rng(73)
    n = 400
    rooms = rng.integers(1, 7, size=n).astype(np.float32)
    area = rng.uniform(30.0, 200.0, size=n).astype(np.float32)
    age = rng.uniform(0.0, 50.0, size=n).astype(np.float32)
    price = (rooms * 28.0 + area * 1.6 - age * 0.9 + rng.normal(0, 12.0, size=n)).astype(np.float32)
    df = pd.DataFrame({"rooms": rooms, "area": area, "age": age, "price": price})
    df.to_csv(output_dir / "sample_reg_catboost.csv", index=False, encoding="utf-8")
    model = CatBoostRegressor(
        iterations=100, depth=4, learning_rate=0.12, verbose=False, random_seed=73
    ).fit(df[["rooms", "area", "age"]], df["price"])
    joblib.dump(model, output_dir / "sample_reg_catboost.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 35 CatBoost samples")
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
    print(f"wrote catboost binary/multi/reg samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
