# backend/scripts/generate_sample_pipeline.py
"""Phase 37 sample bootstrapper — sklearn Pipeline / ColumnTransformer demo data.

Generates (seeded, reproducible):
- ``sample_pipe_numeric.joblib`` / ``sample_pipe_numeric.csv``
  (SimpleImputer median + StandardScaler + LogisticRegression, 4 numeric cols)
- ``sample_pipe_mixed.joblib`` / ``sample_pipe_mixed.csv``
  (ColumnTransformer [num imputer+scaler | cat imputer+OHE] + RandomForest)

Run:  ``python scripts/generate_sample_pipeline.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def generate_numeric(output_dir: Path) -> None:
    rng = np.random.default_rng(37)
    n = 500
    income = rng.uniform(15.0, 150.0, size=n)
    age = rng.uniform(20.0, 65.0, size=n)
    tenure = rng.uniform(0.0, 72.0, size=n)
    balance = rng.normal(50.0, 20.0, size=n)
    df = pd.DataFrame({"income": income, "age": age, "tenure": tenure, "balance": balance})
    logit = -0.04 * df["income"] + 0.06 * df["age"] + 0.03 * df["tenure"] - 0.02 * df["balance"]
    y = (logit + rng.normal(0, 1.5, size=n) > 0).astype(int)
    # Inject ~5% NaNs per column (Imputer path).
    for col in ("income", "age", "tenure", "balance"):
        df.loc[df.sample(frac=0.05, random_state=hash(col) % 10_000).index, col] = np.nan
    df["approved"] = y
    df.to_csv(output_dir / "sample_pipe_numeric.csv", index=False, encoding="utf-8")
    feats = ["income", "age", "tenure", "balance"]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(df[feats], y)
    joblib.dump(pipe, output_dir / "sample_pipe_numeric.joblib", compress=3)


def generate_mixed(output_dir: Path) -> None:
    rng = np.random.default_rng(38)
    n = 600
    income = rng.uniform(15.0, 120.0, size=n)
    age = rng.uniform(20.0, 65.0, size=n)
    city = rng.choice(["Bangkok", "Chiang Mai", "Phuket"], size=n)
    housing = rng.choice(["Rent", "Own", "Mortgage"], size=n)
    df = pd.DataFrame({"income": income, "age": age, "city": city, "housing": housing})
    y = (
        ((df["income"] > 55).astype(int) ^ (df["city"] == "Bangkok").astype(int))
        ^ ((df["age"] > 45).astype(int) & (df["housing"] == "Rent").astype(int))
    ).astype(int).values
    df.loc[df.sample(frac=0.04, random_state=38).index, "income"] = np.nan
    df.loc[df.sample(frac=0.04, random_state=39).index, "age"] = np.nan
    df.loc[df.sample(frac=0.03, random_state=40).index, "city"] = np.nan
    df["approved"] = y
    df.to_csv(output_dir / "sample_pipe_mixed.csv", index=False, encoding="utf-8")
    num_cols = ["income", "age"]
    cat_cols = ["city", "housing"]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), cat_cols),
    ])
    pipe = Pipeline([
        ("preprocessor", pre),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=38, n_jobs=1)),
    ])
    pipe.fit(df[num_cols + cat_cols], y)
    joblib.dump(pipe, output_dir / "sample_pipe_mixed.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 37 pipeline samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_numeric(out)
    generate_mixed(out)
    print(f"wrote pipeline numeric/mixed samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
