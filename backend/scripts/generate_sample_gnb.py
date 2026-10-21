# backend/scripts/generate_sample_gnb.py
"""Phase 24 sample bootstrapper — GaussianNB demo data (offline).

Generates (seeded, reproducible):
- ``sample_iris_gnb.csv`` + ``sample_iris_gnb.joblib``
  (GaussianNB on 3 iris-like blobs — multi-hill demo)
- ``sample_credit_gnb.csv`` + ``sample_credit_gnb.joblib``
  (GaussianNB binary — log-odds waterfall demo)
- ``sample_credit_gnb_priors.csv`` + ``sample_credit_gnb_priors.joblib``
  (GaussianNB with custom ``priors=[0.2, 0.8]`` — prior-vs-count demo)

Run:  ``python scripts/generate_sample_gnb.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _scaled_gnb(**kwargs) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("gnb", GaussianNB(**kwargs))])


def generate_iris(output_dir: Path) -> None:
    rng = np.random.default_rng(24)
    centers = np.array(
        [[5.0, 3.4, 1.5, 0.2], [6.0, 2.8, 4.3, 1.3], [6.6, 3.0, 5.5, 2.0]],
        dtype=np.float32,
    )
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, c in enumerate(centers):
        xs.append(rng.normal(loc=c, scale=[0.35, 0.30, 0.35, 0.15], size=(150, 4)).astype(np.float32))
        ys.extend([k] * 150)
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
    df.to_csv(output_dir / "sample_iris_gnb.csv", index=False, encoding="utf-8")
    model = _scaled_gnb()
    model.fit(
        df[["sepal_length", "sepal_width", "petal_length", "petal_width"]],
        df["species"],
    )
    joblib.dump(model, output_dir / "sample_iris_gnb.joblib", compress=3)


def generate_binary(output_dir: Path, priors: list[float] | None, suffix: str) -> None:
    rng = np.random.default_rng(25 if priors is None else 26)
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
    df.to_csv(output_dir / f"sample_credit_gnb{suffix}.csv", index=False, encoding="utf-8")
    kwargs = {"priors": priors} if priors is not None else {}
    model = _scaled_gnb(**kwargs)
    model.fit(df[["feat_income", "feat_age", "feat_debt", "feat_score"]], df["default"])
    joblib.dump(model, output_dir / f"sample_credit_gnb{suffix}.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 24 GaussianNB samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_iris(out)
    generate_binary(out, None, "")
    generate_binary(out, [0.2, 0.8], "_priors")
    print(f"wrote gnb iris/binary/priors samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
