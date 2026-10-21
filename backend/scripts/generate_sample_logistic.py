# backend/scripts/generate_sample_logistic.py
"""Phase 18 sample bootstrapper — binary + multiclass logistic demo data (offline).

Generates (seed 42, reproducible):
- ``sample_logistic_binary.csv`` (400 rows) + ``sample_credit_default_logistic.joblib``
- ``sample_logistic_multiclass.csv`` (450 rows) + ``sample_iris_logistic.joblib``

Run:  ``python scripts/generate_sample_logistic.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def generate_binary(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 400
    income = rng.uniform(20_000.0, 150_000.0, size=n).astype(np.float32)
    age = rng.uniform(18.0, 70.0, size=n).astype(np.float32)
    debt_ratio = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
    score = rng.uniform(300.0, 850.0, size=n).astype(np.float32)

    logit = (
        -1.2
        + 0.00002 * income
        - 0.03 * age
        + 4.5 * debt_ratio
        - 0.004 * score
    )
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(np.int32)

    df = pd.DataFrame(
        {
            "feat_income": np.round(income, 2),
            "feat_age": np.round(age, 1),
            "feat_debt": np.round(debt_ratio, 3),
            "feat_score": np.round(score, 0).astype(np.int32),
            "default": y,
        }
    )
    df.to_csv(output_dir / "sample_logistic_binary.csv", index=False, encoding="utf-8")

    model = LogisticRegression(max_iter=1000)
    model.fit(
        df[["feat_income", "feat_age", "feat_debt", "feat_score"]],
        df["default"],
    )
    joblib.dump(model, output_dir / "sample_credit_default_logistic.joblib", compress=3)


def generate_multiclass(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_per = 150
    centers = np.array(
        [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 1.0, 0.5], [-3.0, 2.0, -1.0, 1.5]],
        dtype=np.float32,
    )
    parts = []
    labels = []
    for k, c in enumerate(centers):
        parts.append(c + rng.normal(0.0, 0.9, size=(n_per, 4)).astype(np.float32))
        labels.append(np.full((n_per,), k, dtype=np.int32))
    X = np.concatenate(parts, axis=0)
    y = np.concatenate(labels, axis=0)

    df = pd.DataFrame(
        X, columns=["feat_income", "feat_age", "feat_debt", "feat_score"]
    )
    df["segment"] = y
    df.to_csv(output_dir / "sample_logistic_multiclass.csv", index=False, encoding="utf-8")

    model = LogisticRegression(max_iter=1000)
    model.fit(
        df[["feat_income", "feat_age", "feat_debt", "feat_score"]],
        df["segment"],
    )
    joblib.dump(model, output_dir / "sample_iris_logistic.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 18 logistic-regression samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    generate_binary(out)
    generate_multiclass(out)
    print(f"wrote logistic binary + multiclass samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
