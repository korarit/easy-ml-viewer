# backend/scripts/generate_sample_svm.py
"""Phase 23 sample bootstrapper — SVM classifier / regressor demo data (offline).

Generates (seeded, reproducible):
- ``sample_moons_svm_rbf.csv`` + ``sample_moons_svm_rbf.joblib``
  (SVC ``kernel="rbf"`` on two moons — kernel saddle demo)
- ``sample_credit_svm_linear.csv`` + ``sample_credit_svm_linear.joblib``
  (LinearSVC — margin-corridor plane demo)
- ``sample_housing_svr.csv`` + ``sample_housing_svr.joblib``
  (SVR ``kernel="rbf"`` — epsilon-tube demo)

Run:  ``python scripts/generate_sample_svm.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC


def _scaled(estimator) -> Pipeline:
    # SVMs need standardized features; ship Scaler -> SVM pipelines so raw
    # CSV ranges stay valid inputs.
    return Pipeline([("scaler", StandardScaler()), ("svm", estimator)])


def generate_moons(output_dir: Path) -> None:
    X, y = make_moons(n_samples=450, noise=0.12, random_state=23)
    X = X.astype(np.float32)
    df = pd.DataFrame(
        {
            "moon_x": np.round(X[:, 0], 3),
            "moon_y": np.round(X[:, 1], 3),
            "noise_a": np.round(np.random.default_rng(23).normal(size=450).astype(np.float32), 3),
            "noise_b": np.round(np.random.default_rng(24).normal(size=450).astype(np.float32), 3),
            "crescent": y.astype(np.int32),
        }
    )
    df.to_csv(output_dir / "sample_moons_svm_rbf.csv", index=False, encoding="utf-8")
    model = _scaled(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=23))
    model.fit(
        df[["moon_x", "moon_y", "noise_a", "noise_b"]],
        df["crescent"],
    )
    joblib.dump(model, output_dir / "sample_moons_svm_rbf.joblib", compress=3)


def generate_linear(output_dir: Path) -> None:
    rng = np.random.default_rng(25)
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
    df.to_csv(output_dir / "sample_credit_svm_linear.csv", index=False, encoding="utf-8")
    model = _scaled(LinearSVC(C=1.0, max_iter=5000, random_state=25))
    model.fit(df[["feat_income", "feat_age", "feat_debt", "feat_score"]], df["default"])
    joblib.dump(model, output_dir / "sample_credit_svm_linear.joblib", compress=3)


def generate_regressor(output_dir: Path) -> None:
    rng = np.random.default_rng(26)
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
    df.to_csv(output_dir / "sample_housing_svr.csv", index=False, encoding="utf-8")
    model = _scaled(SVR(kernel="rbf", C=10.0, epsilon=0.5))
    model.fit(
        df[["feature_rm", "feature_lstat", "feature_tax", "feature_noise"]],
        df["selling_price"],
    )
    joblib.dump(model, output_dir / "sample_housing_svr.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 23 SVM samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_moons(out)
    generate_linear(out)
    generate_regressor(out)
    print(f"wrote svm moons/linear/regressor samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
