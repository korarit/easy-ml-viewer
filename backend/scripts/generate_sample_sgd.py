# backend/scripts/generate_sample_sgd.py
"""Phase 21 sample bootstrapper — SGDClassifier / SGDRegressor demo data (offline).

Generates (seeded, reproducible):
- ``sample_credit_sgd_hinge.csv`` + ``sample_credit_sgd_hinge.joblib``
  (SGDClassifier ``loss="hinge"`` — Linear SVM mode, no native proba)
- ``sample_credit_sgd_log.csv`` + ``sample_credit_sgd_log.joblib``
  (SGDClassifier ``loss="log_loss"`` — native predict_proba)
- ``sample_housing_sgd.csv`` + ``sample_housing_sgd.joblib``
  (SGDRegressor ``loss="huber"`` — epsilon-tube corridor)

Run:  ``python scripts/generate_sample_sgd.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _scaled_pipeline(estimator) -> Pipeline:
    # Blueprint edge case #4: SGD diverges on unscaled features, so samples
    # ship as a Scaler -> SGD pipeline (raw CSV rows stay valid inputs).
    return Pipeline([("scaler", StandardScaler()), ("sgd", estimator)])


def generate_hinge(output_dir: Path) -> None:
    rng = np.random.default_rng(21)
    n = 500
    velocity = rng.uniform(10.0, 120.0, size=n).astype(np.float32)
    pressure = rng.uniform(0.5, 8.0, size=n).astype(np.float32)
    temp = rng.uniform(-10.0, 45.0, size=n).astype(np.float32)
    vibration = rng.uniform(0.0, 5.0, size=n).astype(np.float32)
    score = 0.08 * velocity + 1.4 * pressure - 0.05 * temp + 2.2 * vibration - 9.0
    y = (score > 0).astype(np.int32)
    df = pd.DataFrame(
        {
            "feature_velocity": np.round(velocity, 2),
            "feature_pressure": np.round(pressure, 3),
            "feature_temp": np.round(temp, 1),
            "feature_vibration": np.round(vibration, 3),
            "failure": y,
        }
    )
    df.to_csv(output_dir / "sample_credit_sgd_hinge.csv", index=False, encoding="utf-8")
    model = _scaled_pipeline(
        SGDClassifier(loss="hinge", alpha=0.0001, max_iter=2000, tol=1e-4, random_state=21)
    )
    model.fit(
        df[["feature_velocity", "feature_pressure", "feature_temp", "feature_vibration"]],
        df["failure"],
    )
    joblib.dump(model, output_dir / "sample_credit_sgd_hinge.joblib", compress=3)


def generate_log(output_dir: Path) -> None:
    rng = np.random.default_rng(22)
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
    df.to_csv(output_dir / "sample_credit_sgd_log.csv", index=False, encoding="utf-8")
    model = _scaled_pipeline(
        SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=2000, tol=1e-4, random_state=22)
    )
    model.fit(df[["feat_income", "feat_age", "feat_debt", "feat_score"]], df["default"])
    joblib.dump(model, output_dir / "sample_credit_sgd_log.joblib", compress=3)


def generate_regressor(output_dir: Path) -> None:
    rng = np.random.default_rng(23)
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
    df.to_csv(output_dir / "sample_housing_sgd.csv", index=False, encoding="utf-8")
    model = _scaled_pipeline(
        SGDRegressor(loss="huber", epsilon=0.5, alpha=0.0001, max_iter=2000, tol=1e-4, random_state=23)
    )
    model.fit(df[["feature_rm", "feature_lstat", "feature_tax", "feature_noise"]], df["selling_price"])
    joblib.dump(model, output_dir / "sample_housing_sgd.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 21 SGD samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_hinge(out)
    generate_log(out)
    generate_regressor(out)
    print(f"wrote sgd hinge/log-loss/regressor samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
