# backend/scripts/generate_sample_mlp.py
"""Phase 36 sample bootstrapper — MLP synapse-brain demo data (offline).

Generates (seeded, reproducible):
- ``sample_binary_mlp.joblib`` / ``sample_binary_mlp.csv``
  (MLPClassifier ``(64, 32)`` relu, churn-style binary task)
- ``sample_multi_mlp.joblib`` / ``sample_multi_mlp.csv``
  (MLPClassifier ``(32, 16)`` 3-class task)
- ``sample_reg_mlp.joblib`` / ``sample_reg_mlp.csv``
  (MLPRegressor ``(32, 16)`` housing-style task)

Run:  ``python scripts/generate_sample_mlp.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor


def generate_binary(output_dir: Path) -> None:
    rng = np.random.default_rng(36)
    n = 500
    tenure = rng.uniform(0.0, 72.0, size=n).astype(float)
    monthly = rng.uniform(20.0, 120.0, size=n).astype(float)
    age = rng.uniform(18.0, 80.0, size=n).astype(float)
    support_calls = rng.integers(0, 10, size=n).astype(float)
    df = pd.DataFrame(
        {"tenure": tenure, "monthly_chg": monthly, "age": age, "support_calls": support_calls}
    )
    logit = -0.08 * tenure + 0.05 * monthly - 0.02 * age + 0.55 * support_calls - 1.2
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < proba).astype(int)
    df["churn"] = y
    df.to_csv(output_dir / "sample_binary_mlp.csv", index=False, encoding="utf-8")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=400, random_state=36)
    model.fit(df[["tenure", "monthly_chg", "age", "support_calls"]].to_numpy(dtype=np.float32), y)
    joblib.dump(model, output_dir / "sample_binary_mlp.joblib", compress=3)


def generate_multi(output_dir: Path) -> None:
    rng = np.random.default_rng(37)
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
    df.to_csv(output_dir / "sample_multi_mlp.csv", index=False, encoding="utf-8")
    feats = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="tanh", max_iter=400, random_state=37)
    model.fit(df[feats].to_numpy(dtype=np.float32), df["species"].to_numpy())
    joblib.dump(model, output_dir / "sample_multi_mlp.joblib", compress=3)


def generate_reg(output_dir: Path) -> None:
    rng = np.random.default_rng(38)
    n = 400
    rooms = rng.integers(1, 7, size=n).astype(np.float32)
    area = rng.uniform(30.0, 200.0, size=n).astype(np.float32)
    age = rng.uniform(0.0, 50.0, size=n).astype(np.float32)
    price = (rooms * 28.0 + area * 1.6 - age * 0.9 + rng.normal(0, 12.0, size=n)).astype(np.float32)
    df = pd.DataFrame({"rooms": rooms, "area": area, "age": age, "price": price})
    df.to_csv(output_dir / "sample_reg_mlp.csv", index=False, encoding="utf-8")
    model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", max_iter=400, random_state=38)
    model.fit(df[["rooms", "area", "age"]].to_numpy(dtype=np.float32), df["price"].to_numpy(dtype=np.float32))
    joblib.dump(model, output_dir / "sample_reg_mlp.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 36 MLP samples")
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
    print(f"wrote mlp binary/multi/reg samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
