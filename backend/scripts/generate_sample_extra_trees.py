# backend/scripts/generate_sample_extra_trees.py
"""Phase 29 sample bootstrapper — extra-trees demo data (offline).

Generates (seeded, reproducible):
- ``sample_binary_extratrees.*`` (ExtraTreesClassifier on 2-D blobs)
- ``sample_multi_extratrees.*`` (ExtraTreesClassifier on 3-class data)
- ``sample_reg_extratrees.*`` (ExtraTreesRegressor on housing-like data)

Run:  ``python scripts/generate_sample_extra_trees.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


def generate_binary(output_dir: Path) -> None:
    rng = np.random.default_rng(31)
    a = rng.normal(size=(250, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(250, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 250 + [1] * 250, dtype=np.int32)
    df = pd.DataFrame({"feat_income": X[:, 0], "feat_age": X[:, 1], "label": y})
    df.to_csv(output_dir / "sample_binary_extratrees.csv", index=False, encoding="utf-8")
    model = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=31).fit(
        df[["feat_income", "feat_age"]], df["label"]
    )
    joblib.dump(model, output_dir / "sample_binary_extratrees.joblib", compress=3)


def generate_multi(output_dir: Path) -> None:
    rng = np.random.default_rng(32)
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
    df.to_csv(output_dir / "sample_multi_extratrees.csv", index=False, encoding="utf-8")
    feats = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    model = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=32).fit(df[feats], df["species"])
    joblib.dump(model, output_dir / "sample_multi_extratrees.joblib", compress=3)


def generate_reg(output_dir: Path) -> None:
    rng = np.random.default_rng(33)
    n = 400
    rooms = rng.integers(1, 7, size=n).astype(np.float32)
    area = rng.uniform(30.0, 200.0, size=n).astype(np.float32)
    age = rng.uniform(0.0, 50.0, size=n).astype(np.float32)
    price = (rooms * 28.0 + area * 1.6 - age * 0.9 + rng.normal(0, 12.0, size=n)).astype(np.float32)
    df = pd.DataFrame({"rooms": rooms, "area": area, "age": age, "price": price})
    df.to_csv(output_dir / "sample_reg_extratrees.csv", index=False, encoding="utf-8")
    model = ExtraTreesRegressor(n_estimators=50, max_depth=8, random_state=33).fit(
        df[["rooms", "area", "age"]], df["price"]
    )
    joblib.dump(model, output_dir / "sample_reg_extratrees.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 29 extra-trees samples")
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
    print(f"wrote extra-trees binary/multi/reg samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
