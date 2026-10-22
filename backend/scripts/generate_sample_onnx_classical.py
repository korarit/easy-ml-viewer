# backend/scripts/generate_sample_onnx_classical.py
"""Phase 38 sample bootstrapper — ONNX Runtime classical ML demo data (offline).

Generates (seeded, reproducible) ``.onnx`` artifacts via ``skl2onnx``
(opset 12, ``zipmap=True`` default) plus matching CSVs:

- ``sample_rf_onnx.onnx`` / ``sample_rf_onnx.csv``
  (RandomForestClassifier 20 trees → ``ai.onnx.ml::TreeEnsembleClassifier``)
- ``sample_logreg_onnx.onnx`` / ``sample_logreg_onnx.csv``
  (LogisticRegression → ``ai.onnx.ml::LinearClassifier``)
- ``sample_svm_onnx.onnx`` / ``sample_svm_onnx.csv``
  (SVC RBF ``probability=True`` → ``ai.onnx.ml::SVMClassifier``)

Run:  ``python scripts/generate_sample_onnx_classical.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def _convert(model: object, n_features: int, name: str):  # type: ignore[no-untyped-def]
    from skl2onnx import convert_sklearn  # noqa: PLC0415
    from skl2onnx.common.data_types import FloatTensorType  # noqa: PLC0415

    return convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset=12,
    )


def generate_forest(output_dir: Path) -> None:
    rng = np.random.default_rng(38)
    n = 500
    amount = rng.uniform(10.0, 5000.0, size=n)
    velocity = rng.uniform(0.0, 20.0, size=n)
    age = rng.uniform(18.0, 80.0, size=n)
    tenure = rng.uniform(0.0, 72.0, size=n)
    df = pd.DataFrame({"amount": amount, "velocity": velocity, "age": age, "tenure": tenure})
    y = (
        ((df["amount"] > 2500).astype(int) ^ (df["velocity"] > 10).astype(int))
        ^ ((df["age"] < 30).astype(int) & (df["tenure"] < 6).astype(int))
    ).astype(int).values
    df["fraud"] = y
    df.to_csv(output_dir / "sample_rf_onnx.csv", index=False, encoding="utf-8")
    feats = ["amount", "velocity", "age", "tenure"]
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=38, n_jobs=1)
    model.fit(df[feats].to_numpy(dtype=np.float32), y)
    onx = _convert(model, len(feats), "fraud_rf")
    (output_dir / "sample_rf_onnx.onnx").write_bytes(onx.SerializeToString())


def generate_logreg(output_dir: Path) -> None:
    rng = np.random.default_rng(39)
    n = 500
    income = rng.uniform(15.0, 150.0, size=n)
    debt_ratio = rng.uniform(0.0, 1.0, size=n)
    age = rng.uniform(20.0, 65.0, size=n)
    df = pd.DataFrame({"income": income, "debt_ratio": debt_ratio, "age": age})
    logit = -0.04 * df["income"] + 4.0 * df["debt_ratio"] - 0.02 * df["age"]
    y = (logit + rng.normal(0, 1.0, size=n) > 0).astype(int).values
    df["default"] = y
    df.to_csv(output_dir / "sample_logreg_onnx.csv", index=False, encoding="utf-8")
    feats = ["income", "debt_ratio", "age"]
    model = LogisticRegression(max_iter=500)
    model.fit(df[feats].to_numpy(dtype=np.float32), y)
    onx = _convert(model, len(feats), "credit_logreg")
    (output_dir / "sample_logreg_onnx.onnx").write_bytes(onx.SerializeToString())


def generate_svm(output_dir: Path) -> None:
    rng = np.random.default_rng(40)
    n = 400
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r0 = rng.normal(2.0, 0.35, size=n)
    r1 = rng.normal(5.0, 0.5, size=n)
    outer = rng.integers(0, 2, size=n).astype(bool)
    r = np.where(outer, r1, r0)
    x0 = (r * np.cos(theta)).astype(np.float32)
    x1 = (r * np.sin(theta)).astype(np.float32)
    df = pd.DataFrame({"ring_x": x0, "ring_y": x1, "outer": outer.astype(int)})
    feats = ["ring_x", "ring_y"]
    model = SVC(kernel="rbf", gamma="scale", probability=True, random_state=40)
    model.fit(df[feats].to_numpy(dtype=np.float32), outer.astype(int))
    onx = _convert(model, len(feats), "moons_svm")
    (output_dir / "sample_svm_onnx.onnx").write_bytes(onx.SerializeToString())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 38 ONNX classical samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_forest(out)
    generate_logreg(out)
    generate_svm(out)
    print("wrote onnx classical rf/logreg/svm samples ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
