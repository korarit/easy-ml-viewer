# backend/scripts/generate_sample_discriminant.py
"""Phase 26 sample bootstrapper — LDA/QDA discriminant demo data (offline).

Generates (seeded, reproducible):
- ``sample_binary_lda.*`` (LDA on correlated 2-blob data — Mahalanobis paradox demo)
- ``sample_multi_qda.*`` (QDA on 3 blobs with distinct covariances — tilted ellipsoids)
- ``sample_lsqr_lda.*`` (LDA ``solver='lsqr', store_covariance=True`` — exact covariance path)

Run:  ``python scripts/generate_sample_discriminant.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


def generate_lda_binary(output_dir: Path) -> None:
    rng = np.random.default_rng(26)
    # Two elongated blobs along different diagonals: Euclidean-near points
    # can be Mahalanobis-far (paradox demo).
    a = rng.normal(size=(250, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(250, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 250 + [1] * 250, dtype=np.int32)
    df = pd.DataFrame({"feat_x": X[:, 0], "feat_y": X[:, 1], "label": y})
    df.to_csv(output_dir / "sample_binary_lda.csv", index=False, encoding="utf-8")
    model = LinearDiscriminantAnalysis().fit(df[["feat_x", "feat_y"]], df["label"])
    joblib.dump(model, output_dir / "sample_binary_lda.joblib", compress=3)


def generate_qda_multi(output_dir: Path) -> None:
    rng = np.random.default_rng(27)
    blobs = [
        (np.array([0.0, 0.0, 2.0]), np.array([[1.8, 1.2, 0.2], [1.2, 1.0, 0.0], [0.2, 0.0, 0.4]]), 120),
        (np.array([4.0, 1.0, -1.0]), np.array([[0.5, -0.4, 0.0], [-0.4, 1.6, 0.3], [0.0, 0.3, 0.9]]), 120),
        (np.array([-1.0, 4.0, 1.0]), np.array([[0.7, 0.0, 0.5], [0.0, 0.6, -0.3], [0.5, -0.3, 1.2]]), 120),
    ]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, (mu, cov, n) in enumerate(blobs):
        xs.append(rng.multivariate_normal(mu, cov, size=n).astype(np.float32))
        ys.extend([k] * n)
    X = np.vstack(xs)
    df = pd.DataFrame({"feat_a": X[:, 0], "feat_b": X[:, 1], "feat_c": X[:, 2], "label": ys})
    df.to_csv(output_dir / "sample_multi_qda.csv", index=False, encoding="utf-8")
    model = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(
        df[["feat_a", "feat_b", "feat_c"]], df["label"]
    )
    joblib.dump(model, output_dir / "sample_multi_qda.joblib", compress=3)


def generate_lsqr_lda(output_dir: Path) -> None:
    rng = np.random.default_rng(28)
    X = np.vstack(
        [rng.normal(c, 0.7, size=(150, 3)) for c in [[0, 0, 0], [3, 2, 1]]]
    ).astype(np.float32)
    y = np.array([0] * 150 + [1] * 150, dtype=np.int32)
    df = pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2], "label": y})
    df.to_csv(output_dir / "sample_lsqr_lda.csv", index=False, encoding="utf-8")
    model = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True).fit(
        df[["f0", "f1", "f2"]], df["label"]
    )
    joblib.dump(model, output_dir / "sample_lsqr_lda.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 26 discriminant samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_lda_binary(out)
    generate_qda_multi(out)
    generate_lsqr_lda(out)
    print(f"wrote discriminant lda-binary/qda-multi/lda-lsqr samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
