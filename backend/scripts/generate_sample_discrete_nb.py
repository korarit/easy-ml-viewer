# backend/scripts/generate_sample_discrete_nb.py
"""Phase 25 sample bootstrapper — discrete Naive Bayes demo data (offline).

Generates (seeded, reproducible):
- ``sample_text_multinomial.*`` (MultinomialNB on word-count vectors)
- ``sample_binary_bernoulli.*`` (BernoulliNB on binary presence vectors)
- ``sample_imbalanced_complement.*`` (ComplementNB on imbalanced counts)
- ``sample_categorical_catnb.*`` (CategoricalNB on integer categories)

Run:  ``python scripts/generate_sample_discrete_nb.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    MultinomialNB,
)

VOCAB = [
    "word_free",
    "word_win",
    "word_offer",
    "word_meeting",
    "word_report",
    "word_lunch",
    "word_money",
    "word_project",
]


def generate_multinomial(output_dir: Path) -> None:
    rng = np.random.default_rng(25)
    n = 600
    lam = np.array(
        [
            [4.0, 3.0, 3.5, 0.2, 0.3, 0.2, 2.5, 0.4],  # spam
            [0.1, 0.2, 1.5, 3.0, 2.5, 1.0, 0.3, 3.5],  # ham/work
            [0.5, 0.5, 2.0, 1.0, 1.0, 2.5, 1.0, 1.0],  # promo
        ]
    )
    y = rng.integers(0, 3, size=n).astype(np.int32)
    X = np.stack([rng.poisson(lam[k]).astype(np.float32) for k in y])
    df = pd.DataFrame(X, columns=VOCAB)
    df["label"] = y
    df.to_csv(output_dir / "sample_text_multinomial.csv", index=False, encoding="utf-8")
    model = MultinomialNB(alpha=1.0).fit(df[VOCAB], df["label"])
    joblib.dump(model, output_dir / "sample_text_multinomial.joblib", compress=3)


def generate_bernoulli(output_dir: Path) -> None:
    rng = np.random.default_rng(26)
    n = 500
    feats = ["has_free", "has_win", "has_meeting", "has_report", "has_link", "has_attach"]
    p_spam = np.array([0.80, 0.70, 0.05, 0.05, 0.60, 0.40])
    p_ham = np.array([0.05, 0.02, 0.60, 0.70, 0.20, 0.50])
    y = (rng.random(n) < 0.4).astype(np.int32)
    X = np.stack(
        [rng.random(len(feats)) < (p_spam if k else p_ham) for k in y]
    ).astype(np.float32)
    df = pd.DataFrame(X, columns=feats)
    df["label"] = y
    df.to_csv(output_dir / "sample_binary_bernoulli.csv", index=False, encoding="utf-8")
    model = BernoulliNB(alpha=1.0, binarize=0.5).fit(df[feats], df["label"])
    joblib.dump(model, output_dir / "sample_binary_bernoulli.joblib", compress=3)


def generate_complement(output_dir: Path) -> None:
    rng = np.random.default_rng(27)
    n = 500
    feats = ["cnt_deal", "cnt_cash", "cnt_minutes", "cnt_pages", "cnt_edits"]
    y = (rng.random(n) < 0.12).astype(np.int32)  # ~12% minority spam
    lam_spam = np.array([5.0, 4.0, 1.0, 0.5, 0.5])
    lam_ham = np.array([0.2, 0.2, 6.0, 4.0, 3.0])
    X = np.stack([rng.poisson(lam_spam if k else lam_ham).astype(np.float32) for k in y])
    df = pd.DataFrame(X, columns=feats)
    df["label"] = y
    df.to_csv(output_dir / "sample_imbalanced_complement.csv", index=False, encoding="utf-8")
    model = ComplementNB(alpha=1.0).fit(df[feats], df["label"])
    joblib.dump(model, output_dir / "sample_imbalanced_complement.joblib", compress=3)


def generate_categorical(output_dir: Path) -> None:
    rng = np.random.default_rng(28)
    n = 500
    # Mixed cardinalities: color(3), size(2), region(4), plan(2).
    feats = ["color", "size", "region", "plan"]
    X = np.stack(
        [
            rng.integers(0, 3, size=n),
            rng.integers(0, 2, size=n),
            rng.integers(0, 4, size=n),
            rng.integers(0, 2, size=n),
        ],
        axis=1,
    ).astype(np.float32)
    y = ((X[:, 0] == 2) | ((X[:, 2] >= 2) & (X[:, 3] == 1))).astype(np.int32)
    df = pd.DataFrame(X, columns=feats)
    df["label"] = y
    df.to_csv(output_dir / "sample_categorical_catnb.csv", index=False, encoding="utf-8")
    model = CategoricalNB(alpha=1.0).fit(df[feats], df["label"])
    joblib.dump(model, output_dir / "sample_categorical_catnb.joblib", compress=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 25 discrete-NB samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_multinomial(out)
    generate_bernoulli(out)
    generate_complement(out)
    generate_categorical(out)
    print(f"wrote discrete-nb multinomial/bernoulli/complement/categorical samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
