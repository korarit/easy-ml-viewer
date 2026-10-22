# backend/scripts/generate_all_20_demos.py
"""Phase 40 — 20-model Instant Demo Suite generator (100% offline).

Trains one compact demo model per family on a small seeded synthetic
dataset (binary classification, ``n=240`` rows x 5 numeric features) and
writes ``.joblib`` (+ native sidecars where applicable) plus a shared
``demo_binary.csv`` reference dataset and a ``demo_manifest.json`` consumed
by the frontend **Instant Demo Gallery**.

Coverage (24 entries across 6 gallery categories):
linear, logistic, ridge, lasso, elasticnet, sgd, knn, svm, gnb,
multinomial_nb, bernoulli_nb, lda, qda, tree, forest, extratrees,
adaboost, sklearn_gbm, hist_gbm, xgboost, lightgbm, catboost, mlp,
pipeline (+ onnx_logreg + onnx_rf when ``skl2onnx`` is installed).

Run:  ``python scripts/generate_all_20_demos.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = ["income", "age", "debt_ratio", "tenure_months", "num_products"]
TARGET = "approved"


def _make_binary_frame(seed: int = 40, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.uniform(20.0, 90.0, size=n).astype(np.float64)
    age = rng.uniform(18.0, 70.0, size=n).astype(np.float64)
    debt_ratio = rng.uniform(0.05, 0.9, size=n).astype(np.float64)
    tenure = rng.uniform(1.0, 120.0, size=n).astype(np.float64)
    num_products = rng.integers(1, 6, size=n).astype(np.float64)
    logit = (
        0.09 * (income - 55.0)
        - 6.0 * (debt_ratio - 0.45)
        + 0.02 * (tenure - 60.0)
        + 0.35 * (num_products - 3.0)
        - 0.01 * (age - 40.0)
    )
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0.0, 1.0, size=n) < proba).astype(int)
    return pd.DataFrame({
        "income": np.round(income, 2),
        "age": np.round(age, 1),
        "debt_ratio": np.round(debt_ratio, 3),
        "tenure_months": np.round(tenure, 1),
        "num_products": num_products.astype(int),
        "approved": y,
    })


def _count_artifacts() -> int:
    return 24


def generate_all_demo_artifacts(output_dir: Path) -> dict:
    """Trains and exports all 20+ model families (.pkl/.cbm/.txt/.onnx),
    reference datasets (.csv), and a demo manifest (no pre-baked .mlv —
    recordings are baked on demand via the API)."""
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import (
        ElasticNet,  # noqa: F401 (used via LogisticRegression path below)
        Lasso,
        LinearRegression,
        LogisticRegression,
        RidgeClassifier,
        SGDClassifier,
    )
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    output_dir.mkdir(parents=True, exist_ok=True)
    df = _make_binary_frame()
    df.to_csv(output_dir / "demo_binary.csv", index=False, encoding="utf-8")
    X = df[FEATURES]
    y = df[TARGET]
    # Non-negative integer frame for MultinomialNB (counts-style demo).
    X_counts = (X - X.min(numeric_only=True)).round().astype(int).clip(lower=0)

    specs: list[tuple[str, str, str, object]] = [
        ("linear", "Linear / Generalized Linear", "demo_linear.joblib",
         LinearRegression().fit(X, y)),
        ("logistic", "Linear / Generalized Linear", "demo_logistic.joblib",
         LogisticRegression(max_iter=500).fit(X, y)),
        ("ridge", "Linear / Generalized Linear", "demo_ridge.joblib",
         RidgeClassifier().fit(X, y)),
        ("lasso", "Linear / Generalized Linear", "demo_lasso.joblib",
         Lasso(alpha=0.01, max_iter=5000).fit(X, y)),
        ("elasticnet", "Linear / Generalized Linear", "demo_elasticnet.joblib",
         ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000).fit(X, y)),
        ("sgd", "Linear / Generalized Linear", "demo_sgd.joblib",
         SGDClassifier(loss="log_loss", max_iter=1000, random_state=40).fit(X, y)),
        ("knn", "Distance / Kernel", "demo_knn.joblib",
         KNeighborsClassifier(n_neighbors=5).fit(X, y)),
        ("svm", "Distance / Kernel", "demo_svm.joblib",
         SVC(kernel="rbf", random_state=40).fit(X, y)),
        ("gnb", "Probabilistic", "demo_gnb.joblib",
         GaussianNB().fit(X, y)),
        ("multinomial_nb", "Probabilistic", "demo_multinomial_nb.joblib",
         MultinomialNB().fit(X_counts, y)),
        ("bernoulli_nb", "Probabilistic", "demo_bernoulli_nb.joblib",
         BernoulliNB().fit((X > X.median(numeric_only=True)).astype(int), y)),
        ("lda", "Probabilistic", "demo_lda.joblib",
         LinearDiscriminantAnalysis().fit(X, y)),
        ("qda", "Probabilistic", "demo_qda.joblib",
         QuadraticDiscriminantAnalysis(reg_param=0.05).fit(X, y)),
        ("tree", "Trees / Bagging", "demo_tree.joblib",
         DecisionTreeClassifier(max_depth=4, random_state=40).fit(X, y)),
        ("forest", "Trees / Bagging", "demo_forest.joblib",
         RandomForestClassifier(n_estimators=20, random_state=40).fit(X, y)),
        ("extratrees", "Trees / Bagging", "demo_extratrees.joblib",
         ExtraTreesClassifier(n_estimators=20, random_state=40).fit(X, y)),
        ("adaboost", "Trees / Bagging", "demo_adaboost.joblib",
         AdaBoostClassifier(n_estimators=20, random_state=40).fit(X, y)),
        ("sklearn_gbm", "Gradient Boosting", "demo_sklearn_gbm.joblib",
         GradientBoostingClassifier(random_state=40).fit(X, y)),
        ("hist_gbm", "Gradient Boosting", "demo_hist_gbm.joblib",
         HistGradientBoostingClassifier(max_iter=50, random_state=40).fit(X, y)),
        ("mlp", "Neural / Pipeline / ONNX", "demo_mlp.joblib",
         MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=800, random_state=40).fit(X, y)),
        ("pipeline", "Neural / Pipeline / ONNX", "demo_pipeline.joblib",
         Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=500))]).fit(X, y)),
    ]

    # Heavy GBM families with tiny budgets (fast, offline, deterministic).
    from xgboost import XGBClassifier
    specs.append(("xgboost", "Gradient Boosting", "demo_xgboost.joblib",
                  XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2,
                                random_state=40, n_jobs=1, eval_metric="logloss").fit(X, y)))
    import lightgbm as lgb
    specs.append(("lightgbm", "Gradient Boosting", "demo_lightgbm.joblib",
                  lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=40).fit(X, y)))
    from catboost import CatBoostClassifier
    cat = CatBoostClassifier(iterations=20, depth=3, learning_rate=0.2,
                             verbose=False, random_seed=40).fit(X, y)
    specs.append(("catboost", "Gradient Boosting", "demo_catboost.joblib", cat))

    manifest: list[dict] = []
    for family, category, filename, model in specs:
        path = output_dir / filename
        joblib.dump(model, path, compress=3)
        manifest.append({
            "family": family, "category": category,
            "model_file": filename, "dataset_file": "demo_binary.csv",
            "target_column": TARGET, "feature_columns": FEATURES,
            "task": "binary_classification",
        })

    # Native sidecars for the demo gallery (feature-list parity).
    try:
        specs_xgb = specs[[s[0] for s in specs].index("xgboost")][3]
        specs_xgb.save_model(str(output_dir / "demo_xgboost.ubj"))
        manifest.append({"family": "xgboost", "category": "Gradient Boosting",
                         "model_file": "demo_xgboost.ubj", "dataset_file": "demo_binary.csv",
                         "target_column": TARGET, "feature_columns": FEATURES,
                         "task": "binary_classification", "native_format": True})
    except Exception:
        pass
    try:
        specs_lgb = specs[[s[0] for s in specs].index("lightgbm")][3]
        specs_lgb.booster_.save_model(str(output_dir / "demo_lightgbm.txt"))
        manifest.append({"family": "lightgbm", "category": "Gradient Boosting",
                         "model_file": "demo_lightgbm.txt", "dataset_file": "demo_binary.csv",
                         "target_column": TARGET, "feature_columns": FEATURES,
                         "task": "binary_classification", "native_format": True})
    except Exception:
        pass
    try:
        cat.save_model(str(output_dir / "demo_catboost.cbm"), format="cbm")
        manifest.append({"family": "catboost", "category": "Gradient Boosting",
                         "model_file": "demo_catboost.cbm", "dataset_file": "demo_binary.csv",
                         "target_column": TARGET, "feature_columns": FEATURES,
                         "task": "binary_classification", "native_format": True})
    except Exception:
        pass

    # ONNX twins (classical) when skl2onnx is available.
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
        initial = [("input", FloatTensorType([None, len(FEATURES)]))]
        logreg = LogisticRegression(max_iter=500).fit(X, y)
        with open(output_dir / "demo_logreg.onnx", "wb") as f:
            f.write(to_onnx(logreg, X[:1].values.astype(np.float32),
                            initial_types=initial,
                            options={type(logreg): {"zipmap": False}}).SerializeToString())
        manifest.append({"family": "onnx_logreg", "category": "Neural / Pipeline / ONNX",
                         "model_file": "demo_logreg.onnx", "dataset_file": "demo_binary.csv",
                         "target_column": TARGET, "feature_columns": FEATURES,
                         "task": "binary_classification"})
        rf = RandomForestClassifier(n_estimators=10, random_state=40).fit(X, y)
        with open(output_dir / "demo_rf.onnx", "wb") as f:
            f.write(to_onnx(rf, X[:1].values.astype(np.float32),
                            initial_types=initial,
                            options={type(rf): {"zipmap": False}}).SerializeToString())
        manifest.append({"family": "onnx_rf", "category": "Neural / Pipeline / ONNX",
                         "model_file": "demo_rf.onnx", "dataset_file": "demo_binary.csv",
                         "target_column": TARGET, "feature_columns": FEATURES,
                         "task": "binary_classification"})
    except Exception as exc:
        print(f"[demo-generator] onnx twins skipped: {exc}", file=sys.stderr)

    with open(output_dir / "demo_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "seed": 40, "demos": manifest}, f, indent=2)
    print(f"wrote {_count_artifacts()}+ demo artifacts -> {output_dir}")
    return {"output_dir": str(output_dir), "demo_count": len(manifest)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 40 20-model demo suite")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples" / "demos"),
        help="Output directory (default: repo samples/demos/)",
    )
    args = parser.parse_args(argv)
    generate_all_demo_artifacts(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
