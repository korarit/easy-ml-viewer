"""Phase 43 — Support Vector halo extraction (counts, weights, bound/free)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from app.domains.model_engine.adapters.distance.distance_2d_extractor import Distance2DExtractor


def test_svm_support_vectors_count_and_weights_parity(spec_svc_linear) -> None:
    """Blueprint §2.2 prototype: SV count and dual weights match the model."""
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    svm = spec_svc_linear

    res = Distance2DExtractor.extract_svm_2d_state(svm, x_idx=0, y_idx=1)
    assert res is not None
    assert len(res["support_vectors_2d"]) == len(svm.support_vectors_)
    assert len(res["dual_coefs"]) == len(svm.dual_coef_[0])
    assert res["n_support_total"] == len(svm.support_)
    for halo, sv_row in zip(res["support_vectors_2d"], svm.support_vectors_):
        assert abs(halo["x"] - float(sv_row[0])) < 1e-9
        assert abs(halo["y"] - float(sv_row[1])) < 1e-9


def test_halo_radius_formula_and_bound_status() -> None:
    """r = 4 + 6·|α|/C; |α| ≈ C marks margin violators (orange)."""
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=9)
    X = X.astype(np.float32)
    # Small C forces bounded (violating) support vectors into existence.
    model = SVC(kernel="rbf", C=0.1).fit(X, y)
    res = Distance2DExtractor.extract_svm_2d_state(model, x_idx=0, y_idx=1)
    assert res is not None and res["c"] == 0.1
    assert res["support_vectors_2d"]
    duals = np.abs(np.asarray(model.dual_coef_[0], dtype=np.float64))
    ranked = np.argsort(-duals, kind="stable")
    for halo, pos in zip(res["support_vectors_2d"][: len(ranked)], ranked.tolist()):
        expected_scale = min(float(duals[pos]) / 0.1, 1.0)
        assert abs(halo["halo_scale"] - expected_scale) < 1e-9
        assert halo["status"] in ("on_margin", "violates_margin")
    statuses = {h["status"] for h in res["support_vectors_2d"]}
    assert "violates_margin" in statuses  # C = 0.1 guarantees bound SVs


def test_halo_truncation_ranked_by_dual() -> None:
    """E43.5: over-budget halo sets truncate by |dual| with an explicit flag."""
    X, y = make_blobs(n_samples=400, centers=2, n_features=2, random_state=13)
    X = X.astype(np.float32)
    model = SVC(kernel="rbf", C=0.5).fit(X, y)
    n_sv = len(model.support_vectors_)
    assert n_sv > 8  # enough to exercise the cap
    res = Distance2DExtractor.extract_svm_2d_state(model, x_idx=0, y_idx=1, max_halos=8)
    assert res is not None and res["halos_truncated"] is True
    assert len(res["support_vectors_2d"]) == 8
    assert res["n_support_total"] == n_sv
    duals = np.abs(np.asarray(model.dual_coef_[0], dtype=np.float64))
    kept = sorted((abs(h["dual_coef"]) for h in res["support_vectors_2d"]), reverse=True)
    top8 = sorted(duals.tolist(), reverse=True)[:8]
    assert all(abs(a - b) < 1e-9 for a, b in zip(kept, top8))


def test_linearsvc_reports_no_sv_plane() -> None:
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    model = LinearSVC(max_iter=5000).fit(X, y)
    res = Distance2DExtractor.extract_svm_2d_state(model, x_idx=0, y_idx=1)
    assert res is not None and res["is_linear_no_sv"] is True
    assert res["support_vectors_2d"] == [] and res["n_support_total"] == 0


def test_svm_pipeline_and_non_svm_return_none() -> None:
    X, y = make_blobs(n_samples=60, centers=2, n_features=2, random_state=21)
    X = X.astype(np.float32)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(X, y)
    assert Distance2DExtractor.extract_svm_2d_state(pipe) is None
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    assert Distance2DExtractor.extract_svm_2d_state(tree) is None


def test_heavy_sv_scene_stays_within_dto_budget() -> None:
    """1462-SV RBF scene at full 120² must respect the 150 KB contour SLA."""
    import json

    from app.domains.model_engine.value_objects import PredictionBatchResult
    from app.domains.simulation.contour_2d_generator import Contour2DGenerator

    class _Adapter:
        def predict_batch(self, model_obj, X_float32):
            preds = np.asarray(model_obj.predict(np.asarray(X_float32, dtype=np.float32))).ravel()
            return PredictionBatchResult(predictions=[float(v) for v in preds.tolist()])

    X, y = make_blobs(
        n_samples=3000, centers=[(0, 0, 0, 0), (1, 1, 1, 1)],
        cluster_std=1.2, n_features=4, random_state=1,
    )
    X = X.astype(np.float32)
    model = SVC(kernel="rbf", C=0.5).fit(X, y)
    assert len(model.support_vectors_) > 800
    out = Contour2DGenerator.generate_contour(
        adapter=_Adapter(),
        model_obj=model,
        final_est=model,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=120,
    )
    assert out.svm_2d is not None and out.svm_2d["halos_truncated"] is True
    assert len(out.svm_2d["support_vectors_2d"]) <= 400
    assert len(json.dumps(out.model_dump(mode="json"))) <= 150_000
