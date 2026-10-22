"""Phase 46 — POST /api/v1/projection/boost-frame-2d endpoint contract."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from sklearn.datasets import make_blobs

from app.main import create_app

client = TestClient(create_app())


def _bootstrap_xgb():
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=7)
    X = X.astype(np.float64)
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, random_state=42,
                              nthread=2, verbosity=0).fit(X, y)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    up = client.post(
        "/api/v1/models/upload",
        files={"file": ("xgb46.joblib", buf, "application/octet-stream")},
    )
    assert up.status_code == 200, up.text
    model_id = up.json()["model_id"]

    import pandas as pd

    frame = pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1]})
    csv = frame.to_csv(index=False).encode()
    ds = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("blobs46.csv", io.BytesIO(csv), "text/csv")},
    )
    assert ds.status_code == 200, ds.text
    dataset_id = ds.json()["dataset_id"]

    auto = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": dataset_id, "model_id": model_id},
    ).json()
    commit = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "items": auto["mapping"]["items"],
        },
    )
    assert commit.status_code == 200, commit.text
    mapping = commit.json()["mapping"]
    cols = [i["dataset_column_name"] or i["model_feature_name"] for i in mapping["items"]]
    return dataset_id, model_id, mapping["mapping_id"], cols


def test_boost_frame_shap_contract():
    ds, mdl, mp, cols = _bootstrap_xgb()
    res = client.post(
        "/api/v1/projection/boost-frame-2d",
        json={
            "dataset_id": ds,
            "model_id": mdl,
            "mapping_id": mp,
            "x_feature": cols[0],
            "y_feature": cols[1],
            "row_index": 3,
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["row_index"] == 3
    assert data["family"] == "xgboost"
    assert len(data["query_world"]) == 2
    assert all(-5.5 - 1e-6 <= v <= 5.5 + 1e-6 for v in data["query_world"])
    shap = data["shap"]
    assert shap is not None and shap["method"] == "xgb_pred_contribs"
    assert shap["magnitude"] >= 0
    assert abs(shap["magnitude"] - float(np.hypot(shap["phi_x"], shap["phi_y"]))) < 1e-3
    assert data["computation_time_ms"] >= 0.0


def test_boost_frame_rejects_row_out_of_range():
    ds, mdl, mp, cols = _bootstrap_xgb()
    res = client.post(
        "/api/v1/projection/boost-frame-2d",
        json={
            "dataset_id": ds,
            "model_id": mdl,
            "mapping_id": mp,
            "x_feature": cols[0],
            "y_feature": cols[1],
            "row_index": 10_000_000,
        },
    )
    assert res.status_code == 422


def test_frame_shap_unit_parity_across_families(blobs_2d, fitted_lgbm,
                                               fitted_hist, fitted_gbm):
    """extract_frame_shap: native SHAP for lgbm, None for sklearn/hist (honest)."""
    from app.domains.model_engine.adapters.boosting.boosting_2d_extractor import (
        Boosting2DExtractor as B,
    )

    X, _ = blobs_2d
    kw = dict(X=X, row_index=5, x_feature_idx=0, y_feature_idx=1)

    lgbm_state = B.extract_frame_shap(model_obj=fitted_lgbm, **kw)
    assert lgbm_state is not None and lgbm_state["family"] == "lightgbm"
    assert lgbm_state["shap_vector"] is not None
    assert lgbm_state["query_raw"] == [round(float(X[5, 0]), 6), round(float(X[5, 1]), 6)]

    hist_state = B.extract_frame_shap(model_obj=fitted_hist, **kw)
    assert hist_state is not None and hist_state["family"] == "hist_gbm"
    assert hist_state["shap_vector"] is None  # no native TreeSHAP: honest null

    gbm_state = B.extract_frame_shap(model_obj=fitted_gbm, **kw)
    assert gbm_state is not None and gbm_state["family"] == "sklearn_gbm"
    assert gbm_state["shap_vector"] is None

    from sklearn.linear_model import LogisticRegression

    X, y = blobs_2d
    lr = LogisticRegression().fit(X, y)
    assert B.extract_frame_shap(model_obj=lr, **kw) is None
