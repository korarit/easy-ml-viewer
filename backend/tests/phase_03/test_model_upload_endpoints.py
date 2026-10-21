"""Model upload endpoints: happy path, warnings, rejections, samples, LRU."""

from __future__ import annotations

import io
import warnings

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression

from app.infrastructure.ml_loaders.joblib_model_repository import JoblibModelRepository
from app.main import app


def _upload(client: TestClient, buf: io.BytesIO, filename: str):
    buf.seek(0)
    return client.post(
        "/api/v1/models/upload",
        files={"file": (filename, buf, "application/octet-stream")},
    )


def _fitted_buffer() -> io.BytesIO:
    X = np.array([[50.0, 2.0], [80.0, 3.0], [120.0, 4.0]], dtype=np.float32)
    y = np.array([3.0e6, 4.5e6, 6.0e6], dtype=np.float32)
    model = LinearRegression().fit(X, y)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf


def test_upload_model_happy_path_returns_metadata() -> None:
    client = TestClient(app)
    resp = _upload(client, _fitted_buffer(), "house_ols.joblib")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["algorithm_key"] == "linear_regression"
    assert data["display_name"] == "Linear Regression (OLS)"
    assert data["task_type"] == "regression"
    assert data["visual_3d_type"] == "flat_hyperplane"
    assert data["n_features_in"] == 2
    assert len(data["sha256_hash"]) == 64
    assert all(f["is_auto_generated_name"] for f in data["features"])
    assert data["intercept_or_base"] is not None
    assert "fit_intercept" in data["hyperparameters"]


def test_upload_model_endpoint_captures_version_mismatch_warning(
    monkeypatch, tmp_path
) -> None:
    buf = _fitted_buffer()
    orig_load = joblib.load

    def mocked_load(*args, **kwargs):
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="LinearRegression",
                current_sklearn_version="1.5.2",
                original_sklearn_version="1.4.0",
            )
        )
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", mocked_load)

    client = TestClient(app)
    resp = _upload(client, buf, "house_ols.joblib")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["algorithm_key"] == "linear_regression"
    assert data["n_features_in"] == 2
    assert len(data["warnings"]) >= 1
    assert "1.4.0" in data["warnings"][0]


def test_upload_unfitted_model_rejected() -> None:
    buf = io.BytesIO()
    joblib.dump(LinearRegression(), buf)
    client = TestClient(app)
    resp = _upload(client, buf, "unfitted.joblib")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MODEL_NOT_FITTED"


def test_reject_dataframe_or_dict_uploaded_as_model() -> None:
    buf = io.BytesIO()
    joblib.dump(pd.DataFrame({"a": [1, 2]}), buf)
    client = TestClient(app)
    resp = _upload(client, buf, "table.pkl")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "NOT_A_VALID_MODEL_ARTIFACT"


def test_supported_lists_registered_algorithms() -> None:
    client = TestClient(app)
    resp = client.get("/api/v1/models/supported")
    assert resp.status_code == 200
    keys = [a["algorithm_key"] for a in resp.json()["algorithms"]]
    assert "linear_regression" in keys


def test_model_repository_enforces_single_thread_and_lru_eviction(tmp_path) -> None:
    repo = JoblibModelRepository(cache_dir=tmp_path, max_models_in_ram=2)
    X = (np.random.rand(50, 4) * 10).astype(np.float32)
    y = (np.random.rand(50) * 5).astype(np.float32)

    for i in range(3):
        mdl = LinearRegression(n_jobs=-1).fit(X, y)
        p = tmp_path / f"mdl_{i}.joblib"
        joblib.dump(mdl, p)
        loaded_obj, _ = repo.load_and_cache(model_id=f"id_{i}", file_path=p)
        assert getattr(loaded_obj, "n_jobs", 1) == 1

    # Only the 2 most recent models stay in RAM.
    assert len(repo.active_model_ids()) == 2
    assert "id_0" not in repo.active_model_ids()
    assert repo.active_model_ids() == ["id_1", "id_2"]
