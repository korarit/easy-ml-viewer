"""Phase 37 — trace-preview / surface-preview REST tests for Pipeline (.joblib)."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def _upload(path_name: str, expected_raw_features: int) -> str:
    from pathlib import Path

    samples = Path(__file__).resolve().parent.parent.parent.parent / "samples"
    buf = io.BytesIO()
    joblib.dump(joblib.load(samples / path_name), buf)
    buf.seek(0)
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (path_name, buf, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["is_pipeline"] is True, body
    assert body["n_features_in"] == expected_raw_features, body
    return body["model_id"]


def test_trace_preview_mixed_pipeline_grouped_raw() -> None:
    from pathlib import Path

    samples = Path(__file__).resolve().parent.parent.parent.parent / "samples"
    df = pd.read_csv(samples / "sample_pipe_mixed.csv").drop(columns=["approved"])
    model_id = _upload("sample_pipe_mixed.joblib", 4)
    rows = df.head(2).to_numpy(dtype=object).tolist()
    # JSON cannot carry NaN — normalize to None (server maps None -> NaN).
    rows = [[None if (isinstance(v, float) and np.isnan(v)) else v for v in r] for r in rows]
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": rows, "row_indices": [0, 1],
              "feature_names": ["income", "age", "city", "housing"]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    for tr in traces:
        assert set(tr["feature_contributions"]) == {"income", "age", "city", "housing"}
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "sklearn_pipeline"
        assert mech["delegated_estimator"] == "RandomForestClassifier"
        assert "feature_lineage" in mech
        assert "pipeline_steps" in mech
        probas = mech["probabilities"]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(probas[1]), rtol=1e-6, atol=1e-6,
        )


def test_surface_preview_mixed_pipeline_raw_units() -> None:
    model_id = _upload("sample_pipe_mixed.joblib", 4)
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": 15.0,
            "x_max": 120.0,
            "y_min": 20.0,
            "y_max": 65.0,
            "resolution": 20,
            "fixed_feature_medians": {},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["z_values"]) == 400
    assert body["telemetry"]["grid_points"] == 400
    ov = body["extra_overlays"]
    assert ov["model_family"] == "sklearn_pipeline"
    assert ov["coordinate_space"] == "raw_domain_units"
    assert ov["raw_axis_x"] == "income"


def test_trace_preview_numeric_pipeline() -> None:
    from pathlib import Path

    samples = Path(__file__).resolve().parent.parent.parent.parent / "samples"
    df = pd.read_csv(samples / "sample_pipe_numeric.csv").drop(columns=["approved"])
    model_id = _upload("sample_pipe_numeric.joblib", 4)
    rows = df.head(2).to_numpy(dtype=np.float32)
    rows = np.where(np.isnan(rows), 0.0, rows).tolist()
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": rows, "row_indices": [0, 1],
              "feature_names": ["income", "age", "tenure", "balance"]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    assert traces[0]["extra_mechanics"]["model_family"] == "sklearn_pipeline"
