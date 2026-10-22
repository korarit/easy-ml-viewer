"""Phase 37 — Pipeline unwrapper math: schema, lineage, parity, waterfall, surface."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app.domains.model_engine.adapters.neural_onnx.pipeline_adapter import (
    SklearnPipelineAdapter,
    build_lineage_map,
    is_sklearn_pipeline,
    preprocessing_steps_of,
    raw_feature_names_of,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import GridSpec3D

# Ensure inner adapters are registered (mirrors app wiring).
import app.domains.model_engine.adapters.linear.logistic_regression_adapter as _lr  # noqa: F401
import app.domains.model_engine.adapters.trees.random_forest as _rf  # noqa: F401
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent.parent.parent / "samples"


def _load(name: str):
    return joblib.load(SAMPLES / name)


def _spec(**over) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": 15.0,
        "x_max": 120.0,
        "y_min": 20.0,
        "y_max": 65.0,
        "resolution": 20,
        "fixed_feature_medians": {},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def test_raw_schema_and_lineage_mixed_pipeline() -> None:
    pipe = _load("sample_pipe_mixed.joblib")
    assert is_sklearn_pipeline(pipe) is True
    assert SklearnPipelineAdapter.can_handle(pipe) is True
    raw = raw_feature_names_of(pipe)
    assert raw == ["income", "age", "city", "housing"]
    pre = preprocessing_steps_of(pipe)
    assert len(pre) == 1 and pre[0][0] == "preprocessor"
    trans_names, lineage = build_lineage_map(pre, raw)
    assert len(trans_names) == 8
    assert set(lineage) == set(raw)
    # Partition: every transformed col has exactly one raw owner.
    owners = [r for r, cols in lineage.items() for _ in cols]
    assert sorted(owners) == sorted(trans_names) or len(owners) == len(trans_names)
    assert lineage["income"] == ["num__income"]
    assert len(lineage["city"]) == 3
    assert len(lineage["housing"]) == 3
    assert set(lineage["city"]).isdisjoint(set(lineage["housing"]))
    # Raw schema via detector reports 4 raw inputs (not 8 transformed).
    schema = UniversalSchemaDetector.extract_schema(pipe)
    assert schema.n_features_in == 4


def test_predict_parity_numeric_and_mixed() -> None:
    adapter = SklearnPipelineAdapter()
    for joblib_name, csv_name, feats in [
        ("sample_pipe_numeric.joblib", "sample_pipe_numeric.csv",
         ["income", "age", "tenure", "balance"]),
        ("sample_pipe_mixed.joblib", "sample_pipe_mixed.csv",
         ["income", "age", "city", "housing"]),
    ]:
        pipe = _load(joblib_name)
        df = pd.read_csv(SAMPLES / csv_name).drop(columns=["approved"]).head(20)
        res = adapter.predict_batch(pipe, df)
        expected_proba = np.asarray(pipe.predict_proba(df), dtype=np.float64)
        got = np.asarray(res.probabilities, dtype=np.float64)
        np.testing.assert_allclose(got, expected_proba, rtol=1e-6, atol=1e-6)
        expected_pred = pipe.predict(df)
        assert [int(v) for v in res.predictions] == [int(v) for v in expected_pred.tolist()]


def test_sparse_densify_and_chunked_transform() -> None:
    from scipy import sparse  # noqa: PLC0415

    pipe = _load("sample_pipe_mixed.joblib")
    adapter = SklearnPipelineAdapter()
    df = pd.read_csv(SAMPLES / "sample_pipe_mixed.csv").drop(columns=["approved"]).head(100)
    Xt, traces, fmap = adapter.trace_preprocessing_steps_for(pipe, df)
    assert Xt.dtype == np.float32
    assert Xt.shape == (100, 8)
    assert not sparse.issparse(Xt)
    assert traces and traces[0].step_name == "preprocessor"
    assert fmap.raw_names == ["income", "age", "city", "housing"]


def test_trace_waterfall_identity_and_flags() -> None:
    pipe = _load("sample_pipe_mixed.joblib")
    adapter = SklearnPipelineAdapter()
    df = pd.read_csv(SAMPLES / "sample_pipe_mixed.csv").drop(columns=["approved"])
    nan_idx = df[df.isna().any(axis=1)].index[:2].tolist()
    assert nan_idx, "sample csv must contain NaN rows for imputation flags"
    sub = df.iloc[nan_idx]
    traces = adapter.trace_internal_flow(
        pipe, sub.to_numpy(dtype=object), list(range(len(nan_idx))),
        ["income", "age", "city", "housing"],
    )
    assert len(traces) == len(nan_idx)
    for tr in traces:
        probas = tr.extra_mechanics["probabilities"]
        # RandomForest binary traces explain the positive class (idx 1).
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(probas[1]),
            rtol=1e-6, atol=1e-6,
        )
        assert set(tr.feature_contributions) == {"income", "age", "city", "housing"}
        assert tr.extra_mechanics["has_imputed_values"] is True
        assert tr.extra_mechanics["model_family"] == "sklearn_pipeline"
        assert tr.extra_mechanics["delegated_estimator"] == "RandomForestClassifier"
        assert len(tr.extra_mechanics["pipeline_steps"]) >= 1
    # OOV: unseen city collapses its OHE block to all-zeros.
    oov = pd.DataFrame([{"income": 60.0, "age": 40.0, "city": "Pattaya", "housing": "Rent"}])
    oov_traces = adapter.trace_internal_flow(
        pipe, oov.to_numpy(dtype=object), [0], ["income", "age", "city", "housing"]
    )
    assert oov_traces[0].extra_mechanics["has_oov_category"] is True
    assert "city" in oov_traces[0].extra_mechanics["oov_columns"]


def test_lineage_aggregation_sums_to_inner() -> None:
    pipe = _load("sample_pipe_mixed.joblib")
    adapter = SklearnPipelineAdapter()
    raw = raw_feature_names_of(pipe)
    pre = preprocessing_steps_of(pipe)
    _, lineage = build_lineage_map(pre, raw)
    trans = {"num__income": 0.5, "num__age": -0.2, "cat__city_Bangkok": 0.3,
             "cat__city_Chiang Mai": 0.1, "cat__city_Phuket": 0.0,
             "cat__housing_Mortgage": 0.05, "cat__housing_Own": -0.05,
             "cat__housing_Rent": 0.2}
    agg = adapter.aggregate_lineage_attributions(trans, lineage)
    assert agg["income"] == 0.5
    assert agg["city"] == 0.4
    assert abs(sum(agg.values()) - sum(trans.values())) < 1e-9


def test_surface_raw_space_shape_and_overlays() -> None:
    pipe = _load("sample_pipe_mixed.joblib")
    adapter = SklearnPipelineAdapter()
    mesh = adapter.compute_3d_surface(pipe, _spec(), n_features=4)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    ov = mesh.extra_overlays
    assert ov["model_family"] == "sklearn_pipeline"
    assert ov["delegated_estimator"] == "RandomForestClassifier"
    assert ov["coordinate_space"] == "raw_domain_units"
    assert ov["raw_axis_x"] == "income"
    assert all(0.0 <= v <= 1.0 for v in mesh.z_values)
    big = adapter.compute_3d_surface(pipe, _spec(resolution=200), n_features=4)
    assert big.resolution == 100


def test_registry_still_resolves_inner_for_simple_pipeline() -> None:
    """Backward compat: scaler-only pipelines stay with the inner adapter.

    Mirrors Phase 03 unwrapping: callers resolve via the terminal estimator,
    and the pipeline adapter itself must NOT claim pure scaler prefixes.
    """
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    import app.domains.model_engine.adapters.linear.linear_regression_adapter as _lin  # noqa: F401

    rng = np.random.default_rng(7)
    X = (rng.random((20, 3)) * 10).astype(np.float32)
    y = (rng.random(20) * 5).astype(np.float32)
    pipe = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())])
    pipe.fit(X, y)
    assert SklearnPipelineAdapter.can_handle(pipe) is False
    _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(pipe)
    assert ModelRegistry.resolve_adapter(final_est).algorithm_key == "linear_regression"
