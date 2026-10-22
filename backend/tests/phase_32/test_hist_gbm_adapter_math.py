"""Phase 32 — HistGBM quantized-bin adapter math identity + staged tests."""

from __future__ import annotations

import io
import warnings

import joblib
import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.boosting.hist_gbm import (
    MAX_TRAJECTORY_POINTS,
    TOP_STAGES,
    HistGBMClassifierAdapter,
    HistGBMRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(44)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    X[::9, 0] = np.nan  # native missing-value routing (edge #2)
    X[::17, 1] = np.nan
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(45)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(46)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    return X, (X[:, 0] * 2.0 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -4.0,
        "x_max": 6.0,
        "y_min": -3.0,
        "y_max": 5.0,
        "resolution": 20,
        "fixed_feature_medians": {2: 0.5, 3: 1.0},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def test_registry_resolves_hist_gbm_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = HistGradientBoostingClassifier(max_iter=10, random_state=1).fit(Xb, yb)
    reg = HistGradientBoostingRegressor(max_iter=10, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "hist_gbm"), (reg, "hist_gbm_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
        assert adapter.supports_nan_input is True


def test_binary_waterfall_matches_raw_with_nan(binary_data) -> None:
    Xb, yb = binary_data
    model = HistGradientBoostingClassifier(max_iter=25, random_state=1).fit(Xb, yb)
    # Edge #5: real iteration count comes from _predictors, not max_iter.
    assert len(model._predictors) == model.n_iter_
    assert len(model._predictors[0]) == 1  # binary: one tree per iter
    adapter = HistGBMClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:12])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:12]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:12])), rtol=1e-9
    )
    raw_final = np.asarray(
        list(model.staged_decision_function(Xb[:12]))[-1], dtype=np.float64
    ).ravel()
    proba = np.asarray(model.predict_proba(Xb[:12]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:12], list(range(12)), ["a", "b"])
    assert MAX_TRAJECTORY_POINTS == 50
    # Rows 0 and 9 carry NaNs and must still resolve exactly.
    assert traces[0].extra_mechanics["has_missing_values"] is True
    for i, tr in enumerate(traces):
        # Raw-margin identity (leaf values already carry lr; edge #4).
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw_final[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr.base_value == pytest.approx(tr.extra_mechanics["base_raw_score"])
        mech = tr.extra_mechanics
        assert mech["model_family"] == "hist_gbm"
        assert mech["link_function"] == "logit"
        assert mech["n_iter_total"] == model.n_iter_
        assert mech["learning_rate"] == pytest.approx(model.learning_rate)
        assert len(mech["staged_predictions"]) <= MAX_TRAJECTORY_POINTS
        assert mech["staged_predictions"][-1] == pytest.approx(float(proba[i]), abs=1e-4)
        assert len(mech["top_residual_stages"]) == min(TOP_STAGES, model.n_iter_)
        for step in mech["top_residual_stages"]:
            assert set(step) >= {
                "stage",
                "leaf_index",
                "delta_raw",
                "cumulative_raw",
                "cumulative_pred",
                "nan_routed_feature_idx",
            }
        assert mech["sampled_stage_indices"][0] == 0
        assert mech["sampled_stage_indices"][-1] == model.n_iter_ - 1
        assert len(mech["leaf_indices_per_stage"]) == min(100, model.n_iter_)
    # Gain importances (edge #1) sum to 1 and match extract_feature_weights.
    gains = traces[0].extra_mechanics["gain_feature_importances"]
    assert len(gains) == 2
    assert sum(gains) == pytest.approx(1.0)
    assert adapter.extract_feature_weights(model, 2) == pytest.approx(gains)


def test_binary_target_zero_negated_view(binary_data) -> None:
    Xb, yb = binary_data
    model = HistGradientBoostingClassifier(max_iter=10, random_state=1).fit(Xb, yb)
    raw_final = float(np.asarray(list(model.staged_decision_function(Xb[:1]))[-1]).ravel()[0])
    tr = HistGBMClassifierAdapter().trace_internal_flow(
        model, Xb[:1], [0], ["a", "b"], target_class_idx=0
    )[0]
    assert tr.extra_mechanics["negated_target_zero_view"] is True
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), -raw_final, rtol=1e-9, atol=1e-9
    )
    proba0 = float(np.asarray(model.predict_proba(Xb[:1]), dtype=np.float64)[0, 0])
    assert tr.extra_mechanics["staged_predictions"][-1] == pytest.approx(proba0, abs=1e-4)


def test_multiclass_per_class_columns(multi_data) -> None:
    Xm, ym = multi_data
    model = HistGradientBoostingClassifier(max_iter=15, random_state=1).fit(Xm, ym)
    assert len(model._predictors[0]) == 3  # multiclass: C trees per iter
    raw_final = np.asarray(list(model.staged_decision_function(Xm[:5]))[-1], dtype=np.float64)
    for k in range(3):
        traces = HistGBMClassifierAdapter().trace_internal_flow(
            model, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=k
        )
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(raw_final[i, k]),
                rtol=1e-9,
                atol=1e-9,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k
            assert tr.extra_mechanics["link_function"] == "softmax"
        proba_k = np.asarray(model.predict_proba(Xm[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            assert tr.extra_mechanics["staged_predictions"][-1] == pytest.approx(
                float(proba_k[i]), abs=1e-4
            )


def test_regressor_losses_with_log_link(reg_data) -> None:
    X, y = reg_data
    for loss, yy in [
        ("squared_error", y),
        ("absolute_error", y),
        ("gamma", np.abs(y) + 0.5),
        ("poisson", np.abs(y) + 0.5),
        ("quantile", y),
    ]:
        kwargs: dict[str, object] = {"loss": loss, "max_iter": 15, "random_state": 1}
        if loss == "quantile":
            kwargs["quantile"] = 0.9
        model = HistGradientBoostingRegressor(**kwargs).fit(  # type: ignore[arg-type]
            X, yy.astype(np.float32)
        )
        adapter = HistGBMRegressorAdapter()
        preds = np.asarray(model.predict(X[:5]), dtype=np.float64)
        staged_all = list(model.staged_predict(X[:5]))
        traces = adapter.trace_internal_flow(model, X[:5], list(range(5)), ["a", "b", "c"])
        for i, tr in enumerate(traces):
            raw_sum = tr.base_value + sum(tr.feature_contributions.values())
            if tr.extra_mechanics["link_function"] == "log":
                # Log-link losses: predict == exp(raw).
                np.testing.assert_allclose(float(np.exp(raw_sum)), float(preds[i]), rtol=1e-9)
            else:
                np.testing.assert_allclose(raw_sum, float(preds[i]), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(float(preds[i]), float(staged_all[-1][i]), rtol=1e-12)
            assert tr.extra_mechanics["is_regressor"] is True
            assert tr.extra_mechanics["staged_predictions"][-1] == pytest.approx(
                float(preds[i]), abs=1e-4
            )


def test_categorical_monotonic_and_wrappers() -> None:
    rng = np.random.default_rng(47)
    n = 300
    cat = rng.integers(0, 6, size=n).astype(np.float64)
    num = rng.normal(size=n)
    X = np.column_stack([cat, num])
    X[::11, 1] = np.nan
    y = ((cat >= 3).astype(int) ^ (np.nan_to_num(num) > 0).astype(int)).astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=10,
        categorical_features=[True, False],
        monotonic_cst=[0, 1],
        random_state=1,
    ).fit(X, y)
    adapter = HistGBMClassifierAdapter()
    traces = adapter.trace_internal_flow(
        model, X[:8].astype(np.float32), list(range(8)), ["cat", "num"]
    )
    raw_final = np.asarray(
        list(model.staged_decision_function(X[:8].astype(np.float32)))[-1], dtype=np.float64
    ).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw_final[i]),
            rtol=1e-9,
            atol=1e-9,
        )
    assert traces[0].extra_mechanics["categorical_feature_idx"] == [0]
    assert traces[0].extra_mechanics["monotonic_cst"] == [0, 1]
    # Default (non-categorical) model exposes an empty categorical index.
    plain = HistGradientBoostingClassifier(max_iter=5, random_state=1).fit(X, y)
    tr = HistGBMClassifierAdapter().trace_internal_flow(
        plain, X[:1].astype(np.float32), [0], ["cat", "num"]
    )[0]
    assert tr.extra_mechanics["categorical_feature_idx"] == []
    # Pipeline + TTR wrappers resolve through unwrapping.
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("hgb", HistGradientBoostingClassifier(max_iter=8, random_state=1))]
    )
    Xb = np.nan_to_num(X).astype(np.float32)
    pipe.fit(Xb, y)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "hist_gbm"
    ptraces = HistGBMClassifierAdapter().trace_internal_flow(pipe, Xb[:2], [0, 1], ["cat", "num"])
    assert [t.final_prediction for t in ptraces] == list(pipe.predict(Xb[:2]))
    Xr = rng.normal(size=(60, 3)).astype(np.float32)
    yr = Xr[:, 0] * 2.0
    ttr = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(max_iter=8, random_state=1)
    )
    ttr.fit(Xr, yr)
    assert ModelRegistry.resolve_adapter(ttr).algorithm_key == "hist_gbm_regressor"
    ttraces = HistGBMRegressorAdapter().trace_internal_flow(ttr, Xr[:2], [0, 1], ["a", "b", "c"])
    preds = np.asarray(ttr.predict(Xr[:2]), dtype=np.float64)
    for i, t in enumerate(ttraces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(preds[i]), rtol=1e-9, atol=1e-9
        )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        HistGBMClassifierAdapter().trace_internal_flow(pipe, Xb[:2], [0, 1], ["cat", "num"])
    assert not [w for w in rec if "feature names" in str(w.message).lower()]


def test_early_stopping_trajectory_cap_and_surface(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    # Pure-noise target plateaus the validation score so early stopping
    # actually halts before max_iter (edge #5).
    rng_es = np.random.default_rng(9)
    X_es = rng_es.normal(size=(600, 4)).astype(np.float32)
    y_es = rng_es.integers(0, 2, size=600)
    es = HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, n_iter_no_change=5, random_state=1
    ).fit(X_es, y_es)
    assert es.n_iter_ < 200
    tr = HistGBMClassifierAdapter().trace_internal_flow(
        es, X_es[:1], [0], ["a", "b", "c", "d"]
    )[0]
    assert tr.extra_mechanics["n_iter_total"] == es.n_iter_
    raw_es = float(np.asarray(list(es.staged_decision_function(X_es[:1]))[-1]).ravel()[0])
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), raw_es, rtol=1e-9, atol=1e-9
    )
    big = HistGradientBoostingClassifier(max_iter=150, random_state=1).fit(Xb, yb)
    traces = HistGBMClassifierAdapter().trace_internal_flow(big, Xb[:3], [0, 1, 2], ["a", "b"])
    for tr in traces:
        assert len(tr.extra_mechanics["staged_predictions"]) <= MAX_TRAJECTORY_POINTS
        assert len(tr.extra_mechanics["sampled_stage_indices"]) == min(big.n_iter_, MAX_TRAJECTORY_POINTS)
        raw_final = float(
            np.asarray(list(big.staged_decision_function(Xb[[tr.row_index]]))[-1]).ravel()[0]
        )
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), raw_final, rtol=1e-9, atol=1e-9
        )
    mesh = HistGBMClassifierAdapter().compute_3d_surface(big, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    grid = np.zeros((400, 2), dtype=np.float32)
    xs = np.linspace(-4.0, 6.0, 20, dtype=np.float32)
    ys = np.linspace(-3.0, 5.0, 20, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    expected = np.asarray(big.predict_proba(grid), dtype=np.float64)[:, 1]
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-9)
    ov = mesh.extra_overlays
    assert ov["model_family"] == "hist_gbm"
    assert ov["n_iter_total"] == big.n_iter_
    assert ov["learning_rate"] == pytest.approx(big.learning_rate)
    assert len(ov["gain_feature_importances"]) == 2
    # Bin-grid overlay payload: downsampled thresholds for both axes.
    assert set(ov["axis_bin_thresholds"]) == {"x", "y"}
    assert len(ov["axis_bin_thresholds"]["x"]) <= 64
    assert len(ov["axis_bin_thresholds"]["y"]) <= 64

    reg = HistGradientBoostingRegressor(max_iter=10, random_state=1).fit(Xr, yr)
    mesh_r = HistGBMRegressorAdapter().compute_3d_surface(
        reg,
        GridSpec3D(
            axis_x_feature_index=0,
            axis_y_feature_index=1,
            x_min=-2.0,
            x_max=2.0,
            y_min=-2.0,
            y_max=2.0,
            resolution=20,
            fixed_feature_medians={2: 0.0},
        ),
        n_features=3,
    )
    assert mesh_r.extra_overlays["is_regressor"] is True
    assert mesh_r.extra_overlays["boundary_threshold"] is None

    big_spec = _spec(resolution=200)
    assert HistGBMClassifierAdapter().compute_3d_surface(big, big_spec, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(big, buf)
    buf.seek(0)
    assert len(HistGBMClassifierAdapter().trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2
