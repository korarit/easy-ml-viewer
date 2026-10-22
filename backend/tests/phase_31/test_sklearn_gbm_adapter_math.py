"""Phase 31 — sklearn GBM staged-residual adapter math identity + staged tests."""

from __future__ import annotations

import io
import warnings

import joblib
import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.boosting.sklearn_gbm import (
    MAX_TRAJECTORY_STAGES,
    TOP_STAGES,
    SklearnGBMClassifierAdapter,
    SklearnGBMRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(43)
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


def test_registry_resolves_gbm_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = GradientBoostingClassifier(n_estimators=10, random_state=1).fit(Xb, yb)
    reg = GradientBoostingRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "sklearn_gbm"), (reg, "sklearn_gbm_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN


def test_binary_single_column_waterfall_matches_raw(binary_data) -> None:
    Xb, yb = binary_data
    model = GradientBoostingClassifier(n_estimators=25, max_depth=2, random_state=1).fit(Xb, yb)
    # Edge #2: binary keeps a single (M, 1) tree column.
    assert model.estimators_.shape == (model.n_estimators_, 1)
    adapter = SklearnGBMClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:10])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:10]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:10])), rtol=1e-9
    )
    raw_final = np.asarray(list(model.staged_decision_function(Xb[:10]))[-1], dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(Xb[:10]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:10], list(range(10)), ["a", "b"])
    assert MAX_TRAJECTORY_STAGES == 50
    for i, tr in enumerate(traces):
        # Raw-margin identity: base + ΣΦ == staged_decision_function final.
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw_final[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr.base_value == pytest.approx(tr.extra_mechanics["base_raw_score"])
        mech = tr.extra_mechanics
        assert mech["model_family"] == "sklearn_gbm"
        assert mech["n_stages_total"] == model.n_estimators_
        assert mech["learning_rate"] == pytest.approx(0.1)
        assert len(mech["staged_predictions"]) <= MAX_TRAJECTORY_STAGES
        # Probability curve resolves through sklearn's own link function.
        assert mech["staged_predictions"][-1] == pytest.approx(float(proba[i]), abs=1e-4)
        assert len(mech["top_residual_stages"]) == min(TOP_STAGES, model.n_estimators_)
        for step in mech["top_residual_stages"]:
            assert set(step) >= {
                "stage",
                "leaf_index",
                "delta_raw",
                "cumulative_raw",
                "cumulative_pred",
            }
        assert mech["sampled_stage_indices"][0] == 0
        assert mech["sampled_stage_indices"][-1] == model.n_estimators_ - 1
        assert len(mech["leaf_indices_per_stage"]) == min(100, model.n_estimators_)


def test_binary_target_zero_negated_view(binary_data) -> None:
    # Edge #5: an explicit class-0 view negates the positive-class margin.
    Xb, yb = binary_data
    model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=1).fit(Xb, yb)
    raw_final = float(np.asarray(list(model.staged_decision_function(Xb[:1]))[-1]).ravel()[0])
    tr = SklearnGBMClassifierAdapter().trace_internal_flow(
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
    model = GradientBoostingClassifier(n_estimators=15, max_depth=2, random_state=1).fit(Xm, ym)
    # Edge #2: multiclass grows C trees per stage.
    assert model.estimators_.shape == (model.n_estimators_, 3)
    raw_final = np.asarray(list(model.staged_decision_function(Xm[:5]))[-1], dtype=np.float64)
    for k in range(3):
        traces = SklearnGBMClassifierAdapter().trace_internal_flow(
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
            assert tr.extra_mechanics["negated_target_zero_view"] is False
        proba_k = np.asarray(model.predict_proba(Xm[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            assert tr.extra_mechanics["staged_predictions"][-1] == pytest.approx(
                float(proba_k[i]), abs=1e-4
            )


def test_regressor_raw_identity_and_losses(reg_data) -> None:
    X, y = reg_data
    for loss in ["squared_error", "absolute_error", "huber", "quantile"]:
        model = GradientBoostingRegressor(loss=loss, n_estimators=15, random_state=1).fit(X, y)
        adapter = SklearnGBMRegressorAdapter()
        preds = np.asarray(model.predict(X[:5]), dtype=np.float64)
        staged_all = list(model.staged_predict(X[:5]))
        traces = adapter.trace_internal_flow(model, X[:5], list(range(5)), ["a", "b", "c"])
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(preds[i]),
                rtol=1e-9,
                atol=1e-9,
            )
            np.testing.assert_allclose(float(preds[i]), float(staged_all[-1][i]), rtol=1e-12)
            assert tr.extra_mechanics["is_regressor"] is True
            assert tr.extra_mechanics["staged_predictions"][-1] == pytest.approx(
                float(preds[i]), abs=1e-4
            )


def test_exponential_loss_uses_native_link(binary_data) -> None:
    # Edge #6: exponential-loss raw scores use a halved-logit link; the
    # probability curve must still come from staged_predict_proba.
    Xb, yb = binary_data
    model = GradientBoostingClassifier(loss="exponential", n_estimators=10, random_state=1).fit(
        Xb, yb
    )
    tr = SklearnGBMClassifierAdapter().trace_internal_flow(model, Xb[:2], [0, 1], ["a", "b"])[0]
    raw_final = float(np.asarray(list(model.staged_decision_function(Xb[:1]))[-1]).ravel()[0])
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), raw_final, rtol=1e-9, atol=1e-9
    )
    proba = float(np.asarray(model.predict_proba(Xb[:1]), dtype=np.float64)[0, 1])
    assert tr.extra_mechanics["staged_predictions"][-1] == pytest.approx(proba, abs=1e-4)


def test_init_zero_early_stopping_and_wrappers(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    # Edge #4: init='zero' resolves through _raw_predict_init.
    zero = GradientBoostingClassifier(n_estimators=10, random_state=1)
    zero.init_ = "zero"
    zero.fit(Xb, yb)
    tr = SklearnGBMClassifierAdapter().trace_internal_flow(zero, Xb[:2], [0, 1], ["a", "b"])[0]
    raw_final = float(np.asarray(list(zero.staged_decision_function(Xb[:1]))[-1]).ravel()[0])
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), raw_final, rtol=1e-9, atol=1e-9
    )
    # Edge #3: early stopping shrinks the real stage count.
    es = GradientBoostingClassifier(
        n_estimators=200, validation_fraction=0.2, n_iter_no_change=3, random_state=1
    ).fit(Xb, yb)
    assert es.estimators_.shape[0] < 200
    tr = SklearnGBMClassifierAdapter().trace_internal_flow(es, Xb[:1], [0], ["a", "b"])[0]
    assert tr.extra_mechanics["n_stages_total"] == es.estimators_.shape[0]
    raw_es = float(np.asarray(list(es.staged_decision_function(Xb[:1]))[-1]).ravel()[0])
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), raw_es, rtol=1e-9, atol=1e-9
    )
    # Pipeline + GridSearchCV-style wrappers resolve via unwrapping.
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("gbm", GradientBoostingClassifier(n_estimators=8, random_state=1))]
    )
    pipe.fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "sklearn_gbm"
    traces = SklearnGBMClassifierAdapter().trace_internal_flow(pipe, Xb[:2], [0, 1], ["a", "b"])
    assert [t.final_prediction for t in traces] == list(pipe.predict(Xb[:2]))
    ttr = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(n_estimators=8, random_state=1)
    )
    ttr.fit(Xr, yr)
    assert ModelRegistry.resolve_adapter(ttr).algorithm_key == "sklearn_gbm_regressor"
    traces = SklearnGBMRegressorAdapter().trace_internal_flow(ttr, Xr[:2], [0, 1], ["a", "b", "c"])
    preds = np.asarray(ttr.predict(Xr[:2]), dtype=np.float64)
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(preds[i]), rtol=1e-9, atol=1e-9
        )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        SklearnGBMClassifierAdapter().trace_internal_flow(pipe, Xb[:2], [0, 1], ["a", "b"])
    assert not [w for w in rec if "feature names" in str(w.message).lower()]


def test_trajectory_cap_and_surface_parity(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    big = GradientBoostingClassifier(n_estimators=250, random_state=1).fit(Xb, yb)
    traces = SklearnGBMClassifierAdapter().trace_internal_flow(big, Xb[:3], [0, 1, 2], ["a", "b"])
    for tr in traces:
        assert len(tr.extra_mechanics["staged_predictions"]) <= MAX_TRAJECTORY_STAGES
        assert len(tr.extra_mechanics["sampled_stage_indices"]) == MAX_TRAJECTORY_STAGES
        # Perf cap: only the first 100 stages walk decision_path, the rest
        # distribute uniformly — the raw identity still holds exactly.
        raw_final = float(np.asarray(list(big.staged_decision_function(Xb[[tr.row_index]]))[-1]).ravel()[0])
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), raw_final, rtol=1e-9, atol=1e-9
        )
    mesh = SklearnGBMClassifierAdapter().compute_3d_surface(big, _spec(), n_features=2)
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
    assert ov["model_family"] == "sklearn_gbm"
    assert ov["n_stages_total"] == big.n_estimators_
    assert ov["learning_rate"] == pytest.approx(0.1)
    assert len(ov["feature_importances"]) == 2

    reg = GradientBoostingRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    mesh_r = SklearnGBMRegressorAdapter().compute_3d_surface(
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
    assert SklearnGBMClassifierAdapter().compute_3d_surface(big, big_spec, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(big, buf)
    buf.seek(0)
    assert len(SklearnGBMClassifierAdapter().trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2
