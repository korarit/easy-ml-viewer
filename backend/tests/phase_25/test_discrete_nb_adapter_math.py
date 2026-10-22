"""Phase 25 — discrete Naive Bayes adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from app.domains.model_engine.adapters.probabilistic.discrete_nb import (
    DiscreteNBAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

VOCAB = ["word_free", "word_win", "word_offer", "word_meeting", "word_report", "word_lunch"]


@pytest.fixture()
def count_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(25)
    lam = np.array(
        [
            [4.0, 3.0, 3.5, 0.2, 0.3, 0.2],
            [0.1, 0.2, 1.5, 3.0, 2.5, 1.0],
            [0.5, 0.5, 2.0, 1.0, 1.0, 2.5],
        ]
    )
    y = rng.integers(0, 3, size=90).astype(np.int32)
    X = np.stack([rng.poisson(lam[k]).astype(np.float32) for k in y])
    return X, y


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(26)
    n = 120
    y = (rng.random(n) < 0.4).astype(np.int32)
    p1 = np.array([0.8, 0.7, 0.05, 0.05, 0.6, 0.4])
    p0 = np.array([0.05, 0.02, 0.6, 0.7, 0.2, 0.5])
    X = np.stack([rng.random(6) < (p1 if k else p0) for k in y]).astype(np.float32)
    return X, y


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -2.0,
        "x_max": 8.0,
        "y_min": -1.0,
        "y_max": 6.0,
        "resolution": 20,
        "fixed_feature_medians": {2: 1.0, 3: 0.5, 4: 0.0, 5: 2.0},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def test_registry_resolves_all_discrete_variants(count_data, binary_data) -> None:
    Xc, yc = count_data
    Xb, yb = binary_data
    models = [
        MultinomialNB().fit(Xc, yc),
        BernoulliNB(binarize=0.5).fit(Xb, yb),
        ComplementNB().fit(Xc, (yc > 0).astype(int)),
        CategoricalNB().fit((Xb > 0.5).astype(int), yb),
    ]
    for model in models:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "discrete_nb"
        assert adapter.visual_3d_type == Visual3DType.DISCRETE_NB_TOWERS


def test_multinomial_waterfall_matches_joint(count_data) -> None:
    X, y = count_data
    model = MultinomialNB(alpha=1.0).fit(X, y)
    adapter = DiscreteNBAdapter()
    batch = adapter.predict_batch(model, X[:10])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:10]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(X[:10])), rtol=1e-6
    )
    joint = np.asarray(model.predict_joint_log_proba(X[:10]), dtype=np.float64)
    # Multiclass: per-target-class waterfall closes over the joint.
    for k in range(3):
        traces = adapter.trace_internal_flow(
            model, X[:10], list(range(10)), VOCAB, target_class_idx=k
        )
        assert len(traces) == 10
        for i, tr in enumerate(traces):
            assert set(tr.feature_contributions) == set(VOCAB)
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(joint[i, k]),
                rtol=1e-6,
                atol=1e-6,
            )
            mech = tr.extra_mechanics
            assert mech["nb_variant"] == "multinomial"
            assert mech["alpha"] == pytest.approx(1.0)
            assert mech["requires_non_negative_input"] is True
            assert mech["joint_log_probs"] == pytest.approx(joint[i].tolist(), rel=1e-6)
            np.testing.assert_allclose(
                np.asarray(mech["probabilities"]),
                np.asarray(model.predict_proba(X[i : i + 1])).ravel(),
                rtol=1e-5,
                atol=1e-9,
            )
    # Binary slice: log-odds-ratio identity.
    binary = MultinomialNB().fit(X, (y > 0).astype(int))
    joint_b = np.asarray(binary.predict_joint_log_proba(X[:10]), dtype=np.float64)
    traces_b = adapter.trace_internal_flow(binary, X[:10], list(range(10)), VOCAB)
    for i, tr in enumerate(traces_b):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(joint_b[i, 1] - joint_b[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_bernoulli_absence_term_identity(binary_data) -> None:
    X, y = binary_data
    model = BernoulliNB(binarize=0.5).fit(X, y)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), VOCAB)
    joint = np.asarray(model.predict_joint_log_proba(X[:10]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        flags = tr.extra_mechanics["presence_flags"]
        assert set(flags) == set(VOCAB)
        assert set(flags.values()) <= {"present", "absent"}
        # An all-absent row still scores (absence penalty is non-trivial).
        assert tr.extra_mechanics["nb_variant"] == "bernoulli"
        assert tr.extra_mechanics["binarize"] == pytest.approx(0.5)
    # Naive X*log-prob-only scoring would NOT match the native joint.
    flp = np.asarray(model.feature_log_prob_)
    naive = X[:10].astype(float) @ flp.T
    assert not np.allclose(naive[:, 1] - naive[:, 0], joint[:, 1] - joint[:, 0], atol=1e-3)


def test_bernoulli_binarize_none(binary_data) -> None:
    X, y = binary_data
    model = BernoulliNB(binarize=None).fit(X, y)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:4], [0, 1, 2, 3], VOCAB)
    joint = np.asarray(model.predict_joint_log_proba(X[:4]), dtype=np.float64)
    assert traces[0].extra_mechanics["binarize"] is None
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_complement_matches_native_sign(count_data) -> None:
    X, y = count_data
    yb = (y > 0).astype(int)
    model = ComplementNB(alpha=1.0).fit(X, yb)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), VOCAB)
    joint = np.asarray(model.predict_joint_log_proba(X[:10]), dtype=np.float64)
    assert traces[0].extra_mechanics["nb_variant"] == "complement"
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_categorical_parity_and_unseen_clamp() -> None:
    rng = np.random.default_rng(28)
    X = np.stack(
        [
            rng.integers(0, 3, size=120),
            rng.integers(0, 2, size=120),
            rng.integers(0, 4, size=120),
            rng.integers(0, 2, size=120),
        ],
        axis=1,
    ).astype(np.float32)
    y = ((X[:, 0] == 2) | (X[:, 2] >= 2)).astype(np.int32)
    names = ["color", "size", "region", "plan"]
    model = CategoricalNB().fit(X, y)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), names)
    joint = np.asarray(model.predict_joint_log_proba(X[:10]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr.extra_mechanics["unseen_category_clamped"] is False
    # Unseen category (e.g. color=9) is clamped, flagged, and finite.
    far = np.array([[9.0, 0.0, 99.0, 1.0]], dtype=np.float32)
    tr_far = adapter.trace_internal_flow(model, far, [0], names)[0]
    assert tr_far.extra_mechanics["unseen_category_clamped"] is True
    assert np.all(np.isfinite(list(tr_far.feature_contributions.values())))


def test_alpha_zero_no_nan(count_data) -> None:
    X, y = count_data
    model = MultinomialNB(alpha=0.0).fit(X, y)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:5], list(range(5)), VOCAB)
    for tr in traces:
        vals = list(tr.feature_contributions.values()) + [tr.base_value]
        assert np.all(np.isfinite(vals))


def test_high_dim_flow_pruning() -> None:
    rng = np.random.default_rng(29)
    d = 600
    X = rng.poisson(1.0, size=(30, d)).astype(np.float32)
    y = (X[:, :50].sum(axis=1) > X[:, 50:100].sum(axis=1)).astype(np.int32)
    names = [f"tok_{i}" for i in range(d)]
    model = MultinomialNB().fit(X, y)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:3], [0, 1, 2], names)
    joint = np.asarray(model.predict_joint_log_proba(X[:3]), dtype=np.float64)
    for i, tr in enumerate(traces):
        mech = tr.extra_mechanics
        assert mech["contributions_pruned"] is True
        assert mech["pruned_feature_count"] == d - 60
        assert len(tr.feature_contributions) == 60
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()) + mech["pruned_remainder_sum"],
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_pipeline_and_negative_clamp(count_data) -> None:
    X, y = count_data
    pipe = Pipeline(
        [("shift", FunctionTransformer(lambda v: v * 1.0)), ("nb", MultinomialNB())]
    ).fit(X, y)
    adapter = DiscreteNBAdapter()
    traces = adapter.trace_internal_flow(pipe, X[:3], [0, 1, 2], VOCAB)
    assert [t.final_prediction for t in traces] == list(pipe.predict(X[:3]))
    assert all(v is not None for v in adapter.extract_feature_weights(pipe[-1], 6))
    # Negative rows are clamped, not crashed.
    neg = np.full((2, 6), -3.0, dtype=np.float32)
    traces_neg = adapter.trace_internal_flow(pipe, neg, [0, 1], VOCAB)
    for tr in traces_neg:
        assert np.all(np.isfinite(list(tr.feature_contributions.values())))

    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert len(adapter.trace_internal_flow(restored, X[:2], [0, 1], VOCAB)) == 2


def test_surface_non_negative_domain_and_bernoulli_passthrough(
    count_data, binary_data
) -> None:
    Xc, yc = count_data
    Xb, yb = binary_data
    adapter = DiscreteNBAdapter()
    multi = MultinomialNB().fit(Xc, yc)
    mesh = adapter.compute_3d_surface(multi, _spec(), n_features=6)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    assert mesh.x_grid[0] >= 0.0 and mesh.y_grid[0] >= 0.0
    assert mesh.extra_overlays["clamped_non_negative"] is True
    assert mesh.extra_overlays["boundary_threshold"] == 0.5
    assert np.all(np.isfinite(np.asarray(mesh.z_values)))

    bern = BernoulliNB(binarize=0.5).fit(Xb, yb)
    mesh_b = adapter.compute_3d_surface(bern, _spec(x_min=-1.0, y_min=-1.0), n_features=6)
    assert mesh_b.x_grid[0] == pytest.approx(-1.0)
    assert mesh_b.extra_overlays["clamped_non_negative"] is False

    big = _spec(resolution=200)
    mesh_big = adapter.compute_3d_surface(multi, big, n_features=6)
    assert mesh_big.resolution == 100

    # Categorical grid on integer categories stays finite.
    rng_cat = np.random.default_rng(28)
    X4 = np.stack(
        [
            rng_cat.integers(0, 3, size=120),
            rng_cat.integers(0, 2, size=120),
            rng_cat.integers(0, 4, size=120),
            rng_cat.integers(0, 2, size=120),
        ],
        axis=1,
    ).astype(np.float32)
    y4 = ((X4[:, 0] == 2) | (X4[:, 2] >= 2)).astype(np.int32)
    cat = CategoricalNB().fit(X4, y4)
    mesh_c = adapter.compute_3d_surface(
        cat,
        GridSpec3D(
            axis_x_feature_index=0,
            axis_y_feature_index=2,
            x_min=0.0,
            x_max=2.0,
            y_min=0.0,
            y_max=3.0,
            resolution=20,
            fixed_feature_medians={1: 0.0, 3: 1.0},
        ),
        n_features=4,
    )
    assert np.all(np.isfinite(np.asarray(mesh_c.z_values)))
