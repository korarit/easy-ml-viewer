"""Phase 07 — evaluation endpoint E2E (sample regression + label harmonizer units)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.application.evaluation_use_case import build_class_index
from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _bootstrap_regression(client: TestClient, target: str | None = "selling_price"):
    ds = client.post("/api/v1/datasets/load-sample/house_prices").json()
    mdl = client.post("/api/v1/models/load-sample/house_price_linreg").json()
    auto = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": ds["dataset_id"], "model_id": mdl["model_id"]},
    ).json()
    items = auto["mapping"]["items"]
    commit = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "items": items,
            "target_column_name": target,
        },
    )
    assert commit.status_code == 200, commit.text
    body = commit.json()
    return ds, mdl, body["mapping"]


def test_regression_compute_e2e(client: TestClient):
    ds, mdl, mapping = _bootstrap_regression(client)
    res = client.post(
        "/api/v1/evaluations/compute",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "top_k": 50,
        },
    )
    assert res.status_code == 200, res.text
    summary = res.json()
    assert summary["task_type"] == "regression"
    assert summary["evaluated_row_count"] > 0
    reg = summary["regression"]
    assert reg is not None
    assert reg["evaluated_rows"] == summary["evaluated_row_count"]
    assert -1.0 <= reg["r2_score"] <= 1.0 or reg["scale_mismatch_warning"]
    assert len(reg["residual_hist_counts"]) == 16
    assert sum(reg["residual_hist_counts"]) == reg["evaluated_rows"]
    assert len(reg["top_worst_abs_error_indices"]) <= 50
    assert summary["ood_outlier_count"] >= 0
    assert isinstance(summary["prediction_distribution"]["pred_mean"], float)


def test_unlabeled_compute_returns_distribution_and_ood(client: TestClient):
    ds, mdl, mapping = _bootstrap_regression(client, target=None)
    res = client.post(
        "/api/v1/evaluations/compute",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "top_k": 50,
        },
    )
    assert res.status_code == 200, res.text
    summary = res.json()
    assert summary["regression"] is None
    assert summary["classification"] is None
    assert summary["prediction_distribution"]["pred_max"] >= summary["prediction_distribution"]["pred_min"]


def test_mapping_mismatch_rejected(client: TestClient):
    ds, mdl, mapping = _bootstrap_regression(client)
    res = client.post(
        "/api/v1/evaluations/compute",
        json={
            "dataset_id": "ds_nope0000",
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "top_k": 50,
        },
    )
    assert res.status_code == 422


def test_label_harmonizer_units():
    classes = [0, 1]
    assert build_class_index("Yes", ["Yes", "No"], None) == 0
    assert build_class_index("1", classes, None) == 1
    assert build_class_index(1, classes, None) == 1
    assert build_class_index("yes ", ["Yes", "No"], None) == 0
    assert build_class_index("Maybe", classes, None) is None
    assert build_class_index(None, classes, None) is None
    assert build_class_index(float("nan"), classes, None) is None
    # Explicit user map wins over auto-guess.
    assert build_class_index("Churn", [1, 0], {"Churn": 1, "Stay": 0}) == 0
    assert build_class_index("Stay", [1, 0], {"Churn": 1, "Stay": 0}) == 1


def test_classification_endpoint_with_synthetic_logreg(client: TestClient):
    """Full classification path through the use-case with a scratch adapter."""
    from sklearn.linear_model import LogisticRegression

    from app.application.evaluation_use_case import ComputeEvaluationRequest, EvaluationUseCase
    from app.domains.mapping.value_objects import FeatureMappingItem, ImputationStrategy, MatchTier
    from app.domains.model_engine.registry import ModelRegistry
    from app.infrastructure.ml_loaders.joblib_model_repository import JoblibModelRepository
    from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)

    from app.domains.model_engine.base_adapter import BaseModelAdapter
    from app.domains.model_engine.value_objects import (
        GridSpec3D,
        InternalFlowRowTrace,
        PredictionBatchResult,
        SurfaceMesh3DData,
        TaskType,
        Visual3DType,
    )

    @ModelRegistry.register("test_logreg_phase07")
    class _LogRegAdapter(BaseModelAdapter):
        task_type = TaskType.BINARY_CLASSIFICATION
        visual_3d_type = Visual3DType.PROBABILITY_SIGMOID_SURFACE

        @classmethod
        def can_handle(cls, estimator, file_ext=".joblib"):
            return isinstance(estimator, LogisticRegression) and getattr(
                estimator, "_phase07_test_tag", False
            )

        def extract_feature_weights(self, estimator, n_features):
            return [0.0] * n_features

        def predict_batch(self, model_obj, X_float32):
            proba = model_obj.predict_proba(np.asarray(X_float32, dtype=np.float32))
            preds = proba.argmax(axis=1)
            return PredictionBatchResult(
                predictions=[int(v) for v in preds],
                probabilities=[[float(a), float(b)] for a, b in proba],
                confidences=[float(max(a, b)) for a, b in proba],
            )

        def trace_internal_flow(self, model_obj, X_float32, row_indices, feature_names):
            return []

        def compute_3d_surface(self, model_obj, grid_spec, n_features):
            raise NotImplementedError

    try:
        clf._phase07_test_tag = True  # type: ignore[attr-defined]
        repo = JoblibModelRepository()
        model_id = "mdl_phase07test"
        repo._models[model_id] = clf

        df = pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "label": y})
        ds_repo = ParquetDatasetRepository()
        dataset_id = "ds_phase07test"
        ds_repo.save_dataset_parquet(dataset_id, df)

        map_repo_holder: dict = {}

        class _MapRepo:
            def get(self, _mid):
                return map_repo_holder["cfg"]

        from app.domains.mapping.value_objects import MappingConfiguration

        cfg = MappingConfiguration(
            mapping_id="map_phase07test",
            dataset_id=dataset_id,
            model_id=model_id,
            items=[
                FeatureMappingItem(
                    feature_index=0,
                    model_feature_name="f0",
                    dataset_column_name="f0",
                    match_tier=MatchTier.MANUAL,
                    match_confidence=1.0,
                    imputation_strategy=ImputationStrategy.MEDIAN,
                    resolved_fill_value=0.0,
                ),
                FeatureMappingItem(
                    feature_index=1,
                    model_feature_name="f1",
                    dataset_column_name="f1",
                    match_tier=MatchTier.MANUAL,
                    match_confidence=1.0,
                    imputation_strategy=ImputationStrategy.MEDIAN,
                    resolved_fill_value=0.0,
                ),
            ],
            target_column_name="label",
            is_valid=True,
        )
        map_repo_holder["cfg"] = cfg

        use_case = EvaluationUseCase(
            dataset_repo=ds_repo,
            model_use_case=ModelIngestUseCaseForTest(repo),
            mapping_repo=_MapRepo(),  # type: ignore[arg-type]
        )
        summary = use_case.compute(
            ComputeEvaluationRequest(
                dataset_id=dataset_id, model_id=model_id, mapping_id="map_phase07test"
            )
        )
        assert summary.classification is not None
        assert summary.classification.evaluated_rows == 120
        assert summary.classification.accuracy > 0.9
        assert len(summary.classification.cells) == 4
        assert len(summary.classification.most_uncertain_indices) <= 50
    finally:
        ModelRegistry._adapters.pop("test_logreg_phase07", None)


class ModelIngestUseCaseForTest:
    def __init__(self, repo):
        self._repo = repo

    def get_model(self, model_id):
        return self._repo.get(model_id)


def test_label_mismatch_returns_mapper_hint():
    """String CSV labels vs numeric classes_ -> needs_label_mapping with hint."""
    import pandas as pd

    from app.application.evaluation_use_case import ComputeEvaluationRequest, EvaluationUseCase
    from app.domains.mapping.value_objects import (
        FeatureMappingItem,
        ImputationStrategy,
        MappingConfiguration,
        MatchTier,
    )
    from app.domains.model_engine.base_adapter import BaseModelAdapter
    from app.domains.model_engine.registry import ModelRegistry
    from app.domains.model_engine.value_objects import (
        PredictionBatchResult,
        TaskType,
        Visual3DType,
    )
    from app.infrastructure.ml_loaders.joblib_model_repository import JoblibModelRepository
    from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    Xn = rng.normal(size=(40, 2)).astype(np.float32)
    yn = (Xn[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(Xn, yn)
    clf._phase07_mm_tag = True  # type: ignore[attr-defined]

    @ModelRegistry.register("test_logreg_mm_phase07")
    class _MMAdapter(BaseModelAdapter):
        task_type = TaskType.BINARY_CLASSIFICATION
        visual_3d_type = Visual3DType.PROBABILITY_SIGMOID_SURFACE

        @classmethod
        def can_handle(cls, estimator, file_ext=".joblib"):
            return isinstance(estimator, LogisticRegression) and getattr(
                estimator, "_phase07_mm_tag", False
            )

        def extract_feature_weights(self, estimator, n_features):
            return [0.0] * n_features

        def predict_batch(self, model_obj, X_float32):
            proba = model_obj.predict_proba(np.asarray(X_float32, dtype=np.float32))
            return PredictionBatchResult(
                predictions=[int(v) for v in proba.argmax(axis=1)],
                probabilities=[[float(a), float(b)] for a, b in proba],
            )

        def trace_internal_flow(self, model_obj, X_float32, row_indices, feature_names):
            return []

        def compute_3d_surface(self, model_obj, grid_spec, n_features):
            raise NotImplementedError

    try:
        repo = JoblibModelRepository()
        repo._models["mdl_mm07"] = clf
        df = pd.DataFrame(
            {"f0": Xn[:, 0], "f1": Xn[:, 1], "label": np.where(yn == 1, "Yes", "No")}
        )
        ds_repo = ParquetDatasetRepository()
        ds_repo.save_dataset_parquet("ds_mm07", df)
        cfg = MappingConfiguration(
            mapping_id="map_mm07",
            dataset_id="ds_mm07",
            model_id="mdl_mm07",
            items=[
                FeatureMappingItem(
                    feature_index=0, model_feature_name="f0", dataset_column_name="f0",
                    match_tier=MatchTier.MANUAL, match_confidence=1.0,
                    imputation_strategy=ImputationStrategy.MEDIAN, resolved_fill_value=0.0,
                ),
                FeatureMappingItem(
                    feature_index=1, model_feature_name="f1", dataset_column_name="f1",
                    match_tier=MatchTier.MANUAL, match_confidence=1.0,
                    imputation_strategy=ImputationStrategy.MEDIAN, resolved_fill_value=0.0,
                ),
            ],
            target_column_name="label",
            is_valid=True,
        )

        class _MapRepo:
            def get(self, _mid):
                return cfg

        use_case = EvaluationUseCase(
            dataset_repo=ds_repo,
            model_use_case=ModelIngestUseCaseForTest(repo),
            mapping_repo=_MapRepo(),  # type: ignore[arg-type]
        )
        # "Yes"/"No" do not auto-match numeric [0, 1] without an explicit map.
        summary = use_case.compute(
            ComputeEvaluationRequest(dataset_id="ds_mm07", model_id="mdl_mm07", mapping_id="map_mm07")
        )
        assert summary.needs_label_mapping is True
        assert summary.classification is None
        assert "Yes" in summary.raw_target_labels or "No" in summary.raw_target_labels

        # With an explicit user map the same data evaluates cleanly.
        summary2 = use_case.compute(
            ComputeEvaluationRequest(
                dataset_id="ds_mm07", model_id="mdl_mm07", mapping_id="map_mm07",
                target_label_map={"Yes": 1, "No": 0},
            )
        )
        assert summary2.needs_label_mapping is False
        assert summary2.classification is not None
        assert summary2.classification.evaluated_rows == 40
    finally:
        ModelRegistry._adapters.pop("test_logreg_mm_phase07", None)
