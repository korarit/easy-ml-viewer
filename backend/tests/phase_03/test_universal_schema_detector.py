"""UniversalSchemaDetector: embedded names, placeholders, fitted validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from app.core.exceptions import ModelLoadError
from app.domains.model_engine.schema_detector import UniversalSchemaDetector


def test_schema_detector_with_dataframe_feature_names() -> None:
    df = pd.DataFrame(
        {
            "area_sqm": [50.0, 80.0, 120.0],
            "bedrooms": [1.0, 2.0, 3.0],
            "distance_bts_km": [5.0, 2.0, 1.0],
        }
    )
    y = np.array([2.5e6, 4.0e6, 6.2e6], dtype=np.float32)
    model = LinearRegression().fit(df, y)

    schema = UniversalSchemaDetector.extract_schema(model)
    assert schema.n_features_in == 3
    assert [f.name for f in schema.features] == ["area_sqm", "bedrooms", "distance_bts_km"]
    assert all(not f.is_auto_generated_name for f in schema.features)
    assert [f.index for f in schema.features] == [0, 1, 2]


def test_schema_detector_fallback_placeholders_for_nameless_numpy_model() -> None:
    X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    y = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    model = LinearRegression().fit(X, y)

    assert not hasattr(model, "feature_names_in_")
    schema = UniversalSchemaDetector.extract_schema(model)
    assert schema.n_features_in == 2
    assert [f.name for f in schema.features] == ["feature_0", "feature_1"]
    assert all(f.is_auto_generated_name for f in schema.features)


def test_unfitted_model_raises_model_load_error() -> None:
    unfitted = LinearRegression()
    with pytest.raises(ModelLoadError) as exc_info:
        UniversalSchemaDetector.validate_is_fitted(unfitted)
    assert exc_info.value.code == "MODEL_NOT_FITTED"
