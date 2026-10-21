"""Phase 06 fixtures — small deterministic frames + FastAPI test client."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def housing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area_sqm": np.array([45.0, 120.0, 65.0, 95.0, 80.0], dtype=np.float32),
            "bedrooms": np.array([1, 4, 2, 3, 2], dtype=np.int32),
            "selling_price": np.array([2.1e6, 7.5e6, 3.2e6, 5.1e6, 4.0e6], dtype=np.float32),
        }
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
