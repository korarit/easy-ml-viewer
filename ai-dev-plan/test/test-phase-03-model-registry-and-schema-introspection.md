# Test Phase 03: Model Engine Core — Abstract Adapter Contract, Plugin Registry & Schema Detector

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 03)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-03-model-registry-and-schema-introspection.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-03-model-registry-and-schema-introspection.md) เพื่อยืนยันความถูกต้องของ Bounded Context **`domains/model_engine/`** ครอบคลุมสัญญามาตรฐาน `BaseModelAdapter`, ระบบลงทะเบียน `@ModelRegistry.register(...)`, ระบบสกัดโครงสร้างตัวแปร `UniversalSchemaDetector` (ทั้งกรณีมี `feature_names_in_` และกรณีเทรนจาก NumPy Array เปล่าที่ต้องสร้าง `feature_0..n`), การคลี่ `Pipeline` และ `GridSearchCV`, การดักจับ `NotFittedError` และ `InconsistentVersionWarning`, ตลอดจนการโหลดโมเดลแบบประหยัดหน่วยความจำด้วย `mmap_mode='r'`

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Model Engine Domain**:
  - `app.domains.model_engine.value_objects` (`TaskType`, `Visual3DType`, `FeatureRequirement`, `GridSpec3D`, `PredictionBatchResult`, `InternalFlowRowTrace`, `SurfaceMesh3DData`)
  - `app.domains.model_engine.base_adapter.BaseModelAdapter`
  - `app.domains.model_engine.registry.ModelRegistry`
  - `app.domains.model_engine.schema_detector.UniversalSchemaDetector`
  - `app.infrastructure.ml_loaders.joblib_model_repository.JoblibModelRepository`
  - REST Endpoints: `POST /api/v1/models/upload`, `GET /api/v1/models/supported`
- **Frontend Model Engine Domain**:
  - `src/domains/model_engine/store/useModelStore.ts`
  - `ModelUploadDropzone.tsx`, `ModelArchitectureCard.tsx`, `ModelFeatureSchemaTable.tsx`, `HyperparameterAccordion.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_03/
├── conftest.py
├── test_model_registry_contract.py
├── test_universal_schema_detector.py
├── test_pipeline_and_gridsearch_unwrapping.py
└── test_model_upload_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_03/test_universal_schema_detector.py
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.core.exceptions import ModelLoadError

def test_schema_detector_with_dataframe_feature_names() -> None:
    """ทดสอบโมเดลที่เทรนด้วย DataFrame ต้องดึง feature_names_in_ ออกมาตรงตามลำดับ Index"""
    df = pd.DataFrame({
        "area_sqm": [50.0, 80.0, 120.0],
        "bedrooms": [1.0, 2.0, 3.0],
        "distance_bts_km": [5.0, 2.0, 1.0],
    })
    y = np.array([2.5e6, 4.0e6, 6.2e6], dtype=np.float32)
    model = LinearRegression().fit(df, y)

    schema = UniversalSchemaDetector.extract_schema(model)
    assert schema.n_features_in == 3
    assert [f.name for f in schema.features] == ["area_sqm", "bedrooms", "distance_bts_km"]
    assert all(not f.is_auto_generated_name for f in schema.features)
    assert [f.index for f in schema.features] == [0, 1, 2]

def test_schema_detector_fallback_placeholders_for_nameless_numpy_model() -> None:
    """ทดสอบโมเดลที่เทรนจาก NumPy Array ล้วน (ไม่มี feature_names_in_) ต้องสร้าง feature_0..n อัตโนมัติ"""
    X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    y = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    model = LinearRegression().fit(X, y)

    assert not hasattr(model, "feature_names_in_")
    schema = UniversalSchemaDetector.extract_schema(model)
    assert schema.n_features_in == 2
    assert [f.name for f in schema.features] == ["feature_0", "feature_1"]
    assert all(f.is_auto_generated_name for f in schema.features)

def test_unwrap_pipeline_and_grid_search_cv() -> None:
    """ทดสอบการคลี่ GridSearchCV และ Pipeline เพื่อดึง Estimator หลักและรายชื่อ Pipeline Steps"""
    X = np.random.rand(20, 3).astype(np.float32)
    y = np.random.rand(20).astype(np.float32)
    pipe = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())])
    grid = GridSearchCV(pipe, param_grid={"ols__fit_intercept": [True, False]}, cv=2)
    grid.fit(X, y)

    unwrapped_obj, final_est, steps = UniversalSchemaDetector.unwrap_model_container(grid)
    assert isinstance(final_est, LinearRegression)
    assert steps == ["StandardScaler", "LinearRegression"]
    adapter = ModelRegistry.resolve_adapter(final_est)
    assert adapter.algorithm_key == "linear_regression"

def test_unfitted_model_raises_model_load_error() -> None:
    """ทดสอบการอัปโหลดโมเดลเปล่าที่ยังไม่ได้ .fit() ต้องโยน ModelLoadError รหัส MODEL_NOT_FITTED"""
    unfitted = LinearRegression()
    with pytest.raises(ModelLoadError) as exc_info:
        UniversalSchemaDetector.validate_is_fitted(unfitted)
    assert exc_info.value.code == "MODEL_NOT_FITTED"
```

```python
# backend/tests/phase_03/test_model_upload_endpoints.py
import io
import warnings
import joblib
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression
from app.main import app

@pytest.mark.asyncio
async def test_upload_model_endpoint_captures_version_mismatch_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """ทดสอบว่าเมื่อเกิด InconsistentVersionWarning ตอนโหลดโมเดล ระบบจะจับใส่ฟิลด์ warnings[] โดยไม่พัง"""
    X = np.array([[50.0, 2.0], [80.0, 3.0]], dtype=np.float32)
    y = np.array([3.0e6, 4.5e6], dtype=np.float32)
    model = LinearRegression().fit(X, y)

    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)

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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        files = {"file": ("house_ols.joblib", buf, "application/octet-stream")}
        resp = await client.post("/api/v1/models/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm_key"] == "linear_regression"
        assert data["n_features_in"] == 2
        assert len(data["warnings"]) >= 1
        assert "1.4.0" in data["warnings"][0]
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/model_engine/__tests__/
├── useModelStore.test.ts
├── ModelArchitectureCard.test.tsx
└── ModelFeatureSchemaTable.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/model_engine/__tests__/ModelArchitectureCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { ModelArchitectureCard } from '../components/ModelArchitectureCard';
import { ModelFeatureSchemaTable } from '../components/ModelFeatureSchemaTable';
import { useModelStore } from '../store/useModelStore';

const mockModelMetadata = {
  model_id: 'mdl_01',
  filename: 'house_price_ols.joblib',
  sha256_hash: '4a8b9c1291e2',
  algorithm_key: 'linear_regression',
  display_name: 'Linear Regression (OLS)',
  family_name: 'Linear Models',
  task_type: 'regression',
  visual_3d_type: 'flat_hyperplane',
  is_pipeline: true,
  pipeline_steps: ['StandardScaler', 'LinearRegression'],
  n_features_in: 2,
  classes: null,
  intercept_or_base: 150000.0,
  features: [
    {
      index: 0,
      name: 'feature_0',
      expected_dtype: 'float32',
      is_auto_generated_name: true,
      learned_weight_or_importance: 42500.0,
    },
    {
      index: 1,
      name: 'bedrooms',
      expected_dtype: 'float32',
      is_auto_generated_name: false,
      learned_weight_or_importance: 350000.0,
    },
  ],
  hyperparameters: { fit_intercept: true, copy_X: true },
  warnings: ['Model was trained on scikit-learn 1.4.0, running on 1.5.2'],
};

describe('Step 2 Model Architecture Card & Feature Schema Inspector', () => {
  beforeEach(() => {
    useModelStore.setState({
      currentModel: mockModelMetadata,
      isLoadingModel: false,
      modelError: null,
    });
  });

  it('แสดงป้ายชื่อโมเดล, ลำดับ Pipeline Steps, และกล่องเตือน InconsistentVersionWarning สีเหลือง', () => {
    render(<ModelArchitectureCard />);
    expect(screen.getByText('Linear Regression (OLS)')).toBeInTheDocument();
    expect(screen.getByText(/StandardScaler/i)).toBeInTheDocument();
    expect(screen.getByText(/scikit-learn 1\.4\.0/i)).toBeInTheDocument();
  });

  it('แสดงป้าย Auto-Generated Placeholder สำหรับฟีเจอร์ที่ไม่มีชื่อฝังมา และเปิดดู Hyperparameters ได้ด้วย Alt+H', () => {
    render(<ModelFeatureSchemaTable />);
    expect(screen.getByText('feature_0')).toBeInTheDocument();
    expect(screen.getByText(/Auto-Generated Placeholder/i)).toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'h', altKey: true });
    expect(screen.getByText(/fit_intercept/i)).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบ LRU Model Cache Eviction และการบังคับ `n_jobs = 1` (`pytest`)
```python
# backend/tests/phase_03/test_model_memory_and_njobs.py
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from app.infrastructure.ml_loaders.joblib_model_repository import JoblibModelRepository

def test_model_repository_enforces_single_thread_and_lru_eviction(tmp_path) -> None:
    """ทดสอบว่าโมเดลที่โหลดเข้ามาถูกปรับ n_jobs=1 เสมอ และ LRU Cache เก็บไม่เกิน 2 โมเดลใน RAM"""
    repo = JoblibModelRepository(cache_dir=tmp_path, max_models_in_ram=2)
    X = np.random.rand(50, 4).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)

    for i in range(3):
        mdl = LinearRegression(n_jobs=-1).fit(X, y)
        p = tmp_path / f"mdl_{i}.joblib"
        joblib.dump(mdl, p)
        loaded_obj, _ = repo.load_and_cache(model_id=f"id_{i}", file_path=p)
        assert getattr(loaded_obj, "n_jobs", 1) == 1

    # ต้องเหลือใน RAM เพียง 2 ตัวล่าสุดคือ id_1 และ id_2
    assert len(repo.active_model_ids()) == 2
    assert "id_0" not in repo.active_model_ids()
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 03)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **เกิด `InconsistentVersionWarning` ตอนโหลดไฟล์ `.pkl` / `.joblib`** | `test_upload_model_endpoint_captures_version_mismatch_warning()` | จำลองการโยน `InconsistentVersionWarning` ระหว่างเรียก `joblib.load()` | API คืนค่า `200 OK` พร้อมบันทึกข้อความเตือนลงใน `warnings: list[str]` |
| **2** | **ผู้ใช้อัปโหลดโมเดลเปล่าที่ยังไม่ได้เทรน (`NotFittedError`)** | `test_unfitted_model_raises_model_load_error()` | อัปโหลด `LinearRegression()` ที่ยังไม่เคยเรียก `.fit(X, y)` | คืนค่า HTTP `400` รหัส `MODEL_NOT_FITTED` พร้อมข้อความแนะนำชัดเจน |
| **3** | **โมเดลไม่มี `feature_names_in_` (เทรนจาก `np.ndarray` ล้วน)** | `test_schema_detector_fallback_placeholders_for_nameless_numpy_model()` | เทรนโมเดลด้วย `np.ndarray` ขนาด `(20, 2)` โดยไม่มีชื่อคอลัมน์ | ระบบสร้างชื่อ `feature_0`, `feature_1` และตั้งค่า `is_auto_generated_name=True` |
| **4** | **โมเดลถูกห่ออยู่ใน `Pipeline` หรือ `GridSearchCV`** | `test_unwrap_pipeline_and_grid_search_cv()` | ห่อ `StandardScaler` + `LinearRegression` ไว้ใน `GridSearchCV` แล้วเซฟเป็น `.joblib` | ระบบดึง `best_estimator_` และระบุ `pipeline_steps = ["StandardScaler", "LinearRegression"]` ได้ถูกต้อง |
| **5** | **ผู้ใช้อัปโหลดไฟล์ `.pkl` ที่ไม่ใช่โมเดล แต่เป็น DataFrame หรือ Dict** | `test_reject_dataframe_or_dict_uploaded_as_model()` | เซฟ `pd.DataFrame` เป็นไฟล์ `.pkl` แล้วอัปโหลดเข้า `/api/v1/models/upload` | คืนค่า HTTP `400` รหัส `NOT_A_VALID_MODEL_ARTIFACT` พร้อมแนะนำให้อัปโหลดใน Step 1 |
| **6** | **โมเดลมี `n_jobs=-1` ติดมา ทำให้เครื่องค้างเวลาเรียกทำนายถี่ๆ** | `test_model_repository_enforces_single_thread_and_lru_eviction()` | เซฟโมเดลที่ตั้งค่า `n_jobs=-1` แล้วโหลดผ่าน `JoblibModelRepository` | แอตทริบิวต์ `n_jobs` ถูกเขียนทับเป็น `1` ทันทีที่โหลดเข้าหน่วยความจำ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Open-Closed Registry Contract**: สามารถสร้าง Dummy Adapter ใหม่และติด `@ModelRegistry.register("dummy")` โดยไม่ต้องแก้โค้ดใน `UniversalSchemaDetector` หรือ `model_endpoints.py`
- [ ] **Schema Extraction Speed SLA**: การโหลดและสกัด Schema ของโมเดลขนาด `< 50 MB` ต้องใช้เวลา `< 100 ms`
- [ ] **LRU RAM Cap**: จำนวนอ็อบเจกต์โมเดลที่ถือครองอยู่ใน Python Heap ต้องไม่เกิน `2 โมเดล` ตลอดเวลา
