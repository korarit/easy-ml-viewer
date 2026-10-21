# Test Phase 04: First Simple Model (Linear Regression) — End-to-End Vertical Slice Proof of Concept

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 04)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-04-first-model-linear-regression-slice.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-04-first-model-linear-regression-slice.md) เพื่อตรวจสอบความถูกต้องเชิงคณิตศาสตร์และประสิทธิภาพของ **`LinearRegressionAdapter` (Ordinary Least Squares - OLS)** ครอบคลุมการแยกพจน์ผลคูณรายตัวแปร ($\text{Contribution}_i = w_i \cdot x_i$) ที่เมื่อรวมกับค่า `intercept` ($b$) แล้วต้องเท่ากับค่าทำนาย $\hat{y}$ ของ `model.predict(X)` ทุกประการ, การสร้างพื้นผิวระนาบ 3 มิติ (`compute_3d_surface` ขนาด `50 x 50` ชนิด `float32`), การรองรับโมเดล 1 มิติ (1D Ribbon), Multi-Target Regression, และสคริปต์สร้างไฟล์ตัวอย่าง `generate_sample_linear.py`

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Linear Adapter & Sample Generator**:
  - `app.domains.model_engine.adapters.linear.linear_regression_adapter.LinearRegressionAdapter`
  - `scripts.generate_sample_linear.generate_house_price_sample`
  - REST Endpoints: `POST /api/v1/models/{model_id}/trace-preview`, `POST /api/v1/models/{model_id}/surface-preview`
- **Frontend Linear Slice Inspector Components**:
  - `LinearFormulaBanner.tsx`, `LinearContributionCard.tsx`, `LinearSurfaceSandbox.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_04/
├── conftest.py
├── test_linear_regression_adapter_math.py
├── test_linear_surface_mesh_3d.py
├── test_generate_sample_linear_script.py
└── test_linear_preview_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_04/test_linear_regression_adapter_math.py
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.domains.model_engine.adapters.linear.linear_regression_adapter import LinearRegressionAdapter
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

def test_linear_trace_contributions_sum_exact_to_prediction() -> None:
    """สมการสำคัญ: base_value + sum(feature_contributions.values()) ต้องเท่ากับ model.predict(X) เป๊ะ"""
    X = np.array([
        [80.0, 2.0, 2.2],
        [150.0, 4.0, 0.5],
        [35.0, 1.0, 12.0],
    ], dtype=np.float32)
    # y = 150,000 + 42,500*x0 + 350,000*x1 - 185,000*x2
    y = (150000.0 + 42500.0 * X[:, 0] + 350000.0 * X[:, 1] - 185000.0 * X[:, 2]).astype(np.float32)

    model = LinearRegression().fit(X, y)
    adapter = LinearRegressionAdapter()

    batch_res = adapter.predict_batch(model, X)
    traces = adapter.trace_internal_flow(
        model_obj=model,
        X_float32=X,
        row_indices=[0, 1, 2],
        feature_names=["area_sqm", "bedrooms", "distance_bts_km"],
    )

    for idx, trace in enumerate(traces):
        expected_pred = batch_res.predictions[idx]
        reconstructed_pred = trace.base_value + sum(trace.feature_contributions.values())
        assert np.isclose(reconstructed_pred, expected_pred, rtol=1e-5, atol=1.0)
        assert np.isclose(trace.final_prediction, expected_pred, rtol=1e-5, atol=1.0)

    # ตรวจสอบค่าผลคูณของแถวที่ 0: area_sqm=80 * 42500 = +3,400,000
    assert np.isclose(traces[0].feature_contributions["area_sqm"], 3400000.0, rtol=1e-4)
    assert np.isclose(traces[0].feature_contributions["distance_bts_km"], -407000.0, rtol=1e-4)

def test_linear_adapter_with_standard_scaler_pipeline() -> None:
    """ทดสอบกรณี LinearRegression อยู่หลัง StandardScaler ใน Pipeline ผลรวมต้องยังคงตรงกับ predict()"""
    rng = np.random.default_rng(123)
    X = rng.uniform(10.0, 200.0, size=(50, 3)).astype(np.float32)
    y = (500.0 + 12.0 * X[:, 0] - 8.5 * X[:, 1] + 3.2 * X[:, 2]).astype(np.float32)

    pipe = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())]).fit(X, y)
    adapter = LinearRegressionAdapter()
    traces = adapter.trace_internal_flow(pipe, X[:5], [0, 1, 2, 3, 4], ["f0", "f1", "f2"])
    preds = pipe.predict(X[:5])

    for i, tr in enumerate(traces):
        total = tr.base_value + sum(tr.feature_contributions.values())
        assert np.isclose(total, preds[i], rtol=1e-4)

def test_multi_target_and_no_intercept_linear_regression() -> None:
    """ทดสอบ Edge Case: โมเดลที่มี y หลายคอลัมน์ (2D coef_) และโมเดลที่ fit_intercept=False"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    Y_multi = np.column_stack([2.0 * X[:, 0] + X[:, 1], -1.0 * X[:, 0] + 3.0 * X[:, 1]]).astype(np.float32)

    model_multi = LinearRegression(fit_intercept=False).fit(X, Y_multi)
    adapter = LinearRegressionAdapter()

    weights = adapter.extract_feature_weights(model_multi, n_features=2)
    assert len(weights) == 2
    traces = adapter.trace_internal_flow(model_multi, X[:1], [0], ["x0", "x1"])
    assert traces[0].base_value == 0.0

def test_compute_3d_surface_hyperplane_and_1d_model_ribbon() -> None:
    """ทดสอบการสร้างพื้นผิว 3D ขนาด 50x50 = 2,500 จุด ทั้งกรณี 3 Features และกรณี 1 Feature (1D Ribbon)"""
    X_1d = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    y_1d = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    model_1d = LinearRegression().fit(X_1d, y_1d)

    adapter = LinearRegressionAdapter()
    spec = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=0,
        x_min=10.0,
        x_max=30.0,
        y_min=-5.0,
        y_max=5.0,
        resolution=50,
        fixed_feature_medians={},
    )
    surface = adapter.compute_3d_surface(model_1d, spec, n_features=1)
    assert surface.visual_type == Visual3DType.FLAT_HYPERPLANE
    assert surface.resolution == 50
    assert len(surface.x_grid) == 50
    assert len(surface.y_grid) == 50
    assert len(surface.z_values) == 2500
```

```python
# backend/tests/phase_04/test_generate_sample_linear_script.py
from pathlib import Path
import joblib
import pandas as pd
from scripts.generate_sample_linear import generate_house_price_sample

def test_sample_generator_creates_valid_csv_and_joblib(tmp_path: Path) -> None:
    """ยืนยันว่าสคริปต์สร้างไฟล์ sample_houses.csv (1,000 แถว) และ sample_house_price_linear.joblib ได้จริง"""
    generate_house_price_sample(tmp_path)
    csv_path = tmp_path / "sample_houses.csv"
    mdl_path = tmp_path / "sample_house_price_linear.joblib"

    assert csv_path.exists() and csv_path.stat().st_size > 10_000
    assert mdl_path.exists() and mdl_path.stat().st_size > 500

    df = pd.read_csv(csv_path)
    assert len(df) == 1000
    assert list(df.columns) == ["area_sqm", "bedrooms", "distance_bts_km", "selling_price"]

    model = joblib.load(mdl_path)
    assert list(model.feature_names_in_) == ["area_sqm", "bedrooms", "distance_bts_km"]
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/model_engine/__tests__/
├── LinearFormulaBanner.test.tsx
├── LinearContributionCard.test.tsx
└── LinearSurfaceSandbox.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/model_engine/__tests__/LinearContributionCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { LinearContributionCard } from '../components/LinearContributionCard';
import { LinearSurfaceSandbox } from '../components/LinearSurfaceSandbox';

describe('Linear Regression Contribution Breakdown & 3D Surface Sandbox UI', () => {
  it('แสดงสมการแยกพจน์ Base Intercept, ค่าบวกสีเขียว, ค่าลบสีแดง และ Residual Error ได้ถูกต้อง', () => {
    const trace = {
      row_index: 0,
      base_value: 150000.0,
      final_prediction: 3843000.0,
      actual_target: 3900000.0,
      residual_error: 57000.0,
      feature_contributions: {
        area_sqm: 3400000.0,
        bedrooms: 700000.0,
        distance_bts_km: -407000.0,
      },
    };

    render(<LinearContributionCard trace={trace} />);
    expect(screen.getByText(/Base Intercept/i)).toBeInTheDocument();
    expect(screen.getByText(/\+3,400,000\.00/i)).toBeInTheDocument();
    expect(screen.getByText(/-407,000\.00/i)).toBeInTheDocument();
    expect(screen.getByText(/3,843,000\.00/i)).toBeInTheDocument();
    expect(screen.getByText(/\+57,000\.00/i)).toBeInTheDocument();
  });

  it('สลับแกน X กับแกน Y อัตโนมัติเมื่อผู้ใช้เลือกตัวแปรเดียวกันซ้ำใน Dropdown', () => {
    const onAxesChange = vi.fn();
    render(
      <LinearSurfaceSandbox
        features={['area_sqm', 'bedrooms', 'distance_bts_km']}
        axisX={0}
        axisY={2}
        onAxesChange={onAxesChange}
      />
    );

    const axisYSelect = screen.getByLabelText(/Axis Y/i);
    // เลือก index 0 ซึ่งชนกับ Axis X ปัจจุบัน -> ต้องสั่งสลับเป็น (2, 0)
    fireEvent.change(axisYSelect, { target: { value: '0' } });
    expect(onAxesChange).toHaveBeenCalledWith(2, 0);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบความเร็ว Vectorized Surface Generation และการจำกัด `resolution <= 100` (`pytest`)
```python
# backend/tests/phase_04/test_surface_performance_and_memory.py
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from app.domains.model_engine.adapters.linear.linear_regression_adapter import LinearRegressionAdapter
from app.domains.model_engine.value_objects import GridSpec3D

def test_surface_generation_under_10ms_and_float32_contiguous() -> None:
    """ยืนยันว่าการสร้างพื้นผิว 50x50 (2,500 จุด) สำหรับโมเดล 20 Features ใช้เวลา < 10ms"""
    X = np.random.rand(100, 20).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    model = LinearRegression().fit(X, y)
    adapter = LinearRegressionAdapter()

    spec = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        resolution=50,
        fixed_feature_medians={i: 0.5 for i in range(2, 20)},
    )

    t0 = time.perf_counter()
    surface = adapter.compute_3d_surface(model, spec, n_features=20)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 10.0
    assert len(surface.z_values) == 2500
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 04)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Multi-Target Linear Regression (`coef_` เป็น 2D Array)** | `test_multi_target_and_no_intercept_linear_regression()` | เทรน `LinearRegression` ด้วย `Y` ขนาด `(N, 2)` | ดึง `weights` ของ `target_idx=0` ได้โดยไม่เกิด `ValueError: shapes not aligned` |
| **2** | **โมเดลตั้งค่า `fit_intercept=False` ตอนเทรน** | `test_multi_target_and_no_intercept_linear_regression()` | เทรนด้วย `LinearRegression(fit_intercept=False)` | `trace.base_value == 0.0` และ UI แสดงป้าย `No Intercept (b = 0)` |
| **3** | **โมเดลมีเพียง 1 ตัวแปรนำเข้า (`n_features_in_ == 1`)** | `test_compute_3d_surface_hyperplane_and_1d_model_ribbon()` | เทรนโมเดลด้วย `X` ขนาด `(N, 1)` แล้วสั่ง `compute_3d_surface` | สร้างระนาบ 3D Ribbon ขนาด `2,500 จุด` ที่ความสูงแปรผันตามแกน $X$ และคงที่ตามแกน $Y$ |
| **4** | **โมเดลอยู่ใน Pipeline ที่มี `StandardScaler`** | `test_linear_adapter_with_standard_scaler_pipeline()` | รัน `trace_internal_flow` กับ `Pipeline([StandardScaler, LinearRegression])` | ระบบแปลง `X` ด้วย `pipeline[:-1].transform` ก่อนคูณ `weights` ทำให้ผลรวมตรงกับ `.predict()` 100% |
| **5** | **ค่า Collinearity สูงทำให้สัมประสิทธิ์มีค่าบวกลบสลับกันระดับล้าน** | `test_impact_bar_percentage_normalization_with_collinear_weights()` | ส่งค่า `feature_contributions = {"f1": 1e7, "f2": -1e7, "f3": 500}` เข้า Component | ความยาวหลอด `% Impact Bar` คำนวณจาก $|c_i| / \sum |c_j|$ อยู่ในช่วง `0% - 100%` ไม่ทะลุกรอบ |
| **6** | **ผู้ใช้เลือกตัวแปรเดียวกันเป็นทั้งแกน X และแกน Y** | `LinearSurfaceSandbox` swap test | เลือก Dropdown แกน $Y$ ให้ตรงกับแกน $X$ ปัจจุบัน | ระบบสลับค่า `(axisX, axisY) -> (axisY, axisX)` อัตโนมัติ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Mathematical Identity Guarantee**: สำหรับทุกแถวข้อมูล $|\text{base\_value} + \sum \text{contributions}_i - \hat{y}| < 10^{-4}$
- [ ] **3D Surface Compute Latency SLA**: `compute_3d_surface` สำหรับความละเอียด `50 x 50` ต้องเสร็จสิ้นภายใน `< 10 ms`
- [ ] **Sample Script Reproducibility**: รัน `generate_sample_linear.py` ด้วย Seed `42` ได้ผลลัพธ์ไฟล์ `.csv` และ `.joblib` เหมือนเดิมทุกครั้ง 100%
