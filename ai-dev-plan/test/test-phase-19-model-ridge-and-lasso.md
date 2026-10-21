# Test Blueprint — Phase 19: Ridge & Lasso Regression Adapters & 3D Regularized Weight Vectors

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-19-model-ridge-and-lasso.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-19-model-ridge-and-lasso.md) เพื่อยืนยันความถูกต้องของคลาส `RegularizedLinearAdapter` (`backend/app/domains/model_engine/adapters/linear/regularized.py`), คอมโพเนนต์ระนาบ `RegularizedPlane3D` (`frontend/src/domains/stage3d/components/surfaces/RegularizedPlane3D.tsx`), และลูกศรเวกเตอร์ `ForceVectorArrows3D` (`frontend/src/domains/stage3d/components/actors/ForceVectorArrows3D.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทุกตัว**:
  - **กลุ่ม $L_2$ (Ridge)**: `Ridge`, `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`
  - **กลุ่ม $L_1$ (Lasso)**: `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`
- **รูปแบบงาน (Tasks)**:
  - Single-target Regression, Multi-output Regression (`coef_` ขนาด `(M, D)`), และ Binary/Multiclass Classification (`RidgeClassifier`)
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - การแยกเซตฟีเจอร์ `active_features` ($|w_j| > 10^{-12}$) และ `zeroed_features` ($|w_j| \le 10^{-12}$)
  - การคำนวณค่า $\|w\|_1$, $\|w\|_2$, และ `sparsity_ratio`
  - การคำนวณระนาบ 3D แบบ Analytical Plane Evaluation โดยไม่ต้อง `np.tile` ตารางขนาดใหญ่

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **Prediction & Waterfall Additivity**:
   - สำหรับทุกรุ่นของ Ridge และ Lasso: $\hat{y}_t = b + \sum_{j=1}^{D} \phi_{t,j}$ โดยที่ $\phi_{t,j} = w_j x_{t,j}$ ต้องตรงกับ `model.predict(X)` ด้วยความแม่นยำ `atol=1e-6`
2. **Lasso Exact Zero Sparsity Partitioning**:
   - ฟีเจอร์ที่อยู่ใน `zeroed_features` ต้องมีค่า `feature_contributions[:, j] == 0.0` ทุกแถว 100%
3. **Analytical 3D Plane Parity**:
   - พื้นผิว `z_values` ที่คำนวณจากสมการย่อ `w[x]*xx + w[y]*yy + off_axis_sum` ต้องตรงกับผลลัพธ์จากการเรียก `model.predict(full_grid_X)` ทุกจุดบนตาราง $40 \times 40$ (`np.allclose(..., atol=1e-5)`)

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_regularized_linear_adapter.py`)
```python
import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import (
    Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV,
    Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
)
from app.domains.model_engine.adapters.linear.regularized import RegularizedLinearAdapter

FEATURE_NAMES = ["area_sqft", "rooms", "age_years", "noise_1", "noise_2"]

@pytest.fixture
def sparse_regression_data():
    X, y, coef = make_regression(
        n_samples=150, n_features=5, n_informative=2, coef=True,
        noise=0.5, random_state=42
    )
    return X, y

@pytest.mark.parametrize("model_cls,kwargs,expect_l1", [
    (Ridge, {"alpha": 2.0}, False),
    (RidgeCV, {"alphas": [0.1, 1.0, 10.0]}, False),
    (Lasso, {"alpha": 5.0}, True),
    (LassoCV, {"cv": 3}, True),
    (LassoLars, {"alpha": 1.0}, True),
    (LassoLarsCV, {"cv": 3}, True),
    (LassoLarsIC, {"criterion": "bic"}, True),
])
def test_ridge_and_lasso_variants_metadata_and_flow(sparse_regression_data, model_cls, kwargs, expect_l1):
    X, y = sparse_regression_data
    model = model_cls(**kwargs).fit(X, y)
    adapter = RegularizedLinearAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    assert (meta["penalty_type"] == "L1 (Lasso)") == expect_l1
    assert len(meta["active_features"]) + len(meta["zeroed_features"]) == 5
    assert 0.0 <= meta["sparsity_ratio"] <= 1.0

    flow = adapter.trace_internal_flow(X[:30])
    preds = np.asarray(flow.predictions)
    contribs = np.asarray(flow.feature_contributions)
    b = flow.base_intercepts[0]

    np.testing.assert_allclose(preds, model.predict(X[:30]), atol=1e-5)
    np.testing.assert_allclose(np.sum(contribs, axis=1) + b, preds, atol=1e-5)

    # Verify that pruned features have strictly 0 contribution
    for zero_feat in meta["zeroed_features"]:
        j_idx = FEATURE_NAMES.index(zero_feat)
        np.testing.assert_allclose(contribs[:, j_idx], 0.0, atol=1e-12)

def test_analytical_surface_mesh_matches_full_predict(sparse_regression_data):
    X, y = sparse_regression_data
    model = Lasso(alpha=10.0, random_state=42).fit(X, y)
    adapter = RegularizedLinearAdapter(model, FEATURE_NAMES)

    fixed = np.mean(X, axis=0)
    mesh = adapter.compute_surface_mesh(
        axis_x_idx=0, axis_y_idx=3,
        x_range=(-2.0, 2.0), y_range=(-2.0, 2.0),
        fixed_features=fixed, grid_resolution=20
    )
    z_analytical = np.asarray(mesh.z_values)

    # Reconstruct full grid to verify analytical shortcut accuracy
    xx, yy = np.meshgrid(mesh.x_grid, mesh.y_grid)
    full_grid = np.tile(fixed, (400, 1))
    full_grid[:, 0] = xx.ravel()
    full_grid[:, 3] = yy.ravel()
    z_expected = model.predict(full_grid).reshape(20, 20)

    np.testing.assert_allclose(z_analytical, z_expected, atol=1e-4)

def test_ridge_classifier_pseudo_proba_and_multiclass():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=4, random_state=42)
    model = RidgeClassifierCV(alphas=[0.1, 1.0]).fit(X, y)
    adapter = RegularizedLinearAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:15])
    assert flow.extra_telemetry["is_pseudo_proba"] is True
    probas = np.asarray(flow.probabilities)
    assert probas.shape == (15, 3)
    np.testing.assert_allclose(np.sum(probas, axis=1), np.ones(15), atol=1e-5)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ UI, 3D Vectors และ What-If Flatline Behavior
1. **`ForceVectorArrows3D` Direction & Magnitude**:
   - ทดสอบว่าลูกศรแกน $X$ (`#38bdf8`), ลูกศรแกน $Y$ (`#c084fc`), และลูกศรเวกเตอร์ลัพธ์ (`#facc15`) มีความยาวแปรผันตรงกับ $|w_x|, |w_y|$ และ $\sqrt{w_x^2 + w_y^2}$
2. **Lasso Sparse Axis Dimming & Warning Hologram**:
   - เมื่อ `auxiliary_grids.is_x_pruned === true` หรือ `is_y_pruned === true` คอมโพเนนต์ `RegularizedPlane3D` ต้องแสดงป้ายเตือน `"L1 PRUNED: Weight = 0.000 (No Effect on Output)"` และเปลี่ยนเส้นกริดแกนนั้นเป็นเส้นประสีเทา (`#475569`)
3. **What-If Sandbox Zero-Weight Invariance**:
   - เมื่อผู้ใช้ลากสไลเดอร์ของฟีเจอร์ที่อยู่ใน `zeroed_features` ค่าทำนาย $\hat{y}_{\text{whatif}}$ ของลูกบอลเงาต้องไม่เปลี่ยนแปลงเลยแม้แต่ `0.000001` พร้อม触发แอนิเมชันกระพริบที่ป้าย `L1 PRUNED`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/surfaces/__tests__/RegularizedPlane3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RegularizedPlane3D, evaluateWhatIfLinear } from "@/domains/stage3d/components/surfaces/RegularizedPlane3D";

describe("Phase 19: RegularizedPlane3D & Lasso Pruned Feature Behavior", () => {
  const mockSurfacePrunedY = {
    x_grid: [-2, 0, 2],
    y_grid: [-2, 0, 2],
    z_values: [
      [10, 20, 30],
      [10, 20, 30],
      [10, 20, 30],
    ],
    auxiliary_grids: {
      wx: 5.0,
      wy: 0.0,
      is_x_pruned: false,
      is_y_pruned: true,
    },
    boundary_threshold: null,
  };

  it("displays L1 PRUNED warning banner when an active 3D axis has weight = 0", () => {
    render(
      <RegularizedPlane3D
        surfaceMesh={mockSurfacePrunedY}
        axisXName="area_sqft"
        axisYName="noise_1"
        testMode={true}
      />
    );
    expect(screen.getByText(/L1 PRUNED: Weight = 0\.000/i)).toBeInTheDocument();
  });

  it("keeps predicted y_hat strictly invariant when dragging a Lasso-pruned What-If slider", () => {
    const coef = [0.84, 0.31, -0.42, 0.0, 0.0];
    const intercept = 180.0;
    const baseFeatures = [100.0, 3.0, 10.0, 1.5, -2.0];

    const yBefore = evaluateWhatIfLinear(coef, intercept, baseFeatures);
    // Change pruned feature index 3 from 1.5 to 9999.0
    const modifiedFeatures = [100.0, 3.0, 10.0, 9999.0, -2.0];
    const yAfter = evaluateWhatIfLinear(coef, intercept, modifiedFeatures);

    expect(yAfter).toBe(yBefore);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **Zero-Tiling Analytical Surface Speed & Memory Test**:
   - ทดสอบเรียก `compute_surface_mesh` บนโมเดลที่มีจำนวนฟีเจอร์สูงมาก ($D = 5,000$ ฟีเจอร์)
   - ยืนยันว่าการคำนวณด้วยสูตร Analytical Plane (`w[axis_x_idx] * xx + w[axis_y_idx] * yy + off_axis_sum`) ไม่มีการจองตารางขนาด $(1600 \times 5000)$ ใน RAM โดยใช้เวลาประมวลผลต่ำกว่า `5 ms` และใช้หน่วยความจำเพิ่มไม่เกิน `1 MB`
2. **การอัปเดต `ArrowHelper` โดยไม่สร้าง Object 3D ใหม่**:
   - ตรวจสอบใน `ForceVectorArrows3D.tsx` ว่าเมื่อสลับแกน $X, Y$ ระบบเรียกใช้ `arrowRef.current.setDirection()` และ `arrowRef.current.setLength()` บน Instance เดิม

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **`RidgeClassifier` ไม่มีเมธอด `predict_proba`** | ส่ง `RidgeClassifier().fit(X, y)` เข้า `trace_internal_flow` | ไม่เกิด `AttributeError` คืนค่า `probabilities` ที่แปลงผ่าน `expit`/`softmax` พร้อมตั้ง `extra_telemetry.is_pseudo_proba = True` |
| 2 | **`Lasso(alpha=1e6)` ตัดน้ำหนักทุกฟีเจอร์เป็น `0.0` ทั้งหมด** | เทรน `Lasso(alpha=1e6)` จน `coef_ == 0` ทุกตัว | `active_weight_count == 0`, `sparsity_ratio == 1.0`, แสดงป้ายเตือน `"All Features Pruned by High L1 Alpha"` และวาดระนาบขนานพื้นราบที่ $Z = b$ |
| 3 | **Multi-output Regression (`coef_` ขนาด `(M, D)`)** | เทรน `Ridge()` และ `Lasso()` บนเป้าหมาย `Y` ขนาด `(N, 3)` | `trace_internal_flow(X, target_idx=2)` และ `compute_surface_mesh(..., target_idx=2)` ดึงสมการของเป้าหมายที่ 2 ได้ถูกต้องโดยไม่เกิด `IndexError` |
| 4 | **`RidgeCV` / `LassoCV` เก็บค่าใน `alpha_` แทน `alpha`** | เรียก `extract_metadata()` บน `RidgeCV`, `LassoCV`, `LassoLarsCV` | ดึงค่า `alpha_` ออกมาเป็น `float` ได้สำเร็จทุกรุ่น |
| 5 | **Floating-Point Noise ใน Coordinate Descent (`w_j = 1e-16`)** | จำลองโมเดลที่มี `coef_[3] = -4.2e-15` | ถูกจัดอยู่ใน `zeroed_features` เนื่องจาก `abs(w_j) <= 1e-12` (`ZERO_TOL`) |
| 6 | **`LassoLarsIC` ใช้เกณฑ์ `aic` หรือ `bic`** | เทรน `LassoLarsIC(criterion='bic')` แล้วตรวจสอบ Metadata | แสดงข้อมูล `criterion: 'bic'` และค่า `alpha_` ที่ถูกเลือกอย่างถูกต้อง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Class Compatibility SLA**: ผ่านการทดสอบ `pytest` ครบทั้ง 9 คลาส (`Ridge`, `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`, `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`)
- [ ] **Analytical Surface SLA**: เวลาคำนวณ `compute_surface_mesh` ต้องต่ำกว่า `5 ms` แม้ในชุดข้อมูลที่มี $D = 5,000$ ฟีเจอร์
- [ ] **What-If Sparsity Gate**: การเลื่อนสไลเดอร์ของฟีเจอร์ใน `zeroed_features` ต้องทำให้ $\Delta \hat{y} = 0.0$ เป๊ะ 100%
- [ ] **Coverage Gate**: Code Coverage ของ `regularized.py`, `RegularizedPlane3D.tsx`, และ `ForceVectorArrows3D.tsx` ไม่ต่ำกว่า `95%`
