# Test Blueprint — Phase 21: SGD Classifier & Regressor Adapter & 3D Margin Boundary Plane

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-21-model-sgd-classifier-regressor.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-21-model-sgd-classifier-regressor.md) เพื่อยืนยันความถูกต้องของคลาส `SGDAdapter` (`backend/app/domains/model_engine/adapters/linear/sgd.py`), คอมโพเนนต์ระนาบขอบเขตคู่ขนาน `MarginCorridorPlanes3D` (`frontend/src/domains/stage3d/components/surfaces/MarginCorridorPlanes3D.tsx`), และแผงตรวจสอบ `SgdMarginInspector` (`frontend/src/domains/hud/components/inspectors/SgdMarginInspector.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทุกตัว**:
  - `sklearn.linear_model.SGDClassifier` (ครบทุก Loss: `hinge`, `log_loss`, `modified_huber`, `squared_hinge`, `perceptron`)
  - `sklearn.linear_model.SGDRegressor` (ครบทุก Loss: `squared_error`, `huber`, `epsilon_insensitive`, `squared_epsilon_insensitive`)
  - `sklearn.linear_model.PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, และ `Perceptron`
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - คะแนนการตัดสินใจดิบ $f(\mathbf{x}_t) = \mathbf{w}^T \mathbf{x}_t + b$
  - ระยะห่างทางเรขาคณิตแบบมีเครื่องหมาย $d_{\text{geom}}(\mathbf{x}_t) = \frac{f(\mathbf{x}_t)}{\|\mathbf{w}\|_2}$
  - การระบุสถานะจุดที่ตกอยู่ในเขตเสี่ยง `inside_margin_mask` ($|f(\mathbf{x}_t)| < 1.0$)
  - การป้องกัน `AttributeError: predict_proba` เมื่อ `loss='hinge'` หรือ `'perceptron'`

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **Loss-Aware Probability vs Margin Score Handling**:
   - เมื่อ `loss in ("log_loss", "modified_huber")`: `has_native_proba` ต้องเป็น `True` และ `probabilities` ต้องตรงกับ `model.predict_proba(X)`
   - เมื่อ `loss in ("hinge", "squared_hinge", "perceptron")`: `has_native_proba` ต้องเป็น `False`, ไม่มีการเรียก `model.predict_proba(X)` โดยตรง และคำนวณ Pseudo-Confidence จาก `expit`/`softmax`
2. **Signed Geometric Distance Accuracy**:
   - ยืนยันว่า `extra_telemetry["geometric_distances"]` เท่ากับ `decision_function(X) / np.linalg.norm(model.coef_, axis=-1)` ทุกแถว (`atol=1e-5`)
3. **Parallel Margin Corridor Surface Mesh ($\pm 1$ / $\pm \epsilon$)**:
   - ใน Classifier: `z_plus_margin - z_values == +1.0` และ `z_minus_margin - z_values == -1.0`
   - ใน Regressor: `z_plus_margin - z_values == +model.epsilon`

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_sgd_adapter.py`)
```python
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    SGDClassifier, SGDRegressor,
    PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron
)
from app.domains.model_engine.adapters.linear.sgd import SGDAdapter

FEATURE_NAMES = ["feat_vel", "feat_press", "feat_temp", "feat_vib"]

@pytest.fixture
def clf_data():
    return make_classification(n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=42)

@pytest.fixture
def reg_data():
    return make_regression(n_samples=120, n_features=4, n_informative=3, noise=0.5, random_state=42)

@pytest.mark.parametrize("model_cls,kwargs,expect_native_proba", [
    (SGDClassifier, {"loss": "hinge"}, False),
    (SGDClassifier, {"loss": "log_loss"}, True),
    (SGDClassifier, {"loss": "modified_huber"}, True),
    (SGDClassifier, {"loss": "squared_hinge"}, False),
    (SGDClassifier, {"loss": "perceptron"}, False),
    (PassiveAggressiveClassifier, {"C": 0.5}, False),
    (Perceptron, {"max_iter": 500}, False),
])
def test_sgd_classifier_all_losses_and_geometric_distance(clf_data, model_cls, kwargs, expect_native_proba):
    X, y = clf_data
    model = model_cls(random_state=42, **kwargs).fit(X, y)
    adapter = SGDAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    assert meta["has_native_proba"] is expect_native_proba

    flow = adapter.trace_internal_flow(X[:25])
    scores = np.asarray(flow.decision_scores)[:, 0]
    contribs = np.asarray(flow.feature_contributions)
    geom_dists = np.asarray(flow.extra_telemetry["geometric_distances"])[:, 0]
    inside_mask = np.asarray(flow.extra_telemetry["inside_margin_mask"])

    # 1. Decision score & Waterfall additivity
    expected_scores = model.decision_function(X[:25])
    np.testing.assert_allclose(scores, expected_scores, atol=1e-5)
    np.testing.assert_allclose(np.sum(contribs, axis=1) + flow.base_intercepts[0], scores, atol=1e-5)

    # 2. Geometric distance f(x) / ||w||_2
    w_norm = np.linalg.norm(model.coef_[0])
    np.testing.assert_allclose(geom_dists, expected_scores / w_norm, atol=1e-5)

    # 3. Inside margin corridor mask (|f(x)| < 1.0)
    np.testing.assert_array_equal(inside_mask, np.abs(expected_scores) < 1.0)

    # 4. Native proba vs Pseudo-proba
    probas = np.asarray(flow.probabilities)
    assert probas.shape == (25, 2)
    if expect_native_proba:
        np.testing.assert_allclose(probas, model.predict_proba(X[:25]), atol=1e-5)

@pytest.mark.parametrize("model_cls,kwargs", [
    (SGDRegressor, {"loss": "squared_error"}),
    (SGDRegressor, {"loss": "huber", "epsilon": 0.25}),
    (SGDRegressor, {"loss": "epsilon_insensitive", "epsilon": 0.5}),
    (PassiveAggressiveRegressor, {"epsilon": 0.3}),
])
def test_sgd_regressor_and_epsilon_corridor_mesh(reg_data, model_cls, kwargs):
    X, y = reg_data
    model = model_cls(random_state=42, **kwargs).fit(X, y)
    adapter = SGDAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:20])
    preds = np.asarray(flow.predictions)
    contribs = np.asarray(flow.feature_contributions)
    np.testing.assert_allclose(preds, model.predict(X[:20]), atol=1e-5)
    np.testing.assert_allclose(np.sum(contribs, axis=1) + flow.base_intercepts[0], preds, atol=1e-5)

    mesh = adapter.compute_surface_mesh(0, 1, (-2.0, 2.0), (-2.0, 2.0), np.mean(X, axis=0), grid_resolution=20)
    z_main = np.asarray(mesh.z_values)
    z_plus = np.asarray(mesh.auxiliary_grids["z_plus_margin"])
    eps = getattr(model, "epsilon", 0.1)
    np.testing.assert_allclose(z_plus - z_main, eps, atol=1e-5)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `MarginCorridorPlanes3D` และ `SgdMarginInspector`
1. **3D Parallel Margin Corridor ($\pm 1$ Planes) & Orthogonal Laser Line**:
   - ทดสอบว่า `MarginCorridorPlanes3D` แสดงแผ่นระนาบหลัก ($f(\mathbf{x}) = 0$) พร้อมระนาบคู่ขนาน $+1$ และ $-1$ เมื่อเปิดสวิตช์ `Show ±1 Margin Planes`
   - ทดสอบการคำนวณจุดปลายของเส้นเลเซอร์ตั้งฉาก (Orthogonal Projection Point บนระนาบ $f(\mathbf{x}) = 0$)
2. **Margin Zone Warning Aura & Status Transitions**:
   - เมื่อ $|f(\mathbf{x}_t)| < 1.0$ แผง HUD ต้องแสดงป้ายเตือน `"INSIDE MARGIN CORRIDOR"` และเปิดวงแหวนเตือนภัยสีเหลืองรอบลูกบอล 3 มิติ
   - เมื่อ `has_native_proba === false` แผง HUD ต้องแสดง `"N/A (Hinge Loss)"` คู่กับค่า `Pseudo-Conf`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/surfaces/__tests__/MarginCorridorPlanes3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SgdMarginInspector } from "@/domains/hud/components/inspectors/SgdMarginInspector";
import { computeOrthogonalPlaneFootprint } from "@/domains/stage3d/components/surfaces/MarginCorridorPlanes3D";

describe("Phase 21: SgdMarginInspector & Orthogonal Margin Projection", () => {
  it("flags INSIDE MARGIN CORRIDOR and disables native proba badge when loss='hinge'", () => {
    const mockMeta = {
      algorithm: "SGDClassifier",
      loss: "hinge",
      has_native_proba: false,
      penalty: "elasticnet",
      alpha: 0.0001,
      total_weight_updates_t: 45000,
    };
    const mockFrame = {
      frameIndex: 315,
      decisionScore: 0.64,
      geometricDistance: 0.285,
      insideMargin: true,
      pseudoConfidence: 0.655,
      contributions: [1.12, -0.18],
      intercept: -0.3,
    };

    render(<SgdMarginInspector metadata={mockMeta} frame={mockFrame} featureNames={["vel", "press"]} />);
    expect(screen.getByText(/INSIDE MARGIN CORRIDOR/i)).toBeInTheDocument();
    expect(screen.getByText(/N\/A \(Hinge Loss\)/i)).toBeInTheDocument();
    expect(screen.getByText(/\+0\.285 units/i)).toBeInTheDocument();
  });

  it("computes exact orthogonal projection footprint from 3D point onto plane wx*x + wy*y - z + b = 0", () => {
    // Plane: 1*x + 0*y - 1*z + 0 = 0 (45-degree plane z = x)
    const foot = computeOrthogonalPlaneFootprint([2.0, 3.0, 0.0], 1.0, 0.0, 0.0);
    // Closest point on z = x to (2, 3, 0) is (1, 3, 1)
    expect(foot[0]).toBeCloseTo(1.0, 5);
    expect(foot[1]).toBeCloseTo(3.0, 5);
    expect(foot[2]).toBeCloseTo(1.0, 5);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **Shared `BufferGeometry` Across 3 Parallel Planes (66% GPU Vertex Memory Saving)**:
   - ตรวจสอบใน `MarginCorridorPlanes3D.tsx` ว่าแผ่นระนาบหลัก ($f(\mathbf{x}) = 0$), แผ่นขอบบน ($+1$), และแผ่นขอบล่าง ($-1$) อ้างอิง `geometry={sharedPlaneGeometry}` ตัวเดียวกัน โดยใช้ `<mesh position={[0, 0, +marginWidth]}>` และ `<mesh position={[0, 0, -marginWidth]}>` แทนการสร้าง `BufferGeometry` แยก 3 ชุด
   - ยืนยันว่า `renderer.info.memory.geometries` เพิ่มขึ้นเพียง `1` เมื่อเปิดโหมด Margin Corridor
2. **Zero-Norm Protection Test**:
   - ทดสอบกรณีที่ `model.coef_` เป็นศูนย์ทุกตัว (`np.zeros((1, D))`) ยืนยันว่า `self.w_norms = np.maximum(np.linalg.norm(...), 1e-12)` ป้องกันไม่ให้เกิด `ZeroDivisionError` หรือ `NaN` ใน JSON Response

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **`SGDClassifier(loss='hinge')` ไม่มีเมธอด `predict_proba`** | เรียก `trace_internal_flow` บน `SGDClassifier(loss='hinge')` | ไม่เกิด `AttributeError` ตั้งค่า `has_native_proba: false` และคืนค่า Pseudo-Probability จาก `expit(decision_function)` |
| 2 | **`loss='modified_huber'` คืนค่าความน่าจะเป็นอิ่มตัวเป็น `0.0` หรือ `1.0` เมื่อ $|f(x)| \ge 1$** | ส่งจุดข้อมูลที่มี $f(\mathbf{x}) = +1.5$ เข้า `SGDClassifier(loss='modified_huber')` | คืนค่าความน่าจะเป็น `1.0` ตรงตาม Scikit-learn พร้อมแสดงคำอธิบาย `"Modified Huber Piecewise Linear Probability"` ใน HUD |
| 3 | **โมเดลเก่าใช้ชื่อ `loss='log'` แทน `'log_loss'`** | จำลอง `model.loss = 'log'` บน Instance ของ `SGDClassifier` | `self.loss_name in self.PROBA_LOSSES` คืนค่า `True` รองรับ Backward Compatibility |
| 4 | **ข้อมูลไม่ได้ทำ Feature Scaling ทำให้ `coef_` สูงระดับ `1e5`** | เทรน SGD บนข้อมูลสเกลใหญ่จน `max(abs(z_values)) > 1e4` | ระบบตรวจพบ Gradient สูงผิดปกติ ทำการ Auto-Normalize สเกลแกน $Z$ ในฉาก 3D และแสดงป้ายเตือน `"Unscaled Features Detected"` |
| 5 | **`Perceptron` ไม่มีแอตทริบิวต์ `loss` ในบางเวอร์ชัน** | ลบแอตทริบิวต์ `loss` หรือใช้ `Perceptron().fit(X, y)` | `getattr(model, "loss", "perceptron")` ทำงานได้โดยไม่พัง |
| 6 | **`SGDRegressor.coef_` มีขนาด 1D `(D,)`** | ส่ง `SGDRegressor` เข้า `SGDAdapter` | `np.atleast_2d(model.coef_)` แปลงเป็น `(1, D)` เสมอ ไม่เกิด `IndexError` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Loss Coverage SLA**: ผ่านชุดทดสอบ `pytest` ครบทุก Loss Function ของ `SGDClassifier`, `SGDRegressor`, `PassiveAggressive*`, และ `Perceptron`
- [ ] **Shared Geometry Gate**: ระนาบทั้ง 3 ชั้นใน `MarginCorridorPlanes3D` ต้องแชร์ `BufferGeometry` เดียวกัน (`geometries allocated == 1`)
- [ ] **Mathematical Parity SLA**: ระยะห่างเรขาคณิต $d_{\text{geom}}$ และผลรวม Waterfall มีความคลาดเคลื่อนไม่เกิน `1e-5`
- [ ] **Coverage Gate**: Code Coverage ของ `sgd.py` และ `MarginCorridorPlanes3D.tsx` ไม่ต่ำกว่า `95%`
