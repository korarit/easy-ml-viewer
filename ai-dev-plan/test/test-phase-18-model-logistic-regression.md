# Test Blueprint — Phase 18: Logistic Regression Adapter & 3D Sigmoid S-Surface Visualizer

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-18-model-logistic-regression.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-18-model-logistic-regression.md) เพื่อยืนยันความถูกต้องของคลาส `LogisticRegressionAdapter` (`backend/app/domains/model_engine/adapters/linear/logistic_regression.py`), คอมโพเนนต์ 3 มิติ `SigmoidSurface3D` (`frontend/src/domains/stage3d/components/surfaces/SigmoidSurface3D.tsx`), และแผงควบคุม `LogisticInspectorPanel` (`frontend/src/domains/hud/components/inspectors/LogisticInspectorPanel.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn**:
  - `sklearn.linear_model.LogisticRegression` (ครบทุก Solver: `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga` และทุก Penalty: `l2`, `l1`, `elasticnet`, `None`)
  - `sklearn.linear_model.LogisticRegressionCV` (ตรวจสอบการสกัดค่า `C_` ที่เป็น Array)
- **รูปแบบงาน (Tasks)**:
  - Binary Classification ($C = 2$): สมการ Sigmoid $\sigma(z_t) = \frac{1}{1 + e^{-z_t}}$
  - Multiclass Classification ($C \ge 3$): สมการ Softmax และ One-vs-Rest (`ovr`)
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - คะแนน `decision_scores` ต้องตรงกับ `model.decision_function(X)` 100%
  - ผลรวม Local Feature Contribution (Waterfall): $b_k + \sum_{j=1}^{D} w_{k,j} x_{t,j} = z_{t,k}$ ความคลาดเคลื่อนไม่เกิน `1e-5`

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **Prediction & Probability Parity**: ค่า `predictions` และ `probabilities` ใน `InternalFlowBatch` ต้องตรงกับ `model.predict(X)` และ `model.predict_proba(X)` ทุกแถว (`np.allclose(..., atol=1e-6)`)
2. **Log-Odds Waterfall Additivity**:
   - สำหรับ Binary ($C=2$): `np.sum(batch.feature_contributions, axis=1) + batch.base_intercepts[0]` ต้องเท่ากับ `batch.decision_scores[:, 0]`
   - สำหรับ Multiclass ($C>2$): ผลรวมของฟีเจอร์ในคลาสที่ชนะ ($\arg\max_k P(y=k)$) บวกกับ `base_intercepts[winning_idx]` ต้องเท่ากับ `decision_scores[t, winning_idx]`
3. **3D Sigmoid S-Surface Mesh Accuracy**:
   - พื้นผิว `z_values` ขนาด $(60, 60)$ ต้องอยู่ในช่วง $[0.0, 1.0]$ เสมอ และจุดที่ $w_x x + w_y y + b_{\text{offset}} = 0$ ต้องมีความสูง $Z = 0.5000 \pm 10^{-6}$

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_logistic_regression_adapter.py`)
```python
import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from app.domains.model_engine.adapters.linear.logistic_regression import LogisticRegressionAdapter

FEATURE_NAMES = ["feat_income", "feat_age", "feat_debt", "feat_score"]

@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0,
        n_classes=2, random_state=42
    )
    return X, y

@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    return X, y

@pytest.mark.parametrize("model_cls,kwargs", [
    (LogisticRegression, {"solver": "lbfgs", "C": 1.0}),
    (LogisticRegression, {"solver": "liblinear", "penalty": "l1", "C": 0.5}),
    (LogisticRegression, {"solver": "lbfgs", "fit_intercept": False}),
    (LogisticRegressionCV, {"cv": 3, "Cs": [0.1, 1.0, 10.0]}),
])
def test_binary_logistic_parity_and_waterfall_additivity(binary_data, model_cls, kwargs):
    X, y = binary_data
    model = model_cls(random_state=42, **kwargs).fit(X, y)
    adapter = LogisticRegressionAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    assert meta["task_type"] == "binary_classification"
    assert isinstance(meta["hyperparameters"]["C"], float)

    flow = adapter.trace_internal_flow(X[:25])
    preds_arr = np.asarray(flow.predictions)
    probas_arr = np.asarray(flow.probabilities)
    scores_arr = np.asarray(flow.decision_scores)
    contribs_arr = np.asarray(flow.feature_contributions)

    # 1. Shape verification: (N, 1) for binary decision_scores, (N, 2) for probas, (N, D) for contribs
    assert scores_arr.shape == (25, 1)
    assert probas_arr.shape == (25, 2)
    assert contribs_arr.shape == (25, 4)

    # 2. Parity with sklearn native methods
    np.testing.assert_array_equal(preds_arr, model.predict(X[:25]))
    np.testing.assert_allclose(probas_arr, model.predict_proba(X[:25]), atol=1e-5)
    np.testing.assert_allclose(scores_arr[:, 0], model.decision_function(X[:25]), atol=1e-5)

    # 3. Exact Log-Odds Waterfall Additivity: sum(w_j * x_j) + b == z_t
    reconstructed_logits = np.sum(contribs_arr, axis=1) + flow.base_intercepts[0]
    np.testing.assert_allclose(reconstructed_logits, scores_arr[:, 0], atol=1e-5)

def test_multiclass_softmax_waterfall_and_surface_mesh(multiclass_data):
    X, y = multiclass_data
    model = LogisticRegression(solver="lbfgs", random_state=42).fit(X, y)
    adapter = LogisticRegressionAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:20])
    scores_arr = np.asarray(flow.decision_scores)
    probas_arr = np.asarray(flow.probabilities)
    contribs_arr = np.asarray(flow.feature_contributions)
    intercepts = np.asarray(flow.base_intercepts)

    assert scores_arr.shape == (20, 3)
    assert probas_arr.shape == (20, 3)
    np.testing.assert_allclose(np.sum(probas_arr, axis=1), np.ones(20), atol=1e-5)

    # Verify winning class waterfall additivity per row
    winning_idx = np.argmax(probas_arr, axis=1)
    expected_winning_logits = scores_arr[np.arange(20), winning_idx]
    reconstructed = np.sum(contribs_arr, axis=1) + intercepts[winning_idx]
    np.testing.assert_allclose(reconstructed, expected_winning_logits, atol=1e-5)

    # Verify 3D Sigmoid Surface Mesh generation (60x60 default)
    mesh = adapter.compute_surface_mesh(
        axis_x_idx=0, axis_y_idx=1,
        x_range=(-3.0, 3.0), y_range=(-3.0, 3.0),
        fixed_features=np.mean(X, axis=0), grid_resolution=60, target_class_idx=1
    )
    z_grid = np.asarray(mesh.z_values)
    assert z_grid.shape == (60, 60)
    assert np.all((z_grid >= 0.0) & (z_grid <= 1.0))
    assert mesh.boundary_threshold == 0.5
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ UI, 3D Geometry และ UX Interaction
1. **`SigmoidSurface3D` Geometry & Contour Line**:
   - ตรวจสอบการสร้าง `PlaneGeometry` ขนาด $60 \times 60$ Vertices ($3,600$ จุด) และการอัปเดตความสูงแกน $Z$ ตามค่าความน่าจะเป็น $\sigma(z) \in [0, 1]$ หรือค่าเชิงเส้น $z$ เมื่อสลับโหมด `Z-Axis Metric Toggle`
   - ตรวจสอบการเรนเดอร์เส้นขอบเขตเรืองแสง `Glowing Iso-Probability Contour Line` ที่ระดับความสูง $Z = \tau$ และการขยับตาม `Decision Threshold Slider` ในช่วง `[0.05, 0.95]`
2. **VCR Playback & Decision Boundary Crossing Alert**:
   - เมื่อเลื่อน Scrubber ข้ามเฟรมที่ค่า $z_t$ เปลี่ยนเครื่องหมาย (เช่น จาก `-0.42` เป็น `+0.68`) ต้องแสดงป้ายสถานะ `"BOUNDARY CROSSED: Class 0 → Class 1"`
3. **What-If Sandbox GPU Shader Phase Shift**:
   - เมื่อขยับสไลเดอร์ของฟีเจอร์แกนอื่นๆ ที่ไม่ได้อยู่บนแกน $X, Y$ (Off-axis feature) ค่า Uniform `interceptOffset` ต้องเปลี่ยนไปตามสมการ $\sum_{m \notin \{x,y\}} w_m x_m + b$ ส่งผลให้คลื่น S-Surface เลื่อนเฟสทันทีโดยไม่ยิง HTTP API ซ้ำ

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/surfaces/__tests__/SigmoidSurface3D.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LogisticInspectorPanel } from "@/domains/hud/components/inspectors/LogisticInspectorPanel";
import { computeSigmoidPhaseShift } from "@/domains/stage3d/components/surfaces/SigmoidSurface3D";

describe("Phase 18: LogisticInspectorPanel & SigmoidSurface3D Math", () => {
  const mockMetadata = {
    model_family: "linear_classification",
    algorithm: "LogisticRegression",
    task_type: "binary_classification",
    classes: [0, 1],
    hyperparameters: { penalty: "l2", C: 1.0, solver: "lbfgs" },
    coef: [[1.5, -0.8, 0.4, -0.2]],
    intercept: [-0.45],
  };

  const mockFrameTelemetry = {
    frameIndex: 142,
    prediction: 1,
    probabilities: [0.15, 0.85],
    decisionScores: [1.734],
    featureContributions: [1.62, 0.81, -0.246, 0.0],
    baseIntercepts: [-0.45],
  };

  it("renders Log-Odds (z_t), Sigmoid probability, and Waterfall contributors accurately", () => {
    const onThresholdChange = vi.fn();
    render(
      <LogisticInspectorPanel
        metadata={mockMetadata}
        frame={mockFrameTelemetry}
        featureNames={["feat_income", "feat_age", "feat_debt", "feat_score"]}
        threshold={0.5}
        onThresholdChange={onThresholdChange}
      />
    );

    expect(screen.getByText(/\+1\.734/)).toBeInTheDocument();
    expect(screen.getByText(/85\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/feat_income/i)).toBeInTheDocument();

    const slider = screen.getByRole("slider", { name: /threshold/i });
    fireEvent.change(slider, { target: { value: "0.75" } });
    expect(onThresholdChange).toHaveBeenCalledWith(0.75);
  });

  it("computes real-time GPU Shader phase shift for off-axis What-If sliders in 0ms", () => {
    const weights = [1.5, -0.8, 0.4, -0.2];
    const baseIntercept = -0.45;
    // Axis X = 0, Axis Y = 1; Off-axis features = index 2 and 3
    const currentFeatures = [2.0, 1.0, 3.0, 0.5];
    const offsetBefore = computeSigmoidPhaseShift(weights, baseIntercept, currentFeatures, 0, 1);
    // 0.4 * 3.0 + (-0.2) * 0.5 - 0.45 = 1.2 - 0.1 - 0.45 = 0.65
    expect(offsetBefore).toBeCloseTo(0.65, 6);

    // Move off-axis feature #2 from 3.0 -> 5.0
    const updatedFeatures = [2.0, 1.0, 5.0, 0.5];
    const offsetAfter = computeSigmoidPhaseShift(weights, baseIntercept, updatedFeatures, 0, 1);
    expect(offsetAfter).toBeCloseTo(1.45, 6);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **การตรวจสอบ `float32` Payload Serialization บน Backend**:
   - ทดสอบเรียก `trace_internal_flow` ด้วยข้อมูลชุดใหญ่ขนาด $(10,000 \times 50)$ ยืนยันว่าการคำนวณใช้ Vectorized NumPy Broadcasting (`X_arr * self.coef_[0:1, :]`) โดยใช้เวลาประมวลผลรวมต่ำกว่า `80 ms` และ Peak RAM เพิ่มขึ้นไม่เกิน `25 MB`
2. **การตรวจสอบการ Reuse `Float32Array` และการเรียก `.dispose()` ใน WebGL**:
   - จำลองการสลับ `Target Class Surface Selector` และการเลื่อน `Decision Threshold Slider` ต่อเนื่อง 1,000 ครั้งใน `vitest`
   - ตรวจสอบว่า `SigmoidSurface3D` อัปเดตค่าใน `geometry.attributes.position.array` เดิมและตั้ง `needsUpdate = true` โดยที่ `renderer.info.memory.geometries` ไม่เพิ่มขึ้นเลยแม้แต่ 1 ชิ้น และเมื่อ Unmount คอมโพเนนต์ต้องเรียก `geometry.dispose()` และ `material.dispose()` ครบ 100%

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **Binary `(1, D)` vs Multiclass `(C, D)` `coef_` Shape** | รัน `trace_internal_flow` เทียบระหว่างโมเดล 2 คลาสและ 4 คลาส | `decision_scores` คืนค่าเป็น 2D Array ขนาด `(N, 1)` สำหรับ Binary และ `(N, 4)` สำหรับ Multiclass เสมอ ไม่เป็น 1D Array |
| 2 | **Extreme Outlier ทำให้เกิด Exponential Overflow ($z_t < -1000$)** | ส่งแถวข้อมูลที่มีค่า $x = -10^6$ จนได้ $z_t \approx -1500$ เข้า Adapter | ไม่เกิด `RuntimeWarning: overflow encountered in exp` และคืนค่าความน่าจะเป็น `0.0` หรือ `1.0` อย่างปลอดภัยด้วย `scipy.special.expit` |
| 3 | **โมเดลตั้งค่า `fit_intercept=False`** | เทรน `LogisticRegression(fit_intercept=False)` แล้วเรียก `extract_metadata` และ `trace_internal_flow` | `base_intercepts` มีค่าเป็น `[0.0]` และ HUD แสดงป้าย `"Intercept: 0.0 (Disabled)"` |
| 4 | **`penalty='l1'` ตัดน้ำหนักทั้งแกน $X$ และแกน $Y$ เป็น `0.0` พร้อมกัน** | เทรน `LogisticRegression(penalty='l1', solver='liblinear', C=0.01)` แล้วเลือกคู่แกนที่ $w_x = 0, w_y = 0$ | พื้นผิว `z_values` แบนราบเท่ากันทุกจุด, ซ่อนเส้น Boundary Tube และแสดงป้ายเตือนบนฉาก 3D ว่าทั้งสองแกนไม่มีอิทธิพลต่อโมเดล |
| 5 | **`LogisticRegressionCV` เก็บค่าใน `model.C_` เป็น `ndarray`** | เทรน `LogisticRegressionCV(cv=3)` ทั้งแบบ Binary และ Multiclass | `extract_metadata()["hyperparameters"]["C"]` แปลงค่าตัวแรกจาก `model.C_` เป็น `float` มาตรฐานโดยไม่เกิด `AttributeError: C` |
| 6 | **ข้อมูลนำเข้าเป็น `scipy.sparse.csr_matrix`** | ส่ง `sparse.csr_matrix(X_batch)` เข้า `trace_internal_flow` | คำนวณ `feature_contributions` ได้ถูกต้องโดยไม่แครชและไม่แปลงทั้งตารางใหญ่เป็น Dense เกินความจำเป็น |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Mathematical Parity SLA**: ผลลัพธ์ `predictions`, `probabilities`, `decision_scores` และผลรวม Waterfall ($\sum \phi_{t,j} + b = z_t$) มีค่าความคลาดเคลื่อนไม่เกิน `1e-5` ทั้งใน Binary และ Multiclass
- [ ] **Performance SLA**: การสร้าง `SurfaceMesh3D` ขนาด $60 \times 60$ ใช้เวลาบน Backend `< 35 ms` และการเลื่อน What-If Slider คำนวณ Phase Shift บน GPU ที่เฟรมเรต `60 FPS` (`< 16.6 ms/frame`)
- [ ] **Memory Gate**: ทดสอบ Scrubbing และสลับคลาส 1,000 รอบ ต้องมี `WebGL Geometry Leak = 0`
- [ ] **Coverage Gate**: Code Coverage ของ `logistic_regression.py` และ `SigmoidSurface3D.tsx` ต้องไม่ต่ำกว่า `95%`
