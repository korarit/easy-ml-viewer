# Test Blueprint — Phase 23: Support Vector Machines (SVC/SVR) Adapter & 3D Kernel Saddle + Support Vector Pillars

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-23-model-svm-kernel-surface.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-23-model-svm-kernel-surface.md) เพื่อยืนยันความถูกต้องของคลาส `SVMAdapter` (`backend/app/domains/model_engine/adapters/distance/svm.py`), คอมโพเนนต์พื้นผิวโค้ง `KernelSaddleSurface3D` (`frontend/src/domains/stage3d/components/surfaces/KernelSaddleSurface3D.tsx`), และเสาพลังงาน `SupportVectorPillars3D` (`frontend/src/domains/stage3d/components/actors/SupportVectorPillars3D.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทั้ง 7 คลาส**:
  - `sklearn.svm.SVC` (ทั้ง `probability=False` และ `probability=True`, ทั้ง 4 Kernels: `rbf`, `poly`, `sigmoid`, `linear`)
  - `sklearn.svm.SVR`, `sklearn.svm.NuSVC`, `sklearn.svm.NuSVR`, `sklearn.svm.OneClassSVM`
  - `sklearn.svm.LinearSVC`, `sklearn.svm.LinearSVR`
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - สมการ Dual Kernel Expansion: $f(\mathbf{x}_t) = \sum_{s=1}^{N_{\text{sv}}} \tilde{\alpha}_s K(\mathbf{x}_s, \mathbf{x}_t) + b$
  - การจำกัดจำนวน Support Vectors ที่ส่งไปแสดงผลไม่เกิน `MAX_DISPLAY_SV = 1000` โดยคัดเลือกเรียงตาม `np.linalg.norm(dual_coef_, axis=0)`
  - การคำนวณ Local Feature Contribution ทั้งแบบ Linear Exact ($w_j x_{t,j}$) และแบบ Non-linear Kernel Directional Attribution

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **All 7 SVM Classes & 4 Kernels Execution Parity**:
   - ทดสอบว่า `trace_internal_flow` และ `compute_surface_mesh` ทำงานได้ครบทั้ง 7 คลาสและ 4 Kernels โดยไม่เกิด `AttributeError` เรื่อง `predict_proba`, `coef_`, หรือ `support_vectors_`
2. **`SVC(probability=False)` Default Safety**:
   - เมื่อ `probability=False` (ค่าเริ่มต้นของ Scikit-learn) `has_native_proba` ต้องเป็น `False` และใช้ `decision_function(X)` แปลงผ่าน `expit`/`softmax` โดยไม่เรียก `model.predict_proba(X)`
3. **Top-1000 Support Vector Prioritization**:
   - เมื่อโมเดลมี `support_vectors_` มากกว่า `1,000` จุด ตัว `_extract_top_support_vectors()` ต้องคืนค่าพิกัดขนาด `(1000, D)` ที่ประกอบด้วยจุดที่มีค่า $|\tilde{\alpha}_s|$ สูงสุด 1,000 อันดับแรก

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_svm_adapter.py`)
```python
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM, LinearSVC, LinearSVR
from app.domains.model_engine.adapters.distance.svm import SVMAdapter

FEATURE_NAMES = ["f0", "f1", "f2", "f3"]

@pytest.fixture
def clf_data():
    return make_classification(n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=42)

@pytest.fixture
def reg_data():
    return make_regression(n_samples=120, n_features=4, n_informative=3, noise=0.5, random_state=42)

@pytest.mark.parametrize("kernel_name,prob_flag", [
    ("rbf", False),
    ("rbf", True),
    ("poly", False),
    ("sigmoid", False),
    ("linear", True),
])
def test_svc_all_kernels_and_probability_modes(clf_data, kernel_name, prob_flag):
    X, y = clf_data
    model = SVC(kernel=kernel_name, probability=prob_flag, random_state=42).fit(X, y)
    adapter = SVMAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    assert meta["kernel"] == kernel_name
    assert meta["has_native_proba"] is prob_flag
    assert meta["total_support_vectors"] == len(model.support_vectors_)

    flow = adapter.trace_internal_flow(X[:20])
    scores = np.asarray(flow.decision_scores)[:, 0]
    contribs = np.asarray(flow.feature_contributions)
    probas = np.asarray(flow.probabilities)

    np.testing.assert_allclose(scores, model.decision_function(X[:20]), atol=1e-5)
    assert probas.shape == (20, 2)
    assert contribs.shape == (20, 4)

    if kernel_name == "linear":
        # Exact linear waterfall additivity
        np.testing.assert_allclose(np.sum(contribs, axis=1) + flow.base_intercepts[0], scores, atol=1e-5)

@pytest.mark.parametrize("model_cls,is_clf", [
    (NuSVC, True),
    (LinearSVC, True),
    (SVR, False),
    (NuSVR, False),
    (LinearSVR, False),
    (OneClassSVM, False),
])
def test_all_svm_family_classes_compatibility(clf_data, reg_data, model_cls, is_clf):
    X, y = clf_data if is_clf else reg_data
    model = model_cls().fit(X, y)
    adapter = SVMAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    flow = adapter.trace_internal_flow(X[:15])
    mesh = adapter.compute_surface_mesh(0, 1, (-2.0, 2.0), (-2.0, 2.0), np.mean(X, axis=0), grid_resolution=25)

    assert len(flow.predictions) == 15
    assert np.asarray(mesh.z_values).shape == (25, 25)
    if isinstance(model, (LinearSVC, LinearSVR)):
        assert meta["support_vectors_3d_data"]["coordinates"] == []

def test_support_vector_capping_at_1000_points():
    rng = np.random.default_rng(42)
    X_big = rng.normal(size=(1500, 4))
    y_big = rng.integers(0, 2, size=1500)
    # High gamma + small C forces > 1000 support vectors
    model = SVC(kernel="rbf", C=0.1, gamma=5.0).fit(X_big, y_big)
    assert len(model.support_vectors_) > 1000

    adapter = SVMAdapter(model, FEATURE_NAMES)
    meta = adapter.extract_metadata()
    sv_coords = meta["support_vectors_3d_data"]["coordinates"]
    assert len(sv_coords) == 1000
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `KernelSaddleSurface3D` และ `SupportVectorPillars3D`
1. **3D Kernel Saddle & Margin Contours ($f(\mathbf{x}) \in \{-1, 0, +1\}$)**:
   - ทดสอบว่า `KernelSaddleSurface3D` เรนเดอร์พื้นผิวโค้งตามค่า `z_values` พร้อมวาดเส้น Contour 3 ระดับคือ `-1.0` (เส้นประสีส้ม), `0.0` (เส้นทึบเรืองแสงสีขาว/เขียว), และ `+1.0` (เส้นประสีฟ้า)
2. **Support Vector Pillars & Top-5 Kernel Force Tethers**:
   - ทดสอบการคำนวณแรงดึงดูด $I_{t,s} = \tilde{\alpha}_s \exp(-\gamma \|\mathbf{x}_s - \mathbf{x}_t\|_2^2)$ ใน `SupportVectorPillars3D` เพื่อเลือกวาดเส้นสายฟ้า 5 เส้นจาก Support Vectors 5 ตัวที่มีค่า $|I_{t,s}|$ สูงที่สุดมายังลูกบอลปัจจุบัน
3. **What-If Sandbox RBF Asymptotic Decay Verification**:
   - เมื่อลากลูกบอลเงา (Ghost Sphere) ออกห่างจากกลุ่ม Support Vectors ทุกตัวไปไกลมาก ($d \to \infty$) ค่าอิทธิพล Kernel $K(\mathbf{x}_s, \mathbf{x}_{\text{ghost}})$ ต้องลดลงเข้าใกล้ `0.0` และค่าคะแนนตัดสินใจต้องลู่เข้าหาค่าคงที่ `intercept` ($b$)

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/actors/__tests__/SupportVectorPillars3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { computeTopKernelForceTethers } from "@/domains/stage3d/components/actors/SupportVectorPillars3D";

describe("Phase 23: SupportVectorPillars3D & Top-5 Kernel Force Tethers", () => {
  it("selects top-5 Support Vectors by |alpha_s * K(x_s, x_t)| and decays to 0 far away", () => {
    const svCoords = [
      [1.0, 1.0, 0.0],
      [1.2, 0.9, 0.0],
      [-2.0, -2.0, 0.0],
      [0.5, 0.5, 0.0],
      [3.0, 3.0, 0.0],
      [-5.0, -5.0, 0.0],
    ];
    const dualWeights = [1.0, 0.8, -1.0, 0.5, -0.9, -1.0];
    const gamma = 0.5;

    // Query near [1.0, 1.0, 0.0]
    const top5Near = computeTopKernelForceTethers([1.0, 1.0, 0.0], svCoords, dualWeights, gamma, 5);
    expect(top5Near).toHaveLength(5);
    // Closest SV at [1.0, 1.0, 0.0] with alpha=1.0 has K=1.0 -> Force = +1.0 (Rank #1)
    expect(top5Near[0].svIndex).toBe(0);
    expect(top5Near[0].force).toBeCloseTo(1.0, 5);

    // What-If Ghost dragged far away to [100, 100, 0] -> all RBF forces decay to ~0
    const top5Far = computeTopKernelForceTethers([100.0, 100.0, 0.0], svCoords, dualWeights, gamma, 5);
    expect(Math.abs(top5Far[0].force)).toBeLessThan(1e-10);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **Top-1000 Support Vectors Memory & `cdist` Benchmark**:
   - ทดสอบเรียก `_compute_local_contributions` ด้วย `scipy.spatial.distance.cdist(X_arr, sv, metric="sqeuclidean")`
   - ยืนยันว่าการใช้ `cdist` ระดับ C-BLAS เร็วกว่าการเขียน Loop ใน Python กว่า $50\times$ โดยใช้เวลา `< 25 ms` สำหรับ Batch 100 แถวเทียบกับ 1,000 Support Vectors
2. **InstancedMesh Rendering สำหรับ Support Vector Rings & Pillars**:
   - ตรวจสอบว่า `SupportVectorPillars3D.tsx` ใช้ `THREE.InstancedMesh` ในการวาดเสาแสงและวงแหวนทั้ง 1,000 ต้นใน Draw Call เดียว

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **`SVC(probability=False)` เป็นค่าเริ่มต้นของ Scikit-learn** | เทรน `SVC()` โดยไม่ส่งพารามิเตอร์ใดๆ แล้วรัน `trace_internal_flow` | ไม่เกิด `AttributeError: predict_proba is not available` คืนค่า `has_native_proba: false` และใช้คะแนน `decision_function` เป็นหลัก |
| 2 | **การเข้าถึง `model.coef_` บน Non-linear Kernel (`rbf`, `poly`, `sigmoid`)** | เทรน `SVC(kernel='rbf')` แล้วรัน `extract_metadata` และ `_compute_local_contributions` | ไม่มีการเรียก `model.coef_` จึงไม่เกิด `AttributeError: coef_ is only available when using a linear kernel` |
| 3 | **`LinearSVC` และ `LinearSVR` ไม่มี `support_vectors_`** | ส่ง `LinearSVC().fit(X, y)` เข้า `SVMAdapter` | ตรวจเช็ก `hasattr(model, "support_vectors_")` คืนค่ารายการพิกัดว่าง `[]` และสลับไปใช้โหมดระนาบเชิงเส้นโดยไม่แครช |
| 4 | **Multiclass `SVC(decision_function_shape='ovo')`** | เทรน `SVC(decision_function_shape='ovo')` บนข้อมูล 4 คลาส (ได้ 6 คอลัมน์คู่เปรียบเทียบ) | `trace_internal_flow` และ `compute_surface_mesh` จัดการมิติ `(N, 6)` ได้โดยไม่เกิด `IndexError` |
| 5 | **ความขัดแย้งระหว่าง `predict()` กับ `argmax(predict_proba())` จาก Platt Scaling** | จำลองจุดใกล้ขอบเขต $f(\mathbf{x}) = +0.01$ ที่ Platt Scaling ให้ `P(y=1) = 0.48` | `predictions` ยึดค่าจาก `model.predict(X)` เป็นหลักเสมอ และ UI แสดงป้ายแจ้งเตือน `Platt Calibration Offset` |
| 6 | **`OneClassSVM` คืนค่าทำนายเป็น `{-1, +1}` และไม่มี `classes_`** | ส่ง `OneClassSVM().fit(X)` เข้า `SVMAdapter` | ทำงานได้สมบูรณ์โดยแมป `+1 -> Inlier` และ `-1 -> Outlier` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Full SVM Matrix Gate**: ผ่านชุดทดสอบ `pytest` ครบทั้ง 7 คลาส (`SVC`, `SVR`, `NuSVC`, `NuSVR`, `OneClassSVM`, `LinearSVC`, `LinearSVR`) และครบทั้ง 4 Kernels (`rbf`, `poly`, `sigmoid`, `linear`)
- [ ] **Support Vector Cap SLA**: จำนวน Support Vectors ที่ส่งไปยัง Frontend ต้องไม่เกิน `1,000` จุดเสมอ
- [ ] **Performance SLA**: การคำนวณ `compute_surface_mesh` ขนาด $50 \times 50$ ใช้เวลา `< 60 ms` และเรนเดอร์เสา Support Vector Pillars ที่ `60 FPS`
- [ ] **Coverage Gate**: Code Coverage ของ `svm.py`, `KernelSaddleSurface3D.tsx`, และ `SupportVectorPillars3D.tsx` ไม่ต่ำกว่า `95%`
