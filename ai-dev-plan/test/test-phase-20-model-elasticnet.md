# Test Blueprint — Phase 20: ElasticNet Adapter & L1/L2 Penalty Ratio 3D Inspector

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-20-model-elasticnet.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-20-model-elasticnet.md) เพื่อยืนยันความถูกต้องของคลาส `ElasticNetAdapter` (`backend/app/domains/model_engine/adapters/linear/elasticnet.py`), คอมโพเนนต์วงแหวนขอบเขตข้อจำกัด `ElasticNetConstraintRing3D` (`frontend/src/domains/stage3d/components/actors/ElasticNetConstraintRing3D.tsx`), และแผงมาตรวัด `ElasticNetGaugePanel` (`frontend/src/domains/hud/components/inspectors/ElasticNetGaugePanel.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทุกตัว**:
  - `sklearn.linear_model.ElasticNet`
  - `sklearn.linear_model.ElasticNetCV` (รวมถึงกรณีส่ง `l1_ratio` เป็น List เช่น `[0.1, 0.5, 0.9]`)
  - `sklearn.linear_model.MultiTaskElasticNet`
  - `sklearn.linear_model.MultiTaskElasticNetCV`
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - การแยกสัมประสิทธิ์โทษปรับ $\lambda_1 = \alpha \rho$ และ $\lambda_2 = \frac{1}{2}\alpha(1 - \rho)$ โดยที่ $\rho = \text{l1\_ratio}$
  - การคำนวณภาระโทษปรับจริง `l1_penalty_burden` และ `l2_penalty_burden`
  - การคัดเลือกฟีเจอร์ร่วมกันข้ามทุก Task (`Joint Feature Selection` ผ่าน $L_{2,1}$ Norm) ใน `MultiTaskElasticNet`
  - สมการวงแหวน Superellipse $\rho(|u| + |v|) + (1 - \rho)(u^2 + v^2) = r^2$ ในทุกค่าของ $\rho \in [0, 1]$

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **Penalty Decomposition Verification**:
   - ตรวจสอบว่า `meta["lambda_1"] == meta["alpha"] * meta["l1_ratio"]` และ `meta["lambda_2"] == 0.5 * meta["alpha"] * (1.0 - meta["l1_ratio"])`
2. **MultiTask Joint Sparsity Verification**:
   - ใน `MultiTaskElasticNet` ที่มี $M$ Tasks ฟีเจอร์ที่อยู่ใน `zeroed_features` ต้องมีน้ำหนัก $w_{m,j} = 0$ ในทุกๆ Task $m \in \{0, \dots, M-1\}$ พร้อมกัน
3. **Task-Specific Waterfall Additivity**:
   - สำหรับทุก `task_idx` ใน `MultiTaskElasticNet`: ผลรวม `np.sum(flow.feature_contributions, axis=1) + flow.base_intercepts[task_idx]` ต้องเท่ากับ `flow.decision_scores[:, task_idx]` และตรงกับ `model.predict(X)[:, task_idx]`

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_elasticnet_adapter.py`)
```python
import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_regression
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
)
from app.domains.model_engine.adapters.linear.elasticnet import ElasticNetAdapter

FEATURE_NAMES = ["feat_rm", "feat_lstat", "feat_tax", "feat_nox", "feat_dis"]

@pytest.fixture
def single_target_data():
    X, y = make_regression(n_samples=120, n_features=5, n_informative=3, noise=1.0, random_state=42)
    return X, y

@pytest.fixture
def multi_task_data():
    X, Y = make_regression(
        n_samples=120, n_features=5, n_informative=3, n_targets=3, noise=1.0, random_state=42
    )
    return X, Y

@pytest.mark.parametrize("model_cls,kwargs", [
    (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.7}),
    (ElasticNet, {"alpha": 0.2, "l1_ratio": 0.3, "positive": True}),
    (ElasticNetCV, {"l1_ratio": [0.1, 0.5, 0.9], "cv": 3}),
])
def test_single_target_elasticnet_penalty_math_and_flow(single_target_data, model_cls, kwargs):
    X, y = single_target_data
    model = model_cls(random_state=42, **kwargs).fit(X, y)
    adapter = ElasticNetAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    alpha = meta["alpha"]
    rho = meta["l1_ratio"]
    assert np.isclose(meta["lambda_1"], alpha * rho, atol=1e-7)
    assert np.isclose(meta["lambda_2"], 0.5 * alpha * (1.0 - rho), atol=1e-7)

    w = np.asarray(meta["coef"])[0]
    expected_l1_burden = (alpha * rho) * np.sum(np.abs(w))
    expected_l2_burden = 0.5 * alpha * (1.0 - rho) * np.sum(w ** 2)
    assert np.isclose(meta["l1_penalty_burden"], expected_l1_burden, atol=1e-6)
    assert np.isclose(meta["l2_penalty_burden"], expected_l2_burden, atol=1e-6)

    flow = adapter.trace_internal_flow(X[:20])
    preds = np.asarray(flow.predictions)
    contribs = np.asarray(flow.feature_contributions)
    np.testing.assert_allclose(preds, model.predict(X[:20]), atol=1e-5)
    np.testing.assert_allclose(np.sum(contribs, axis=1) + flow.base_intercepts[0], preds, atol=1e-5)

@pytest.mark.parametrize("model_cls,kwargs", [
    (MultiTaskElasticNet, {"alpha": 1.0, "l1_ratio": 0.8}),
    (MultiTaskElasticNetCV, {"l1_ratio": [0.2, 0.8], "cv": 3}),
])
def test_multitask_elasticnet_joint_feature_selection_and_task_switching(multi_task_data, model_cls, kwargs):
    X, Y = multi_task_data
    model = model_cls(random_state=42, **kwargs).fit(X, Y)
    adapter = ElasticNetAdapter(model, FEATURE_NAMES)

    meta = adapter.extract_metadata()
    assert meta["task_type"] == "multitask_regression"
    assert meta["n_tasks"] == 3

    # Test switching across all 3 tasks
    expected_all_preds = model.predict(X[:15])
    for t_idx in range(3):
        flow = adapter.trace_internal_flow(X[:15], task_idx=t_idx)
        preds_t = np.asarray(flow.predictions)
        contribs_t = np.asarray(flow.feature_contributions)
        b_t = flow.base_intercepts[t_idx]

        np.testing.assert_allclose(preds_t, expected_all_preds[:, t_idx], atol=1e-5)
        np.testing.assert_allclose(np.sum(contribs_t, axis=1) + b_t, preds_t, atol=1e-5)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `ElasticNetConstraintRing3D` และ `ElasticNetGaugePanel`
1. **3D Superellipse Constraint Ring Geometry (`computeSuperellipseRingPoints`)**:
   - เมื่อ `l1_ratio = 0.0` (Pure Ridge $L_2$): จุดทุกจุดบนวงแหวน 256 จุดต้องมีระยะห่างจากจุดกำเนิด $r = \sqrt{u^2 + v^2}$ คงที่เท่ากันทุกมุม $\theta$ (เป็นวงกลมสมบูรณ์แบบ)
   - เมื่อ `l1_ratio = 1.0` (Pure Lasso $L_1$): จุดทุกจุดต้องสอดคล้องกับสมการ $|u| + |v| = c$ โดยไม่เกิด `NaN` หรือ `Infinity` จากการหารด้วย $a = 1 - \rho = 0$
   - เมื่อ `l1_ratio = 0.7`: ต้องได้รูปทรง Rounded Diamond ที่ทุกจุดสอดคล้องกับ $\rho(|u| + |v|) + (1 - \rho)(u^2 + v^2) = c$
2. **L1 / L2 Balance Gauge & Convergence Banner**:
   - ทดสอบการแสดงผลสัดส่วนเปอร์เซ็นต์ `70% L1 / 30% L2`, ค่า $\lambda_1, \lambda_2$ และการแสดงป้ายเตือนสีเหลืองเมื่อ `n_iter >= max_iter`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/actors/__tests__/ElasticNetConstraintRing3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { computeConstraintRingPoints } from "@/domains/stage3d/components/actors/ElasticNetConstraintRing3D";
import { ElasticNetGaugePanel } from "@/domains/hud/components/inspectors/ElasticNetGaugePanel";

describe("Phase 20: ElasticNetConstraintRing3D & L1/L2 Gauge Tests", () => {
  it("generates finite valid vertices across all l1_ratio regimes including edge cases 0.0 and 1.0", () => {
    for (const rho of [0.0, 0.25, 0.5, 0.75, 1.0]) {
      const points = computeConstraintRingPoints(rho, 1.0, 256, false);
      expect(points).toHaveLength(256);
      for (const [u, v, z] of points) {
        expect(Number.isFinite(u)).toBe(true);
        expect(Number.isFinite(v)).toBe(true);
        expect(z).toBe(0);
        // Verify points satisfy the constraint equation rho*(|u|+|v|) + (1-rho)*(u^2+v^2) = 1.0
        const lhs = rho * (Math.abs(u) + Math.abs(v)) + (1 - rho) * (u * u + v * v);
        expect(lhs).toBeCloseTo(1.0, 4);
      }
    }
  });

  it("renders Convergence Warning badge when n_iter reaches max_iter", () => {
    const mockMeta = {
      algorithm: "ElasticNet",
      alpha: 0.12,
      l1_ratio: 0.7,
      lambda_1: 0.084,
      lambda_2: 0.018,
      l1_penalty_burden: 0.312,
      l2_penalty_burden: 0.094,
      n_iter: 1000,
      max_iter: 1000,
      n_tasks: 1,
    };
    render(<ElasticNetGaugePanel metadata={mockMeta} activeTaskIdx={0} onTaskChange={() => {}} />);
    expect(screen.getByText(/Convergence Warning: Reached max_iter/i)).toBeInTheDocument();
    expect(screen.getByText(/70% L1/i)).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **Multi-Task Client-Side Switching Without Extra API Calls**:
   - ทดสอบว่าเมื่อโหลดโมเดล `MultiTaskElasticNet` ที่มี $M = 10$ Tasks ตัว `trace_internal_flow` จะส่ง `decision_scores` ขนาด $(N, 10)$ และ `coef` ขนาด $(10, D)$ มาให้ครบในครั้งเดียว
   - ยืนยันด้วยการ Spy บน `fetch` / API Client ว่าเมื่อผู้ใช้คลิกสลับปุ่ม `Task 0 -> Task 1 -> Task 2` ใน HUD ระบบสามารถคำนวณระนาบ 3D และ Waterfall ใหม่บนฝั่ง Client ได้ทันทีโดยมีจำนวน HTTP Request เพิ่มขึ้นเป็น `0`
2. **Constraint Ring Memoization**:
   - ตรวจสอบว่า `computeConstraintRingPoints` ถูกห่อหุ้มด้วย `useMemo([l1_ratio, positive])` จึงไม่ถูกคำนวณซ้ำในทุกเฟรมขณะที่ VCR Playback กำลังวิ่ง

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **`ElasticNetCV` รับ `l1_ratio=[0.1, 0.5, 0.9]` เป็น List** | เทรน `ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9])` แล้วเรียก `extract_metadata()` | อ่านค่าสเกลาร์ที่ชนะจาก `model.l1_ratio_` เป็น `float` เดี่ยวโดยไม่เกิด `TypeError` ตอนคูณกับ `alpha_` |
| 2 | **`MultiTaskElasticNet` มี `coef_` ขนาด 2D `(M, D)`** | ส่งทั้ง `ElasticNet` (`1D coef_`) และ `MultiTaskElasticNet` (`2D coef_`) เข้า Adapter | `self.coef_2d` มีขนาด `(1, D)` และ `(M, D)` ตามลำดับอย่างสม่ำเสมอ |
| 3 | **ค่าสุดโต่ง `l1_ratio == 1.0` ทำให้สัมประสิทธิ์กำลังสอง $a = 1 - \rho = 0$** | เรียก `computeConstraintRingPoints(1.0, 1.0, 256)` | สลับไปใช้สมการเชิงเส้น $r = \frac{c}{|\cos\theta| + |\sin\theta|}$ โดยไม่เกิดการหารด้วยศูนย์ (`NaN` / `Inf`) |
| 4 | **Coordinate Descent ไม่ลู่เข้า (`n_iter_ == max_iter`)** | เทรน `ElasticNet(max_iter=2)` บนข้อมูลที่มี Collinearity สูง | Metadata ตรวจพบ `n_iter >= max_iter` และแสดงป้าย `"Convergence Warning: Reached max_iter"` ใน HUD |
| 5 | **พารามิเตอร์ `positive=True` บังคับน้ำหนักไม่ติดลบ** | เทรน `ElasticNet(positive=True)` | วาดวงแหวน Constraint Aura เฉพาะในจตุภาคที่ 1 ($u \ge 0, v \ge 0$) และแสดงป้าย `"Non-negative Weights Constraint Active"` |
| 6 | **ข้อมูลนำเข้าเป็น Sparse Matrix ใน `MultiTaskElasticNet`** | ส่ง `sparse.csr_matrix(X)` เข้า `trace_internal_flow` ของ `MultiTaskElasticNet` | แปลงเป็น Dense เฉพาะ Batch ปัจจุบันด้วย `.toarray()` อัตโนมัติโดยไม่โยน `TypeError: A sparse matrix was passed` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Mathematical Parity SLA**: สมการแยกโทษปรับ $\lambda_1, \lambda_2$, ผลรวม Waterfall และสมการวงแหวน Superellipse มีความคลาดเคลื่อนไม่เกิน `1e-5`
- [ ] **Zero-Division Safety Gate**: วงแหวน `ElasticNetConstraintRing3D` ต้องไม่มีค่า `NaN` หรือ `Infinity` ในทุกค่าของ `l1_ratio` ตั้งแต่ `0.000` ถึง `1.000`
- [ ] **Multi-Task Switching SLA**: การสลับดูแต่ละ Task ใน `MultiTaskElasticNet` บน UI ใช้เวลา `< 5 ms` โดยไม่เรียก Backend ซ้ำ
- [ ] **Coverage Gate**: Code Coverage ของ `elasticnet.py`, `ElasticNetConstraintRing3D.tsx`, และ `ElasticNetGaugePanel.tsx` ไม่ต่ำกว่า `95%`
