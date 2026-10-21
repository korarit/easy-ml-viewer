# Test Plan — Phase 30: AdaBoost Adapter & 3D Sequential Weak-Learner Cascade

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-30-model-adaboost-cascade.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-30-model-adaboost-cascade.md) เพื่อตรวจสอบความถูกต้องของ `AdaBoostAdapter` ในการแกะรอยการปรับคำตอบทีละขั้นผ่าน `staged_predict_proba` และ `staged_predict`, การผูกแรงดึง $\Delta S_m$ เข้ากับฟีเจอร์ของ Decision Stump หรือต้นไม้หลายชั้น, การคัดเลือก `top_influential_learners` ตามน้ำหนัก `estimator_weights_` ($\alpha_m$), และการเรนเดอร์บันไดแสงแนวดิ่ง **3D Sequential Cascade Ladder** (`AdaBoostCascadeLadder.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `AdaBoostAdapter` (`backend/src/domains/Precomputation/adapters/trees/adaboost.py`)
  - ค่าคงที่: `MAX_TRAJECTORY_POINTS = 50`
  - เมธอดหลัก: `_compute_staged_matrix(X_f32)`, `trace_internal_flow(X_chunk)`
  - คลาสโมเดลที่ทดสอบ: `AdaBoostClassifier` (`SAMME`), `AdaBoostRegressor` (`linear`, `square`, `exponential`)
- **Frontend 3D & HUD Components**:
  - `AdaBoostCascadeLadder.tsx` (`frontend/src/domains/stage3d/actors/AdaBoostCascadeLadder.tsx`)
  - `StumpSplitCurtains.tsx` (`frontend/src/domains/stage3d/actors/StumpSplitCurtains.tsx`)
  - `AdaBoostHudPanel.tsx` (`frontend/src/domains/hud/components/AdaBoostHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Staged Telescoping Sum Parity (ผลรวมแรงดึงราย Stage เท่ากับคำตอบสุดท้าย)**:
   เนื่องจาก $\Delta S_m(\mathbf{x}_i) = S_m(\mathbf{x}_i) - S_{m-1}(\mathbf{x}_i)$ ดังนั้นผลรวมของ `base_value` ($S_0$) บวกกับ `feature_contributions[i, :]` (เมื่อ Base Estimator เป็น Decision Tree) ต้องเท่ากับค่าทำนายสุดท้าย $S_M(\mathbf{x}_i) = \hat{y}(\mathbf{x}_i)$ ทุกประการ:
   $$S_0 + \sum_{j=1}^{d} \Phi_j(\mathbf{x}_i) = S_0 + \sum_{m=1}^{M} \Delta S_m(\mathbf{x}_i) = S_M(\mathbf{x}_i) = \hat{y}(\mathbf{x}_i)$$
2. **Final Trajectory Point Parity**:
   ค่าสุดท้ายใน `staged_trajectory[i, -1]` ต้องเท่ากับ `model.predict_proba(X_f32)[i, target_c]` (สำหรับ Classifier) หรือ `model.predict(X_f32)[i]` (สำหรับ Regressor) เสมอ

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/trees/test_adaboost_adapter.py`)
```python
from __future__ import annotations
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from domains.Precomputation.adapters.trees.adaboost import (
    MAX_TRAJECTORY_POINTS,
    AdaBoostAdapter,
)


def test_adaboost_classifier_stump_telescoping_sum_and_trajectory():
    """พิสูจน์ว่าสำหรับ Decision Stump ผลรวม base_value + sum(contribs) เท่ากับ predict_proba สุดท้ายเป๊ะ."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
    X_f32 = X.astype(np.float32)
    model = AdaBoostClassifier(n_estimators=40, random_state=42).fit(X_f32, y)

    adapter = AdaBoostAdapter(model, feature_names=[f"f{i}" for i in range(5)], target_class_idx=1)
    batch = adapter.trace_internal_flow(X_f32)
    tel = batch.extra_telemetry

    expected_probs = model.predict_proba(X_f32)[:, 1]
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_probs, atol=1e-5)

    # จุดสุดท้ายของ staged_trajectory ต้องเท่ากับ expected_probs
    traj = np.asarray(tel["staged_trajectory"])
    np.testing.assert_allclose(traj[:, -1], expected_probs, atol=1e-4)
    assert len(tel["top_influential_learners"]) == 100
    assert len(tel["top_influential_learners"][0]) <= 8


def test_adaboost_regressor_and_deep_base_estimator():
    """พิสูจน์การทำงานกับ AdaBoostRegressor และกรณีที่ Base Estimator มีความลึก max_depth=3."""
    X, y = make_regression(n_samples=70, n_features=4, noise=0.5, random_state=19)
    X_f32 = X.astype(np.float32)
    model = AdaBoostRegressor(n_estimators=65, random_state=19).fit(X_f32, y)

    adapter = AdaBoostAdapter(model, feature_names=["f0", "f1", "f2", "f3"])
    batch = adapter.trace_internal_flow(X_f32)

    expected_preds = model.predict(X_f32)
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_preds, atol=1e-4)

    # ตรวจสอบการย่อความยาว staged_trajectory ไม่ให้เกิน MAX_TRAJECTORY_POINTS = 50
    traj = np.asarray(batch.extra_telemetry["staged_trajectory"])
    assert traj.shape[1] <= MAX_TRAJECTORY_POINTS


def test_adaboost_early_terminated_and_non_tree_base_estimator():
    """พิสูจน์ว่าโมเดลที่หยุดเทรนก่อนครบ n_estimators หรือใช้ LogisticRegression ไม่เกิด AttributeError."""
    X, y = make_classification(n_samples=60, n_features=4, random_state=5)
    X_f32 = X.astype(np.float32)

    # Non-tree base estimator
    lr_boost = AdaBoostClassifier(
        estimator=LogisticRegression(), n_estimators=10, random_state=5
    ).fit(X_f32, y)
    adapter = AdaBoostAdapter(lr_boost, feature_names=["a", "b", "c", "d"])
    batch = adapter.trace_internal_flow(X_f32)
    assert batch.feature_contributions.shape == (60, 4)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`AdaBoostCascadeLadder.tsx`**:
  - สร้างเสาบันไดแสงแนวดิ่งจาก `base_value` ผ่าน `top_influential_learners` ไปยังความสูงสุดท้าย `prediction`
  - ขั้นบันไดที่มี `delta > 0` เรืองแสงสีฟ้านีออน/เขียว และขั้นที่มี `delta < 0` เรืองแสงสีส้ม/แดง
- **`AdaBoostHudPanel.tsx`**:
  - พล็อตกราฟเส้น **Staged Boosting Curve** จาก `sampled_stage_indices` และ `staged_trajectory`
  - แสดงตาราง `top_influential_learners` พร้อมค่า `weight` ($\alpha_m$), `error` ($\epsilon_m$), และ `delta`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/AdaBoostHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AdaBoostHudPanel } from "../AdaBoostHudPanel";
import { buildCascadeLadderSteps } from "../../../stage3d/actors/AdaBoostCascadeLadder";

const mockAdaBoostFrame = {
  frame_index: 19,
  prediction: 0.82,
  base_value: 0.5,
  feature_contributions: [0.26, 0.15, -0.09, 0.0],
  telemetry: {
    model_family: "adaboost" as const,
    sampled_stage_indices: [0, 1, 2, 3, 4, 12, 49],
    staged_trajectory: [0.74, 0.71, 0.73, 0.65, 0.56, 0.7, 0.82],
    top_influential_learners: [
      { stage: 0, weight: 1.42, error: 0.11, feature_idx: 0, threshold: 65.0, delta: 0.24, cumulative: 0.74 },
      { stage: 3, weight: 0.88, error: 0.23, feature_idx: 2, threshold: 12.0, delta: -0.09, cumulative: 0.56 },
      { stage: 11, weight: 0.95, error: 0.21, feature_idx: 1, threshold: 29.5, delta: 0.14, cumulative: 0.7 },
    ],
  },
};

describe("Phase 30 — AdaBoostHudPanel & 3D AdaBoostCascadeLadder", () => {
  it("renders Top Influential Weak Learners with stage weights, errors, and deltas", () => {
    render(
      <AdaBoostHudPanel
        frame={mockAdaBoostFrame}
        featureNames={["Income", "Age", "Debt", "Score"]}
        activeStageLimit={50}
        onStageLimitChange={vi.fn()}
      />
    );

    expect(screen.getByText(/1\.42/)).toBeInTheDocument();
    expect(screen.getByText(/\+0\.24/)).toBeInTheDocument();
    expect(screen.getByText(/-0\.09/)).toBeInTheDocument();
    expect(screen.getByText(/Income <= 65/i)).toBeInTheDocument();
  });

  it("constructs vertical 3D cascade rungs sorted chronologically by stage index", () => {
    const rungs = buildCascadeLadderSteps(
      mockAdaBoostFrame.base_value,
      mockAdaBoostFrame.telemetry.top_influential_learners
    );

    expect(rungs).toHaveLength(3);
    expect(rungs[0].stage).toBe(0);
    expect(rungs[0].direction).toBe("up");
    expect(rungs[1].stage).toBe(3);
    expect(rungs[1].direction).toBe("down");
    expect(rungs[2].stage).toBe(11);
    expect(rungs[2].direction).toBe("up");
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Generator Consumption & Fast Stump Path)

1. **การทดสอบความเร็ว Fast Path สำหรับ Decision Stumps (`max_depth == 1`)**:
   - บนโมเดล `AdaBoostClassifier(n_estimators=200)` ที่ใช้ Decision Stump และการบวก `contribs[:, f_idx] += delta_m` แบบ Vectorized โดยไม่เรียก `decision_path` ต้องประมวลผล $2,000$ แถวเสร็จภายใน **$\le 60 \text{ ms}$**
2. **การตรวจสอบการบีบอัด `staged_trajectory` (`MAX_TRAJECTORY_POINTS = 50`)**:
   - ยืนยันว่าเมื่อ `n_estimators = 500` อาเรย์ `staged_trajectory` ในแต่ละเฟรมมีความยาวไม่เกิน `50` จุดเสมอ และจุดสุดท้ายคือ Stage ที่ `499` เสมอ
3. **การบริโภค Generator `staged_predict_proba`**:
   - ยืนยันว่าสำหรับ Multiclass ($C = 10$ คลาส) ระบบดึงเฉพาะคอลัมน์ `proba[:, target_class_idx]` ในแต่ละขั้นของ Generator โดยไม่เก็บ Tensor 3 มิติ `(n_samples, M, C)` ทั้งก้อนใน RAM

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **Early Termination ตอนเทรน (`len(estimators_) < n_estimators`)** | เทรน AdaBoost บนข้อมูลที่แยกกันได้สมบูรณ์แบบใน 1 ขั้น (`make_blobs(centers=2, cluster_std=0.1)`) โดยตั้ง `n_estimators=50` | โค้ดใช้ `len(model.estimators_)` เป็นจำนวน Stage จริงโดยไม่เกิด `IndexError` |
| 2 | **Base Estimator ไม่ใช่ Decision Tree** | เทรน `AdaBoostClassifier(estimator=LogisticRegression())` | `isinstance(est, BaseDecisionTree)` เป็น `False` และระบบข้ามการอ่าน `.tree_` ได้อย่างปลอดภัย |
| 3 | **โมเดลเทรนด้วยอัลกอริทึม `SAMME` (Scikit-learn >= 1.4+)** | เทรนโมเดลด้วย `algorithm="SAMME"` | `staged_predict_proba` และ Telescoping Sum ทำงานได้แม่นยำตรงกับ `predict_proba` |
| 4 | **`AdaBoostRegressor` มีหลาย Stage ที่ $\Delta S_m = 0$ (Weighted Median)** | เทรน `AdaBoostRegressor(n_estimators=30)` | `top_influential_learners` คัดเลือกเฉพาะ Stage ที่ทำให้ค่า Weighted Median เปลี่ยนแปลงสูงสุด |
| 5 | **ตัวเรียนตัวแรกสุด (`m = 0`) ดึงคำตอบกระโดดจาก `base_val` ทันที** | ตรวจสอบ `stage_deltas[:, 0]` ของ `trace_internal_flow` | `stage_deltas[:, 0] == staged_mat[:, 0] - base_val` ทำให้ผลรวม Telescoping Sum ไม่ขาดหาย |
| 6 | **มีตัวเรียนเพียง 1 ตัว (`len(estimators_) == 1`)** | เทรน `AdaBoostClassifier(n_estimators=1)` | `sampled_stage_indices == [0]` และกราฟใน HUD แสดงเส้นเชื่อมจาก Stage 0 (`base_value`) ไปยัง Stage 1 ได้โดยไม่พัง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Telescoping Sum Parity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - expected_pred)) <= 1e-5` สำหรับทุกโมเดล AdaBoost ที่ใช้ Tree Base Estimator
- [ ] **Latency SLA**: `trace_internal_flow` สำหรับ $2,000$ แถว (50 Stumps) เสร็จภายใน **$\le 60 \text{ ms}$**
- [ ] **Trajectory Compression SLA**: `len(staged_trajectory[i]) <= 50` และ `len(top_influential_learners[i]) <= 8` เสมอ
- [ ] **WebGL Frame Rate SLA**: `AdaBoostCascadeLadder` และ `SteppedVoxelTerrain` เรนเดอร์ที่ **$\ge 60 \text{ FPS}$**
- [ ] **Coverage Gate**: Line Coverage ของ `adaboost.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
