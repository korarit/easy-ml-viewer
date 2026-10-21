# Test Plan — Phase 31: Scikit-Learn Gradient Boosting (GBM) Adapter & Staged Residual Trajectory

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-31-model-sklearn-gbm.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-31-model-sklearn-gbm.md) เพื่อตรวจสอบความถูกต้องของ `SklearnGBMAdapter` ในการแกะโครงสร้าง `model.estimators_` (ซึ่งเป็นอาเรย์ 2 มิติของ `DecisionTreeRegressor` เสมอ), การคูณด้วย `model.learning_rate` ($\eta$), การดึงค่าเริ่มต้นจาก `model._raw_predict_init`, การแยกแยะดัชนีคอลัมน์ระหว่าง Binary Classification (`n_cols == 1`) กับ Multiclass (`n_cols == C`), และการเรนเดอร์เกลียวสปริง **3D Residual Spiral Ladder** (`ResidualSpiralLadder.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `SklearnGBMAdapter` (`backend/src/domains/Precomputation/adapters/boosting/sklearn_gbm.py`)
  - ค่าคงที่: `MAX_TRAJECTORY_STAGES = 50`, `MAX_TRACED_STAGES = 100`
  - เมธอดหลัก: `_get_base_raw_score(X_sample)`, `trace_internal_flow(X_chunk)`
  - คลาสโมเดลที่ทดสอบ: `GradientBoostingClassifier`, `GradientBoostingRegressor`
- **Frontend 3D & HUD Components**:
  - `ResidualSpiralLadder.tsx` (`frontend/src/domains/stage3d/actors/ResidualSpiralLadder.tsx`)
  - `SteppedVoxelTerrain.tsx` (`frontend/src/domains/stage3d/actors/SteppedVoxelTerrain.tsx`)
  - `SklearnGbmHudPanel.tsx` (`frontend/src/domains/hud/components/SklearnGbmHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Raw Margin / Decision Function Additivity Parity**:
   เมื่อจำนวน Stage $M \le 100$ ผลรวมของ `base_value` ($F_0$) บวกกับผลรวมของ `feature_contributions[i, :]` (ซึ่งคูณ `learning_rate` ในทุกกิ่งของ `DecisionTreeRegressor` แล้ว) ต้องเท่ากับค่า Raw Decision Function `model.decision_function(X_f32)[i]` (สำหรับ Classifier) หรือ `model.predict(X_f32)[i]` (สำหรับ Regressor) ทุกประการ:
   $$\left| F_0 + \sum_{j=1}^{d} \Phi_j(\mathbf{x}_i) - F_M(\mathbf{x}_i) \right| \le 10^{-4}$$
2. **Stage Raw Delta Lookup vs `staged_decision_function` Parity**:
   ค่า `stage_raw_deltas[:, m]` ที่ดึงผ่าน `learning_rate * reg_tree.tree_.value[leaf_matrix[:, m], 0, 0]` เมื่อบวกสะสมตั้งแต่วินาทีแรกพร้อม `base_value` ต้องเท่ากับ `list(model.staged_decision_function(X_f32))[-1]` ทุกแถว!

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/boosting/test_sklearn_gbm_adapter.py`)
```python
from __future__ import annotations
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from domains.Precomputation.adapters.boosting.sklearn_gbm import (
    MAX_TRACED_STAGES,
    MAX_TRAJECTORY_STAGES,
    SklearnGBMAdapter,
)


def test_gbm_classifier_binary_raw_margin_additivity_and_single_col_indexing():
    """พิสูจน์ว่า Binary GBM (estimators_.shape == (M, 1)) ไม่เกิด IndexError เมื่อ target_class_idx=1 และบวกได้ตรงกับ decision_function."""
    X, y = make_classification(n_samples=90, n_features=6, n_informative=4, random_state=42)
    X_f32 = X.astype(np.float32)
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X_f32, y)

    assert model.estimators_.shape == (50, 1)
    adapter = SklearnGBMAdapter(model, feature_names=[f"f{i}" for i in range(6)], target_class_idx=1)
    assert adapter.tree_col == 0  # ต้องล็อกคอลัมน์เป็น 0 สำหรับ Binary GBM!

    batch = adapter.trace_internal_flow(X_f32)
    tel = batch.extra_telemetry

    expected_raw_margin = model.decision_function(X_f32)
    reconstructed_margin = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed_margin, expected_raw_margin, atol=1e-4)

    # ผลรวมของ stage_raw_deltas_sample + base_value ต้องเท่ากับ decision_function
    deltas_sum = batch.base_value + np.sum(tel["stage_raw_deltas_sample"], axis=1)
    np.testing.assert_allclose(deltas_sum, expected_raw_margin, atol=1e-3)


def test_gbm_classifier_multiclass_column_indexing():
    """พิสูจน์ว่า Multiclass GBM (estimators_.shape == (M, C)) ดึงคอลัมน์ target_class_idx ได้ตรงกับ decision_function[:, c]."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_classes=3, n_clusters_per_class=1, random_state=7
    )
    X_f32 = X.astype(np.float32)
    model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.15, max_depth=3, random_state=7).fit(X_f32, y)

    assert model.estimators_.shape == (30, 3)
    raw_dec = model.decision_function(X_f32)  # shape: (100, 3)

    for c_idx in range(3):
        adapter = SklearnGBMAdapter(model, feature_names=[f"f{i}" for i in range(5)], target_class_idx=c_idx)
        batch = adapter.trace_internal_flow(X_f32)
        reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
        np.testing.assert_allclose(reconstructed, raw_dec[:, c_idx], atol=1e-4)


def test_gbm_regressor_and_init_zero_support():
    """พิสูจน์ความถูกต้องของ GradientBoostingRegressor ทั้งกรณี init ปกติและ init='zero'."""
    X, y = make_regression(n_samples=80, n_features=4, noise=0.5, random_state=12)
    X_f32 = X.astype(np.float32)

    for init_mode in [None, "zero"]:
        model = GradientBoostingRegressor(
            n_estimators=40, learning_rate=0.08, max_depth=3, init=init_mode, random_state=12
        ).fit(X_f32, y)
        adapter = SklearnGBMAdapter(model, feature_names=["f0", "f1", "f2", "f3"])
        batch = adapter.trace_internal_flow(X_f32)

        if init_mode == "zero":
            assert batch.base_value == pytest.approx(0.0, abs=1e-6)

        expected_preds = model.predict(X_f32)
        reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
        np.testing.assert_allclose(reconstructed, expected_preds, atol=1e-4)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`ResidualSpiralLadder.tsx`**:
  - สร้างจุดยอด 3 มิติของเกลียวสปริงแนวดิ่งจาก `base_value` ไต่ขึ้น/ลงทีละขั้นตาม `stage_raw_deltas_sample`
- **`SklearnGbmHudPanel.tsx`**:
  - แสดงกราฟ **Staged Residual & Loss Convergence Curve** จาก `sampled_stage_indices` และ `staged_predictions`
  - รองรับการเลื่อน **Stage Evolution Scrubber (`Stage m / M`)** เพื่อจำลองการตัดจำนวนต้นไม้ที่ Stage $m$

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/SklearnGbmHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SklearnGbmHudPanel } from "../SklearnGbmHudPanel";
import { generateResidualSpiralVertices } from "../../../stage3d/actors/ResidualSpiralLadder";

const mockGbmFrame = {
  frame_index: 5,
  prediction: 0.891,
  base_value: -0.619,
  feature_contributions: [1.12, 1.85, 0.0, -0.25],
  telemetry: {
    model_family: "sklearn_gbm" as const,
    learning_rate: 0.1,
    n_stages_total: 100,
    sampled_stage_indices: [0, 2, 14, 41, 99],
    staged_predictions: [0.47, 0.62, 0.58, 0.81, 0.891],
    stage_raw_deltas_sample: [0.74, 0.12, 0.62, -0.05],
    leaf_indices_per_stage: [4, 2, 7, 1],
  },
};

describe("Phase 31 — SklearnGbmHudPanel & 3D ResidualSpiralLadder", () => {
  it("renders learning rate, init base score, and top residual stages", () => {
    render(
      <SklearnGbmHudPanel
        frame={mockGbmFrame}
        featureNames={["f0", "f1", "f2", "f3"]}
        activeStage={100}
        onActiveStageChange={vi.fn()}
      />
    );

    expect(screen.getByText(/0\.1/)).toBeInTheDocument();
    expect(screen.getByText(/-0\.62/)).toBeInTheDocument();
    expect(screen.getByText(/\+0\.74/)).toBeInTheDocument();
  });

  it("updates Stage Evolution Scrubber when user slides to an earlier stage m", () => {
    const onStageChange = vi.fn();
    render(
      <SklearnGbmHudPanel
        frame={mockGbmFrame}
        featureNames={["f0", "f1", "f2", "f3"]}
        activeStage={100}
        onActiveStageChange={onStageChange}
      />
    );

    const slider = screen.getByRole("slider", { name: /stage/i });
    fireEvent.change(slider, { target: { value: "42" } });
    expect(onStageChange).toHaveBeenCalledWith(42);
  });

  it("generates 3D spiral vertices starting from base_value and accumulating stage_raw_deltas", () => {
    const vertices = generateResidualSpiralVertices(
      [1.0, 2.0], // x, y ball position
      mockGbmFrame.base_value,
      mockGbmFrame.telemetry.stage_raw_deltas_sample,
      0.5 // spiral radius
    );

    // 1 base vertex + 4 stage vertices = 5 vertices
    expect(vertices).toHaveLength(5);
    expect(vertices[0].z).toBeCloseTo(-0.619, 3);
    expect(vertices[4].z).toBeCloseTo(-0.619 + 0.74 + 0.12 + 0.62 - 0.05, 3);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, `MAX_TRACED_STAGES=100` & WebGL Tests)

1. **การทดสอบความเร็วของการดึง `stage_raw_deltas` ผ่าน `model.apply(X_f32)`**:
   - พิสูจน์ว่าการใช้ `reg_tree.tree_.value[leaf_matrix[:, m], 0, 0]` แทนการวนเรียก `reg_tree.predict(X_f32)` ทำให้เวลาในการดึงค่าของโมเดล 200 Stages ($2,000$ แถว) เสร็จภายใน **$\le 30 \text{ ms}$**
2. **การจำกัดจำนวน Stage ที่แกะ `decision_path` (`MAX_TRACED_STAGES = 100`)**:
   - เมื่อโมเดลมี `n_estimators = 1,000` ยืนยันว่าลูป `decision_path` รันเพียง 100 Stage แรก และขนาดอาเรย์ `leaf_indices_per_stage` และ `stage_raw_deltas_sample` ถูกตัดที่ `100` คอลัมน์พอดี
3. **การเรนเดอร์ `ResidualSpiralLadder.tsx` ใน 1 Draw Call**:
   - ยืนยันว่าเส้นเกลียวสปริงถูกวาดด้วย `Line` BufferGeometry เดียว (1 Draw Call)

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **`model.estimators_` เป็น `DecisionTreeRegressor` เสมอแม้ใน Classifier** | เทรน `GradientBoostingClassifier` แล้วรัน `trace_internal_flow` | โค้ดอ่าน `reg_tree.tree_.value[:, 0, 0]` และคูณ `learning_rate` ได้ตรงกับ `decision_function` |
| 2 | **Binary vs Multiclass Shape ของ `estimators_` (`(M, 1)` vs `(M, C)`)** | ทดสอบทั้ง `n_classes=2` (`target_class_idx=1`) และ `n_classes=4` (`target_class_idx=3`) | สำหรับ Binary ค่า `self.tree_col == 0` ไม่เกิด `IndexError: index 1 is out of bounds` |
| 3 | **Early Stopping (`n_iter_no_change=5`) เปิดใช้งาน** | เทรนโมเดลโดยตั้ง `n_estimators=200, n_iter_no_change=5` จนหยุดที่รอบที่ 28 | `adapter.n_stages == 28` (อ้างอิงจาก `estimators_.shape[0]`) ไม่ใช่ 200 |
| 4 | **ผู้ใช้ตั้ง `init='zero'`** | เทรน `GradientBoostingRegressor(init="zero")` | `_get_base_raw_score` คืนค่า `0.0` โดยไม่เกิด `AttributeError` บน `model.init_` |
| 5 | **Binary Classification เมื่อเลือกดูมุมมอง `target_class_idx == 0`** | สร้าง Adapter ด้วย `target_class_idx=0` บน Binary GBM | ความน่าจะเป็นใน `staged_predictions` แสดงค่าของคลาส `0` ($1 - P(y=1)$) ได้อย่างถูกต้อง |
| 6 | **Loss แบบ `exponential` ใน `GradientBoostingClassifier`** | เทรน `GradientBoostingClassifier(loss="exponential")` | `staged_predict_proba` และ `decision_function` ทำงานสอดคล้องกันโดยไม่มีความคลาดเคลื่อน |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Raw Margin Additivity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - decision_function)) <= 1e-4` เมื่อ $M \le 100$
- [ ] **Latency SLA**: `trace_internal_flow` สำหรับ $1,000$ แถว (100 Stages, `max_depth=3`) เสร็จภายใน **$\le 180 \text{ ms}$**
- [ ] **Payload Cap SLA**: `stage_raw_deltas_sample` และ `leaf_indices_per_stage` มีความยาวไม่เกิน `100` Stages และ `staged_predictions` ไม่เกิน `50` จุด
- [ ] **WebGL Frame Rate SLA**: `ResidualSpiralLadder` และ `SteppedVoxelTerrain` เรนเดอร์ที่ **$\ge 60 \text{ FPS}$**
- [ ] **Coverage Gate**: Line Coverage ของ `sklearn_gbm.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
