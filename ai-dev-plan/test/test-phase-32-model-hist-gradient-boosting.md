# Test Plan — Phase 32: HistGradientBoosting Adapter & 3D Quantized Histogram Bin Terraces

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-32-model-hist-gradient-boosting.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-32-model-hist-gradient-boosting.md) เพื่อตรวจสอบความถูกต้องของ `HistGBMAdapter` ในการแกะโครงสร้างโหนดระดับ Cython จาก `model._predictors` (`TreePredictor.nodes`), การสกัดขอบเขตถังฮิสโตแกรม 255 ถังจาก `model._bin_mapper.bin_thresholds_`, การคำนวณความสำคัญฟีเจอร์จากผลรวม `gain` (แทน `feature_importances_` ที่ไม่มีใน HistGBM), การเดินต้นไม้เมื่อข้อมูลมีค่าว่าง (`NaN` ผ่านแฟล็ก `missing_go_to_left`), และการแสดงผลตาราง **3D Quantized Histogram Bin Grid** (`HistogramBinGridOverlay.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `HistGBMAdapter` (`backend/src/domains/Precomputation/adapters/boosting/hist_gbm.py`)
  - ค่าคงที่: `MAX_TRACED_ITERS = 100`, `MAX_TRAJECTORY_POINTS = 50`
  - เมธอดหลัก: `compute_gain_feature_importances()`, `extract_bin_thresholds(max_bins_export=128)`, `_walk_single_predictor(predictor, X_f32, contribs)`, `trace_internal_flow(X_chunk)`
  - คลาสโมเดลที่ทดสอบ: `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`
- **Frontend 3D & HUD Components**:
  - `HistogramBinGridOverlay.tsx` (`frontend/src/domains/stage3d/actors/HistogramBinGridOverlay.tsx`)
  - `HistGbmHudPanel.tsx` (`frontend/src/domains/hud/components/HistGbmHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Custom `_walk_single_predictor` vs Native Cython `decision_function` / `predict` Parity**:
   เนื่องจาก `HistGradientBoosting` ไม่มี `.decision_path()` และไม่มี `.apply()` เราจึงต้องพิสูจน์ว่าฟังก์ชัน `_walk_single_predictor` ที่เดินตาม `predictor.nodes` ให้ผลรวมค่าใบไม้ (`leaf_vals`) และผลรวม `feature_contributions` บวกกับ `_baseline_prediction` **เท่ากับค่า `model.decision_function(X_f32)` (หรือ `model.predict(X_f32)`) ของ Cython เป๊ะ 100%** ทั้งกรณีข้อมูลปกติและข้อมูลที่มีค่า `NaN`!
2. **Gain Feature Importance Normalization**:
   ผลลัพธ์จาก `compute_gain_feature_importances()` ต้องมีขนาด `(n_features,)` ทุกค่า $\ge 0$ และผลรวมเท่ากับ `1.0` (เมื่อต้นไม้มีการแตกกิ่งอย่างน้อย 1 โหนด)

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/boosting/test_hist_gbm_adapter.py`)
```python
from __future__ import annotations
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from domains.Precomputation.adapters.boosting.hist_gbm import HistGBMAdapter


def test_hist_gbm_classifier_tree_walker_and_nan_routing_parity():
    """พิสูจน์ว่า _walk_single_predictor เดินตาม missing_go_to_left ได้ตรงกับ Cython decision_function แม้มีค่า NaN."""
    rng = np.random.default_rng(42)
    X, y = make_classification(n_samples=150, n_features=5, n_informative=4, random_state=42)
    X_f32 = X.astype(np.float32)
    # ฉีดค่า NaN ลงไป 20% ของตารางข้อมูลเพื่อทดสอบ missing_go_to_left
    nan_mask = rng.random(X_f32.shape) < 0.20
    X_f32[nan_mask] = np.nan

    model = HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=15, random_state=42).fit(X_f32, y)
    adapter = HistGBMAdapter(model, feature_names=[f"f{i}" for i in range(5)], target_class_idx=1)

    batch = adapter.trace_internal_flow(X_f32)
    tel = batch.extra_telemetry

    expected_raw = model.decision_function(X_f32)
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_raw, atol=1e-4)

    # ผลรวมของ stage_raw_deltas_sample + base_value ต้องเท่ากับ expected_raw เช่นกัน
    stage_sum = batch.base_value + np.sum(tel["stage_raw_deltas_sample"], axis=1)
    np.testing.assert_allclose(stage_sum, expected_raw, atol=1e-4)
    assert tel["has_missing_values"] == np.isnan(X_f32).any(axis=1).tolist()


def test_hist_gbm_regressor_gain_importance_and_bin_thresholds():
    """พิสูจน์ว่า compute_gain_feature_importances และ extract_bin_thresholds ทำงานได้โดยไม่ต้องพึ่ง feature_importances_."""
    X, y = make_regression(n_samples=120, n_features=4, noise=0.3, random_state=17)
    X_f32 = X.astype(np.float32)
    model = HistGradientBoostingRegressor(max_iter=35, max_bins=64, random_state=17).fit(X_f32, y)

    assert not hasattr(model, "feature_importances_")
    adapter = HistGBMAdapter(model, feature_names=["f0", "f1", "f2", "f3"])

    importances = adapter.compute_gain_feature_importances()
    assert importances.shape == (4,)
    assert np.all(importances >= 0.0)
    assert np.sum(importances) == pytest.approx(1.0, abs=1e-5)

    bins = adapter.extract_bin_thresholds(max_bins_export=32)
    assert set(bins.keys()) == {0, 1, 2, 3}
    for f_idx, th_list in bins.items():
        assert len(th_list) <= 64
        # ขอบถังต้องเรียงจากน้อยไปมากเสมอ
        assert th_list == sorted(th_list)


def test_hist_gbm_multiclass_parity():
    """พิสูจน์ความถูกต้องของ HistGBMAdapter บนงาน Multiclass (C=3 คลาส)."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_classes=3, n_clusters_per_class=1, random_state=88
    )
    X_f32 = X.astype(np.float32)
    model = HistGradientBoostingClassifier(max_iter=25, random_state=88).fit(X_f32, y)

    raw_margins = model.decision_function(X_f32)  # shape: (100, 3)
    for c_idx in range(3):
        adapter = HistGBMAdapter(model, feature_names=[f"f{i}" for i in range(5)], target_class_idx=c_idx)
        batch = adapter.trace_internal_flow(X_f32)
        reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
        np.testing.assert_allclose(reconstructed, raw_margins[:, c_idx], atol=1e-4)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`HistogramBinGridOverlay.tsx`**:
  - วาดเส้นแบ่งขอบเขตถังฮิสโตแกรมบนพื้นเวที 3 มิติจาก `bin_thresholds` และระบุหมายเลขถัง (`Bin #b`) ที่ลูกบอลข้อมูลยืนอยู่
- **`HistGbmHudPanel.tsx`**:
  - แสดงป้าย **Active Histogram Bin Badge** และกราฟ **Computed Total Gain Importance**
  - มีปุ่มทดลอง **`[Inject NaN]`** ในโหมด What-If Sandbox เพื่อจำลองการส่งค่าว่างเข้าสู่เส้นทาง `missing_go_to_left`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/HistGbmHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { HistGbmHudPanel } from "../HistGbmHudPanel";
import { findHistogramBinIndex } from "../../../stage3d/actors/HistogramBinGridOverlay";

const mockHistGbmFrame = {
  frame_index: 64,
  prediction: 0.762,
  base_value: -0.405,
  feature_contributions: [0.89, 0.68, 0.0],
  telemetry: {
    model_family: "hist_gbm" as const,
    n_iter_total: 100,
    has_missing_values: true,
    sampled_stage_indices: [0, 10, 50, 99],
    staged_predictions: [0.42, 0.59, 0.71, 0.762],
    stage_raw_deltas_sample: [0.18, 0.14, -0.05, 0.09],
    leaf_indices_per_stage: [12, 9, 15, 7],
  },
};

describe("Phase 32 — HistGbmHudPanel & 3D HistogramBinGridOverlay", () => {
  it("displays Missing Value Routing indicator when frame has_missing_values is true", () => {
    render(
      <HistGbmHudPanel
        frame={mockHistGbmFrame}
        featureNames={["Age", "Income", "Credit"]}
        gainImportances={[0.315, 0.482, 0.203]}
        onToggleInjectNaN={vi.fn()}
      />
    );

    expect(screen.getByText(/NaN/i)).toBeInTheDocument();
    expect(screen.getByText(/48\.2%/i)).toBeInTheDocument();
  });

  it("triggers Inject NaN callback in What-If Sandbox mode", () => {
    const onToggleInjectNaN = vi.fn();
    render(
      <HistGbmHudPanel
        frame={mockHistGbmFrame}
        featureNames={["Age", "Income", "Credit"]}
        gainImportances={[0.315, 0.482, 0.203]}
        onToggleInjectNaN={onToggleInjectNaN}
      />
    );

    const nanBtn = screen.getByRole("button", { name: /inject nan/i });
    fireEvent.click(nanBtn);
    expect(onToggleInjectNaN).toHaveBeenCalledTimes(1);
  });

  it("maps continuous feature value to quantized Histogram Bin index or Bin #255 for NaN", () => {
    const thresholds = [10.0, 20.0, 30.0, 40.0];
    expect(findHistogramBinIndex(5.0, thresholds)).toBe(0);
    expect(findHistogramBinIndex(25.0, thresholds)).toBe(2);
    expect(findHistogramBinIndex(99.0, thresholds)).toBe(4);
    expect(findHistogramBinIndex(Number.NaN, thresholds)).toBe(255);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Fast Gain Extraction & WebGL Tests)

1. **การทดสอบความเร็วของ `compute_gain_feature_importances()`**:
   - บนโมเดล `HistGradientBoostingClassifier(max_iter=200)` การคำนวณความสำคัญฟีเจอร์จาก `predictor.nodes["gain"]` ต้องเสร็จภายใน **$\le 2 \text{ ms}$** (เทียบกับ `permutation_importance` ที่ใช้เวลา $> 2,000 \text{ ms}$)
2. **การทดสอบการตัดขนาด `extract_bin_thresholds(max_bins_export=128)`**:
   - ยืนยันว่าขอบเขตถังที่ส่งออกไปยัง `.mlv` มีความยาวไม่เกิน `128` ค่าต่อฟีเจอร์ และเส้นตารางใน `HistogramBinGridOverlay.tsx` ถูกรวมอยู่ใน `LineSegments` Geometry เดียว (1 Draw Call)

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **`HistGradientBoosting` ไม่มี `feature_importances_`** | เรียกใช้งาน Adapter บน `HistGradientBoostingRegressor` | โค้ดใช้ `compute_gain_feature_importances()` โดยไม่เรียก `model.feature_importances_` |
| 2 | **ข้อมูลมีค่า `NaN` จำนวนมาก (`np.isnan(x_val)`)** | ส่ง `X_chunk` ที่มีค่า `NaN` ทั้งแถวหรือบางคอลัมน์ | `_walk_single_predictor` เช็ก `np.isnan(x_val)` ก่อนเสมอและเดินตาม `missing_go_to_left` ได้ผลลัพธ์ตรงกับ Cython 100% |
| 3 | **Native Categorical Split (`categorical_features`)** | เทรน `HistGradientBoostingClassifier(categorical_features=[0])` | ระบบรองรับโหนดที่มี `is_categorical == 1` โดยไม่พัง |
| 4 | **`nodes["value"]` คูณ `learning_rate` มาให้แล้ว** | เทรนด้วย `learning_rate=0.03` แล้วตรวจสอบผลรวม `contribs` | โค้ดไม่คูณ `learning_rate` ซ้ำ และ `base_value + sum(contribs) == decision_function` เป๊ะ |
| 5 | **Early Stopping ทำให้ `len(_predictors) < max_iter`** | เทรนด้วย `early_stopping=True, max_iter=300` จนหยุดที่รอบที่ 22 | `adapter.n_iter == 22` ตามความยาวจริงของ `model._predictors` |
| 6 | **`_baseline_prediction` มี Shape `(1,)` vs `(1, C)`** | ทดสอบทั้ง Binary/Regression และ Multiclass 4 คลาส | `float(np.ravel(self.model._baseline_prediction)[self.tree_col])` ดึงค่าสเกลาร์ได้โดยไม่เกิด ValueError |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Tree Walker & NaN Routing Parity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - decision_function)) <= 1e-4` แม้ข้อมูลจะมีค่า `NaN` ปะปนอยู่
- [ ] **Gain Importance Latency SLA**: `compute_gain_feature_importances()` เสร็จภายใน **$\le 2 \text{ ms}$**
- [ ] **Chunk Throughput SLA**: `trace_internal_flow` สำหรับ $1,000$ แถว (100 Iterations) เสร็จภายใน **$\le 120 \text{ ms}$**
- [ ] **WebGL Frame Rate SLA**: `HistogramBinGridOverlay` และพื้นผิวขั้นบันไดเรนเดอร์ที่ **$\ge 60 \text{ FPS}$**
- [ ] **Coverage Gate**: Line Coverage ของ `hist_gbm.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
