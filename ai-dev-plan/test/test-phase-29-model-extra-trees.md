# Test Plan — Phase 29: Extra Trees Adapter & Randomized Split Variance 3D Visualizer

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-29-model-extra-trees.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-29-model-extra-trees.md) เพื่อตรวจสอบความถูกต้องของ `ExtraTreesAdapter` ในการคำนวณความแปรปรวนระหว่างต้นไม้ (`inter_tree_std` ด้วย `ddof=0`), รัศมีหมอกความไม่แน่นอน (`uncertainty_radius` $\in [0, 1]$), ช่วงเปอร์เซ็นไทล์ที่ 5–95 (`prediction_p05`, `prediction_p95`), ความแปรปรวนของแรงผลักรายฟีเจอร์ (`contribution_std`), การสกัดการกระจายตัวของจุดตัดสุ่ม (`extract_split_threshold_distribution`), และการแสดงผล **3D Uncertainty Halo** (`UncertaintyHaloActor.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `ExtraTreesAdapter` (`backend/src/domains/Precomputation/adapters/trees/extra_trees.py` สืบทอดจาก `BaseForestAdapter`)
  - เมธอดหลัก: `extract_split_threshold_distribution(max_splits_per_feature=200)`, `trace_internal_flow(X_chunk)`
  - คลาสโมเดลที่ทดสอบ: `ExtraTreesClassifier`, `ExtraTreesRegressor`
- **Frontend 3D & HUD Components**:
  - `UncertaintyHaloActor.tsx` (`frontend/src/domains/stage3d/actors/UncertaintyHaloActor.tsx`)
  - `SplitDensityCarpet.tsx` (`frontend/src/domains/stage3d/actors/SplitDensityCarpet.tsx`)
  - `ExtraTreesHudPanel.tsx` (`frontend/src/domains/hud/components/ExtraTreesHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Epistemic Uncertainty & Percentile Bounds**:
   - สำหรับทุกแถว $i$: `prediction_p05[i] <= inter_tree_mean[i] <= prediction_p95[i]`
   - ค่า `uncertainty_radius[i]` ต้องอยู่ในช่วงปิด $[0.0, 1.0]$ เสมอ และห้ามมีค่า `NaN` หรือ `inf` แม้ในกรณีที่ `n_estimators == 1` หรือข้อมูลทุกแถวทำนายได้ค่าเท่ากันหมด
2. **Feature Contribution Mean & Standard Deviation Parity**:
   - ค่า `feature_contributions` คือค่าเฉลี่ย $\frac{1}{K'} \sum_{k=1}^{K'} \phi_{k, i, j}$ และต้องสอดคล้องกับสมการ Local Accuracy: `base_value + sum(feature_contributions[i]) == mean(tree_preds[i, :K'])`
   - ค่า `contribution_std` ต้องมีค่า $\ge 0$ ทุกช่อง และเท่ากับ `np.std(per_tree_contribs, axis=0, ddof=0)`

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/trees/test_extra_trees_adapter.py`)
```python
from __future__ import annotations
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from domains.Precomputation.adapters.trees.extra_trees import ExtraTreesAdapter


def test_extra_trees_regressor_uncertainty_telemetry_and_local_accuracy():
    """พิสูจน์ความถูกต้องของ inter_tree_std, p05/p95, uncertainty_radius และ contribution_std."""
    X, y = make_regression(n_samples=90, n_features=5, noise=5.0, random_state=42)
    X_f32 = X.astype(np.float32)
    model = ExtraTreesRegressor(n_estimators=50, max_depth=6, random_state=42).fit(X_f32, y)

    adapter = ExtraTreesAdapter(model, feature_names=[f"f{i}" for i in range(5)])
    batch = adapter.trace_internal_flow(X_f32)
    tel = batch.extra_telemetry

    expected_preds = model.predict(X_f32)
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_preds, atol=1e-3)

    p05 = np.asarray(tel["prediction_p05"])
    mean = np.asarray(tel["inter_tree_mean"])
    p95 = np.asarray(tel["prediction_p95"])
    assert np.all(p05 <= mean + 1e-4)
    assert np.all(mean <= p95 + 1e-4)

    halo = np.asarray(tel["uncertainty_radius"])
    assert not np.isnan(halo).any()
    assert np.all((halo >= 0.0) & (halo <= 1.0))

    contrib_std = np.asarray(tel["contribution_std"])
    assert contrib_std.shape == (90, 5)
    assert np.all(contrib_std >= 0.0)


def test_extra_trees_single_estimator_ddof_zero_prevents_nan():
    """เมื่อ n_estimators == 1 การคำนวณ std ต้องใช้ ddof=0 และคืนค่า 0.0 โดยไม่เกิด NaN."""
    X, y = make_classification(n_samples=40, n_features=4, random_state=13)
    X_f32 = X.astype(np.float32)
    model = ExtraTreesClassifier(n_estimators=1, random_state=13).fit(X_f32, y)

    adapter = ExtraTreesAdapter(model, feature_names=["f0", "f1", "f2", "f3"])
    batch = adapter.trace_internal_flow(X_f32)
    tel = batch.extra_telemetry

    assert not np.isnan(tel["inter_tree_std"]).any()
    assert not np.isnan(tel["uncertainty_radius"]).any()
    assert not np.isnan(tel["contribution_std"]).any()
    np.testing.assert_allclose(tel["inter_tree_std"], 0.0, atol=1e-7)
    np.testing.assert_allclose(tel["uncertainty_radius"], 0.0, atol=1e-7)


def test_extract_split_threshold_distribution_respects_cap():
    """พิสูจน์ว่าการสกัดจุดตัดสุ่มของแต่ละฟีเจอร์ไม่เกิน max_splits_per_feature ที่กำหนด."""
    X, y = make_classification(n_samples=200, n_features=4, random_state=21)
    model = ExtraTreesClassifier(n_estimators=60, max_depth=8, random_state=21).fit(X, y)

    adapter = ExtraTreesAdapter(model, feature_names=["f0", "f1", "f2", "f3"])
    dist = adapter.extract_split_threshold_distribution(max_splits_per_feature=50)

    assert set(dist.keys()) == {0, 1, 2, 3}
    for f_idx, cutpoints in dist.items():
        assert len(cutpoints) <= 50
        assert all(isinstance(val, float) for val in cutpoints)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`UncertaintyHaloActor.tsx`**:
  - คำนวณขนาดทรงกลมหมอกเรืองแสงรอบลูกบอลข้อมูลจาก `uncertainty_radius` ($r_{\text{halo}} \in [0, 1]$)
  - เปลี่ยนสี Shader จากสีเขียวมรกต (ความมั่นใจสูง $r_{\text{halo}} < 0.2$) ไปเป็นสีม่วง/ส้มแอมเบอร์เมื่อต้นไม้มีความเห็นขัดแย้งกันสูง ($r_{\text{halo}} > 0.5$)
- **`SplitDensityCarpet.tsx`**:
  - แสดงแถบ Rug Plot / Barcode บนขอบแกน $X, Y$ จาก `split_threshold_distribution`
- **`ExtraTreesHudPanel.tsx`**:
  - แสดงหนวดกุ้ง Error Bars ($\pm \text{contribution\_std}$) บนกราฟแท่งแรงผลักฟีเจอร์ และปุ่มสลับพื้นผิว `Surface Mode Switcher` (`Mean Prediction` vs `Epistemic Uncertainty`)

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/ExtraTreesHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExtraTreesHudPanel } from "../ExtraTreesHudPanel";
import { computeHaloScaleAndColor } from "../../../stage3d/actors/UncertaintyHaloActor";

const mockExtraTreesFrame = {
  frame_index: 7,
  prediction: 42.85,
  base_value: 35.0,
  feature_contributions: [9.2, -2.15, 0.8],
  telemetry: {
    model_family: "extra_trees" as const,
    inter_tree_mean: 42.85,
    inter_tree_std: 6.42,
    uncertainty_radius: 0.68,
    prediction_p05: 30.12,
    prediction_p95: 55.04,
    contribution_std: [3.1, 0.82, 0.21],
    tree_predictions_sample: [48.1, 39.5, 44.2, 31.8],
    leaf_indices_sample: [39, 18, 27, 12],
  },
};

describe("Phase 29 — ExtraTreesHudPanel & 3D UncertaintyHaloActor", () => {
  it("renders uncertainty radius badge, 95% tree range, and contribution error bars", () => {
    render(
      <ExtraTreesHudPanel
        frame={mockExtraTreesFrame}
        featureNames={["Temp", "Humid", "Wind"]}
        surfaceMode="prediction"
        onSurfaceModeChange={vi.fn()}
      />
    );

    expect(screen.getByText(/±\s*6\.42/)).toBeInTheDocument();
    expect(screen.getByText(/30\.1/)).toBeInTheDocument();
    expect(screen.getByText(/55\.0/)).toBeInTheDocument();
    expect(screen.getByText(/±\s*3\.1/)).toBeInTheDocument();
  });

  it("toggles Surface Mode between Mean Prediction Surface and Uncertainty Surface", () => {
    const onSurfaceModeChange = vi.fn();
    render(
      <ExtraTreesHudPanel
        frame={mockExtraTreesFrame}
        featureNames={["Temp", "Humid", "Wind"]}
        surfaceMode="prediction"
        onSurfaceModeChange={onSurfaceModeChange}
      />
    );

    const toggleBtn = screen.getByRole("button", { name: /uncertainty/i });
    fireEvent.click(toggleBtn);
    expect(onSurfaceModeChange).toHaveBeenCalledWith("uncertainty");
  });

  it("scales UncertaintyHaloActor radius proportionally to uncertainty_radius", () => {
    const lowUncertainty = computeHaloScaleAndColor(0.05, 1.0);
    const highUncertainty = computeHaloScaleAndColor(0.75, 1.0);

    expect(highUncertainty.scale).toBeGreaterThan(lowUncertainty.scale);
    expect(highUncertainty.severityLevel).toBe("high");
    expect(lowUncertainty.severityLevel).toBe("low");
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, `per_tree_contribs` Buffer & WebGL Tests)

1. **การตรวจสอบหน่วยความจำของ `per_tree_contribs`**:
   - ใน `ExtraTreesAdapter.trace_internal_flow` อาเรย์ `per_tree_contribs` ขนาด `(k_export=100, n_samples=2000, n_features=50)` ชนิด `float32` ต้องใช้หน่วยความจำเพียง $40 \text{ MB}$ ชั่วคราวต่อ Chunk และถูกคืนให้กับระบบทันทีหลังคำนวณ `mean_contribs` และ `std_contribs` เสร็จ
2. **การควบคุมขนาด `extract_split_threshold_distribution`**:
   - ทดสอบกับป่า Extra Trees ที่มีโหนดรวมกันมากกว่า $100,000$ โหนด ยืนยันว่าลิสต์จุดตัดต่อฟีเจอร์ถูกตัดที่ `max_splits_per_feature = 200` ค่าเสมอ ใช้เวลาสกัดไม่เกิน $10 \text{ ms}$
3. **การทดสอบ `UncertaintyHaloActor.tsx` บน GPU**:
   - ยืนยันว่า `SphereGeometry` และ `ShaderMaterial` ของวงรัศมีหมอกถูกสร้างเพียงครั้งเดียวใน `useMemo` และอัปเดตเฉพาะ Uniform `uUncertaintyRadius` กับ `mesh.scale` ใน `useFrame`

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **ป่าที่มีต้นไม้เพียงต้นเดียว (`n_estimators == 1`)** | เทรน `ExtraTreesRegressor(n_estimators=1)` | `inter_tree_std == 0.0`, `contribution_std == 0.0` และไม่มี `NaN` 100% |
| 2 | **งาน Regression ที่ทุกแถวใน Chunk ทำนายได้ค่าเท่ากัน (`ptp == 0`)** | ส่ง `X_chunk` ที่ทุกแถวมีค่าเหมือนกันทั้งหมดเข้า `trace_internal_flow` | การหารด้วย `pred_span = np.ptp(inter_tree_mean) + 1e-6` ไม่เกิด `ZeroDivisionError` หรือ `inf` |
| 3 | **ต้นไม้ใน Extra Trees ลึกและมีโหนดเยอะกว่า RF 3 เท่า** | เทรน `ExtraTreesClassifier(n_estimators=150, max_depth=None)` | `predict_all_trees_fast` และ `k_export = 100` ทำงานเสร็จภายใน SLA $\le 500 \text{ ms}$ |
| 4 | **`bootstrap=False` ทำให้ไม่มี `oob_score_`** | ใช้ค่าเริ่มต้นของ Extra Trees (`bootstrap=False`) | HUD ไม่พยายามอ่าน `model.oob_score_` และไม่เกิด `AttributeError` |
| 5 | **ฟีเจอร์ที่เป็น Binary (0/1) หรือค่าคงที่** | เทรนด้วยฟีเจอร์ที่มีเฉพาะค่า `0` และ `1` | `SplitDensityCarpet.tsx` จัดกลุ่มจุดตัดลง Histogram Bins โดยไม่วาดเส้นซ้อนทับกันเกินพิกัด |
| 6 | **Multiclass Classification ($C = 5$ คลาส)** | เทรน `ExtraTreesClassifier` บน 5 คลาส | คำนวณ `inter_tree_std` และ `uncertainty_radius` ของคลาสเป้าหมายได้อย่างถูกต้อง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Zero-NaN Uncertainty SLA**: `uncertainty_radius`, `inter_tree_std`, และ `contribution_std` ต้องไม่มีค่า `NaN` หรือ `inf` ในทุกกรณี (รวมถึง `n_estimators=1` และ `ptp=0`)
- [ ] **Mathematical Parity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - expected_pred)) <= 1e-4` เมื่อ $K \le 100$
- [ ] **Latency SLA**: `trace_internal_flow` สำหรับ $1,000$ แถว (100 ต้น) เสร็จภายใน **$\le 500 \text{ ms}$**
- [ ] **WebGL Frame Rate SLA**: `UncertaintyHaloActor` และ `SplitDensityCarpet` เรนเดอร์ร่วมกับฉากหลักที่ **$\ge 60 \text{ FPS}$**
- [ ] **Coverage Gate**: Line Coverage ของ `extra_trees.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
