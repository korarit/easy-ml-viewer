# Test Plan — Phase 28: Random Forest Adapter & 3D Multi-Tree Voting Arena + Active Branch Glow

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-28-model-random-forest-consensus.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-28-model-random-forest-consensus.md) เพื่อพิสูจน์ความถูกต้องและประสิทธิภาพของ `BaseForestAdapter` และ `RandomForestAdapter` ในการดึงผลโหวตของต้นไม้ทุกต้นในป่าผ่าน `model.apply(X_f32)` + Node Expectation Lookup, การคำนวณ `consensus_ratio`, การป้องกัน `UserWarning` เรื่อง `feature_names_in_`, และการเรนเดอร์วงแหวนเสาคริสตัล **3D Forest Voting Ring** (`ForestVotingArena.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `BaseForestAdapter`, `RandomForestAdapter` (`backend/src/domains/Precomputation/adapters/trees/random_forest.py`)
  - ค่าคงที่: `MAX_EXPORTED_TREES = 100`
  - เมธอดหลัก: `_extract_node_expectations()`, `predict_all_trees_fast(X_f32)`, `trace_internal_flow(X_chunk)`
  - คลาสโมเดลที่ทดสอบ: `RandomForestClassifier`, `RandomForestRegressor`
- **Frontend 3D & HUD Components**:
  - `ForestVotingArena.tsx` (`frontend/src/domains/stage3d/actors/ForestVotingArena.tsx`)
  - `SteppedVoxelTerrain.tsx` (`frontend/src/domains/stage3d/actors/SteppedVoxelTerrain.tsx`)
  - `RandomForestHudPanel.tsx` (`frontend/src/domains/hud/components/RandomForestHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Fast Leaf Lookup vs Individual `tree.predict` Parity**:
   ผลลัพธ์ `tree_preds` จาก `predict_all_trees_fast(X_f32)` (ที่ใช้ `model.apply(X_f32)` เปิดตาราง `_node_exp_per_tree`) ต้องเท่ากับผลลัพธ์จากการวนเรียก `[est.predict_proba(X_f32)[:, target_c] for est in model.estimators_]` (หรือ `est.predict(X_f32)` ในงาน Regression) ทุกต้นไม้และทุกแถว (`atol=1e-5`)
2. **Ensemble Mean & Local Accuracy Parity**:
   - ค่า `inter_tree_mean` ต้องเท่ากับ `model.predict_proba(X_f32)[:, target_c]` (หรือ `model.predict(X_f32)`)
   - เมื่อจำนวนต้นไม้ $K \le 100$ ผลรวม `base_value + np.sum(feature_contributions, axis=1)` ต้องเท่ากับ `inter_tree_mean` ทุกประการ (`atol=1e-5`)
3. **Zero `UserWarning` Guarantee (`feature_names_in_`)**:
   เมื่อโมเดลถูกเทรนด้วย `pd.DataFrame` ที่มีชื่อคอลัมน์ การรัน `trace_internal_flow` ต้องไม่ปล่อย `UserWarning: X has feature names, but DecisionTreeClassifier was fitted without feature names` ออกมาแม้แต่ครั้งเดียว

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/trees/test_random_forest_adapter.py`)
```python
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from domains.Precomputation.adapters.trees.random_forest import (
    MAX_EXPORTED_TREES,
    RandomForestAdapter,
)


def test_random_forest_classifier_fast_lookup_and_local_accuracy():
    """พิสูจน์ว่า predict_all_trees_fast ตรงกับ estimators_[k].predict_proba และผลรวม contributions ตรงกับค่าทำนายป่า."""
    X, y = make_classification(n_samples=80, n_features=5, n_informative=3, random_state=42)
    X_f32 = X.astype(np.float32)
    model = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42).fit(X_f32, y)

    adapter = RandomForestAdapter(model, feature_names=[f"f{i}" for i in range(5)], target_class_idx=1)
    tree_preds, leaf_matrix = adapter.predict_all_trees_fast(X_f32)

    # 1. ตรวจสอบความเท่ากันรายต้นไม้กับ est.predict_proba
    manual_preds = np.column_stack([est.predict_proba(X_f32)[:, 1] for est in model.estimators_])
    np.testing.assert_allclose(tree_preds, manual_preds, atol=1e-5)

    # 2. ตรวจสอบ Local Accuracy ของทั้งป่า
    batch = adapter.trace_internal_flow(X_f32)
    expected_forest_prob = model.predict_proba(X_f32)[:, 1]
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_forest_prob, atol=1e-4)
    np.testing.assert_allclose(batch.extra_telemetry["inter_tree_mean"], expected_forest_prob, atol=1e-4)

    # 3. ตรวจสอบขอบเขตของ consensus_ratio ต้องอยู่ใน [0.5, 1.0]
    consensus = np.asarray(batch.extra_telemetry["consensus_ratio"])
    assert np.all((consensus >= 0.5) & (consensus <= 1.0))


def test_random_forest_regressor_parity_and_truncation_over_100_trees():
    """พิสูจน์ว่าเมื่อ n_estimators=250 ระบบจะเก็บตัวอย่างรายต้น 100 ต้นแรก แต่คำนวณ mean/std จากทั้ง 250 ต้น."""
    X, y = make_regression(n_samples=60, n_features=4, noise=0.2, random_state=11)
    X_f32 = X.astype(np.float32)
    model = RandomForestRegressor(n_estimators=250, max_depth=4, random_state=11).fit(X_f32, y)

    adapter = RandomForestAdapter(model, feature_names=["f0", "f1", "f2", "f3"])
    batch = adapter.trace_internal_flow(X_f32)

    # สถิติรวมต้องมาจากทั้ง 250 ต้น
    expected_mean = model.predict(X_f32)
    np.testing.assert_allclose(batch.extra_telemetry["inter_tree_mean"], expected_mean, atol=1e-3)

    # อาเรย์รายต้นใน Telemetry ต้องถูกตัดให้เหลือ MAX_EXPORTED_TREES = 100 ต้นพอดี
    sample_preds = np.asarray(batch.extra_telemetry["tree_predictions_sample"])
    sample_leaves = np.asarray(batch.extra_telemetry["leaf_indices_sample"])
    assert sample_preds.shape == (60, MAX_EXPORTED_TREES)
    assert sample_leaves.shape == (60, MAX_EXPORTED_TREES)


def test_no_feature_names_in_warning_when_fitted_on_dataframe():
    """ห้ามเกิด UserWarning เรื่อง feature_names_in_ เมื่อโมเดลเทรนด้วย DataFrame."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=9)
    df = pd.DataFrame(X, columns=["age", "income", "debt", "score"])
    model = RandomForestClassifier(n_estimators=15, random_state=9).fit(df, y)

    adapter = RandomForestAdapter(model, feature_names=list(df.columns))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = adapter.trace_internal_flow(df.to_numpy(dtype=np.float32))
        feature_warnings = [w for w in caught if "feature names" in str(w.message).lower()]
        assert len(feature_warnings) == 0
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`ForestVotingArena.tsx`**:
  - วาดเสาคริสตัลจำนวน `min(n_estimators, 100)` ต้นเรียงเป็นวงแหวนด้วย `THREE.InstancedMesh` เพียง 1 ตัว
  - อัปเดตความสูง (`scale.y`) และสีของแต่ละอินสแตนซ์ตาม `tree_predictions_sample[k]` ในเฟรมปัจจุบัน
- **`RandomForestHudPanel.tsx`**:
  - แสดงแถบ **Consensus & Vote Distribution Bar** (เช่น `84.0% Agreement`)
  - รองรับการเลือก **Tree Selector (`Tree #k Explorer`)** เพื่อสลับโหมดฉาก 3 มิติจาก `Ensemble Surface` ไปเป็นการส่องต้นไม้เดี่ยว `Tree #k` พร้อมแสดง `leaf_indices_sample[k]`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/RandomForestHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RandomForestHudPanel } from "../RandomForestHudPanel";
import { computeVotingPillarTransforms } from "../../../stage3d/actors/ForestVotingArena";

const mockForestFrame = {
  frame_index: 108,
  prediction: 0.812,
  base_value: 0.5,
  feature_contributions: [0.15, 0.0, 0.19, 0.0, -0.028],
  telemetry: {
    model_family: "random_forest" as const,
    n_estimators_total: 200,
    n_estimators_exported: 5,
    consensus_ratio: 0.84,
    inter_tree_mean: 0.812,
    inter_tree_std: 0.145,
    tree_predictions_sample: [0.92, 0.88, 0.15, 0.94, 0.79],
    leaf_indices_sample: [14, 22, 7, 23, 19],
  },
};

describe("Phase 28 — RandomForestHudPanel & 3D ForestVotingArena", () => {
  it("displays Ensemble Consensus ratio, inter-tree std, and individual tree vote distribution", () => {
    render(
      <RandomForestHudPanel
        frame={mockForestFrame}
        featureNames={["f0", "f1", "f2", "f3", "f4"]}
        selectedTreeIndex={null}
        onSelectTree={vi.fn()}
      />
    );

    expect(screen.getByText(/84\.0%/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.812/)).toBeInTheDocument();
    expect(screen.getByText(/0\.145/)).toBeInTheDocument();
  });

  it("allows selecting a specific Tree #k and displays its individual prediction and Leaf ID", () => {
    const onSelectTree = vi.fn();
    render(
      <RandomForestHudPanel
        frame={mockForestFrame}
        featureNames={["f0", "f1", "f2", "f3", "f4"]}
        selectedTreeIndex={3}
        onSelectTree={onSelectTree}
      />
    );

    // Tree #3 has prediction 0.94 and leaf_id 23
    expect(screen.getByText(/0\.94/)).toBeInTheDocument();
    expect(screen.getByText(/#23/)).toBeInTheDocument();
  });

  it("computes ring positions and pillar heights for 3D ForestVotingArena InstancedMesh", () => {
    const transforms = computeVotingPillarTransforms(
      mockForestFrame.telemetry.tree_predictions_sample,
      10.0 // arena radius
    );
    expect(transforms).toHaveLength(5);
    expect(transforms[0].scaleY).toBeGreaterThan(transforms[2].scaleY); // 0.92 > 0.15
    expect(transforms[0].isPositiveVote).toBe(true);
    expect(transforms[2].isPositiveVote).toBe(false);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, `MAX_EXPORTED_TREES=100` Pruning & WebGL Tests)

1. **การทดสอบความเร็วของ `predict_all_trees_fast` เทียบกับ Naive Python Loop**:
   - บนโมเดล `RandomForestClassifier(n_estimators=200)` และ `X_chunk` ขนาด $2,000$ แถว วัดเวลาการทำงานของ `predict_all_trees_fast(X_f32)` ต้องเสร็จภายใน **$\le 25 \text{ ms}$** (เร็วกว่าการส่ง `DataFrame` เข้า `[tree.predict_proba(df) for tree in model.estimators_]` อย่างน้อย **$10\times$**)
2. **การทดสอบขนาด Payload เมื่อ `n_estimators = 2,000`**:
   - ยืนยันว่าขนาด JSON ของ `extra_telemetry` ต่อ 1 เฟรมไม่เกิน **$2.5 \text{ KB}$** เนื่องจากถูกตัดด้วย `MAX_EXPORTED_TREES = 100`
3. **การทดสอบ `InstancedMesh` Draw Call ของเสาคริสตัล 100 ต้น**:
   - ใน `ForestVotingArena.tsx` ตรวจสอบว่าเสาคริสตัลทั้ง 100 ต้นถูกวาดด้วย `THREE.InstancedMesh` เพียง 1 อ็อบเจกต์ (`count = 100`, Draw Call $= 1$)

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **`estimators_[k]` ไม่มี `feature_names_in_`** | เทรนโมเดลด้วย `pd.DataFrame` แล้วส่งเข้า `RandomForestAdapter` | ไม่มี `UserWarning` เรื่อง `feature_names_in_` และทำงานด้วย `np.float32` ตลอดสาย |
| 2 | **ป่าขนาดใหญ่มาก (`n_estimators = 1500`)** | สร้างป่า 1,500 ต้น แล้วรัน `trace_internal_flow` | `len(tree_predictions_sample[0]) == 100` แต่ `inter_tree_mean` คำนวณจากทั้ง 1,500 ต้นครบถ้วน |
| 3 | **โมเดลถูกเซฟมาพร้อม `n_jobs = -1`** | ตั้ง `model.n_jobs = -1` ก่อนส่งเข้า Worker ของ `ComputePool` | ระบบจำกัด Thread ภายใน Worker ไม่ให้เกิด CPU Oversubscription และไม่เกิด Deadlock |
| 4 | **ต้นไม้บางต้นในป่ามีโหนดเดียว (`node_count == 1`)** | บังคับให้ `estimators_[0]` เป็น Stump ที่ไม่แตกกิ่ง (`max_depth=0` หรือข้อมูลซ้ำ) | ลูปแกะ `decision_path` ข้ามต้นไม้ที่มีโหนดเดียวไปอย่างปลอดภัยโดยไม่เกิด IndexError หรือ ZeroDivisionError |
| 5 | **Multiclass ที่ต้นไม้ย่อยบางต้นเห็นคลาสไม่ครบตอน Bootstrap** | สร้างข้อมูลที่คลาสที่ 3 มีตัวอย่างน้อยมากจนบาง Bootstrap Sample ไม่มีคลาสที่ 3 | `_extract_node_expectations` จัดการขนาด `probs.shape[1]` ได้โดยไม่เกิด `IndexError` |
| 6 | **โมเดลไม่ได้เปิด `oob_score=True` ตอนเทรน** | เทรน `RandomForestClassifier(oob_score=False)` (ค่าเริ่มต้น) | โค้ดอ่าน `getattr(model, "oob_score_", None)` ได้ค่า `None` โดยไม่เกิด `AttributeError` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Fast Lookup Parity SLA**: `np.max(np.abs(tree_preds - manual_estimator_preds)) <= 1e-5` สำหรับทุกต้นไม้ในป่า
- [ ] **Throughput SLA**: สกัดข้อมูล `trace_internal_flow` จำนวน $1,000$ แถว (สำหรับป่า 100 ต้น) เสร็จภายใน **$\le 450 \text{ ms}$**
- [ ] **Payload Size SLA**: อาเรย์ `tree_predictions_sample` และ `leaf_indices_sample` มีความยาวไม่เกิน `MAX_EXPORTED_TREES = 100` เสมอ
- [ ] **WebGL Draw Call SLA**: `ForestVotingArena.tsx` วาดเสาคริสตัล 100 ต้นใน **`1 Draw Call`** และรักษาเฟรมเรต **$\ge 60 \text{ FPS}$**
- [ ] **Coverage Gate**: Line Coverage ของ `random_forest.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
