# Test Plan — Phase 34: LightGBM Adapter (`.pkl`, `.txt`, `.model`) & 3D Leaf-Wise Split Hierarchy

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-34-model-lightgbm.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-34-model-lightgbm.md) เพื่อตรวจสอบความถูกต้องของ `LightGBMAdapter` ในการโหลดโมเดลทั้งจาก `LGBMClassifier`, `LGBMRegressor` และไฟล์ Native `lightgbm.Booster` (`.txt`, `.model`), การดึงค่า **Exact C++ TreeSHAP (`pred_contrib=True`)** พร้อมการ **`.reshape(n_samples, n_classes, n_features + 1)`** สำหรับงาน Multiclass, การดึง `pred_leaf=True`, การแกะโครงสร้างการแตกกิ่งแบบอสมมาตรจาก `booster.dump_model(num_iteration=100)`, และการแสดงผล **3D Leaf-Wise Asymmetric Tree Hierarchy** (`LeafWiseTreeVisualizer.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `LightGBMAdapter` (`backend/src/domains/Precomputation/adapters/boosting/lightgbm_adapter.py`)
  - ค่าคงที่: `MAX_EXPORTED_TREES = 100`, `MAX_TRAJECTORY_POINTS = 50`
  - เมธอดหลัก: `extract_model_dump_summary(max_trees=100)`, `trace_internal_flow(X_chunk)`
  - คลาสและไฟล์โมเดลที่ทดสอบ: `lgb.LGBMClassifier`, `lgb.LGBMRegressor`, `lgb.Booster` (ไฟล์ `.pkl`, `.joblib`, `.txt`, `.model`)
- **Frontend 3D & HUD Components**:
  - `LeafWiseTreeVisualizer.tsx` (`frontend/src/domains/stage3d/actors/LeafWiseTreeVisualizer.tsx`)
  - `TreeShapForceArrows.tsx` (`frontend/src/domains/stage3d/actors/TreeShapForceArrows.tsx`)
  - `LightgbmHudPanel.tsx` (`frontend/src/domains/hud/components/LightgbmHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Exact C++ TreeSHAP Additivity (`pred_contrib=True` vs `raw_score=True`)**:
   ผลรวมของ `base_value` บวกกับผลรวมของ `feature_contributions` ที่สกัดจาก `booster.predict(X_f32, pred_contrib=True)` ต้องเท่ากับค่า Raw Score ที่ได้จาก `booster.predict(X_f32, raw_score=True)` ทุกแถวและทุกคลาส:
   $$\left| \text{Bias}_c + \sum_{j=1}^{d} \phi_{i, c, j} - \hat{y}_{\text{raw}, c}(\mathbf{x}_i) \right| \le 10^{-4}$$
2. **Multiclass 2D Flattened Array Reshape Verification**:
   พิสูจน์ว่าสำหรับโมเดล Multiclass ($C \ge 3$ คลาส) ที่ `booster.predict(X_f32, pred_contrib=True)` คืนอาเรย์ 2 มิติขนาด `(n_samples, C * (n_features + 1))` โค้ดของ `LightGBMAdapter` ทำการ `.reshape(n_samples, C, n_features + 1)` ได้อย่างถูกต้องและให้ผลบวกตรงกับ `raw_score[:, c]` ของแต่ละคลาส 100%!

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/boosting/test_lightgbm_adapter.py`)
```python
from __future__ import annotations
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from domains.Precomputation.adapters.boosting.lightgbm_adapter import LightGBMAdapter


def test_lightgbm_binary_and_native_txt_booster_treeshap_parity(tmp_path: Path):
    """พิสูจน์ว่า pred_contrib=True บวกได้ตรงกับ raw_score=True และไฟล์ .txt ให้ผลเหมือน LGBMClassifier."""
    X, y = make_classification(n_samples=100, n_features=6, n_informative=4, random_state=42)
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    feat_names = [f"f{i}" for i in range(6)]

    clf = lgb.LGBMClassifier(n_estimators=25, num_leaves=15, learning_rate=0.1, random_state=42, verbose=-1)
    clf.fit(X_f32, y, feature_name=feat_names)

    # 1. ทดสอบผ่าน Scikit-learn Wrapper
    adapter_wrap = LightGBMAdapter(clf, feature_names=feat_names, target_class_idx=1)
    batch_wrap = adapter_wrap.trace_internal_flow(X_f32)

    expected_raw = clf.booster_.predict(X_f32, raw_score=True)
    reconstructed = batch_wrap.base_value + np.sum(batch_wrap.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_raw, atol=1e-4)

    # 2. บันทึกเป็นไฟล์ .txt แล้วโหลดผ่าน lgb.Booster(model_file=...)
    txt_path = tmp_path / "lgb_model.txt"
    clf.booster_.save_model(str(txt_path))
    adapter_txt = LightGBMAdapter(txt_path, feature_names=feat_names, target_class_idx=1)
    batch_txt = adapter_txt.trace_internal_flow(X_f32)

    np.testing.assert_allclose(batch_txt.feature_contributions, batch_wrap.feature_contributions, atol=1e-5)
    assert batch_txt.extra_telemetry["leaf_indices_sample"] == batch_wrap.extra_telemetry["leaf_indices_sample"]


def test_lightgbm_multiclass_2d_to_3d_reshape_parity():
    """พิสูจน์การ Reshape อาเรย์ 2 มิติ (n_samples, C * (d + 1)) ของ Multiclass pred_contrib=True."""
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_classes=4, n_clusters_per_class=1, random_state=33
    )
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    clf = lgb.LGBMClassifier(n_estimators=20, num_leaves=12, random_state=33, verbose=-1).fit(X_f32, y)

    expected_raw_scores = clf.booster_.predict(X_f32, raw_score=True)  # shape: (90, 4)

    for c_idx in range(4):
        adapter = LightGBMAdapter(clf, feature_names=[f"f{i}" for i in range(5)], target_class_idx=c_idx)
        batch = adapter.trace_internal_flow(X_f32)
        assert batch.feature_contributions.shape == (90, 5)

        reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
        np.testing.assert_allclose(reconstructed, expected_raw_scores[:, c_idx], atol=1e-4)

        leaf_mat = np.asarray(batch.extra_telemetry["leaf_indices_sample"])
        assert leaf_mat.shape == (90, 20)


def test_lightgbm_single_tree_1d_leaf_array_and_dump_model():
    """พิสูจน์ว่าเมื่อ n_estimators == 1 อาเรย์ pred_leaf ที่เป็น 1D จะถูก reshape เป็น (n_samples, 1) อย่างปลอดภัย."""
    X, y = make_regression(n_samples=45, n_features=4, random_state=8)
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    reg = lgb.LGBMRegressor(n_estimators=1, num_leaves=8, random_state=8, verbose=-1).fit(X_f32, y)

    adapter = LightGBMAdapter(reg, feature_names=["a", "b", "c", "d"])
    batch = adapter.trace_internal_flow(X_f32)
    leaf_mat = np.asarray(batch.extra_telemetry["leaf_indices_sample"])
    assert leaf_mat.shape == (45, 1)

    dump = adapter.extract_model_dump_summary(max_trees=5)
    assert len(dump["tree_info"]) == 1
    assert "tree_structure" in dump["tree_info"][0]
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`LeafWiseTreeVisualizer.tsx`**:
  - แปลงโครงสร้าง `"tree_structure"` จาก `dump_model()` ให้เป็นกราฟต้นไม้ 3 มิติแบบอสมมาตร (Leaf-wise Hierarchy) และคำนวณความลึกของใบไม้แต่ละใบ (`minLeafDepth`, `maxLeafDepth`, `activeLeafDepth`)
  - ไฮไลต์กิ่งที่วิ่งจาก Root ลงไปยัง `leaf_index` ของแถวปัจจุบัน
- **`LightgbmHudPanel.tsx`**:
  - แสดงมาตรวัด **Leaf-Wise Asymmetry & Depth Gauge** และกราฟ **Exact TreeSHAP Waterfall** (ใช้คอมโพเนนต์ร่วมกับ Phase 33)
  - แปลเงื่อนไขโหนด Categorical (`decision_type === "=="`, `threshold: "0||2"`) ให้เป็นชื่อหมวดหมู่ที่อ่านเข้าใจง่าย

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/LightgbmHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { LightgbmHudPanel } from "../LightgbmHudPanel";
import { traceLeafWisePathToLeaf } from "../../../stage3d/actors/LeafWiseTreeVisualizer";

const mockTreeStructure = {
  split_index: 0,
  split_feature: 1,
  threshold: 25.0,
  decision_type: "<=",
  left_child: {
    leaf_index: 0,
    leaf_value: -0.45,
  },
  right_child: {
    split_index: 1,
    split_feature: 3,
    threshold: "0||2",
    decision_type: "==",
    left_child: {
      leaf_index: 1,
      leaf_value: 0.82,
    },
    right_child: {
      leaf_index: 2,
      leaf_value: 0.12,
    },
  },
};

const mockLgbFrame = {
  frame_index: 88,
  prediction: 0.91,
  base_value: -1.1,
  feature_contributions: [-0.15, 1.95, 0.0, 1.62],
  telemetry: {
    model_family: "lightgbm" as const,
    n_iterations: 150,
    sampled_iteration_indices: [0, 10, 45, 95, 149],
    staged_trajectory: [0.35, 0.61, 0.79, 0.88, 0.91],
    leaf_indices_sample: [1, 14, 0, 22, 18],
  },
};

describe("Phase 34 — LightgbmHudPanel & 3D LeafWiseTreeVisualizer", () => {
  it("renders Exact TreeSHAP Waterfall and Active Leaf Depth telemetry", () => {
    render(
      <LightgbmHudPanel
        frame={mockLgbFrame}
        featureNames={["Age", "Income", "Debt", "City"]}
        treeStructure={mockTreeStructure}
        selectedTreeIdx={0}
        onSelectTreeIdx={vi.fn()}
      />
    );

    expect(screen.getByText(/-1\.1/)).toBeInTheDocument();
    expect(screen.getByText(/\+1\.95/)).toBeInTheDocument();
    expect(screen.getByText(/Leaf #1/i)).toBeInTheDocument();
  });

  it("traces asymmetric leaf-wise path from root to target leaf_index and parses categorical '||' splits", () => {
    const pathToLeaf0 = traceLeafWisePathToLeaf(mockTreeStructure, 0);
    const pathToLeaf1 = traceLeafWisePathToLeaf(mockTreeStructure, 1);

    // Leaf #0 is shallow (depth 1), while Leaf #1 is deeper (depth 2)
    expect(pathToLeaf0.depth).toBe(1);
    expect(pathToLeaf1.depth).toBe(2);
    expect(pathToLeaf1.steps[1].isCategorical).toBe(true);
    expect(pathToLeaf1.steps[1].categoricalIndices).toEqual([0, 2]);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, `dump_model(num_iteration=100)` & `num_threads=2`)

1. **การจำกัดขนาด `booster.dump_model(num_iteration=100)`**:
   - บนโมเดลที่มี $1,500$ ต้น การเรียก `extract_model_dump_summary(max_trees=100)` ต้องส่งพารามิเตอร์ `num_iteration=100` เข้าไปในระดับ C++ ของ LightGBM ทำให้คืนต้นไม้กลับมาไม่เกิน 100 ต้น ใช้ RAM ไม่เกิน **$15 \text{ MB}$**
2. **การตรวจสอบการใช้ `num_threads=2` ในทุกคำสั่ง `booster.predict`**:
   - ยืนยันว่าทุกการเรียก `booster.predict` ใน `trace_internal_flow` ส่ง `num_threads=2` เสมอเพื่อป้องกัน OpenMP Thread Contention

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **Multiclass `pred_contrib=True` คืนค่าเป็น 2D Flattened Array** | เทรน `LGBMClassifier` บน 4 คลาส แล้วรัน `trace_internal_flow` | โค้ดทำ `.reshape(n_samples, 4, n_features + 1)` และได้ผลรวมตรงกับ `raw_score[:, c]` ทุกคลาส |
| 2 | **พารามิเตอร์ชื่อ `pred_contrib=True` (เอกพจน์)** | ตรวจสอบการเรียก `booster.predict` ใน `LightGBMAdapter` | ใช้ `pred_contrib=True` เสมอ ไม่เกิด Warning เรื่องพารามิเตอร์ผิดชื่อ |
| 3 | **เมื่อมีเพียง 1 ต้นไม้ (`n_trees == 1`) `pred_leaf=True` คืน 1D Array** | เทรน `LGBMRegressor(n_estimators=1)` | `raw_leaves.ndim == 1` ถูกแปลงด้วย `.reshape(n_samples, -1)` เป็น 2D `(n_samples, 1)` เสมอ |
| 4 | **Categorical Splits ใน `dump_model()` ใช้สตริง `"0||3||5"`** | เทรนโมเดลที่มี Categorical Feature (`decision_type == "=="`) | `traceLeafWisePathToLeaf` แยกสตริงด้วย `.split("||")` โดยไม่เกิด `ValueError: could not convert string to float` |
| 5 | **อินพุตเป็น Sparse Matrix ทำให้ `pred_contrib` คืน `csr_matrix`** | จำลองการส่ง `scipy.sparse.csr_matrix` เข้า `trace_internal_flow` | แปลงเป็น Dense `np.float32` ด้วย `hasattr(raw_contrib, "toarray")` อย่างปลอดภัย |
| 6 | **ชื่อฟีเจอร์ใน Dataset มีเว้นวรรค เช่น `"sepal length (cm)"`** | เทรนด้วยชื่อฟีเจอร์ที่มีช่องว่าง (ซึ่ง LightGBM แปลงเป็น `"sepal_length_(cm)"`) | การส่ง `np.ascontiguousarray(X_chunk, dtype=np.float32)` อ้างอิงตามลำดับคอลัมน์ได้ถูกต้องโดยไม่พังเพราะชื่อฟีเจอร์ไม่ตรง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Exact C++ TreeSHAP Parity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - raw_score)) <= 1e-4` ทั้ง Binary, Multiclass (หลัง Reshape 3D), และ Regression
- [ ] **Throughput SLA**: `trace_internal_flow` สำหรับ $2,000$ แถว (100 Iterations) เสร็จภายใน **$\le 75 \text{ ms}$**
- [ ] **Dump Memory SLA**: `extract_model_dump_summary(max_trees=100)` ใช้หน่วยความจำไม่เกิน **$20 \text{ MB}$**
- [ ] **Format Compatibility SLA**: โหลดโมเดลจาก `.pkl`, `.joblib`, `.txt`, และ `.model` ได้ครบทุกนามสกุล
- [ ] **Coverage Gate**: Line Coverage ของ `lightgbm_adapter.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
