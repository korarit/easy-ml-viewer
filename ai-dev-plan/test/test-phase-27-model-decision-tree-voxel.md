# Test Plan — Phase 27: Decision Tree Adapter, `decision_path` Tracer & 3D Stepped Voxel Terraces

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-27-model-decision-tree-voxel.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-27-model-decision-tree-voxel.md) เพื่อตรวจสอบความถูกต้องของการแกะโครงสร้าง `model.tree_`, การดึงเส้นทางการตัดสินใจด้วย CSR Sparse Matrix (`decision_path`), การคำนวณค่าอิทธิพลรายโหนดแบบ Exact Tree Path Feature Contribution ($\Delta_{u \to v}$), และการเรนเดอร์ภูมิประเทศ **3D Stepped Voxel Terraces** ร่วมกับเส้นแสง **Active Tree Branch Glow**

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `DecisionTreeAdapter` (`backend/src/domains/Precomputation/adapters/trees/decision_tree.py`)
  - เมธอดหลัก: `_compute_node_expectations()`, `extract_tree_topology(max_export_nodes=512)`, `trace_internal_flow(X_chunk)`
  - คลาสโมเดลที่ทดสอบ: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreeRegressor` (ทั้งแบบเดี่ยวและใน `Pipeline`)
- **Frontend 3D & HUD Components**:
  - `SteppedVoxelTerrain.tsx` (`frontend/src/domains/stage3d/actors/SteppedVoxelTerrain.tsx`)
  - `SplitBoundaryLines.tsx` (`frontend/src/domains/stage3d/actors/SplitBoundaryLines.tsx`)
  - `ActiveTreeBranchOverlay.tsx` (`frontend/src/domains/stage3d/actors/ActiveTreeBranchOverlay.tsx`)
  - `DecisionTreeHudPanel.tsx` (`frontend/src/domains/hud/components/DecisionTreeHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Local Accuracy Parity (ความแม่นยำของการแยกองค์ประกอบรายโหนด)**:
   สำหรับทุกแถวข้อมูล $\mathbf{x}_i$ ผลรวมของ `base_value` ($E_{\text{root}}$) บวกกับผลรวมของ `feature_contributions[i, :]` ต้องเท่ากับค่าความน่าจะเป็นหรือค่าทำนายของโหนดใบไม้ $E_{\text{leaf}}$ ทุกประการ:
   $$\left| E_{\text{root}} + \sum_{j=1}^{d} \phi_{i, j} - \hat{y}_{\text{target}}(\mathbf{x}_i) \right| \le 10^{-5}$$
2. **CSR Sparse Path & `apply()` Consistency**:
   โหนดสุดท้ายใน `active_paths[i][-1]["next_node_id"]` ต้องตรงกับหมายเลขใบไม้ที่ได้จาก `model.apply(X_chunk)[i]` เสมอ และทุกขั้นตอนใน `active_paths[i]` ต้องมี `delta == round(E_v - E_u, 5)`

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/trees/test_decision_tree_adapter.py`)
```python
from __future__ import annotations
from unittest.mock import patch
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from domains.Precomputation.adapters.trees.decision_tree import DecisionTreeAdapter


@pytest.mark.parametrize("tree_cls", [DecisionTreeClassifier, ExtraTreeClassifier])
def test_decision_tree_classifier_local_accuracy_and_leaf_parity(tree_cls):
    """พิสูจน์ว่า base_value + sum(feature_contributions) == predict_proba[:, 1] และใบไม้ตรงกับ apply()."""
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4, random_state=42)
    X_f32 = X.astype(np.float32)
    model = tree_cls(max_depth=5, random_state=42).fit(X_f32, y)

    adapter = DecisionTreeAdapter(model, feature_names=[f"f{i}" for i in range(6)], target_class_idx=1)
    batch = adapter.trace_internal_flow(X_f32)

    expected_probs = model.predict_proba(X_f32)[:, 1]
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_probs, atol=1e-5)

    expected_leaves = model.apply(X_f32)
    assert batch.extra_telemetry["leaf_ids"] == expected_leaves.astype(np.int32).tolist()
    for i, path in enumerate(batch.extra_telemetry["active_paths"]):
        assert path[-1]["next_node_id"] == int(expected_leaves[i])
        assert path[0]["node_id"] == 0


@pytest.mark.parametrize("reg_cls", [DecisionTreeRegressor, ExtraTreeRegressor])
def test_decision_tree_regressor_exact_path_contributions(reg_cls):
    """พิสูจน์ความถูกต้องของสมการ Delta รายโหนดในงาน Regression."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.5, random_state=7)
    X_f32 = X.astype(np.float32)
    model = reg_cls(max_depth=6, random_state=7).fit(X_f32, y)

    adapter = DecisionTreeAdapter(model, feature_names=[f"f{i}" for i in range(5)])
    batch = adapter.trace_internal_flow(X_f32)

    expected_preds = model.predict(X_f32)
    reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed, expected_preds, atol=1e-4)


def test_multiclass_decision_tree_target_class_indexing():
    """พิสูจน์การคำนวณความน่าจะเป็นและ Contributions สำหรับ Multiclass (C=4 คลาส)."""
    X, y = make_classification(
        n_samples=150, n_features=6, n_informative=4, n_classes=4, n_clusters_per_class=1, random_state=99
    )
    X_f32 = X.astype(np.float32)
    model = DecisionTreeClassifier(max_depth=4, random_state=99).fit(X_f32, y)

    for target_c in range(4):
        adapter = DecisionTreeAdapter(model, feature_names=[f"f{i}" for i in range(6)], target_class_idx=target_c)
        batch = adapter.trace_internal_flow(X_f32)
        expected_c_prob = model.predict_proba(X_f32)[:, target_c]
        reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
        np.testing.assert_allclose(reconstructed, expected_c_prob, atol=1e-5)


def test_csr_decision_path_never_calls_toarray():
    """บังคับว่าห้ามเรียก .toarray() บน CSR matrix ของ decision_path โดยเด็ดขาด."""
    X, y = make_classification(n_samples=32, n_features=4, random_state=1)
    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X, y)
    adapter = DecisionTreeAdapter(model, feature_names=["a", "b", "c", "d"])

    with patch.object(csr_matrix, "toarray", side_effect=AssertionError("Forbidden .toarray() call on CSR!")):
        batch = adapter.trace_internal_flow(X.astype(np.float32))
        assert batch.feature_contributions.shape == (32, 4)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`SteppedVoxelTerrain.tsx`**: สร้าง `THREE.InstancedMesh` สำหรับตาราง $80 \times 80 = 6,400$ บล็อก โดยใช้ Flat Shading (ไม่มีการเกลี่ยความสูงแบบ Smooth) เพื่อรักษาขอบหน้าผาตั้งฉากตาม `threshold`
- **`ActiveTreeBranchOverlay.tsx`**: เรนเดอร์เฉพาะโหนดที่อยู่ใน `telemetry.active_path` ของเฟรมปัจจุบัน พร้อมไฮไลต์กิ่งที่ `went_left === true` หรือ `false`
- **`DecisionTreeHudPanel.tsx` & What-If Cliff Jump**: เมื่อผู้ใช้ลากค่าฟีเจอร์แกน $X$ ข้ามค่า `threshold` ของโหนดปัจจุบัน ค่าทำนายต้องกระโดดเปลี่ยนขั้นทันที และ Breadcrumb ต้องเปลี่ยน `leaf_id` ทันที

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/DecisionTreeHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DecisionTreeHudPanel } from "../DecisionTreeHudPanel";
import { computeSteppedVoxelHeightAt } from "../../../stage3d/actors/SteppedVoxelTerrain";

const mockFramePayload = {
  frame_index: 42,
  prediction: 0.45,
  base_value: 0.5,
  feature_contributions: [-0.18, 0.13, 0.0, 0.0],
  telemetry: {
    model_family: "decision_tree" as const,
    leaf_id: 15,
    leaf_impurity: 0.24,
    leaf_n_samples: 38,
    active_path: [
      { node_id: 0, next_node_id: 1, feature_idx: 0, threshold: 52.5, went_left: true, delta: -0.18 },
      { node_id: 1, next_node_id: 15, feature_idx: 1, threshold: 41.0, went_left: false, delta: 0.13 },
    ],
  },
};

describe("Phase 27 — DecisionTreeHudPanel & 3D Stepped Voxel Terraces", () => {
  it("renders Active Path Rule Breadcrumb with exact thresholds, directions, and leaf telemetry", () => {
    render(
      <DecisionTreeHudPanel
        frame={mockFramePayload}
        featureNames={["Income", "Age", "Debt", "Score"]}
        maxTreeDepth={6}
        selectedDepth={6}
        onDepthChange={vi.fn()}
      />
    );

    expect(screen.getByText(/Income <= 52\.5/i)).toBeInTheDocument();
    expect(screen.getByText(/Age > 41/i)).toBeInTheDocument();
    expect(screen.getByText(/Leaf #15/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.24/)).toBeInTheDocument();
  });

  it("triggers Depth Clipping Slider callback and clips displayed path steps to selectedDepth", () => {
    const onDepthChange = vi.fn();
    render(
      <DecisionTreeHudPanel
        frame={mockFramePayload}
        featureNames={["Income", "Age", "Debt", "Score"]}
        maxTreeDepth={6}
        selectedDepth={1}
        onDepthChange={onDepthChange}
      />
    );

    const slider = screen.getByRole("slider", { name: /depth/i });
    fireEvent.change(slider, { target: { value: "3" } });
    expect(onDepthChange).toHaveBeenCalledWith(3);
  });

  it("simulates What-If Sandbox cliff jump when feature X crosses orthogonal split threshold", () => {
    const gridSplitsX = [52.5];
    const terraceHeights = [0.65, 0.45]; // left of 52.5 -> 0.65, right of 52.5 -> 0.45

    // ขณะลากอยู่ในช่วง <= 52.5 ความสูงขั้นบันไดต้องคงที่ที่ 0.65
    expect(computeSteppedVoxelHeightAt(40.0, gridSplitsX, terraceHeights)).toBeCloseTo(0.65, 5);
    expect(computeSteppedVoxelHeightAt(52.5, gridSplitsX, terraceHeights)).toBeCloseTo(0.65, 5);

    // ทันทีที่ข้ามเส้น 52.5 -> 52.51 ลูกบอลกระโดดลงหน้าผาไปยัง 0.45 ทันที!
    expect(computeSteppedVoxelHeightAt(52.51, gridSplitsX, terraceHeights)).toBeCloseTo(0.45, 5);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, `max_export_nodes=512` Pruning & WebGL Tests)

1. **การทดสอบหน่วยความจำกับต้นไม้ขนาดใหญ่ (`node_count > 20,000`)**:
   - สร้าง `DecisionTreeRegressor(max_depth=None, min_samples_split=2)` บนข้อมูลสุ่ม $15,000$ แถว จนได้โหนดมากกว่า $20,000$ โหนด
   - รัน `tracemalloc` ขณะเรียก `adapter.trace_internal_flow(X_chunk[:2000])` และตรวจสอบว่า Peak Memory Allocation เพิ่มขึ้นไม่เกิน **$35 \text{ MB}$** (พิสูจน์ว่าไม่มีการแปลง CSR Sparse Matrix เป็น Dense Array)
2. **การทดสอบการตัดขนาด Topology (`extract_tree_topology(max_export_nodes=512)`)**:
   - ยืนยันว่าแม้ `model.tree_.node_count` จะมีค่า $25,000$ โหนด แต่ลิสต์ `children_left`, `children_right`, `feature`, `threshold` ที่ส่งออกใน Header มีความยาวสูงสุดไม่เกิน `512` รายการ และ `active_paths` ของแต่ละแถวถูกจำกัดความลึกไม่เกิน `32` ขั้นแรก
3. **การทดสอบประสิทธิภาพ `InstancedMesh` ใน `SteppedVoxelTerrain.tsx`**:
   - ตรวจสอบว่า `renderer.info.render.calls` สำหรับการวาดบล็อกขั้นบันได $80 \times 80 = 6,400$ แท่ง มีค่าเท่ากับ **`1 Draw Call`** และไม่มี Memory Leak ของ `BufferGeometry` เมื่อเปลี่ยนแกนฟีเจอร์ 20 ครั้งติดต่อกัน

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **ต้นไม้ลึกมาก (`max_depth > 25`, โหนดหลักหมื่น)** | เทรน `DecisionTreeRegressor(max_depth=35)` บนข้อมูล 10,000 แถว | `len(topology["feature"]) <= 512` และแต่ละ `active_path` มีความยาว `<= 32` ขั้น |
| 2 | **ต้นไม้มีโหนดเดียว (`node_count == 1`)** | เทรนต้นไม้บนข้อมูลที่ค่า $y$ ทุกแถวเป็นค่าเดียวกันทั้งหมด (`y = np.zeros(50)`) | `trace_internal_flow` คืน `active_paths = [[] * 50]`, `feature_contributions` เป็น `0.0` ทั้งหมดโดยไม่เกิด `IndexError` |
| 3 | **Multiclass ที่ `target_class_idx` เกินจำนวนคลาสจริง** | เทรนต้นไม้ 3 คลาส แต่สร้าง Adapter ด้วย `target_class_idx = 9` | โค้ดใช้ `min(target_class_idx, probs.shape[1] - 1)` เลือกคลาสสุดท้าย (`idx = 2`) อย่างปลอดภัย |
| 4 | **Multi-output Tree (`n_outputs_ > 1`)** | เทรน `DecisionTreeRegressor` ที่ `Y` มี 3 คอลัมน์ (`shape: (100, 3)`) | `_compute_node_expectations()` ดึง `vals[:, 0, 0]` ของ Output แรกได้โดยไม่พังด้วย ValueError |
| 5 | **ข้อมูลมีค่า `NaN` (Scikit-learn >= 1.3 Native Missing Values)** | ส่ง `X_chunk` ที่มี `np.nan` ในฟีเจอร์ที่ใช้แบ่งโหนด Root | ทิศทาง `went_left` ถูกตัดสินใจสอดคล้องกับโหนดลูกจริงใน `csr.indices` และ `reconstructed == expected_pred` 100% |
| 6 | **`feature_importances_` เป็นศูนย์ทั้งหมด** | จำลองต้นไม้ที่ `model.feature_importances_` มีค่า `[0.0, 0.0, 0.0]` | ระบบเลือกแกน $X, Y$ เริ่มต้นจาก 2 ฟีเจอร์ที่มี `np.nanvar(X, axis=0)` สูงสุดแทนโดยไม่เกิดการหารด้วยศูนย์ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Mathematical Parity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - expected_pred)) <= 1e-5` สำหรับทุกคลาส (`DecisionTreeClassifier/Regressor`, `ExtraTreeClassifier/Regressor`)
- [ ] **Latency SLA**: การรัน `trace_internal_flow` สำหรับ `X_chunk` ขนาด $2,000$ แถว $\times 20$ ฟีเจอร์ (ความลึก $\le 15$) ต้องเสร็จภายใน **$\le 35 \text{ ms}$** บน CPU มาตรฐาน
- [ ] **Memory SLA**: Peak RAM ของ Worker ขณะแกะ `decision_path` ต้องไม่เกิน **$50 \text{ MB}$** และห้ามมีการเรียก `.toarray()` บน CSR Matrix
- [ ] **WebGL Frame Rate SLA**: ฉาก 3 มิติพร้อม `SteppedVoxelTerrain` ($6,400$ Voxels) และ `ActiveTreeBranchOverlay` เรนเดอร์ที่ **$\ge 60 \text{ FPS}$** (Draw Calls ของพื้นผิว $= 1$)
- [ ] **Coverage Gate**: Line Coverage ของ `backend/src/domains/Precomputation/adapters/trees/decision_tree.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
