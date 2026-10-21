# Test Plan — Phase 33: XGBoost Adapter (`.pkl`, `.json`, `.ubj`), Tree Dump Parser & 3D Boosted Leaf Flow

---

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)

แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับ [phase-33-model-xgboost.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-33-model-xgboost.md) เพื่อตรวจสอบความถูกต้องของ `XGBoostAdapter` ในการโหลดโมเดลทั้งจาก Scikit-learn Wrapper (`XGBClassifier`, `XGBRegressor`) และไฟล์ Native Booster (`.json`, `.ubj`, `.model`), การใช้ `xgb.DMatrix` เพียงครั้งเดียวต่อ Chunk เพื่อดึงทั้ง **Exact C++ TreeSHAP (`pred_contribs=True, strict_shape=True`)** และ **Leaf Index (`pred_leaf=True, strict_shape=True`)**, การแกะ JSON Tree Dump (`get_dump(dump_format="json", with_stats=True)`), และการแสดงผลลูกศร **3D Exact TreeSHAP Force Vectors** (`TreeShapForceArrows.tsx`)

### ขอบเขตคลาสและคอมโพเนนต์ที่ต้องตรวจสอบ (Target Under Test)
- **Backend Adapter**: `XGBoostAdapter` (`backend/src/domains/Precomputation/adapters/boosting/xgboost_adapter.py`)
  - ค่าคงที่: `MAX_EXPORTED_TREES = 100`, `MAX_TRAJECTORY_POINTS = 50`
  - เมธอดหลัก: `parse_trees_topology(max_trees=100)`, `trace_internal_flow(X_chunk)`
  - คลาสและไฟล์โมเดลที่ทดสอบ: `xgb.XGBClassifier`, `xgb.XGBRegressor`, `xgb.Booster` (ไฟล์ `.pkl`, `.joblib`, `.json`, `.ubj`)
- **Frontend 3D & HUD Components**:
  - `TreeShapForceArrows.tsx` (`frontend/src/domains/stage3d/actors/TreeShapForceArrows.tsx`)
  - `SteppedVoxelTerrain.tsx` (`frontend/src/domains/stage3d/actors/SteppedVoxelTerrain.tsx`)
  - `XgboostHudPanel.tsx` (`frontend/src/domains/hud/components/XgboostHudPanel.tsx`)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Tree/Boosting Internal Flow Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

### 2.1 สมการและเงื่อนไข Parity ที่ต้องพิสูจน์ในชุดทดสอบ
1. **Exact C++ TreeSHAP Additivity (`pred_contribs=True` vs `output_margin=True`)**:
   ผลรวมของ `base_value` (`raw_contribs[:, grp_idx, -1]`) บวกกับผลรวมของ `feature_contributions` (`raw_contribs[:, grp_idx, :-1]`) ต้องเท่ากับค่า Raw Margin ที่ได้จาก `booster.predict(dmat, output_margin=True)` ทุกแถวและทุกคลาส:
   $$\left| \text{Bias}_c + \sum_{j=1}^{d} \phi_{i, c, j} - \hat{y}_{\text{margin}, c}(\mathbf{x}_i) \right| \le 10^{-4}$$
2. **Multiclass 3D Tensor Shape & Portable File Format Parity (`.json` / `.ubj`)**:
   - เมื่อใช้ `strict_shape=True` อาเรย์ `raw_contribs` ต้องมีขนาด 3 มิติ `(n_samples, n_groups, n_features + 1)` เสมอ และ `raw_leaves` ต้องมีขนาด 4 มิติ `(n_samples, n_rounds, n_groups, n_parallel_trees)` เสมอ
   - โมเดลที่เซฟเป็น `.ubj` หรือ `.json` แล้วโหลดกลับมาเป็น Native `xgb.Booster` ต้องให้ค่า `feature_contributions` และ `leaf_indices_sample` ตรงกับ `XGBClassifier` ต้นฉบับทุกประการ

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/unit/adapters/boosting/test_xgboost_adapter.py`)
```python
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, wraps
import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from domains.Precomputation.adapters.boosting.xgboost_adapter import XGBoostAdapter


def test_xgboost_binary_and_ubj_native_booster_treeshap_parity(tmp_path: Path):
    """พิสูจน์ว่า Exact TreeSHAP (pred_contribs=True) บวกได้เท่ากับ output_margin=True และไฟล์ .ubj ให้ผลตรงกับ Wrapper."""
    X, y = make_classification(n_samples=100, n_features=6, n_informative=4, random_state=42)
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    feat_names = [f"feat_{i}" for i in range(6)]

    clf = xgb.XGBClassifier(n_estimators=30, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X_f32, y)

    # 1. ทดสอบผ่าน Scikit-learn Wrapper
    adapter_wrap = XGBoostAdapter(clf, feature_names=feat_names, target_class_idx=1)
    batch_wrap = adapter_wrap.trace_internal_flow(X_f32)

    dmat = xgb.DMatrix(X_f32, feature_names=feat_names)
    expected_margin = clf.get_booster().predict(dmat, output_margin=True)
    reconstructed_margin = batch_wrap.base_value + np.sum(batch_wrap.feature_contributions, axis=1)
    np.testing.assert_allclose(reconstructed_margin, expected_margin, atol=1e-4)

    # 2. บันทึกเป็นไฟล์ .ubj แล้วโหลดเป็น Native Booster โดยตรง
    ubj_path = tmp_path / "model.ubj"
    clf.save_model(ubj_path)
    adapter_ubj = XGBoostAdapter(ubj_path, feature_names=feat_names, target_class_idx=1)
    batch_ubj = adapter_ubj.trace_internal_flow(X_f32)

    np.testing.assert_allclose(batch_ubj.feature_contributions, batch_wrap.feature_contributions, atol=1e-5)
    assert batch_ubj.extra_telemetry["leaf_indices_sample"] == batch_wrap.extra_telemetry["leaf_indices_sample"]


def test_xgboost_multiclass_strict_shape_3d_contribs_and_4d_leaves():
    """พิสูจน์การแยกคลาสของ Multiclass (C=4) จาก pred_contribs (3D) และ pred_leaf (4D)."""
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=4, n_classes=4, n_clusters_per_class=1, random_state=19
    )
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=19).fit(X_f32, y)

    dmat = xgb.DMatrix(X_f32)
    expected_margins = clf.get_booster().predict(dmat, output_margin=True)  # shape: (80, 4)

    for c_idx in range(4):
        adapter = XGBoostAdapter(clf, feature_names=[f"f{i}" for i in range(5)], target_class_idx=c_idx)
        batch = adapter.trace_internal_flow(X_f32)
        reconstructed = batch.base_value + np.sum(batch.feature_contributions, axis=1)
        np.testing.assert_allclose(reconstructed, expected_margins[:, c_idx], atol=1e-4)
        assert np.asarray(batch.extra_telemetry["leaf_indices_sample"]).shape == (80, 20)


def test_single_dmatrix_instantiation_per_chunk_and_json_tree_dump(monkeypatch):
    """บังคับว่า trace_internal_flow ต้องสร้าง xgb.DMatrix เพียง 1 ครั้งต่อ Chunk เท่านั้น."""
    X, y = make_regression(n_samples=50, n_features=4, random_state=3)
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    reg = xgb.XGBRegressor(n_estimators=12, max_depth=3, random_state=3).fit(X_f32, y)
    adapter = XGBoostAdapter(reg, feature_names=["a", "b", "c", "d"])

    orig_dmatrix = xgb.DMatrix
    call_counter = {"count": 0}

    def counting_dmatrix(*args, **kwargs):
        call_counter["count"] += 1
        return orig_dmatrix(*args, **kwargs)

    monkeypatch.setattr(xgb, "DMatrix", counting_dmatrix)
    _ = adapter.trace_internal_flow(X_f32)
    assert call_counter["count"] == 1

    trees = adapter.parse_trees_topology(max_trees=5)
    assert len(trees) == 5
    assert "nodeid" in trees[0]
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, Tree HUD & What-If UX Tests (พร้อมโค้ดต้นแบบ `vitest`)

### 3.1 รายการพฤติกรรม UI/3D ที่ต้องตรวจสอบ
- **`TreeShapForceArrows.tsx`**:
  - วาดลูกศรเวกเตอร์แรงผลัก 3 มิติพุ่งออกจากลูกบอลข้อมูลตามแกน $X$ และแกน $Y$ โดยความยาวและทิศทางแปรผันตาม `feature_contributions[axisX]` และ `feature_contributions[axisY]`
- **`XgboostHudPanel.tsx`**:
  - แสดงกราฟน้ำตก **Exact TreeSHAP Waterfall Chart** ที่เรียงลำดับฟีเจอร์ตามขนาด $|\phi_j|$ จากมากไปน้อย พร้อมแสดง `Bias (Base Margin)` และ `Raw Margin Sum`
  - แสดงเส้นทางโหนดใน JSON Tree Dump ของต้นไม้ที่เลือก พร้อมค่า `gain` และ `cover`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/hud/components/__tests__/XgboostHudPanel.test.tsx`)
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { XgboostHudPanel } from "../XgboostHudPanel";
import { computeTreeShapArrowVectors } from "../../../stage3d/actors/TreeShapForceArrows";

const mockXgbFrame = {
  frame_index: 215,
  prediction: 0.7824,
  base_value: -0.35,
  feature_contributions: [0.68, 0.0, 1.42, 0.0, 0.0, -0.49],
  telemetry: {
    model_family: "xgboost" as const,
    n_boost_rounds: 250,
    sampled_round_indices: [0, 10, 50, 120, 249],
    staged_trajectory: [0.46, 0.58, 0.69, 0.75, 0.7824],
    leaf_indices_sample: [6, 11, 4, 19, 8],
  },
};

describe("Phase 33 — XgboostHudPanel & 3D TreeShapForceArrows", () => {
  it("renders Exact C++ TreeSHAP Waterfall with Bias, top features, and Raw Margin Sum", () => {
    render(
      <XgboostHudPanel
        frame={mockXgbFrame}
        featureNames={["Age", "Gender", "Income", "City", "Tenure", "Debt"]}
        selectedTreeIdx={0}
        onSelectTreeIdx={vi.fn()}
      />
    );

    expect(screen.getByText(/-0\.35/)).toBeInTheDocument();
    expect(screen.getByText(/\+1\.42/)).toBeInTheDocument();
    expect(screen.getByText(/\+0\.68/)).toBeInTheDocument();
    expect(screen.getByText(/-0\.49/)).toBeInTheDocument();
    // Raw Margin Sum = -0.35 + 0.68 + 1.42 - 0.49 = +1.26
    expect(screen.getByText(/\+1\.26/)).toBeInTheDocument();
  });

  it("computes 3D force arrow vectors for selected X and Y axes from Exact TreeSHAP values", () => {
    const vectors = computeTreeShapArrowVectors(
      mockXgbFrame.feature_contributions,
      2, // X axis = Income (+1.42)
      5  // Y axis = Debt (-0.49)
    );

    expect(vectors.arrowX.length).toBeGreaterThan(vectors.arrowY.length);
    expect(vectors.arrowX.sign).toBe(1);
    expect(vectors.arrowY.sign).toBe(-1);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Single `DMatrix` Reuse, `nthread=2` & WebGL Tests)

1. **การยืนยันการสร้าง `xgb.DMatrix` เพียงครั้งเดียวต่อ Chunk**:
   - ใช้ `monkeypatch` นับจำนวนครั้งที่ `xgb.DMatrix` ถูกเรียกใน `trace_internal_flow` ต้องเท่ากับ **`1 ครั้ง`** เท่านั้นต่อ Chunk
2. **การตรวจสอบการล็อก `nthread = 2`**:
   - ยืนยันว่า `json.loads(adapter.booster.save_config())["learner"]["generic_param"]["nthread"] == "2"` เพื่อป้องกันการแย่งคอร์ CPU ใน `ComputePool`
3. **การจำกัดจำนวนต้นไม้ใน `parse_trees_topology(max_trees=100)`**:
   - เมื่อโมเดลมี $2,000$ ต้น การเรียก `parse_trees_topology` ต้องแปลง JSON เฉพาะ 100 ต้นแรกเท่านั้น ใช้เวลาไม่เกิน $25 \text{ ms}$

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| # | กรณีศึกษา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Setup) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :-: | :--- | :--- | :--- |
| 1 | **Multiclass `pred_contribs` Shape (2D vs 3D)** | ทดสอบทั้ง Binary (`n_classes=2`) และ Multiclass (`n_classes=5`) | การเปิด `strict_shape=True` ทำให้จัดการทั้งสองกรณีได้โดยไม่มี `IndexError` หรือ `ValueError` |
| 2 | **Native Booster ไม่มี `feature_names` (`booster.feature_names is None`)** | เทรนด้วย NumPy Array ล้วนแล้วเซฟเป็น `.ubj` โหลดกลับโดยที่ `booster.feature_names` เป็น `None` | สร้าง `DMatrix` ได้โดยไม่เกิด `feature_names mismatch` และแมป `"f0" -> feature_names[0]` ได้ถูกต้อง |
| 3 | **Early Stopping (`best_iteration`) ในโมเดลที่เซฟไว้** | เทรนพร้อม `early_stopping_rounds=3` | ระบบอ่าน `best_iteration` และทำนายภายในช่วง `iteration_range` ที่ถูกต้อง |
| 4 | **`gblinear` Booster แทนที่จะเป็น `gbtree`** | เทรน `xgb.XGBRegressor(booster="gblinear")` | ระบบดึง `pred_contribs=True` ได้ตามปกติและข้ามการเรียก `pred_leaf=True` อย่างปลอดภัย |
| 5 | **Categorical Splits ใน JSON Dump (`"split_condition": [0, 2]`)** | เทรนด้วย `enable_categorical=True` บนคอลัมน์ `category` | JSON Tree Parser รองรับ `split_condition` ที่เป็น `list` โดยไม่พยายามแปลงเป็น `float` |
| 6 | **โมเดลชนิด `DART` (`booster="dart"`)** | เทรน `xgb.XGBClassifier(booster="dart", n_estimators=30)` | `pred_contribs=True` และ `pred_leaf=True` ทำงานได้ครบถ้วนภายใน SLA |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)

- [ ] **Exact C++ TreeSHAP Parity SLA**: `np.max(np.abs(base_value + contribs.sum(axis=1) - output_margin)) <= 1e-4` ทั้ง Binary, Multiclass, และ Regression
- [ ] **Single DMatrix SLA**: สร้าง `xgb.DMatrix` เพียง **`1 ครั้งต่อ Chunk`** เท่านั้น
- [ ] **Throughput SLA**: `trace_internal_flow` สำหรับ $2,000$ แถว (100 Boost Rounds) เสร็จภายใน **$\le 90 \text{ ms}$**
- [ ] **Format Compatibility SLA**: โหลดไฟล์ `.pkl`, `.joblib`, `.json`, `.ubj`, และ `.model` ได้ครบทุกนามสกุล
- [ ] **Coverage Gate**: Line Coverage ของ `xgboost_adapter.py` ต้อง **$\ge 95\%$** และผ่าน Edge Cases ทั้ง 6 ข้อครบถ้วน
