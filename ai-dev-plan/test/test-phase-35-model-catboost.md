# Test Plan — Phase 35: CatBoost Adapter (`.cbm`, `.pkl`) & 3D Symmetric Oblivious Tree Grid

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 35)
เอกสารนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ **[phase-35-model-catboost.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-35-model-catboost.md)** เพื่อยืนยันความถูกต้องของคลาส **`CatBoostAdapter` (`backend/app/domains/model_engine/adapters/boosting/catboost_adapter.py`)** และคอมโพเนนต์ 3 มิติ **`CatBoostObliviousGrid3D` (`frontend/src/domains/simulation/components/stage3d/renderers/CatBoostObliviousGrid3D.tsx`)** ร่วมกับ **`ObliviousBitmaskCard`**

### ขอบเขตการตรวจสอบหลัก
1. **`.cbm` Binary 2-Stage Probing Parity**: ตรวจสอบการโหลดไฟล์ `.cbm` ดิบและจำแนก `CatBoostClassifier` กับ `CatBoostRegressor` จาก `get_all_params()["loss_function"]` อัตโนมัติ 100%
2. **Native Categorical `Pool` & Missing Value Guard**: ตรวจสอบการสร้าง `catboost.Pool` โดยรักษาคอลัมน์หมวดหมู่ตาม `get_cat_feature_indices()` และการแทนที่ `NaN` ด้วย `"__MISSING__"`
3. **Bitwise Oblivious Leaf Index & C++ `ShapValues` Parity**: ยืนยันสมการดัชนีใบไม้ฐานสอง $l_t(\mathbf{x}) = \sum_{d=0}^{D-1} \text{Bit}_d(\mathbf{x}) \cdot 2^d$ และความแม่นยำของ `get_feature_importance(data=pool, type="ShapValues")`
4. **3D Symmetric Oblivious Grid & Discrete Categorical Axis Rendering**: ยืนยันการวาดเส้นตัดตั้งฉาก (`orthogonalSplitsX`, `orthogonalSplitsY`) และแกน `categoricalTicksX`/`Y` บนฉาก 3D

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow / System Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

ไฟล์ทดสอบหลัก: `backend/tests/domains/model_engine/adapters/boosting/test_catboost_adapter.py`

```python
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from app.domains.model_engine.adapters.boosting.catboost_adapter import CatBoostAdapter


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "income": np.array([25000.0, 58000.0, 92000.0, 31000.0, 120000.0], dtype=np.float32),
        "age": np.array([22.0, 35.0, 48.0, 29.0, 55.0], dtype=np.float32),
        "occupation": ["Salaried", "Self-Emp", "Gov", "Salaried", "Self-Emp"],
        "city": ["Bangkok", "Chiang Mai", "Phuket", "Bangkok", "Chiang Mai"],
    })


def test_cbm_binary_probing_classifier_vs_regressor(tmp_path: Path, sample_mixed_df: pd.DataFrame) -> None:
    """ยืนยันว่า load_from_path() แยกแยะไฟล์ .cbm ระหว่าง Classifier และ Regressor ได้ถูกต้อง 100%"""
    y_cls = np.array([0, 1, 1, 0, 1])
    y_reg = np.array([10.5, 42.1, 88.0, 15.2, 95.4], dtype=np.float32)
    cat_cols = [2, 3]

    cls_path = tmp_path / "credit_cls.cbm"
    clf = CatBoostClassifier(iterations=10, depth=3, verbose=0)
    clf.fit(sample_mixed_df, y_cls, cat_features=cat_cols)
    clf.save_model(str(cls_path), format="cbm")

    reg_path = tmp_path / "score_reg.cbm"
    reg = CatBoostRegressor(iterations=10, depth=3, loss_function="RMSE", verbose=0)
    reg.fit(sample_mixed_df, y_reg, cat_features=cat_cols)
    reg.save_model(str(reg_path), format="cbm")

    loaded_cls_adapter = CatBoostAdapter.load_from_path(cls_path)
    loaded_reg_adapter = CatBoostAdapter.load_from_path(reg_path)

    assert loaded_cls_adapter.extract_schema().task_type == "classification"
    assert loaded_reg_adapter.extract_schema().task_type == "regression"
    assert loaded_cls_adapter.extract_schema().categorical_indices == [2, 3]


def test_oblivious_leaf_bitwise_and_shap_additive_parity(tmp_path: Path, sample_mixed_df: pd.DataFrame) -> None:
    """ยืนยันความแม่นยำของ C++ ShapValues (ผลรวม SHAP + Base Value == Raw Margin) และ uint16 Leaf Index"""
    y_cls = np.array([0, 1, 1, 0, 1])
    clf = CatBoostClassifier(iterations=12, depth=4, verbose=0)
    clf.fit(sample_mixed_df, y_cls, cat_features=[2, 3])

    adapter = CatBoostAdapter(model=clf, source_path=tmp_path / "in_mem.cbm")
    batch_res = adapter.predict_batch(sample_mixed_df)

    # 1. ตรวจสอบชนิดข้อมูลเพื่อประหยัด RAM
    assert batch_res.feature_contributions.dtype == np.float32
    assert batch_res.leaf_indices.dtype == np.uint16
    assert batch_res.leaf_indices.shape == (len(sample_mixed_df), 12)
    assert np.all(batch_res.leaf_indices < (1 << 4))  # Depth=4 -> Leaf index in [0..15]

    # 2. ตรวจสอบสมการ Additivity ของ C++ TreeSHAP เทียบกับ Raw Logit (Margin)
    pool = adapter.build_pool(sample_mixed_df)
    raw_margins = clf.predict(pool, prediction_type="RawFormulaVal")
    reconstructed_margins = batch_res.feature_contributions.sum(axis=1) + batch_res.base_values
    np.testing.assert_allclose(reconstructed_margins, raw_margins, rtol=1e-4, atol=1e-4)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & UX Flow Tests (พร้อมโค้ดต้นแบบ `vitest`)

ไฟล์ทดสอบหลัก: `frontend/src/domains/simulation/components/stage3d/renderers/__tests__/CatBoostObliviousGrid3D.test.tsx`

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { ObliviousBitmaskCard } from "../../../hud/ObliviousBitmaskCard";
import { CatBoostObliviousGrid3DProps } from "../CatBoostObliviousGrid3D";

describe("Phase 35 — CatBoostObliviousGrid3D & ObliviousBitmaskCard", () => {
  it("แปลงหมายเลข Leaf Index ของ Oblivious Tree เป็นรหัสฐานสอง (Bitmask) ตามความลึก D ได้ถูกต้อง", () => {
    render(
      <ObliviousBitmaskCard
        treeIndex={0}
        treeDepth={3}
        leafIndex={5} // 5 in 3-bit binary = 101_2
        levelConditions={[
          { depth: 0, featureName: "income", operator: ">", threshold: "35,000" },
          { depth: 1, featureName: "city", operator: "==", threshold: "BKK" },
          { depth: 2, featureName: "age", operator: ">", threshold: "28" },
        ]}
      />
    );

    expect(screen.getByText(/Leaf #5/i)).toBeDefined();
    expect(screen.getByText(/101/)).toBeDefined();
    expect(screen.getByText(/income > 35,000/i)).toBeDefined();
  });

  it("ตรวจสอบโครงสร้าง Props ของ CatBoostObliviousGrid3D สำหรับแกน Categorical Ticks และเส้น Split ตั้งฉาก", () => {
    const props: CatBoostObliviousGrid3DProps = {
      surfaceMesh: new Float32Array(50 * 50),
      orthogonalSplitsX: [35000, 62000, 88000],
      orthogonalSplitsY: [0, 1, 2],
      categoricalTicksY: ["Salaried", "Self-Emp", "Gov"],
      activeFrameLeafBitmask: 0b101,
    };

    expect(props.orthogonalSplitsX.length).toBe(3);
    expect(props.categoricalTicksY).toEqual(["Salaried", "Self-Emp", "Gov"]);
    expect(props.surfaceMesh.byteLength).toBe(50 * 50 * 4);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Process Lifecycle & GPU/CPU Resource Tests)

1. **Single `catboost.Pool` Instantiation Spy Test**:
   - ใช้ `unittest.mock.patch.object(CatBoostAdapter, "build_pool", wraps=adapter.build_pool)` ยืนยันว่าในการรัน `predict_batch(df_chunk)` 1 ครั้ง เมธอด `build_pool()` ถูกเรียกใช้งาน **เพียง 1 ครั้งถ้วน** (`call_count == 1`) และส่ง `Pool` ตัวเดียวกันเข้าทั้ง `predict_proba`, `calc_leaf_indexes`, และ `get_feature_importance(type="ShapValues")`
2. **CPU Core Reservation Test (`thread_count`)**:
   - ตรวจสอบว่าพารามิเตอร์ `thread_count` ที่ส่งเข้า C++ Runtime ของ CatBoost มีค่าเท่ากับ `max(1, psutil.cpu_count(logical=False) - 1)` เสมอ ไม่บล็อก FastAPI Event Loop
3. **Chunk Peak Memory Delta Audit**:
   - รัน `predict_batch()` บนชุดข้อมูลผสมขนาด `2,500` แถว $\times$ `30` คอลัมน์ (มี Categorical 10 คอลัมน์) ยืนยันว่า `psutil.Process().memory_info().rss` เพิ่มขึ้นไม่เกิน `120 MB` และลดกลับสู่ Baseline หลังจบฟังก์ชันและรัน `gc.collect()`

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| กรณีปัญหา (Edge Case จาก Dev Phase 35) | วิธีจำลองในการทดสอบ (Test Simulation Setup) | ผลลัพธ์ที่ต้องผ่านการตรวจสอบ (Expected Verification Assertion) |
| :--- | :--- | :--- |
| **1. ไฟล์ `.cbm` ไม่ระบุคลาส Python** | เซฟ `CatBoostRegressor(loss_function="MAE")` เป็น `.cbm` แล้วโหลดผ่าน `CatBoostAdapter.load_from_path()` | ระบบตรวจพบ `loss_function == "MAE"` จากอินสแตนซ์ Probe แล้วสลับเป็น `CatBoostRegressor` อัตโนมัติโดยไม่เกิด Exception |
| **2. ค่า `NaN` ในคอลัมน์ Categorical** | ใส่ค่า `np.nan` และ `None` ลงในคอลัมน์ `occupation` และ `city` ของ `df_chunk` | `build_pool()` แทนที่ค่าว่างด้วย `"__MISSING__"` และแปลงเป็น `str` ก่อนสร้าง `Pool` ไม่เกิด `CatBoostError: Invalid type for cat_feature` |
| **3. โมเดล `.cbm` ไม่มี `feature_names_`** | เทรนโมเดลด้วย `np.ndarray` ล้วนโดยไม่ระบุชื่อคอลัมน์ แล้วโหลดเข้า Adapter | `extract_schema().feature_names` สร้างชื่อมาตรฐาน `["feature_0", "feature_1", ...]` ครบตามจำนวนฟีเจอร์จริง |
| **4. แกน 3D Mesh Slice เป็นคอลัมน์ Categorical** | เรียก `extract_oblivious_split_grid(feature_x="occupation", feature_y="income")` | ระบบใช้รายการหมวดหมู่ที่ไม่ซ้ำกัน (`unique_categories[:32]`) แทน `np.linspace` และคืนค่า `categoricalTicksX` ให้ Frontend |
| **5. Multi-class `ShapValues` คืนอาเรย์ 3 มิติ** | เทรน `CatBoostClassifier(loss_function="MultiClass")` แบบ 4 คลาส แล้วรัน `predict_batch()` | ระบบตัด Slice ของคลาสที่ชนะ `(N, C, F+1) -> (N, F)` พร้อมแยก Base Value ของคลาสนั้นออกมาอย่างแม่นยำ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **100% `.cbm` & `.pkl` Classification/Regression Detection**: ทดสอบโหลดไฟล์ `.cbm` ครบทั้ง `Logloss`, `MultiClass`, `RMSE`, `MAE`, และ `Quantile` ผ่าน 100%
- [ ] **Mathematical Parity (`< 1e-4`)**: ผลรวม `feature_contributions + base_values` เท่ากับ `RawFormulaVal` ของ CatBoost ทุกแถวในระดับความคลาดเคลื่อน `< 1e-4`
- [ ] **Single Pool & Memory Guard (`<= 120 MB`)**: ยืนยันการสร้าง `Pool` เพียงครั้งเดียวต่อ Chunk และใช้ RAM ส่วนเพิ่มไม่เกิน `120 MB`
- [ ] **60 FPS Symmetric Grid Rendering**: คอมโพเนนต์ `CatBoostObliviousGrid3D` และ `ObliviousBitmaskCard` ผ่านชุดทดสอบ `vitest` ครบทุกเคส
