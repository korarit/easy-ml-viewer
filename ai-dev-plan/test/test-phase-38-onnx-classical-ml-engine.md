# Test Plan — Phase 38: ONNX Runtime Classical ML Inspector (`TreeEnsemble`, `Linear`, `SVM` Operators)

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 38)
เอกสารนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ **[phase-38-onnx-classical-ml-engine.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-38-onnx-classical-ml-engine.md)** เพื่อยืนยันความถูกต้องของคลาส **`ONNXClassicalAdapter` (`backend/app/domains/model_engine/adapters/neural_onnx/onnx_classical.py`)** และคอมโพเนนต์ 3 มิติ **`ONNXClassicalStageRouter`** ร่วมกับ **`ONNXGraphInspectorCard`**

### ขอบเขตการตรวจสอบหลัก
1. **`ai.onnx.ml` Operator Decompilation Parity**: ตรวจสอบการสแกนโหนด `TreeEnsembleClassifier`/`Regressor`, `LinearClassifier`/`Regressor`, และ `SVMClassifier`/`Regressor` พร้อมการสกัดแอตทริบิวต์ (`nodes_featureids`, `nodes_values`, `coefficients`, `support_vectors`)
2. **In-Memory `ZipMap` Bypass & Fallback Unpacker**: ยืนยันว่าการตัดโหนด `ZipMap` ในหน่วยความจำหรือการแกะ `seq(map(int64/string, float))` คืนค่า `np.ndarray` ขนาด `(N, C)` (`float32`) ที่ตรงกับโมเดลต้นฉบับก่อนแปลงเป็น ONNX
3. **Single 2D Tensor vs Multi-Input Named Tensors**: ยืนยันว่า `_build_feed_dict()` จัดการได้ทั้งอินพุตแบบก้อนเดียว `float_input: [None, F]` และแบบแยกชื่อคอลัมน์หลายอินพุต
4. **Auto-Morphing 3D Stage Routing**: ยืนยันว่า `ONNXClassicalStageRouter` เลือกเรนเดอร์ฉาก 3D (`LinearHyperplane3D`, `DecisionTreeVoxel3D`, `SVMKernelSurface3D`) ได้ตรงตามชนิดของ Operator ในไฟล์ `.onnx`

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow / System Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

ไฟล์ทดสอบหลัก: `backend/tests/domains/model_engine/adapters/neural_onnx/test_onnx_classical.py`

```python
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from app.domains.model_engine.adapters.neural_onnx.onnx_classical import ONNXClassicalAdapter


@pytest.fixture
def sample_xy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=["feature_0", "feature_1", "feature_2", "feature_3"])
    return df, y


def test_onnx_tree_ensemble_decompilation_and_zipmap_bypass(
    tmp_path: Path, sample_xy: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """ยืนยันการแกะ TreeEnsembleClassifier, การตัด ZipMap, และความแม่นยำเทียบกับ RandomForest ต้นฉบับ"""
    df, y = sample_xy
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=123)
    rf.fit(df.to_numpy(), y)

    onnx_proto = convert_sklearn(rf, initial_types=[("float_input", FloatTensorType([None, 4]))])
    onnx_path = tmp_path / "rf_zipmap.onnx"
    onnx.save_model(onnx_proto, str(onnx_path))

    adapter = ONNXClassicalAdapter(model_path=onnx_path, onnx_proto=onnx_proto)
    meta = adapter._inspect_ml_operators()
    assert meta.detected_operator_family == "TreeEnsemble"
    assert meta.has_zipmap is True

    # ตรวจสอบการดึงจุดตัดแกน X (feature 0) และแกน Y (feature 1) จากแอตทริบิวต์ ONNX
    splits = adapter.decompile_tree_ensemble_splits(feat_x_idx=0, feat_y_idx=1)
    assert len(splits["splits_x"]) > 0
    assert len(splits["splits_y"]) > 0

    # ตรวจสอบความแม่นยำของความน่าจะเป็นหลัง Bypass ZipMap เทียบกับ Scikit-learn
    batch_res = adapter.predict_batch(df)
    assert batch_res.probabilities.dtype == np.float32
    assert batch_res.probabilities.shape == (30, 2)
    np.testing.assert_allclose(batch_res.probabilities, rf.predict_proba(df.to_numpy()), rtol=1e-5, atol=1e-5)


def test_onnx_multi_column_named_inputs_and_linear_decompiler(
    tmp_path: Path, sample_xy: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """ยืนยันการทำงานของ _build_feed_dict เมื่อไฟล์ ONNX มีหลาย Named Inputs แยกรายคอลัมน์"""
    df, y = sample_xy
    clf = LogisticRegression().fit(df, y)

    initial_types = [(col, FloatTensorType([None, 1])) for col in df.columns]
    onnx_proto = convert_sklearn(clf, initial_types=initial_types)
    onnx_path = tmp_path / "linear_multi_in.onnx"
    onnx.save_model(onnx_proto, str(onnx_path))

    adapter = ONNXClassicalAdapter(model_path=onnx_path, onnx_proto=onnx_proto)
    schema = adapter.extract_schema()
    assert schema.feature_names == ["feature_0", "feature_1", "feature_2", "feature_3"]

    batch_res = adapter.predict_batch(df)
    np.testing.assert_allclose(batch_res.probabilities, clf.predict_proba(df), rtol=1e-5, atol=1e-5)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & UX Flow Tests (พร้อมโค้ดต้นแบบ `vitest`)

ไฟล์ทดสอบหลัก: `frontend/src/domains/simulation/components/stage3d/renderers/__tests__/ONNXClassicalStageRouter.test.tsx`

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { ONNXGraphInspectorCard } from "../../../hud/ONNXGraphInspectorCard";

describe("Phase 38 — ONNXClassicalStageRouter & ONNXGraphInspectorCard", () => {
  it("แสดงรายละเอียดโหนด ai.onnx.ml, สถานะ ZipMap Bypass และ Input Tensor Signatures ได้ถูกต้อง", () => {
    render(
      <ONNXGraphInspectorCard
        domain="ai.onnx.ml"
        detectedOperator="TreeEnsembleClassifier"
        opsetVersion={18}
        zipMapBypassed={true}
        inputs={[{ name: "float_input", shape: ["batch_size", 10], tensorType: "tensor(float)" }]}
        nodeChain={["Scaler", "TreeEnsembleClassifier", "ZipMap (Bypassed)"]}
      />
    );

    expect(screen.getByText(/TreeEnsembleClassifier/i)).toBeDefined();
    expect(screen.getByText(/ZipMap \(Bypassed\)/i)).toBeDefined();
    expect(screen.getByText(/tensor\(float\)/i)).toBeDefined();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Process Lifecycle & GPU/CPU Resource Tests)

1. **`ZipMap` Bypass Memory & Speed Benchmark**:
   - ทดสอบรัน `predict_batch()` บนข้อมูล `2,500` แถว สำหรับโมเดล 10 คลาส ยืนยันว่าการอ่าน Raw Float Tensor ก่อนเข้า `ZipMap` ไม่สร้าง `list[dict]` ใน Python Heap และใช้เวลาประมวลผล `< 25 ms` ต่อ Chunk
2. **C-Contiguous Array Zero-Copy Verification**:
   - ตรวจสอบว่าทุกอาเรย์ใน `feed_dict` ที่สร้างโดย `_build_feed_dict()` มีคุณสมบัติ `arr.flags["C_CONTIGUOUS"] is True` ก่อนส่งเข้า `ort.InferenceSession.run()`

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| กรณีปัญหา (Edge Case จาก Dev Phase 38) | วิธีจำลองในการทดสอบ (Test Simulation Setup) | ผลลัพธ์ที่ต้องผ่านการตรวจสอบ (Expected Verification Assertion) |
| :--- | :--- | :--- |
| **1. `ZipMap` คืนค่า `list[dict]` แทน `ndarray`** | แปลง Classifier ด้วย `skl2onnx` โดยเปิด `zipmap=True` (ค่าเริ่มต้น) | `_build_optimized_session()` ตัดโหนด `ZipMap` ออกหรือแปลง `list[dict]` เป็น `np.float32` ขนาด `(N, C)` ได้สำเร็จ 100% |
| **2. Single 2D Input vs Multi-Input Tensors** | ทดสอบทั้งไฟล์ `.onnx` ที่มี 1 Input `[None, 4]` และไฟล์ที่มี 4 Inputs `[None, 1]` | `_build_feed_dict()` จัดรูปเมทริกซ์ให้ตรงตาม `sess.get_inputs()` ทุกประการโดยไม่เกิด `InvalidArgument` |
| **3. ชนิดข้อมูล `tensor(double)` vs `tensor(float)`** | แปลงโมเดลด้วย `DoubleTensorType([None, 4])` แล้วส่ง DataFrame `float32` เข้าไป | ระบบตรวจพบ `inp.type == "tensor(double)"` จึงแคสต์ Input เฉพาะตอนส่งเข้า Session เป็น `np.float64` แล้วแปลงเอาต์พุตกลับเป็น `np.float32` |
| **4. `TreeEnsemble` OpSet 5 (ONNX 1.16+) เปลี่ยนชื่อ Attribute** | สร้างกราฟ ONNX ที่ใช้ทั้งโหนด `TreeEnsembleClassifier` (OpSet 3) และ `TreeEnsemble` (OpSet 5) | `decompile_tree_ensemble_splits()` สกัดจุดตัดแกน $X, Y$ ออกมาได้ทั้งสองเวอร์ชันโดยไม่เกิด `AttributeError` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **100% Operator Family Detection**: จำแนกตระกูล `TreeEnsemble`, `Linear`, และ `SVM` ในโดเมน `ai.onnx.ml` และเลือกฉาก 3D ได้ถูกต้อง 100%
- [ ] **Numerical Parity (`< 1e-5`)**: ผลลัพธ์จาก `ONNXClassicalAdapter` ตรงกับโมเดล Scikit-learn/XGBoost/LightGBM ต้นฉบับในระดับ `< 1e-5`
- [ ] **Zero-Dict Overhead**: ยืนยันว่า `ZipMap` Bypass ทำงานสำเร็จและคืนค่า `np.float32` โดยตรง

- [ ] **SVM & Linear ONNX Decompiler Accuracy**: ยืนยันว่าการดึง `support_vectors`, `kernel_params`, และ `coefficients` จากโหนด `ai.onnx.ml.SVMClassifier` และ `LinearClassifier` ให้พิกัดเสาแสง Support Vectors และความชันระนาบ 3D ตรงกับโมเดลต้นฉบับ 100%

### คำสั่งรันชุดทดสอบประจำเฟส 38 (CLI Test Execution Commands)
```bash
# รันชุดทดสอบ Backend ONNXClassicalAdapter (TreeEnsemble, Linear, SVM, ZipMap)
pytest backend/tests/domains/model_engine/adapters/neural_onnx/test_onnx_classical.py -v

# รันชุดทดสอบ Frontend ONNXClassicalStageRouter & ONNXGraphInspectorCard
cd frontend && npx vitest run src/domains/simulation/components/stage3d/renderers/__tests__/ONNXClassicalStageRouter.test.tsx
```
