# Test Plan — Phase 37: Scikit-Learn `Pipeline` & `ColumnTransformer` Unwrapper & Multi-Step 3D Tunnel

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 37)
เอกสารนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ **[phase-37-sklearn-pipeline-unwrapper.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-37-sklearn-pipeline-unwrapper.md)** เพื่อยืนยันความถูกต้องของคลาส **`SklearnPipelineAdapter` (`backend/app/domains/model_engine/adapters/neural_onnx/pipeline_adapter.py`)** และคอมโพเนนต์ UI **`PipelineStepBreadcrumb`** ร่วมกับ **`PipelineTransformationCard`**

### ขอบเขตการตรวจสอบหลัก
1. **Raw Input Schema Extraction & Recursive Unwrapping**: ยืนยันการดึงชื่อคอลัมน์ดิบต้นทางจาก `pipeline.feature_names_in_` และการแกะ `ColumnTransformer` ที่ซ้อน `Pipeline` ย่อยภายใน
2. **Step-by-Step Transformation Tracer & Imputation Flagging**: ยืนยันการติดตามค่าตั้งแต่ `Raw DataFrame` $\to$ `SimpleImputer` $\to$ `StandardScaler / OneHotEncoder` พร้อมตรวจจับแถวที่ถูกเติมค่าว่าง (`NaN`)
3. **Feature Lineage Attribution Aggregation**: ยืนยันการรวมค่าอิทธิพลของคอลัมน์ที่แตกตัวจาก `OneHotEncoder` ($\sum_{m \in \mathcal{G}_{\text{lineage}}(j)} \phi_{\text{trans}}(m)$) กลับมายังชื่อคอลัมน์ดิบต้นทางอย่างถูกต้องตามทฤษฎี Shapley Additivity
4. **3D Raw Domain Units vs Transformed Z-Score Space Slicing**: ยืนยันว่าการสร้างพื้นผิว 3D Mesh ในหน่วยข้อมูลดิบส่ง `pd.DataFrame` ผ่าน `pipeline[:-1].transform()` ก่อนส่งให้ Adapter ของโมเดลตัวสุดท้าย

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow / System Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

ไฟล์ทดสอบหลัก: `backend/tests/domains/model_engine/adapters/neural_onnx/test_pipeline_adapter.py`

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.domains.model_engine.adapters.linear.logistic_adapter import LogisticRegressionAdapter
from app.domains.model_engine.adapters.neural_onnx.pipeline_adapter import SklearnPipelineAdapter


@pytest.fixture
def production_pipeline_and_df() -> tuple[Pipeline, pd.DataFrame]:
    df = pd.DataFrame({
        "income": [45000.0, np.nan, 82000.0, 29000.0, 110000.0, 64000.0],
        "age": [28.0, 35.0, 49.0, np.nan, 52.0, 41.0],
        "city": ["Bangkok", "Chiang Mai", "Phuket", "Bangkok", "Chiang Mai", "Phuket"],
    })
    y = np.array([0, 0, 1, 0, 1, 1])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=True, handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, ["income", "age"]),
        ("cat", cat_pipe, ["city"]),
    ])
    full_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression()),
    ])
    full_pipe.fit(df, y)
    return full_pipe, df


def test_pipeline_schema_lineage_and_sparse_to_dense_handling(
    production_pipeline_and_df: tuple[Pipeline, pd.DataFrame]
) -> None:
    """ยืนยันการสกัด Raw Schema (3 คอลัมน์), การแปลง Sparse OHE เป็น Dense float32, และการรวม Attribution"""
    pipe, df = production_pipeline_and_df
    inner_adapter = LogisticRegressionAdapter(pipe.named_steps["clf"])
    adapter = SklearnPipelineAdapter(pipeline=pipe, inner_estimator_adapter=inner_adapter)

    schema = adapter.extract_schema()
    assert schema.feature_names == ["income", "age", "city"]

    # ทดสอบการติดตามขั้นตอนย่อยและการตรวจจับค่าว่างที่ถูก Impute ในแถวที่ 1 (income=NaN) และแถวที่ 3 (age=NaN)
    X_trans, step_traces = adapter.trace_preprocessing_steps(df)
    assert X_trans.dtype == np.float32
    assert X_trans.shape == (6, 5)  # 2 numeric + 3 one-hot categories
    assert step_traces[0].imputed_mask[1, 0] is True or bool(step_traces[0].imputed_mask[1, 0]) is True

    # ทดสอบการทำนายและรวมค่าอิทธิพลจาก 5 คอลัมน์หลังแปลง กลับมาเป็น 3 คอลัมน์ดิบต้นทาง
    batch_res = adapter.predict_batch(df)
    assert batch_res.feature_contributions.shape == (6, 3)
    np.testing.assert_allclose(batch_res.probabilities, pipe.predict_proba(df), rtol=1e-5, atol=1e-5)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & UX Flow Tests (พร้อมโค้ดต้นแบบ `vitest`)

ไฟล์ทดสอบหลัก: `frontend/src/domains/simulation/components/hud/__tests__/PipelineStepBreadcrumb.test.tsx`

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { PipelineStepBreadcrumb } from "../PipelineStepBreadcrumb";
import { PipelineTransformationCard } from "../PipelineTransformationCard";

describe("Phase 37 — PipelineStepBreadcrumb & Transformation Tracer UI", () => {
  it("แสดงลำดับขั้นตอนใน Pipeline และสลับมุมมองแกน 3D ระหว่าง Raw Units กับ Transformed Space ได้", () => {
    const onToggleSpace = vi.fn();
    render(
      <PipelineStepBreadcrumb
        steps={[
          { stepName: "Raw Input", inputDim: 3, outputDim: 3 },
          { stepName: "ColumnTransformer (Imputer + Scaler + OHE)", inputDim: 3, outputDim: 5 },
          { stepName: "LogisticRegression", inputDim: 5, outputDim: 1 },
        ]}
        coordinateSpace="raw"
        onToggleCoordinateSpace={onToggleSpace}
      />
    );

    expect(screen.getByText(/ColumnTransformer/i)).toBeDefined();
    const toggleBtn = screen.getByRole("button", { name: /Transformed Space/i });
    fireEvent.click(toggleBtn);
    expect(onToggleSpace).toHaveBeenCalledWith("transformed");
  });

  it("แสดงป้ายแจ้งเตือน [Imputed!] เมื่อแถวข้อมูลปัจจุบันมีค่าว่างที่ถูกเติมโดย SimpleImputer", () => {
    render(
      <PipelineTransformationCard
        rowIndex={19}
        entries={[
          { rawColumn: "income", rawValue: "45,000", transformedSummary: "num__income = +0.742", isImputed: false },
          { rawColumn: "age", rawValue: "NaN", transformedSummary: "num__age = 0.000 (Median=34)", isImputed: true },
        ]}
      />
    );

    expect(screen.getByText(/Imputed/i)).toBeDefined();
    expect(screen.getByText(/45,000/)).toBeDefined();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Process Lifecycle & GPU/CPU Resource Tests)

1. **Sparse One-Hot Matrix Explosion Guard Test**:
   - สร้าง `ColumnTransformer` ที่มี `OneHotEncoder(sparse_output=True)` บนคอลัมน์ที่มี 200 หมวดหมู่ แล้วรันผ่าน `SklearnPipelineAdapter` บน Chunk ขนาด `2,500` แถว ยืนยันว่าระบบแปลงเป็น `np.float32` เฉพาะภายใน Chunk ปัจจุบัน และตัดส่ง Preview ไปยัง UI เฉพาะ `Top-16 non-zero active columns` ต่อแถว
2. **Zero-Copy Inner Adapter Delegation Test**:
   - ตรวจสอบว่า `X_transformed` ที่ส่งเข้า `self.inner_estimator_adapter` ใช้ Buffer เดียวกันในหน่วยความจำ (`np.shares_memory`) โดยไม่มีการ Serialize/Deserialize ซ้ำซ้อน

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| กรณีปัญหา (Edge Case จาก Dev Phase 37) | วิธีจำลองในการทดสอบ (Test Simulation Setup) | ผลลัพธ์ที่ต้องผ่านการตรวจสอบ (Expected Verification Assertion) |
| :--- | :--- | :--- |
| **1. `ColumnTransformer` โยน Error เมื่อส่ง `np.ndarray`** | สร้าง `ColumnTransformer` ที่อ้างอิงชื่อคอลัมน์เป็น String แล้วเรียกทั้ง `predict_batch` และ 3D Surface Generator | ระบบห่อข้อมูลเข้า `pd.DataFrame(columns=feature_names_in_)` เสมอ ไม่เกิด `ValueError: Specifying the columns using strings is only supported for pandas DataFrames` |
| **2. `OneHotEncoder` คืนค่า `scipy.sparse.csr_matrix`** | ตั้งค่า `OneHotEncoder(sparse_output=True)` คู่กับโมเดลปลายทางที่ไม่รับ Sparse Matrix | `trace_preprocessing_steps()` ตรวจจับ `sparse.issparse(X)` และแปลงเป็น Dense `np.float32` อัตโนมัติ |
| **3. Nested Pipeline ซ้อนอยู่ใน `ColumnTransformer`** | สร้าง `Pipeline` ย่อย 3 ชั้นซ้อนใน `ColumnTransformer` แล้วซ้อนใน `Pipeline` หลัก | `_unwrap_transformers_recursively()` ดึงรายชื่อขั้นตอนและสร้าง `FeatureLineageMap` ได้ครบถ้วนโดยไม่ตกหล่น |
| **4. โมเดลใน What-If Sandbox ปรับค่าในหน่วย Raw vs Scaled** | ปรับค่า `income` จาก `45,000` เป็น `95,000` บาทใน What-If Sandbox | ระบบนำแถวหน่วยดิบผ่าน `pipeline[:-1].transform()` ก่อนส่งเข้า `inner_estimator_adapter` ได้ผลลัพธ์ตรงกับ `pipeline.predict_proba()` 100% |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **100% Delegation Compatibility**: ทดสอบ `Pipeline` ที่ลงท้ายด้วยโมเดลตัวแทนจากทุกกลุ่ม (Logistic, SVM, RandomForest, XGBoost, LightGBM, CatBoost, MLP) ผ่านครบ 100%
- [ ] **Lineage Additivity Parity (`< 1e-5`)**: ผลรวมของ Attribution ที่ยุบรวมกลับมาเป็นคอลัมน์ดิบต้นทาง ($\sum \Phi_{\text{raw}}$) เท่ากับผลรวม Attribution ในปริภูมิแปลงแล้ว ($\sum \phi_{\text{trans}}$) ทุกแถว
- [ ] **Raw-Space 3D Surface Accuracy**: พื้นผิว 3D ในโหมด `Raw Domain Units` มีความสูง $Z$ ณ พิกัด $(x_{\text{raw}}, y_{\text{raw}})$ ตรงกับการเรียก `pipeline.predict_proba()` โดยตรง
