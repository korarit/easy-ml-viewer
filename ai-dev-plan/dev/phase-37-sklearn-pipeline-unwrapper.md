# Phase 37: Scikit-Learn `Pipeline` & `ColumnTransformer` Unwrapper & Multi-Step 3D Tunnel

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
พัฒนา Adapter สำหรับ **`sklearn.pipeline.Pipeline`** และ **`sklearn.compose.ColumnTransformer`** ซึ่งเป็นหัวใจสำคัญของการนำโมเดลใช้งานจริงในระดับ Production เนื่องจากโมเดลในโลกจริงมักถูกห่อรวมกับขั้นตอนเตรียมข้อมูล (เช่น `SimpleImputer` $\to$ `StandardScaler` / `RobustScaler` / `OneHotEncoder` $\to$ `Final Estimator`) ไว้ในไฟล์ `.pkl` หรือ `.joblib` เพียงไฟล์เดียว

### ขอบเขตการพัฒนา (Scope)
1. **Recursive Pipeline & ColumnTransformer Introspector**: แกะรายชื่อคอลัมน์ต้นทาง (Raw Input Schema) ที่ Pipeline ต้องการตั้งแต่ด่านแรกสุด (`feature_names_in_`) และติดตามการแตกตัวของฟีเจอร์หลังผ่าน `OneHotEncoder` หรือ `PolynomialFeatures` (`get_feature_names_out()`)
2. **Step-by-Step Transformation Tracer**: บันทึกค่าของข้อมูลในแต่ละแถวตั้งแต่ `Raw Input Space` $\to$ `Imputed Space` $\to$ `Scaled / Encoded Space` เพื่อแสดงให้เห็นว่าข้อมูลเปลี่ยนรูปไปอย่างไรก่อนเข้าสู่โมเดล
3. **Final Estimator Adapter Delegation (Composite Pattern)**: ส่งต่อเมทริกซ์ข้อมูลที่ผ่านการแปลงแล้ว ($X_{\text{transformed}}$) ไปยัง Adapter เฉพาะทางของโมเดลตัวสุดท้าย (`pipeline.steps[-1][1]`) ที่สร้างไว้ใน Phase 04 และ Phase 18–36 โดยอัตโนมัติ
4. **3D Raw-Space vs Transformed-Space Coordinate Switcher**: ให้ผู้ใช้สลับพิกัดแกน $X, Y$ บนฉาก 3D ได้ทันทีระหว่างหน่วยจริงในชีวิตประจำวัน (เช่น อายุ `45 ปี`, รายได้ `55,000 บาท`) กับหน่วยทางคณิตศาสตร์หลังแปลง (เช่น Z-Score `+1.24 SD`)

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 คณิตศาสตร์ของการแปลงปริภูมิหลายขั้นตอน (Composite Function Composition)
กำหนดให้ `Pipeline` ประกอบด้วยขั้นตอนการแปลงข้อมูล $K$ ขั้นตอน $T_1, T_2, \dots, T_K$ ตามด้วยโมเดลตัดสินใจตัวสุดท้าย $M_{\text{final}}$ สำหรับเวกเตอร์ข้อมูลดิบ $\mathbf{x}_{\text{raw}} \in \mathcal{X}_{\text{raw}}$ (ซึ่งอาจมีทั้งตัวเลข, ข้อความ, และค่าว่าง `NaN`):

$$
\mathbf{h}^{(0)} = \mathbf{x}_{\text{raw}}, \qquad \mathbf{h}^{(k)} = T_k\left(\mathbf{h}^{(k-1)}\right) \quad \forall k \in \{1, \dots, K\}
$$

และผลลัพธ์การทำนายสุดท้ายคือ:

$$
\hat{y} = M_{\text{final}}\left(\mathbf{h}^{(K)}\right) = \left(M_{\text{final}} \circ T_K \circ \dots \circ T_1\right)(\mathbf{x}_{\text{raw}})
$$

### 2.2 การแกะโครงสร้าง `ColumnTransformer` และการจับคู่ฟีเจอร์ต้นทางสู่ฟีเจอร์ปลายทาง (Lineage Mapping)
ใน `ColumnTransformer` ข้อมูลดิบ $\mathbf{x}_{\text{raw}}$ จะถูกแยกเข้าท่อย่อยตามกลุ่มคอลัมน์ $\{(name_m, trans_m, cols_m)\}_{m=1}^M$ แล้วนำมาต่อกัน (Concatenate):

$$
T_{\text{CT}}(\mathbf{x}_{\text{raw}}) = \bigoplus_{m=1}^{M} trans_m\left(\mathbf{x}_{\text{raw}}[cols_m]\right) \in \mathbb{R}^{d_{\text{out}}}
$$

ปัญหายากที่สุดคือเมื่อมี `OneHotEncoder` จำนวนคอลัมน์หลังแปลง $d_{\text{out}}$ จะมากกว่าจำนวนคอลัมน์ดิบ $d_{\text{in}}$ (เช่น คอลัมน์ `city` 1 คอลัมน์ แตกออกเป็น `city_BKK`, `city_CNX`, `city_HKT` 3 คอลัมน์) ระบบจึงสร้าง **Feature Lineage Graph** $\mathcal{G}_{\text{lineage}}: \text{RawCol}_j \mapsto \{\text{TransCol}_{m_1}, \dots, \text{TransCol}_{m_r}\}$:
1. ดึงชื่อคอลัมน์หลังแปลงจาก `preprocessor.get_feature_names_out()`
2. เมื่อ Adapter ของโมเดลตัวสุดท้ายคำนวณค่าอิทธิพลรายฟีเจอร์ในปริภูมิแปลงแล้ว $\phi_{\text{trans}}(m)$ (เช่น SHAP หรือ $\beta_m x_m$) ระบบสามารถ **รวมค่าอิทธิพลย้อนกลับมายังคอลัมน์ดิบต้นทาง (Aggregated Raw Feature Attribution)** ตามกฎการบวกของ Shapley Values:
   $$
   \Phi_{\text{raw}}(\text{RawCol}_j) = \sum_{m \in \mathcal{G}_{\text{lineage}}(\text{RawCol}_j)} \phi_{\text{trans}}(m)
   $$

### 2.3 การสร้างพื้นผิว 3D ในหน่วยจริง (Raw-Space 3D Surface Slicing)
เมื่อผู้ใช้เลือกแสดงฉาก 3D ในโหมด **Raw Units Space** บนแกนคอลัมน์ดิบ $X_{\text{raw}}, Y_{\text{raw}}$:
1. ระบบสร้างตาราง Grid $50 \times 50$ จุดในหน่วยข้อมูลดิบจริง (เช่น `income` ตั้งแต่ `15,000` ถึง `150,000` บาท)
2. นำตาราง Grid ขนาด `2,500` แถวในรูป `pd.DataFrame` (ที่มีชื่อคอลัมน์ตรงกับ `pipeline.feature_names_in_` ทุกประการ) ส่งผ่าน `pipeline[:-1].transform(grid_df)` ก่อนส่งเข้า `final_adapter`
3. ทำให้พื้นผิว 3D ของโมเดลตัวสุดท้าย (ไม่ว่าจะเป็น SVM, Random Forest, XGBoost หรือ MLP) ถูovาดลงบนสเกลหน่วยเงินบาทและอายุจริงได้อย่างแม่นยำ 100%!

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)
1. **Interactive Pipeline Step Breadcrumb Bar (แถบแสดงลำดับท่อข้อมูลด้านบนฉาก 3D)**:
   - แสดงบล็อกขั้นตอนเรียงต่อกัน เช่น `[Raw CSV (8 cols)] ➔ [SimpleImputer] ➔ [ColumnTransformer (Scaler + OHE -> 19 cols)] ➔ [XGBClassifier]`
   - ผู้ใช้สามารถคลิกที่บล็อกใดก็ได้เพื่อเปิดดูตารางเปรียบเทียบค่า **Before vs After** ของแถวปัจจุบันในขั้นตอนนี้
2. **3D Axis Space Toggle (`[Raw Domain Units (บาท/ปี)]` $\leftrightarrow$ `[Transformed Z-Score Space]`)**:
   - สวิตช์สลับหน่วยแกน $X, Y$ บนฉาก 3D พร้อมแอนิเมชันยืด/หดสเกลพื้นผิวจากค่าจริงไปเป็นค่า Standardized Z-Score
3. **Aggregated vs Expanded Waterfall Toggle**:
   - ในแผง Explainability ด้านขวา ผู้ใช้เลือกได้ว่าจะดูอิทธิพลแบบ **Grouped by Original Column** (รวม `city_BKK + city_CNX` เป็นก้อน `city` เดียว) หรือดูแบบ **Expanded Encoded Features**
4. **Imputation & Out-of-Vocabulary Warning Badges**:
   - หากแถวข้อมูลที่กำลังเล่นใน VCR มีค่าว่าง (`NaN`) ที่ถูก `SimpleImputer` เติมค่าให้ หรือมีหมวดหมู่ใหม่ที่ `OneHotEncoder(handle_unknown='ignore')` แปลงเป็นศูนย์ล้วน ระบบจะขึ้นป้ายไฟกะพริบ `[Imputed Value]` บน HUD ทันที

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

### UX Flow
1. ผู้ใช้อัปโหลดไฟล์ `loan_pipeline.pkl` คู่กับ `applicants.csv`
2. ระบบตรวจพบ `sklearn.pipeline.Pipeline` $\to$ แกะพบว่าข้างในมี `ColumnTransformer` และปิดท้ายด้วย `RandomForestClassifier`
3. ระบบจับคู่คอลัมน์ดิบจาก `pipeline.feature_names_in_` เข้ากับ CSV อัตโนมัติ และเลือกใช้ `RandomForestAdapter` (จาก Phase 28) เป็นตัววาดฉาก 3D
4. ขณะเลื่อน VCR Scrubber ผู้ใช้เห็นทั้งฉาก 3D ของ Random Forest และแถบ Pipeline Inspector ด้านล่างที่แสดงว่า `income = 45,000` ถูก `StandardScaler` แปลงเป็น `z = +0.74` ก่อนเข้าต้นไม้ตัดสินใจ

### ASCII Wireframe
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer]  Pipeline: [Raw Input (6)] -> [ColumnTransformer (14)] -> [RandomForest (100)]  |
| 3D Coordinate Space: (o) Raw Domain Units (THB, Years)    ( ) Transformed Space (Scaled Z-Score) |
+------------------------------------------------------------------+-------------------------------+
| 3D DELEGATED MODEL STAGE (RandomForest Voxel Terraces)           | PIPELINE STEP TRACER (Row #19)|
|                                                                  |                               |
|       Z (Approval Prob)                                          | [Step 0: Raw Input DataFrame] |
|       ^       +---------+---------+                              | - income : 45,000 THB         |
|   1.0 |      /         /         /|                              | - age    : NaN [Missing!]     |
|       |     +---------+---------+ |                              | - city   : "Chiang Mai"       |
|   0.5 |     |   (*)   |         | +                              |              |                |
|       |    /  Row #19 /         /                                |              v                |
|   0.0 +---+---------+---------+---> X: income (15,000 - 120,000) | [Step 1: ColumnTransformer]   |
|      /                                                           | - num__income : +0.742 (Z)    |
|     v Y: age (20 - 65 Years) [Imputed Median = 34.0]             | - num__age    :  0.000 [Imp!] |
|                                                                  | - cat__city_CM:  1.000 (OHE)  |
| +--------------------------------------------------------------+ |              |                |
| | PIPELINE TUNNEL: [Raw: 45k, NaN, 'CM'] ==> [0.74, 0.0, 1.0]  | |              v                |
| +--------------------------------------------------------------+ | [Step 2: RandomForest Output] |
|                                                                  | Approval Prob : 0.810 (81.0%) |
|                                                                  | Waterfall Mode: [Grouped Raw] |
+------------------------------------------------------------------+-------------------------------+
| [|<] [<] [ || Pause ] [>] [>|]    Frame: 0019 / 1500   [====*===================]  Speed: [1.0x v]|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD แยกรายชั้นพร้อมชื่อไฟล์และ Code Signatures

### 5.1 Backend (`backend/app/domains/model_engine/`)
- **`adapters/neural_onnx/pipeline_adapter.py`**
  ```python
  import numpy as np
  import pandas as pd
  from scipy import sparse
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  from app.domains.model_engine.domain.ports import BaseModelAdapter
  from app.domains.model_engine.domain.entities import (
      ModelSchema, PredictionBatchResult, PipelineStepTrace, FeatureLineageMap
  )

  class SklearnPipelineAdapter(BaseModelAdapter):
      def __init__(self, pipeline: Pipeline, inner_estimator_adapter: BaseModelAdapter) -> None: ...

      def extract_schema(self) -> ModelSchema:
          """Extracts raw feature_names_in_ from Step 0 and builds FeatureLineageMap to Step K."""
          ...

      def trace_preprocessing_steps(self, df_chunk: pd.DataFrame) -> tuple[np.ndarray, list[PipelineStepTrace]]:
          """Executes pipeline[:-1] step-by-step, converting sparse matrices to dense float32,
          and records intermediate values and imputation flags."""
          ...

      def aggregate_lineage_attributions(self, transformed_contribs: np.ndarray) -> np.ndarray:
          """Sums feature contributions of one-hot/polynomial columns back to raw input columns."""
          ...

      def predict_batch(self, df_chunk: pd.DataFrame) -> PredictionBatchResult:
          """Transforms df_chunk and delegates internal flow extraction to inner_estimator_adapter."""
          ...
  ```

### 5.2 Frontend (`frontend/src/domains/simulation/`)
- **`components/hud/PipelineStepBreadcrumb.tsx`**: แถบแสดงลำดับ Step ของ Pipeline พร้อม Drawer เปิดดูค่าก่อน/หลังแปลง
- **`components/hud/PipelineTransformationCard.tsx`**: การ์ดแสดงการแปลงค่าของแถวปัจจุบันตั้งแต่ Raw Value $\to$ Imputed $\to$ Scaled/Encoded

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)
1. **Sparse-to-Dense Memory Guard (`scipy.sparse.issparse`)**:
   - `OneHotEncoder` ใน Scikit-learn มีค่าเริ่มต้นเป็น `sparse_output=True` (คืนค่า `scipy.sparse.csr_matrix`) หากแปลงเป็น Dense Array ทั้งไฟล์ใหญ่อาจทำให้ RAM พุ่งสูง ระบบจึงบังคับแปลง `.toarray().astype(np.float32)` เฉพาะภายใน Chunk ขนาดไม่เกิน `2,500` แถว และจำกัดจำนวนคอลัมน์หลัง One-Hot สูงสุดที่ `512` คอลัมน์ในการส่ง Telemetry ไปยัง Frontend
2. **Zero-Copy Final Estimator Delegation**:
   - เมทริกซ์ `X_transformed` (`float32`) ที่ได้จาก `pipeline[:-1].transform(df_chunk)` จะถูกส่งต่อให้ `self.inner_estimator_adapter` โดยตรงในหน่วยความจำโดยไม่ต้องสร้างไฟล์ชั่วคราวหรือ Serialize ซ้ำ
3. **Top-16 Intermediate Step Preview Truncation**:
   - ในการเก็บ `PipelineStepTrace` สำหรับส่งไปแสดงใน UI ด้านขวา หากขั้นตอนใดสร้างคอลัมน์ออกมามากกว่า 16 คอลัมน์ ระบบจะส่งเฉพาะ 16 คอลัมน์ที่มีค่าไม่เป็นศูนย์ (`non-zero active columns`) ของแถวนั้นเพื่อประหยัดขนาด Payload

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| กรณีปัญหา (Edge Case) | สาเหตุเชิงเทคนิค | แนวทางป้องกันและแก้ไขในระบบ (Mitigation Strategy) |
| :--- | :--- | :--- |
| **1. `ColumnTransformer` โยน Error เมื่อส่ง `np.ndarray`** | `ColumnTransformer` มักอ้างอิงคอลัมน์ด้วยชื่อ String (`["income", "city"]`) จึงบังคับว่า Input ต้องเป็น `pd.DataFrame` เท่านั้น | ใน `SklearnPipelineAdapter` ห้ามแปลง `df_chunk` เป็น `np.ndarray` ก่อนเข้า Step แรกเด็ดขาด ต้องส่ง `pd.DataFrame` ที่มีชื่อคอลัมน์ตรงกับ `feature_names_in_` เสมอ |
| **2. `OneHotEncoder` คืนค่า `csr_matrix` (Sparse)** | โมเดลหรือโค้ดคำนวณ SHAP/Surface บางตัวไม่รองรับ Sparse Matrix | ตรวจสอบ `if sparse.issparse(X_step): X_step = X_step.toarray()` และแคสต์เป็น `np.float32` ทันทีในระดับ Chunk |
| **3. Nested Pipeline ซ้อนอยู่ใน `ColumnTransformer`** | นักพัฒนามักสร้าง `num_pipe = Pipeline([('imputer', ...), ('scaler', ...)])` ซ้อนไว้ใน `ColumnTransformer` | เขียนฟังก์ชันแกะ Schema แบบ Recursive (`_unwrap_transformers_recursively`) เพื่อดึงรายชื่อขั้นตอนย่อยและสถานะ `SimpleImputer` ออกมาได้ครบทุกระดับ |
| **4. โมเดลใน What-If Sandbox ปรับค่าในหน่วย Raw vs Scaled** | ผู้ใช้ลากสไลเดอร์ What-If ในหน่วยเงินบาทจริง แต่โมเดลตัวสุดท้ายรับค่า Z-Score | เมื่อผู้ใช้ปรับค่าใน What-If Sandbox ระบบจะส่งแถวข้อมูลหน่วยดิบผ่าน `pipeline[:-1].transform()` ก่อนส่งเข้า `inner_estimator_adapter` เสมอ |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] อัปโหลดไฟล์ `.pkl`/`.joblib` ที่เป็น `Pipeline` (ทั้งแบบเส้นตรงและแบบที่มี `ColumnTransformer` ซ้อน `Pipeline` ย่อย) ได้โดยสกัดรายชื่อคอลัมน์ดิบต้นทางถูกต้อง 100%
- [ ] ส่งต่อการทำงานไปยัง Adapter ของโมเดลตัวสุดท้าย (เช่น Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost, MLP) และเรนเดอร์ฉาก 3D ประจำตระกูลนั้นได้ครบถ้วน
- [ ] สลับมุมมองแกน 3D ระหว่าง `Raw Domain Units` และ `Transformed Z-Score Space` ได้โดยพื้นผิว 3D และตำแหน่งลูกบอลสอดคล้องกัน
- [ ] แสดงค่าการเปลี่ยนแปลงทีละขั้นใน `PipelineStepTracer` พร้อมแจ้งเตือนเมื่อแถวข้อมูลมีค่าว่างที่ถูกเติมโดย `Imputer`
- [ ] รวมค่าอิทธิพลของคอลัมน์ที่ถูกแตกด้วย `OneHotEncoder` กลับมาเป็นชื่อคอลัมน์ดิบต้นทางในกราฟ Waterfall ได้อย่างถูกต้อง
