# Phase 35: CatBoost Adapter (`.cbm`, `.pkl`) & 3D Symmetric Oblivious Tree Grid

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
พัฒนา Adapter เฉพาะทางสำหรับ **`CatBoost` (`CatBoostClassifier`, `CatBoostRegressor`)** ซึ่งเป็นสุดยอดอัลกอริทึม Gradient Boosting จาก Yandex ที่โดดเด่นเรื่องการจัดการข้อมูลประเภทหมวดหมู่ (Categorical Features) โดยตรง และการสร้างต้นไม้ตัดสินใจแบบสมมาตร (**Symmetric / Oblivious Decision Trees**) โดยรองรับทั้งไฟล์ Binary ดั้งเดิมของ CatBoost (`.cbm`) และไฟล์ตระกูล Python Serialization (`.pkl`, `.joblib`)

### ขอบเขตการพัฒนา (Scope)
1. **Multi-Format Loader**: โหลดและตรวจสอบไฟล์ `.cbm` แบบอัตโนมัติ (แยกแยะ Classifier กับ Regressor จาก Metadata ภายในไฟล์ `.cbm`) รวมถึง `.pkl`/`.joblib`
2. **Native Categorical `Pool` Engine**: สร้าง `catboost.Pool` ที่รักษาชนิดคอลัมน์ String/Int Categorical ตามดัชนี `get_cat_feature_indices()` โดยไม่ต้องทำ One-Hot Encoding ล่วงหน้า
3. **Exact Oblivious Tree & C++ SHAP Extractor**: สกัด Leaf Index (`calc_leaf_indexes`) และค่าอิทธิพลรายฟีเจอร์ (`get_feature_importance(type="ShapValues")`) ในระดับ C++
4. **3D Symmetric Oblivious Hypercube Grid**: เรนเดอร์พื้นผิวการตัดสินใจ 3 มิติที่เป็นตารางหมากรุกสมมาตร (Orthogonal Symmetric Terraces) พร้อมแกน 3D ที่รองรับ Discrete Categorical Ticks

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 โครงสร้างต้นไม้สมมาตร (Oblivious / Symmetric Trees)
ต่างจาก XGBoost (Depth-wise) และ LightGBM (Leaf-wise) ที่แต่ละโหนดในระดับความลึกเดียวกันสามารถใช้เงื่อนไขการตัด (`feature_id`, `threshold`) ต่างกันได้ ต้นไม้แบบ **Oblivious Tree** ของ CatBoost บังคับให้ **ทุกโหนดในระดับความลึก $d$ เดียวกัน ($d \in \{0, \dots, D-1\}$) ต้องใช้ฟีเจอร์ $f_d$ และจุดตัด $\tau_d$ เดียวกันทั้งหมด**:

$$
\text{Bit}_{d}(\mathbf{x}) = \mathbb{I}\left(x_{f_d} > \tau_d\right) \in \{0, 1\}
$$

ดัชนีใบไม้ (Leaf Index $l_t$) ของต้นไม้ต้นที่ $t$ ที่ความลึก $D$ จึงคำนวณได้ด้วยระดับบิต (Bitwise Encoding) ใน $O(D)$ โดยไม่ต้องเดินตาม Pointer:

$$
l_t(\mathbf{x}) = \sum_{d=0}^{D-1} \text{Bit}_{d}(\mathbf{x}) \cdot 2^d \in \{0, 1, \dots, 2^D - 1\}
$$

ผลลัพธ์ทางเรขาคณิตในปริภูมิ 3 มิติ คือ เส้นแบ่งเขตของแต่ละต้นไม้จะลากยาวทะลุทั้งปริภูมิเป็นเส้นตรงตั้งฉากกัน (`Orthogonal Hyperplanes`) ก่อให้เกิดภูมิประเทศแบบ **ตารางหมากรุก 3 มิติที่สมมาตร (3D Symmetric Grid Terraces)**

### 2.2 การตรวจสอบไฟล์ `.cbm` อัตโนมัติ (`.cbm` Binary Inspection)
ไฟล์ `.cbm` ไม่มีคลาส Python หุ้มอยู่ภายนอก ระบบจึงใช้อัลกอริทึม 2-Stage Probing:
1. สร้างอินสแตนซ์ชั่วคราว `probe = CatBoostClassifier()` แล้วเรียก `probe.load_model(str(file_path), format="cbm")`
2. อ่านค่า `loss_fn = probe.get_all_params().get("loss_function", "")`
   - หาก `loss_fn` อยู่ในกลุ่ม `{"RMSE", "MAE", "Quantile", "LogLinQuantile", "Poisson", "MAPE", "RMSEWithUncertainty"}` ระบบจะสลับไปโหลดด้วย `CatBoostRegressor().load_model(str(file_path))` ทันที
   - หากเป็น `{"Logloss", "CrossEntropy", "MultiClass", "MultiClassOneVsAll"}` จะคงสถานะเป็น `CatBoostClassifier`

### 2.3 การจัดการ Native Categorical `Pool` และ `ShapValues`
- **Cat Feature Preservation**: ดึงดัชนีคอลัมน์หมวดหมู่จาก `cat_indices = model.get_cat_feature_indices()` ในขั้นตอนเตรียมข้อมูลเข้า `catboost.Pool(data=df_chunk, cat_features=cat_indices)` คอลัมน์ที่อยู่ใน `cat_indices` จะถูกแปลงเป็น `str` (หรือ `int` ตามต้นฉบับ) ส่วนคอลัมน์ตัวเลขจะถูกบังคับเป็น `np.float32`
- **Exact TreeSHAP Extraction**: เรียก `shap_raw = model.get_feature_importance(data=pool, type="ShapValues")`
  - สำหรับ **Regression / Binary Classification**: คืนค่าอาเรย์ขนาด `(N, F + 1)` โดยคอลัมน์ที่ `0..F-1` คือค่า SHAP $\phi_j(\mathbf{x}_i)$ และคอลัมน์สุดท้าย `[:, -1]` คือ Base Value (Expected Value $E[f(X)]$)
  - สำหรับ **Multi-class Classification ($C$ คลาส)**: คืนค่าอาเรย์ 3 มิติขนาด `(N, C, F + 1)` ระบบจะเลือก Slice ของคลาสที่โมเดลทายชนะ $\hat{y}_i = \arg\max_c P(y=c|\mathbf{x}_i)$ มาแสดงเป็น Waterfall หลัก พร้อมแนบ Base Value ของคลาสนั้น

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)
1. **Symmetric Grid Laser Overlay (บนฉาก 3D Stage)**:
   - เส้นเลเซอร์สีฟ้าอมม่วงลากตัดผ่านแกน $X$ และ $Y$ ตลอดความกว้างของแผ่นพื้นผิว 3D เพื่อแสดงให้เห็นว่าเงื่อนไข Split ของ Oblivious Tree พาดผ่านทั้งปริภูมิพร้อมกัน
2. **Categorical Axis Discrete Ticks & Labels**:
   - เมื่อผู้ใช้เลือกแกน $X$ หรือ $Y$ เป็นคอลัมน์ที่เป็น Categorical Feature (มีไอคอนป้ายกำกับ `CAT` สีทอง) แกน 3D จะเปลี่ยนจากสเกลตัวเลขต่อเนื่องเป็นช่องตารางตามหมวดหมู่จริง (เช่น `["Bangkok", "Chiang Mai", "Phuket"]`)
3. **Bitwise Leaf Binary Badge (ใน HUD & Right Drawer)**:
   - แสดงหมายเลข Leaf ของแต่ละต้นไม้ควบคู่กับรหัสฐานสอง (เช่น `Tree #0: Leaf 5 [Bitmask: 101₂]`) เพื่ออธิบายว่าแถวข้อมูลปัจจุบันตอบ Yes/No ในแต่ละชั้นความลึกอย่างไร
4. **Target Statistics (CTR) Inspector Pill**:
   - แสดงป้ายแจ้งเตือนเมื่อฟีเจอร์ที่ส่งผลสูงสุดในแถวนั้นเป็น Categorical Feature ที่ผ่านกลไก Ordered Target Statistics ของ CatBoost

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

### UX Flow
1. ผู้ใช้อัปโหลดไฟล์ `credit_catboost.cbm` คู่กับ `customers.csv` (ที่มีคอลัมน์ข้อความ เช่น `occupation`, `housing_type`)
2. ระบบตรวจพบ `.cbm` $\to$ ระบุว่าเป็น `CatBoostClassifier` $\to$ ดึง `feature_names_` และไฮไลต์คอลัมน์ `occupation`, `housing_type` เป็น `Native Categorical` ในหน้า Column Mapping โดยไม่ต้องแปลงค่าเป็นตัวเลข
3. เมื่อเข้าสู่หน้า 3D Studio พื้นผิวจะแสดงเป็น **ตารางหมากรุก 3 มิติ (Symmetric Oblivious Terraces)** พร้อมเส้น Grid ตัดตรงแนวตั้งและแนวนอน
4. ผู้ใช้เลื่อน VCR Scrubber ไปที่แถวที่ต้องการ ลูกบอลแสงจะกระโดดไปตามช่องตารางหมากรุก พร้อมแสดง Bitmask `10110₂` ของต้นไม้แต่ละต้นในแผงด้านขวา

### ASCII Wireframe
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer]  Model: credit_catboost.cbm (CatBoostClassifier | 200 Oblivious Trees, Depth=6) |
+------------------------------------------------------------------+-------------------------------+
| 3D SYMMETRIC OBLIVIOUS TREE STAGE                                | OBLIVIOUS TREE & SHAP HUD     |
|                                                                  |                               |
|       Z (Prob: Default)                                          | Row #42 Prediction: 0.842     |
|       ^     +-------+-------+-------+                            | Base Value (Logit): -0.410    |
|   1.0 |    /       /       /       /|  <-- Orthogonal Split      |                               |
|       |   +-------+-------+-------+ |      Laser Planes          | [Symmetric Tree #0 Inspector] |
|       |   |  0.84 |  0.62 |  0.91 | +                            | Depth 0 (income > 35k)  : YES |
|   0.5 |   +-------+-------+-------+                              | Depth 1 (city == 'BKK') : NO  |
|       |  /  0.31 /  0.12 /  0.45 /   * <-- Active Row #42 Sphere | Depth 2 (age > 28)      : YES |
|   0.0 | +-------+-------+-------+      (Bitmask: 101_2 = Leaf 5) | -> Resulting Leaf Index : #5  |
|       +-----------------------------> X: occupation [CAT]        |    (Binary Path: 101_2)       |
|      /   "Salaried"  "Self-Emp" "Gov"                            |                               |
|     v Y: income (Numeric)                                        | [C++ Native SHAP Waterfall]   |
|                                                                  | occupation="Self" : +0.62 [+] |
| [x] Show Oblivious Split Grid Lines   [x] Categorical Axis Ticks | income=22,500     : +0.41 [+] |
+------------------------------------------------------------------+-------------------------------+
| [|<] [<] [ || Pause ] [>] [>|]  Frame: 0042 / 1000  [=======*================]  Speed: [1.0x v]  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD แยกรายชั้นพร้อมชื่อไฟล์และ Code Signatures

### 5.1 Backend (`backend/app/domains/model_engine/`)
- **`adapters/boosting/catboost_adapter.py`**
  ```python
  from pathlib import Path
  import numpy as np
  import pandas as pd
  from catboost import CatBoostClassifier, CatBoostRegressor, Pool
  from app.domains.model_engine.domain.ports import BaseModelAdapter
  from app.domains.model_engine.domain.entities import (
      ModelSchema, PredictionBatchResult, ObliviousTreeStructure
  )

  class CatBoostAdapter(BaseModelAdapter):
      def __init__(self, model: CatBoostClassifier | CatBoostRegressor, source_path: Path) -> None: ...
      
      @classmethod
      def load_from_path(cls, file_path: Path) -> "CatBoostAdapter":
          """Loads .cbm or .pkl/.joblib and auto-detects Classifier vs Regressor."""
          ...

      def extract_schema(self) -> ModelSchema:
          """Extracts feature_names_, get_cat_feature_indices(), and task type."""
          ...

      def build_pool(self, df_chunk: pd.DataFrame) -> Pool:
          """Constructs catboost.Pool preserving native categorical columns as str/int."""
          ...

      def predict_batch(self, df_chunk: pd.DataFrame) -> PredictionBatchResult:
          """Runs predict/predict_proba, calc_leaf_indexes(pool), and ShapValues in one Pool."""
          ...

      def extract_oblivious_split_grid(self, feature_x: str, feature_y: str) -> ObliviousTreeStructure:
          """Extracts level-wise symmetric split thresholds along active 3D axes."""
          ...
  ```

### 5.2 Frontend (`frontend/src/domains/simulation/`)
- **`components/stage3d/renderers/CatBoostObliviousGrid3D.tsx`**
  ```tsx
  export interface CatBoostObliviousGrid3DProps {
    surfaceMesh: Float32Array;
    orthogonalSplitsX: number[];
    orthogonalSplitsY: number[];
    categoricalTicksX?: string[];
    categoricalTicksY?: string[];
    activeFrameLeafBitmask: number;
  }
  export const CatBoostObliviousGrid3D: React.FC<CatBoostObliviousGrid3DProps> = (props) => { ... };
  ```
- **`components/hud/ObliviousBitmaskCard.tsx`**: แสดงตารางความลึก $d = 0 \dots D-1$ พร้อมสถานะบิต `0/1` ของต้นไม้ที่เลือก

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)
1. **Single `catboost.Pool` Reuse per Chunk**:
   - การสร้าง `catboost.Pool` มีต้นทุนในการแปลง Hash ของ Categorical String สูง ห้ามสร้าง `Pool` แยกกันระหว่าง `predict()`, `calc_leaf_indexes()`, และ `get_feature_importance()` เด็ดขาด ต้องสร้าง `pool = self.build_pool(df_chunk)` เพียงครั้งเดียวต่อ Chunk (ขนาดไม่เกิน `2,500` แถว) แล้วส่ง `pool` เดิมเข้าทั้ง 3 ฟังก์ชัน จากนั้นสั่ง `del pool` ทันทีเมื่อจบ Chunk
2. **Thread Control (`thread_count`)**:
   - กำหนด `thread_count = max(1, psutil.cpu_count(logical=False) - 1)` ในทุกการเรียก `calc_leaf_indexes` และ `get_feature_importance` เพื่อป้องกันไม่ให้ C++ Runtime ของ CatBoost แย่ง CPU 100% จน Event Loop ของ FastAPI ค้าง
3. **Downcasting Output Arrays**:
   - แปลงผลลัพธ์ `shap_values` จาก `float64` เป็น `np.float32` และ `leaf_indexes` จาก `int32` เป็น `np.uint16` (เนื่องจากความลึกสูงสุดของ CatBoost คือ 16 ชั้น มีใบไม้ไม่เกิน $2^{16} = 65,536$ ใบ) ช่วยลดขนาด RAM และ Payload ลงถึง 50%

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| กรณีปัญหา (Edge Case) | สาเหตุเชิงเทคนิค | แนวทางป้องกันและแก้ไขในระบบ (Mitigation Strategy) |
| :--- | :--- | :--- |
| **1. ไฟล์ `.cbm` ไม่ระบุคลาส Python** | ไฟล์ `.cbm` เป็น Binary ล้วน ไม่มี Header บอกว่าเป็น Classifier หรือ Regressor | โหลดด้วย `CatBoostClassifier` ก่อนแล้วเช็ก `get_all_params()["loss_function"]` หากเป็นตระกูล Regression ให้สลับเป็น `CatBoostRegressor` อัตโนมัติ |
| **2. ค่า `NaN` ในคอลัมน์ Categorical** | Pandas อ่านค่าว่างในคอลัมน์ String เป็น `np.nan` (`float`) ทำให้ `catboost.Pool` โยน `CatBoostError: Invalid type for cat_feature` | บังคับรัน `df_chunk[cat_col] = df_chunk[cat_col].fillna("__MISSING__").astype(str)` ก่อนส่งเข้า `Pool` เสมอ |
| **3. โมเดล `.cbm` ไม่มี `feature_names_`** | เทรนด้วย `np.ndarray` ล้วนโดยไม่ได้ส่ง `Pool(feature_names=...)` | หาก `not model.feature_names_` ให้สร้างชื่อมาตรฐาน `["feature_0", "feature_1", ...]` ตามจำนวน `model.tree_count_` / Input Dimension |
| **4. แกน 3D Mesh Slice เป็นคอลัมน์ Categorical** | ไม่สามารถใช้ `np.linspace(min, max, 50)` สร้าง Grid บนคอลัมน์ที่เป็นข้อความได้ | หากแกน $X$ หรือ $Y$ เป็น Categorical ระบบจะเปลี่ยน Grid Generator ของแกนนั้นเป็นรายการหมวดหมู่ที่ไม่ซ้ำกัน (`unique_categories[:32]`) แล้วเรียงลำดับตามความถี่ |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] อัปโหลดไฟล์ `.cbm` ทั้งแบบ Binary Classification, Multi-class Classification, และ Regression ได้โดยระบบเลือกคลาส Adapter ถูกต้อง 100%
- [ ] รองรับชุดข้อมูลที่มีคอลัมน์ Categorical เป็น String แท้ๆ ได้โดยไม่ต้องแปลงเป็นตัวเลขก่อน และไม่เกิด `CatBoostError` เมื่อมีค่าว่าง (`NaN`)
- [ ] ฉาก 3D แสดงพื้นผิว Symmetric Oblivious Grid พร้อมเส้นตัดตั้งฉาก (Orthogonal Split Planes) และแกน Categorical Ticks ได้อย่างถูกต้อง
- [ ] แผง HUD แสดงค่า Exact C++ `ShapValues` พร้อม Base Value และแสดงรหัสฐานสอง (Bitmask) ของใบไม้ในแต่ละต้นไม้ได้ถูกต้องตรงตามสมการ
- [ ] ทดสอบรันผ่าน `pytest` ด้วยหน่วยความจำส่วนเพิ่ม (Peak RAM Delta) ต่อ Chunk ไม่เกิน `120 MB` และคืนหน่วยความจำ `Pool` สะอาดหลังจบงาน
