# Phase 38: ONNX Runtime Classical ML Inspector (`TreeEnsemble`, `Linear`, `SVM` Operators)

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
พัฒนา Adapter สำหรับไฟล์มาตรฐานสากล **`.onnx` (Open Neural Network Exchange)** ที่ถูกแปลงมาจากโมเดล Classical Machine Learning (ผ่าน `skl2onnx`, `onnxmltools`, หรือ ` hummingbird`) ซึ่งใช้ชุดคำสั่งในโดเมน **`ai.onnx.ml`** เพื่อให้ระบบสามารถวิเคราะห์โครงสร้างภายในของไฟล์ `.onnx` และเลือกเรนเดอร์ฉาก 3D (ระนาบแบน Linear, ขั้นบันไดต้นไม้ Tree Voxel, หรือพื้นผิวโค้ง SVM) ได้ตรงตามอัลกอริทึมที่แท้จริงข้างในไฟล์ `.onnx`

### ขอบเขตการพัฒนา (Scope)
1. **`ONNXClassicalAdapter` (`adapters/neural_onnx/onnx_classical.py`)**: โหลดกราฟด้วย `onnx` เพื่ออ่านโครงสร้างโหนด และรันการทำนายความเร็วสูงด้วย `onnxruntime.InferenceSession`
2. **`ai.onnx.ml` Operator Attribute Decompiler**:
   - แกะ `TreeEnsembleClassifier` / `TreeEnsembleRegressor`: สกัดโครงสร้างต้นไม้ จุดตัด (`nodes_values`) และรหัสใบไม้จาก Attributes ในกราฟ ONNX
   - แกะ `LinearClassifier` / `LinearRegressor`: สกัดเวกเตอร์สัมประสิทธิ์ (`coefficients`) และจุดตัดแกน (`intercepts`)
   - แกะ `SVMClassifier` / `SVMRegressor`: สกัด `support_vectors`, `coefficients`, และ `kernel_params` ($\gamma, c_0, d$)
3. **Single Tensor vs Multi-Input Tensor Resolver**: รองรับทั้งไฟล์ ONNX ที่รับอินพุตเป็นก้อนเดียว `float_input: [None, F]` และแบบแยกชื่อคอลัมน์หลายอินพุต
4. **ONNX `ZipMap` Sequence-of-Maps Unpacker**: แปลงผลลัพธ์ความน่าจะเป็นรูปแบบ `seq(map(int64, float))` หรือ `seq(map(string, float))` ของ `skl2onnx` ให้เป็นอาเรย์ `np.ndarray` ขนาด `(N, C)` (`float32`) มาตรฐาน

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 การสแกนและถอดรหัสพารามิเตอร์จาก `ai.onnx.ml` Graph Nodes
เมื่อโหลดไฟล์ `.onnx` ด้วย `proto = onnx.load(file_path)` ระบบจะวนสำรวจ `proto.graph.node` เพื่อหาโหนดหลักในโดเมน `ai.onnx.ml`:

1. **กรณี `LinearClassifier` / `LinearRegressor`**:
   - ดึงแอตทริบิวต์ `coefs = get_attr_floats(node, "coefficients")` และ `intercepts = get_attr_floats(node, "intercepts")`
   - จัดรูปเมทริกซ์น้ำหนัก $\mathbf{W} \in \mathbb{R}^{C \times F}$ และ $\mathbf{b} \in \mathbb{R}^C$
   - คำนวณอิทธิพลรายฟีเจอร์ของแถว $\mathbf{x}_i$ ได้ทันทีด้วยสมการแม่นยำ 100%: $\phi_j(\mathbf{x}_i) = w_{c^*, j} \cdot x_{i, j}$ และเรนเดอร์ฉาก **3D Hyperplane / Sigmoid Surface**

2. **กรณี `TreeEnsembleClassifier` / `TreeEnsembleRegressor` (รองรับทั้ง OpSet $\le 3$ และ `TreeEnsemble` OpSet 5)**:
   - โครงสร้างต้นไม้ทั้งหมดถูกเก็บเป็น Flattened Parallel Arrays ใน Attributes ของโหนด:
     - `nodes_treeids`: รหัสต้นไม้ของแต่ละโหนดย่อย
     - `nodes_nodeids`: รหัสโหนดภายในต้นไม้นั้น
     - `nodes_featureids`: ดัชนีฟีเจอร์ที่ใช้ตัด ($f_k$)
     - `nodes_values`: ค่าจุดตัด ($\tau_k$) หรือค่าใบไม้
     - `nodes_modes`: โหมดการเปรียบเทียบ (`BRANCH_LEQ` คือ $x_{f_k} \le \tau_k$, `BRANCH_LT`, `BRANCH_GTE`, หรือ `LEAF`)
     - `nodes_truenodeids` / `nodes_falsenodeids`: ตัวชี้ไปยังโหนดลูกซ้าย/ขวา
     - `target_ids` / `target_weights` (หรือ `class_ids` / `class_weights`): น้ำหนักคะแนนของแต่ละใบไม้
   - ระบบสร้าง C-style NumPy Lookup Table จากอาเรย์เหล่านี้เพื่อดึงเส้นทาง `decision_path` และจุดตัดแกน $X, Y$ สำหรับวาด **3D Voxel Stepped Terraces**!

3. **กรณี `SVMClassifier` / `SVMRegressor`**:
   - ดึง `support_vectors` มาจัดรูปเป็นเมทริกซ์ $\mathbf{SV} \in \mathbb{R}^{n_{\text{sv}} \times F}$ ดึง `kernel_type` (`RBF`, `POLY`, `LINEAR`, `SIGMOID`) และ `kernel_params = [gamma, coef0, degree]`
   - วาดเสาแสง 3D Support Vector Pillars และพื้นผิวโค้ง Kernel Saddle เหมือนโมเดล SVM ต้นฉบับทุกประการ

### 2.2 การจัดการ Single Tensor vs Multi-Input Tensors ใน `InferenceSession`
โมเดลที่แปลงด้วย `skl2onnx` มีรูปแบบ Input Signature 2 แบบหลักที่พบในงานจริง:
- **แบบที่ 1 (Single 2D Tensor Input)**: `sess.get_inputs()` มีเพียง 1 รายการ เช่น `name="float_input", shape=[None, 12], type="tensor(float)"`
  - ระบบแตกชื่อฟีเจอร์จำลองเป็น `["feature_0", "feature_1", ..., "feature_11"]` (หรืออ่านจาก `proto.metadata_props` หากมี) และสร้าง feed dict เป็น `{"float_input": X_chunk.astype(np.float32)}`
- **แบบที่ 2 (Multi-Column Named Inputs — มาจาก `DataFrame` Pipeline)**: `sess.get_inputs()` มี $F$ รายการ เช่น `[{"name": "age", "shape": [None, 1], "type": "tensor(float)"}, {"name": "city", "shape": [None, 1], "type": "tensor(string)"}]`
  - ระบบใช้ `inp.name` เป็นรายชื่อฟีเจอร์โดยตรง และสร้าง feed dict แยกรายคอลัมน์ `{col: df_chunk[col].to_numpy().reshape(-1, 1).astype(onnx_to_numpy_dtype(inp.type))}`

### 2.3 การแกะผลลัพธ์ `ZipMap` (`seq(map(int64/string, float))`)
ใน `skl2onnx` เอาต์พุตตัวที่ 2 (`output_probability`) มักผ่านตัวดำเนินการ `ZipMap` ซึ่งคืนค่าเป็น `list[dict[int | str, float]]` แทนที่จะเป็น `np.ndarray`
- **กลไกการแกะความเร็วสูง**:
  1. หากตรวจพบโหนดปลายทาง `op_type == "ZipMap"` ระบบจะตรวจสอบก่อนว่าสามารถตัดโหนด `ZipMap` ออกจาก `proto.graph.output` ในหน่วยความจำเพื่ออ่าน Tensor `probabilities` (`tensor(float)` ขนาด `[N, C]`) ก่อนเข้า `ZipMap` โดยตรงได้หรือไม่ (ช่วยเพิ่มความเร็วขึ้น 15 เท่า!)
  2. หากกราฟล็อกโครงสร้างไว้ ระบบมี Fallback Unpacker แปลง `list[dict]` เป็น `np.float32` ขนาด `(N, C)` ตามลำดับ `classlabels_int64s` หรือ `classlabels_strings`

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)
1. **ONNX Graph Architecture & Domain Badge**:
   - แสดงป้ายกำกับอัจฉริยะด้านบน เช่น `[ONNX ai.onnx.ml | Detected: TreeEnsembleClassifier (100 Trees, OpSet 18)]` เพื่อให้ผู้ใช้รู้ทันทีว่าข้างในไฟล์ `.onnx` คือโมเดลตระกูลใด
2. **Adaptive 3D Stage Renderer (Auto-Morphing 3D View)**:
   - เปลี่ยนฉาก 3D อัตโนมัติตาม Operator ที่ตรวจพบ:
     - `TreeEnsemble*` $\to$ 3D Stepped Voxel Terraces + Split Planes
     - `Linear*` $\to$ 3D Tilted Hyperplane / Sigmoid Wave
     - `SVM*` $\to$ 3D RBF Kernel Saddle + Support Vector Pillars
3. **ONNX Input Signature & Tensor Type Inspector Modal**:
   - ตารางแสดงรายการ `session.get_inputs()` และ `session.get_outputs()` พร้อม Shape (`[batch_size, 14]`), ชนิดข้อมูล (`tensor(float)`), และสถานะการตัด `ZipMap`
4. **ONNX Operator Pipeline Mini-Graph**:
   - แผนผังบล็อกแสดงลำดับโหนดในกราฟ ONNX เช่น `[Input: float_input] ➔ [Scaler (ai.onnx.ml)] ➔ [TreeEnsembleClassifier] ➔ [Cast / ZipMap]`

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

### UX Flow
1. ผู้ใช้อัปโหลดไฟล์ `fraud_rf.onnx` คู่กับ `transactions.csv`
2. ระบบสแกนกราฟ ONNX $\to$ พบโหนด `Scaler` ต่อด้วย `TreeEnsembleClassifier` (50 ต้น) และ `ZipMap`
3. ระบบตัด `ZipMap` ออกในหน่วยความจำเพื่อดึง Tensor ความน่าจะเป็นโดยตรง และสกัดจุดตัดต้นไม้ทั้ง 50 ต้นจากแอตทริบิวต์ `nodes_values`
4. หน้าจอ 3D Studio เรนเดอร์ภูมิประเทศขั้นบันได 3D Voxel Terraces ของ Random Forest จากไฟล์ `.onnx` พร้อมเล่น VCR 60 FPS ได้เหมือนไฟล์ `.pkl` ต้นฉบับทุกประการ

### ASCII Wireframe
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer]  Model: fraud_rf.onnx (ONNX Runtime | ai.onnx.ml.TreeEnsembleClassifier)        |
| Execution Provider: [CPUExecutionProvider (AVX2)]   Graph Optimization: [ZipMap Bypass Active]   |
+------------------------------------------------------------------+-------------------------------+
| 3D ADAPTIVE ONNX STAGE (Mode: TreeEnsemble Voxel Terraces)       | ONNX GRAPH & TELEMETRY HUD    |
|                                                                  |                               |
|       Z (Fraud Prob)                                             | [ONNX Graph Node Chain]       |
|       ^       +---------+---------+                              | 1. Input: float_input [N, 10] |
|   1.0 |      /         /   (*)   /| <-- Row #77 Sphere           | 2. Op: ai.onnx.ml::Scaler     |
|       |     +---------+---------+ |     (Prob = 0.942)           | 3. Op: TreeEnsembleClassifier |
|   0.5 |     |         |         | +                              |    - Trees: 50 | Nodes: 1,420 |
|       |    /         /         /                                 |    - Mode : BRANCH_LEQ        |
|   0.0 +---+---------+---------+---> X: feature_2 (amount)        | 4. Out: probabilities [N, 2]  |
|      /                                                           |                               |
|     v Y: feature_5 (tx_velocity)                                 | [TreeEnsemble Leaf Trace]     |
|                                                                  | Active Leaves Hit:            |
| [x] Render Extracted ONNX Split Planes (nodes_values)            | Tree #0: Node 14 (w = +0.018) |
| [x] Perturbation / Margin Sensitivity Waterfall                  | Tree #1: Node 29 (w = +0.020) |
+------------------------------------------------------------------+-------------------------------+
| [|<] [<] [ || Pause ] [>] [>|]    Frame: 0077 / 5000   [==*====================]  Speed: [1.0x v]|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD แยกรายชั้นพร้อมชื่อไฟล์และ Code Signatures

### 5.1 Backend (`backend/app/domains/model_engine/`)
- **`adapters/neural_onnx/onnx_classical.py`**
  ```python
  from pathlib import Path
  import numpy as np
  import pandas as pd
  import onnx
  import onnxruntime as ort
  from app.domains.model_engine.domain.ports import BaseModelAdapter
  from app.domains.model_engine.domain.entities import (
      ModelSchema, PredictionBatchResult, ONNXClassicalMetadata
  )

  class ONNXClassicalAdapter(BaseModelAdapter):
      def __init__(self, model_path: Path, onnx_proto: onnx.ModelProto) -> None: ...

      def _inspect_ml_operators(self) -> ONNXClassicalMetadata:
          """Scans graph.node for TreeEnsemble*, Linear*, SVM*, Scaler, and ZipMap."""
          ...

      def _build_optimized_session(self) -> ort.InferenceSession:
          """Bypasses ZipMap if present to expose raw float probability tensor directly."""
          ...

      def _build_feed_dict(self, df_chunk: pd.DataFrame) -> dict[str, np.ndarray]:
          """Handles both single 2D tensor input and multi-column named tensor inputs."""
          ...

      def decompile_tree_ensemble_splits(self, feat_x_idx: int, feat_y_idx: int) -> dict[str, list[float]]:
          """Extracts split thresholds from nodes_featureids and nodes_values attributes."""
          ...

      def predict_batch(self, df_chunk: pd.DataFrame) -> PredictionBatchResult: ...
  ```

### 5.2 Frontend (`frontend/src/domains/simulation/`)
- **`components/stage3d/renderers/ONNXClassicalStageRouter.tsx`**: สลับเรนเดอร์ระหว่าง `LinearHyperplane3D`, `DecisionTreeVoxel3D`, และ `SVMKernelSurface3D` ตาม `onnxMetadata.detectedOperatorFamily`
- **`components/hud/ONNXGraphInspectorCard.tsx`**: แสดงผังโหนด `ai.onnx.ml` และตาราง Input/Output Tensors

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)
1. **In-Memory `ZipMap` Removal Before Session Creation**:
   - การปล่อยให้ `onnxruntime` สร้าง Python `list[dict[int, float]]` สำหรับ `2,500` แถวต่อ Chunk จะสร้าง Python Objects นับหมื่นตัวและกิน RAM สูงกว่า `np.ndarray` ถึง 10 เท่า ระบบจึงตรวจสอบว่าหากเอาต์พุตมาจากโหนด `ZipMap` จะเปลี่ยน `graph.output` ให้ชี้ไปยัง Input Tensor ของโหนด `ZipMap` (ซึ่งเป็น `float32[N, C]` ล้วน) ก่อนส่งเข้า `ort.InferenceSession(proto.SerializeToString())`
2. **Contiguous Array Memory Enforcement (`np.ascontiguousarray`)**:
   - C++ Binding ของ `onnxruntime` ต้องการหน่วยความจำแบบ C-Contiguous หากส่ง Slice ของ DataFrame ที่ไม่ต่อเนื่องไป `onnxruntime` จะแอบก็อปปี้หน่วยความจำซ้ำข้างใน ดังนั้นต้องบังคับ `np.ascontiguousarray(arr, dtype=np.float32)` ก่อนใส่ใน `feed_dict` เสมอ
3. **Session Thread Pool Tuning**:
   - ตั้งค่า `sess_options = ort.SessionOptions()` โดยกำหนด `sess_options.intra_op_num_threads = max(1, cpu_cores - 1)` และ `sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL` เพื่อป้องกันการแย่ง Thread กับ Worker ของระบบ

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| กรณีปัญหา (Edge Case) | สาเหตุเชิงเทคนิค | แนวทางป้องกันและแก้ไขในระบบ (Mitigation Strategy) |
| :--- | :--- | :--- |
| **1. `ZipMap` คืนค่า `list[dict]` แทน `ndarray`** | ค่าเริ่มต้นของ `skl2onnx` เปิด `zipmap=True` เสมอสำหรับ Classifier | ใช้เทคนิค Graph Output Rewiring อ่าน Tensor ความน่าจะเป็นก่อนเข้า `ZipMap` พร้อมมีฟังก์ชันแปลง `list[dict]` เป็น Fallback |
| **2. Single 2D Input vs Multi-Input Tensors** | โมเดลจาก NumPy ใช้ Input เดียว `[None, F]` แต่โมเดลจาก DataFrame ใช้ $F$ Inputs `[None, 1]` | ตรวจสอบ `len(session.get_inputs())` หากเท่ากับ 1 และ `shape[1] > 1` ให้รวมคอลัมน์เป็น 2D Matrix แต่ถ้ามีหลาย Inputs ให้ส่งแยกตามชื่อคอลัมน์ |
| **3. ชนิดข้อมูล `tensor(double)` vs `tensor(float)`** | บางโมเดลถูก Export ด้วย `DoubleTensorType` (`float64`) หากส่ง `float32` จะเกิด `InvalidArgument` | สร้างแผนที่แปลงชนิด `ONNX_TYPE_TO_NP = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(int64)": np.int64, "tensor(string)": np.object_}` ตาม `inp.type` จริงของแต่ละอินพุต |
| **4. `TreeEnsemble` OpSet 5 (ONNX 1.16+) เปลี่ยนชื่อ Attribute** | ใน `ai.onnx.ml` OpSet 5 โหนดรวมเป็น `TreeEnsemble` และเปลี่ยนโครงสร้างแอตทริบิวต์บางส่วน | รองรับการอ่านแอตทริบิวต์ทั้งมาตรฐาน OpSet 1–3 (`nodes_featureids`, `nodes_values`) และ OpSet 5 (`nodes_splits`, `leaf_weights`) |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] โหลดไฟล์ `.onnx` ที่แปลงมาจาก Logistic Regression, Linear Regression, SVM, Decision Tree, Random Forest, XGBoost, และ LightGBM ได้ครบถ้วน
- [ ] ระบุตระกูลอัลกอริทึมจากโหนด `ai.onnx.ml` และเลือกแสดงฉาก 3D (ระนาบแบน / ขั้นบันไดต้นไม้ / พื้นผิวโค้ง SVM) ได้ตรงตามชนิดโมเดลจริง
- [ ] รองรับทั้งไฟล์ `.onnx` แบบ Single 2D Input Tensor และแบบ Multi-Column Named Inputs โดยจับคู่คอลัมน์กับ CSV ได้ถูกต้อง
- [ ] จัดการโหนด `ZipMap` ได้อย่างสมบูรณ์และคืนค่าความน่าจะเป็นขนาด `(N, C)` (`float32`) โดยไม่เกิด Memory Bloat
- [ ] แสดงข้อมูลโครงสร้างกราฟ ONNX และอิทธิพลรายฟีเจอร์ในแผง Explainability HUD ได้อย่างแม่นยำ
