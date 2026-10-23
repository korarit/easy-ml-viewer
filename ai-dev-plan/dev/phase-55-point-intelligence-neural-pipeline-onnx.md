# Phase 55: 2D Point Intelligence Integration for Neural Networks, Pipelines & Universal ONNX (MLP, Scaled Pipelines, ONNX Latents)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เชื่อมต่อระบบ **2D Point Intelligence (A, B, C, D)** เข้ากับกลุ่มโมเดลโครงข่ายประสาทเทียม, ไปป์ไลน์การแปลงข้อมูล, และโมเดลมาตรฐานสากล ONNX (**Neural Networks, Pipelines & Universal ONNX**) ได้แก่ **MLP, Sklearn Pipeline, ONNX Classical ML**, และ **ONNX Deep Learning Latent Manifold** เพื่อเผยพฤติกรรมในปริภูมิแฝง (Latent Space), ปัญหาการทำนายมั่นใจเกินจริงในพื้นที่ว่าง (Data Desert / Overconfident Extrapolation), และเวกเตอร์เกรเดียนต์ของ Loss Function ในมุมมองระนาบ 2 มิติ

### ขอบเขตโมเดลที่รองรับในเฟสนี้
- **`MLPClassifier` & `MLPRegressor`** (Multi-Layer Perceptron)
- **Scikit-Learn `Pipeline` & `ColumnTransformer`** (StandardScaler, RobustScaler, MinMaxScaler)
- **ONNX Classical Runtime** (Linear, TreeEnsemble, SVM Operators)
- **ONNX Deep Learning Models** (Intermediate Feature Extractors & Embeddings)

---

## 2. กลไกคณิตศาสตร์และการประยุกต์ใช้ Point Intelligence เชิงลึก (Mathematical & Geometric Integration)

### 2.1 กลไก A: Dual-Space Trajectory (Raw Input vs Scaled Latent Comet Trail)
เมื่อโมเดลทำงานผ่าน Pipeline (เช่น `StandardScaler` $\to$ `MLP`):
- **Space-Swapping Comet Trail**:
  - ผู้ใช้สามารถสลับมุมมองระหว่าง **Raw Feature Space** (หน่วยจริง เช่น บาท, ปี, กิโลกรัม) และ **Transformed Scaled Space** ($z = \frac{x - \mu}{\sigma}$)
  - เส้นหาง Comet Trail จะคำนวณการแปลงพิกัดแบบ 1-to-1 ทำให้เห็นความยืดหดของระยะทาง (Metric Distortion) ระหว่างสองปริภูมิ
- **Latent Geodesic Path (สำหรับ Deep ONNX)**:
  - ในโมเดล ONNX Deep Learning เส้นทาง Comet Trail จะสะท้อนระยะทางบน Latent Manifold เพื่อดูว่าข้อมูลวิ่งผ่านคลัสเตอร์ใดในเชิงเรขาคณิต

### 2.2 กลไก B: Softmax Logit Gap Dual-Ring
1. **Logit Margin Spread (สำหรับ MLP & ONNX Classifier)**:
   - ให้ $z_{(1)}$ และ $z_{(2)}$ เป็น Logits อันดับที่ 1 และ 2 ตามลำดับ ก่อนผ่าน Softmax:
   - คำนวณ Logit Gap:
     $$\Delta z = z_{(1)} - z_{(2)}$$
   - วงแหวนรอบนอก (Outer Ring):
     - หาก $\Delta z > 3.0$ (Saturated Confident): วงแหวนทึบสีคมชัด (โมเดลฟันธงเด็ดขาด)
     - หาก $\Delta z < 0.5$ (Tempting Tie / High Temperature): วงแหวนกระจายแสงแบบ Neon Diffuse เพื่อเตือนว่าโมเดลลังเลระหว่างสองคลาสสูงสุด
2. **Hidden Layer Activation Halo**:
   - สำหรับ MLP แสดงสีของวงแหวนตาม Activation Value ของ Neuron ใน Hidden Layer ชั้นสุดท้าย

### 2.3 กลไก C: Input Loss Gradient Spokes ($\nabla_{\mathbf{x}} \mathcal{L}$)
- ใน Neural Networks เราสามารถหาอนุพันธ์ของ Loss เทียบกับ Input Features ได้:
  $$\mathbf{g}_i = \nabla_{(x_1, x_2)} \mathcal{L}\left(f(\mathbf{x}_i), y_i\right)$$
- **Adversarial Gradient Spoke**:
  - ก้าน Spoke จะชี้ไปในทิศทาง $\mathbf{g}_i$ บ่งบอกว่าถ้าขยับค่าของ $X_1, X_2$ ไปในทิศทางนี้เพียงเล็กน้อย ค่าความผิดพลาด (Loss) จะพุ่งสูงขึ้นเร็วที่สุด (Local Sensitivity / Adversarial Fragility)

### 2.4 กลไก D: Data Desert & Overconfident Extrapolation Hexbin
โมเดล Neural Networks มักมีข้อเสียคือทำนายด้วยความมั่นใจสูงลิ่วแม้ในบริเวณที่ไม่มีข้อมูลฝึกสอน (Data Desert):
- **Extrapolation Hazard Detection**:
  - เปรียบเทียบระหว่างความหนาแน่นใน Hexbin กับความมั่นใจของโครงข่ายประสาทเทียม:
    $$\text{Hazard}(u, v) = \mathbb{I}\left(\text{Count}(u, v) == 0\right) \times \max_c P(y=c | \mathbf{x}_c(u, v))$$
  - ช่องรังผึ้งที่ตกอยู่ในสภาวะ $\text{Hazard} > 0.85$ จะแสดงไอคอนโครงกระดูก/ป้ายเตือน $\triangle \text{ Data Desert}$ เพื่อเตือนว่าบริเวณนี้โมเดลทำนายแบบ Overconfident โดยไม่มีข้อมูลจริง

---

## 3. สถาปัตยกรรมคอมโพเนนต์ (`frontend/src/domains/stage3d/components/surfaces_2d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D NEURAL & ONNX POINT INTELLIGENCE OVERLAYS                                                     |
+--------------------------+-----------------------------------------------------------------------+
| Component                | บทบาทเฉพาะตัวสำหรับ Neural, Pipeline & ONNX Family                    |
+--------------------------+-----------------------------------------------------------------------+
| DualSpaceCometTrail2D    | หาง Comet Trail สลับการแสดงผลระหว่าง Raw Units และ Scaled Units       |
| SoftmaxLogitGapRing2D    | วงแหวน Dual-Ring แสดงส่วนต่าง Logit Gap (Delta z) และ Softmax Spread  |
| LossGradientSpoke2D      | ก้านแสดงทิศทาง Input Gradient (nabla_x L) และความอ่อนไหวต่อสิ่งรบกวน  |
| DataDesertHexbin2D       | กริดรังผึ้งตรวจจับโซนที่ไม่มีข้อมูลแต่โมเดลมั่นใจเกินจริง (Extrapolate)|
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: MLPClassifier (Pipeline Scaled)     [Trails (T)] [DualRing (R)]        |
| Space: [● Scaled (Z)] [Raw Units]                              [Errors (E)] [Hexbin (H)]          |
+--------------------------------------------------------------------------------------------------+
| Y: Feature_2 (Z-score: -3..+3)                                                                   |
|  +3.0 |              [ DATA DESERT ZONE: Count = 0, P_fraud = 94% (Hazard!) ]                     |
|       |                ⬡   ⬡   ⬡   ⬡  (Empty Hexbins with Overconfident Glow)                |
|  +1.5 |                                                                                          |
|       |                     ● [Misclassified Input]                                              |
|   0.0 |                     |                                                                    |
|       |                     +--> Adversarial Spoke (nabla_x Loss = [0.82, -0.41])                |
|  -1.5 |                                                                                          |
|       |        ~~~~~~~●-●-◉ [Active Point: Logit Gap Delta z = 0.18 (Diffuse Halo)]              |
|  -3.0 |                                                                                          |
|       +------------------------------------------------------------------------------------------+
|      -3.0            -1.5             0.0            +1.5            +3.0 (X: 1, Z-score)        |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Definition of Done (DoD)
1. สวิตช์สลับ Raw Space vs Scaled Space ปรับพิกัดจุดและหาง Comet Trail ได้อย่างถูกต้อง
2. วงแหวน Softmax Logit Gap คำนวณ $\Delta z$ ตรงตามเอาต์พุตของโมเดลก่อนผ่าน Sigmoid/Softmax
3. ช่องรังผึ้งที่ไม่มีข้อมูลแต่ความน่าจะเป็นสูงถูกติดป้าย Data Desert อย่างแม่นยำ
4. ทดสอบความเร็วในการอนุมาน Input Gradient และเรนเดอร์คงที่ที่ $60$ FPS

