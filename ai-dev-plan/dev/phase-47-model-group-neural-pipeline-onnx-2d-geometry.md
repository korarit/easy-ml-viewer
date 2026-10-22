# Phase 47: 2D Decision Geometries & Visualizers for Neural Networks, Pipelines & Universal ONNX (MLP, Pipeline, ONNX ML & Deep)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
พัฒนาชุดเรขาคณิตและการแสดงผล 2 มิติเฉพาะทางสำหรับกลุ่มโมเดล **โครงข่ายประสาทเทียม ไพป์ไลน์ และโมเดลมาตรฐานสากล ONNX (Neural Networks, Pipelines & Universal ONNX)** ได้แก่ **Multi-Layer Perceptron (MLP)**, **Scikit-Learn Pipeline & ColumnTransformer**, **ONNX Classical ML**, และ **ONNX Deep Learning** โดยในโหมด 2 มิติ ผู้ใช้จะได้เห็นกลไกเบื้องหลังที่โมเดล Deep Learning และโมเดลที่ผ่านการแปลงสเกลข้อมูลสร้างขึ้น:
- **MLP First-Layer Neuron Hyperplanes**: เส้นตรงขอบเขตเปิด/ปิดการทำงานของแต่ละ Hidden Neuron ในชั้นแรก ($h_k = 0$) ทำให้เห็นชัดเจนว่าโครงข่ายประสาทเทียมนำเส้นตรงหลายเส้นมาประกอบกันเป็นขอบเขตโค้งมนได้อย่างไร
- **Pipeline Space Transformation Toggle**: สวิตช์สลับมุมมอง `[Raw Feature Space]` $\leftrightarrow$ `[Scaled Space]` เพื่อดูผลกระทบของการทำ Standard/MinMax Scaler ต่อเส้นแบ่งเขตใน 2D
- **ONNX Deep Learning 2D Latent Projection Plane**: การฉายเวกเตอร์แฝง (Penultimate Latent Embedding) ลงบนระนาบ 2D ผ่าน 2D PCA / SVD เพื่อดูกลุ่มคลัสเตอร์การตัดสินใจในอวกาศแฝง

### ขอบเขตโมเดลและเครื่องยนต์ที่รองรับ (In-Scope Models)
1. **`MLPClassifier` & `MLPRegressor`** (`sklearn.neural_network`)
2. **`Pipeline` & `ColumnTransformer`** (`sklearn.pipeline`, `compose`)
3. **`ONNX Classical ML`** (`ai.onnx.ml`: `TreeEnsemble`, `LinearClassifier`, `SVMClassifier`)
4. **`ONNX Deep Learning`** (`onnxruntime`: Fully Connected, Conv, Embeddings)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน 2 มิติ (Mathematical Formulation & 2D Geometry)

### 2.1 เรขาคณิตของ MLP Neural Network (Hidden Neuron Boundary Cuts)
1. **สมการเส้นตรงของแต่ละเซลล์ประสาทใน Hidden Layer 1**:
   กำหนดให้เซลล์ประสาทตัวที่ $k$ ในชั้นแรกมีน้ำหนัก $w_{k,1}^{(1)}, w_{k,2}^{(1)}$ และค่าไบแอสประสิทธิผล $b_{k, \text{eff}}^{(1)}$:
   $$z_k^{(1)} = w_{k,1}^{(1)} x_1 + w_{k,2}^{(1)} x_2 + b_{k, \text{eff}}^{(1)}$$
   - สำหรับ Activation Function แบบ ReLU ($h_k = \max(0, z_k)$):
   - เส้นตรง $z_k^{(1)} = 0$ คือ **เส้นแบ่งเขตเปิด/ปิด (Switching Hyperplane)** ของเซลล์ประสาทตัวที่ $k$
   - ระบบจะวาดเส้นตรงสีฟ้าบางๆ ของทุก Neuron ในชั้นแรกซ้อนลงบนระนาบ 2D ทำให้ผู้ใช้เห็นว่าเส้นแบ่งเขตสุดท้าย (Non-linear Boundary) ก่อตัวขึ้นจากการพับงอตามแนวเส้นตรงของ Neurons เหล่านี้!
2. **การตรวจจับ Dead ReLU บนระนาบ 2D**:
   - หากเซลล์ประสาท $k$ ใดมีค่า $z_k < 0$ ทั่วทั้งพื้นที่ระนาบ $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$ จะถูกติดป้ายสถานะ **Dead ReLU in this 2D Slice**

### 2.2 Scikit-Learn Pipeline: การบิดระนาบพิกัด (Raw vs Scaled Space)
1. **การแปลงข้อมูลผ่าน Transformer (เช่น `StandardScaler`)**:
   $$x_1' = \frac{x_1 - \mu_1}{\sigma_1}, \quad x_2' = \frac{x_2 - \mu_2}{\sigma_2}$$
2. **Dual-Space Visualization**:
   - ในโหมด `Scaled Space`: ผู้ใช้เห็นพิกัดมาตรฐานของโมเดล (มักอยู่แถว $[-3.0, +3.0]$)
   - ในโหมด `Raw Space`: ระบบแปลงขอบเขตและเส้นแบ่งเขตย้อนกลับด้วย Inverse Transform ทำให้เห็นเส้นแบ่งเขตทับบนมาตราส่วนจริงของข้อมูลต้นฉบับ (เช่น รายได้เป็นบาท, อายุเป็นปี)

### 2.3 ONNX Deep Learning 2D Latent SVD Projection Plane
1. **การดึงเวกเตอร์แฝง (Intermediate Hooking)**:
   ดึงเอาท์พุตของเลเยอร์ก่อนสุดท้าย $\mathbf{z}_t \in \mathbb{R}^K$
2. **การฉายภาพ 2 มิติด้วย Singular Value Decomposition (SVD / PCA 2D)**:
   $$\mathbf{z}_{t, \text{2D}} = \mathbf{V}_2^T (\mathbf{z}_t - \bar{\mathbf{z}})$$
   - วาดจุดข้อมูลในอวกาศแฝง 2D พร้อมเส้นคอนทัวร์ความน่าจะเป็น ทำให้ผู้ใช้ตรวจสอบได้ว่าโมเดลแยกแยะกลุ่มข้อมูลในเลเยอร์ลึกได้ดีเพียงใด

---

## 3. UI Feature List & 2D Overlays ของกลุ่มโมเดล Neural, Pipeline & ONNX

```text
+--------------------------------------------------------------------------------------------------+
| 2D NEURAL & UNIVERSAL OVERLAYS MATRIX                                                            |
+--------------------------+-----------------------------------------------------------------------+
| UI Component             | ความสามารถและองค์ประกอบที่แสดงในโหมด 2D Planar                         |
+--------------------------+-----------------------------------------------------------------------+
| MlpNeuronHyperplanes2D   | • เส้นตรงเปิด/ปิดการทำงานของ Hidden Neurons ในชั้นแรก (Layer 1 Cuts)   |
|                          | • แสดงสถานะ Dead ReLU บนระนาบ พร้อมแถบสถิติ Active Neuron Ratio       |
|                          | • เส้นขอบเขตความน่าจะเป็นแบบ Non-linear Smooth Contour                |
+--------------------------+-----------------------------------------------------------------------+
| PipelineSpaceToggle2D    | • สวิตช์สลับมุมมอง "[● Raw Space] <-> [Scaled Space]"                 |
|                          | • แสดงการบิดสเกลพิกัดและสมการปรับระดับค่าข้อมูล                       |
+--------------------------+-----------------------------------------------------------------------+
| OnnxLatentManifold2D     | • แผนผังจุดข้อมูลในอวกาศแฝง 2D (Latent SVD Projection Plane)          |
|                          | • วงแหวนคลัสเตอร์และเส้นแบ่งเขตใน Latent Space                         |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ 2D Neural & ONNX Group

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: MLPClassifier (Hidden: 64x32, ReLU)          [● 2D Mode] [3D Mode] (V)|
+--------------------------------------------------------------------------------------------------+
| Y: Feature 2                                                     NEURONS: 42 Active / 4 Dead ReLUs|
|   4.0 |         / - - - - - - - - - - - - - \                                                    |
|       |        /   Neuron #1 Line            \                                                   |
|   2.0 | - - - / - - - - - - - - - - - - - - - \ - - - - - - (Neuron #4 Cut: z_4 = 0)             |
|       |      /        (Non-linear Boundary)    \                                                 |
|   0.0 |     /   =============================   \                                                |
|       |    /   /                             \   \                                               |
|  -2.0 |   /   /       [ ZONE 1 ]              \   \ - - - - - (Neuron #12 Cut)                   |
|       |  /   /         ● [Active Row]          \   \                                             |
|  -4.0 +-/---/-----------------------------------\---\-----------------> X: Feature 1            |
|       -4.0        -2.0           0.0            2.0         4.0                                   |
|                                                                                                   |
| MLP OVERLAYS: [x] Non-linear Boundary   [x] First-Layer Neuron Cuts   [x] Highlight Dead ReLUs    |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Signatures)

### 5.1 โครงสร้างไฟล์
- `frontend/src/domains/stage3d/components/surfaces_2d/MlpNeuronHyperplanes2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/PipelineSpaceToggle2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/OnnxLatentManifold2D.tsx`
- `backend/app/domains/model_engine/adapters/neural_onnx/neural_2d_extractor.py`

### 5.2 โค้ดต้นแบบ: `neural_2d_extractor.py`
```python
from typing import List, Dict, Any
import numpy as np

class Neural2DExtractor:
    """สกัดเส้นตรงของเซลล์ประสาทในชั้นแรกของโครงข่าย MLP เพื่อแสดงในระนาบ 2D"""

    @staticmethod
    def extract_mlp_first_layer_hyperplanes(
        mlp_model: Any,
        x_idx: int,
        y_idx: int,
        pinned_features: np.ndarray
    ) -> List[Dict[str, float]]:
        # coefs_[0] มีขนาด (n_features, n_hidden_1)
        w_all = mlp_model.coefs_[0]
        b_all = mlp_model.intercepts_[0]

        w1 = w_all[x_idx, :]
        w2 = w_all[y_idx, :]

        # คำนวณ b_eff จากฟีเจอร์ที่ถูก Pin
        mask = np.ones(w_all.shape[0], dtype=bool)
        mask[x_idx] = False
        mask[y_idx] = False

        other_contrib = np.dot(pinned_features[mask], w_all[mask, :])
        b_eff = b_all + other_contrib

        hyperplanes = []
        for i in range(len(b_eff)):
            hyperplanes.append({
                "neuron_idx": i,
                "w1": float(w1[i]),
                "w2": float(w2[i]),
                "b_eff": float(b_eff[i])
            })
        return hyperplanes
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] เส้นตรงของ Hidden Neurons ในชั้นแรกถูกคำนวณและวาดลงในระนาบ 2D ตรงตามสมการ $z_k^{(1)} = 0$
- [ ] สวิตช์สลับมุมมอง Raw Space $\leftrightarrow$ Scaled Space ของ Pipeline แปลงพิกัดได้อย่างถูกต้อง 100%
- [ ] โมเดล ONNX Deep Learning สามารถดึง Latent Embeddings และฉายลงบน 2D SVD Plane ได้อย่างเสถียร
- [ ] เรนเดอร์บนหน้าจอ 2D ได้ที่ความเร็ว **60 FPS**

