# Phase 36: Multi-Layer Perceptron (MLP) Adapter & 3D Floating Synapse Network + Layer Pulses

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
พัฒนา Adapter สำหรับโครงข่ายประสาทเทียมแบบเชื่อมต่อสมบูรณ์ **`MLPClassifier`** และ **`MLPRegressor`** จาก `sklearn.neural_network` เพื่อเปิดกล่องดำของ Neural Network ให้เห็นการไหลของสัญญาณประสาททีละชั้น (Layer-by-Layer Signal Propagation) ในรูปแบบ **3D Floating Synapse Network (สมองกลโครงข่ายประสาท 3 มิติ)** ควบคู่กับพื้นผิวการตัดสินใจ 3 มิติที่โค้งมน

### ขอบเขตการพัฒนา (Scope)
1. **`MLPAdapter` (`adapters/neural_onnx/mlp_adapter.py`)**: สกัดน้ำหนัก (`coefs_`), ไบแอส (`intercepts_`), จำนวนชั้น (`n_layers_`), และฟังก์ชันกระตุ้น (`activation` และ `out_activation_`)
2. **Exact Layer-by-Layer Forward Pass Engine**: คำนวณสถานะ Pre-activation $Z^{(l)}$ และ Post-activation $A^{(l)}$ ของทุก Hidden Layer สำหรับข้อมูลแต่ละแถว พร้อมคำนวณอิทธิพลรายฟีเจอร์ด้วย **Input $\times$ Gradient / First-Layer Signal Attribution**
3. **Top-32 Neuron & Top-K Synapse Downsampler**: อัลกอริทึมคัดกรองโหนดประสาทและเส้นเชื่อมที่มีพลังงานสูงสุดต่อชั้นเพื่อรักษามาตรฐานความลื่นไหล 60 FPS แม้โมเดลจะมีขนาดหลายร้อยโหนดต่อชั้น
4. **Dead ReLU Neuron Detector**: ตรวจจับโหนดประสาทที่ตายสนิท ($A^{(l)}_j = 0$ ตลอดทั้งชุดข้อมูล) และแสดงเป็นโหนดสีเทาด้านบนฉาก 3D

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 สมการการแพร่สัญญาณไปข้างหน้าทีละชั้น (Layer-by-Layer Forward Propagation)
กำหนดให้โครงข่ายมีทั้งหมด $L$ ชั้น (โดยชั้นที่ $0$ คือ Input Layer $\mathbf{A}^{(0)} = \mathbf{X} \in \mathbb{R}^{N \times d_0}$ และชั้นที่ $L-1$ คือ Output Layer) สำหรับแต่ละชั้น $l \in \{1, 2, \dots, L-1\}$ มีเมทริกซ์น้ำหนัก $\mathbf{W}^{(l)} = \text{coefs\_}[l-1] \in \mathbb{R}^{d_{l-1} \times d_l}$ และเวกเตอร์ไบแอส $\mathbf{b}^{(l)} = \text{intercepts\_}[l-1] \in \mathbb{R}^{d_l}$:

$$
\mathbf{Z}^{(l)} = \mathbf{A}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}, \qquad \mathbf{A}^{(l)} = \sigma_l\left(\mathbf{Z}^{(l)}\right)
$$

โดยที่ฟังก์ชันกระตุ้น $\sigma_l(z)$ สำหรับ Hidden Layers ($1 \le l \le L-2$) คือฟังก์ชันที่ระบุใน `model.activation`:
- **ReLU (`relu`)**: $\sigma(z) = \max(0, z)$
- **Hyperbolic Tangent (`tanh`)**: $\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- **Logistic Sigmoid (`logistic`)**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Identity (`identity`)**: $\sigma(z) = z$

สำหรับชั้นสุดท้าย $l = L-1$ จะใช้ฟังก์ชัน `model.out_activation_` (`softmax` สำหรับ Multiclass, `logistic` สำหรับ Binary/Multilabel, หรือ `identity` สำหรับ Regression)

### 2.2 การคำนวณพลังงานของเส้นใยประสาท (Synapse Signal Flux) และการคัดกรอง Top-32 Neurons
หาก Hidden Layer ที่ $l$ มีจำนวนโหนด $d_l > 32$ (เช่น `hidden_layer_sizes=(256, 128)`) การวาดเส้นเชื่อมทั้งหมด $256 \times 128 = 32,768$ เส้นพร้อมกันจะทำให้ภาพรกและกิน GPU ดังนั้นระบบจึงใช้อัลกอริทึม **Dynamic Top-32 Neuron Downsampling**:
1. คำนวณพลังงานเฉลี่ยและความตื่นตัวของแต่ละโหนด $j$ ในชั้น $l$:
   $$
   E^{(l)}_j = \left|A^{(l)}_j\right| \cdot \max_{k} \left|W^{(l+1)}_{j, k}\right|
   $$
2. คัดเลือกดัชนีโหนดที่มีค่า $E^{(l)}_j$ สูงสุด 32 อันดับแรกของแต่ละชั้น $\mathcal{S}_l = \text{TopIndices}(E^{(l)}, 32)$
3. คำนวณความแรงของกระแสประสาทบนเส้นเชื่อม (Synapse Flux $F^{(l)}_{u \to v}$) เฉพาะคู่โหนด $(u \in \mathcal{S}_{l-1}, v \in \mathcal{S}_l)$:
   $$
   F^{(l)}_{u \to v} = A^{(l-1)}_u \cdot W^{(l)}_{u, v}
   $$
   - ถ้า $F^{(l)}_{u \to v} > 0$ (Excitatory Synapse): แสดงเส้นลำแสงสีฟ้าสว่าง (Cyan Pulse)
   - ถ้า $F^{(l)}_{u \to v} < 0$ (Inhibitory Synapse): แสดงเส้นลำแสงสีส้มแดง (Crimson Inhibition Pulse)

### 2.3 การตรวจจับโหนดประสาทตายใน ReLU (Dead ReLU Detection)
เมื่อ `model.activation == "relu"` ระบบจะตรวจสอบสถิติตลอดทั้ง Chunk ขนาด $N$ แถว:
$$
\text{DeadRatio}^{(l)}_j = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(A^{(l)}_{i, j} \le 0\right)
$$
โหนดที่มี $\text{DeadRatio}^{(l)}_j = 1.0$ (มีค่าเป็น `0.0` ในทุกแถวข้อมูล 100%) จะถูกทำเครื่องหมายเป็น **`Dead Neuron`** เพื่อแจ้งเตือนให้นักพัฒนาทราบว่าโครงข่ายมีปัญหา Dying ReLU จาก Learning Rate หรือการตั้งค่า Initialization

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)
1. **Dual 3D View Mode Switcher (`[3D Synapse Brain View]` / `[3D Decision Surface View]` / `[Split Dual View]`)**:
   - **Synapse Brain View**: แสดงชั้นของโหนดประสาทเป็นทรงกลมเรืองแสง (`InstancedMesh`) ลอยเรียงตามแกน $X$ (จาก Input $\to$ Hidden 1 $\to$ Hidden 2 $\to$ Output) พร้อมเส้นใยประสาทที่ปล่อยคลื่นแสงวิ่งตามจังหวะ VCR Scrubber
   - **Decision Surface View**: แสดงพื้นผิวการตัดสินใจ 3 มิติที่โค้งมนพร้อมลูกบอลข้อมูลวิ่งบนพื้นผิว
2. **Dead ReLU Diagnostic Banner & Hollow Grey Nodes**:
   - โหนดที่เป็น Dead ReLU จะแสดงเป็นทรงกลมสีเทาด้านมีวงแหวนประสีเหลืองล้อมรอบ พร้อมป้ายสรุปด้านบน เช่น `"Layer 1: 14 / 128 Dead ReLU Neurons (10.9%)"`
3. **Interactive Neuron Hover & Synapse Spotlight**:
   - เมื่อเอาเมาส์ชี้ที่โหนดใดในชั้น Hidden Layer ระบบจะหรี่แสงเส้นอื่นลง แล้วไฮไลต์เฉพาะเส้น Input ที่ส่งพลังงานเข้าโหนดนั้นและเส้น Output ที่ออกจากโหนดนั้น พร้อมแสดงสมการ $\sigma(\sum a_i w_{ij} + b_j)$
4. **First-Layer Effective Weight Attribution Waterfall**:
   - คำนวณผลคูณเมทริกซ์สะสมผ่านชั้นที่เปิดใช้งาน (Active Linearized Path $\mathbf{g}_i = \prod_{l=1}^{L-1} \text{diag}(\sigma'_l(\mathbf{Z}^{(l)}_i)) \mathbf{W}^{(l)}$) เพื่อแสดงกราฟ Waterfall ว่าฟีเจอร์ต้นทางตัวใดผลักคำตอบขึ้นหรือลงในแถวนั้น

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

### UX Flow
1. ผู้ใช้อัปโหลดไฟล์ `churn_mlp.joblib` (`MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu')`)
2. ระบบสกัดโครงสร้าง `[Input(10) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(2, Softmax)]`
3. เมื่อกด Play ใน VCR Studio ผู้ใช้จะเห็นพลังงานแสงวิ่งจากโหนดฟีเจอร์ด้านซ้าย ทะลุผ่านชั้น Hidden 1 และ Hidden 2 ไปยังโหนด Output ด้านขวาแบบเรียลไทม์ 60 FPS
4. ผู้ใช้คลิกปุ่ม `[Highlight Dead Neurons]` ระบบจะชี้ให้เห็นโหนดที่ตายสนิทในชั้นที่ 1 จำนวน 5 โหนดทันที

### ASCII Wireframe
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer]  Model: churn_mlp.joblib (MLPClassifier | Layers: 10 -> 64 -> 32 -> 2 | ReLU)   |
| View Mode: (o) 3D Synapse Brain + Surface Split   ( ) 3D Brain Only   ( ) 3D Surface Only        |
+-----------------------------------+------------------------------+-------------------------------+
| 3D FLOATING SYNAPSE NETWORK       | 3D NON-LINEAR SURFACE        | NEURAL LAYER TELEMETRY        |
|                                   |                              |                               |
|  In(10)   H1(Top32/64)  H2(32) Out|      Z (Prob)                | Row #108 Output: 0.914 (Churn)|
|   (O)======>(*)-------->(*)       |      ^    _.-~-._            |                               |
|    \       /  \        /   \      |  1.0 |  /   (*)   \          | [Layer Activation Health]     |
|   (o)---->(x)  \====>(*)===>(O)   |      | /  Row #108 \         | Layer 1 (64 ReLU):            |
|    /       ^                 /    |  0.0 +----------------> X    | - Active Neurons : 59 / 64    |
|   (O)======+====>(o)------->+     |     /                        | - Dead ReLU (0.0):  5 / 64 [!] |
|                  (x) <--Dead ReLU |    v Y                       | - Max Activation : +4.82      |
|                                   |                              |                               |
| Legend: === Excitatory (+Flux)    | [x] Show Top-32 Active Nodes | [Gradient x Input Waterfall]  |
|         --- Inhibitory (-Flux)    | [x] Pulse Animation Speed 1x | tenure      : +0.48 [+++++++] |
|         (x) Dead Neuron (0.0)     | Threshold Filter: [0.15 ===] | monthly_chg : +0.31 [+++++  ] |
+-----------------------------------+------------------------------+-------------------------------+
| [|<] [<] [ > Play ] [>] [>|]    Frame: 0108 / 2000   [==========*=============]   Speed: [1.0x v]|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD แยกรายชั้นพร้อมชื่อไฟล์และ Code Signatures

### 5.1 Backend (`backend/app/domains/model_engine/`)
- **`adapters/neural_onnx/mlp_adapter.py`**
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.neural_network import MLPClassifier, MLPRegressor
  from app.domains.model_engine.domain.ports import BaseModelAdapter
  from app.domains.model_engine.domain.entities import (
      ModelSchema, PredictionBatchResult, NeuralForwardTrace, LayerTelemetry
  )

  class MLPAdapter(BaseModelAdapter):
      MAX_RENDERED_NEURONS_PER_LAYER: int = 32
      MAX_SYNAPSES_PER_LAYER_PAIR: int = 128

      def __init__(self, model: MLPClassifier | MLPRegressor) -> None: ...

      def extract_schema(self) -> ModelSchema:
          """Extracts feature_names_in_, layer dimensions, activation, and out_activation_."""
          ...

      def compute_forward_trace(self, X_chunk: np.ndarray) -> NeuralForwardTrace:
          """Computes exact Z^(l) and A^(l) across all layers, downsamples Top-32 nodes,
          and identifies Dead ReLU neurons across the chunk."""
          ...

      def compute_input_gradient_attribution(self, X_chunk: np.ndarray) -> np.ndarray:
          """Computes analytical Input * Gradient attribution via chain-rule Jacobian."""
          ...

      def predict_batch(self, df_chunk: pd.DataFrame) -> PredictionBatchResult: ...
  ```

### 5.2 Frontend (`frontend/src/domains/simulation/`)
- **`components/stage3d/renderers/MLPNeuralSynapse3D.tsx`**
  ```tsx
  export interface MLPNeuralSynapse3DProps {
    layerDimensions: number[];
    renderedNodeIndices: number[][];
    nodeActivations: Float32Array[]; // Per-layer activations for current frame
    deadNeuronMasks: boolean[][];
    synapseEdges: Array<{ fromLayer: number; u: number; v: number; flux: number }>;
  }
  export const MLPNeuralSynapse3D: React.FC<MLPNeuralSynapse3DProps> = (props) => { ... };
  ```
- ใช้ `THREE.InstancedMesh` เพียง 1 Draw Call สำหรับวาดโหนดประสาททุกชั้นรวมกัน และใช้ `THREE.LineSegments` พร้อม `Float32BufferAttribute` สำหรับสีและความโปร่งใสของเส้นใยประสาท

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)
1. **In-Place Activation Computation & Downsampling Before Serialization**:
   - หากโมเดลมีโครงสร้างใหญ่ เช่น `(1024, 512, 256)` การเก็บ Activation ของทุกโหนดสำหรับ `2,500` แถวจะเปลือง RAM โดยไม่จำเป็น ดังนั้น `compute_forward_trace()` จะคำนวณสถิติ `DeadRatio` และคัดเลือก `Top-32 Neurons` บน NumPy ทันที แล้วตัดเก็บเฉพาะอาเรย์ขนาด `(N_chunk, 32)` ต่อชั้นในรูป `np.float16` / `np.float32` ก่อนส่งผ่าน Binary MsgPack ไปยัง Frontend
2. **Zero-Allocation WebGL Frame Scrubbing**:
   - ในฝั่ง React Three Fiber ห้ามสร้าง `new THREE.Vector3()` หรือ `new THREE.Color()` ใน `useFrame()` หรือเมื่อเลื่อน Scrubber ให้เขียนค่าสีและขนาดโหนดลงใน `instancedMesh.instanceColor.array` และ `instanceMatrix.array` โดยตรง แล้วตั้งค่า `.needsUpdate = true`
3. **Chunked Jacobian Calculation**:
   - การคำนวณ Input $\times$ Gradient Attribution ใช้การคูณเมทริกซ์ย้อนกลับทีละชั้น (Backpropagation w.r.t. Input) โดยจำกัดขนาด Batch ย่อยที่ `512` แถวเพื่อควบคุม Peak RAM ไม่ให้เกิน `80 MB`

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| กรณีปัญหา (Edge Case) | สาเหตุเชิงเทคนิค | แนวทางป้องกันและแก้ไขในระบบ (Mitigation Strategy) |
| :--- | :--- | :--- |
| **1. Hidden Layer ขนาดใหญ่มาก (เช่น 1,000 โหนด)** | การเรนเดอร์เส้นเชื่อม $1000 \times 1000 = 10^6$ เส้นจะทำให้ WebGL ค้างทันที | บังคับเพดาน `MAX_RENDERED_NEURONS_PER_LAYER = 32` และ `MAX_SYNAPSES_PER_LAYER_PAIR = 128` พร้อมแสดงป้าย `"Showing Top 32 / 1000 Active Neurons"` |
| **2. Overflow ในฟังก์ชัน `logistic` หรือ `softmax`** | ค่า $Z^{(l)}$ ที่มีขนาดติดลบหรือบวกสูงมากทำให้ `np.exp(-Z)` เกิด `RuntimeWarning: overflow` | ใช้ `scipy.special.expit(Z)` สำหรับ Logistic Sigmoid และ `scipy.special.softmax(Z, axis=-1)` ซึ่งทำ Log-Sum-Exp Trick ป้องกัน Overflow เสมอ |
| **3. โมเดลที่มี Hidden Layer เพียงชั้นเดียว vs หลายชั้น** | `model.coefs_` เป็น `list` ของเมทริกซ์ที่มีขนาดไม่เท่ากัน Неสามารถแปลงเป็น 3D `np.ndarray` ก้อนเดียวได้ | จัดเก็บโครงสร้างน้ำหนักเป็น `list[np.ndarray]` แยกตามชั้น $l$ และส่งเฉพาะ Metadata ของโหนดที่คัดกรองแล้ว |
| **4. ค่า Activation ใน ReLU พุ่งสูงผิดปกติ (Exploding Activations)** | ReLU ไม่มีเพดานบน ($[0, \infty)$) ทำให้ขนาดทรงกลมใน 3D ใหญ่ล้นจอ | ใช้ฟังก์ชันบีบอัดขนาดแสดงผล $r = r_{\min} + (r_{\max} - r_{\min}) \cdot \tanh(|a| / \sigma_a)$ เพื่อคุมขนาดโหนดใน 3D ให้อยู่ในกรอบสวยงามเสมอ |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] รองรับทั้ง `MLPClassifier` และ `MLPRegressor` ทุกชนิดของ `activation` (`relu`, `tanh`, `logistic`, `identity`)
- [ ] คำนวณค่า Output จาก `compute_forward_trace()` ชั้นสุดท้ายตรงกับ `model.predict_proba()` / `model.predict()` ของ Scikit-learn ในระดับความแม่นยำ `< 1e-5`
- [ ] แสดง 3D Floating Synapse Network ที่คัดกรอง Top-32 Neurons และเปลี่ยนสี/ความสว่างของโหนดและเส้นเชื่อมตามการเลื่อน VCR Scrubber ที่ 60 FPS
- [ ] ตรวจจับและไฮไลต์โหนด Dead ReLU (`Activation == 0.0` ทุกแถว) พร้อมแสดงสถิติสุขภาพของแต่ละ Layer ได้อย่างถูกต้อง
- [ ] แสดงกราฟ Waterfall จากการคำนวณ Analytical Input $\times$ Gradient Attribution ในแผง Explainability HUD ได้ครบถ้วน
