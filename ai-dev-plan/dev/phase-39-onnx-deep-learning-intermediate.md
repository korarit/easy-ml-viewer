# Phase 39: ONNX Deep Neural Network Intermediate Layer Extractor & 3D Latent Manifold

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
พัฒนา Adapter สำหรับไฟล์ **`.onnx` ที่เป็นโครงข่ายประสาทเทียมเชิงลึก (Deep Neural Networks / Tabular Deep Learning ที่ Export มาจาก PyTorch, TensorFlow, Keras, หรือ JAX)** โดยใช้เทคนิค **In-Memory ONNX Graph Output Injection** เพื่อเจาะดึงค่า Tensor จากชั้นซ่อนเร้นภายใน (Intermediate Hidden Layers) ออกมาแสดงเป็น **3D Latent Space Embedding Manifold** และคลื่นพลังงานประสาทโดยไม่ต้องแก้ไฟล์ `.onnx` ต้นฉบับบนดิสก์

### ขอบเขตการพัฒนา (Scope)
1. **`ONNXDeepLearningAdapter` (`adapters/neural_onnx/onnx_deep.py`)**: ตรวจจับกราฟ ONNX ที่ไม่มีโหนดตระกูล `ai.onnx.ml` แต่ประกอบด้วยตัวดำเนินการเชิงลึก (`Gemm`, `MatMul`, `Relu`, `Gelu`, `Silu`, `Sigmoid`, `Tanh`, `BatchNormalization`, `LayerNormalization`, `Attention`, `Add`, `Softmax`)
2. **In-Memory Intermediate Output Injection**: ดัดแปลงโครงสร้าง `onnx.ModelProto.graph.output` ในหน่วยความจำเพื่อแนบจุดดักฟัง (Probe Tensors) ที่ปลายของแต่ละชั้นซ่อนเร้น
3. **3D Latent Space PCA Manifold Projection**: นำเวกเตอร์แทนความหมายจากชั้นรองสุดท้าย (Penultimate Layer Representation $\mathbf{h}^{(L-1)} \in \mathbb{R}^{d_{\text{latent}}}$) มาลดมิติด้วย 3D Singular Value Decomposition (SVD / PCA) เพื่อแสดงให้เห็นว่าสมองส่วนลึกของ Deep Learning จัดกลุ่มข้อมูลในอวกาศ 3 มิติอย่างไร
4. **Static `batch_size=1` Fallback & Execution Provider Auto-Selector**: รองรับโมเดลที่ล็อกขนาด Batch คงที่ `[1, F]` ตอน Export จาก PyTorch และสลับใช้ GPU (`CUDA`/`DirectML`/`CoreML`) หรือ `CPUExecutionProvider` อัตโนมัติ

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 กลไกการฝังจุดดักฟังในกราฟ ONNX (In-Memory Graph Output Injection)
ไฟล์ `.onnx` ของ Deep Learning โดยทั่วไปจะเปิดเผยเฉพาะโหนด Input $\mathbf{X}$ และโหนด Output สุดท้าย $\hat{\mathbf{Y}}$ เท่านั้น เพื่อดึงค่า Activation ของชั้นซ่อนเร้นที่ $l$ ($\mathbf{H}^{(l)}$) ออกมาในการรัน `session.run()` เพียงครั้งเดียว ระบบดำเนินการดังนี้:
1. โหลด `model_proto = onnx.load(str(file_path))` เข้าสู่ RAM และรัน `inferred_proto = onnx.shape_inference.infer_shapes(model_proto)`
2. ค้นหาโหนดที่เป็นจุดสิ้นสุดของแต่ละบล็อกชั้นประสาท (เช่น โหนด `Relu`, `Gelu`, `Tanh`, `Sigmoid`, `LayerNormalization` หรือโหนด `Gemm`/`MatMul` ที่อยู่ก่อนหน้า Output Layer)
3. สำหรับแต่ละโหนดเป้าหมาย $n_l$ ดึงชื่อเอาต์พุต `tensor_name = n_l.output[0]` แล้วสร้าง `ValueInfoProto` เพิ่มเข้าไปใน `inferred_proto.graph.output`:
   ```python
   intermediate_info = onnx.helper.make_empty_tensor_value_info(tensor_name)
   inferred_proto.graph.output.append(intermediate_info)
   ```
4. สร้าง `ort.InferenceSession(inferred_proto.SerializeToString(), providers=...)` เมื่อรัน `outputs = sess.run(None, feed_dict)` จะได้ทั้งผลลัพธ์สุดท้าย $\hat{\mathbf{Y}}$ และรายการเมทริกซ์ชั้นซ่อนเร้น $\{\mathbf{H}^{(1)}, \mathbf{H}^{(2)}, \dots, \mathbf{H}^{(L-1)}\}$ ออกมาพร้อมกันใน C++ Pass เดียว!

### 2.2 คณิตศาสตร์ของการฉายปริภูมิแฝง 3 มิติ (3D Latent Space PCA Manifold)
กำหนดให้ $\mathbf{H} = \mathbf{H}^{(L-1)} \in \mathbb{R}^{N \times D}$ คือเมทริกซ์ Activation ของชั้นรองสุดท้าย (Penultimate Layer) ก่อนเข้าชั้น Output Linear/Softmax:
1. หาก $D > 3$ ระบบทำ Centering และแยกองค์ประกอบหลักด้วย Truncated SVD:
   $$
   \tilde{\mathbf{H}} = \mathbf{H} - \frac{1}{N}\mathbf{1}\mathbf{1}^T\mathbf{H}, \qquad \tilde{\mathbf{H}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
   $$
2. ฉายพิกัดของข้อมูลทุกแถวลงสู่อวกาศแฝง 3 มิติ (3D Latent Manifold Coordinates):
   $$
   \mathbf{Z}_{\text{latent}} = \tilde{\mathbf{H}} \mathbf{V}_{:, 1:3} \in \mathbb{R}^{N \times 3}
   $$
3. เนื่องจากชั้นสุดท้ายของ Deep Network มักเป็นชั้น Linear $\text{Logits} = \mathbf{H}\mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}}$ ในปริภูมิแฝง $\mathbf{H}$ ขอบเขตการตัดสินใจของโมเดลที่ซับซ้อนในปริภูมิข้อมูลดิบ **จะถูกคลี่ออกจนกลายเป็นระนาบแบน (Linear Separating Hyperplane) ในปริภูมิแฝง 3 มิติ**! การแสดงทั้ง 2 มุมมองคู่กันจึงทำให้นักพัฒนาเข้าใจแก่นแท้ของ Deep Learning ได้ทันที

### 2.3 การรับมือกับ Static `batch_size=1` ในโมเดลจาก PyTorch
เมื่อผู้ใช้ Export โมเดลจาก PyTorch ด้วย `torch.onnx.export(model, dummy_input, "model.onnx")` โดยลืมกำหนด `dynamic_axes={'input': {0: 'batch_size'}}` มิติแรกของ Input Tensor จะถูกล็อกตายตัวที่ `shape = [1, F]` (แทนที่จะเป็น `['batch_size', F]` หรือ `[None, F]`):
- หากส่ง `X_chunk` ขนาด `(2500, F)` เข้าไปตรงๆ `onnxruntime` จะโยน `InvalidArgument: Got invalid dimensions for input: expected 1, got 2500`
- **กลไกแก้ไข**: ระบบตรวจสอบ `dim_0 = sess.get_inputs()[0].shape[0]`
  - หาก `isinstance(dim_0, int) and dim_0 > 0 and dim_0 != len(X_chunk)`: ระบบจะ嘗試แก้ไข `dim_param = "batch_size"` ใน `inferred_proto.graph.input[0]` และ `value_info` ก่อนสร้าง Session
  - หากภายในกราฟมีโหนด `Reshape` ที่ฝังค่าคงที่ `1` ไว้จนแก้ Graph Shape ไม่ได้ ระบบจะสลับเข้าสู่โหมด **Pre-Allocated Micro-Batch Loop (`batch_size = dim_0`)** ด้วย `ort.IoBinding()` เพื่อรันทีละ 1 แถวอย่างรวดเร็วโดยไม่ให้เกิด Exception

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)
1. **Dual 3D Stage: `[Input Feature Surface 3D]` $\leftrightarrow$ `[3D Latent Space PCA Manifold]`**:
   - สลับหรือเปิดดู 2 จอพร้อมกัน: ด้านซ้ายคือพื้นผิวการตัดสินใจที่บิดโค้งซับซ้อนในปริภูมิฟีเจอร์จริง $(X, Y, \hat{Y})$ ส่วนด้านขวาคือกลุ่มก้อนข้อมูลที่ถูกคลี่ออกเป็นกลุ่มชัดเจนในอวกาศแฝง 3 มิติ $(PC_1, PC_2, PC_3)$ ของชั้น Penultimate Layer!
2. **ONNX Deep Layer Pulse & Activation Histogram Bar**:
   - แสดงกราฟแท่งและคลื่นแสงของค่าเฉลี่ย/ค่าสูงสุด Activation ในแต่ละชั้นที่ถูกดักฟัง (`Gemm_0 -> Relu_1 -> BatchNorm_2 -> Relu_3 -> Gemm_4`) ขณะเลื่อน VCR Scrubber
3. **Static Batch & Execution Provider Telemetry Pill**:
   - แสดงสถานะว่าโมเดลรองรับ Dynamic Batch หรือใช้โหมด Auto-Unrolled `batch_size=1` พร้อมแสดงฮาร์ดแวร์ที่ใช้รัน (`CUDAExecutionProvider` / `DmlExecutionProvider` / `CPUExecutionProvider`)
4. **Finite-Difference / Integrated Input Sensitivity Waterfall**:
   - คำนวณอิทธิพลของแต่ละฟีเจอร์ต้นทางต่อเอาต์พุตของกราฟ ONNX สำหรับแถวที่เลือกแสดงใน Waterfall HUD

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

### UX Flow
1. ผู้ใช้อัปโหลดไฟล์ `pytorch_tabular_resnet.onnx` คู่กับ `sensor_readings.csv`
2. ระบบตรวจพบกราฟ Deep Learning (มีโหนด `MatMul`, `BatchNormalization`, `Relu`, `Add`) และตรวจพบว่า `input.shape = [1, 16]` (Static Batch Size = 1)
3. ระบบฝังจุดดักฟังที่โหนด `Relu` ทุกชั้น และเปิดโหมด Micro-Batch Runner ดึงค่า Penultimate Layer ออกมาทำ 3D PCA Manifold
4. เมื่อผู้ใช้กด Play ลูกบอลข้อมูลจะเคลื่อนที่ไปบนพื้นผิว 3D ด้านซ้าย พร้อมกับเคลื่อนที่ผ่านกลุ่มหมอกดาว (3D Latent Cluster Cloud) ด้านขวาไปพร้อมกันแบบเรียลไทม์

### ASCII Wireframe
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer]  Model: pytorch_tabular_resnet.onnx (ONNX Deep Net | 5 Hidden Blocks | OpSet 17)|
| Provider: [DmlExecutionProvider (GPU)]   Batch Mode: [Static batch_size=1 Auto-Unroll Active]    |
+-----------------------------------+------------------------------+-------------------------------+
| 3D INPUT FEATURE SURFACE          | 3D LATENT MANIFOLD (Penult.) | DEEP LAYER PROBE TELEMETRY    |
|                                   |                              |                               |
|       Z (Anomaly Prob)            |        Latent PC3            | [Injected Layer Probes (5)]   |
|       ^    _.-~-._                |          ^    . o o .        | 1. /fc1/Relu_output_0  (D=128)|
|   1.0 |   /  (*)  \   _.-.        |     . .  |   o  (*)  o       |    Mean: 0.42 | Sparsity: 38% |
|       |  / Row #88 \_/    \       |   . x x .|    . o o .        | 2. /res1/Add_output_0  (D=128)|
|   0.0 +--------------------> X    | ---------+-----------> PC1   |    Mean: 0.61 | Sparsity: 12% |
|      /                            |         /|  Class 1 Cluster  | 3. /penult/Relu_out    (D=64) |
|     v Y: vibration_hz             |        v PC2                 |    3D PCA Variance: 84.6%     |
|                                   |                              |                               |
| [x] Synchronize Dual 3D Cameras   | (*) Active Row #88 in Latent | [Local Sensitivity Waterfall] |
| [x] Show Latent Trajectory Trail  |     Space: [+2.1, -0.8, +1.4]| vibration_hz : +0.52 [++++++] |
+-----------------------------------+------------------------------+-------------------------------+
| [|<] [<] [ > Play ] [>] [>|]    Frame: 0088 / 3000   [===*====================]   Speed: [1.0x v]|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD แยกรายชั้นพร้อมชื่อไฟล์และ Code Signatures

### 5.1 Backend (`backend/app/domains/model_engine/`)
- **`adapters/neural_onnx/onnx_deep.py`**
  ```python
  from pathlib import Path
  import numpy as np
  import pandas as pd
  import onnx
  import onnxruntime as ort
  from app.domains.model_engine.domain.ports import BaseModelAdapter
  from app.domains.model_engine.domain.entities import (
      ModelSchema, PredictionBatchResult, ONNXDeepTraceResult
  )

  class ONNXDeepLearningAdapter(BaseModelAdapter):
      MAX_PROBED_LAYERS: int = 8
      MAX_NEURONS_PER_PROBED_LAYER: int = 32

      def __init__(self, model_path: Path, onnx_proto: onnx.ModelProto, providers: list[str]) -> None: ...

      def _inject_intermediate_outputs(self, proto: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
          """Clones ModelProto in-memory and appends top hidden activation outputs to graph.output."""
          ...

      def _run_session_with_batch_guard(self, X_chunk: np.ndarray) -> list[np.ndarray]:
          """Executes sess.run() in full batch if dynamic, or micro-batch loop if static batch_size=1."""
          ...

      def compute_latent_pca_manifold(self, penultimate_activations: np.ndarray) -> np.ndarray:
          """Projects (N, D_latent) activations into (N, 3) float32 coordinates via SVD."""
          ...

      def predict_batch(self, df_chunk: pd.DataFrame) -> PredictionBatchResult: ...
  ```

### 5.2 Frontend (`frontend/src/domains/simulation/`)
- **`components/stage3d/renderers/ONNXLatentManifold3D.tsx`**
  ```tsx
  export interface ONNXLatentManifold3DProps {
    latentCoords3D: Float32Array; // Flat [x0, y0, z0, x1, y1, z1, ...]
    predictedClasses: Uint16Array;
    activeFrameIndex: number;
    layerTelemetry: Array<{ layerName: string; meanActivation: number; sparsity: number }>;
  }
  export const ONNXLatentManifold3D: React.FC<ONNXLatentManifold3DProps> = (props) => { ... };
  ```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)
1. **Selective Probe Injection Cap (`MAX_PROBED_LAYERS = 8`)**:
   - หากโมเดล Deep Learning มีโหนดย่อยมากกว่า 100 โหนด การดึงเอาต์พุตของทุกโหนดออกมาพร้อมกันจะทำให้ RAM ของ `onnxruntime` พุ่งสูงหลายกิกะไบต์ ระบบจึงคัดเลือกเฉพาะโหนด Activation หลักไม่เกิน `8` จุดที่กระจายตัวสม่ำเสมอตั้งแต่ต้นกราฟจนถึงโหนด Penultimate Layer
2. **Immediate Activation Downsampling & Garbage Collection**:
   - ทันทีที่ `sess.run()` คืนค่ารายการ `outputs` สำหรับแต่ละ Chunk ระบบจะคำนวณ 3D PCA บน Penultimate Layer และตัดเก็บเฉพาะ `Top-32` โหนดของชั้นที่เหลือในรูป `np.float32` จากนั้นสั่ง `del outputs` ทันทีเพื่อคืนหน่วยความจำของ Intermediate Tensors ขนาดเต็มให้ระบบปฏิบัติการ
3. **Hardware Execution Provider Priority Fallback**:
   - ตรวจสอบ `ort.get_available_providers()` ตามลำดับ `["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]` โดยกำหนด `arena_extend_strategy = "kSameAsRequested"` เพื่อไม่ให้ GPU จอง VRAM เกินความจำเป็น

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| กรณีปัญหา (Edge Case) | สาเหตุเชิงเทคนิค | แนวทางป้องกันและแก้ไขในระบบ (Mitigation Strategy) |
| :--- | :--- | :--- |
| **1. Static `batch_size = 1` ใน Input Tensor** | โมเดลที่ Export จาก PyTorch โดยไม่ใส่ `dynamic_axes` จะล็อกมิติที่ 0 เป็น `1` | ตรวจเช็ก `input_shape[0]` หากเป็นค่าคงที่ `N_fixed` ให้แบ่ง `X_chunk` รันทีละ `N_fixed` แถวใน C++ Session แล้ว `np.concatenate` ผลลัพธ์เข้าด้วยกัน |
| **2. เอาต์พุตของโมเดลเป็น Raw Logits (ไม่มี Softmax)** | โมเดล PyTorch มักจบด้วย `nn.Linear` เพราะใช้ `CrossEntropyLoss` ตอนเทรน | ตรวจสอบโหนดสุดท้ายของกราฟ หากเป็น Classification แต่โหนดสุดท้ายไม่ใช่ `Softmax` หรือ `Sigmoid` (และผลรวมแถวไม่เท่ากับ `1.0`) ระบบจะครอบ `scipy.special.softmax` / `expit` ให้อัตโนมัติ |
| **3. โหนดซ่อนเร้นมีมิติมากกว่า 2 มิติ (เช่น 1D CNN `[N, C, L]`)** | โมเดล Tabular CNN หรือ Attention อาจมี Intermediate Tensor เป็น 3 มิติ `(N, C, L)` | ก่อนคำนวณสถิติหรือทำ 3D Latent PCA ให้ทำ Global Average Pooling ตามแกนลำดับ `tensor.mean(axis=-1)` หรือ Flatten ให้เป็น `(N, D)` เสมอ |
| **4. `onnx.shape_inference` ล้มเหลวใน Custom Ops** | บางโมเดลมีโหนดพิเศษที่ `infer_shapes` ไม่รู้จัก | ห่อ `onnx.shape_inference.infer_shapes` ด้วย `try...except` และใช้ `onnx.helper.make_empty_tensor_value_info(name)` ซึ่งยอมให้ `onnxruntime` คำนวณ Shape ตอนรันจริงได้โดยไม่ต้องระบุ Shape ล่วงหน้า |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] โหลดไฟล์ `.onnx` ของโครงข่ายประสาทเทียม (PyTorch / Keras / TensorFlow Tabular DNN) ได้ทั้งแบบ Dynamic Batch และ Static `batch_size=1`
- [ ] เจาะดึงค่า Intermediate Layer Tensors ด้วยเทคนิค In-Memory Graph Output Injection โดยไม่แก้ไขไฟล์ `.onnx` ต้นฉบับบนดิสก์
- [ ] คำนวณและแสดงฉาก **3D Latent Space PCA Manifold** ควบคู่กับ **3D Input Feature Decision Surface** ได้อย่างลื่นไหลที่ 60 FPS
- [ ] แปลง Raw Logits เป็นความน่าจะเป็น (`Softmax`/`Sigmoid`) อัตโนมัติเมื่อกราฟไม่ได้แนบชั้น Activation สุดท้ายมาด้วย
- [ ] ควบคุมการใช้หน่วยความจำระหว่างการดึง Intermediate Tensors ให้ไม่เกินเพดาน `150 MB` ต่อ Chunk
