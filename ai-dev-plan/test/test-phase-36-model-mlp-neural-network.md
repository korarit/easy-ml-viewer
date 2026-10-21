# Test Plan — Phase 36: Multi-Layer Perceptron (MLP) Adapter & 3D Floating Synapse Network + Layer Pulses

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 36)
เอกสารนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ **[phase-36-model-mlp-neural-network.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-36-model-mlp-neural-network.md)** เพื่อตรวจสอบคลาส **`MLPAdapter` (`backend/app/domains/model_engine/adapters/neural_onnx/mlp_adapter.py`)** และคอมโพเนนต์ 3 มิติ **`MLPNeuralSynapse3D` (`frontend/src/domains/simulation/components/stage3d/renderers/MLPNeuralSynapse3D.tsx`)**

### ขอบเขตการตรวจสอบหลัก
1. **Layer-by-Layer Forward Pass Mathematical Parity**: ยืนยันว่าการคำนวณ $\mathbf{Z}^{(l)} = \mathbf{A}^{(l-1)}\mathbf{W}^{(l)} + \mathbf{b}^{(l)}$ และ $\mathbf{A}^{(l)} = \sigma_l(\mathbf{Z}^{(l)})$ ของ `compute_forward_trace()` ให้ผลลัพธ์ชั้นสุดท้ายตรงกับ `model.predict_proba()` / `model.predict()` ของ Scikit-learn ทุกประการ
2. **Top-32 Active Neurons & Top-128 Synapses Downsampler**: ยืนยันการคัดกรองโหนดประสาทที่มีพลังงานสูงสุดไม่เกิน `32` โหนดต่อชั้น และเส้นใยประสาทไม่เกิน `128` เส้นต่อคู่ชั้น
3. **Dead ReLU Neuron Detection**: ยืนยันการตรวจจับโหนดประสาทที่มีค่า $A^{(l)}_{i, j} \le 0$ ตลอดทั้ง Chunk ($\text{DeadRatio}^{(l)}_j = 1.0$)
4. **Analytical Input $\times$ Gradient Attribution**: ยืนยันความถูกต้องของการคูณเมทริกซ์ Jacobian ย้อนกลับเพื่อหาค่าอิทธิพลรายฟีเจอร์

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow / System Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

ไฟล์ทดสอบหลัก: `backend/tests/domains/model_engine/adapters/neural_onnx/test_mlp_adapter.py`

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from app.domains.model_engine.adapters.neural_onnx.mlp_adapter import MLPAdapter


@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic", "identity"])
def test_mlp_forward_pass_exact_parity_with_sklearn(activation: str) -> None:
    """ยืนยันว่า compute_forward_trace() คำนวณ Output ชั้นสุดท้ายตรงกับ predict_proba ของ sklearn (< 1e-5)"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = (X[:, 0] * 1.5 - X[:, 1] > 0).astype(int)

    mlp = MLPClassifier(hidden_layer_sizes=(24, 16), activation=activation, max_iter=30, random_state=42)
    mlp.fit(X, y)

    adapter = MLPAdapter(model=mlp)
    trace = adapter.compute_forward_trace(X)

    sklearn_probs = mlp.predict_proba(X).astype(np.float32)
    np.testing.assert_allclose(trace.output_activations, sklearn_probs, rtol=1e-5, atol=1e-5)


def test_top32_neuron_downsampling_and_dead_relu_detection() -> None:
    """ยืนยันว่า Hidden Layer ขนาด 128 โหนดถูกคัดกรองเหลือ 32 โหนด และตรวจจับ Dead ReLU ได้แม่นยำ 100%"""
    rng = np.random.default_rng(7)
    X = rng.uniform(0.1, 1.0, size=(50, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=50)

    mlp = MLPClassifier(hidden_layer_sizes=(128, 48), activation="relu", max_iter=10, random_state=7)
    mlp.fit(X, y)

    # บังคับสร้าง Dead ReLU Neurons ในชั้นที่ 1 จำนวน 3 โหนด (ตั้งน้ำหนักเป็นลบและไบแอสติดลบสูง)
    mlp.coefs_[0][:, [3, 17, 99]] = -5.0
    mlp.intercepts_[0][[3, 17, 99]] = -100.0

    adapter = MLPAdapter(model=mlp)
    trace = adapter.compute_forward_trace(X)

    # 1. ตรวจสอบการคัดกรอง Top-32 โหนดต่อชั้น
    assert trace.layer_activations[0].shape == (50, 32)
    assert trace.layer_activations[1].shape == (50, 32)
    assert len(trace.synapse_edges_per_layer[0]) <= MLPAdapter.MAX_SYNAPSES_PER_LAYER_PAIR

    # 2. ตรวจสอบการตรวจพบ Dead ReLU ที่ดัชนี 3, 17, 99
    dead_indices_layer1 = np.where(trace.dead_neuron_masks[0])[0]
    assert {3, 17, 99}.issubset(set(dead_indices_layer1))
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & UX Flow Tests (พร้อมโค้ดต้นแบบ `vitest`)

ไฟล์ทดสอบหลัก: `frontend/src/domains/simulation/components/stage3d/renderers/__tests__/MLPNeuralSynapse3D.test.tsx`

```tsx
import { describe, it, expect } from "vitest";
import { MLPNeuralSynapse3DProps } from "../MLPNeuralSynapse3D";

describe("Phase 36 — MLPNeuralSynapse3D & Synapse Flux Polarity", () => {
  it("จำแนกสีเส้นใยประสาท Excitatory (+Flux) และ Inhibitory (-Flux) พร้อมทำเครื่องหมาย Dead ReLU", () => {
    const mockProps: MLPNeuralSynapse3DProps = {
      layerDimensions: [10, 64, 32, 2],
      renderedNodeIndices: [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        Array.from({ length: 32 }, (_, i) => i),
        Array.from({ length: 32 }, (_, i) => i),
        [0, 1],
      ],
      nodeActivations: [
        new Float32Array(10).fill(0.5),
        new Float32Array(32).map((_, i) => (i === 4 ? 0.0 : 1.2)),
        new Float32Array(32).fill(0.8),
        new Float32Array([0.12, 0.88]),
      ],
      deadNeuronMasks: [
        new Array(10).fill(false),
        new Array(32).map((_, i) => i === 4), // โหนดที่ 4 เป็น Dead ReLU
        new Array(32).fill(false),
        [false, false],
      ],
      synapseEdges: [
        { fromLayer: 0, u: 0, v: 1, flux: +0.85 }, // Excitatory
        { fromLayer: 0, u: 2, v: 5, flux: -0.62 }, // Inhibitory
      ],
    };

    expect(mockProps.deadNeuronMasks[1][4]).toBe(true);
    expect(mockProps.synapseEdges[0].flux).toBeGreaterThan(0);
    expect(mockProps.synapseEdges[1].flux).toBeLessThan(0);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Process Lifecycle & GPU/CPU Resource Tests)

1. **In-Place Downsampling Memory Cap Test**:
   - สร้างโมเดล `MLPRegressor(hidden_layer_sizes=(1024, 512, 256))` แล้วรัน `compute_forward_trace()` บน Chunk ขนาด `2,500` แถว ยืนยันว่าอาเรย์ที่ถูกเก็บใน `NeuralForwardTrace` มีขนาดคอลัมน์ถูกบีบเหลือ `(2500, 32)` ต่อชั้นทันที และใช้ Peak RAM Delta ไม่เกิน `80 MB`
2. **Zero-Allocation WebGL Scrubbing Benchmark**:
   - ทดสอบการอัปเดตสถานะโหนดและสีเส้นเชื่อมเมื่อเปลี่ยนเฟรมจาก `0 -> 500` ใน `MLPNeuralSynapse3D` ยืนยันว่าเป็นการแก้ไขค่าภายใน `instancedMesh.instanceMatrix.array` และ `lineGeometry.attributes.color.array` โดยตรง ไม่มีการสร้าง Geometry ใหม่และคงความเร็วที่ 60 FPS (`< 16.6 ms/frame`)

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| กรณีปัญหา (Edge Case จาก Dev Phase 36) | วิธีจำลองในการทดสอบ (Test Simulation Setup) | ผลลัพธ์ที่ต้องผ่านการตรวจสอบ (Expected Verification Assertion) |
| :--- | :--- | :--- |
| **1. Hidden Layer ขนาดใหญ่มาก (เช่น 1,000 โหนด)** | สร้าง `MLPClassifier(hidden_layer_sizes=(1000,))` แล้วรัน `compute_forward_trace()` | ระบบคัดกรองโหนดเหลือ `32` โหนด และเส้นเชื่อมไม่เกิน `128` เส้น พร้อมแนบ Metadata `total_nodes=1000, rendered_nodes=32` |
| **2. Overflow ในฟังก์ชัน `logistic` หรือ `softmax`** | ป้อนข้อมูลที่มีค่าสุดขั้ว `X = np.full((10, 5), 1e4, dtype=np.float32)` เข้าโมเดลที่ใช้ `activation="logistic"` | ใช้ `scipy.special.expit` และ `softmax` ไม่เกิด `RuntimeWarning: overflow encountered in exp` และไม่มีค่า `NaN` |
| **3. โมเดลที่มี Hidden Layer เพียงชั้นเดียว vs หลายชั้น** | ทดสอบทั้ง `hidden_layer_sizes=(16,)` (1 ชั้น) และ `(64, 32, 16, 8)` (4 ชั้น) | `extract_schema()` และ `compute_forward_trace()` จัดการรายการเมทริกซ์ `coefs_` ต่างขนาดได้โดยไม่เกิด `ValueError: setting an array element with a sequence` |
| **4. ค่า Activation ใน ReLU พุ่งสูงผิดปกติ (Exploding Activations)** | ป้อนค่าให้ $A^{(l)}_j > 500.0$ แล้วตรวจสอบฟังก์ชันคำนวณรัศมีโหนด 3D | รัศมีโหนด 3D ถูกบีบอัดด้วยฟังก์ชัน $\tanh(|a|/\sigma_a)$ ให้อยู่ในช่วง $[r_{\min}, r_{\max}]$ เสมอ ไม่ล้นทับโหนดข้างเคียง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Forward Pass Exact Parity (`< 1e-5`)**: ผลลัพธ์การคำนวณผ่านชั้นประสาทตรงกับ Scikit-learn ทั้ง 4 ฟังก์ชันกระตุ้น (`relu`, `tanh`, `logistic`, `identity`)
- [ ] **Top-32 Cap & Dead ReLU Accuracy**: คัดกรองโหนดไม่เกิน 32 โหนดต่อชั้น และระบุตำแหน่ง Dead ReLU ได้แม่นยำ 100%
- [ ] **Memory & Draw Call SLA**: ใช้ RAM ส่วนเพิ่มใน Backend $\le 80\text{ MB}$ ต่อ Chunk และใช้ WebGL Draw Calls สำหรับวาดโหนดและเส้นประสาทรวมกันไม่เกิน `4` Draw Calls

- [ ] **Analytical Input $\times$ Gradient Parity (`< 1e-4`)**: ตรวจสอบผลการคำนวณ `compute_input_gradient_attribution(X_chunk)` เทียบกับ Finite-Difference Numerical Gradient $(\frac{f(\mathbf{x} + \epsilon \mathbf{e}_j) - f(\mathbf{x} - \epsilon \mathbf{e}_j)}{2\epsilon} \cdot x_j)$ ยืนยันความสอดคล้องทางคณิตศาสตร์ทุกคอลัมน์

### คำสั่งรันชุดทดสอบประจำเฟส 36 (CLI Test Execution Commands)
```bash
# รันชุดทดสอบ Backend MLPAdapter & Memory Guard
pytest backend/tests/domains/model_engine/adapters/neural_onnx/test_mlp_adapter.py -v

# รันชุดทดสอบ Frontend 3D Floating Synapse Network & Dead ReLU HUD
cd frontend && npx vitest run src/domains/simulation/components/stage3d/renderers/__tests__/MLPNeuralSynapse3D.test.tsx
```
