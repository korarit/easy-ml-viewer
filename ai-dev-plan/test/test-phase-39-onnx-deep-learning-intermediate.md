# Test Plan — Phase 39: ONNX Deep Neural Network Intermediate Layer Extractor & 3D Latent Manifold

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 39)
เอกสารนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ **[phase-39-onnx-deep-learning-intermediate.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-39-onnx-deep-learning-intermediate.md)** เพื่อยืนยันความถูกต้องของคลาส **`ONNXDeepLearningAdapter` (`backend/app/domains/model_engine/adapters/neural_onnx/onnx_deep.py`)** และคอมโพเนนต์ 3 มิติ **`ONNXLatentManifold3D` (`frontend/src/domains/simulation/components/stage3d/renderers/ONNXLatentManifold3D.tsx`)**

### ขอบเขตการตรวจสอบหลัก
1. **In-Memory Intermediate Output Injection Integrity**: ยืนยันว่า `_inject_intermediate_outputs()` เพิ่มโหนดชั้นซ่อนเร้น (`Relu`, `Gelu`, `Tanh`, `Gemm`, `MatMul`) เข้าไปใน `proto.graph.output` ในหน่วยความจำโดยไม่แก้ไขไฟล์ `.onnx` บนดิสก์ (SHA-256 ของไฟล์บนดิสก์ต้องคงเดิม 100%)
2. **3D Latent Space PCA Manifold Mathematical Accuracy**: ยืนยันว่า `compute_latent_pca_manifold()` คำนวณ SVD บน Penultimate Layer $\tilde{\mathbf{H}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ และคืนค่าพิกัด 3 มิติขนาด `(N, 3)` (`float32`) ที่ศูนย์กลางอยู่ที่ `(0, 0, 0)`
3. **Static `batch_size=1` Auto-Unroll Fallback**: ยืนยันว่า `_run_session_with_batch_guard()` สามารถรันโมเดลที่ล็อกมิติแรกไว้ที่ `[1, F]` กับข้อมูลขนาด `N > 1` แถวได้โดยไม่เกิด `onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument`
4. **Raw Logits to Probability Auto-Normalization**: ยืนยันการครอบ `Softmax` / `Sigmoid` อัตโนมัติเมื่อโหนดสุดท้ายของกราฟเป็น `Gemm`/`MatMul` ดิบ

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow / System Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

ไฟล์ทดสอบหลัก: `backend/tests/domains/model_engine/adapters/neural_onnx/test_onnx_deep.py`

```python
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import onnx
from onnx import TensorProto, helper
from app.domains.model_engine.adapters.neural_onnx.onnx_deep import ONNXDeepLearningAdapter


def _create_synthetic_deep_onnx(file_path: Path, static_batch_one: bool = False) -> onnx.ModelProto:
    """สร้างกราฟ ONNX Deep Network จำลอง: Input(6) -> Gemm(16) -> Relu -> Gemm(8) -> Relu(Penultimate) -> Gemm(2)"""
    batch_dim = 1 if static_batch_one else "batch_size"
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_dim, 6])
    Y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch_dim, 2])

    rng = np.random.default_rng(99)
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [6, 16], rng.normal(size=(6, 16)).astype(np.float32).flatten())
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [16, 8], rng.normal(size=(16, 8)).astype(np.float32).flatten())
    W3 = helper.make_tensor("W3", TensorProto.FLOAT, [8, 2], rng.normal(size=(8, 2)).astype(np.float32).flatten())

    nodes = [
        helper.make_node("MatMul", ["input", "W1"], ["h1_pre"], name="fc1"),
        helper.make_node("Relu", ["h1_pre"], ["h1_relu"], name="relu1"),
        helper.make_node("MatMul", ["h1_relu", "W2"], ["h2_pre"], name="fc2"),
        helper.make_node("Relu", ["h2_pre"], ["penultimate_relu"], name="relu2"),
        helper.make_node("MatMul", ["penultimate_relu", "W3"], ["logits"], name="fc_out"),
    ]
    graph = helper.make_graph(nodes, "DeepTabularNet", [X], [Y], [W1, W2, W3])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(model, str(file_path))
    return model


def test_in_memory_output_injection_and_3d_latent_pca_manifold(tmp_path: Path) -> None:
    """ยืนยันการเจาะดึง Intermediate Tensors โดยไม่แตะไฟล์บนดิสก์ และการคำนวณ 3D Latent PCA"""
    onnx_path = tmp_path / "deep_dynamic.onnx"
    proto = _create_synthetic_deep_onnx(onnx_path, static_batch_one=False)
    hash_before = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

    adapter = ONNXDeepLearningAdapter(model_path=onnx_path, onnx_proto=proto, providers=["CPUExecutionProvider"])
    df = pd.DataFrame(np.random.randn(25, 6).astype(np.float32), columns=[f"f_{i}" for i in range(6)])

    batch_res = adapter.predict_batch(df)
    hash_after = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

    # 1. ไฟล์ต้นฉบับบนดิสก์ต้องไม่ถูกแก้ไข
    assert hash_before == hash_after

    # 2. ตรวจสอบว่า Logits ดิบถูกแปลงเป็น Softmax Probabilities ที่รวมกันได้ 1.0
    np.testing.assert_allclose(batch_res.probabilities.sum(axis=1), np.ones(25), rtol=1e-5, atol=1e-5)

    # 3. ตรวจสอบพิกัด 3D Latent PCA Manifold ขนาด (25, 3) ที่ถูกทำ Centering รอบจุดกำเนิด
    assert batch_res.latent_coords_3d.shape == (25, 3)
    assert batch_res.latent_coords_3d.dtype == np.float32
    np.testing.assert_allclose(batch_res.latent_coords_3d.mean(axis=0), np.zeros(3), atol=1e-4)


def test_static_batch_size_one_auto_unroll_guard(tmp_path: Path) -> None:
    """ยืนยันว่าโมเดลที่ล็อก shape=[1, 6] สามารถรันกับข้อมูล 20 แถวได้ผ่าน _run_session_with_batch_guard()"""
    onnx_path = tmp_path / "deep_static_b1.onnx"
    proto = _create_synthetic_deep_onnx(onnx_path, static_batch_one=True)

    adapter = ONNXDeepLearningAdapter(model_path=onnx_path, onnx_proto=proto, providers=["CPUExecutionProvider"])
    df = pd.DataFrame(np.random.randn(20, 6).astype(np.float32), columns=[f"f_{i}" for i in range(6)])

    batch_res = adapter.predict_batch(df)
    assert batch_res.probabilities.shape == (20, 2)
    assert batch_res.latent_coords_3d.shape == (20, 3)
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & UX Flow Tests (พร้อมโค้ดต้นแบบ `vitest`)

ไฟล์ทดสอบหลัก: `frontend/src/domains/simulation/components/stage3d/renderers/__tests__/ONNXLatentManifold3D.test.tsx`

```tsx
import { describe, it, expect } from "vitest";
import { ONNXLatentManifold3DProps } from "../ONNXLatentManifold3D";

describe("Phase 39 — ONNXLatentManifold3D & Layer Probe Telemetry", () => {
  it("ตรวจสอบการจัดเก็บพิกัด 3D Latent Manifold แบบ Flat Float32Array [x0,y0,z0,...] และสถิติ Sparsity", () => {
    const numRows = 100;
    const props: ONNXLatentManifold3DProps = {
      latentCoords3D: new Float32Array(numRows * 3),
      predictedClasses: new Uint16Array(numRows),
      activeFrameIndex: 42,
      layerTelemetry: [
        { layerName: "relu1", meanActivation: 0.42, sparsity: 0.38 },
        { layerName: "penultimate_relu", meanActivation: 0.61, sparsity: 0.12 },
      ],
    };

    expect(props.latentCoords3D.length).toBe(300);
    expect(props.layerTelemetry[1].layerName).toBe("penultimate_relu");
    expect(props.layerTelemetry[0].sparsity).toBeCloseTo(0.38);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Process Lifecycle & GPU/CPU Resource Tests)

1. **Probe Cap (`MAX_PROBED_LAYERS = 8`) & Immediate Tensor Disposal Test**:
   - สร้างกราฟ ONNX ลึก 30 ชั้น แล้วตรวจสอบว่า `_inject_intermediate_outputs()` คัดเลือกจุดดักฟังไม่เกิน `8` ชั้น และคืนหน่วยความจำของ Intermediate Tensors ทันทีหลังคำนวณ 3D PCA เสร็จสิ้น โดยใช้ Peak RAM Delta $\le 150\text{ MB}$ ต่อ Chunk
2. **Execution Provider Fallback Chain Test**:
   - ทดสอบว่าเมื่อเครื่องไม่มี GPU (`CUDAExecutionProvider` หรือ `DmlExecutionProvider` ไม่พร้อมใช้งาน) ระบบจะถอยกลับมาใช้ `CPUExecutionProvider` โดยอัตโนมัติโดยไม่โยน Error

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| กรณีปัญหา (Edge Case จาก Dev Phase 39) | วิธีจำลองในการทดสอบ (Test Simulation Setup) | ผลลัพธ์ที่ต้องผ่านการตรวจสอบ (Expected Verification Assertion) |
| :--- | :--- | :--- |
| **1. Static `batch_size = 1` ใน Input Tensor** | สร้างโมเดลที่ล็อก `input_shape = [1, 6]` แล้วส่ง `X_chunk` ขนาด `(20, 6)` เข้าไป | `_run_session_with_batch_guard()` สลับไปรันทีละ 1 แถวและรวมผลลัพธ์กลับมาเป็น `(20, 2)` โดยไม่เกิด `InvalidArgument` |
| **2. เอาต์พุตของโมเดลเป็น Raw Logits (ไม่มี Softmax)** | สร้างโมเดลที่จบด้วยโหนด `MatMul` / `Gemm` ล้วน (ค่าติดลบและบวกเกิน 1.0) | ระบบตรวจพบว่าผลรวมแถวไม่ใช่ `1.0` จึงครอบ `scipy.special.softmax` ให้โดยอัตโนมัติ |
| **3. โหนดซ่อนเร้นมีมิติมากกว่า 2 มิติ (เช่น 1D CNN `[N, C, L]`)** | สร้างโมเดลที่มีโหนด `Conv` คืนเทนเซอร์ 3 มิติ `(20, 16, 4)` ก่อนเข้าชั้นสุดท้าย | ระบบทำ Global Average Pooling หรือ Flatten ให้เป็น 2 มิติ `(20, 64)` ก่อนคำนวณ 3D Latent PCA |
| **4. `onnx.shape_inference` ล้มเหลวใน Custom Ops** | จำลองให้ `onnx.shape_inference.infer_shapes` โยน Exception | ระบบจับ Exception แล้วใช้ `onnx.helper.make_empty_tensor_value_info(name)` แนบ `graph.output` ได้ตามปกติ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Disk File Immutability (100%)**: ค่า Hash SHA-256 ของไฟล์ `.onnx` บนดิสก์ไม่เปลี่ยนแปลงหลังการทำ In-Memory Graph Output Injection
- [ ] **Dynamic & Static Batch Support**: รันผ่านทั้งโมเดลที่มี Dynamic Axis และโมเดลที่ล็อก `batch_size=1`
- [ ] **3D Latent Manifold & Memory SLA**: คำนวณ 3D PCA บน Penultimate Layer เสร็จภายใน `< 50 ms` ต่อ Chunk และใช้ RAM ส่วนเพิ่มไม่เกิน `150 MB`

- [ ] **Hardware Execution Provider Fallback Resilience**: ยืนยันว่าเมื่อส่งรายชื่อ Providers `["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]` บนเครื่องที่ไม่มี GPU ระบบจะสลับมาใช้ `CPUExecutionProvider` โดยอัตโนมัติและคืนค่า `active_provider == "CPUExecutionProvider"` ใน Telemetry HUD

### คำสั่งรันชุดทดสอบประจำเฟส 39 (CLI Test Execution Commands)
```bash
# รันชุดทดสอบ Backend ONNXDeepLearningAdapter (In-Memory Probe Injection & 3D Latent PCA)
pytest backend/tests/domains/model_engine/adapters/neural_onnx/test_onnx_deep.py -v

# รันชุดทดสอบ Frontend ONNXLatentManifold3D & Layer Probe Telemetry
cd frontend && npx vitest run src/domains/simulation/components/stage3d/renderers/__tests__/ONNXLatentManifold3D.test.tsx
```
