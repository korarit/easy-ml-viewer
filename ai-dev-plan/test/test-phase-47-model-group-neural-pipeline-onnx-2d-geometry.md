# Test Phase 47: 2D Decision Geometries & Visualizers for Neural Networks, Pipelines & Universal ONNX

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 47)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-47-model-group-neural-pipeline-onnx-2d-geometry.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-47-model-group-neural-pipeline-onnx-2d-geometry.md) เพื่อยืนยันความถูกต้องของเรขาคณิต 2 มิติสำหรับกลุ่มโครงข่ายประสาทเทียม ไพป์ไลน์ และโมเดล ONNX (**MLP, Pipeline, ONNX ML, ONNX Deep**) ครอบคลุมการคำนวณสมการเส้นตรงของ Hidden Neurons ในชั้นแรก, การแปลงพิกัดสเกลข้อมูลใน Pipeline (Raw vs Scaled), การฉายภาพเวกเตอร์แฝงลงบน 2D Latent SVD Plane, และการทำงานที่ความเร็ว 60 FPS ใน Frontend

### ขอบเขตโมเดลที่ถูกทดสอบในเฟสนี้
- `MLPClassifier`, `MLPRegressor`, `Pipeline`, `ColumnTransformer`, `ONNX Classical ML`, `ONNX Deep Learning`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Mathematical Parity Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_47/
├── conftest.py
├── test_mlp_first_layer_hyperplanes_math.py
├── test_pipeline_2d_raw_to_scaled_inversion.py
└── test_onnx_deep_latent_svd_projection.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_47/test_mlp_first_layer_hyperplanes_math.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from app.domains.model_engine.adapters.neural_onnx.neural_2d_extractor import Neural2DExtractor

def test_mlp_hidden_neuron_lines_match_first_layer_weights() -> None:
    """ทดสอบว่าค่าความชันและจุดตัดของเส้นตรง Hidden Neuron ในชั้นแรกตรงกับ coefs_[0] และ intercepts_[0]"""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=10, random_state=42).fit(X, y)

    lines = Neural2DExtractor.extract_mlp_first_layer_hyperplanes(
        mlp, x_idx=0, y_idx=1, pinned_features=np.zeros(2, dtype=np.float32)
    )

    assert len(lines) == 8
    for i, line in enumerate(lines):
        assert abs(line["w1"] - mlp.coefs_[0][0, i]) < 1e-6
        assert abs(line["w2"] - mlp.coefs_[0][1, i]) < 1e-6
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/MlpNeuronHyperplanes2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MlpNeuronHyperplanes2D } from '../../components/surfaces_2d/MlpNeuronHyperplanes2D';

describe('MlpNeuronHyperplanes2D Component Rendering', () => {
  it('renders dashed lines for first layer neurons', () => {
    const hyperplanes = [
      { neuron_idx: 0, w1: 1.0, w2: -1.0, b_eff: 0.0 },
      { neuron_idx: 1, w1: 0.5, w2: 2.0, b_eff: -1.0 },
    ];

    const { container } = render(
      <svg>
        <MlpNeuronHyperplanes2D
          hyperplanes={hyperplanes}
          xRange={[-5, 5]}
          yRange={[-5, 5]}
          visible={true}
        />
      </svg>
    );

    const lines = container.querySelectorAll('line');
    expect(lines.length).toBe(2);
  });
});
```

---

## 4. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E47.1** | โครงข่าย MLP มี Hidden Layer แรกขนาดใหญ่ (1,024 Neurons) | การวาดเส้น 1,024 เส้นใน 2D ทำให้ SVG พัง | แสดงเฉพาะ Top-32 Neurons ที่มีค่าน้ำหนักสูงสุดในชั้นถัดไป |
| **E47.2** | เซลล์ประสาททั้งหมดในชั้นแรกเป็น Dead ReLU (Activation 0) | โมเดลทำนายค่าคงที่สม่ำเสมอ | แสดงป้ายแจ้งเตือนสีเหลือง *"All neurons dead in this 2D region"* |
| **E47.3** | Pipeline มี Transformer ซ้อนกัน 5 ชั้น (Imputer, Scaler, Poly, PCA) | การถอด Inverse Transform ล้มเหลว | ถอดค่า Inverse ทีละชั้นแบบย้อนกลับอย่างเป็นระบบ (LIFO) |
| **E47.4** | โมเดล ONNX Deep Learning รับ Input ขนาดแปลกประหลาด | การ Hook กราฟล้มเหลว | ตรวจสอบ Tensor Shape อัตโนมัติและ Reshape ให้ตรงกับโมเดล |
| **E47.5** | สลับปุ่ม Raw vs Scaled Space ขณะเล่น VCR 60 FPS | สเกลตัวเลขกระโดดฉับพลัน | ใช้ Smooth Coordinate Interpolation ปรับสเกลนุ่มนวล |
| **E47.6** | โมเดล ONNX ไม่มี Metadata ของคลาสแนบมาในโมเดล | แสดงชื่อคลาสเป็นตัวเลขดิบ | Fallback กำหนดชื่อคลาสเป็น `Class 0`, `Class 1` อัตโนมัติ |

---

## 5. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit Test ฝั่ง Backend ใน `backend/tests/phase_47/` ผ่าน 100%
- [ ] ชุดทดสอบ Component Test ใน `frontend/src/domains/stage3d/__tests__/surfaces_2d/MlpNeuronHyperplanes2D.test.tsx` ผ่าน 100%
- [ ] สลับดู Raw vs Scaled Space ของ Pipeline ได้อย่างถูกต้องไร้รอยต่อ
- [ ] ผ่านเกณฑ์ Edge Case Matrix ทั้ง 6 กรณีครบถ้วน

