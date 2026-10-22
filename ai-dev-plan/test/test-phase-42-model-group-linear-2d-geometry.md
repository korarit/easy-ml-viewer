# Test Phase 42: 2D Decision Geometries & Visualizers for Linear & Generalized Linear Models

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 42)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-42-model-group-linear-2d-geometry.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-42-model-group-linear-2d-geometry.md) เพื่อยืนยันความถูกต้องทางคณิตศาสตร์ 100% ของการแสดงผลระนาบ 2 มิติสำหรับกลุ่มโมเดลเชิงเส้น (**Linear & Generalized Linear Models**) ครอบคลุมการคำนวณจุดตัดของเส้นแบ่งเขตการตัดสินใจ ($w_1 x_1 + w_2 x_2 + b_{\text{eff}} = 0$), ขนาดและทิศทางของเวกเตอร์ตั้งฉาก ($\mathbf{w}_{1,2}$), ความกว้างของทางเดินมาร์จิน ($W_{\text{2D}} = 2/\|\mathbf{w}\|$), การตรวจจับฟีเจอร์ที่เป็นศูนย์จากโทษปรับ $L_1$ ของ Lasso, และการเรนเดอร์คอมโพเนนต์ 2D บน Frontend ที่ 60 FPS

### ขอบเขตโมเดลที่ถูกทดสอบในเฟสนี้
- `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDClassifier`, `SGDRegressor`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Mathematical Parity Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_42/
├── conftest.py
├── test_linear_boundary_formula_math.py
├── test_logistic_sigmoid_isolines_math.py
├── test_sgd_margin_corridor_math.py
└── test_lasso_zero_axis_detection.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_42/test_linear_boundary_formula_math.py
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from app.domains.model_engine.adapters.linear.linear_2d_extractor import Linear2DExtractor

def test_linear_decision_line_exact_slope_and_intercept() -> None:
    """ทดสอบว่าสูตรเส้นตรง 2D คำนวณความชันและจุดตัดแกนตรงกับสมการทฤษฎีเป๊ะ"""
    w1, w2, b = 2.0, -4.0, 1.0
    pinned_features = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    weights = np.array([w1, w2, 0.5], dtype=np.float32)

    # b_eff = b + 0.5 * 5.0 = 1.0 + 2.5 = 3.5
    b_eff = b + weights[2] * pinned_features[2]
    expected_slope = -w1 / w2 # -2.0 / -4.0 = 0.5
    expected_intercept = -b_eff / w2 # -3.5 / -4.0 = 0.875

    result = Linear2DExtractor.extract_2d_line_params(
        weights=weights, bias=b, x_idx=0, y_idx=1, pinned=pinned_features
    )
    assert abs(result["slope"] - expected_slope) < 1e-6
    assert abs(result["intercept"] - expected_intercept) < 1e-6

def test_lasso_zero_weight_generates_horizontal_boundary() -> None:
    """ทดสอบกรณี Lasso บีบให้น้ำหนัก w1 เป็น 0 เส้นแบ่งเขตต้องขนานกับแกน X1 (slope = 0)"""
    weights = np.array([0.0, 3.0], dtype=np.float32) # w1 = 0
    bias = -6.0

    result = Linear2DExtractor.extract_2d_line_params(
        weights=weights, bias=bias, x_idx=0, y_idx=1, pinned=np.zeros(2)
    )
    assert result["is_zero_axis_x"] is True
    assert abs(result["slope"]) < 1e-6 # เส้นนอน
    assert abs(result["intercept"] - 2.0) < 1e-6 # x2 = -(-6)/3 = 2.0

def test_sgd_margin_width_formula_exact_parity() -> None:
    """ทดสอบความกว้าง Margin ของ SGD: W = 2 / sqrt(w1^2 + w2^2)"""
    w1, w2 = 3.0, 4.0 # norm = 5.0
    weights = np.array([w1, w2], dtype=np.float32)

    margin_width = Linear2DExtractor.compute_margin_width(weights, x_idx=0, y_idx=1)
    assert abs(margin_width - (2.0 / 5.0)) < 1e-6 # 0.40
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/LinearBoundary2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { LinearBoundary2D } from '../../components/surfaces_2d/LinearBoundary2D';

describe('LinearBoundary2D SVG Geometry Rendering', () => {
  it('renders line with correct endpoints across xRange', () => {
    // x2 = -0.5 * x1 + 2.0 บน xRange [0, 10]
    // x1 = 0 -> x2 = 2.0, x1 = 10 -> x2 = -3.0
    const { container } = render(
      <svg>
        <LinearBoundary2D
          w1={1.0}
          w2={2.0}
          bEff={-4.0}
          xRange={[0, 10]}
          yRange={[-5, 5]}
          showNormalVector={true}
        />
      </svg>
    );

    const line = container.querySelector('line');
    expect(line).toBeTruthy();
    expect(line?.getAttribute('x1')).toBe('0');
    expect(line?.getAttribute('y1')).toBe('2');
    expect(line?.getAttribute('x2')).toBe('10');
    expect(line?.getAttribute('y2')).toBe('-3');
  });
});
```

---

## 4. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E42.1** | น้ำหนัก $w_2 = 0$ แต่ $w_1 \ne 0$ | เกิดข้อผิดพลาดหารด้วยศูนย์ (Division by Zero ในสูตร Slope) | สลับไปวาดเส้นดิ่ง $x_1 = -b_{\text{eff}}/w_1$ อัตโนมัติ |
| **E42.2** | น้ำหนักทั้ง $w_1 = 0$ และ $w_2 = 0$ (Lasso ตัดทิ้งทั้ง 2 ฟีเจอร์) | ไม่มีเส้นแบ่งเขตในระนาบ 2D | ซ่อนเส้น และแสดงป้ายเตือน *"Both visualized features zeroed out"* |
| **E42.3** | เส้นแบ่งเขตอยู่นอกช่วงพิกัด $[x_{\min}, x_{\max}]$ และ $[y_{\min}, y_{\max}]$ | จุดตัดอยู่นอก Frustum จอภาพ | แสดงลูกศรบอกทิศทางบนขอบจอว่าเส้นแบ่งเขตอยู่ห่างออกไปทางใด |
| **E42.4** | โมเดล Logistic Regression หลายคลาส (Multiclass 10 คลาส) | เส้นแบ่งเขตระหว่างคู่คลาสซ้อนทับกันจนมองไม่เห็น | แสดงเฉพาะเส้นขอบเขตของคลาสที่เป็นผู้ชนะในระนาบ |
| **E42.5** | ฟีเจอร์ที่ถูก Pin มีค่า Extreme Outlier | ค่า $b_{\text{eff}}$ พุ่งสูงจนเส้นหลุดกรอบ | ปรับสเกลให้อยู่ในกรอบ หรือแสดงป้ายเตือน Pin Outlier |
| **E42.6** | สลับคู่แกน $X-Y$ ในโหมด 2D ขณะกำลังเล่น VCR ที่ 60 FPS | คำนวณสมการเส้นใหม่ไม่ทัน เกิดการกระตุก | คำนวณค่า $w_1, w_2, b_{\text{eff}}$ ล่วงหน้าและแคชไว้ใน Store |

---

## 5. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit Test ครอบคลุม 6 โมเดลเชิงเส้นใน `backend/tests/phase_42/` ผ่าน 100%
- [ ] ชุดทดสอบ Component Test ใน `frontend/src/domains/stage3d/__tests__/surfaces_2d/LinearBoundary2D.test.tsx` ผ่าน 100%
- [ ] การเรนเดอร์เส้นและมาร์จินในโหมด 2D รักษาความเร็วที่ **60 FPS**
- [ ] ผ่านเกณฑ์ Edge Case Matrix ทั้ง 6 กรณีครบถ้วน

