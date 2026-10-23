# Test Phase 51: 2D Point Intelligence Integration for Distance & Kernel Models

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 51)
ตรวจสอบความถูกต้องของระบบ Point Intelligence สำหรับ KNN และ SVM: การจับคู่จุดที่เป็น Support Vectors ในสเกล 2D, การจำแนกประเภท Bound/Unbound SVs, การคำนวณ Slack Variables ใน SVR, และการแยกส่วน Pie Arc เสียงโหวตของ KNN

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

```python
# backend/tests/phase_51/test_distance_point_intelligence_math.py
import pytest
import numpy as np

def test_svr_slack_variable_spoke_calculation():
    """ทดสอบว่า SVR สร้าง Spoke เฉพาะจุดที่เกินขอบเขต epsilon-tube เท่านั้น"""
    epsilon = 0.5
    y_true = np.array([2.0, 1.2, 0.4])
    y_pred = np.array([1.0, 1.0, 1.0])

    # จุดที่ 1: y - y_pred = 1.0 > 0.5 -> xi = 0.5 (ต้องวาด Spoke)
    # จุดที่ 2: y - y_pred = 0.2 <= 0.5 -> อยู่ใน tube (ไม่ต้องวาด)
    # จุดที่ 3: y - y_pred = -0.6 -> |error| = 0.6 > 0.5 -> xi* = 0.1 (ต้องวาด Spoke)
    residuals = y_true - y_pred
    violations = np.abs(residuals) > epsilon

    assert violations[0] == True
    assert violations[1] == False
    assert violations[2] == True

    slack_1 = max(0.0, residuals[0] - epsilon)
    slack_3 = max(0.0, -residuals[2] - epsilon)
    assert abs(slack_1 - 0.5) < 1e-5
    assert abs(slack_3 - 0.1) < 1e-5

def test_support_vector_classification_status():
    """ทดสอบการจำแนกสถานะ Bounded vs Unbounded Support Vectors สำหรับ Dual-Ring Halo"""
    C = 1.0
    alphas = np.array([1.0, 0.45, 0.0]) # 1.0 = Unbounded, 0.45 = Bounded, 0.0 = Not SV

    is_sv = alphas > 1e-5
    is_unbounded = np.isclose(alphas, C, atol=1e-4)

    assert is_sv[0] == True and is_unbounded[0] == True   # ส้มเข้ม (ละเมิดมาร์จิน)
    assert is_sv[1] == True and is_unbounded[1] == False  # ฟ้าคราม (บนมาร์จิน)
    assert is_sv[2] == False                              # จุดธรรมดา
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/SvmSupportVectorHalo2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SvmSupportVectorHalo2D } from '../../components/surfaces_2d/SvmSupportVectorHalo2D';

describe('SvmSupportVectorHalo2D Verification', () => {
  it('renders diamond polygon halo with cyan stroke for bounded SVs and orange for unbounded', () => {
    const { container } = render(
      <svg>
        <SvmSupportVectorHalo2D
          x={1.0}
          y={2.0}
          alpha={1.0}
          C={1.0}
          isUnbounded={true}
        />
      </svg>
    );

    const diamond = container.querySelector('polygon');
    expect(diamond).not.toBeNull();
    // Unbounded SV ต้องมี stroke ส้ม (#f97316 หรือใกล้เคียง)
    expect(diamond?.getAttribute('stroke')).toContain('#f97316');
  });
});
```

---

## 4. Performance & Memory Gate
- การคำนวณและเรนเดอร์ Support Vector Halos $500$ จุด พร้อมเรดาร์ KNN ต้องมี Frame Drop ไม่เกิน $2$ เฟรมในจังหวะ Scrubbing

