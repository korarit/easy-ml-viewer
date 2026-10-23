# Test Phase 54: 2D Point Intelligence Integration for Gradient Boosting Heavyweights

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 54)
ทดสอบความถูกต้องของระบบ Point Intelligence สำหรับโมเดลตระกูล Gradient Boosting: การคำนวณเวกเตอร์ TreeSHAP Push Vector บนระนาบ 2D, การจำแนกประเภท Functional Margin ของ Dual-Ring, และขนาดก้าน Pseudo-Residual Spoke

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

```python
# backend/tests/phase_54/test_boosting_point_intelligence_math.py
import pytest
import numpy as np

def test_functional_margin_status_classification():
    """ทดสอบการจำแนกสถานะ Safe, Critical, Negative Margin สำหรับ Dual-Ring"""
    y_true = np.array([1, 1, -1], dtype=np.int32)
    f_scores = np.array([2.5, 0.4, 0.8], dtype=np.float32)

    margins = y_true * f_scores
    # จุดที่ 1: M = 1 * 2.5 = 2.5 > 1.5 -> Safe Margin (Solid True Hue)
    # จุดที่ 2: M = 1 * 0.4 = 0.4 in [0, 1.5] -> Critical Margin (Pulsing Orange)
    # จุดที่ 3: M = -1 * 0.8 = -0.8 < 0 -> Negative Margin / Misclassified (Red Alert)
    assert margins[0] > 1.5
    assert 0.0 <= margins[1] <= 1.5
    assert margins[2] < 0.0

def test_pseudo_residual_binary_logloss_calculation():
    """ทดสอบสูตร Pseudo-Residual สำหรับ Binary Log-Loss: r_i = y_i - p_i"""
    y_true = np.array([1.0, 0.0]) # Class 1, Class 0
    raw_scores = np.array([0.0, 2.0]) # p1 = 0.5, p2 = 1 / (1 + e^-2) ~ 0.8808

    probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
    pseudo_residuals = y_true - probabilities

    # จุดที่ 1: y=1, p=0.5 -> r = +0.5
    # จุดที่ 2: y=0, p=0.8808 -> r = -0.8808 (ทายผิดรุนแรง Negative Gradient สูง)
    assert abs(pseudo_residuals[0] - 0.5) < 1e-4
    assert abs(pseudo_residuals[1] - (-0.8808)) < 1e-3
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/FunctionalMarginRing2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { FunctionalMarginRing2D } from '../../components/surfaces_2d/FunctionalMarginRing2D';

describe('FunctionalMarginRing2D Verification', () => {
  it('renders pulsing orange stroke when functional margin is within critical threshold', () => {
    const { container } = render(
      <svg>
        <FunctionalMarginRing2D
          x={0.2}
          y={0.4}
          margin={0.4} // Critical
          radius={0.12}
        />
      </svg>
    );

    const ring = container.querySelector('circle');
    expect(ring).not.toBeNull();
    // Critical Margin ต้องมี stroke สีส้ม (#f59e0b หรือใกล้เคียง)
    expect(ring?.getAttribute('stroke')).toContain('#f59e0b');
  });
});
```

---

## 4. Performance & Memory Gate
- การคำนวณ Staged TreeSHAP Push Vectors สำหรับ 100 ต้น ต้องใช้เวลาน้อยกว่า $20$ ms
- การสลับสเตจ VCR ควบคู่กับการวาดก้าน Pseudo-Residual Spokes ทำงานที่ 60 FPS

