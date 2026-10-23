# Test Phase 50: 2D Point Intelligence Integration for Linear & Generalized Linear Models

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 50)
ทดสอบความแม่นยำทางคณิตศาสตร์ของการเชื่อมต่อฟีเจอร์ Point Intelligence (A, B, C, D) เข้ากับโมเดลกลุ่ม Linear ได้แก่ ความถูกต้องของจุดฉาย Orthogonal Projection Spoke, การตรวจจับสถานะ Margin Violation ของ Dual-Ring, และการแสดงผล Residual Stem

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

```python
# backend/tests/phase_50/test_linear_point_intelligence_math.py
import pytest
import numpy as np

def test_orthogonal_projection_spoke_coordinates():
    """ทดสอบพิกัดปลายเส้นตั้งฉากจากจุดข้อมูลไปยังเส้นตรง w1*x1 + w2*x2 + b = 0"""
    w1, w2, b = 3.0, 4.0, -10.0 # เส้นตรง 3x + 4y - 10 = 0
    point = np.array([0.0, 0.0], dtype=np.float32) # (0, 0)
    
    # ระยะทางตั้งฉากทฤษฎี d = |-10| / sqrt(3^2 + 4^2) = 10 / 5 = 2.0
    expected_d = 2.0
    # จุดฉายตั้งฉาก p_proj = (0, 0) - (-10 / 25) * [3, 4] = [1.2, 1.6]
    expected_proj = np.array([1.2, 1.6], dtype=np.float32)

    val = w1 * point[0] + w2 * point[1] + b
    norm_sq = w1**2 + w2**2
    p_proj = point - (val / norm_sq) * np.array([w1, w2])
    d_perp = np.linalg.norm(point - p_proj)

    assert abs(d_perp - expected_d) < 1e-5
    assert np.allclose(p_proj, expected_proj, atol=1e-5)

def test_margin_corridor_violation_status():
    """ทดสอบเงื่อนไขการระบุสถานะ Margin Corridor สำหรับ Dual-Ring"""
    weights = np.array([1.0, 1.0], dtype=np.float32)
    bias = 0.0
    
    # จุดที่ 1: z = 1(0.3) + 1(0.4) = 0.7 (|z| <= 1 -> Inside Margin Corridor)
    # จุดที่ 2: z = 1(1.0) + 1(1.0) = 2.0 (|z| > 1 -> Confident Safe)
    p1 = np.array([0.3, 0.4])
    p2 = np.array([1.0, 1.0])

    z1 = np.dot(weights, p1) + bias
    z2 = np.dot(weights, p2) + bias

    assert abs(z1) <= 1.0 # ต้องแสดง Amber Ring
    assert abs(z2) > 1.0  # แสดง Normal Ring
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/LinearOrthogonalSpoke2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { LinearOrthogonalSpoke2D } from '../../components/surfaces_2d/LinearOrthogonalSpoke2D';

describe('LinearOrthogonalSpoke2D Rendering Verification', () => {
  it('renders dashed projection lines connecting misclassified points to hyperplane', () => {
    const mockMisclassifiedPoints = [
      { id: 1, x: 0, y: 0, projX: 1.2, projY: 1.6, distance: 2.0 }
    ];

    const { container } = render(
      <svg>
        <LinearOrthogonalSpoke2D spokes={mockMisclassifiedPoints} />
      </svg>
    );

    const line = container.querySelector('line');
    expect(line).not.toBeNull();
    expect(Number(line?.getAttribute('x1'))).toBe(0);
    expect(Number(line?.getAttribute('y1'))).toBe(0);
    expect(Number(line?.getAttribute('x2'))).toBeCloseTo(1.2);
    expect(Number(line?.getAttribute('y2'))).toBeCloseTo(1.6);
  });
});
```

---

## 4. Performance & Memory Gate
- การคำนวณเวกเตอร์ Orthogonal Spokes สำหรับจุดผิด 500 จุด ต้องใช้เวลาน้อยกว่า $2$ ms
- การเรนเดอร์ร่วมกับเส้นตรง Decision Boundary และ Hexbins ทำงานที่ 60 FPS อย่างเสถียร

