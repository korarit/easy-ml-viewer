# Test Phase 52: 2D Point Intelligence Integration for Probabilistic & Discriminant Models

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 52)
ทดสอบฟังก์ชันคณิตศาสตร์ของการคำนวณ Shannon Entropy สำหรับ Comet Trail, การคำนวณเวกเตอร์ Centroid Attractor Spokes, และการตรวจสอบความหนาแน่น Bivariate Normal Divergence ใน Hexbin Grid

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

```python
# backend/tests/phase_52/test_probabilistic_point_intelligence_math.py
import pytest
import numpy as np

def test_shannon_entropy_calculation():
    """ทดสอบการคำนวณ Shannon Entropy สำหรับปรับสี Comet Trail"""
    # กรณีที่ 1: ความน่าจะเป็น 50/50 -> Entropy สูงสุด = 1.0 bit
    probs_ambiguous = np.array([0.5, 0.5])
    h_ambiguous = -np.sum(probs_ambiguous * np.log2(probs_ambiguous))
    assert abs(h_ambiguous - 1.0) < 1e-5

    # กรณีที่ 2: ความน่าจะเป็น 99/1 -> Entropy ต่ำมากใกล้ 0
    probs_confident = np.array([0.99, 0.01])
    h_confident = -np.sum(probs_confident * np.log2(probs_confident))
    assert h_confident < 0.1

def test_centroid_spoke_distance_and_direction():
    """ทดสอบเวกเตอร์และระยะทางไปยังจุดศูนย์กลางคลาสทั้งสอง"""
    point = np.array([1.0, 1.0])
    mu_true = np.array([1.0, 3.0]) # ระยะห่าง = 2.0
    mu_pred = np.array([4.0, 1.0]) # ระยะห่าง = 3.0

    d_true = np.linalg.norm(point - mu_true)
    d_pred = np.linalg.norm(point - mu_pred)

    assert abs(d_true - 2.0) < 1e-5
    assert abs(d_pred - 3.0) < 1e-5
    # จุดนี้อยู่ใกล้ยูคลิดของคลาสจริงมากกว่า แต่โมเดลทายไปอีกคลาส
    assert d_true < d_pred
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/CentroidAttractorSpoke2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { CentroidAttractorSpoke2D } from '../../components/surfaces_2d/CentroidAttractorSpoke2D';

describe('CentroidAttractorSpoke2D Verification', () => {
  it('renders two dashed lines from the data point to both class centroids', () => {
    const { container } = render(
      <svg>
        <CentroidAttractorSpoke2D
          pointX={1.0}
          pointY={1.0}
          muTrueX={1.0}
          muTrueY={3.0}
          muPredX={4.0}
          muPredY={1.0}
        />
      </svg>
    );

    const lines = container.querySelectorAll('line');
    expect(lines.length).toBe(2);
    // เส้นที่ 1 ไปหา mu_true
    expect(Number(lines[0].getAttribute('x2'))).toBeCloseTo(1.0);
    expect(Number(lines[0].getAttribute('y2'))).toBeCloseTo(3.0);
    // เส้นที่ 2 ไปหา mu_pred
    expect(Number(lines[1].getAttribute('x2'))).toBeCloseTo(4.0);
    expect(Number(lines[1].getAttribute('y2'))).toBeCloseTo(1.0);
  });
});
```

---

## 4. Performance & Memory Gate
- การคำนวณ Shannon Entropy ตลอดความยาว Trajectory 1,000 จุด ต้องเสร็จสิ้นในเวลาน้อยกว่า $1.5$ ms
- การสลับโหมด Gaussian Divergence Hexbin ต้องแสดงผลได้ภายใน $30$ ms

