# Test Phase 55: 2D Point Intelligence Integration for Neural Networks, Pipelines & Universal ONNX

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 55)
ทดสอบความถูกต้องของระบบ Point Intelligence สำหรับ Neural Networks, Pipeline Scalers, และ ONNX: การแปลงพิกัด Trajectory ข้ามปริภูมิ (Raw $\leftrightarrow$ Scaled Space), การคำนวณ Logit Gap $\Delta z$, และการระบุพิกัด Data Desert ใน Hexbin Grid

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

```python
# backend/tests/phase_55/test_neural_point_intelligence_math.py
import pytest
import numpy as np

def test_softmax_logit_gap_calculation():
    """ทดสอบการคำนวณส่วนต่าง Logit Gap (Delta z) ก่อน Softmax"""
    # กรณีที่ 1: Logits ห่างกันมาก (ฟันธงเด็ดขาด)
    logits_confident = np.array([5.0, 1.0, 0.5])
    sorted_logits = np.sort(logits_confident)[::-1]
    gap_confident = sorted_logits[0] - sorted_logits[1]
    assert abs(gap_confident - 4.0) < 1e-5

    # กรณีที่ 2: Logits สูสีกันมาก (เกิดความลังเล)
    logits_uncertain = np.array([2.1, 2.0, -1.0])
    sorted_uncertain = np.sort(logits_uncertain)[::-1]
    gap_uncertain = sorted_uncertain[0] - sorted_uncertain[1]
    assert abs(gap_uncertain - 0.1) < 1e-5

def test_data_desert_hazard_detection():
    """ทดสอบเงื่อนไขตรวจจับ Data Desert: Count == 0 แต่ Max Probability > 0.85"""
    hexbin_counts = np.array([0, 15, 0])
    max_probabilities = np.array([0.94, 0.92, 0.55])

    hazards = (hexbin_counts == 0) & (max_probabilities > 0.85)

    assert hazards[0] == True   # เข้าข่าย Data Desert (Overconfident ในพื้นที่ว่าง)
    assert hazards[1] == False  # มีข้อมูลจริงหนาแน่น ไม่ใช่ทะเลทราย
    assert hazards[2] == False  # เป็นพื้นที่ว่าง แต่โมเดลก็ลังเล (P=0.55) ปลอดภัย
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/SoftmaxLogitGapRing2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SoftmaxLogitGapRing2D } from '../../components/surfaces_2d/SoftmaxLogitGapRing2D';

describe('SoftmaxLogitGapRing2D Verification', () => {
  it('renders diffuse glow filter when logit gap is very small', () => {
    const { container } = render(
      <svg>
        <SoftmaxLogitGapRing2D
          x={0.0}
          y={0.0}
          logitGap={0.15} // ใกล้เคียงกันมาก
          radius={0.12}
        />
      </svg>
    );

    const circle = container.querySelector('circle');
    expect(circle).not.toBeNull();
    // ต้องมี filter diffuse หรือคลาส diffuse glow
    expect(circle?.getAttribute('filter')).toContain('diffuse');
  });
});
```

---

## 4. Performance & Memory Gate
- การแปลงพิกัดไป-กลับระหว่าง Raw Space และ Scaled Space สำหรับจุด 10,000 จุด ต้องเสร็จในเวลาน้อยกว่า $3$ ms
- อัตราเฟรมคงที่ที่ 60 FPS ขณะแสดงผล Data Desert Warning Badges

