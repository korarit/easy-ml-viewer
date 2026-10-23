# Test Phase 53: 2D Point Intelligence Integration for Trees & Bagging Ensembles

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 53)
ตรวจสอบความถูกต้องของระบบ Point Intelligence สำหรับโมเดลตระกูล Trees & Bagging: การคำนวณสัดส่วนเสียงโหวต Forest Agreement Variance, การสเกลขนาดวงแหวนตาม AdaBoost Sample Weights, และการจัดตำแหน่ง Leaf Minority Spokes

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

```python
# backend/tests/phase_53/test_trees_point_intelligence_math.py
import pytest
import numpy as np

def test_forest_consensus_variance_calculation():
    """ทดสอบการคำนวณ Agreement Variance จากเสียงโหวตของ Random Forest"""
    # กรณีที่ 1: 100 ต้น โหวตคลาส A 100 เสียง -> Variance = 0
    votes_unanimous = np.array([100, 0])
    variance_unanimous = 1.0 - (np.max(votes_unanimous) / 100.0)
    assert abs(variance_unanimous - 0.0) < 1e-5

    # กรณีที่ 2: 100 ต้น โหวตคลาส A 52 เสียง, คลาส B 48 เสียง -> Variance = 0.48
    votes_split = np.array([52, 48])
    variance_split = 1.0 - (np.max(votes_split) / 100.0)
    assert abs(variance_split - 0.48) < 1e-5

def test_adaboost_sample_weight_ring_scaling():
    """ทดสอบการแปลงค่าน้ำหนักตัวอย่าง sample_weight เป็นรัศมีของวงแหวน"""
    weights = np.array([0.1, 0.4, 0.9])
    base_radius = 0.1
    mean_weight = np.mean(weights) # ~0.4667

    scaled_radii = base_radius * np.sqrt(weights / mean_weight)
    # จุดที่มี weight สูงสุด (0.9) ต้องมีรัศมีวงแหวนใหญ่กว่าจุดแรก (0.1) อย่างมีนัยสำคัญ
    assert scaled_radii[2] > scaled_radii[0] * 2.5
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/ForestConsensusRing2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ForestConsensusRing2D } from '../../components/surfaces_2d/ForestConsensusRing2D';

describe('ForestConsensusRing2D Component Verification', () => {
  it('applies stroke-dasharray styling when forest agreement variance is high', () => {
    const { container } = render(
      <svg>
        <ForestConsensusRing2D
          x={0.5}
          y={0.5}
          variance={0.48} // เสียงแตก
          radius={0.15}
        />
      </svg>
    );

    const circle = container.querySelector('circle');
    expect(circle).not.toBeNull();
    // ต้องมี strokeDasharray สำหรับแสดงความไม่แน่นอน
    expect(circle?.getAttribute('stroke-dasharray')).not.toBeNull();
  });
});
```

---

## 4. Performance & Memory Gate
- การคำนวณ Leaf Bounded Hexbin Grid ที่มีกล่องแบ่ง 128 กล่อง ต้องเสร็จในเวลาไม่เกิน $15$ ms
- อัตราเฟรมคงที่ที่ 60 FPS เมื่อสลับดูโครงสร้างต้นไม้

