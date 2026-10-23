# Test Phase 49: Universal 2D Point Intelligence Engine (Comet Trails, Dual-Ring Glyphs, Error Lens & Hexbin Density)

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 49)
ยืนยันความถูกต้อง 100% ของสถาปัตยกรรม Point Intelligence Engine ในระนาบ 2 มิติ ครอบคลุมความถูกต้องของอัลกอริทึม Hexbin Tessellation, การคำนวณหาง Comet Decay Kernel, การแยกแยะสีของวงแหวนคู่ Dual-Ring, และการกรอง Error Lens ภายใต้เงื่อนไข $60$ FPS

---

## 2. รายการทดสอบฝั่ง Backend Unit Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_49/
├── conftest.py
├── test_hexbin_tessellation_math.py
├── test_point_binning_aggregation.py
└── test_point_intelligence_dto_serialization.py
```

### 2.2 โค้ดทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_49/test_hexbin_tessellation_math.py
import pytest
import numpy as np
from app.domains.spatial_projection.services.hexbin_engine import HexbinEngine

def test_hexbin_grid_coverage_and_coordinates():
    """ยืนยันว่าการสร้างจุดศูนย์กลางของ Hexbin สอดคล้องกับขนาดรัศมี R_hex อย่างถูกต้อง"""
    engine = HexbinEngine(radius=0.5, x_bounds=(-5.0, 5.0), y_bounds=(-5.0, 5.0))
    cells = engine.generate_empty_grid()
    assert len(cells) > 0
    # ระยะห่างแนวแกน X ระหว่างคอลัมน์คู่ต้องเท่ากับ sqrt(3) * R
    expected_dx = np.sqrt(3) * 0.5
    assert abs(cells[1].center_x - cells[0].center_x - expected_dx) < 1e-4

def test_point_aggregation_counts_and_error_rates():
    """ทดสอบการจัดกลุ่มจุดลงใน Hexbin พร้อมคำนวณอัตราความผิดพลาด"""
    points_x = np.array([0.0, 0.1, 0.05, 4.0], dtype=np.float32)
    points_y = np.array([0.0, 0.05, 0.1, 4.0], dtype=np.float32)
    y_true = np.array([1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1]) # จุดที่ 2 ทายผิด

    engine = HexbinEngine(radius=0.5, x_bounds=(-5.0, 5.0), y_bounds=(-5.0, 5.0))
    result = engine.aggregate_points(points_x, points_y, y_true, y_pred)

    center_cell = next(c for c in result if abs(c.center_x) < 0.3 and abs(c.center_y) < 0.3)
    assert center_cell.count == 3
    # ใน 3 จุด มีทายผิด 1 จุด -> Error Rate = 1/3
    assert abs(center_cell.error_rate - (1.0 / 3.0)) < 1e-4
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/DualRingPoint2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DualRingPoint2D } from '../../components/surfaces_2d/DualRingPoint2D';

describe('DualRingPoint2D Component Verification', () => {
  it('renders correct distinct inner and outer colors for misclassified points', () => {
    const { container } = render(
      <svg>
        <DualRingPoint2D
          x={0}
          y={0}
          actualClass={1}
          predictedClass={0}
          isMispredicted={true}
          isActive={false}
        />
      </svg>
    );

    const circles = container.querySelectorAll('circle');
    expect(circles.length).toBeGreaterThanOrEqual(2);
    // วงใน (Actual) และวงนอก (Pred) ต้องมี fill color ที่ต่างกัน
    expect(circles[0].getAttribute('fill')).not.toBe(circles[1].getAttribute('fill'));
    // ต้องมีกากบาทหรือสัญลักษณ์เตือนความผิดพลาด
    const cross = container.querySelector('[data-error-cross="true"]');
    expect(cross).not.toBeNull();
  });
});
```

---

## 4. Performance & Memory Gate
- **Hexbin Engine Execution**: ข้อมูล $10,000$ แถวต้องทำการ Tessellate และ Aggregate เสร็จภายใน $< 25$ ms
- **Rendering Budget**: การวาด Comet Trail $W=10$ ร่วมกับ Dual-Ring 1,000 จุด ต้องใช้เวลาเรนเดอร์ $< 16.6$ ms (60 FPS) บน Chrome / Firefox / Safari
- **Memory Consumption**: Float32 Array สำหรับ Hexbin vertices ต้องไม่เกิน $2$ MB

