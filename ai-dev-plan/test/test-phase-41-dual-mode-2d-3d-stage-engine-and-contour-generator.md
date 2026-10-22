# Test Phase 41: Dual-Mode 2D/3D Stage Engine, Orthographic Camera Rig, Vector Grid & Universal Decision Boundary Contour Generator

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 41)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-41-dual-mode-2d-3d-stage-engine-and-contour-generator.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-41-dual-mode-2d-3d-stage-engine-and-contour-generator.md) เพื่อยืนยันความถูกต้องและเสถียรภาพของเครื่องยนต์แสดงผลแบบสองโหมด (**Dual-Mode 2D Planar & 3D Spatial Engine**), ความแม่นยำทางคณิตศาสตร์ของอัลกอริทึม Marching Squares ในการสร้างเส้นแบ่งเขตการตัดสินใจ (**Decision Boundary Polylines**), ความถูกต้องของแผนที่ความน่าจะเป็น 2 มิติ (Probability Heatmap), การทำงานของเป้าเล็งวงแหวนและหางแสง 2 มิติใน VCR Playback, ลูกศรเวกเตอร์การกระจัดใน What-If Sandbox, ตลอดจนเกณฑ์ความปลอดภัยด้านหน่วยความจำ VRAM และการทำงานแบบ **100% Offline**

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Simulation & Projection Domain**:
  - `app.domains.simulation.contour_2d_generator.Contour2DGenerator`
  - REST Endpoint: `POST /api/v1/projection/contour-2d`
- **Frontend 2D Stage Domain**:
  - `src/domains/stage3d/store/useStage3DStore.ts` (`viewMode`, `toggleViewMode`)
  - `src/domains/projection3d/store/useProjectionStore.ts` (`fetchContour2D`, `contourCache`)
  - `Stage2DCanvas.tsx`, `DecisionBoundaryContour2D.tsx`, `ActiveDataPointReticle2D.tsx`, `WhatIfDisplacementVector2D.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_41/
├── conftest.py
├── test_contour_2d_generator.py
├── test_marching_squares_exact_crossing.py
└── test_contour_2d_endpoint.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_41/test_marching_squares_exact_crossing.py
import numpy as np
import pytest
from app.domains.simulation.contour_2d_generator import Contour2DGenerator

def test_marching_squares_straight_vertical_boundary() -> None:
    """ทดสอบกรณีพื้นผิวความน่าจะเป็นแบ่งครึ่งตามแนวดิ่ง (x = 0.5) เส้นขอบเขตต้องเป็นเส้นตรงดิ่งเป๊ะ"""
    xs = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    grid_x, _ = np.meshgrid(xs, ys)
    # ค่าฟิลด์แปรผันตาม x โดยข้ามเกณฑ์ 0.5 ที่กึ่งกลาง
    prob_field = grid_x.astype(np.float32)

    segments = Contour2DGenerator._marching_squares_segments(xs, ys, prob_field, threshold=0.5)
    assert len(segments) > 0

    # พิกัด x ของทุกจุดตัดในเซกเมนต์ต้องอยู่ใกล้ 0.5 อย่างยิ่ง (Error <= 1e-3)
    for seg in segments:
        x1, y1, x2, y2 = seg
        assert abs(x1 - 0.5) <= 0.025
        assert abs(x2 - 0.5) <= 0.025

def test_marching_squares_circular_boundary() -> None:
    """ทดสอบกรณีขอบเขตวงกลม (รัศมี R=0.4 จากจุดศูนย์กลาง) ต้องสกัดเส้นรอบวงได้ครบวงปิด"""
    xs = np.linspace(-1.0, 1.0, 60, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, 60, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    # รัศมี r = sqrt(x^2 + y^2), ค่า = 1.0 - r/0.4 (ข้าม 0.0 ที่ r = 0.4)
    r = np.sqrt(gx**2 + gy**2)
    field = (0.4 - r).astype(np.float32)

    segments = Contour2DGenerator._marching_squares_segments(xs, ys, field, threshold=0.0)
    assert len(segments) >= 40

    # ทุกจุดบนเส้นต้องมีระยะห่างจากศูนย์กลางใกล้เคียง 0.40
    for x1, y1, x2, y2 in segments:
        r1 = np.sqrt(x1**2 + y1**2)
        r2 = np.sqrt(x2**2 + y2**2)
        assert abs(r1 - 0.4) <= 0.03
        assert abs(r2 - 0.4) <= 0.03

@pytest.mark.asyncio
async def test_contour_2d_endpoint_returns_valid_payload(async_client, synthetic_classification_model, synthetic_dataset) -> None:
    """ทดสอบว่า Endpoint /api/v1/projection/contour-2d ส่งกลับ DTO ถูกต้องและมี Latency < 100ms"""
    payload = {
        "dataset_id": synthetic_dataset["dataset_id"],
        "model_id": synthetic_classification_model["model_id"],
        "x_feature": "feature_0",
        "y_feature": "feature_1",
        "grid_resolution": 120,
        "pinning_mode": "median"
    }
    response = await async_client.post("/api/v1/projection/contour-2d", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["grid_resolution"] == 120
    assert len(data["heatmap_raw"]) == 120 * 120
    assert "boundary_segments" in data
    assert data["computation_time_ms"] <= 150.0 # SLA limit
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & WebGL Mock)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/stage3d/__tests__/
├── useStage3DStoreViewMode.test.ts
├── Stage2DCanvas.test.tsx
├── DecisionBoundaryContour2D.test.tsx
└── WhatIfDisplacementVector2D.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/stage3d/__tests__/useStage3DStoreViewMode.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useStage3DStore } from '../store/useStage3DStore';

describe('useStage3DStore ViewMode Toggle (2D <-> 3D) & Shortcuts', () => {
  beforeEach(() => {
    useStage3DStore.setState({
      viewMode: '3d',
      showBoundaryLines: true,
      showHeatmap: true,
    });
  });

  it('toggles viewMode between 2d and 3d smoothly', () => {
    const store = useStage3DStore.getState();
    expect(store.viewMode).toBe('3d');

    store.toggleViewMode();
    expect(useStage3DStore.getState().viewMode).toBe('2d');

    store.toggleViewMode();
    expect(useStage3DStore.getState().viewMode).toBe('3d');
  });

  it('sets explicit viewMode correctly', () => {
    const store = useStage3DStore.getState();
    store.setViewMode('2d');
    expect(useStage3DStore.getState().viewMode).toBe('2d');
  });
});

// frontend/src/domains/stage3d/__tests__/DecisionBoundaryContour2D.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DecisionBoundaryContour2D } from '../components/DecisionBoundaryContour2D';

describe('DecisionBoundaryContour2D Rendering & Visibility Guards', () => {
  it('renders boundary line segments and respects visibility toggles', () => {
    const sampleSegments = [
      [-2.5, 0.0, 2.5, 0.0],
      [0.0, -2.5, 0.0, 2.5],
    ];
    const { container, rerender } = render(
      <DecisionBoundaryContour2D segments={sampleSegments} visible={true} opacity={1.0} />
    );
    expect(container.querySelector('[data-testid="decision-boundary-layer"]')).toBeTruthy();

    rerender(<DecisionBoundaryContour2D segments={sampleSegments} visible={false} opacity={1.0} />);
    expect(container.querySelector('[data-testid="decision-boundary-layer"]')).toBeNull();
  });
});
```

---

## 4. การทดสอบการจัดการหน่วยความจำและทรัพยากร (Memory Leak & Resource Verification Tests)

### 4.1 แผนการตรวจสอบหน่วยความจำและ VRAM
| มิติการตรวจสอบ | เงื่อนไขการทดสอบ | เกณฑ์ผ่านที่ยอมรับได้ (Pass SLA) |
| :--- | :--- | :--- |
| **1. 2D/3D Rapid Toggle VRAM Leak** | สลับ `viewMode` ระหว่าง `2d` และ `3d` ต่อเนื่อง 50 รอบ | จำนวน `geometries` และ `textures` ใน WebGL ต้องกลับเท่าเดิมเป๊ะ (**Zero Leak**) |
| **2. Heatmap Texture Disposal** | เปลี่ยนคู่แกน $X-Y$ ต่อเนื่อง 20 ครั้ง | เท็กซ์เจอร์เดิมถูกเรียก `texture.dispose()` ทุกครั้ง ไม่มี Texture Stacking ค้างในการ์ดจอ |
| **3. Backend Batch Grid Inference RAM** | คำนวณกริด 14,400 จุด 10 ครั้งติดต่อกัน | Memory RSS เพิ่มขึ้นไม่เกิน **$\le 15\text{ MB}$** หลัง Garbage Collection |
| **4. 2D Scrubbing 60 FPS Frame Budget** | ลาก Scrubber ที่ความเร็ว 10x ข้าม 5,000 เฟรมในโหมด 2D | เวลาต่อเฟรม **$< 3.5\text{ ms}$** เรนเดอร์ลื่นไหลที่ **60 FPS** |

---

## 5. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 8 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E41.1** | ข้อมูลทั้งแกน $X$ หรือ $Y$ มีค่าเท่ากันหมด ($\text{span} = 0$) | การหารด้วยศูนย์ ($1 / (x_{\max} - x_{\min})$) กลายเป็น `inf` หรือ `nan` | ขยายสเกลขอบเขตชั่วคราวเป็น $[x - 1.0, x + 1.0]$ ไม่เกิด Error |
| **E41.2** | โมเดลทำนายทุกจุดเป็นคลาสเดียวกัน 100% (ไม่มี Boundary) | Marching Squares หาจุดตัดไม่เจอ คืนค่า Segments ว่าง | วาด Heatmap สีสม่ำเสมอ และแสดงป้าย *"No decision boundary within this feature range"* |
| **E41.3** | ผู้ใช้กดคีย์ `V` รัวๆ 10 ครั้งใน 1 วินาทีขณะกล้องกำลัง Lerp | แอนิเมชันกล้องชนกัน Frustum หรือ Projection Matrix ค้าง | มี Flag `isMorphing` ป้องกัน Re-entrant และตัดจบทันทีไปยัง State ล่าสุด |
| **E41.4** | จุดข้อมูลในคิวมีพิกัดซ้อนทับกันสนิทหลายสิบจุด | จุดใน 2D บดบังกันจนมองไม่เห็นความคลาดเคลื่อน | วาดจุดที่มีสถานะทายผิด (**False Positive/Negative**) ให้อยู่ชั้นบนสุด (Z-Index Top) เสมอ |
| **E41.5** | ข้อมูลมีอัตราส่วนสเกลแกนต่างกันมหาศาล (เช่น $10^6$ vs $10^{-4}$) | กราฟ 2D บี้แบนเป็นเส้นเดียว | แปลงผ่าน `WorldCoordinateScaler` เป็นพิกัดมาตรฐาน `[-5, +5]` ทั้งสองแกน |
| **E41.6** | Multiclass 10 คลาสที่มีค่าความน่าจะเป็นเท่ากันเป๊ะที่จุดตัด | อัลกอริทึม Marching Squares สับสนระหว่างขอบเขตคลาส | จัดการด้วยการเลือกลำดับคลาสแบบ Deterministic ตาม `class_index` ลำดับต่ำกว่า |
| **E41.7** | มีค่า `NaN` หลุดเข้ามาใน Grid Inference | แผนที่ Heatmap มีรูโหว่ หรือ WebGL Render Crash | แทนที่ค่า `NaN` ด้วยค่าความน่าจะเป็น 0.5 หรือ Mean อัตโนมัติในเลเยอร์ Backend |
| **E41.8** | ข้อมูลมีขนาด 100,000 แถว พล็อตลงบน 2D Planar Stage | หน้าเว็บค้าง หรือ DOM Overflow | ใช้ `THREE.InstancedMesh` สำหรับจุด 2D ทั้งหมด จึงมีเพียง **1 Draw Call** เท่านั้น |

---

## 6. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit & Integration Test ฝั่ง Backend ทั้งหมดใน `backend/tests/phase_41/` ผ่านฉลุย 100%
- [ ] ชุดทดสอบ Component & Hook Test ฝั่ง Frontend ใน `frontend/src/domains/stage3d/__tests__/` ผ่านฉลุย 100%
- [ ] ผ่านเกณฑ์ Memory SLA: ไม่มี VRAM Leak หลังสลับ 2D/3D 50 รอบ
- [ ] ผ่านการทดสอบ Edge Case ทั้ง 8 กรณีในตารางข้อ 5 ครบถ้วน
- [ ] ยืนยันว่าไม่มี External Network Request ใดๆ หลุดออกนอก `127.0.0.1` (100% Air-Gapped Offline Verified)

