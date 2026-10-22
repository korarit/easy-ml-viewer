# Test Phase 48: Dual-Mode Synchronized Compare (2D/3D), 2D .mlv Persistence, High-Res Vector/Video Export & Hardening

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 48)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-48-dual-mode-compare-mlv-baker-export-and-hardening.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-48-dual-mode-compare-mlv-baker-export-and-hardening.md) เพื่อยืนยันความถูกต้องและเสถียรภาพของการทำงานในส่วนปลายทางของโหมด 2 มิติ ได้แก่:
1. การคำนวณพื้นที่ขัดแย้งของสองโมเดลในโหมด Dual 2D Compare (**2D Disagreement Area & Mask Computation**)
2. ความถูกต้องของการเข้ารหัสและคลายการบีบอัดแคช 2D Contour ในฟอร์แมตไฟล์ `.mlv` (MsgPack + Zstandard)
3. ความสมบูรณ์ของโครงสร้าง XML ในไฟล์ส่งออก **Vector SVG** และความคมชัดของภาพ 4K PNG / คลิปวิดีโอ 2D WebM
4. การทดสอบความทนทานระดับ **100,000 แถว** ในโหมด 2D (Stress Benchmark) และการตรวจสอบ **100% Offline Air-Gapped Operation**

### ขอบเขตโมเดลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Recording & Export Domain**:
  - `app.domains.recording.contour_2d_baker.Contour2DBaker`
  - `app.domains.recording.svg_export_generator.SvgExportGenerator`
  - Integration with `BinaryCodec` (.mlv serialization)
- **Frontend Dual Compare & Export Domain**:
  - `DualCompare2DCanvas.tsx`, `SvgExportModal.tsx`, `useDualCompareStore`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Integration Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_48/
├── conftest.py
├── test_dual_compare_2d_disagreement_math.py
├── test_mlv_2d_contour_serialization.py
└── test_svg_export_generator.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_48/test_mlv_2d_contour_serialization.py
import numpy as np
from app.core.binary_codec import BinaryCodec

def test_mlv_roundtrip_preserves_2d_contour_cache() -> None:
    """ทดสอบว่าข้อมูล 2D Contours และ Heatmap ถูกบีบอัดและคลายออกได้ตรงกัน 100% ไม่มีข้อมูลตกหล่น"""
    original_heatmap = np.random.uniform(0.0, 1.0, size=(120, 120)).astype(np.float32)
    original_segments = [
        [-2.5, 1.0, -1.0, 1.5],
        [-1.0, 1.5, 0.5, 2.0]
    ]

    payload = {
        "version": "2.1.0",
        "contour_2d_cache": {
            "x_feature": "income",
            "y_feature": "credit_score",
            "grid_resolution": 120,
            "heatmap": original_heatmap.tolist(),
            "boundary_polylines": original_segments
        }
    }

    # บีบอัดด้วย MsgPack + Zstd
    encoded_bytes = BinaryCodec.encode(payload)
    assert len(encoded_bytes) < 60_000 # ต้องบีบอัดได้กะทัดรัด < 60 KB

    # คลายบีบอัด
    decoded = BinaryCodec.decode(encoded_bytes)
    restored_cache = decoded["contour_2d_cache"]

    assert restored_cache["grid_resolution"] == 120
    assert len(restored_cache["boundary_polylines"]) == 2
    restored_heatmap = np.array(restored_cache["heatmap"], dtype=np.float32)
    assert np.allclose(original_heatmap, restored_heatmap, atol=1e-5)

def test_svg_generator_produces_well_formed_xml() -> None:
    """ทดสอบว่า SvgExportGenerator สร้าง SVG ที่ผ่านการ Parse แบบ XML และมี Element ครบถ้วน"""
    import xml.etree.ElementTree as ET
    from app.domains.recording.svg_export_generator import SvgExportGenerator

    svg_str = SvgExportGenerator.generate_svg(
        x_label="Feature X",
        y_label="Feature Y",
        boundary_segments=[[-1.0, 0.0, 1.0, 0.0]],
        data_points=[{"x": 0.0, "y": 0.5, "actual": 0}],
        width=800,
        height=600
    )

    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "800"
    assert root.attrib["height"] == "600"
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component & Dual Compare Tests (`vitest`)

```tsx
// frontend/src/domains/recording/__tests__/DualCompare2DCanvas.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DualCompare2DCanvas } from '../components/DualCompare2DCanvas';

describe('DualCompare2DCanvas Split & Superimposed Modes', () => {
  it('renders both model boundary layers and disagreement overlay', () => {
    const { container } = render(
      <DualCompare2DCanvas
        modelAName="RandomForest"
        modelBName="SVM-RBF"
        disagreementAreaPct={14.2}
        showConflictHatch={true}
      />
    );

    expect(screen.getByText(/14.2% Disagreement Area/i)).toBeTruthy();
    expect(container.querySelector('[data-testid="conflict-hatch-pattern"]')).toBeTruthy();
  });
});
```

---

## 4. การทดสอบการจัดการหน่วยความจำและ 100k-Row Stress Verification

| รายการทดสอบ | เงื่อนไขการทดสอบ | เกณฑ์ผ่านที่ยอมรับได้ (Pass SLA) |
| :--- | :--- | :--- |
| **1. 100k-Row 2D Playback Frame Rate** | เล่นวิดีโอ 100,000 แถวในโหมด 2D Planar | อัตราเฟรมคงที่ที่ **60 FPS** (Render Time $< 3.0\text{ ms}$) |
| **2. Dual 2D Compare Peak Memory** | โหลดไฟล์ `.mlv` 2 ไฟล์ขนาดใหญ่พร้อมกัน | Peak RAM Delta ของ Backend ไม่เกิน **$\le 180\text{ MB}$** |
| **3. SVG Large Dataset Export** | ส่งออก SVG ที่มีจุดข้อมูล 20,000 จุด | ใช้เวลาสร้างไฟล์ **$< 350\text{ ms}$** และขนาดไฟล์ SVG $\le 4.5\text{ MB}$ |
| **4. Zero Network Request Guarantee** | ตรวจสอบการส่งออกและเล่นไฟล์ในโหมด Offline | ไม่มีการยิง Network Traffic ออกนอก `127.0.0.1` (**0 External Requests**) |

---

## 5. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E48.1** | โมเดล A และ B ทายตรงกัน 100% ทุกแถวและทุกพิกัด | สัดส่วนความขัดแย้งเป็น 0% ไม่มี Conflict Zone ให้ระบาย | แสดงป้าย *"100% Consensus: Models are completely in agreement"* |
| **E48.2** | โมเดล A และ B ทายขัดแย้งกัน 100% ทุกจุด (Total Disagreement) | ลายเฉียงสีม่วงคลุมเต็มหน้าจอ 2D | แสดงป้ายเตือน *"100% Inverted Predictions across entire feature domain"* |
| **E48.3** | ไฟล์ `.mlv` รุ่นเก่า (v1.x) ที่ไม่มีบล็อก `contour_2d_cache` | การเปิดโหมด 2D ล้มเหลวหรือเกิด Null Reference | Fallback คำนวณ 2D Contour ใหม่แบบ On-the-Fly อย่างนุ่มนวล |
| **E48.4** | การส่งออก SVG ที่ชื่อคอลัมน์มีตัวอักษรพิเศษ XML (`<`, `>`, `&`, `"`) | โครงสร้าง XML ของไฟล์ SVG เสียหาย ไม่สามารถเปิดดูได้ | ทำการ Escape ตัวอักษร (`&amp;`, `&lt;`, `&gt;`) อัตโนมัติในตัวสร้าง SVG |
| **E48.5** | การเปิดไฟล์ `.mlv` 2 ไฟล์ที่แกน $X$ หรือ $Y$ ไม่ตรงกัน | การเปรียบเทียบในระนาบ 2D ผิดสเกล | แจ้งเตือนผู้ใช้ให้เลือกจับคู่แกน $X-Y$ ให้ตรงกันก่อนแสดงผล Superimposed |
| **E48.6** | บราวเซอร์มีหน่วยความจำต่ำมากขณะดาวน์โหลด 4K PNG Snapshot | เกิด Out-of-Memory (Canvas toBlob Crash) | ปรับลดความละเอียดเป็น 1080p อัตโนมัติพร้อมแจ้งเตือนผู้ใช้ |

---

## 6. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Backend ทั้งหมดใน `backend/tests/phase_48/` ผ่าน 100%
- [ ] ชุดทดสอบ Frontend ใน `frontend/src/domains/recording/__tests__/` ผ่าน 100%
- [ ] ผ่านเกณฑ์ 100,000-Row Stress Benchmark ในโหมด 2D ที่ 60 FPS
- [ ] ยืนยันว่าไฟล์ `.mlv` สามารถจัดเก็บและอ่านแคช 2D ได้อย่างสมบูรณ์
- [ ] ยืนยันการทำงานแบบ 100% Offline Air-Gapped บนทุกระบบปฏิบัติการ

