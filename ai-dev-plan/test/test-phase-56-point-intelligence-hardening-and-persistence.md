# Test Phase 56: 2D Point Intelligence Unified Controls, .mlv Multi-Track Recording & Hardening

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 56)
ทดสอบระบบควบคุมแบบรวมศูนย์ (Point Intelligence Toolbar & Shortcuts), ความสมบูรณ์ของการบันทึกและโหลดไฟล์ `.mlv` ที่มี Point Intelligence Tracks, ความถูกต้องของการส่งออกไฟล์เวกเตอร์ SVG, และการทดสอบความทนทาน (Stress & Memory Leak Gate)

---

## 2. รายการทดสอบฝั่ง Backend Unit & Serialization Tests (`pytest`)

```python
# backend/tests/phase_56/test_mlv_point_intelligence_serialization.py
import pytest
from app.core.binary_codec import BinaryCodec

def test_mlv_point_intelligence_track_roundtrip():
    """ทดสอบการ Encode และ Decode แทร็ก Point Intelligence ในไฟล์ .mlv ให้ได้ผลลัพธ์ตรงเป๊ะ"""
    payload = {
        "flags": {"trails": True, "dual_ring": True, "error_lens": False, "hexbin": True},
        "hexbin_cells": [
            {"cx": 0.5, "cy": 1.2, "count": 14, "error_rate": 0.2857, "dom_class": 1}
        ],
        "density_contours": [
            {"threshold": 0.5, "segments": [[0.0, 0.0, 1.0, 1.0]]}
        ]
    }

    # เข้ารหัสด้วย MsgPack + Zstd
    encoded_bytes = BinaryCodec.encode_payload(payload)
    assert len(encoded_bytes) > 0
    # ขนาดต้องกะทัดรัด
    assert len(encoded_bytes) < 1024

    # ถอดรหัสกลับคืน
    decoded = BinaryCodec.decode_payload(encoded_bytes)
    assert decoded["flags"] == payload["flags"]
    assert decoded["hexbin_cells"][0]["count"] == 14
    assert abs(decoded["hexbin_cells"][0]["error_rate"] - 0.2857) < 1e-4
```

---

## 3. รายการทดสอบฝั่ง Frontend Integration & Export Tests (`vitest`)

```tsx
// frontend/src/domains/stage3d/__tests__/PointDisplayToolbar2D.test.tsx
import { render, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { PointDisplayToolbar2D } from '../components/PointDisplayToolbar2D';
import { useStage3DStore } from '../store/useStage3DStore';

describe('PointDisplayToolbar2D & Shortcut Integration', () => {
  it('toggles error lens state when E key is pressed', () => {
    render(<PointDisplayToolbar2D />);
    
    expect(useStage3DStore.getState().showErrorsOnly).toBe(false);

    // จำลองกดปุ่มคีย์บอร์ด E
    fireEvent.keyDown(window, { key: 'e' });
    expect(useStage3DStore.getState().showErrorsOnly).toBe(true);

    // กดซ้ำเพื่อปิด
    fireEvent.keyDown(window, { key: 'e' });
    expect(useStage3DStore.getState().showErrorsOnly).toBe(false);
  });
});
```

---

## 4. Stress Test & Memory Leak Gate
1. **Scrubbing Loop Stress Test**: ทำการ Scrubbing ซ้ำไป-มา $5,000$ ครั้งอย่างต่อเนื่อง หน่วยความจำ JS Heap ต้องไม่เพิ่มขึ้นเกิน $5$ MB (Zero Memory Leak)
2. **Scale Stress Test**: ข้อมูล $10,000$ จุด ร่วมกับกริด Hexbin 300 ช่อง ต้องเรนเดอร์ได้ในเวลา $< 16$ ms ต่อเฟรม
3. **SVG Export Integrity**: ไฟล์ SVG ที่ส่งออกต้องประกอบด้วยแท็ก `<polygon>` ของ Hexbin และ `<circle>` ของ Dual-Ring ครบถ้วน ไม่มีการตัดทอนพิกัด

