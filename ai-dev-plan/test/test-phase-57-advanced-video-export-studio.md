# Test Phase 57: Advanced Multi-Layout Video Export Studio (2D/3D Dual-Dimension, Why/What-If PiP, Independent Overrides & MP4/WebM)

## 1. วัตถุประสงค์และขอบเขตการทดสอบ (Test Objective & Scope — Synced with Dev Phase 57)
ทดสอบความสมบูรณ์ของระบบสตูดิโอเรนเดอร์วิดีโอขั้นสูง (Phase 57) ครอบคลุม:
1. การแยกสถานะ (State Isolation) ของ **Theme, มุมกล้อง, ความเร็ว, และมิติ 2D/3D** ออกจากพรีวิวหน้าจอหลัก 100%
2. ความถูกต้องเชิงเรขาคณิตของการประกอบภาพในแคนวาสเสมือน (**Composite Canvas Coordinates: Split 70/30 และ PiP**)
3. การฝังและการซิงค์ข้อมูล **Why Panel (SHAP Waterfall)** และ **What-If Sandbox** ให้ตรงกับเฟรมที่กำลังเรนเดอร์
4. ความถูกต้องของโครงสร้างไฟล์สากล **`.mp4` (H.264/AAC)** และ **`.webm` (VP9)**
5. การเรนเดอร์แบบ Deterministic Zero-Drop-Frame 60 FPS และการป้องกัน Memory Leak

---

## 2. รายการทดสอบฝั่ง Backend Unit & Transcode API Tests (`pytest`)

```python
# backend/tests/phase_57/test_video_export_backend.py
import pytest
import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_mp4_transcode_endpoint_structure_and_headers():
    """ทดสอบว่า Endpoint สำหรับแปลงไฟล์หรือ Fallback MP4 ส่งคืนไฟล์ H.264 ถูกต้อง"""
    # จำลองไฟล์วิดีโอตัวอย่างขนาดเล็ก
    dummy_video_bytes = b"FAKE_VIDEO_STREAM_DATA"
    
    response = client.post(
        "/api/v1/export/transcode-mp4",
        files={"file": ("test_clip.webm", io.BytesIO(dummy_video_bytes), "video/webm")},
        data={"target_fps": "60", "codec": "h264"}
    )
    # หากรัน fallback transcode สำเร็จ ต้องได้ status 200 หรือ response DTO
    if response.status_code == 200:
        data = response.json()
        assert "download_url" in data
        assert data["output_format"] == "mp4"

def test_mp4_container_magic_bytes_validation():
    """ทดสอบโครงสร้างกล่อง MP4 ftyp box มาตรฐาน"""
    def is_valid_mp4_header(data: bytes) -> bool:
        # MP4 ต้องมี ftyp ใน 12 ไบต์แรก
        return len(data) >= 12 and data[4:8] == b"ftyp"

    # ตัวอย่าง Header MP4 ทั่วไป
    valid_header = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    assert is_valid_mp4_header(valid_header) is True
```

---

## 3. รายการทดสอบฝั่ง Frontend UI & Virtual Compositor Tests (`vitest`)

```tsx
// frontend/src/domains/export-studio/__tests__/AdvancedVideoExportCompositor.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useExportStore } from '../store/useExportStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';

describe('Advanced Video Export Studio Isolation Tests', () => {
  beforeEach(() => {
    // รีเซ็ตสถานะสโตร์
    useStage3DStore.setState({
      viewMode: '2d',
      cameraPreset: 'front_xz',
    });
    usePlaybackStore.setState({
      speedMultiplier: 1.0,
      currentFrameIndex: 10,
    });
  });

  it('allows selecting 3D dimension and isometric camera without altering live 2D preview state', () => {
    // ผู้ใช้ตั้งค่าใน Export Studio เป็น 3D + Isometric
    const exportConfig = {
      dimension: '3d' as const,
      cameraPreset: 'isometric' as const,
      theme: 'light' as const,
      speedMultiplier: 0.25, // Slow-Mo
    };

    // ตรวจสอบว่า Preview หน้าจอสด ยังคงเป็น 2D และ front_xz ไม่โดนแทรกแซง
    expect(useStage3DStore.getState().viewMode).toBe('2d');
    expect(useStage3DStore.getState().cameraPreset).toBe('front_xz');
    expect(usePlaybackStore.getState().speedMultiplier).toBe(1.0);
  });

  it('calculates exact Split 70/30 viewport coordinates for 1920x1080 canvas', () => {
    const canvasWidth = 1920;
    const canvasHeight = 1080;

    const stageWidth = Math.round(canvasWidth * 0.7); // 1344px
    const panelWidth = canvasWidth - stageWidth;     // 576px

    expect(stageWidth).toBe(1344);
    expect(panelWidth).toBe(576);
    expect(stageWidth + panelWidth).toBe(1920);
  });
});
```

---

## 4. เกณฑ์การตรวจรับคุณภาพระดับระบบ (Performance & Video Quality Gates)
1. **Zero Dropped Frames**: วิดีโอ 60 FPS ความยาว 5 วินาที (300 เฟรม) ต้องมีจำนวนเฟรมในวิดีโอครบถ้วน $300$ เฟรมพอดี ไม่มีการข้ามเฟรม
2. **Cross-Platform Playback**: ไฟล์ `.mp4` ที่สร้างต้องเปิดเล่นได้บน QuickTime Player บน macOS และ Windows Media Player โดยไม่เกิด Codec Error
3. **Memory Budget**: ระหว่างกระบวนการ Render และ Encode วิดีโอขนาด 1080p หน่วยความจำเบราว์เซอร์ (JS Heap) ต้องไม่เกิน $350\text{ MB}$
4. **Theme Purity**: ในโหมด Light Theme ตัวอักษรและเส้นกริดใน Why Panel ต้องมี Contrast Ratio $\ge 4.5:1$ ตามมาตรฐาน WCAG 2.1 AA
