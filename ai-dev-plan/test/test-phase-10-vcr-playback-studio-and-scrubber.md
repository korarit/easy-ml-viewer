# Test Phase 10: VCR Playback Studio — 60FPS Scrubber, Play/Pause/Step/Speed/Loop & Motion Trail

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 10)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-10-vcr-playback-studio-and-scrubber.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-10-vcr-playback-studio-and-scrubber.md) เพื่อตรวจสอบความแม่นยำของระบบควบคุมการเล่นวิดีโอ (**VCR Playback Studio**), การคำนวณ `exactFrameFloat` ด้วย `requestAnimationFrame` Delta Time, โหมดการเล่นวนซ้ำ (`Loop All` และ `A-B Sub-range Loop`), การรีเซ็ตหางแสง (`MotionTrail3D`) เมื่อกระโดดข้ามเฟรม, และการรับประกันว่าคอมโพเนนต์ React จะไม่เกิด Re-render ซ้ำซ้อนทุกๆ 16.67ms ขณะเล่นที่ 60 FPS

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Domain & API**:
  - `backend/app/domains/playback/value_objects.py`: `FrameTrajectoryPointVO`
  - `backend/app/domains/playback/entities.py`: `PlaybackSequenceManifestEntity`
  - `backend/app/domains/playback/services.py`: `TrajectoryInterpolationService.detect_discontinuity_jumps()`
  - `backend/app/presentation/api/v1/playback_router.py`: `GET /api/v1/playback/{session_id}/manifest`
- **Frontend Store, Hooks & 3D Components**:
  - `frontend/src/domains/playback/store/usePlaybackStore.ts`
  - `frontend/src/domains/playback/hooks/usePlaybackRafLoop.ts`, `usePlaybackKeyboardShortcuts.ts`
  - `frontend/src/domains/playback/components/VcrTransportDeck.tsx`, `TimelineScrubberBar.tsx`, `ActiveSampleSphere3D.tsx`, `MotionTrail3D.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `TrajectoryInterpolationService`
ไฟล์ทดสอบ: `backend/tests/unit/domains/test_playback_service.py`

```python
import numpy as np
import pytest
from app.domains.playback.services import TrajectoryInterpolationService


def test_detect_discontinuity_jumps_identifies_large_spatial_gaps():
    """ตรวจสอบการตรวจจับเฟรมที่มีการกระโดดข้ามพื้นที่ฉาก 3D ไกลเกินเกณฑ์ threshold_world_dist"""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],  # เฟรม 1: ขยับนิดเดียว
            [4.8, -3.5, 2.0], # เฟรม 2: กระโดดไกลเกิน 4.5 หน่วย!
            [4.9, -3.4, 2.1], # เฟรม 3: ขยับนิดเดียว
        ],
        dtype=np.float32,
    )
    jump_frames = TrajectoryInterpolationService.detect_discontinuity_jumps(
        positions, threshold_world_dist=4.5
    )
    assert jump_frames == [2]


def test_detect_discontinuity_jumps_single_or_empty_frame():
    """ตรวจสอบว่าเมื่อมีเพียง 0 หรือ 1 เฟรม จะคืนค่ารายการว่างโดยไม่โยน IndexError"""
    single_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert TrajectoryInterpolationService.detect_discontinuity_jumps(single_pos) == []
```

### 2.2 API Integration Tests สำหรับ `GET /api/v1/playback/{session_id}/manifest`
ไฟล์ทดสอบ: `backend/tests/integration/api/test_playback_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_playback_manifest_returns_valid_sequence(async_client: AsyncClient, seeded_session_id: str):
    """ตรวจสอบว่า Manifest คืนจำนวนเฟรม พิกัด 3D และค่าทำนายครบทุกเฟรม"""
    response = await async_client.get(f"/api/v1/playback/{seeded_session_id}/manifest")
    assert response.status_code == 200

    body = response.json()
    assert body["session_id"] == seeded_session_id
    assert body["total_frames"] == len(body["frames_summary"])
    first_frame = body["frames_summary"][0]
    assert len(first_frame["position_world"]) == 3
    assert 0.0 <= first_frame["confidence"] <= 1.0
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Unit Tests สำหรับ `usePlaybackStore` (State Machine, Delta Time, A-B Loop & Trail Reset)
ไฟล์ทดสอบ: `frontend/src/domains/playback/__tests__/usePlaybackStore.test.ts`

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { usePlaybackStore } from '../store/usePlaybackStore';

describe('Phase 10: usePlaybackStore State Machine & 60FPS Timing', () => {
  beforeEach(() => {
    usePlaybackStore.getState().initializeQueue(500);
  });

  it('คำนวณการเดินเฟรมด้วย advanceByDeltaMs ตาม baseFps และ speedMultiplier อย่างแม่นยำ', () => {
    const store = usePlaybackStore.getState();
    store.play();
    expect(usePlaybackStore.getState().status).toBe('playing');

    // baseFps = 5, speedMultiplier = 1.0 -> ในเวลา 100ms (0.1 วินาที) ต้องเลื่อนไป 0.5 เฟรม
    usePlaybackStore.getState().advanceByDeltaMs(100);
    expect(usePlaybackStore.getState().exactFrameFloat).toBeCloseTo(0.5, 4);
    expect(usePlaybackStore.getState().currentFrameIndex).toBe(0);

    // เดินต่ออีก 100ms -> รวมเป็น 1.0 เฟรม
    usePlaybackStore.getState().advanceByDeltaMs(100);
    expect(usePlaybackStore.getState().exactFrameFloat).toBeCloseTo(1.0, 4);
    expect(usePlaybackStore.getState().currentFrameIndex).toBe(1);
  });

  it('ป้องกันการกระโดดข้ามเฟรมเมื่อสลับแท็บเบราว์เซอร์นานเกิน 100ms (Delta Time Clamping)', () => {
    usePlaybackStore.getState().play();
    // จำลองว่าผู้ใช้สลับแท็บไป 5 วินาที (5000ms) ระบบต้อง Clamp เหลือสูงสุด 100ms เท่านั้น
    usePlaybackStore.getState().advanceByDeltaMs(5000);
    expect(usePlaybackStore.getState().exactFrameFloat).toBeCloseTo(0.5, 4);
  });

  it('รองรับโหมด A-B Sub-range Loop และวนกลับมายังจุด A พร้อมเพิ่ม trailResetCounter เมื่อถึงจุด B', () => {
    const store = usePlaybackStore.getState();
    store.setLoopRangePoint('A', 10);
    store.setLoopRangePoint('B', 20);
    store.seekToFrame(19, true);
    store.play();

    const initialResetCount = usePlaybackStore.getState().trailResetCounter;
    // เดินหน้าเกินเฟรมที่ 20 (เช่น +300ms = +1.5 เฟรม -> 20.5 >= 20)
    usePlaybackStore.getState().advanceByDeltaMs(300);

    const afterState = usePlaybackStore.getState();
    expect(afterState.currentFrameIndex).toBe(10);
    expect(afterState.exactFrameFloat).toBe(10);
    expect(afterState.trailResetCounter).toBe(initialResetCount + 1);
  });

  it('จดจำสถานะ wasPlayingBeforeScrub เมื่อผู้ใช้ลาก Scrubber และคืนสถานะเดิมเมื่อปล่อยเมาส์', () => {
    usePlaybackStore.getState().play();
    expect(usePlaybackStore.getState().status).toBe('playing');

    usePlaybackStore.getState().startScrubbing();
    expect(usePlaybackStore.getState().status).toBe('scrubbing');
    expect(usePlaybackStore.getState().wasPlayingBeforeScrub).toBe(true);

    usePlaybackStore.getState().seekToFrame(250, true);
    usePlaybackStore.getState().endScrubbing();
    expect(usePlaybackStore.getState().status).toBe('playing');
  });

  it('Edge Case #1: เพิ่มค่า trailResetCounter ทันทีเมื่อกระโดดข้ามเฟรมไกลกว่า 5 เฟรม', () => {
    const beforeCounter = usePlaybackStore.getState().trailResetCounter;
    usePlaybackStore.getState().seekToFrame(300, true);
    expect(usePlaybackStore.getState().trailResetCounter).toBe(beforeCounter + 1);
  });

  it('Edge Case #2: เมื่อคิวมีเพียง 1 แถว (totalFrames === 1) ต้องไม่ยอมให้เปลี่ยนสถานะเป็น playing', () => {
    usePlaybackStore.getState().initializeQueue(1);
    usePlaybackStore.getState().play();
    expect(usePlaybackStore.getState().status).toBe('paused');
  });
});
```

### 3.2 Keyboard Shortcuts & Input Focus Guard Test
```typescript
import { describe, it, expect } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { VcrTransportDeck } from '../components/VcrTransportDeck';

describe('Phase 10: Keyboard Shortcuts & Input Guard', () => {
  it('Edge Case #3: ไม่สั่ง Play/Pause เมื่อผู้ใช้กด Spacebar ขณะโฟกัสอยู่ในช่อง <input>', () => {
    usePlaybackStore.getState().initializeQueue(100);
    const { container } = render(
      <div>
        <input data-testid="search-input" />
        <VcrTransportDeck />
      </div>
    );

    const inputEl = container.querySelector('[data-testid="search-input"]')!;
    fireEvent.keyDown(inputEl, { key: ' ', code: 'Space' });
    expect(usePlaybackStore.getState().status).toBe('paused');
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การทดสอบ Zero-React-Re-render ระหว่าง Sub-frame Interpolation ที่ 60 FPS
ไฟล์ทดสอบ: `frontend/src/domains/playback/__tests__/zeroReRender60Fps.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import { usePlaybackStore } from '../store/usePlaybackStore';

describe('60FPS Zero-React-Re-render Verification', () => {
  it('คอมโพเนนต์แผงควบคุมไม่ต้อง Re-render เมื่อ exactFrameFloat เปลี่ยนค่าในระดับทศนิยมระหว่างเฟรม (เช่น 5.1 -> 5.9)', () => {
    usePlaybackStore.getState().initializeQueue(100);
    usePlaybackStore.getState().seekToFrame(5);
    usePlaybackStore.getState().play();

    const renderSpy = vi.fn();
    const InspectorConsumer: React.FC = () => {
      // Subscribe เฉพาะ currentFrameIndex (Integer) ตามกฎสถาปัตยกรรม Phase 10
      const currentFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
      renderSpy(currentFrameIndex);
      return <div>Frame {currentFrameIndex}</div>;
    };

    render(<InspectorConsumer />);
    expect(renderSpy).toHaveBeenCalledTimes(1);

    // จำลองการเดินเวลา 4 ครั้งในเฟรมที่ 5 (5.2, 5.4, 5.6, 5.8)
    act(() => {
      usePlaybackStore.getState().advanceByDeltaMs(40); // 5.2
      usePlaybackStore.getState().advanceByDeltaMs(40); // 5.4
      usePlaybackStore.getState().advanceByDeltaMs(40); // 5.6
      usePlaybackStore.getState().advanceByDeltaMs(40); // 5.8
    });

    // ต้องไม่เกิด React Re-render เลยแม้แต่ครั้งเดียวระหว่างที่ยังอยู่ในเฟรมที่ 5!
    expect(renderSpy).toHaveBeenCalledTimes(1);

    // เมื่อก้าวข้ามเข้าสู่เฟรมที่ 6.0 จึง Re-render เพียง 1 ครั้ง
    act(() => {
      usePlaybackStore.getState().advanceByDeltaMs(40); // 6.0
    });
    expect(renderSpy).toHaveBeenCalledTimes(2);
  });
});
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **การกระโดดข้ามเฟรมไกลๆ ตอนลาก Scrubber** | เรียก `seekToFrame(400, true)` จากเฟรม `5` | `trailResetCounter` เพิ่มขึ้น 1 ค่า และหางแสง `MotionTrail3D` ล้างจุดเก่ารีเซ็ตใหม่ที่เฟรม 400 |
| 2 | **คิวข้อมูลมีเพียง 1 แถว (`totalFrames === 1`)** | เรียก `initializeQueue(1)` แล้วตรวจตำแหน่งหัวอ่าน Scrubber | ตำแหน่งหัวอ่านไม่เป็น `NaN` (`0 / 0` ถูกป้องกัน) และปุ่ม Play/Loop อยู่ในสถานะ `disabled` |
| 3 | **กด `Spacebar` ขณะพิมพ์ใน `<input>` หรือ `<textarea>`** | ยิง `keydown` event ปุ่ม Spacebar โดยมี `target` เป็น `<input>` | ตัวจัดการคีย์ลัดข้ามการทำงาน (`return` ทันที) วิดีโอไม่เล่นหรือหยุดเอง |
| 4 | **ตั้งจุด `Loop A` มากกว่า `Loop B`** | ตั้ง `Loop B = 50` แล้วสั่ง `setLoopRangePoint('A', 120)` | ระบบ Clamp ให้ `Loop A = 49` (`Math.min(frameIndex, currentB - 1)`) เสมอ |
| 5 | **สลับแท็บเบราว์เซอร์ไปนาน 10 วินาทีขณะเปิด Play** | เรียก `advanceByDeltaMs(10000)` | `safeDeltaSec` ถูกจำกัดสูงสุดไว้ที่ `0.1s` (100ms) เฟรมไม่กระโดดข้ามไปท้ายคลิป |
| 6 | **Index Out-of-Bounds ที่เฟรมสุดท้าย (`499.8`)** | ตั้ง `exactFrameFloat = 499.8` ในคิว 500 เฟรม แล้วคำนวณ `nextFrameIdx` สำหรับ Lerp | `Math.min(totalFrames - 1, Math.floor(499.8) + 1)` คืนค่า `499` ไม่ทะลุไป `500` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Frame Rate SLA**: การลากแถบ Scrubber ไปมาอย่างต่อเนื่องบนคิว 10,000 เฟรม ต้องรักษาอัตราเฟรมเรตเฉลี่ย $\ge 58\text{ FPS}$
- [ ] **Zero Sub-Frame React Re-render**: ระหว่างการเคลื่อนที่จากเฟรม $t$ ไปยัง $t+1$ (เช่น $5.01 \dots 5.99$) คอมโพเนนต์ตารางและแผงด้านข้างต้องมีจำนวนครั้งการ Re-render เท่ากับ `0 ครั้ง`
- [ ] **Pre-allocated Trail Buffer**: เส้นหางแสง `MotionTrail3D` ต้องใช้ `Float32Array` ที่จองไว้ล่วงหน้า ไม่มีการสร้าง `new THREE.Vector3()` ใน `useFrame`
- [ ] **Test Coverage**: Unit Test Coverage ของ `usePlaybackStore.ts` และ `TrajectoryInterpolationService` ต้อง बराबर $100\%$
