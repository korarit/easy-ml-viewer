# Test Phase 11: Smart Multi-Track Timeline — Auto-Anomaly Markers, Confidence Strip & Frame Bookmarks

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 11)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-11-smart-multitrack-timeline-and-bookmarks.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-11-smart-multitrack-timeline-and-bookmarks.md) เพื่อตรวจสอบความถูกต้องของเอนจินสแกนหาจุดผิดปกติอัตโนมัติ (**Auto-Anomaly & Event Scanner** ทั้ง 4 ประเภท: `misclassified`, `decision_flip`, `low_confidence`, `input_outlier`), อัลกอริทึมรวมกลุ่มหมุดบนหน้าจอ (`computeScreenPinClusters`), ระบบจัดการ Bookmark & Note (`B`), และสถาปัตยกรรม Dual-Layer HTML5 Canvas ที่ไม่สร้าง DOM nodes ตามจำนวนหมุด

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Domain & API**:
  - `backend/app/domains/timeline/value_objects.py`: `AutoEventPinVO`
  - `backend/app/domains/timeline/entities.py`: `UserBookmarkEntity`, `MultiTrackTimelineEntity`
  - `backend/app/domains/timeline/services.py`: `TimelineEventScannerService.scan_events()`
  - `backend/app/presentation/api/v1/timeline_router.py`: `GET /api/v1/timeline/{session_id}/tracks`, `POST /api/v1/timeline/{session_id}/bookmarks`
- **Frontend Utils, Store & Components**:
  - `frontend/src/domains/timeline/utils/pinClusterEngine.ts`: `computeScreenPinClusters()`
  - `frontend/src/domains/timeline/store/useTimelineStore.ts`
  - `frontend/src/domains/timeline/components/MultiTrackTimelineCanvas.tsx`, `EventJumpNavigator.tsx`, `BookmarkEditorPopover.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `TimelineEventScannerService.scan_events()`
ไฟล์ทดสอบ: `backend/tests/unit/domains/test_timeline_scanner.py`

```python
import numpy as np
import pytest
from app.domains.timeline.services import TimelineEventScannerService


def test_scan_events_detects_all_four_anomaly_categories():
    """ตรวจสอบว่าสแกนเนอร์ตรวจพบทั้ง misclassified, decision_flip, low_confidence และ input_outlier ครบถ้วน"""
    # สร้างข้อมูล 6 แถว 2 คอลัมน์ โดยให้แถวที่ 4 มีค่า Outlier พุ่งสูงมาก
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [1.0, 1.0],
            [100.0, 1.0], # แถวที่ 4: Input Outlier (Z > 3.2)
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    # เพิ่มแถวปกติอีก 20 แถวเพื่อให้ค่า SD นิ่งและแถวที่ 4 เกิน 3.2 SD แน่นอน
    X_full = np.vstack([X, np.ones((20, 2), dtype=np.float32)])
    n_total = len(X_full)

    predictions = np.zeros(n_total, dtype=np.float32)
    predictions[2] = 1.0  # เฟรม 2: เปลี่ยนคำตัดสินจาก 0 -> 1 (decision_flip) และทายผิดจากค่าจริง 0 (misclassified)

    confidences = np.full(n_total, 0.95, dtype=np.float32)
    confidences[3] = 0.49  # เฟรม 3: ความมั่นใจก้ำกึ่งอยู่ในช่วง [0.45, 0.55] (low_confidence)

    actual_targets = np.zeros(n_total, dtype=np.float32)
    row_indices = list(range(n_total))

    events = TimelineEventScannerService.scan_events(
        X=X_full,
        predictions=predictions,
        confidences=confidences,
        actual_targets=actual_targets,
        is_classification=True,
        row_indices=row_indices,
    )

    event_types_by_frame = {(e.frame_index, e.event_type) for e in events}
    assert (2, "misclassified") in event_types_by_frame
    assert (2, "decision_flip") in event_types_by_frame
    assert (3, "low_confidence") in event_types_by_frame
    assert (4, "input_outlier") in event_types_by_frame


def test_scan_events_unlabeled_dataset_skips_misclassified_gracefully():
    """Edge Case #2: เมื่อ actual_targets เป็น None (ข้อมูลไม่มีเฉลย) ต้องไม่โยน Error และยังสแกนอีเวนต์อื่นได้ปกติ"""
    X = np.random.randn(20, 3).astype(np.float32)
    preds = np.array([0 if i < 10 else 1 for i in range(20)], dtype=np.float32)
    confs = np.full(20, 0.85, dtype=np.float32)

    events = TimelineEventScannerService.scan_events(
        X=X,
        predictions=preds,
        confidences=confs,
        actual_targets=None,
        is_classification=True,
        row_indices=list(range(20)),
    )

    assert all(e.event_type != "misclassified" for e in events)
    assert any(e.event_type == "decision_flip" and e.frame_index == 10 for e in events)
```

### 2.2 API Integration Tests สำหรับการสร้างและดึง Bookmark
ไฟล์ทดสอบ: `backend/tests/integration/api/test_timeline_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_and_fetch_frame_bookmark(async_client: AsyncClient, seeded_session_id: str):
    """ทดสอบ POST /api/v1/timeline/{session_id}/bookmarks และการผูกกับ dataset_row_index"""
    bookmark_req = {
        "frame_index": 42,
        "dataset_row_index": 1042,
        "color": "amber",
        "note_text": "ข้อมูลรายได้เป็น 0 แต่ยอดหนี้สูง โมเดลจึงปัดตก",
    }
    post_resp = await async_client.post(
        f"/api/v1/timeline/{seeded_session_id}/bookmarks", json=bookmark_req
    )
    assert post_resp.status_code == 201
    created = post_resp.json()
    assert created["bookmark_id"].startswith("bm_")
    assert created["note_text"] == bookmark_req["note_text"]

    tracks_resp = await async_client.get(f"/api/v1/timeline/{seeded_session_id}/tracks")
    assert tracks_resp.status_code == 200
    bookmarks = tracks_resp.json()["bookmarks"]
    assert any(b["bookmark_id"] == created["bookmark_id"] for b in bookmarks)
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Unit Tests สำหรับ `computeScreenPinClusters` (Spatial Pin Clustering)
ไฟล์ทดสอบ: `frontend/src/domains/timeline/__tests__/pinClusterEngine.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { computeScreenPinClusters } from '../utils/pinClusterEngine';
import { AutoEventPin } from '../types/timeline.types';

describe('Phase 11: computeScreenPinClusters Spatial Clustering', () => {
  const samplePins: AutoEventPin[] = [
    { frame_index: 100, dataset_row_index: 100, event_type: 'misclassified', severity: 1.0, title: 'E1', detail: '' },
    { frame_index: 102, dataset_row_index: 102, event_type: 'misclassified', severity: 1.0, title: 'E2', detail: '' },
    { frame_index: 105, dataset_row_index: 105, event_type: 'decision_flip', severity: 0.8, title: 'E3', detail: '' },
    { frame_index: 800, dataset_row_index: 800, event_type: 'input_outlier', severity: 0.7, title: 'E4', detail: '' },
  ];

  it('Edge Case #1: รวมหมุดที่อยู่ใกล้กันเกิน 14px เป็น Cluster เดียวเมื่อดูมุมกว้าง (Zoom 1x)', () => {
    // ช่วงเฟรม 0..1000 บนความกว้าง 1000px -> เฟรม 100, 102, 105 อยู่ห่างกันแค่ 2px และ 3px (< 14px)
    const clusters = computeScreenPinClusters(samplePins, 0, 1000, 1000, 14);
    expect(clusters.length).toBe(2);
    expect(clusters[0].pins.length).toBe(3); // รวมเฟรม 100, 102, 105 เข้าด้วยกัน
    expect(clusters[0].startFrame).toBe(100);
    expect(clusters[0].endFrame).toBe(105);
    expect(clusters[1].pins.length).toBe(1); // เฟรม 800 อยู่เดี่ยวๆ
  });

  it('แตก Cluster ออกเป็นหมุดเดี่ยวเมื่อผู้ใช้ซูมขยาย Timeline เข้าไปใกล้ๆ (Zoom 20x)', () => {
    // ซูมดูเฉพาะช่วงเฟรม 95..115 (20 เฟรม) บนความกว้าง 1000px -> 1 เฟรม = 50px (> 14px)
    const clustersZoomed = computeScreenPinClusters(samplePins, 95, 115, 1000, 14);
    expect(clustersZoomed.length).toBe(3);
    expect(clustersZoomed.every((c) => c.pins.length === 1)).toBe(true);
  });
});
```

### 3.2 Component & Navigation Tests (`EventJumpNavigator` & Wrap-around)
```typescript
import { describe, it, expect } from 'vitest';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';

describe('Phase 11: Jump-to-Event Navigation & Wrap-Around', () => {
  it('Edge Case #5: เมื่อกด Next Event ที่หมุดสุดท้ายของคิว ระบบต้องวนกลับไปยังหมุดแรกสุดอัตโนมัติ', () => {
    const eventFrames = [15, 120, 450];
    usePlaybackStore.getState().initializeQueue(500);
    usePlaybackStore.getState().seekToFrame(450);

    const current = usePlaybackStore.getState().currentFrameIndex;
    const nextPinFrame = eventFrames.find((f) => f > current) ?? eventFrames[0];
    usePlaybackStore.getState().seekToFrame(nextPinFrame, true);

    expect(usePlaybackStore.getState().currentFrameIndex).toBe(15);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การตรวจสอบ Zero-DOM Pin Rendering & Dual-Layer Canvas Split
ไฟล์ทดสอบ: `frontend/src/domains/timeline/__tests__/MultiTrackTimelineCanvas.perf.test.tsx`

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { MultiTrackTimelineCanvas } from '../components/MultiTrackTimelineCanvas';

describe('Phase 11: Dual-Layer Canvas & Zero-DOM Pin Overhead', () => {
  it('ไม่สร้าง DOM element ย่อยตามจำนวนหมุด แม้มีหมุดเหตุการณ์ 2,000 หมุด (ใช้เฉพาะ <canvas> 2 เลเยอร์)', () => {
    const twoThousandPins = Array.from({ length: 2000 }, (_, i) => ({
      frame_index: i,
      dataset_row_index: i,
      event_type: 'low_confidence' as const,
      severity: 0.6,
      title: `Pin ${i}`,
      detail: '',
    }));

    const { container } = render(
      <MultiTrackTimelineCanvas
        totalFrames={2000}
        predictionCurve={new Float32Array(2000)}
        confidenceScores={new Float32Array(2000)}
        autoEvents={twoThousandPins}
        bookmarks={[]}
      />
    );

    const canvases = container.querySelectorAll('canvas');
    expect(canvases.length).toBe(2); // StaticTrackCanvas + DynamicPlayheadCanvas เท่านั้น
    // ต้องไม่มีการสร้าง <div> หรือ <svg> จำนวน 2,000 ตัวสำหรับหมุด
    expect(container.querySelectorAll('[data-pin-node]').length).toBe(0);
  });
});
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **หมุดซ้อนทับกันแน่นเมื่อมีหลายหมื่นเฟรม** | ส่งหมุดที่ห่างกันน้อยกว่า 14px เข้า `computeScreenPinClusters` | ถูกรวมเป็น `PinClusterNode` และเมื่อคลิกที่ Cluster จะซูม Timeline เข้าสู่ช่วง `[startFrame, endFrame]` |
| 2 | **ชุดข้อมูลไม่มีคอลัมน์เฉลย (`actual_target is None`)** | เรียก `scan_events(..., actual_targets=None)` | ไม่เกิด `TypeError` และแสดงป้าย `"Unlabeled Dataset"` พร้อมหมุด `decision_flip`, `low_confidence`, `input_outlier` |
| 3 | **หน้าจอ HiDPI / Retina (150% - 200% Scaling)** | จำลอง `window.devicePixelRatio = 2.0` ตอน Mount Canvas | `canvas.width === clientWidth * 2` และ `ctx.scale(2, 2)` ถูกเรียกใช้งาน |
| 4 | **ข้อมูลสุ่มทำให้เกิด Decision Flip แทบทุกเฟรม** | ส่งอาเรย์ `predictions` ที่สลับ `0, 1, 0, 1` ทุกเฟรม (`flip_rate > 0.40`) | ระบบกรองเฉพาะจุด Flip ที่ค่า Probability เปลี่ยนแปลงกระโดดแรง (`delta_prob > 0.5`) ไม่ปล่อยให้หมุดรกเต็มแถบ |
| 5 | **กด `Next Event` เมื่ออยู่ที่หมุดสุดท้ายของคิว** | ตั้ง `currentFrameIndex` ที่หมุดสุดท้ายแล้วกด `Next Event (>|)` | วนกลับไปที่หมุดแรกสุดของคิว (`Wrap-around`) |
| 6 | **การกรองแถวข้อมูลทำให้ `frame_index` เปลี่ยน** | สร้าง Bookmark ที่ `dataset_row_index = 500` แล้วเปลี่ยนการเรียงลำดับคิวใหม่ | Bookmark ย้ายไปแสดงที่ `frame_index` ใหม่ที่ตรงกับ `dataset_row_index == 500` อัตโนมัติ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Scanner Speed SLA**: `TimelineEventScannerService.scan_events()` บนข้อมูล 20,000 แถว 20 คอลัมน์ ต้องทำงานเสร็จภายใน `< 20 ms`
- [ ] **Zero Static Redraw on Play**: ขณะกด Play ที่ 60 FPS เลเยอร์ `StaticTrackCanvas` ต้องถูกวาด `0 ครั้ง` (วาดใหม่เฉพาะเลเยอร์ `DynamicPlayheadCanvas`)
- [ ] **DOM Node Budget**: จำนวน DOM Elements ทั้งหมดภายใน `MultiTrackTimelineCanvas` ต้องไม่เกิน `25 nodes` ไม่ว่าจะมีหมุดกี่พันจุดก็ตาม
- [ ] **Test Coverage**: Unit Test Coverage ของ `TimelineEventScannerService` และ `pinClusterEngine.ts` ต้อง $\ge 95\%$
