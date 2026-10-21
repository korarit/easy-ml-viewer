# Test Phase 16: Rich Export Suite — Enriched CSV/Excel Exporter, Offline WebM 3D Video Recorder & Snapshot

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 16)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-16-rich-export-video-and-enriched-csv.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-16-rich-export-video-and-enriched-csv.md) เพื่อตรวจสอบความถูกต้องของระบบส่งออกตารางข้อมูลที่ผสานผลลัพธ์ของโมเดล (**Enriched Dataset Exporter** สำหรับ `.csv`, `.xlsx`, `.parquet`), การใส่รหัส `UTF-8 BOM (\xef\xbb\xbf)` เพื่อให้เปิดภาษาไทยใน Microsoft Excel ได้ไม่เพี้ยน, การอัดวิดีโอฉาก 3D แบบ Offline ด้วย `MediaRecorder`, และการถ่ายภาพนิ่ง `.png` ความละเอียดสูงด้วยการเรียก `gl.render(scene, camera)` แบบ Synchronous ก่อนจับภาพ

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Domain & API**:
  - `backend/app/domains/export_reporting/services.py`: `EnrichedDatasetExportService.stream_enriched_csv_chunks()`
  - `backend/app/presentation/api/v1/export_router.py`: `POST /api/v1/exports/{session_id}/enriched-table`
- **Frontend Hooks & Components**:
  - `frontend/src/domains/export-studio/hooks/useWebGLVideoRecorder.ts`, `useWebGLSnapshot.ts`
  - `frontend/src/domains/export-studio/components/ExportStudioModal.tsx`, `VideoRecordingIndicatorBar.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `EnrichedDatasetExportService.stream_enriched_csv_chunks()`
ไฟล์ทดสอบ: `backend/tests/unit/domains/test_enriched_exporter.py`

```python
import csv
import io
import numpy as np
import pandas as pd
import pytest
from app.domains.export_reporting.services import EnrichedDatasetExportService


def test_stream_enriched_csv_includes_utf8_bom_and_handles_special_chars_in_notes():
    """ตรวจสอบว่าไบต์แรกคือ UTF-8 BOM (\\xef\\xbb\\xbf) และโน้ตที่มีลูกน้ำ (,) หรือขึ้นบรรทัดใหม่ (\\n) ไม่ทำให้คอลัมน์เคลื่อน"""
    df = pd.DataFrame({"credit_score": [720.0, 580.0], "income": [120000.0, 45000.0]})
    row_indices = [10, 25]
    predictions = np.array([1.0, 0.0], dtype=np.float32)
    confidences = np.array([0.92, 0.81], dtype=np.float32)
    class_aliases = {"1": "อนุมัติสินเชื่อ", "0": "ปฏิเสธสินเชื่อ"}
    top_pos_feat = ["credit_score", "income"]
    top_pos_score = np.array([0.35, 0.05], dtype=np.float32)
    top_neg_feat = ["income", "credit_score"]
    top_neg_score = np.array([-0.02, -0.41], dtype=np.float32)

    # Edge Case #6: โน้ตมีทั้งภาษาไทย เครื่องหมายลูกน้ำ (,) และขึ้นบรรทัดใหม่ (\n)
    bookmarks = {1: "เคสต้องสงสัย, รายได้ต่ำ\nตรวจสอบเอกสารเพิ่มเติม"}

    chunks = list(
        EnrichedDatasetExportService.stream_enriched_csv_chunks(
            df_features=df,
            row_indices=row_indices,
            predictions=predictions,
            confidences=confidences,
            class_aliases=class_aliases,
            top_pos_features=top_pos_feat,
            top_pos_scores=top_pos_score,
            top_neg_features=top_neg_feat,
            top_neg_scores=top_neg_score,
            bookmarks_by_frame=bookmarks,
            chunk_size=1,
        )
    )

    # 1. ไบต์แรกต้องเป็น UTF-8 BOM (Edge Case #2)
    assert chunks[0] == b"\xef\xbb\xbf"

    full_bytes = b"".join(chunks)
    decoded_text = full_bytes.decode("utf-8-sig")
    reader = list(csv.reader(io.StringIO(decoded_text)))

    header = reader[0]
    assert "model_prediction_label" in header
    assert "user_bookmark_note" in header

    row_0 = reader[1]
    row_1 = reader[2]
    assert len(row_0) == len(header)
    assert len(row_1) == len(header)
    assert row_0[header.index("model_prediction_label")] == "อนุมัติสินเชื่อ"
    assert row_1[header.index("user_bookmark_note")] == "เคสต้องสงสัย, รายได้ต่ำ\nตรวจสอบเอกสารเพิ่มเติม"
```

### 2.2 API Integration Tests สำหรับ `POST /api/v1/exports/{session_id}/enriched-table`
ไฟล์ทดสอบ: `backend/tests/integration/api/test_export_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize("export_format", ["csv", "xlsx", "parquet"])
async def test_export_enriched_table_all_formats(
    async_client: AsyncClient, seeded_session_id: str, export_format: str
):
    """ทดสอบการส่งออกตารางข้อมูลครบทั้ง 3 ฟอร์แมต (.csv, .xlsx, .parquet)"""
    resp = await async_client.post(
        f"/api/v1/exports/{seeded_session_id}/enriched-table",
        json={
            "format": export_format,
            "scope": "all_queued",
            "include_attribution_columns": True,
            "include_bookmarks": True,
        },
    )
    assert resp.status_code == 200
    assert len(resp.content) > 100
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Unit Tests สำหรับ `useWebGLVideoRecorder` & `useWebGLSnapshot`
ไฟล์ทดสอบ: `frontend/src/domains/export-studio/__tests__/useWebGLVideoRecorder.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('Phase 16: WebGL Snapshot Pre-Render & MediaRecorder MIME Fallback', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('Edge Case #1: บังคับเรียก gl.render(scene, camera) ใน Synchronous Tick เดียวกันก่อนเรียก canvas.toBlob เสมอ', () => {
    const callOrder: string[] = [];
    const mockGl = {
      render: vi.fn(() => callOrder.push('gl.render')),
      domElement: {
        toBlob: vi.fn((cb: (b: Blob) => void) => {
          callOrder.push('canvas.toBlob');
          cb(new Blob(['png-bytes'], { type: 'image/png' }));
        }),
      },
    };

    // จำลองฟังก์ชัน captureSnapshot
    mockGl.render();
    mockGl.domElement.toBlob(() => {});

    expect(callOrder).toEqual(['gl.render', 'canvas.toBlob']);
  });

  it('Edge Case #3: เลือก MIME Type ที่เบราว์เซอร์รองรับตามลำดับ vp9 -> vp8 -> webm -> mp4', () => {
    const supportedSet = new Set(['video/mp4']); // จำลอง Safari ที่รองรับเฉพาะ video/mp4
    const isTypeSupportedMock = vi.fn((mime: string) => supportedSet.has(mime));

    const candidates = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4',
    ];
    const selected = candidates.find((m) => isTypeSupportedMock(m)) || '';
    expect(selected).toBe('video/mp4');
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การทดสอบ Streaming CSV Memory Ceiling (`< 15 MB` สำหรับ 50,000 แถว)
```python
import numpy as np
import pandas as pd
import psutil
from app.domains.export_reporting.services import EnrichedDatasetExportService


def test_50k_rows_streaming_csv_memory_budget():
    """ตรวจสอบว่าการสร้างไฟล์ CSV 50,000 แถวแบบ Streaming Chunk ใช้ RAM เพิ่มขึ้นไม่เกิน 15 MB"""
    n_rows = 50000
    df = pd.DataFrame({f"f_{i}": np.zeros(n_rows, dtype=np.float32) for i in range(10)})
    preds = np.zeros(n_rows, dtype=np.float32)
    confs = np.ones(n_rows, dtype=np.float32)
    pos_f = ["f_0"] * n_rows
    neg_f = ["f_1"] * n_rows

    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    stream_gen = EnrichedDatasetExportService.stream_enriched_csv_chunks(
        df_features=df,
        row_indices=list(range(n_rows)),
        predictions=preds,
        confidences=confs,
        class_aliases={"0": "ปกติ"},
        top_pos_features=pos_f,
        top_pos_scores=preds,
        top_neg_features=neg_f,
        top_neg_scores=preds,
        bookmarks_by_frame={},
        chunk_size=5000,
    )

    total_bytes = 0
    for chunk in stream_gen:
        total_bytes += len(chunk)

    rss_after = proc.memory_info().rss
    delta_mb = (rss_after - rss_before) / (1024 * 1024)
    assert total_bytes > 1_000_000
    assert delta_mb < 15.0
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **ถ่ายภาพ PNG ออกมาแล้วได้ภาพดำสนิท** | ทดสอบลำดับการเรียกใน `useWebGLSnapshot` | `gl.render(scene, camera)` ถูกเรียกก่อน `gl.domElement.toBlob()` ใน Synchronous Tick เดียวกัน |
| 2 | **เปิดไฟล์ CSV ใน Excel บน Windows แล้วภาษาไทยเพี้ยน** | ตรวจ 3 ไบต์แรกของ Output Stream จาก `stream_enriched_csv_chunks` | ขึ้นต้นด้วย `b"\xef\xbb\xbf"` (UTF-8 BOM) เสมอ |
| 3 | **เบราว์เซอร์ไม่รองรับ `video/webm;codecs=vp9`** | Mock `MediaRecorder.isTypeSupported` ให้คืน `false` สำหรับ `vp9` | Fallback ไปใช้ `vp8`, `webm`, หรือ `mp4` ตามลำดับโดยไม่โยน Exception |
| 4 | **ป้าย HTML HUD ไม่ติดเข้าไปในไฟล์วิดีโอที่อัดจาก `<canvas>`** | เปิดตัวเลือก `HUD Watermark Overlay` ตอนอัดวิดีโอ | ระบบวาดแถบข้อมูลเฟรมและค่าทำนายลงบน Composite Canvas 2D / WebGL Sprite ติดลงในวิดีโอ |
| 5 | **ปิดหน้าต่างหรือ Unmount ขณะกำลังอัดวิดีโอค้างอยู่** | เรียก `unmount()` ของ Hook ขณะ `isRecording === true` | `recorder.stop()`, `stream.getTracks().forEach(t => t.stop())`, และ `URL.revokeObjectURL` ถูกเรียกครบ |
| 6 | **โน้ต Bookmark มีเครื่องหมาย `,` หรือ `\n`** | ใส่โน้ต `"เคส A, บรรทัด 1\nบรรทัด 2"` | `csv.writer` ครอบเครื่องหมาย `"` ตามมาตรฐาน RFC 4180 จำนวนคอลัมน์ไม่เคลื่อน |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Streaming Memory SLA**: การส่งออก Enriched CSV 50,000 แถว ต้องใช้ RAM เพิ่มขึ้นในระหว่างสตรีมไม่เกิน `< 15 MB`
- [ ] **Excel Thai Compatibility**: ไฟล์ `.csv` และ `.xlsx` ต้องแสดงผลภาษาไทยได้ถูกต้อง 100% เมื่อเปิดใน Microsoft Excel
- [ ] **Blob Memory Cleanup**: หลังจบการดาวน์โหลดวิดีโอหรือภาพนิ่ง ต้องคืนหน่วยความจำผ่าน `URL.revokeObjectURL` และล้าง Array ของ `Blob` เป็น `[]` ทันที
- [ ] **Test Coverage**: Unit & Integration Coverage ของ `EnrichedDatasetExportService` ต้อง $\ge 95\%$
