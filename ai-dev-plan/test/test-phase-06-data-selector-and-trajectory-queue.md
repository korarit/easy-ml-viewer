# Test Phase 06: Data Selector Studio — Virtualized Table, Multi-Select, Query Filter & Trajectory Queue

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 06)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-06-data-selector-and-trajectory-queue.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-06-data-selector-and-trajectory-queue.md) เพื่อตรวจสอบความถูกต้องและประสิทธิภาพของระบบคัดเลือกแถวข้อมูล (**Data Selector Studio & Trajectory Playback Queue - Step 4**) ครอบคลุมการกรองข้อมูลด้วย `FilterRule` ที่ปลอดภัย (ไม่ใช้ `eval`), การเรียงลำดับคิวการเล่นแบบ Stable Sort (`kind="mergesort"`), การรักษาหมายเลขแถวต้นฉบับ (`_original_row_index`), การเลือกแถวแบบ Checkbox / `Shift + Click` / Range + Stride / Presets, และการเรนเดอร์ตาราง `VirtualizedDatasetTable` ด้วย `@tanstack/react-virtual` ที่จำกัดจำนวน `<tr>` ใน DOM ไม่เกิน 40 โหนด

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Dataset Query Service**:
  - `app.domains.dataset.query_service` (`FilterRule`, `DatasetQueryRequest`, `DatasetQueryResponse`, `DatasetQueryService.execute_query`)
  - REST Endpoints: `POST /api/v1/datasets/{dataset_id}/query`, `POST /api/v1/datasets/{dataset_id}/resolve-selection`
- **Frontend Trajectory Queue Domain**:
  - `src/domains/trajectory_queue/store/useTrajectoryQueueStore.ts`
  - `DataSelectorToolbar.tsx`, `FilterConditionBar.tsx`, `VirtualizedDatasetTable.tsx`, `TrajectoryQueueDock.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_06/
├── conftest.py
├── test_dataset_query_service.py
├── test_original_row_index_preservation.py
└── test_dataset_query_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_06/test_dataset_query_service.py
import numpy as np
import pandas as pd
from app.domains.dataset.query_service import (
    DatasetQueryRequest,
    DatasetQueryService,
    FilterRule,
)

def test_filter_and_sort_preserves_original_csv_row_index() -> None:
    """
    ทดสอบว่าเมื่อกรองและเรียงลำดับข้อมูลใหม่ ฟิลด์ _original_row_index
    จะต้องยังคงเป็นหมายเลขแถวเดิมในไฟล์ CSV ต้นฉบับเสมอ
    """
    df = pd.DataFrame({
        "area_sqm": np.array([45.0, 120.0, 65.0, 95.0, 80.0], dtype=np.float32),
        "bedrooms": np.array([1, 4, 2, 3, 2], dtype=np.int32),
        "selling_price": np.array([2.1e6, 7.5e6, 3.2e6, 5.1e6, 4.0e6], dtype=np.float32),
    })

    # กรองเฉพาะ area_sqm >= 60.0 (ตรงกับแถวเดิม #1, #2, #3, #4) แล้วเรียงจากน้อยไปมาก
    req = DatasetQueryRequest(
        filters=[FilterRule(column="area_sqm", operator="gte", value=60.0)],
        sort_by="area_sqm",
        sort_ascending=True,
        offset=0,
        limit=10,
    )

    res = DatasetQueryService.execute_query(df, req)
    assert res.total_dataset_rows == 5
    assert res.total_filtered_rows == 4

    # ลำดับหลังเรียง area_sqm จากน้อยไปมากต้องเป็น: 65.0 (#2) -> 80.0 (#4) -> 95.0 (#3) -> 120.0 (#1)
    assert res.matching_row_indices_preview == [2, 4, 3, 1]
    assert [r["_original_row_index"] for r in res.rows] == [2, 4, 3, 1]
    assert [r["area_sqm"] for r in res.rows] == [65.0, 80.0, 95.0, 120.0]

def test_stable_mergesort_and_nan_positioned_last() -> None:
    """ทดสอบกรณีมีค่าซ้ำกันต้องรักษาลำดับเดิม (Stable Sort) และแถวที่มีค่า NaN ในคอลัมน์ที่ใช้ Sort ต้องอยู่ท้ายสุด"""
    df = pd.DataFrame({
        "bedrooms": [2.0, np.nan, 2.0, 1.0],
        "area_sqm": [50.0, 90.0, 60.0, 40.0],
    })
    req = DatasetQueryRequest(sort_by="bedrooms", sort_ascending=True, offset=0, limit=10)
    res = DatasetQueryService.execute_query(df, req)

    # ลำดับต้องเป็น: bedrooms=1.0 (#3) -> bedrooms=2.0 ตัวแรก (#0) -> bedrooms=2.0 ตัวที่สอง (#2) -> NaN (#1)
    assert [r["_original_row_index"] for r in res.rows] == [3, 0, 2, 1]
    assert res.rows[-1]["bedrooms"] is None
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/trajectory_queue/__tests__/
├── useTrajectoryQueueStore.test.ts
├── VirtualizedDatasetTable.test.tsx
└── TrajectoryQueueDock.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/trajectory_queue/__tests__/useTrajectoryQueueStore.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useTrajectoryQueueStore } from '../store/useTrajectoryQueueStore';

describe('useTrajectoryQueueStore Multi-Select, Shift+Click & Range Stride', () => {
  beforeEach(() => {
    useTrajectoryQueueStore.setState({
      selectedRowIndices: [],
      selectedLookupSet: new Set(),
      lastClickedRowIndex: null,
      filters: [],
      sortBy: null,
      sortAscending: true,
      totalFilteredRows: 1000,
    });
  });

  it('รองรับการเลือกแบบ Range พร้อม Step Stride และรักษา O(1) Lookup Set ให้ตรงกับ Array เสมอ', () => {
    useTrajectoryQueueStore.getState().selectRange(10, 20, 3);
    const state = useTrajectoryQueueStore.getState();

    // 10, 13, 16, 19
    expect(state.selectedRowIndices).toEqual([10, 13, 16, 19]);
    expect(state.selectedLookupSet.has(13)).toBe(true);
    expect(state.selectedLookupSet.has(14)).toBe(false);
  });

  it('รองรับ Shift+Click โดยเลือกเฉพาะแถวที่ผ่านการกรองอยู่ใน visibleIndices เท่านั้น (ไม่รวมแถวที่ถูกซ่อน)', () => {
    const visibleFilteredIndices = [5, 12, 29, 44, 81];
    const store = useTrajectoryQueueStore.getState();

    // คลิกแถว #12 ก่อน
    store.toggleRowSelection(12, false, visibleFilteredIndices);
    // กด Shift+Click ที่แถว #44 -> ต้องได้ [12, 29, 44] เท่านั้น ห้ามติดเลข 13..28 ที่ถูกกรองออก
    store.toggleRowSelection(44, true, visibleFilteredIndices);

    expect(useTrajectoryQueueStore.getState().selectedRowIndices).toEqual([12, 29, 44]);
  });
});
```

```tsx
// frontend/src/domains/trajectory_queue/__tests__/VirtualizedDatasetTable.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { VirtualizedDatasetTable } from '../components/VirtualizedDatasetTable';

describe('VirtualizedDatasetTable DOM Node Cap Test', () => {
  it('เรนเดอร์จำนวน <tr> ใน DOM ไม่เกิน 40 แถว แม้ว่าข้อมูลจะมี 100,000 แถว', () => {
    const sampleWindowRows = Array.from({ length: 200 }, (_, idx) => ({
      _original_row_index: idx,
      area_sqm: 50 + idx,
      bedrooms: 2,
      selling_price: 3000000 + idx * 10000,
    }));

    const { container } = render(
      <div style={{ height: '500px', overflow: 'auto' }}>
        <VirtualizedDatasetTable
          totalRows={100000}
          loadedRows={sampleWindowRows}
          columns={['area_sqm', 'bedrooms', 'selling_price']}
        />
      </div>
    );

    const renderedTrElements = container.querySelectorAll('tbody tr');
    expect(renderedTrElements.length).toBeGreaterThan(0);
    expect(renderedTrElements.length).toBeLessThanOrEqual(40);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบเพดาน `Max Active Animated Trajectory (10,000 Frames)` (`pytest`)
```python
# backend/tests/phase_06/test_selection_cap_and_query_speed.py
import time
import numpy as np
import pandas as pd
from app.domains.dataset.query_service import DatasetQueryRequest, DatasetQueryService, FilterRule

def test_query_100k_rows_completes_under_30ms_and_caps_indices() -> None:
    """ทดสอบการกรองข้อมูล 100,000 แถว ต้องเสร็จภายใน < 30ms และจำกัด preview indices ไม่เกิน 10,000 แถว"""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "f0": rng.uniform(0.0, 100.0, size=100_000).astype(np.float32),
        "f1": rng.integers(1, 5, size=100_000).astype(np.int32),
    })

    req = DatasetQueryRequest(
        filters=[FilterRule(column="f0", operator="gte", value=10.0)],
        offset=0,
        limit=200,
    )

    t0 = time.perf_counter()
    res = DatasetQueryService.execute_query(df, req)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 30.0
    assert len(res.rows) == 200
    assert len(res.matching_row_indices_preview) <= 10_000
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 06)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **หมายเลขแถวหลัง Filter/Sort ไม่ตรงกับแถวในไฟล์ CSV ต้นฉบับ** | `test_filter_and_sort_preserves_original_csv_row_index()` | กรองข้อมูลให้เหลือครึ่งเดียวแล้วเรียงย้อนกลับ | `_original_row_index` ยังคงชี้ไปยังเลขแถวเดิมใน CSV ต้นฉบับอย่างแม่นยำ |
| **2** | **เบราว์เซอร์ค้างเมื่อผู้ใช้กด `Select All` กับไฟล์ 500,000 แถว** | `test_query_100k_rows_completes_under_30ms_and_caps_indices()` | กด `Select All Matching` บนชุดข้อมูล 100,000+ แถว | ระบบจำกัดคิวแอนิเมชันไว้ที่เพดาน `10,000 เฟรม` (ด้วย Stride Sampling) ไม่ทำให้เบราว์เซอร์ค้าง |
| **3** | **การเรียงลำดับ (`sort_values`) ทำให้ลำดับของแถวที่มีค่าเท่ากันสลับไปมา** | `test_stable_mergesort_and_nan_positioned_last()` | เรียงลำดับคอลัมน์ที่มีค่าซ้ำกันหลายแถว | ใช้ `kind="mergesort"` รักษาลำดับก่อน-หลังตามแถวเดิมเสมอ |
| **4** | **ผู้ใช้กด `Shift + Click` ข้ามช่วงที่ถูกกรองออกไปแล้ว** | `useTrajectoryQueueStore` Shift+Click test | กรองให้เหลือแถว `[5, 12, 29, 44, 81]` แล้วกด `Shift+Click` จาก `12` ไป `44` | คิวมีเฉพาะ `[12, 29, 44]` ไม่มีแถวที่ถูกซ่อนปนเข้ามา |
| **5** | **ผู้ใช้เผลอกดปุ่มไปต่อโดยที่มี `0 แถว` ในคิว** | `test_launch_3d_button_disabled_when_queue_empty()` | กด `Clear Queue` จน `selectedRowIndices = []` | ปุ่ม `Launch 3D Simulation Studio ->` มีสถานะ `disabled` |
| **6** | **คอลัมน์ที่ใช้ Sort มีค่า `NaN` ปนอยู่** | `test_stable_mergesort_and_nan_positioned_last()` | เรียงลำดับคอลัมน์ที่มีค่า `NaN` ทั้งแบบ Ascending และ Descending | แถวที่มีค่า `NaN` ถูกจัดไว้ท้ายสุด (`na_position="last"`) เสมอ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **DOM Node Ceiling SLA**: จำนวน `<tr>` ที่ถูกสร้างใน DOM ของ `VirtualizedDatasetTable` ต้องไม่เกิน `40 แถว` ตลอดการเลื่อนตาราง
- [ ] **Query Latency SLA**: การกรองและตัดหน้าข้อมูล (`offset/limit`) บนชุดข้อมูล `100,000 แถว` ต้องใช้เวลา `< 30 ms`
- [ ] **$O(1)$ Checkbox Lookup**: การตรวจสอบสถานะการเลือกแถวใช้ `Set.has()` เสมอ ไม่มีการใช้ `Array.includes()` ใน Render Loop
