# Test Phase 02: Dataset Domain — CSV/XLSX/Parquet Ingestion, Profiling & `float32` Memory Guard

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 02)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-02-dataset-ingestion-and-memory-guard.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-02-dataset-ingestion-and-memory-guard.md) เพื่อตรวจสอบความถูกต้องของ Bounded Context **`domains/dataset/`** และ **`core/memory_guard.py`** ครอบคลุมการอัปโหลดไฟล์ `.csv`, `.tsv`, `.xlsx`, `.parquet`, การตรวจจับรหัสอักขระภาษาไทย (`cp874`, `tis-620`, `utf-8-sig`), การลดขนาด RAM ลง 50% ด้วย `float32` Downcasting, การล้างค่า `±inf` และ `#DIV/0!`, การคำนวณสถิติ `ColumnProfile`, การจัดเก็บลงไฟล์ `.parquet` แบบ Memory-mapped และการแสดงผลบน UI Step 1

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Core & Dataset Domain**:
  - `app.core.memory_guard.MemoryGuard` (`assert_can_ingest_file`, `downcast_dataframe`)
  - `app.domains.dataset.value_objects.ColumnProfile` และ `app.domains.dataset.entities.DatasetEntity`
  - `app.infrastructure.storage.tabular_reader.TabularReader` และ `parquet_dataset_repository.ParquetDatasetRepository`
  - REST Endpoints: `POST /api/v1/datasets/upload`, `POST /api/v1/datasets/load-sample/{sample_name}`
- **Frontend Dataset Domain**:
  - `src/domains/dataset/store/useDatasetStore.ts`
  - `DatasetDropzone.tsx`, `SheetSelectorBar.tsx`, `MemorySavingsBanner.tsx`, `ColumnProfileTable.tsx`, `DatasetSamplePreview.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_02/
├── conftest.py
├── test_memory_guard_downcast.py
├── test_thai_encoding_and_excel_reader.py
├── test_column_profiler_service.py
└── test_dataset_upload_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_02/test_memory_guard_downcast.py
import numpy as np
import pandas as pd
import pytest
from app.core.memory_guard import MemoryGuard
from app.core.exceptions import MemoryLimitExceededError

def test_float32_downcast_reduces_ram_and_sanitizes_infinity() -> None:
    """ทดสอบว่า downcast_dataframe แปลง float64 -> float32, int64 -> int32 และเปลี่ยน ±inf เป็น NaN"""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "price_f64": rng.uniform(1000.0, 500000.0, size=10000).astype(np.float64),
        "rooms_i64": rng.integers(1, 10, size=10000).astype(np.int64),
        "id_large_i64": np.arange(10_000_000_000, 10_000_010_000, dtype=np.int64),
    })
    df.loc[0, "price_f64"] = np.inf
    df.loc[1, "price_f64"] = -np.inf

    df_opt, orig_mb, opt_mb, inf_count = MemoryGuard.downcast_dataframe(df)

    assert df_opt["price_f64"].dtype == np.float32
    assert df_opt["rooms_i64"].dtype in (np.int8, np.int16, np.int32)
    # คอลัมน์จำนวนเต็มขนาดใหญ่เกิน int32 ต้องคงเป็น int64 เพื่อไม่ให้เลขท้ายเพี้ยน
    assert df_opt["id_large_i64"].dtype == np.int64
    assert inf_count == 2
    assert np.isnan(df_opt.loc[0, "price_f64"])
    assert np.isnan(df_opt.loc[1, "price_f64"])
    assert opt_mb < orig_mb * 0.70

def test_thai_cp874_and_utf8_sig_csv_ingestion(tmp_path) -> None:
    """ทดสอบการอ่านไฟล์ CSV ภาษาไทยที่เข้ารหัสแบบ CP874 (Excel Windows) และมี #DIV/0!"""
    from app.infrastructure.storage.tabular_reader import TabularReader

    csv_content = "เขตพื้นที่,ราคาประเมิน,อัตราส่วน\nจตุจักร,3500000.50,#DIV/0!\nบางรัก,8200000.00,1.25\n"
    cp874_file = tmp_path / "thai_excel_cp874.csv"
    cp874_file.write_bytes(csv_content.encode("cp874"))

    df, meta = TabularReader.read_file(cp874_file)
    assert meta.detected_encoding in ("cp874", "tis-620")
    assert list(df.columns) == ["เขตพื้นที่", "ราคาประเมิน", "อัตราส่วน"]
    assert df.loc[0, "เขตพื้นที่"] == "จตุจักร"
    assert pd.isna(df.loc[0, "อัตราส่วน"])
    assert df["อัตราส่วน"].dtype == np.float32
```

```python
# backend/tests/phase_02/test_dataset_upload_endpoints.py
import io
import json
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_upload_csv_endpoint_returns_json_compliant_profiles() -> None:
    """ทดสอบ POST /api/v1/datasets/upload ต้องไม่พังเมื่อมีค่า NaN/Inf และคืนค่า ColumnProfile ครบถ้วน"""
    csv_bytes = (
        " area_sqm , area_sqm ,bedrooms,district,price\n"
        "55.0,55.0,2,Chatuchak,3200000\n"
        "80.5,80.5,#N/A,Bangrak,inf\n"
        "120.0,120.0,4,Sathorn,9500000\n"
    ).encode("utf-8-sig")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        files = {"file": ("test_houses.csv", io.BytesIO(csv_bytes), "text/csv")}
        resp = await client.post("/api/v1/datasets/upload", files=files)
        assert resp.status_code == 200

        # ตรวจสอบว่าเป็น Standard JSON ที่ไม่มี NaN/Infinity literal
        raw_text = resp.text
        assert "NaN" not in raw_text and "Infinity" not in raw_text
        payload = json.loads(raw_text)

        assert payload["row_count"] == 3
        assert payload["detected_encoding"] == "utf-8-sig"
        assert payload["inf_sanitized_count"] == 1
        # ชื่อคอลัมน์ต้องถูก strip ช่องว่างและเติม suffix _1 เมื่อซ้ำกัน
        col_names = [c["name"] for c in payload["columns"]]
        assert col_names[0] == "area_sqm"
        assert col_names[1] == "area_sqm_1"
        assert len(payload["sample_rows"]) == 3
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/dataset/__tests__/
├── useDatasetStore.test.ts
├── DatasetDropzone.test.tsx
├── MemorySavingsBanner.test.tsx
└── ColumnProfileTable.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/dataset/__tests__/ColumnProfileTable.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { ColumnProfileTable } from '../components/ColumnProfileTable';
import { MemorySavingsBanner } from '../components/MemorySavingsBanner';
import { useDatasetStore } from '../store/useDatasetStore';

const mockDatasetResponse = {
  dataset_id: 'ds_test01',
  filename: 'thai_housing_data.csv',
  row_count: 125000,
  column_count: 2,
  original_ram_mb: 11.44,
  optimized_ram_mb: 5.72,
  inf_sanitized_count: 8,
  available_sheets: [],
  selected_sheet: null,
  columns: [
    {
      index: 0,
      name: 'area_sqm',
      logical_type: 'numeric' as const,
      storage_dtype: 'float32',
      null_count: 0,
      null_percentage: 0.0,
      unique_count: 840,
      min_value: 22.0,
      max_value: 450.0,
      mean_value: 84.5,
      std_value: 32.1,
      median_value: 78.0,
      histogram_bins: [22, 100, 200, 300, 450],
      histogram_counts: [100, 500, 300, 100],
      sample_values: [55.0, 82.5],
    },
    {
      index: 1,
      name: 'district_name',
      logical_type: 'categorical' as const,
      storage_dtype: 'string',
      null_count: 15,
      null_percentage: 0.012,
      unique_count: 50,
      min_value: null,
      max_value: null,
      mean_value: null,
      std_value: null,
      median_value: null,
      histogram_bins: [],
      histogram_counts: [],
      sample_values: ['จตุจักร', 'บางรัก'],
    },
  ],
  sample_rows: [{ area_sqm: 55.0, district_name: 'จตุจักร' }],
};

describe('Step 1 Dataset Memory Banner & Column Profile Inspector', () => {
  beforeEach(() => {
    useDatasetStore.setState({
      currentDataset: mockDatasetResponse,
      isUploading: false,
      uploadError: null,
      searchQuery: '',
      typeFilter: 'all',
    });
  });

  it('แสดงจำนวน RAM ที่ประหยัดได้ 50% และแจ้งเตือนจำนวน Infinity ที่ถูกล้างเป็น NaN', () => {
    render(<MemorySavingsBanner />);
    expect(screen.getByText(/11\.44 MB/i)).toBeInTheDocument();
    expect(screen.getByText(/5\.72 MB/i)).toBeInTheDocument();
    expect(screen.getByText(/50\.0%/i)).toBeInTheDocument();
    expect(screen.getByText(/8 Infinity/i)).toBeInTheDocument();
  });

  it('กรองคอลัมน์ด้วยช่องค้นหาและปุ่ม Filter ชนิดข้อมูล (Numeric vs Categorical) ได้ถูกต้อง', () => {
    render(<ColumnProfileTable />);
    expect(screen.getByText('area_sqm')).toBeInTheDocument();
    expect(screen.getByText('district_name')).toBeInTheDocument();

    const catFilterBtn = screen.getByRole('button', { name: /Categorical/i });
    fireEvent.click(catFilterBtn);
    expect(screen.queryByText('area_sqm')).not.toBeInTheDocument();
    expect(screen.getByText('district_name')).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบการคืน RAM หลังแปลง Parquet และ LRU Dataset Cache (`pytest`)
```python
# backend/tests/phase_02/test_parquet_memory_eviction.py
import os
import psutil
import numpy as np
import pandas as pd
from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository

def test_dataframe_is_evicted_from_ram_after_parquet_cache(tmp_path) -> None:
    """ยืนยันว่าหลังจากบันทึกไฟล์ .parquet แล้ว Repository จะไม่เก็บ DataFrame ค้างไว้ใน Python Dict"""
    repo = ParquetDatasetRepository(cache_dir=tmp_path, max_cached_datasets=3)
    proc = psutil.Process(os.getpid())

    # สร้างชุดข้อมูลจำลอง 4 ชุดเพื่อทดสอบ LRU Eviction (เก็บสูงสุด 3 ชุดบนดิสก์)
    for idx in range(4):
        df = pd.DataFrame({
            "x": np.random.rand(50_000).astype(np.float32),
            "y": np.random.rand(50_000).astype(np.float32),
        })
        repo.save_dataset_parquet(dataset_id=f"ds_{idx}", df=df)
        del df

    cached_files = list(tmp_path.glob("*.parquet"))
    assert len(cached_files) == 3
    assert not (tmp_path / "ds_0.parquet").exists()

    # ทดสอบการอ่านแบบ Memory-Mapped เฉพาะคอลัมน์เดียว
    table_slice = repo.read_columns_mmap("ds_3", columns=["x"], offset=0, limit=100)
    assert table_slice.shape == (100, 1)
    assert table_slice["x"].dtype == np.float32
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 02)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **ภาษาไทยกลายเป็นภาษาต่างดาว (Mojibake)** | `test_thai_cp874_and_utf8_sig_csv_ingestion()` | สร้างไฟล์ `.csv` เข้ารหัสด้วย `cp874`, `tis-620`, และ `utf-8-sig` แล้วอัปโหลด | อ่านชื่อคอลัมน์และข้อความภาษาไทยได้ตรง 100% โดยไม่เกิด `UnicodeDecodeError` |
| **2** | **JSON Serialization พังเพราะค่า `NaN` หรือ `Infinity`** | `test_upload_csv_endpoint_returns_json_compliant_profiles()` | ใส่ค่า `inf`, `-inf`, และค่าว่างลงในไฟล์ CSV แล้วเรียก API | API คืนสถานะ `200 OK` และไม่มีคำว่า `NaN` หรือ `Infinity` ปนอยู่ใน Raw JSON String |
| **3** | **สูตร Excel ติด Error (`#DIV/0!`, `#VALUE!`, `#N/A`)** | `test_excel_error_tokens_coerced_to_nan()` | สร้างตารางที่มีสตริง `#DIV/0!`, `#VALUE!`, `#N/A`, `#REF!` ในคอลัมน์ตัวเลข | ระบบแปลงเป็น `NaN` อัตโนมัติและยังคงตรวจจับคอลัมน์นั้นเป็น `numeric (float32)` |
| **4** | **ตัวเลขที่อยู่ในช่วงเกินขีดจำกัด `float32` (`> 3.4e38`)** | `test_extreme_float64_overflow_protection()` | ใส่ตัวเลข `1e100` ลงในคอลัมน์ `float64` | ระบบตรวจพบว่าเกิน `np.finfo(np.float32).max` จึงไม่แปลงเป็น `inf` แต่แจ้งเตือนหรือเก็บในช่วงปลอดภัย |
| **5** | **ชื่อคอลัมน์ซ้ำกันหรือมีช่องว่างนำหน้า/ต่อท้าย** | `test_duplicate_and_whitespace_column_names()` | อัปโหลด CSV ที่มีหัวตาราง `" area_sqm , area_sqm "` | ชื่อคอลัมน์ถูก `.strip()` และเปลี่ยนชื่อซ้ำเป็น `area_sqm`, `area_sqm_1` อัตโนมัติ |
| **6** | **คอลัมน์ ID ขนาดใหญ่ถูกแปลงเป็น `float32` จนเลขท้ายเพี้ยน** | `test_float32_downcast_reduces_ram_and_sanitizes_infinity()` | ส่งคอลัมน์ `int64` ที่มีค่า `10,000,000,001` เข้า `downcast_dataframe()` | คอลัมน์จำนวนเต็มใช้ `downcast="integer"` ทำให้ค่าไม่ถูกแปลงเป็น `float32` และเลขท้ายไม่เพี้ยน |
| **7** | **ไฟล์ว่างเปล่าหรือมีแต่บรรทัด Header (0 แถวข้อมูล)** | `test_empty_zero_row_csv_rejected()` | อัปโหลดไฟล์ CSV ที่มีเฉพาะบรรทัดชื่อคอลัมน์แต่ไม่มีข้อมูล | คืนค่า HTTP `400 Bad Request` พร้อมรหัส `EMPTY_DATASET_ERROR` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Memory Reduction SLA**: ชุดข้อมูลตัวเลข `float64` ทุกชุดที่ผ่าน `MemoryGuard.downcast_dataframe()` ต้องใช้ RAM ลดลงอย่างน้อย `45% - 50%`
- [ ] **Ingestion Speed SLA**: อัปโหลดและทำ Data Profiling ไฟล์ CSV ขนาด `100,000 แถว x 15 คอลัมน์` ต้องเสร็จสิ้นภายใน `< 1.5 วินาที`
- [ ] **Thai Encoding Compatibility**: ทดสอบไฟล์ CSV ภาษาไทยจากทั้ง Windows Excel (`cp874`) และ macOS/UTF-8 BOM (`utf-8-sig`) ผ่าน 100%
- [ ] **Bounded JSON Payload**: ขนาด Response JSON ของ `/api/v1/datasets/upload` ต้องมีขนาดไม่เกิน `100 KB` ไม่ว่าไฟล์ต้นฉบับจะมีกี่แสนแถวก็ตาม
