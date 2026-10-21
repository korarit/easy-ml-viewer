# Test Phase 17: Compute & Performance Core — Multi-Core CPU Pool, GPU Detection & Chunked Processing

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 17)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-17-compute-engine-cpu-gpu-and-chunking.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-17-compute-engine-cpu-gpu-and-chunking.md) เพื่อตรวจสอบความเสถียรของศูนย์กลางบริหารทรัพยากรการคำนวณ (**`AdaptiveComputeEngine`** ใน `backend/app/core/compute_pool.py`), การตั้งค่า `optimal_workers = max(1, physical_cores - 1)`, การตรวจจับ GPU Execution Providers พร้อมระบบ Fallback กลับสู่ CPU อัตโนมัติ, การป้องกันการหมดหน่วยความจำ (`InsufficientMemoryBudgetError`), และการแบ่งประมวลผลเป็นก้อน (`process_in_chunks`) ที่หยุดได้ทันทีด้วย `CancellationToken`

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Core, Domain & API**:
  - `backend/app/core/compute_pool.py`: `AdaptiveComputeEngine`, `CancellationToken`, `InsufficientMemoryBudgetError`, `JobCancelledError`
  - `backend/app/presentation/api/v1/compute_router.py`: `GET /api/v1/compute/telemetry`, `POST /api/v1/compute/jobs/{job_id}/cancel`, `POST /api/v1/compute/cache/purge`
- **Frontend Store & Components**:
  - `frontend/src/domains/compute/store/useComputeStatusStore.ts`
  - `frontend/src/domains/compute/components/HardwareStatusPill.tsx`, `ComputeDiagnosticsModal.tsx`, `ChunkedJobProgressBanner.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `AdaptiveComputeEngine` & `CancellationToken`
ไฟล์ทดสอบ: `backend/tests/unit/core/test_compute_pool.py`

```python
import os
import numpy as np
import pytest
from app.core.compute_pool import (
    AdaptiveComputeEngine,
    CancellationToken,
    InsufficientMemoryBudgetError,
    JobCancelledError,
)


def test_adaptive_compute_engine_reserves_one_core_and_sets_omp_env():
    """ตรวจสอบว่า optimal_workers เว้นอย่างน้อย 1 คอร์ไว้ให้ UI/OS และตั้งค่า OMP_NUM_THREADS ถูกต้อง"""
    engine = AdaptiveComputeEngine()
    assert engine.optimal_workers >= 1
    if engine.physical_cores > 1:
        assert engine.optimal_workers == engine.physical_cores - 1
    assert "OMP_NUM_THREADS" in os.environ
    assert " CPUExecutionProvider" in " " + " ".join(engine.get_preferred_onnx_providers(prefer_gpu=False))


def test_process_in_chunks_reports_progress_and_yields_all_batches():
    """ตรวจสอบว่าการประมวลผลแบบแบ่งก้อน (Chunked Generator) ครบทุกแถวและส่งค่า Progress % ถูกต้อง"""
    engine = AdaptiveComputeEngine()
    X_large = np.ones((12500, 10), dtype=np.float64)
    progress_log: list[tuple[int, int, float]] = []

    def chunk_worker(chunk_arr: np.ndarray, start_i: int, end_i: int) -> tuple[int, int, str]:
        assert chunk_arr.dtype == np.float32
        assert chunk_arr.flags["C_CONTIGUOUS"]
        return (start_i, end_i, f"rows_{len(chunk_arr)}")

    results = list(
        engine.process_in_chunks(
            X=X_large,
            chunk_fn=chunk_worker,
            chunk_size=5000,
            on_progress=lambda done, total, pct: progress_log.append((done, total, pct)),
        )
    )

    assert len(results) == 3
    assert results[0] == (0, 5000, "rows_5000")
    assert results[1] == (5000, 10000, "rows_5000")
    assert results[2] == (10000, 12500, "rows_2500")
    assert [p[2] for p in progress_log] == [40.0, 80.0, 100.0]


def test_process_in_chunks_stops_immediately_when_cancellation_token_triggered():
    """Edge Case #4: เมื่อผู้ใช้สั่ง cancel_token.cancel() ลูปต้องหยุดทันทีในก้อนถัดไปและโยน JobCancelledError"""
    engine = AdaptiveComputeEngine()
    X_data = np.zeros((20000, 5), dtype=np.float32)
    token = CancellationToken()
    processed_chunks = 0

    def worker(chunk_arr: np.ndarray, s: int, e: int) -> int:
        nonlocal processed_chunks
        processed_chunks += 1
        if processed_chunks == 2:
            token.cancel()  # จำลองผู้ใช้กดปุ่ม Cancel หลังก้อนที่ 2
        return len(chunk_arr)

    with pytest.raises(JobCancelledError, match="ถูกยกเลิกโดยผู้ใช้"):
        list(engine.process_in_chunks(X_data, worker, chunk_size=5000, cancel_token=token))

    # ต้องประมวลผลไปเพียง 2 ก้อนแล้วหยุดทันที ไม่รันก้อนที่ 3 และ 4 ต่อ!
    assert processed_chunks == 2


def test_check_memory_budget_raises_insufficient_memory_when_ram_exceeded(monkeypatch):
    """ตรวจสอบว่าเมื่อต้องการ RAM เกินกว่าที่มีอยู่จริง ระบบเรียก gc.collect() แล้วโยน InsufficientMemoryBudgetError"""
    engine = AdaptiveComputeEngine()
    # ขอหน่วยความจำ 999 TB เพื่อจำลองสถานการณ์ RAM ไม่พอ
    with pytest.raises(InsufficientMemoryBudgetError, match="หน่วยความจำระบบไม่เพียงพอ"):
        engine.check_memory_budget(required_extra_bytes=999 * 1024**4)
```

### 2.2 API Integration Tests สำหรับ Hardware Telemetry & Cache Purge
ไฟล์ทดสอบ: `backend/tests/integration/api/test_compute_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_compute_telemetry_and_purge_cache(async_client: AsyncClient):
    """ทดสอบ GET /api/v1/compute/telemetry และ POST /api/v1/compute/cache/purge"""
    tel_resp = await async_client.get("/api/v1/compute/telemetry")
    assert tel_resp.status_code == 200
    body = tel_resp.json()
    assert body["physical_cores"] >= 1
    assert body["optimal_workers"] >= 1
    assert body["process_rss_mb"] > 0.0

    purge_resp = await async_client.post("/api/v1/compute/cache/purge")
    assert purge_resp.status_code == 200
    assert "process_rss_mb_after" in purge_resp.json()
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Component Tests สำหรับ `HardwareStatusPill.tsx` & `ChunkedJobProgressBanner.tsx`
ไฟล์ทดสอบ: `frontend/src/domains/compute/__tests__/HardwareStatusPill.test.tsx`

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HardwareStatusPill } from '../components/HardwareStatusPill';
import { ChunkedJobProgressBanner } from '../components/ChunkedJobProgressBanner';

describe('Phase 17: HardwareStatusPill & ChunkedJobProgressBanner', () => {
  it('แสดงไฟสถานะสีแดง (Memory Guard Active) เมื่อการใช้ RAM ของระบบเกิน 80%', () => {
    render(
      <HardwareStatusPill
        physicalCores={8}
        logicalCores={16}
        processRssMb={1250}
        systemMemoryPercent={84.5}
        activeProvider="DmlExecutionProvider"
      />
    );

    expect(screen.getByText(/8 Cores/i)).toBeInTheDocument();
    expect(screen.getByTestId('memory-status-indicator')).toHaveAttribute('data-severity', 'critical');
  });

  it('แสดงเปอร์เซ็นต์ความคืบหน้าของ Chunked Job และเรียก onCancel เมื่อคลิกปุ่ม Cancel หรือกด Esc', () => {
    const onCancelMock = vi.fn();
    render(
      <ChunkedJobProgressBanner
        jobId="job_01"
        currentChunk={3}
        totalChunks={10}
        progressPercent={30}
        onCancel={onCancelMock}
      />
    );

    expect(screen.getByText(/30%/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Cancel/i }));
    expect(onCancelMock).toHaveBeenCalledWith('job_01');
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การพิสูจน์ Flat Memory Profile ใน `process_in_chunks` และกฎห้ามใช้ `ProcessPoolExecutor` บน Windows
```python
import inspect
import numpy as np
import psutil
import app.core.compute_pool as compute_pool_mod
from app.core.compute_pool import AdaptiveComputeEngine


def test_compute_pool_uses_thread_pool_and_never_spawns_process_pool_on_windows():
    """Edge Case #1: ยืนยันว่า compute_pool.py ใช้ ThreadPoolExecutor และไม่ใช้ ProcessPoolExecutor"""
    source_code = inspect.getsource(compute_pool_mod)
    assert "ThreadPoolExecutor" in source_code
    assert "ProcessPoolExecutor" not in source_code


def test_flat_memory_profile_across_10_chunks_of_50k_rows():
    """พิสูจน์ว่าการใช้ process_in_chunks ประมวลผล 50,000 แถว มีการใช้ RAM แบนราบ (Flat Memory Profile)"""
    engine = AdaptiveComputeEngine()
    X_50k = np.ones((50000, 20), dtype=np.float32)
    proc = psutil.Process()
    rss_samples_mb: list[float] = []

    def heavy_chunk_op(chunk: np.ndarray, s: int, e: int) -> float:
        # คำนวณชั่วคราวภายใน Chunk
        temp = np.sin(chunk) * 2.0
        val = float(temp.mean())
        rss_samples_mb.append(proc.memory_info().rss / (1024 * 1024))
        return val

    _ = list(engine.process_in_chunks(X_50k, heavy_chunk_op, chunk_size=5000))

    # ความต่างของ RAM ระหว่างก้อนที่ 1 กับก้อนที่ 10 ต้องไม่โตขึ้นสะสม (น้อยกว่า 10 MB)
    ram_drift_mb = rss_samples_mb[-1] - rss_samples_mb[0]
    assert ram_drift_mb < 10.0
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **`ProcessPoolExecutor` บน Windows กิน RAM ซ้ำและช้า** | ตรวจ Source & Runtime ของ `AdaptiveComputeEngine` | ใช้ `ThreadPoolExecutor` ร่วมกับ C++ `n_jobs` แชร์หน่วยความจำโมเดล Zero-Copy |
| 2 | **Thread Oversubscription ($16 \times 16$ Threads)** | ตรวจค่า `os.environ["OMP_NUM_THREADS"]` และ `MKL_NUM_THREADS` | ถูกจำกัดไว้ไม่เกิน `optimal_workers` เสมอ |
| 3 | **GPU Provider ล้มเหลวหรือไม่มีไดรเวอร์ CUDA** | Mock `onnxruntime.get_available_providers` ให้โยน Exception | Fallback กลับเป็น `["CPUExecutionProvider"]` โดยอัตโนมัติ ไม่เกิด Crash |
| 4 | **ผู้ใช้กดปุ่ม Cancel Job ระหว่างประมวลผลชุดข้อมูลใหญ่** | เรียก `cancel_token.cancel()` ในระหว่างรัน Chunk ที่ 2 | โยน `JobCancelledError` ทันทีก่อนเริ่ม Chunk ที่ 3 (ภายใน `< 50ms`) |
| 5 | **`psutil.cpu_count(logical=False)` คืนค่าเป็น `None` บน VM** | Monkeypatch `psutil.cpu_count` ให้คืน `None` | Fallback ไปใช้ค่าเริ่มต้น `2` และได้ `optimal_workers >= 1` เสมอ |
| 6 | **ข้อมูลตารางขนาดเล็กถูกส่งไป GPU แล้วเสียเวลา PCIe Copy** | ทดสอบ Smart Router ด้วยข้อมูลขนาด 500 แถว | เลือกใช้ Vectorized CPU โดยอัตโนมัติ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Flat Memory SLA**: การประมวลผล 50,000 แถวผ่าน `process_in_chunks` ต้องมีค่า RAM Drift ระหว่างก้อนแรกกับก้อนสุดท้าย `< 10 MB`
- [ ] **Cancellation Responsiveness**: เมื่อกดยกเลิกงาน (`Cancel Job`) ลูปการคำนวณต้องหยุดภายใน `< 60 ms`
- [ ] **UI Core Reservation**: บนเครื่องที่มี CPU $\ge 2$ Physical Cores ระบบต้องเว้นอย่างน้อย 1 คอร์ให้ระบบ UI และ OS เสมอ (`optimal_workers = physical_cores - 1`)
- [ ] **Test Coverage**: Unit & Integration Coverage ของ `compute_pool.py` ต้อง $\ge 95\%$
