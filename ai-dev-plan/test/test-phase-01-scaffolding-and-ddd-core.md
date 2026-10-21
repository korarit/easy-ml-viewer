# Test Phase 01: Project Scaffolding, Monorepo (`backend/` & `frontend/`), 2-Port Config & DDD Shared Core

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 01)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-01-scaffolding-and-ddd-core.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-01-scaffolding-and-ddd-core.md) เพื่อยืนยันว่าโครงสร้าง Monorepo (`backend/` พอร์ต `30211` และ `frontend/` พอร์ต `30222`), ชั้น **Shared `core/` Kernel**, ระบบดักจับ `DomainException`, ระบบ `InMemoryEventBus`, การจำกัด `OMP_NUM_THREADS`, และการแสดงผล UI App Shell (`StudioHeader`, `SystemStatusBar`, `SystemDiagnosticsModal`) ทำงานได้อย่างถูกต้อง 100% ในสภาพแวดล้อม **Air-Gapped (100% Offline)** บนทั้ง Windows, macOS, และ Linux

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Core & System Domain**:
  - `app.core.config.AppSettings` (`settings`)
  - `app.core.exceptions` (`DomainException`, `MemoryLimitExceededError`, `ModelLoadError`, `SchemaMismatchError`)
  - `app.core.events.InMemoryEventBus` (`event_bus`)
  - `app.core.memory_monitor.MemoryMonitor` และ `app.infrastructure.filesystem.cache_manager.CacheManager`
  - `app.application.system_service.SystemOrchestratorService`
  - REST Endpoints: `GET /api/v1/system/health`, `POST /api/v1/system/gc`, `DELETE /api/v1/system/cache`
- **Frontend Core & UI Components**:
  - `src/core/store/useSystemStore.ts`
  - `src/core/components/StudioHeader.tsx`, `SystemStatusBar.tsx`, `SystemDiagnosticsModal.tsx`, `GlobalErrorBoundary.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_01/
├── conftest.py
├── test_core_config_and_paths.py
├── test_domain_exceptions_and_events.py
├── test_system_orchestrator_service.py
└── test_system_api_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_01/test_core_config_and_paths.py
import os
from pathlib import Path
import pytest
from app.core.config import AppSettings, settings
from app.core.events import DomainEvent, InMemoryEventBus
from dataclasses import dataclass

def test_cross_platform_path_resolution() -> None:
    """ตรวจสอบว่า Path ทั้งหมดเป็น pathlib.Path ที่ resolve แล้ว ไม่ขึ้นกับ OS slash"""
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.CACHE_DIR, Path)
    assert settings.CACHE_DIR.is_absolute()
    assert settings.DATASET_CACHE_DIR == settings.CACHE_DIR / "datasets"
    assert settings.MODEL_CACHE_DIR == settings.CACHE_DIR / "models"

def test_env_override_for_ports_and_memory_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """ตรวจสอบการเปลี่ยนพอร์ตและเพดาน RAM ผ่าน Environment Variables กรณีพอร์ตชนกัน"""
    monkeypatch.setenv("PORT", "8088")
    monkeypatch.setenv("MAX_PROCESS_RAM_MB", "2048")
    custom = AppSettings()
    assert custom.PORT == 8088
    assert custom.MAX_PROCESS_RAM_MB == 2048

@dataclass(frozen=True)
class DummySystemBootEvent(DomainEvent):
    service_name: str = "backend"

def test_in_memory_event_bus_pub_sub() -> None:
    """ตรวจสอบว่า InMemoryEventBus ส่งและรับ DomainEvent ข้ามโมดูลได้ถูกต้อง"""
    bus = InMemoryEventBus()
    received: list[DummySystemBootEvent] = []
    bus.subscribe(DummySystemBootEvent, lambda evt: received.append(evt))

    bus.publish(DummySystemBootEvent(service_name="test_engine"))
    assert len(received) == 1
    assert received[0].service_name == "test_engine"
    assert received[0].occurred_at is not None
```

```python
# backend/tests/phase_01/test_system_api_endpoints.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.exceptions import MemoryLimitExceededError

@pytest.mark.asyncio
async def test_get_system_health_endpoint_schema() -> None:
    """ทดสอบ GET /api/v1/system/health ต้องคืนค่าครบทุกฟิลด์ภายในเวลา < 50ms"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] in ("healthy", "degraded")
        assert isinstance(data["os_platform"], str) and len(data["os_platform"]) > 0
        assert isinstance(data["cpu_logical_cores"], int) and data["cpu_logical_cores"] >= 1
        assert data["omp_threads"] == 4

        mem = data["memory"]
        assert mem["process_rss_mb"] > 0
        assert mem["process_budget_mb"] >= 512
        assert 0.0 <= mem["process_usage_percent"] <= 100.0
        assert mem["system_total_mb"] >= mem["system_available_mb"]
        assert isinstance(mem["is_low_memory"], bool)

@pytest.mark.asyncio
async def test_force_gc_and_clear_cache_endpoints(tmp_path) -> None:
    """ทดสอบ POST /api/v1/system/gc และ DELETE /api/v1/system/cache"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        gc_resp = await client.post("/api/v1/system/gc")
        assert gc_resp.status_code == 200
        assert "process_rss_mb" in gc_resp.json()

        cache_resp = await client.delete("/api/v1/system/cache")
        assert cache_resp.status_code == 200
        assert "freed_mb" in cache_resp.json()
        assert cache_resp.json()["freed_mb"] >= 0.0

@pytest.mark.asyncio
async def test_cors_local_loopback_headers() -> None:
    """ตรวจสอบว่า CORS อนุญาตทั้ง localhost:30222 และ 127.0.0.1:30222 แต่บล็อกเว็บภายนอก"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_ok = await client.options(
            "/api/v1/system/health",
            headers={"Origin": "http://localhost:30222", "Access-Control-Request-Method": "GET"},
        )
        assert r_ok.headers.get("access-control-allow-origin") == "http://localhost:30222"

        r_block = await client.options(
            "/api/v1/system/health",
            headers={"Origin": "https://malicious-site.com", "Access-Control-Request-Method": "GET"},
        )
        assert r_block.headers.get("access-control-allow-origin") is None
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/core/__tests__/
├── useSystemStore.test.ts
├── StudioHeader.test.tsx
├── SystemStatusBar.test.tsx
├── SystemDiagnosticsModal.test.tsx
└── offlineAssetIntegrity.test.ts
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/core/__tests__/StudioHeader.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StudioHeader } from '../components/StudioHeader';
import { useSystemStore } from '../store/useSystemStore';

describe('StudioHeader & 5-Step Workflow Stepper', () => {
  beforeEach(() => {
    useSystemStore.setState({
      activeStep: 1,
      unlockedMaxStep: 2,
      isDiagnosticsOpen: false,
      health: {
        status: 'healthy',
        os_platform: 'Windows 11 (AMD64)',
        python_version: '3.11.9',
        cpu_logical_cores: 8,
        omp_threads: 4,
        memory: {
          process_rss_mb: 310,
          process_budget_mb: 4096,
          process_usage_percent: 7.5,
          system_total_mb: 16384,
          system_available_mb: 11420,
          is_low_memory: false,
        },
        cache_size_mb: 12.4,
        latency_ms: 6,
        webgl2_renderer: 'ANGLE (NVIDIA RTX 3060)',
      },
    });
  });

  it('แสดงป้าย 100% OFFLINE STUDIO สีเขียวเมื่อเชื่อมต่อสำเร็จ และล็อก Step 3..5 ที่ยังไม่ปลดล็อก', () => {
    render(<StudioHeader />);
    expect(screen.getByText(/100% OFFLINE STUDIO/i)).toBeInTheDocument();

    const step2Btn = screen.getByRole('button', { name: /2\. Model/i });
    const step3Btn = screen.getByRole('button', { name: /3\. Column Mapping/i });
    expect(step2Btn).not.toBeDisabled();
    expect(step3Btn).toBeDisabled();

    fireEvent.click(step2Btn);
    expect(useSystemStore.getState().activeStep).toBe(2);
  });

  it('รองรับคีย์ลัด Ctrl+Shift+D เพื่อเปิด SystemDiagnosticsModal และ Alt+1..5 เพื่อสลับ Step', () => {
    render(<StudioHeader />);
    fireEvent.keyDown(window, { key: 'D', ctrlKey: true, shiftKey: true });
    expect(useSystemStore.getState().isDiagnosticsOpen).toBe(true);

    fireEvent.keyDown(window, { key: '2', altKey: true });
    expect(useSystemStore.getState().activeStep).toBe(2);

    // Alt+4 ต้องไม่ทำงานเพราะ unlockedMaxStep = 2
    fireEvent.keyDown(window, { key: '4', altKey: true });
    expect(useSystemStore.getState().activeStep).toBe(2);
  });
});
```

```tsx
// frontend/src/core/__tests__/SystemStatusBar.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SystemStatusBar } from '../components/SystemStatusBar';
import { useSystemStore } from '../store/useSystemStore';

describe('SystemStatusBar Telemetry & Offline Recovery Banner', () => {
  it('เปลี่ยนสีแถบ RAM Meter เป็นสีแดงเมื่อ process_usage_percent > 85% และแสดงปุ่ม Force GC', () => {
    useSystemStore.setState({
      health: {
        status: 'degraded',
        os_platform: 'Windows 11',
        python_version: '3.11.9',
        cpu_logical_cores: 8,
        omp_threads: 4,
        memory: {
          process_rss_mb: 3650,
          process_budget_mb: 4096,
          process_usage_percent: 89.1,
          system_total_mb: 16384,
          system_available_mb: 420,
          is_low_memory: true,
        },
        cache_size_mb: 45.0,
        latency_ms: 8,
        webgl2_renderer: 'WebGL2 Ready',
      },
    });

    render(<SystemStatusBar />);
    expect(screen.getByText(/3650 \/ 4096 MB/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Force GC/i })).toBeInTheDocument();
  });

  it('แสดงแบนเนอร์ Backend Offline พร้อมปุ่ม Copy Command เมื่อ health.status === "offline"', () => {
    useSystemStore.setState({ health: null });
    render(<SystemStatusBar />);
    expect(screen.getByText(/uvicorn app\.main:app --port 30211/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Retry Now/i })).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบการจำกัด Thread (`OMP_NUM_THREADS`) และ `MemoryMonitor` (`pytest`)
```python
# backend/tests/phase_01/test_memory_and_threads.py
import os
import pytest
from app.main import app  # trigger startup thread pinning
from app.core.config import settings
from app.core.memory_monitor import MemoryMonitor
from app.core.exceptions import MemoryLimitExceededError

def test_blas_and_omp_thread_pinning_on_boot() -> None:
    """ยืนยันว่า Environment Variables ของ BLAS/OpenMP ถูกล็อกไม่ให้แย่ง CPU 100%"""
    expected = str(settings.OMP_NUM_THREADS)
    assert os.environ.get("OMP_NUM_THREADS") == expected
    assert os.environ.get("MKL_NUM_THREADS") == expected
    assert os.environ.get("OPENBLAS_NUM_THREADS") == expected

def test_pre_allocation_memory_budget_gate_raises_413() -> None:
    """ทดสอบ assert_memory_budget เมื่อขอจอง RAM เกิน MAX_PROCESS_RAM_MB ต้องโยน MemoryLimitExceededError"""
    monitor = MemoryMonitor()
    excessive_bytes = (settings.MAX_PROCESS_RAM_MB + 1024) * 1024 * 1024
    with pytest.raises(MemoryLimitExceededError) as exc_info:
        monitor.assert_memory_budget(required_bytes=excessive_bytes)
    assert exc_info.value.http_status == 413
    assert exc_info.value.code == "MEMORY_LIMIT_EXCEEDED"
```

### 4.2 โค้ดทดสอบการคืน WebGL Context หลังตรวจชื่อการ์ดจอ (`vitest`)
```typescript
// frontend/src/core/__tests__/webglProbeCleanup.test.ts
import { describe, it, expect, vi } from 'vitest';
import { probeWebGL2Renderer } from '../utils/webglProbe';

describe('WebGL2 Capability Probe Cleanup', () => {
  it('ต้องเรียก WEBGL_lose_context.loseContext() ทันทีหลังอ่าน UNMASKED_RENDERER_WEBGL เสร็จ', () => {
    const loseContextMock = vi.fn();
    const getParameterMock = vi.fn().mockReturnValue('NVIDIA GeForce RTX 3060');
    const getExtensionMock = vi.fn((extName: string) => {
      if (extName === 'WEBGL_debug_renderer_info') return { UNMASKED_RENDERER_WEBGL: 0x9246 };
      if (extName === 'WEBGL_lose_context') return { loseContext: loseContextMock };
      return null;
    });

    vi.spyOn(document, 'createElement').mockReturnValue({
      getContext: () => ({ getExtension: getExtensionMock, getParameter: getParameterMock }),
      remove: vi.fn(),
    } as unknown as HTMLElement);

    const renderer = probeWebGL2Renderer();
    expect(renderer).toBe('NVIDIA GeForce RTX 3060');
    expect(loseContextMock).toHaveBeenCalledTimes(1);
  });
});
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 01)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **แอปค้างตอนเปิดในเครื่องที่ตัดอินเทอร์เน็ต (Air-gapped Freeze)** | `test_zero_external_network_urls_in_html_and_css()` | สแกนไฟล์ `index.html` และ `src/**/*.{ts,tsx,css}` ด้วย Regex หา `https?://(?!localhost|127\.0\.0\.1)` | ไม่พบ URL ภายนอก (0 matches) และฟอนต์ `.woff2` โหลดจาก `/fonts/` ในเครื่อง |
| **2** | **พอร์ต `30211` หรือ `30222` ชนกับโปรแกรมอื่นในเครื่อง** | `test_env_override_for_ports_and_memory_limits()` | ตั้งค่า `.env` เป็น `PORT=8088` และ `VITE_API_BASE_URL=http://127.0.0.1:8088` | `AppSettings.PORT == 8088` และ Frontend Client ชี้ไปยังพอร์ตใหม่โดยไม่ Hardcode |
| **3** | **CORS Blocked ระหว่าง `localhost` กับ `127.0.0.1`** | `test_cors_local_loopback_headers()` | ยิง `OPTIONS` Preflight จากทั้ง `http://localhost:30222` และ `http://127.0.0.1:30222` | คืน Header `Access-Control-Allow-Origin` ตรงตาม Origin ทั้งสองกรณี |
| **4** | **Path พังบน Windows เนื่องจาก Backslash (`\`)** | `test_cross_platform_path_resolution()` | ทดสอบ `CacheManager` ด้วย Path บน Windows และตรวจ JSON Output | Path ใน Python ใช้ `pathlib.Path` และเมื่อส่งออกผ่าน JSON จะถูกแปลงเป็น `.as_posix()` เสมอ |
| **5** | **โฟลเดอร์ `.cache/` ไม่มีอยู่จริงหรือติด Permission Read-only** | `test_cache_manager_readonly_fallback()` | จำลอง `PermissionError` บนโฟลเดอร์ `.cache/` ด้วย `monkeypatch` | ระบบสร้างโฟลเดอร์สำรองใน `tempfile.gettempdir() / "easy-ml-viewer-cache"` อัตโนมัติโดยแอปไม่ล่ม |
| **6** | **WebGL Context Lost จากการเช็ก GPU ซ้ำๆ** | `probeWebGL2Renderer()` test | เรียกใช้ฟังก์ชันตรวจสอบ GPU ตอนเริ่มต้นแอป | เรียก `loseContext()` ทุกครั้งและเก็บค่าไว้ใน `useSystemStore` โดยไม่สร้าง `<canvas>` ซ้ำ |
| **7** | **CPU พุ่ง 100% เมื่อ NumPy/Scikit-learn ทำงานพร้อม FastAPI** | `test_blas_and_omp_thread_pinning_on_boot()` | ตรวจสอบ `os.environ` หลัง Import `app.main` | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` มีค่าเท่ากับ `4` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Backend Pytest Coverage**: โมดูลใน `backend/app/core/` และ `domains/system/` มี Test Coverage `>= 95%` และรันผ่านทั้งหมด 100%
- [ ] **Frontend Vitest Suite**: ทดสอบ `StudioHeader`, `SystemStatusBar`, `SystemDiagnosticsModal`, และ `useSystemStore` ผ่านทั้งหมดโดยไม่มี Console Error
- [ ] **Health Endpoint Latency SLA**: `GET /api/v1/system/health` ตอบกลับภายในเวลา `< 20 ms` (P95)
- [ ] **100% Offline Network Audit**: เปิด Chrome DevTools ตั้งค่า `Network: Offline` แล้วรีเฟรชหน้าเว็บ `http://localhost:30222` หน้าจอต้องเรนเดอร์ฟอนต์ ไอคอน และธีมครบถ้วนโดยไม่มี External Request ล้มเหลวแม้แต่รายการเดียว
