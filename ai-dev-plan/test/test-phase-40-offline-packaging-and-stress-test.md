# Test Plan — Phase 40: Cross-Platform Offline Launchers, 20-Model Demo Suite, Multi-Recording Compare & Hardening

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 40)
เอกสารนี้กำหนดแผนการทดสอบขั้นสุดท้ายแบบ 1-ต่อ-1 สำหรับ **[phase-40-offline-packaging-and-stress-test.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-40-offline-packaging-and-stress-test.md)** เพื่อตรวจรับความพร้อมระดับ Production ทั้งระบบ ได้แก่ สคริปต์สร้างโมเดลตัวอย่าง **`generate_all_20_demos.py`**, บริการเปรียบเทียบ 2 โมเดล **`DualRecordingCompareService` (`backend/app/domains/recording/application/dual_compare_service.py`)**, คอมโพเนนต์หน้าจอคู่ **`DualStageCompareWorkspace`**, สคริปต์เปิดโปรแกรมข้ามแพลตฟอร์ม (`start-windows.bat`, `start-windows.ps1`, `start-unix.sh`), และชุดทดสอบความทนทาน **100,000-Row Stress Benchmark & 100% Air-Gapped Offline Audit**

### ขอบเขตการตรวจสอบหลัก
1. **20-Model Demo Suite Completeness**: ยืนยันว่าสคริปต์ `generate_all_20_demos.py` สร้างไฟล์โมเดลครบทั้ง 20+ ตระกูล พร้อมไฟล์ `.csv` และ `.mlv` ที่เปิดเล่นได้ทันที
2. **Synchronized Dual `.mlv` Compare & Disagreement Detection**: ยืนยันความถูกต้องของการคำนวณ `disagreement_indices` ($D_i = \mathbb{I}(\hat{y}^{(A)}_i \ne \hat{y}^{(B)}_i)$) และ `probability_divergence` ($\Delta P_i$) รวมถึงการซิงค์มุมกล้อง 3D และแถบเวลา VCR
3. **Cross-Platform Process Lifecycle & Graceful Port Cleanup**: ยืนยันการปิดโปรเซสลูกทั้งพอร์ต `30211` และ `30222` โดยไม่เหลือ Zombie Process และการจัดการขนาด `.cache/` (`<= 1.0 GB`)
4. **100,000-Row Stress Benchmark & 100% Air-Gapped Offline Audit**: ยืนยันเพดานหน่วยความจำ (`<= 600 MB` RSS), ความเร็วการเล่น 60 FPS, และการไม่มี Network Request ออกนอก `127.0.0.1`

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow / System Parity Tests (พร้อมโค้ดต้นแบบ `pytest`)

ไฟล์ทดสอบหลัก: `backend/tests/stress/test_phase40_compare_and_100k_benchmark.py`

```python
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import pytest
from app.domains.recording.application.dual_compare_service import DualRecordingCompareService
from app.domains.recording.domain.entities import MLVRecordingPayload


def test_dual_recording_compare_disagreement_and_divergence_math() -> None:
    """ยืนยันความแม่นยำของสมการตรวจจับความขัดแย้ง D_i และ Total Variation Divergence Delta P_i"""
    rec_a = MLVRecordingPayload(
        model_name="RandomForest.mlv",
        task_type="classification",
        predicted_classes=np.array([0, 1, 0, 1, 1], dtype=np.uint16),
        probabilities=np.array([
            [0.80, 0.20],
            [0.30, 0.70],
            [0.55, 0.45],  # Frame 2: Model A says Class 0
            [0.10, 0.90],
            [0.48, 0.52],  # Frame 4: Model A says Class 1
        ], dtype=np.float32),
    )
    rec_b = MLVRecordingPayload(
        model_name="CatBoost.mlv",
        task_type="classification",
        predicted_classes=np.array([0, 1, 1, 1, 0], dtype=np.uint16),
        probabilities=np.array([
            [0.85, 0.15],
            [0.25, 0.75],
            [0.20, 0.80],  # Frame 2: Model B says Class 1 -> Disagreement!
            [0.10, 0.90],
            [0.70, 0.30],  # Frame 4: Model B says Class 0 -> Disagreement!
        ], dtype=np.float32),
    )

    service = DualRecordingCompareService()
    summary = service.compare_recordings(rec_a, rec_b)

    assert summary.total_aligned_frames == 5
    np.testing.assert_array_equal(summary.disagreement_indices, np.array([2, 4], dtype=np.uint32))
    assert summary.agreement_rate == pytest.approx(3 / 5)
    # Frame 2 TV Divergence = 0.5 * (|0.55 - 0.20| + |0.45 - 0.80|) = 0.35
    assert summary.probability_divergence[2] == pytest.approx(0.35, abs=1e-5)


def test_100k_rows_ingestion_and_chunked_simulation_memory_budget(tmp_path: Path) -> None:
    """Stress Test: ทดสอบข้อมูล 100,000 แถว x 25 คอลัมน์ ยืนยันว่าระบบใช้ RAM ไม่เกินเพดานและทำงานเสร็จตาม SLA"""
    proc = psutil.Process()
    gc.collect()
    baseline_rss_mb = proc.memory_info().rss / (1024 * 1024)

    rng = np.random.default_rng(2026)
    n_rows, n_cols = 100_000, 25
    data = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    df_100k = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_cols)])

    parquet_cache = tmp_path / "stress_100k.parquet"
    t0 = time.perf_counter()
    df_100k.to_parquet(parquet_cache, engine="pyarrow")
    elapsed_write = time.perf_counter() - t0

    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)
    ram_delta_mb = peak_rss_mb - baseline_rss_mb

    assert parquet_cache.exists()
    assert elapsed_write < 2.5
    assert ram_delta_mb <= 250.0, f"RAM Delta exceeded 250 MB budget: {ram_delta_mb:.1f} MB"
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & UX Flow Tests (พร้อมโค้ดต้นแบบ `vitest`)

ไฟล์ทดสอบหลัก: `frontend/src/domains/simulation/components/compare/__tests__/DualStageCompareWorkspace.test.tsx`

```tsx
import { describe, it, expect } from "vitest";
import { useDualCompareStore } from "../../../../recording/store/useDualCompareStore";

describe("Phase 40 — Synchronized Dual .mlv Compare & Disagreement Navigation", () => {
  it("ซิงค์ตำแหน่งเฟรม VCR ของทั้งสองโมเดลและกระโดดไปยังเฟรมที่ขัดแย้งถัดไป (Jump to Next Conflict) ได้ถูกต้อง", () => {
    const store = useDualCompareStore.getState();
    store.setDisagreementIndices(new Uint32Array([14, 42, 108, 255]));
    store.setCurrentFrame(10);

    store.jumpToNextDisagreement();
    expect(useDualCompareStore.getState().currentFrame).toBe(14);

    store.jumpToNextDisagreement();
    expect(useDualCompareStore.getState().currentFrame).toBe(42);

    store.jumpToPrevDisagreement();
    expect(useDualCompareStore.getState().currentFrame).toBe(14);
  });

  it("ป้องกัน Infinite Camera Sync Loop ด้วยการตรวจสอบ activeControlSource", () => {
    const store = useDualCompareStore.getState();
    store.setCameraSyncEnabled(true);
    store.syncCameraFromSource("A", { position: [5, 5, 5], target: [0, 0, 0] });

    const state = useDualCompareStore.getState();
    expect(state.lastCameraSource).toBe("A");
    expect(state.sharedCameraPose.position).toEqual([5, 5, 5]);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Process Lifecycle & GPU/CPU Resource Tests)

1. **Cross-Platform Launcher & Zombie Process Termination Test**:
   - ทดสอบเปิดโปรเซสจำลองผ่านสคริปต์ Launcher แล้วส่งสัญญาณ `SIGINT` / `SIGTERM` ยืนยันว่าพอร์ต `30211` และ `30222` ถูกคืนให้ระบบปฏิบัติการภายใน `< 2.0 วินาที` (`socket.connect_ex(('127.0.0.1', 30211)) != 0`)
2. **Single WebGL Context Scissor View Test**:
   - ยืนยันว่าในโหมด `DualStageCompareWorkspace` การแสดงผล 3D ซ้ายและขวาใช้ `<View>` บน WebGL Context เดียวกัน หรือมีการเรียก `.dispose()` คืน VRAM ทันทีเมื่อออกจากโหมดเปรียบเทียบ
3. **100% Air-Gapped Offline Static & Runtime Audit**:
   - สแกนซอร์สโค้ดทั้งหมดใน `frontend/index.html`, `frontend/src/`, และ CSS ยืนยันว่าไม่มี URL ที่ชี้ไปยัง `https://fonts.googleapis.com`, `https://cdn.`, หรือ External Domain ใดๆ นอกเหนือจาก `127.0.0.1` / `localhost`

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| กรณีปัญหา (Edge Case จาก Dev Phase 40) | วิธีจำลองในการทดสอบ (Test Simulation Setup) | ผลลัพธ์ที่ต้องผ่านการตรวจสอบ (Expected Verification Assertion) |
| :--- | :--- | :--- |
| **1. Zombie Process ค้างพอร์ต `30211` หรือ `30222`** | จำลองโปรเซสยึดพอร์ตค้างไว้ก่อนรันสคริปต์ และจำลองการส่งสัญญาณปิดเทอร์มินัล | สคริปต์ตรวจพบพอร์ตซ้ำและทำความสะอาด Process Tree ทั้งหมดเมื่อจบการทำงาน |
| **2. เปรียบเทียบ 2 ไฟล์ `.mlv` ที่มีจำนวนแถวไม่เท่ากัน** | ส่ง `rec_a` (500 เฟรม) และ `rec_b` (1,200 เฟรม) เข้า `compare_recordings()` | ระบบตัดเทียบเฉพาะ `total_aligned_frames = 500` ($N_{\min}$) และตั้งค่า `frame_count_mismatch = True` เพื่อแจ้งเตือนบน UI |
| **3. การซิงค์มุมกล้อง 3D เกิด Infinite Loop** | ยิง Event `onChange` จาก `OrbitControls` ฝั่งซ้ายและขวาต่อเนื่องกัน | ตัวแปร `activeControlSourceRef` บล็อกการสะท้อนกลับของ Event เดิม ไม่เกิด `Maximum update depth exceeded` |
| **4. โฟลเดอร์ `.cache/` มีขนาดเกินโควตา `1.0 GB`** | สร้างไฟล์ `.parquet` จำลองใน `.cache/` ให้ขนาดรวมเกิน `1.05 GB` แล้วเรียก `CacheCleanupService` | ระบบลบไฟล์ที่เก่าที่สุดแบบ LRU จนขนาดรวมของ `.cache/` ลดลงต่ำกว่าเพดานที่กำหนด หรือเคลียร์ทั้งหมดเมื่อกดปุ่ม `Clear Temp Cache` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **20-Model Instant Demo Suite**: สคริปต์ `generate_all_20_demos.py` สร้างไฟล์ครบทั้ง 20+ ตระกูลและโหลดผ่าน `ModelRegistry` ได้สำเร็จ 100%
- [ ] **100,000-Row Stress Benchmark**: ประมวลผลไฟล์ 100,000 แถวโดยใช้ Peak RAM Delta $\le 250\text{ MB}$ (RSS รวม $\le 600\text{ MB}$) และเล่น VCR ที่ 60 FPS
- [ ] **Dual `.mlv` Compare Accuracy**: คำนวณจุดขัดแย้งระหว่าง 2 โมเดลและซิงค์ฉาก 3D ซ้าย-ขวาได้แม่นยำระดับเฟรมต่อเฟรม
- [ ] **Zero External Network Calls (100% Offline)**: ผ่านการตรวจสอบทั้ง Static Code Scan และ Runtime Offline Mode โดยไม่มี Request ออกนอก `127.0.0.1`
