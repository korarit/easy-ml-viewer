# Test Phase 15: `.mlv` Recording Format — MsgPack + Zstandard Baker, Streaming Writer & Instant Loader

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 15)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-15-mlv-file-format-baker-and-loader.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-15-mlv-file-format-baker-and-loader.md) เพื่อตรวจสอบความสมบูรณ์แบบ 100% (Bit-Exact / Float32 Round-Trip Fidelity) ของตัวแปลงไฟล์ไบนารี **`.mlv` (Machine Learning Video Package v1.0)** ที่ใช้ Magic Header `b"EMLV\x01\x00\x00\x00"`, `CRC32` Checksum, `msgpack`, และ `zstandard` รวมถึงการป้องกันไฟล์ปลอม, ไฟล์เสียหายระหว่างดาวน์โหลด, และการป้องกัน Decompression Bomb (`MAX_DECOMPRESSED_BYTES = 250 MB`)

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Core, Domain & API**:
  - `backend/app/core/binary_codec.py`: `MLV_MAGIC_HEADER`, `MAX_DECOMPRESSED_BYTES`, `MlvFormatError`, `encode_mlv_package()`, `decode_mlv_package()`, `_msgpack_default_encoder()`, `_msgpack_object_hook()`
  - `backend/app/domains/recording_mlv/entities.py`: `MlvHeaderSection`
  - `backend/app/presentation/api/v1/recording_router.py`: `POST /api/v1/recordings/{session_id}/bake`, `POST /api/v1/recordings/load`
- **Frontend Store & Components**:
  - `frontend/src/domains/recording/store/useRecordingStore.ts`
  - `frontend/src/domains/recording/components/BakeMlvModal.tsx`, `MlvInstantDropzone.tsx`, `MlvReplayModeBanner.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `backend/app/core/binary_codec.py`
ไฟล์ทดสอบ: `backend/tests/unit/core/test_binary_codec.py`

```python
import numpy as np
import pytest
from app.core.binary_codec import (
    MLV_MAGIC_HEADER,
    MlvFormatError,
    decode_mlv_package,
    encode_mlv_package,
)


def test_encode_and_decode_mlv_package_roundtrip_with_numpy_and_thai_utf8():
    """ตรวจสอบการแพ็กและแตกไฟล์ .mlv ว่าคงค่า NumPy ndarray, scalar, และข้อความภาษาไทยได้ตรงกัน 100%"""
    original_payload = {
        "header": {
            "schema_version": "1.0",
            "title": "ทดสอบการบันทึกโมเดลสินเชื่อ Q3",
            "total_frames": np.int64(1000),
            "is_classification": np.bool_(True),
        },
        "static_world": {
            "z_heights_world": np.linspace(-3.0, 3.0, 3600, dtype=np.float64),
        },
        "frames_track": {
            "positions_world": np.random.randn(1000, 3).astype(np.float32),
            "predictions": np.random.uniform(0, 1, size=1000).astype(np.float32),
        },
    }

    mlv_bytes = encode_mlv_package(original_payload, compression_level=3)
    assert mlv_bytes[:8] == MLV_MAGIC_HEADER

    decoded = decode_mlv_package(mlv_bytes)
    assert decoded["header"]["title"] == "ทดสอบการบันทึกโมเดลสินเชื่อ Q3"
    assert decoded["header"]["total_frames"] == 1000
    assert decoded["header"]["is_classification"] is True

    # ตรวจสอบว่า ndarray ถูกแปลงเป็น float32 ที่แก้ไขค่าได้ (WRITEABLE=True)
    pos_arr = decoded["frames_track"]["positions_world"]
    assert isinstance(pos_arr, np.ndarray)
    assert pos_arr.dtype == np.float32
    assert pos_arr.shape == (1000, 3)
    assert pos_arr.flags["WRITEABLE"] is True
    assert np.allclose(pos_arr, original_payload["frames_track"]["positions_world"], atol=1e-6)


def test_decode_mlv_package_rejects_invalid_magic_header():
    """Edge Case #3: เมื่อผู้ใช้อัปโหลดไฟล์ CSV หรือ PKL ที่เปลี่ยนนามสกุลเป็น .mlv ต้องโยน MlvFormatError"""
    fake_file = b"col1,col2,col3\n1,2,3\n4,5,6\n"
    with pytest.raises(MlvFormatError, match="Invalid Magic Header"):
        decode_mlv_package(fake_file)


def test_decode_mlv_package_detects_corrupted_crc32_checksum():
    """Edge Case #4: เมื่อไบต์ภายในไฟล์ .mlv เสียหายแม้แต่ 1 ไบต์ การตรวจ CRC32 ต้องจับได้ทันที"""
    valid_bytes = bytearray(encode_mlv_package({"test": np.arange(50, dtype=np.float32)}))
    # ดัดแปลงไบต์สุดท้ายของ Payload ให้เพี้ยนไป 1 บิต
    valid_bytes[-1] ^= 0xFF

    with pytest.raises(MlvFormatError, match="CRC32 Mismatch"):
        decode_mlv_package(bytes(valid_bytes))


def test_encode_mlv_handles_non_contiguous_sliced_numpy_arrays():
    """Edge Case #6: อาเรย์ที่ผ่านการ Transpose (.T) หรือ Slice ข้ามแถว ต้องถูกแปลงด้วย ascontiguousarray อย่างถูกต้อง"""
    mat = np.arange(100, dtype=np.float32).reshape(10, 10)
    non_contiguous_slice = mat.T[:, ::2]  # Non-C-contiguous
    assert not non_contiguous_slice.flags["C_CONTIGUOUS"]

    encoded = encode_mlv_package({"sliced": non_contiguous_slice})
    decoded = decode_mlv_package(encoded)
    assert np.array_equal(decoded["sliced"], non_contiguous_slice)
```

### 2.2 API Integration Tests สำหรับ Bake & Instant Load
ไฟล์ทดสอบ: `backend/tests/integration/api/test_recording_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_bake_and_instant_load_mlv_workflow(async_client: AsyncClient, seeded_session_id: str):
    """ทดสอบวงจรครบจบ: สั่ง Bake .mlv จาก Session ปัจจุบัน แล้วนำไฟล์ที่ได้ไป Load เปิดเป็น Replay Session ใหม่"""
    bake_resp = await async_client.post(
        f"/api/v1/recordings/{seeded_session_id}/bake",
        json={"title": "Audit Replay Test", "notes": "Note 1", "include_explainability": True},
    )
    assert bake_resp.status_code == 200
    mlv_binary = bake_resp.content
    assert mlv_binary.startswith(b"EMLV\x01\x00\x00\x00")

    # นำ Binary ที่ได้ไปอัปโหลดเข้า POST /api/v1/recordings/load
    files = {"file": ("audit_replay.mlv", mlv_binary, "application/octet-stream")}
    load_resp = await async_client.post("/api/v1/recordings/load", files=files)
    assert load_resp.status_code == 200

    loaded_data = load_resp.json()
    assert loaded_data["is_prebaked_replay"] is True
    assert loaded_data["header"]["title"] == "Audit Replay Test"
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Component Tests สำหรับ `MlvReplayModeBanner.tsx` & `BakeMlvModal.tsx`
ไฟล์ทดสอบ: `frontend/src/domains/recording/__tests__/MlvReplayModeBanner.test.tsx`

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MlvReplayModeBanner } from '../components/MlvReplayModeBanner';

describe('Phase 15: MlvReplayModeBanner & Pre-baked Replay UX', () => {
  it('แสดงป้าย REPLAY MODE พร้อมชื่อไฟล์ จำนวนเฟรม และปุ่มบันทึกทับเมื่อเพิ่ม Bookmark ใหม่', () => {
    render(
      <MlvReplayModeBanner
        fileName="credit_audit_q3.mlv"
        schemaVersion="1.0"
        totalFrames={2500}
        hasUnsavedBookmarkChanges={true}
      />
    );

    expect(screen.getByText(/REPLAY MODE/i)).toBeInTheDocument();
    expect(screen.getByText(/credit_audit_q3\.mlv/i)).toBeInTheDocument();
    expect(screen.getByText(/2,500 Frames/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Update & Re-bake \.mlv/i })).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การทดสอบขนาดไฟล์บีบอัด ความเร็ว Decompress (`< 30ms`) และ Decompression Bomb Guard (`250 MB`)
```python
import time
import numpy as np
import pytest
import zstandard as zstd
from app.core.binary_codec import (
    MAX_DECOMPRESSED_BYTES,
    decode_mlv_package,
    encode_mlv_package,
)


def test_20k_frames_compression_ratio_and_decompression_speed():
    """ตรวจสอบว่าข้อมูล 20,000 เฟรมพร้อมพื้นผิว 3D บีบอัดเหลือไม่เกิน 5 MB และแตกไฟล์เสร็จใน < 30ms"""
    payload_20k = {
        "header": {"schema_version": "1.0", "total_frames": 20000},
        "static_world": {"z_heights": np.zeros(3600, dtype=np.float32)},
        "frames_track": {
            "positions": np.linspace(-5.0, 5.0, 20000 * 3, dtype=np.float32).reshape(20000, 3),
            "predictions": np.sin(np.linspace(0, 50, 20000, dtype=np.float32)),
            "top_contribs": np.zeros((20000, 10), dtype=np.float32),
        },
    }

    compressed_bytes = encode_mlv_package(payload_20k, compression_level=3)
    size_mb = len(compressed_bytes) / (1024 * 1024)
    assert size_mb < 5.0

    t0 = time.perf_counter()
    _ = decode_mlv_package(compressed_bytes)
    decompress_ms = (time.perf_counter() - t0) * 1000.0
    assert decompress_ms < 30.0


def test_zstd_decompression_bomb_guard_blocks_oversized_payload(monkeypatch):
    """ตรวจสอบว่าหากไฟล์ .mlv ขยายตัวเกิน MAX_DECOMPRESSED_BYTES จะถูกบล็อกทันทีก่อนที่ RAM จะเต็ม"""
    import app.core.binary_codec as codec_mod

    # จำลองการตั้งเพดานไว้ที่ 1 KB เพื่อทดสอบการบล็อกเมื่อไฟล์ขยายเกินเพดาน
    monkeypatch.setattr(codec_mod, "MAX_DECOMPRESSED_BYTES", 1024)
    large_payload = {"data": np.zeros(5000, dtype=np.float32)}  # ~20 KB หลังแตกไฟล์
    packed = encode_mlv_package(large_payload)

    with pytest.raises(zstd.ZstdError):
        decode_mlv_package(packed)
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **`msgpack` เจอชนิดข้อมูล `np.float32`, `np.int64`, `np.bool_`** | ส่ง Dictionary ที่มี NumPy scalars และ `ndarray` เข้า `encode_mlv_package` | แพ็กสำเร็จผ่าน `_msgpack_default_encoder` โดยไม่โยน `TypeError` |
| 2 | **อาเรย์จาก `np.frombuffer` เป็น Read-Only (`WRITEABLE=False`)** | แตกไฟล์ด้วย `decode_mlv_package` แล้วลองแก้ไขค่า `arr[0] = 1.0` | แก้ไขค่าได้สำเร็จเพราะ `_msgpack_object_hook` เรียก `.copy()` เสมอ |
| 3 | **อัปโหลดไฟล์ `.csv` หรือไฟล์ที่มีขนาดสั้นกว่า 12 ไบต์** | ส่ง `b"short"` และ `b"col1,col2\n"` เข้า `decode_mlv_package` | โยน `MlvFormatError` (`Truncated File` / `Invalid Magic Header`) |
| 4 | **ไฟล์ `.mlv` เสียหายระหว่างคัดลอก (Bit Flip Corruption)** | สลับบิตใน `compressed_payload` 1 ไบต์ | โยน `MlvFormatError` (`CRC32 Mismatch`) ทันที |
| 5 | **เพิ่ม Bookmark ใหม่ขณะอยู่ในโหมด Replay (`.mlv`)** | เพิ่ม Bookmark ในโหมด Replay แล้วกด `Update & Re-bake .mlv` | บันทึกไฟล์ `.mlv` ใหม่พร้อม Bookmark ล่าสุดได้สำเร็จโดยไม่ต้องใช้ไฟล์โมเดล |
| 6 | **Non-Contiguous NumPy Array (`arr.T[:, ::2]`)** | ส่ง Sliced Transpose Array เข้า `encode_mlv_package` | `np.ascontiguousarray` จัดเรียงไบต์ใหม่ให้ถูกต้อง ค่าไม่เพี้ยนหลังแตกไฟล์ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Decompression Speed SLA**: การแตกไฟล์ `.mlv` ขนาด 20,000 เฟรม ต้องใช้เวลา `< 30 ms`
- [ ] **Compression Ratio SLA**: ขนาดไฟล์ `.mlv` สำหรับ 20,000 เฟรมพร้อมพื้นผิว 3D Standard LOD ต้อง `< 5 MB`
- [ ] **Checksum & Bomb Security**: ต้องตรวจจับไฟล์ที่บิตเพี้ยนด้วย `CRC32` ได้ 100% และจำกัดขนาดแตกไฟล์สูงสุดที่ `250 MB`
- [ ] **Test Coverage**: Unit Test Coverage ของ `binary_codec.py` ต้องเท่ากับ $100\%$
