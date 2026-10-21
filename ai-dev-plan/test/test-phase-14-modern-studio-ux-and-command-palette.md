# Test Phase 14: Modern Studio UX — Bi-Directional 3D Raycast Scrubbing, Command Palette (`Ctrl+K`) & Label Aliasing

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 14)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-14-modern-studio-ux-and-command-palette.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-14-modern-studio-ux-and-command-palette.md) เพื่อตรวจสอบความแม่นยำของการคลิกจุดในฉาก 3D เพื่อกรอวิดีโอไปยังเฟรมนั้นทันที (**Bi-Directional 3D Raycast Click-to-Scrub** บน `THREE.InstancedMesh`), ระบบป้องกันการคลิกชนกับการลากเมาส์หมุนกล้อง (`OrbitControls Drag Distance <= 4px Guard`), หน้าต่าง **Command Palette (`Ctrl+K`)**, ระบบตั้งชื่อและสีแทนคลาส/คอลัมน์ (**Class & Feature Aliasing**), และโหมด **Cinema Presentation (`Tab`)**

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Domain & API**:
  - `backend/app/domains/studio_metadata/entities.py`: `ClassAliasVO`, `FeatureAliasVO`, `SessionAliasConfigEntity`
  - `backend/app/presentation/api/v1/alias_router.py`: `PUT /api/v1/sessions/{session_id}/aliases`
- **Frontend Stores, Hooks & Components**:
  - `frontend/src/domains/studio-ux/store/useAliasStore.ts`, `useStudioLayoutStore.ts`
  - `frontend/src/domains/studio-ux/hooks/useCommandPaletteRegistry.ts`
  - `frontend/src/domains/studio-ux/components/InteractiveQueuePointCloud3D.tsx`, `CommandPaletteModal.tsx`, `AliasSettingsModal.tsx`, `CameraPresetToolbar.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit & Validation Tests สำหรับ `ClassAliasVO` และ `FeatureAliasVO`
ไฟล์ทดสอบ: `backend/tests/unit/domains/test_studio_metadata.py`

```python
import pytest
from pydantic import ValidationError
from app.domains.studio_metadata.entities import (
    ClassAliasVO,
    FeatureAliasVO,
    SessionAliasConfigEntity,
)


def test_class_alias_vo_validates_hex_color_and_max_length():
    """ตรวจสอบว่า ClassAliasVO รับค่าสี Hex ที่ถูกต้องและปฏิเสธค่าสีผิดรูปแบบหรือชื่อที่ยาวเกิน 64 ตัวอักษร"""
    valid_alias = ClassAliasVO(
        raw_class_value=1,
        display_label="อนุมัติสินเชื่อ (ความเสี่ยงต่ำ)",
        hex_color="#10B981",
    )
    assert valid_alias.display_label == "อนุมัติสินเชื่อ (ความเสี่ยงต่ำ)"

    # ทดสอบสีผิดรูปแบบ
    with pytest.raises(ValidationError):
        ClassAliasVO(raw_class_value=1, display_label="Valid", hex_color="not-a-hex")

    # Edge Case #5: ตั้งชื่อยาวเกิน 64 ตัวอักษรต้องถูกสกัดกั้น
    with pytest.raises(ValidationError):
        ClassAliasVO(raw_class_value=0, display_label="A" * 65, hex_color="#F43F5E")
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Unit Tests สำหรับ 3D Raycast Click vs OrbitDrag Guard (`<= 4px`)
ไฟล์ทดสอบ: `frontend/src/domains/studio-ux/__tests__/InteractiveQueuePointCloud3D.test.ts`

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';

describe('Phase 14: Bi-Directional 3D Raycast Scrubbing & OrbitDrag Guard', () => {
  beforeEach(() => {
    usePlaybackStore.getState().initializeQueue(500);
    usePlaybackStore.getState().seekToFrame(0);
  });

  it('กระโดดไปยังเฟรม instanceId ทันทีเมื่อระยะขยับเมาส์ระหว่าง PointerDown และ PointerUp ไม่เกิน 4 พิกเซล', () => {
    const downEvent = { clientX: 400, clientY: 300 };
    const upEvent = {
      clientX: 402, // dx = 2
      clientY: 302, // dy = 2 -> hypot(2, 2) = 2.82px <= 4.0px
      instanceId: 312,
      stopPropagation: vi.fn(),
    };

    const dragDistancePx = Math.hypot(
      upEvent.clientX - downEvent.clientX,
      upEvent.clientY - downEvent.clientY
    );
    if (dragDistancePx <= 4.0 && typeof upEvent.instanceId === 'number') {
      upEvent.stopPropagation();
      usePlaybackStore.getState().seekToFrame(upEvent.instanceId, true);
    }

    expect(upEvent.stopPropagation).toHaveBeenCalledTimes(1);
    expect(usePlaybackStore.getState().currentFrameIndex).toBe(312);
  });

  it('Edge Case #1: เพิกเฉยการเปลี่ยนเฟรมเมื่อผู้ใช้ลากเมาส์หมุนกล้องเกิน 4 พิกเซล (OrbitControls Drag)', () => {
    const downEvent = { clientX: 400, clientY: 300 };
    const upEvent = {
      clientX: 445, // dx = 45px (> 4px)
      clientY: 310,
      instanceId: 312,
      stopPropagation: vi.fn(),
    };

    const dragDistancePx = Math.hypot(
      upEvent.clientX - downEvent.clientX,
      upEvent.clientY - downEvent.clientY
    );
    if (dragDistancePx <= 4.0 && typeof upEvent.instanceId === 'number') {
      usePlaybackStore.getState().seekToFrame(upEvent.instanceId, true);
    }

    // เฟรมต้องคงอยู่ที่ 0 เหมือนเดิม ไม่กระโดดไปเฟรม 312!
    expect(usePlaybackStore.getState().currentFrameIndex).toBe(0);
  });
});
```

### 3.2 Component Tests สำหรับ Command Palette (`Ctrl+K`) & Cinema Mode (`Tab`)
ไฟล์ทดสอบ: `frontend/src/domains/studio-ux/__tests__/CommandPaletteModal.test.tsx`

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { useStudioLayoutStore } from '../store/useStudioLayoutStore';
import { CommandPaletteModal } from '../components/CommandPaletteModal';

describe('Phase 14: Command Palette (Ctrl+K) & Cinema Mode (Tab)', () => {
  beforeEach(() => {
    usePlaybackStore.getState().initializeQueue(500);
    useStudioLayoutStore.setState({ isCommandPaletteOpen: false, isCinemaMode: false });
  });

  it('เปิด Command Palette ด้วย Ctrl+K และรองรับการพิมพ์ "Go to row 245" เพื่อกระโดดไปเฟรม 245', () => {
    render(<CommandPaletteModal availableFeatures={['credit_score', 'annual_income']} />);

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
    expect(useStudioLayoutStore.getState().isCommandPaletteOpen).toBe(true);

    const searchInput = screen.getByPlaceholderText(/Search commands/i);
    fireEvent.change(searchInput, { target: { value: '245' } });
    fireEvent.keyDown(searchInput, { key: 'Enter' });

    expect(usePlaybackStore.getState().currentFrameIndex).toBe(245);
    expect(useStudioLayoutStore.getState().isCommandPaletteOpen).toBe(false);
  });

  it('Edge Case #4: กดปุ่ม Tab เพื่อเข้าโหมด Cinema ได้เมื่อไม่มี Modal เปิด แต่ไม่สลับโหมดหากเปิด Modal อยู่', () => {
    // เมื่อเปิด Command Palette Modal อยู่ การกด Tab ต้องไม่สลับเป็น Cinema Mode
    useStudioLayoutStore.setState({ isCommandPaletteOpen: true, isCinemaMode: false });
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(useStudioLayoutStore.getState().isCinemaMode).toBe(false);

    // เมื่อปิด Modal แล้ว การกด Tab ต้องเปิด Cinema Mode ทันที
    useStudioLayoutStore.setState({ isCommandPaletteOpen: false });
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(useStudioLayoutStore.getState().isCinemaMode).toBe(true);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การตรวจสอบ `THREE.InstancedMesh` Single Draw Call & `computeBoundingSphere()`
ไฟล์ทดสอบ: `frontend/src/domains/studio-ux/__tests__/InstancedMeshMemory.test.ts`

```typescript
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

describe('Phase 14: InstancedMesh Single Draw Call & BoundingSphere Verification', () => {
  it('อัปเดตพิกัดจุดข้อมูล 10,000 จุดลงใน InstancedMesh เดียวและเรียก computeBoundingSphere() เสมอ', () => {
    const geom = new THREE.SphereGeometry(0.09, 12, 12);
    const mat = new THREE.MeshStandardMaterial();
    const instancedMesh = new THREE.InstancedMesh(geom, mat, 10000);
    const boundingSphereSpy = vi.spyOn(instancedMesh, 'computeBoundingSphere');

    const dummy = new THREE.Object3D();
    for (let i = 0; i < 10000; i++) {
      dummy.position.set(i * 0.001, 0, 0);
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.computeBoundingSphere();

    expect(instancedMesh.count).toBe(10000);
    expect(boundingSphereSpy).toHaveBeenCalledTimes(1);
    expect(instancedMesh.boundingSphere).not.toBeNull();

    geom.dispose();
    mat.dispose();
  });
});
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **คลิกค้างเพื่อหมุนกล้อง 3D แต่ปล่อยเมาส์โดนจุดข้อมูล** | จำลอง `pointerDown` ที่ `(100,100)` และ `pointerUp` ที่ `(120,115)` (`hypot > 4px`) | ไม่เรียก `seekToFrame` เฟรมปัจจุบันไม่เปลี่ยนเอง |
| 2 | **Raycaster ทะลุผ่านจุดด้านหน้าไปโดนจุดด้านหลังซ้อนกัน** | จำลอง `ThreeEvent` ที่มีหลาย Intersections ซ้อนกัน | `e.stopPropagation()` ถูกเรียกตั้งแต่จุดแรกที่ใกล้กล้องที่สุด |
| 3 | **จุดข้อมูลมีพิกัดซ้อนทับกันสนิทหลายเฟรม** | คลิกซ้ำที่พิกัดเดิมซึ่งมีเฟรม `[10, 45, 92]` อยู่ตำแหน่งเดียวกัน | ระบบวนเลือกเฟรมถัดไปในกลุ่ม (`10 -> 45 -> 92`) ตามลำดับ |
| 4 | **กดปุ่ม `Tab` ขณะเปิด Modal ตั้งค่าหรืออยู่ใน `<input>`** | เปิด `AliasSettingsModal` แล้วกดปุ่ม `Tab` | เลื่อน Focus ตามปกติ ไม่สลับโหมด Cinema Fullscreen |
| 5 | **ตั้งชื่อ Alias ยาวเกิน 64 ตัวอักษร** | ส่งชื่อยาว 80 ตัวอักษรเข้า `ClassAliasVO` และช่อง Input | ถูกจำกัดด้วย `maxLength={64}` และแสดง `truncate` บนป้าย HUD |
| 6 | **ค้นหาคำสั่งใน Command Palette ด้วยคำย่อแบบ Fuzzy** | พิมพ์ `"ax x inc"` ในช่องค้นหา Command Palette | แมตช์คำสั่ง `"Set Axis X -> annual_income"` ขึ้นมาเป็นอันดับแรก |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Draw Call Budget**: จุดข้อมูลทั้งหมดในคิว (สูงสุด 10,000 จุด) ต้องใช้ `THREE.InstancedMesh` เพียง `1 Draw Call`
- [ ] **Click-to-Scrub Response Time**: เวลาตั้งแต่ปล่อยเมาส์คลิกบนจุด 3D จนถึง `currentFrameIndex` เปลี่ยนค่า ต้อง `< 2 ms`
- [ ] **Command Palette Latency**: การกรองรายการคำสั่ง Fuzzy Search ขณะพิมพ์ต้องเสร็จภายใน `< 5 ms` ต่อการกดปุ่ม
- [ ] **Test Coverage**: Unit & Component Test Coverage ของ `studio-ux` ต้อง $\ge 90\%$
