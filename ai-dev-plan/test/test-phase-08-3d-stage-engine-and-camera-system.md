# Test Phase 08: 3D Stage Core — React Three Fiber Setup, Camera Presets, Drop-Lines, Shadows & Grid

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 08)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-08-3d-stage-engine-and-camera-system.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-08-3d-stage-engine-and-camera-system.md) เพื่อยืนยันความถูกต้องของเครื่องยนต์แสดงผล 3 มิติ (**3D Coordinate Stage Engine & Spatial Orientation System**) ครอบคลุมการแปลงสเกลข้อมูลจริงเข้าสู่กล่องพิกัดมาตรฐาน `[-5.0, +5.0]` (`WorldCoordinateScaler` ใน `math_3d.py`), การสลับมุมกล้องทั้ง 5 โหมด (`Isometric`, `Top X-Y`, `Front X-Z`, `Side Y-Z`, `Follow Ball`) พร้อมคีย์ลัด `1-5, R`, การวาดเส้นประทิ้งดิ่งตั้งฉาก (`Orthogonal Drop-Lines`) และวงแหวนเงาบนพื้น, การวาดจุดข้อมูลพื้นหลังด้วย `THREE.InstancedMesh` ใน 1 Draw Call, การทำงานแบบ **100% Offline** (ไม่มีการดึงฟอนต์ `.woff` หรือ `.hdr` จาก CDN), และการคืนหน่วยความจำการ์ดจอ (`geometry.dispose()` / `material.dispose()`)

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend 3D Math & Stage Endpoint**:
  - `app.core.math_3d` (`WORLD_MIN`, `WORLD_MAX`, `AxisScaleMeta`, `WorldCoordinateScaler.build_axis_scale`, `WorldCoordinateScaler.to_world_coords`)
  - REST Endpoint: `POST /api/v1/stage3d/initialize-scene`
- **Frontend 3D Stage Domain**:
  - `src/domains/stage3d/store/useStage3DStore.ts`
  - `src/domains/stage3d/hooks/useDisposableThreeResources.ts`
  - `Stage3DCanvas.tsx`, `StageAxisControlBar.tsx`, `CameraPresetHUD.tsx`, `CameraRigController.tsx`, `CoordinateCage3D.tsx`, `ParametricSurfaceMesh.tsx`, `InstancedPointCloud.tsx`, `ActiveDataPointSphere.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_08/
├── conftest.py
├── test_world_coordinate_scaler_math.py
└── test_stage3d_initialize_endpoint.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_08/test_world_coordinate_scaler_math.py
import numpy as np
from app.core.math_3d import WORLD_MAX, WORLD_MIN, WorldCoordinateScaler

def test_world_coordinate_normalization_maps_min_max_to_minus5_plus5() -> None:
    """ทดสอบว่าค่าต่ำสุดและสูงสุดของข้อมูลจริงถูกแปลงเป็น [-5.0, +5.0] เป๊ะ และป้าย Tick เก็บค่าจริงไว้ครบถ้วน"""
    prices = np.array([1_000_000.0, 5_500_000.0, 10_000_000.0], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("selling_price", prices, feature_idx=None, n_ticks=5)

    assert scale.raw_min == 1_000_000.0
    assert scale.raw_max == 10_000_000.0
    assert scale.tick_world_positions == [-5.0, -2.5, 0.0, 2.5, 5.0]
    assert scale.tick_raw_labels[0] == 1_000_000.0
    assert scale.tick_raw_labels[-1] == 10_000_000.0

    world_coords = WorldCoordinateScaler.to_world_coords(prices, scale)
    assert world_coords.dtype == np.float32
    assert np.allclose(world_coords, [-5.0, 0.0, 5.0], atol=1e-4)

def test_zero_span_constant_feature_does_not_divide_by_zero() -> None:
    """ทดสอบกรณีข้อมูลทุกแถวในแกนนั้นมีค่าเท่ากันหมด (เช่น 42.0 ทุกแถว) ต้องไม่เกิด NaN และจุดอยู่ที่ World 0.0"""
    constant_vals = np.array([42.0, 42.0, 42.0], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("const_col", constant_vals, feature_idx=0)

    assert scale.raw_min == 41.0
    assert scale.raw_max == 43.0
    world_coords = WorldCoordinateScaler.to_world_coords(constant_vals, scale)
    assert not np.isnan(world_coords).any()
    assert np.allclose(world_coords, [0.0, 0.0, 0.0], atol=1e-4)

def test_extreme_outlier_clamped_within_safe_frustum_bounds() -> None:
    """ทดสอบกรณีมีจุด Outlier ทะลุสเกลปกติ ต้องถูก Clip ให้อยู่ในช่วง [-7.0, +7.0] เพื่อไม่ให้หลุด Frustum กล้อง"""
    normal_vals = np.array([0.0, 100.0], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("x", normal_vals, feature_idx=0)

    extreme_vals = np.array([-5000.0, 50.0, 9000.0], dtype=np.float32)
    world_coords = WorldCoordinateScaler.to_world_coords(extreme_vals, scale)
    assert np.allclose(world_coords, [-7.0, 0.0, 7.0], atol=1e-4)
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & WebGL Mock)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/stage3d/__tests__/
├── useStage3DStore.test.ts
├── CameraPresetHUD.test.tsx
├── useDisposableThreeResources.test.ts
└── offlineDreiAudit.test.ts
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/stage3d/__tests__/CameraPresetHUD.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { CameraPresetHUD } from '../components/CameraPresetHUD';
import { useStage3DStore } from '../store/useStage3DStore';

describe('CameraPresetHUD 1-Click Orientation & Keyboard Shortcuts (1-5, R, G)', () => {
  beforeEach(() => {
    useStage3DStore.setState({
      cameraPreset: 'isometric',
      showDropLines: true,
      showSurface: true,
      showResidualBars: true,
      showBackgroundCloud: true,
    });
  });

  it('เปลี่ยนมุมกล้องได้ทันทีเมื่อคลิกปุ่มหรือกดคีย์ลัดเลข 1 ถึง 5 และเปิด/ปิด Drop-Lines ด้วยปุ่ม G', () => {
    render(<CameraPresetHUD />);

    fireEvent.keyDown(window, { key: '2' });
    expect(useStage3DStore.getState().cameraPreset).toBe('top_xy');

    fireEvent.keyDown(window, { key: '3' });
    expect(useStage3DStore.getState().cameraPreset).toBe('front_xz');

    fireEvent.keyDown(window, { key: '4' });
    expect(useStage3DStore.getState().cameraPreset).toBe('side_yz');

    fireEvent.keyDown(window, { key: '5' });
    expect(useStage3DStore.getState().cameraPreset).toBe('follow_ball');

    fireEvent.keyDown(window, { key: 'g' });
    expect(useStage3DStore.getState().showDropLines).toBe(false);
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบการคืน VRAM (`geometry.dispose()` & `material.dispose()`) และการห้ามใช้ `<Environment>` ออนไลน์ (`vitest`)
```typescript
// frontend/src/domains/stage3d/__tests__/useDisposableThreeResources.test.ts
import { renderHook } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { useDisposableThreeResources } from '../hooks/useDisposableThreeResources';

describe('Zero VRAM Leak Policy — Explicit Geometry & Material Disposal', () => {
  it('เรียก geometry.dispose() และ material.dispose() ทันทีเมื่อเปลี่ยนข้อมูลพื้นผิวหรือ Unmount Component', () => {
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.MeshStandardMaterial();
    const geoDisposeSpy = vi.spyOn(geo, 'dispose');
    const matDisposeSpy = vi.spyOn(mat, 'dispose');

    const { unmount } = renderHook(() => useDisposableThreeResources([geo], [mat]));
    unmount();

    expect(geoDisposeSpy).toHaveBeenCalledTimes(1);
    expect(matDisposeSpy).toHaveBeenCalledTimes(1);
  });
});
```

```typescript
// frontend/src/domains/stage3d/__tests__/offlineDreiAudit.test.ts
import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect } from 'vitest';

describe('100% Offline @react-three/drei Static Code Audit', () => {
  it('ห้ามมี <Environment preset=... /> หรือ <Text> ที่ไม่ระบุ Local Font ในโฟลเดอร์ stage3d เด็ดขาด', () => {
    const componentsDir = path.resolve(__dirname, '../components');
    if (!fs.existsSync(componentsDir)) return;

    const files = fs.readdirSync(componentsDir).filter((f) => f.endsWith('.tsx'));
    for (const file of files) {
      const content = fs.readFileSync(path.join(componentsDir, file), 'utf-8');
      expect(content).not.toMatch(/<Environment\b/);
      //ตรวจสอบว่าไม่มีการสร้าง new THREE.Vector3() ใน useFrame
      const useFrameBlocks = content.split('useFrame(').slice(1);
      for (const block of useFrameBlocks) {
        expect(block.slice(0, 400)).not.toMatch(/new\s+THREE\.(Vector3|Matrix4|Color)\b/);
      }
    }
  });
});
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 08)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **ฉาก 3D มืดสนิทหรือค้างในเครื่องที่ไม่มีอินเทอร์เน็ต** | `offlineDreiAudit.test.ts` | สแกนซอร์สโค้ดใน `src/domains/stage3d/components/*.tsx` | ไม่มี `<Environment` และใช้ `<Html>` + `<ambientLight>` + `<directionalLight>` เท่านั้น |
| **2** | **ข้อมูลจริงมีสเกลต่างกันล้านเท่า (`bedrooms: 1..5` vs `price: 1M..50M`)** | `test_world_coordinate_normalization_maps_min_max_to_minus5_plus5()` | ส่งอาเรย์ที่มีสเกล `1..5` และ `1,000,000..10,000,000` เข้า `WorldCoordinateScaler` | ทั้งสองแกนถูกแปลงเป็น `[-5.0, +5.0]` เท่ากันโดยป้าย `tick_raw_labels` ยังแสดงค่าจริง |
| **3** | **ข้อมูลในคอลัมน์ที่เลือกเป็นแกนมีค่าเท่ากันทุกแถว (`raw_max == raw_min`)** | `test_zero_span_constant_feature_does_not_divide_by_zero()` | ส่งอาเรย์ `[42.0, 42.0, 42.0]` เข้า `build_axis_scale()` | ขยายขอบเขตเป็น `[41.0, 43.0]` อัตโนมัติและคืนพิกัดโลก `0.0` โดยไม่มี `NaN` |
| **4** | **จุดข้อมูล Outlier มีค่าทะลุสเกลจนลูกบอลลอยหลุดออกจากจอ** | `test_extreme_outlier_clamped_within_safe_frustum_bounds()` | ส่งค่า Outlier `-5000.0` และ `+9000.0` เข้า `to_world_coords()` | ถูกจำกัดขอบเขตด้วย `np.clip` ให้อยู่ในช่วง `[-7.0, +7.0]` เสมอ |
| **5** | **ผู้ใช้งงทิศทางและความลึกในอวกาศ 3 มิติ (Parallax Illusion)** | `CameraPresetHUD.test.tsx` & Drop-Lines test | ทดสอบการเรนเดอร์ `ActiveDataPointSphere` และการสลับปุ่มกล้อง `1-5` | วาดเส้นประทิ้งดิ่งลงพื้นพร้อมวงแหวนเงา และสลับมุมมองตรง (`Top X-Y`, `Front X-Z`, `Side Y-Z`) ได้ทันที |
| **6** | **VRAM รั่วไหล (Memory Leak) เมื่อผู้ใช้สลับแกน $X/Y$ ไปมาหลายครั้ง** | `useDisposableThreeResources.test.ts` | จำลองการสลับแกนและ Unmount Component | `geometry.dispose()` และ `material.dispose()` ถูกเรียกครบ 100% ทุกครั้ง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **60 FPS & Draw Call Budget**: การวาดฉาก 3D พร้อมจุดข้อมูลพื้นหลัง `50,000 จุด` ด้วย `THREE.InstancedMesh` ต้องใช้ Draw Calls รวมทั้งฉากไม่เกิน `15 Draw Calls` และรักษาอัตราเฟรมที่ `60 FPS`
- [ ] **Zero Allocation Inside `useFrame`**: ผ่านการตรวจสอบ Static Audit ว่าไม่มีการสั่ง `new THREE.Vector3()` หรือ `new THREE.Matrix4()` ภายในลูป `useFrame`
- [ ] **Zero VRAM Leak**: เมื่อสลับแกน $X/Y$ ติดต่อกัน 20 ครั้ง ค่า `gl.info.memory.geometries` ใน Three.js Renderer ต้องคงที่ (ไม่เพิ่มขึ้นสะสม)
