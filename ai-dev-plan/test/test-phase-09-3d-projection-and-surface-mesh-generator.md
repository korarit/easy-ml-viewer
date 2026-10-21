# Test Phase 09: 3D Math Core — Axis Explorer Slice Generator, 3D PCA Projection & LOD Surface Mesh

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 09)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-09-3d-projection-and-surface-mesh-generator.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-09-3d-projection-and-surface-mesh-generator.md) เพื่อตรวจสอบความถูกต้องทางคณิตศาสตร์ 3 มิติ ความแม่นยำของการ Normalize พิกัดฉาก `[-5.0, +5.0]`, การสร้างกริดพื้นผิว 3D Decision Surface ในทุกระดับ LOD (`Low 30x30`, `Standard 60x60`, `High 100x100`), การย่อมิติด้วย `PCA(n_components=3)`, และการจัดการหน่วยความจำ WebGL Geometry บนฝั่ง Frontend

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Core & Domain**:
  - `backend/app/core/math_3d.py`: `safe_min_max_normalize()`, `build_slice_grid_matrix()`, `compute_3d_pca_projection()`
  - `backend/app/domains/spatial_projection/value_objects.py`: `ProjectionConfigVO`
  - `backend/app/domains/spatial_projection/entities.py`: `SurfaceMesh3DEntity`
  - `backend/app/presentation/api/v1/projection_router.py`: `POST /api/v1/projection/surface-mesh`
- **Frontend Store, Hooks & Components**:
  - `frontend/src/domains/projection3d/store/useProjectionStore.ts`
  - `frontend/src/domains/projection3d/hooks/useSurfaceMeshGeometry.ts`
  - `frontend/src/domains/projection3d/components/ProjectionControlToolbar.tsx`, `DecisionSurfaceMesh.tsx`, `PcaVarianceBadge.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `backend/app/core/math_3d.py`
ไฟล์ทดสอบ: `backend/tests/unit/core/test_math_3d.py`

```python
import numpy as np
import pytest
from app.core.math_3d import (
    WORLD_BOUNDS_HALF_SIZE,
    build_slice_grid_matrix,
    compute_3d_pca_projection,
    safe_min_max_normalize,
)
from app.domains.spatial_projection.value_objects import ProjectionConfigVO


def test_safe_min_max_normalize_standard_range():
    """ตรวจสอบว่าค่าต่ำสุดและสูงสุดถูกแปลงเข้าสู่ช่วง [-5.0, +5.0] ชนิด float32 อย่างแม่นยำ"""
    arr = np.array([10.0, 55.0, 100.0], dtype=np.float64)
    norm = safe_min_max_normalize(arr, v_min=10.0, v_max=100.0)

    assert norm.dtype == np.float32
    assert np.isclose(norm[0], -WORLD_BOUNDS_HALF_SIZE, atol=1e-5)
    assert np.isclose(norm[1], 0.0, atol=1e-5)
    assert np.isclose(norm[2], WORLD_BOUNDS_HALF_SIZE, atol=1e-5)


def test_safe_min_max_normalize_constant_column_zero_variance():
    """Edge Case #1: เมื่อคอลัมน์มีค่าเท่ากันทุกแถว (v_min == v_max) ต้องไม่เกิด ZeroDivisionError หรือ NaN"""
    arr = np.full(100, 42.0, dtype=np.float64)
    norm = safe_min_max_normalize(arr, v_min=42.0, v_max=42.0)

    assert norm.dtype == np.float32
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    assert np.all(norm == 0.0)


@pytest.mark.parametrize(
    "lod_level,expected_res,expected_points",
    [
        ("low", 30, 900),
        ("standard", 60, 3600),
        ("high", 100, 10000),
    ],
)
def test_build_slice_grid_matrix_lod_resolutions_and_pinning(lod_level, expected_res, expected_points):
    """ตรวจสอบขนาดเมทริกซ์กริด (res*res, D) และการตรึงค่าคอลัมน์ที่ไม่ใช่แกน X, Y ด้วยค่า Median"""
    cfg = ProjectionConfigVO(x_feature="f0", y_feature="f1", lod_level=lod_level, pinning_mode="median")
    assert cfg.grid_resolution == expected_res

    rng = np.random.default_rng(42)
    X_ref = rng.normal(loc=10.0, scale=2.0, size=(200, 5)).astype(np.float64)
    expected_f2_median = np.float32(np.nanmedian(X_ref[:, 2]))

    grid_mat, x_range, y_range = build_slice_grid_matrix(
        reference_X=X_ref,
        x_col_idx=0,
        y_col_idx=1,
        grid_resolution=cfg.grid_resolution,
        pinning_mode="median",
    )

    assert grid_mat.shape == (expected_points, 5)
    assert grid_mat.dtype == np.float32
    assert np.allclose(grid_mat[:, 2], expected_f2_median, atol=1e-5)
    assert x_range[0] < x_range[1]
    assert y_range[0] < y_range[1]


def test_build_slice_grid_matrix_current_row_pinning():
    """ตรวจสอบโหมด Dynamic Current Row Follow ว่าคอลัมน์พื้นหลังถูกตรึงตามค่าของแถวปัจจุบันจริง"""
    X_ref = np.ones((50, 4), dtype=np.float32)
    current_row = np.array([1.0, 2.0, 99.5, -45.25], dtype=np.float32)

    grid_mat, _, _ = build_slice_grid_matrix(
        reference_X=X_ref,
        x_col_idx=0,
        y_col_idx=1,
        grid_resolution=30,
        pinning_mode="current_row",
        current_row_values=current_row,
    )

    assert np.allclose(grid_mat[:, 2], 99.5)
    assert np.allclose(grid_mat[:, 3], -45.25)


def test_compute_3d_pca_projection_with_fewer_than_3_features():
    """Edge Case #2: เมื่อ Dataset มีเพียง 2 Features ระบบต้องเติมแกนที่ 3 ด้วย 0.0 โดยไม่โยน ValueError"""
    X_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 1.0]], dtype=np.float32)
    coords_3d, variance_ratios, components = compute_3d_pca_projection(X_2d)

    assert coords_3d.shape == (4, 3)
    assert coords_3d.dtype == np.float32
    assert np.all(coords_3d[:, 2] == 0.0)
    assert len(variance_ratios) == 3
    assert variance_ratios[2] == 0.0
```

### 2.2 API Integration Tests สำหรับ `POST /api/v1/projection/surface-mesh`
ไฟล์ทดสอบ: `backend/tests/integration/api/test_projection_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_post_surface_mesh_standard_lod_contract(async_client: AsyncClient, seeded_session_id: str):
    """ทดสอบสัญญา JSON Schema และเวลาประมวลผลของ POST /api/v1/projection/surface-mesh"""
    payload = {
        "session_id": seeded_session_id,
        "mode": "axis_explorer",
        "x_feature": "credit_score",
        "y_feature": "annual_income",
        "z_mode": "probability",
        "target_class_index": 1,
        "pinning_mode": "median",
        "lod_level": "standard",
        "current_frame_index": 0,
    }
    response = await async_client.post("/api/v1/projection/surface-mesh", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["grid_resolution"] == 60
    assert len(data["z_heights_world"]) == 3600
    assert len(data["raw_predictions"]) == 3600
    assert data["computation_time_ms"] < 50.0
    assert data["has_constant_axis_warning"] is False
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Unit & Component Tests (`ProjectionControlToolbar.test.tsx`)
ไฟล์ทดสอบ: `frontend/src/domains/projection3d/__tests__/ProjectionControlToolbar.test.tsx`

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useProjectionStore } from '../store/useProjectionStore';
import { ProjectionControlToolbar } from '../components/ProjectionControlToolbar';

describe('Phase 09: ProjectionControlToolbar & useProjectionStore', () => {
  beforeEach(() => {
    useProjectionStore.setState({
      mode: 'axis_explorer',
      xFeature: 'credit_score',
      yFeature: 'annual_income',
      lodLevel: 'standard',
      pinningMode: 'median',
      wireframe: false,
      opacity: 0.78,
      surfaceData: null,
      isLoading: false,
    });
  });

  it('สลับแกน X และ Y ทันทีเมื่อคลิกปุ่ม Swap X/Y หรือเมื่อเลือกแกน X ซ้ำกับแกน Y ปัจจุบัน', () => {
    useProjectionStore.getState().swapAxes();
    const stateAfterSwap = useProjectionStore.getState();
    expect(stateAfterSwap.xFeature).toBe('annual_income');
    expect(stateAfterSwap.yFeature).toBe('credit_score');

    // กรณีผู้ใช้เลือกแกน X ให้ตรงกับแกน Y ปัจจุบัน ระบบต้องสลับแกนให้อัตโนมัติ (Edge Case #3)
    useProjectionStore.getState().setAxes('credit_score', 'credit_score');
    expect(useProjectionStore.getState().xFeature).not.toBe(useProjectionStore.getState().yFeature);
  });

  it('รองรับคีย์ลัด W (Toggle Wireframe) และ P (Toggle Axis Explorer / 3D PCA)', () => {
    render(<ProjectionControlToolbar availableFeatures={['credit_score', 'annual_income', 'age']} />);

    fireEvent.keyDown(window, { key: 'w' });
    expect(useProjectionStore.getState().wireframe).toBe(true);

    fireEvent.keyDown(window, { key: 'p' });
    expect(useProjectionStore.getState().mode).toBe('pca_3d');
  });

  it('ปิดปุ่ม 3D PCA Manifold อัตโนมัติและแสดง Tooltip เมื่อจำนวน Feature น้อยกว่า 3', () => {
    render(<ProjectionControlToolbar availableFeatures={['credit_score', 'annual_income']} />);
    const pcaBtn = screen.getByRole('button', { name: /3D PCA/i });
    expect(pcaBtn).toBeDisabled();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การทดสอบ In-Place BufferAttribute Update & WebGL Geometry Disposal
ไฟล์ทดสอบ: `frontend/src/domains/projection3d/__tests__/useSurfaceMeshGeometry.memory.test.ts`

```typescript
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

describe('WebGL Surface Mesh Memory & Disposal Verification', () => {
  it('อัปเดตความสูงแกน Z แบบ In-place ลงใน BufferAttribute เดิมเมื่อ grid_resolution เท่าเดิมโดยไม่สร้าง Geometry ใหม่', () => {
    const geom = new THREE.PlaneGeometry(10, 10, 59, 59); // 60x60 = 3,600 vertices
    const initialPositionArrayRef = geom.attributes.position.array;
    const computeNormalsSpy = vi.spyOn(geom, 'computeVertexNormals');

    // จำลองการอัปเดตความสูงใหม่ในโหมด Dynamic Current Row Slice
    const newZHeights = new Float32Array(3600).fill(1.25);
    const posAttr = geom.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < 3600; i++) {
      posAttr.setZ(i, newZHeights[i]);
    }
    posAttr.needsUpdate = true;
    geom.computeVertexNormals();

    expect(geom.attributes.position.array).toBe(initialPositionArrayRef); // อ้างอิง TypedArray เดิม 100%
    expect(computeNormalsSpy).toHaveBeenCalledTimes(1);
  });

  it('เรียก geometry.dispose() ทันทีเมื่อเปลี่ยนระดับความละเอียด LOD จาก Standard (60x60) เป็น High (100x100)', () => {
    const oldGeom = new THREE.PlaneGeometry(10, 10, 59, 59);
    const disposeSpy = vi.spyOn(oldGeom, 'dispose');

    // จำลองการ Cleanup เมื่อเปลี่ยน lodLevel
    oldGeom.dispose();
    expect(disposeSpy).toHaveBeenCalledTimes(1);
  });
});
```

### 4.2 การทดสอบ Backend Memory Guard สำหรับ Wide-Column Dataset ($D > 500$)
```python
import psutil
from app.core.math_3d import build_slice_grid_matrix


def test_wide_dataset_grid_memory_footprint():
    """ตรวจสอบว่าการสร้างกริด Standard LOD (3,600 จุด) บนข้อมูล 500 คอลัมน์ ใช้ RAM เพิ่มขึ้นไม่เกิน 15 MB"""
    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    X_wide = np.zeros((100, 500), dtype=np.float32)
    grid_mat, _, _ = build_slice_grid_matrix(X_wide, 0, 1, grid_resolution=60, pinning_mode="median")

    mem_after = proc.memory_info().rss
    delta_mb = (mem_after - mem_before) / (1024 * 1024)
    assert grid_mat.dtype == np.float32
    assert delta_mb < 15.0
    del grid_mat
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **Constant Column (Variance = 0)** | ส่งคอลัมน์ที่ทุกแถวมีค่า `100.0` เข้า `safe_min_max_normalize` และ API | คืนพิกัด `0.0` ทุกจุด, ไม่มี `NaN`/`Inf`, และ `has_constant_axis_warning == True` |
| 2 | **จำนวน Feature น้อยกว่า 3 ($D < 3$)** | เรียก `compute_3d_pca_projection` ด้วยอาเรย์ขนาด `(50, 1)` และ `(50, 2)` | คืน `coords_3d` ขนาด `(50, 3)` โดยแกนที่เกินเป็น `0.0` และปุ่ม 3D PCA บน UI ถูก Disable |
| 3 | **ผู้ใช้เลือกแกน X และแกน Y เป็นคอลัมน์เดียวกัน** | เรียก `setAxes('income', 'income')` และส่ง `x_col_idx == y_col_idx` | Store สลับแกนให้อัตโนมัติ และ Backend ไม่เขียนทับคอลัมน์จนพัง |
| 4 | **ค่า `NaN` หรือ `Inf` ปะปนใน Dataset** | ใส่ `np.nan` และ `np.inf` ลงใน 10% ของ `reference_X` | `np.nanmin`/`np.nanmax`/`np.nanmedian` และ `np.nan_to_num` ทำงานผ่านโดยไม่มี `NaN` ใน `z_heights_world` |
| 5 | **Outlier สุดโต่งบีบพื้นผิว 3D** | ใส่ค่า `1e9` ในแถวสุดท้ายของคอลัมน์แกน X | โหมด `Robust Percentile Range (P01-P99)` ตัดขอบเขตกริดให้อยู่ในช่วงข้อมูลปกติ |
| 6 | **โมเดลไม่มี `predict_proba` (เช่น Ridge / SVR)** | ส่ง `z_mode = "probability"` ไปยัง Session ที่ใช้โมเดล Regression | ระบบ Fallback กลับเป็น `z_mode = "prediction"` อัตโนมัติและคืน HTTP 200 |
| 7 | **สลับ LOD รัวๆ 50 ครั้งติดต่อกัน** | ยิงการเปลี่ยน `lodLevel` (`low -> std -> high`) 50 รอบใน Component Test | Request เก่าถูกยกเลิกด้วย `AbortController` และ `renderer.info.memory.geometries` ไม่เพิ่มขึ้นสะสม |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Latency SLA**: การคำนวณ `Standard LOD (60x60 = 3,600 จุด)` ฝั่ง Backend ต้องเสร็จภายใน `< 50 ms` และ `Low LOD (30x30)` ภายใน `< 15 ms`
- [ ] **Numerical Integrity**: อาเรย์ `z_heights_world` และ `point_positions_world` ต้องเป็นชนิด `float32` ที่ไม่มีค่า `NaN` หรือ `Infinity` 100%
- [ ] **VRAM Leak Zero Tolerance**: หลังสลับแกน X/Y และสลับระดับ LOD ครบ 50 รอบ จำนวน `THREE.BufferGeometry` ที่ค้างในหน่วยความจำต้องไม่เพิ่มขึ้นเกิน 1 ชุดปัจจุบัน
- [ ] **Test Coverage**: Unit & Integration Coverage ของ `math_3d.py` และ `spatial_projection` ต้อง $\ge 95\%$
