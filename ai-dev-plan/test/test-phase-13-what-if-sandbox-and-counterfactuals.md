# Test Phase 13: Interactive "What-If" Sandbox, Counterfactual Finder & 3D Ghost Sphere Comparison

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 13)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-13-what-if-sandbox-and-counterfactuals.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-13-what-if-sandbox-and-counterfactuals.md) เพื่อตรวจสอบความถูกต้องของการจำลองผลลัพธ์เมื่อผู้ใช้ปรับค่า Feature แบบสดๆ (**Live What-If Mutation**), การแสดงลูกบอลเงาต้นฉบับคู่กับลูกบอลใหม่และเส้นลูกศรประในฉาก 3D (`GhostComparisonGroup3D`), และอัลกอริทึม **1-Click Nearest Counterfactual Finder** (`KDTree` + 12-step Binary Line Search) ที่ต้องเคารพการล็อกกุญแจ `🔒 Immutable Feature Lock`

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Domain & API**:
  - `backend/app/domains/simulation_whatif/value_objects.py`: `FeatureChangeDeltaVO`
  - `backend/app/domains/simulation_whatif/entities.py`: `WhatIfEvaluationResultEntity`, `CounterfactualSearchResultEntity`
  - `backend/app/domains/simulation_whatif/services.py`: `CounterfactualFinderService.find_minimum_change_counterfactual()`
  - `backend/app/presentation/api/v1/whatif_router.py`: `POST /api/v1/simulation/{session_id}/what-if`, `POST /api/v1/simulation/{session_id}/counterfactual`
- **Frontend Store & Components**:
  - `frontend/src/domains/whatif/store/useWhatIfStore.ts`
  - `frontend/src/domains/whatif/components/WhatIfSandboxPanel.tsx`, `CounterfactualFinderCard.tsx`, `GhostComparisonGroup3D.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `CounterfactualFinderService`
ไฟล์ทดสอบ: `backend/tests/unit/domains/test_counterfactual_finder.py`

```python
import numpy as np
import pytest
from app.domains.simulation_whatif.services import CounterfactualFinderService


def dummy_credit_classifier(X: np.ndarray) -> np.ndarray:
    """โมเดลจำลอง: อนุมัติ (Class 1) เมื่อ (0.01 * credit_score - 2.0 * debt_ratio + 0.02 * age) >= 6.0"""
    score = 0.01 * X[:, 0] - 2.0 * X[:, 1] + 0.02 * X[:, 2]
    return (score >= 6.0).astype(np.float32)


def test_find_minimum_change_counterfactual_flips_class_and_respects_immutable_locks():
    """ตรวจสอบว่า Counterfactual ค้นหาจุดที่เปลี่ยนคำตอบจาก Class 0 -> Class 1 ได้จริง โดยไม่แตะคอลัมน์ที่ถูกล็อก (age)"""
    rng = np.random.default_rng(42)
    X_dataset = np.column_stack([
        rng.uniform(500, 800, 300), # col 0: credit_score
        rng.uniform(0.1, 0.8, 300), # col 1: debt_ratio
        rng.integers(20, 65, 300),  # col 2: age (เราจะล็อกกุญแจคอลัมน์นี้!)
    ]).astype(np.float32)
    y_preds = dummy_credit_classifier(X_dataset)

    # ตัวอย่างลูกค้าที่ถูกปฏิเสธ: credit=550, debt=0.60, age=35 -> score = 5.5 - 1.2 + 0.7 = 5.0 (< 6.0 -> Class 0)
    base_row = np.array([550.0, 0.60, 35.0], dtype=np.float32)
    assert dummy_credit_classifier(base_row.reshape(1, -1))[0] == 0.0

    found, cf_vec, msg = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=dummy_credit_classifier,
        X_dataset=X_dataset,
        y_predictions=y_preds,
        base_row=base_row,
        locked_feature_indices=[2],  # ล็อกคอลัมน์ที่ 2 (age = 35.0 ห้ามเปลี่ยนเด็ดขาด!)
        target_class_or_threshold=1.0,
        is_classification=True,
    )

    assert found is True
    assert cf_vec[2] == 35.0  # อายุต้องคงเดิมที่ 35.0 เป๊ะ!
    assert dummy_credit_classifier(cf_vec.reshape(1, -1))[0] == 1.0  # ผลทำนายต้องพลิกเป็น Class 1 สำเร็จ!

    # ตรวจสอบว่า Binary Line Search บีบค่าให้เข้าใกล้เส้นแบ่งเขต 6.0 มากที่สุด (ประหยัดการปรับค่าที่สุด)
    final_score = 0.01 * cf_vec[0] - 2.0 * cf_vec[1] + 0.02 * cf_vec[2]
    assert 6.0 <= final_score <= 6.15


def test_find_counterfactual_when_all_critical_features_locked_returns_false_gracefully():
    """Edge Case #3: เมื่อผู้ใช้ล็อกกุญแจทุกคอลัมน์ที่มีผลต่อการเปลี่ยนคลาส ต้องคืน found=False พร้อมข้อความอธิบาย"""
    X_dataset = np.array([[500.0, 0.5, 30.0], [780.0, 0.1, 30.0]], dtype=np.float32)
    y_preds = dummy_credit_classifier(X_dataset)
    base_row = np.array([500.0, 0.5, 30.0], dtype=np.float32)

    found, cf_vec, msg = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=dummy_credit_classifier,
        X_dataset=X_dataset,
        y_predictions=y_preds,
        base_row=base_row,
        locked_feature_indices=[0, 1, 2],  # ล็อกทุกคอลัมน์!
        target_class_or_threshold=1.0,
        is_classification=True,
    )

    assert found is False
    assert "Immutable Lock" in msg
    assert np.allclose(cf_vec, base_row)
```

### 2.2 API Integration Tests สำหรับ `POST /api/v1/simulation/{session_id}/what-if`
ไฟล์ทดสอบ: `backend/tests/integration/api/test_whatif_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_post_what_if_mutation_returns_ghost_and_mutated_coordinates(
    async_client: AsyncClient, seeded_session_id: str
):
    """ทดสอบการส่งค่า Feature ที่ปรับแล้ว และตรวจรับพิกัดเดิม (Ghost) คู่กับพิกัดใหม่ (Mutated) และธง OOD"""
    req_body = {
        "base_frame_index": 5,
        "mutated_feature_values": {
            "credit_score": 950.0,  # เกินขอบเขตสูงสุด 850 ของ Dataset -> ต้องติดธง is_out_of_distribution
        },
    }
    resp = await async_client.post(f"/api/v1/simulation/{seeded_session_id}/what-if", json=req_body)
    assert resp.status_code == 200

    data = resp.json()
    assert data["base_frame_index"] == 5
    assert len(data["original_position_world"]) == 3
    assert len(data["mutated_position_world"]) == 3
    assert data["world_distance"] > 0.0
    assert data["changed_features"][0]["is_out_of_distribution"] is True
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 UX Flow Tests: Auto-Pause on Slider Edit & Auto-Reset on Frame Change
ไฟล์ทดสอบ: `frontend/src/domains/whatif/__tests__/WhatIfSandboxPanel.test.tsx`

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useWhatIfStore } from '../store/useWhatIfStore';
import { WhatIfSandboxPanel } from '../components/WhatIfSandboxPanel';

describe('Phase 13: WhatIfSandboxPanel UX Flow & Auto-Guards', () => {
  beforeEach(() => {
    usePlaybackStore.getState().initializeQueue(100);
    usePlaybackStore.getState().play();
    useWhatIfStore.getState().resetSandbox();
  });

  it('หยุดการเล่นวิดีโออัตโนมัติ (Auto-Pause) ทันทีที่ผู้ใช้แตะขยับ Slider ในแผง What-If', () => {
    expect(usePlaybackStore.getState().status).toBe('playing');

    render(<WhatIfSandboxPanel />);
    const slider = screen.getByTestId('whatif-slider-credit_score');
    fireEvent.change(slider, { target: { value: '720' } });

    expect(usePlaybackStore.getState().status).toBe('paused');
    expect(useWhatIfStore.getState().isSandboxActive).toBe(true);
  });

  it('Edge Case #1: รีเซ็ตสถานะ What-If Sandbox อัตโนมัติเมื่อผู้ใช้เปลี่ยนเฟรมบนแถบ Scrubber', () => {
    useWhatIfStore.getState().setMutatedFeature('credit_score', 720);
    expect(useWhatIfStore.getState().isSandboxActive).toBe(true);

    // ผู้ใช้คลิกเลื่อน Scrubber ไปยังเฟรมที่ 40
    usePlaybackStore.getState().seekToFrame(40, true);

    expect(useWhatIfStore.getState().isSandboxActive).toBe(false);
    expect(useWhatIfStore.getState().mutatedValues).toEqual({});
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การทดสอบ Subsampled KDTree Memory & 30ms Slider Throttling + `AbortController`
```python
import time
import numpy as np
from app.domains.simulation_whatif.services import CounterfactualFinderService


def test_counterfactual_finder_performance_under_25ms():
    """ตรวจสอบว่าการค้นหา Counterfactual บนชุดข้อมูล 5,000 แถว 20 คอลัมน์ ใช้เวลาน้อยกว่า 25ms"""
    rng = np.random.default_rng(99)
    X_5k = rng.normal(size=(5000, 20)).astype(np.float32)
    weights = rng.normal(size=20).astype(np.float32)

    def fast_clf(X: np.ndarray) -> np.ndarray:
        return (X @ weights > 0.0).astype(np.float32)

    y_preds = fast_clf(X_5k)
    base_row = X_5k[np.where(y_preds == 0)[0][0]]

    t0 = time.perf_counter()
    found, _, _ = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=fast_clf,
        X_dataset=X_5k,
        y_predictions=y_preds,
        base_row=base_row,
        locked_feature_indices=[0, 1],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert found is True
    assert elapsed_ms < 25.0
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **เปลี่ยนเฟรมบน Scrubber ขณะค้างอยู่ในโหมด What-If** | ปรับค่า Slider แล้วเรียก `seekToFrame(nextFrame)` | `useWhatIfStore` ล้าง `mutatedValues` และซ่อน `GhostSphere` ทันที |
| 2 | **คอลัมน์ Integer / Binary (`0/1`) ถูกลากเป็นทศนิยม** | ตรวจสอบ `step` ของ Slider สำหรับคอลัมน์ `num_children (int64)` | `slider.step === "1"` และค่าใน Counterfactual ถูกปัดเป็นจำนวนเต็มเสมอ |
| 3 | **ล็อกกุญแจ `🔒` ครบทุกคอลัมน์แล้วกด Find Counterfactual** | ล็อกทุก Feature แล้วคลิกปุ่ม `Find Minimum Change` | แจ้งเตือนบน UI ทันทีโดยไม่แครช และคืน `found = False` |
| 4 | **พิมพ์ค่า Out-of-Distribution สูงสุดโต่ง (เช่น `999,999,999`)** | พิมพ์ค่า `1e9` ในช่อง Numeric Spinbox | แสดงป้ายเตือน `⚠️ Out-of-Distribution` และพิกัด 3D ถูก Clamp ไว้ไม่เกินขอบฉาก `[-7.5, +7.5]` |
| 5 | **ปรับค่า Feature ที่ไม่ได้อยู่บนแกน X หรือ Y ของฉาก 3D** | ปรับคอลัมน์ที่ 4 ขณะแกน X=คอลัมน์ 1, Y=คอลัมน์ 2 | ลูกบอล Mutated ขยับขึ้น/ลงตามแกน $Z$ ($\Delta Z$) และแสดงป้าย `"ΔZ Vertical Shift"` |
| 6 | **Dataset ทั้งชุดมีผลทำนายเป็นคลาสเดียวกันหมด** | ส่ง `y_predictions` ที่เป็น `0` ทั้งหมดเข้า `CounterfactualFinderService` | ระบบลองสุ่ม Synthetic Grid Search หรือคืนข้อความแจ้งเตือนอย่างปลอดภัยโดยไม่เกิด `ValueError` ใน `KDTree` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **What-If Mutation Latency**: การประมวลผล `POST /api/v1/simulation/{session_id}/what-if` ต้องตอบกลับภายใน `< 20 ms`
- [ ] **Counterfactual Search SLA**: การค้นหา `POST /api/v1/simulation/{session_id}/counterfactual` ต้องตอบกลับภายใน `< 25 ms`
- [ ] **Immutable Constraint Guarantee**: คอลัมน์ที่ถูกล็อกกุญแจ `🔒` ต้องมีค่า `mutated_value == original_value` เสมอ 100%
- [ ] **Test Coverage**: Unit & Integration Coverage ของ `CounterfactualFinderService` และ `useWhatIfStore` ต้อง $\ge 95\%$
