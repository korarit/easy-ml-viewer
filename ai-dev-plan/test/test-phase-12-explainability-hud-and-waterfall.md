# Test Phase 12: Explainability & 3D HUD — Floating 3D Badges, Local Feature Contribution & Waterfall Panel

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 12)
แผนการทดสอบนี้เชื่อมต่อแบบ 1-ต่อ-1 กับไฟล์ [`phase-12-explainability-hud-and-waterfall.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-12-explainability-hud-and-waterfall.md) เพื่อตรวจสอบความแม่นยำทางคณิตศาสตร์ของ **Local Feature Attribution** (คุณสมบัติ Additivity: $E[f(X)] + \sum \phi_i = f(x_t)$), ความเร็วในการคำนวณ Percentile ด้วย Binary Search (`np.searchsorted`), การรวมค่า `other_contributions_sum` เมื่อมีจำนวน Feature มากกว่า Top-K, และกฎเหล็กฝั่ง Frontend ที่ต้องมี `<Html>` HUD เพียง 1 Instance เดียวในฉาก 3D

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Domain & API**:
  - `backend/app/domains/explainability/value_objects.py`: `FeatureAttributionItemVO`
  - `backend/app/domains/explainability/entities.py`: `FrameExplanationEntity`
  - `backend/app/domains/explainability/services.py`: `FastLocalAttributionService` (`compute_linear_contributions`, `compute_fast_perturbation_contributions`, `compute_percentiles_for_row`)
  - `backend/app/presentation/api/v1/explainability_router.py`: `GET /api/v1/explainability/{session_id}/frames/{frame_index}`
- **Frontend Store & Components**:
  - `frontend/src/domains/explainability/store/useExplainabilityStore.ts`
  - `frontend/src/domains/explainability/components/ActiveBallHudOverlay.tsx`, `FeatureWaterfallPanel.tsx`, `FrameFeatureInspector.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 Unit Tests สำหรับ `FastLocalAttributionService`
ไฟล์ทดสอบ: `backend/tests/unit/domains/test_fast_local_attribution.py`

```python
import numpy as np
import pytest
from app.domains.explainability.services import FastLocalAttributionService


def test_compute_linear_contributions_satisfies_exact_additivity():
    """ตรวจสอบว่า Exact Linear SHAP (w_i * (x_i - mean_i)) บวกกับ base_val แล้วเท่ากับผลทำนายจริง 100%"""
    weights = np.array([2.5, -1.5, 0.8], dtype=np.float32)
    intercept = 1.2
    X_batch = np.array(
        [
            [10.0, 4.0, 2.0],
            [20.0, 8.0, 5.0],
            [15.0, 6.0, 3.0],
        ],
        dtype=np.float32,
    )
    means = X_batch.mean(axis=0)

    contribs, base_val = FastLocalAttributionService.compute_linear_contributions(
        X_batch=X_batch,
        feature_means=means,
        weights=weights,
        intercept=intercept,
    )

    assert contribs.dtype == np.float32
    for row_idx in range(len(X_batch)):
        exact_pred = float(np.dot(X_batch[row_idx], weights) + intercept)
        reconstructed_pred = float(base_val + np.sum(contribs[row_idx]))
        assert np.isclose(reconstructed_pred, exact_pred, atol=1e-4)


def test_compute_fast_perturbation_contributions_single_batch_and_zero_sum_guard():
    """ตรวจสอบว่า Perturbation เรียก predict_fn เพียง 2 ครั้ง (1 สำหรับแถวปัจจุบัน + 1 สำหรับเมทริกซ์ D แถว) และกันหารด้วยศูนย์"""
    call_sizes: list[int] = []

    def mock_nonlinear_predict(X_in: np.ndarray) -> np.ndarray:
        call_sizes.append(len(X_in))
        return (X_in[:, 0] ** 2 - 3.0 * X_in[:, 1] + 0.5 * X_in[:, 2]).astype(np.float32)

    medians = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    base_pred = float(mock_nonlinear_predict(medians.reshape(1, -1))[0])
    call_sizes.clear()

    x_row = np.array([5.0, -2.0, 6.0], dtype=np.float32)
    expected_row_pred = float(mock_nonlinear_predict(x_row.reshape(1, -1))[0])
    call_sizes.clear()

    contribs = FastLocalAttributionService.compute_fast_perturbation_contributions(
        predict_fn=mock_nonlinear_predict,
        x_row=x_row,
        feature_medians=medians,
        base_prediction=base_pred,
    )

    # ต้องเรียก predict_fn แค่ 2 ครั้ง (ขนาด 1 และขนาด D=3) เท่านั้น ไม่วนลูปเรียกทีละคอลัมน์!
    assert call_sizes == [1, 3]
    assert np.isclose(base_pred + float(np.sum(contribs)), expected_row_pred, atol=1e-4)

    # Edge Case #1: เมื่อ x_row มีค่าเท่ากับ medians เป๊ะ (sum_raw == 0.0) ต้องไม่เกิด ZeroDivisionError
    zero_contribs = FastLocalAttributionService.compute_fast_perturbation_contributions(
        predict_fn=mock_nonlinear_predict,
        x_row=medians.copy(),
        feature_medians=medians,
        base_prediction=base_pred,
    )
    assert not np.isnan(zero_contribs).any()
    assert np.all(zero_contribs == 0.0)


def test_compute_percentiles_for_row_binary_search_accuracy():
    """ตรวจสอบการหา Percentile ด้วย np.searchsorted ว่าได้ค่าตรงตามตำแหน่งลำดับข้อมูลจริง"""
    sorted_cols = np.column_stack([
        np.linspace(1.0, 100.0, 100, dtype=np.float32),
        np.linspace(0.0, 10.0, 100, dtype=np.float32),
    ])
    row = np.array([50.0, 9.0], dtype=np.float32)
    pcts = FastLocalAttributionService.compute_percentiles_for_row(row, sorted_cols)

    assert pcts.dtype == np.float32
    assert np.isclose(pcts[0], 50.0, atol=1.0)
    assert np.isclose(pcts[1], 90.0, atol=1.0)
```

### 2.2 API Integration Tests สำหรับ `GET /api/v1/explainability/{session_id}/frames/{frame_index}`
ไฟล์ทดสอบ: `backend/tests/integration/api/test_explainability_router.py`

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_frame_explanation_top_k_and_other_sum(async_client: AsyncClient, seeded_wide_session_id: str):
    """Edge Case #2: เมื่อโมเดลมี 25 Features แต่ขอ top_k=5 ต้องคืน 5 อันดับแรกพร้อมคำนวณ other_contributions_sum"""
    response = await async_client.get(
        f"/api/v1/explainability/{seeded_wide_session_id}/frames/0?top_k=5"
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body["attributions"]) == 5
    # ตรวจสอบว่าเรียงตาม |contribution| จากมากไปน้อย
    abs_vals = [abs(item["contribution"]) for item in body["attributions"]]
    assert abs_vals == sorted(abs_vals, reverse=True)

    # ตรวจสอบคุณสมบัติ Additivity: base_value + sum(top_5) + other_sum == output_value
    top_5_sum = sum(item["contribution"] for item in body["attributions"])
    reconstructed = body["base_value"] + top_5_sum + body["other_contributions_sum"]
    assert abs(reconstructed - body["output_value"]) < 1e-3
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 Component Tests สำหรับ `FeatureWaterfallPanel.tsx` & `FrameFeatureInspector.tsx`
ไฟล์ทดสอบ: `frontend/src/domains/explainability/__tests__/FeatureWaterfallPanel.test.tsx`

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FeatureWaterfallPanel } from '../components/FeatureWaterfallPanel';
import { FrameExplanation } from '../types/explainability.types';

const sampleExplanation: FrameExplanation = {
  frame_index: 42,
  dataset_row_index: 1042,
  base_value: 0.42,
  output_value: 0.892,
  other_contributions_sum: 0.015,
  attributions: [
    {
      feature_index: 0,
      feature_name: 'credit_score',
      raw_value: 780.0,
      percentile: 94.2,
      z_score: 1.85,
      contribution: 0.312,
    },
    {
      feature_index: 1,
      feature_name: 'debt_ratio',
      raw_value: 0.48,
      percentile: 99.4, // > P99 Outlier!
      z_score: 3.1,
      contribution: -0.145,
    },
  ],
};

describe('Phase 12: FeatureWaterfallPanel & Percentile Outlier Badge', () => {
  it('แสดงค่า Base Value, Output Value, แท่งบวก (+0.312) และแท่งลบ (-0.145) ด้วย GPU scaleX', () => {
    const { container } = render(<FeatureWaterfallPanel explanation={sampleExplanation} />);

    expect(screen.getByText(/0.420/)).toBeInTheDocument();
    expect(screen.getByText(/0.892/)).toBeInTheDocument();
    expect(screen.getByText(/\+0.312/)).toBeInTheDocument();
    expect(screen.getByText(/-0.145/)).toBeInTheDocument();

    // ตรวจสอบว่าแท่งกราฟใช้ transform: scaleX(...) เพื่อหลีกเลี่ยง Layout Thrashing
    const barEl = container.querySelector('[data-testid="waterfall-bar-credit_score"]') as HTMLElement;
    expect(barEl.style.transform).toContain('scaleX');
  });

  it('แสดงป้ายเตือน Outlier เมื่อค่า Percentile ของ Feature อยู่ในกลุ่ม >= P99 หรือ <= P01', () => {
    render(<FeatureWaterfallPanel explanation={sampleExplanation} />);
    expect(screen.getByText(/Outlier/i)).toBeInTheDocument();
  });

  it('รองรับคีย์ลัด H เพื่อซ่อน/แสดง 3D HUD และ E เพื่อพับเก็บแผง Explainability', () => {
    const { container } = render(<FeatureWaterfallPanel explanation={sampleExplanation} />);
    fireEvent.keyDown(window, { key: 'h' });
    expect(container.querySelector('[data-hud-hidden="true"]')).toBeDefined();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 การบังคับกฎ Single `<Html>` Instance ในฉาก 3D และห้ามใช้ `shap.KernelExplainer`
```python
import sys
import time
import numpy as np
from app.domains.explainability.services import FastLocalAttributionService


def test_fast_attribution_latency_and_no_kernel_explainer_import():
    """ตรวจสอบว่าเวลาคำนวณต่อเฟรมบน 50 Features เร็วกว่า 5ms และไม่มีการเรียกใช้ KernelExplainer"""
    rng = np.random.default_rng(123)
    X_50 = rng.normal(size=(100, 50)).astype(np.float32)
    medians = np.median(X_50, axis=0)
    weights = rng.normal(size=50).astype(np.float32)

    def fast_predict(batch: np.ndarray) -> np.ndarray:
        return batch @ weights

    t0 = time.perf_counter()
    _ = FastLocalAttributionService.compute_fast_perturbation_contributions(
        predict_fn=fast_predict,
        x_row=X_50[0],
        feature_medians=medians,
        base_prediction=float(medians @ weights),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 5.0
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | เคสปัญหา (Edge Case จากไฟล์ Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องตรวจผ่าน (Expected Assertion) |
| :---: | :--- | :--- | :--- |
| 1 | **ผลรวม Contribution ทุกตัวเป็น `0.0` (`x_row == medians`)** | ส่ง `x_row` ที่มีค่าเท่ากับ `feature_medians` ทุกคอลัมน์ | คืนอาเรย์ `0.0` ชนิด `float32` โดยไม่เกิด `ZeroDivisionError` หรือ `NaN` |
| 2 | **โมเดลมีจำนวน Feature สูงมาก (เช่น $D = 300$ คอลัมน์)** | เรียก API พร้อม `top_k=10` บน Session 300 คอลัมน์ | คืน `attributions` 10 รายการแรก และรวมที่เหลืออีก 290 คอลัมน์ไว้ใน `other_contributions_sum` |
| 3 | **ป้าย 3D HUD บังการคลิกจุดข้อมูลในฉาก 3D** | ตรวจสอบ CSS class บน `<Html>` container ของ `ActiveBallHudOverlay` | มีคลาส `pointer-events-none` และตำแหน่ง Offset `[0, 0.65, 0]` เหนือลูกบอล |
| 4 | **Log-Odds vs Probability Scale ใน GBDT (`pred_contribs`)** | ทดสอบกับโมเดล XGBoost Binary Classification | ค่า Contribution ถูกแปลงสเกลหรือแสดงป้ายหน่วยชัดเจน และผลรวมสอดคล้องกับ Output |
| 5 | **Multi-Class Classification ($C = 4$ คลาส)** | ส่งข้อมูลแถวที่โมเดลทายชนะที่ `Class 2` | ระบบดึง Contribution ของ `Class 2` มาแสดงบน Waterfall ตรงกับคลาสที่ชนะ |
| 6 | **ตัวเลข Raw Value มีทศนิยมยาว 12 ตำแหน่ง (เช่น `0.000012345678`)** | ส่งค่า `raw_value = 0.000012345678` เข้า `FrameFeatureInspector` | แสดงผลด้วย Scientific Notation `1.23e-5` ไม่ล้นกรอบตาราง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Attribution Latency SLA**: เวลาคำนวณ Local Attribution ต่อเฟรมต้อง `< 5 ms` สำหรับ Linear/Tree และ `< 10 ms` สำหรับ Black-box Perturbation
- [ ] **Additivity Accuracy**: ความคลาดเคลื่อนระหว่าง `base_value + sum(attributions) + other_contributions_sum` กับ `output_value` ต้อง `< 1e-4`
- [ ] **Single `<Html>` Instance Rule**: ในฉาก 3D ต้องมี DOM Node ที่สร้างจาก `<Html>` ของ `@react-three/drei` เพียง `1 ชุด` เท่านั้น
- [ ] **Test Coverage**: Unit & Integration Coverage ของ `FastLocalAttributionService` ต้อง $\ge 95\%$
