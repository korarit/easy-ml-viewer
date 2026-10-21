# Test Phase 07: Global Evaluation Lens — Interactive Confusion Matrix, Residual Slicer & Error Auto-Queue

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 07)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-07-global-evaluation-and-error-slicer.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-07-global-evaluation-and-error-slicer.md) เพื่อยืนยันความถูกต้องของ **Global Evaluation Lens** ที่เชื่อมโยงการประเมินผลภาพรวมทั้งชุดข้อมูลเข้ากับการคัดเลือกแถวเข้าคิว (`useTrajectoryQueueStore`) ครอบคลุมการคำนวณ $R^2$, $\text{RMSE}$, $\text{MAE}$, การคัดกรอง `Top 50 Worst Absolute Errors`, `Over-predicted`, `Under-predicted`, การคำนวณตาราง `InteractiveConfusionMatrix` พร้อมฟีเจอร์ **Click-Cell-to-Queue**, และการตรวจจับ `Out-of-Distribution (OOD) Feature Outliers (|Z-Score| > 3.0)`

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Simulation & Evaluation Domain**:
  - `app.domains.simulation.evaluation_value_objects` (`RegressionEvaluationReport`, `ConfusionMatrixCell`, `ClassificationEvaluationReport`, `GlobalEvaluationSummary`)
  - `app.domains.simulation.evaluation_service.GlobalEvaluationService` (`compute_ood_outliers`, `evaluate_regression`, `evaluate_classification`)
  - REST Endpoint: `POST /api/v1/evaluations/compute`
- **Frontend Evaluation Domain**:
  - `src/domains/evaluation/store/useEvaluationStore.ts`
  - `GlobalMetricsRibbon.tsx`, `ResidualSlicerPanel.tsx`, `InteractiveConfusionMatrix.tsx`, `OutlierSlicerCard.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_07/
├── conftest.py
├── test_regression_residual_slicer.py
├── test_classification_confusion_matrix_slicer.py
├── test_ood_zscore_outlier_detector.py
└── test_evaluation_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_07/test_regression_residual_slicer.py
import numpy as np
from app.domains.simulation.evaluation_service import GlobalEvaluationService

def test_evaluate_regression_sorts_worst_over_and_under_predictions_correctly() -> None:
    """
    ตรวจสอบว่า evaluate_regression คำนวณ residual (e = y_true - y_pred)
    และจัดอันดับ Top Worst, Over-predicted (y_pred > y_true), และ Under-predicted (y_pred < y_true)
    โดยอ้างอิงกลับไปยัง original_row_index ได้ถูกต้องแม้มี NaN คั่น
    """
    y_true = np.array([100.0, np.nan, 200.0, 300.0, 400.0], dtype=np.float32)
    # แถว #0: ตรงเป๊ะ (e = 0)
    # แถว #1: NaN (ต้องถูกข้ามอย่างปลอดภัย)
    # แถว #2: ทำนาย 290 (Over-predicted 90 -> e = -90)
    # แถว #3: ทำนาย 150 (Under-predicted 150 -> e = +150, ผิดหนักสุด!)
    # แถว #4: ทำนาย 395 (Under-predicted 5 -> e = +5)
    y_pred = np.array([100.0, 500.0, 290.0, 150.0, 395.0], dtype=np.float32)

    report = GlobalEvaluationService.evaluate_regression(y_true, y_pred, top_k=3)

    assert report.max_abs_error == 150.0
    # ผิดหนักสุดเรียงจาก |e| มากไปน้อย: แถว #3 (|150|) -> แถว #2 (|-90|) -> แถว #4 (|5|)
    assert report.top_worst_abs_error_indices == [3, 2, 4]
    # Over-predicted สูงสุด (e ติดลบมากสุด): แถว #2 (e = -90)
    assert report.top_over_predicted_indices[0] == 2
    # Under-predicted สูงสุด (e เป็นบวกมากสุด): แถว #3 (e = +150)
    assert report.top_under_predicted_indices[0] == 3
    # แม่นยำที่สุด (|e| น้อยสุด): แถว #0 (|e| = 0)
    assert report.top_accurate_indices[0] == 0

def test_ood_outlier_detector_handles_zero_variance_constant_column() -> None:
    """ทดสอบว่าหากมีคอลัมน์ที่มีค่าคงที่เท่ากันหมด (SD = 0) ระบบจะไม่เกิด ZeroDivisionError และตรวจพบแถวที่ > 3 SD ได้"""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(200, 3)).astype(np.float32)
    # คอลัมน์ที่ 1 มีค่าคงที่ = 5.0 ทั้งคอลัมน์ (SD = 0)
    X[:, 1] = 5.0
    # บังคับให้แถวที่ 77 และ 142 มีค่าสุดโต่งในคอลัมน์ที่ 0
    X[77, 0] = 9.5
    X[142, 2] = -12.0

    outliers = GlobalEvaluationService.compute_ood_outliers(X, top_k=10)
    assert outliers[0] == 142
    assert outliers[1] == 77
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/evaluation/__tests__/
├── useEvaluationStore.test.ts
├── ResidualSlicerPanel.test.tsx
└── InteractiveConfusionMatrix.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/evaluation/__tests__/InteractiveConfusionMatrix.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { InteractiveConfusionMatrix } from '../components/InteractiveConfusionMatrix';
import { ResidualSlicerPanel } from '../components/ResidualSlicerPanel';
import { useEvaluationStore } from '../store/useEvaluationStore';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';

describe('Global Evaluation Slicer -> TrajectoryQueue Integration', () => {
  beforeEach(() => {
    useTrajectoryQueueStore.setState({
      selectedRowIndices: [],
      selectedLookupSet: new Set(),
    });
  });

  it('กดปุ่ม Queue Top 50 Worst Errors แล้วส่งรายการแถวที่ผิดหนักสุดเข้า useTrajectoryQueueStore ทันที', () => {
    useEvaluationStore.setState({
      summary: {
        task_type: 'regression',
        evaluated_row_count: 1000,
        ood_outlier_indices: [12, 99],
        regression: {
          r2_score: 0.894,
          rmse: 245120,
          mae: 182400,
          max_abs_error: 1620000,
          residual_hist_bins: [-1e6, 0, 1e6],
          residual_hist_counts: [50, 50],
          top_worst_abs_error_indices: [812, 104, 55],
          top_over_predicted_indices: [104],
          top_under_predicted_indices: [812],
          top_accurate_indices: [1, 2],
        },
        classification: null,
      },
      isComputing: false,
      activeSlicerTag: null,
    });

    render(<ResidualSlicerPanel />);
    const worstBtn = screen.getByRole('button', { name: /Queue Top 50 Worst/i });
    fireEvent.click(worstBtn);

    expect(useTrajectoryQueueStore.getState().selectedRowIndices).toEqual([812, 104, 55]);
  });

  it('คลิกช่อง False Positives ใน InteractiveConfusionMatrix แล้วโหลดเฉพาะแถวในช่องนั้นเข้าคิว', () => {
    useEvaluationStore.setState({
      summary: {
        task_type: 'binary_classification',
        evaluated_row_count: 100,
        ood_outlier_indices: [],
        regression: null,
        classification: {
          accuracy: 0.92,
          macro_precision: 0.91,
          macro_recall: 0.90,
          macro_f1: 0.905,
          class_labels: ['Stay (0)', 'Churn (1)'],
          all_misclassified_indices: [7, 19, 42],
          most_uncertain_indices: [33],
          cells: [
            { actual_class_idx: 0, predicted_class_idx: 0, actual_label: '0', predicted_label: '0', row_count: 50, percentage: 50, sample_row_indices: [0, 1] },
            { actual_class_idx: 0, predicted_class_idx: 1, actual_label: '0', predicted_label: '1', row_count: 3, percentage: 3, sample_row_indices: [7, 19, 42] },
            { actual_class_idx: 1, predicted_class_idx: 0, actual_label: '1', predicted_label: '0', row_count: 0, percentage: 0, sample_row_indices: [] },
            { actual_class_idx: 1, predicted_class_idx: 1, actual_label: '1', predicted_label: '1', row_count: 47, percentage: 47, sample_row_indices: [2, 3] },
          ],
        },
      },
      isComputing: false,
      activeSlicerTag: null,
    });

    render(<InteractiveConfusionMatrix />);
    // คลิกช่อง FP (Actual 0, Predicted 1 -> 3 rows)
    const fpCell = screen.getByTestId('cm-cell-0-1');
    fireEvent.click(fpCell);
    expect(useTrajectoryQueueStore.getState().selectedRowIndices).toEqual([7, 19, 42]);

    // ช่อง FN ที่มี 0 แถว (cm-cell-1-0) ต้องเป็น disabled กดไม่ได้
    const emptyCell = screen.getByTestId('cm-cell-1-0');
    expect(emptyCell).toBeDisabled();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบการประมวลผลแบบแบ่งก้อน (`Chunked Evaluation`) บนข้อมูล 100,000 แถว (`pytest`)
```python
# backend/tests/phase_07/test_chunked_evaluation_memory.py
import os
import psutil
import numpy as np
from app.domains.simulation.evaluation_service import GlobalEvaluationService

def test_evaluate_100k_rows_uses_bounded_ram_under_50mb_delta() -> None:
    """ยืนยันว่าการประเมินผลข้อมูล 100,000 แถว ใช้หน่วยความจำเพิ่มขึ้นชั่วคราวไม่เกิน 50 MB"""
    proc = psutil.Process(os.getpid())
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    y_true = np.random.rand(100_000).astype(np.float32)
    y_pred = (y_true + np.random.normal(0, 0.1, size=100_000)).astype(np.float32)

    report = GlobalEvaluationService.evaluate_regression(y_true, y_pred, top_k=50)
    rss_after_mb = proc.memory_info().rss / (1024 * 1024)

    assert len(report.top_worst_abs_error_indices) == 50
    assert (rss_after_mb - rss_before_mb) < 50.0
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 07)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **ค่าในคอลัมน์ `Actual Target` ไม่ตรงกับชนิดของ `model.classes_`** | `test_target_label_harmonizer_maps_string_to_int_classes()` | CSV เก็บ `"Yes"/"No"` แต่ `model.classes_` เป็น `[0, 1]` | `TargetLabelHarmonizer` แปลงค่าตามแผนผัง Label Map ก่อนเข้า `confusion_matrix` โดยไม่โยน `ValueError` |
| **2** | **คอลัมน์ `Actual Target` มีค่าว่าง (`NaN`) ในบางแถว** | `test_evaluate_regression_sorts_worst_over_and_under_predictions_correctly()` | ใส่ `np.nan` ที่แถว `#1` ใน `y_true` | ระบบข้ามแถว `#1` และคืนดัชนีแถวจริง `[3, 2, 4]` ได้ตรงกับตำแหน่งเดิม |
| **3** | **ตัวแปรนำเข้ามีค่าคงที่ทั้งคอลัมน์ ($\text{SD} = 0$) ตอนคำนวณ Z-Score** | `test_ood_outlier_detector_handles_zero_variance_constant_column()` | กำหนดให้คอลัมน์ `X[:, 1] = 5.0` ทุกแถว | แทนค่า `std < 1e-8` ด้วย `1.0` ทำให้ไม่เกิด `NaN` หรือ `ZeroDivisionError` |
| **4** | **ช่องใน Confusion Matrix มี `0 แถว` แล้วผู้ใช้เผลอคลิก** | `InteractiveConfusionMatrix` empty cell test | คลิกช่องที่มี `row_count === 0` | ปุ่มของช่องนั้นมีสถานะ `disabled` และไม่ล้างคิวเดิม |
| **5** | **ค่า $R^2$ ติดลบมหาศาลเมื่อผู้ใช้เลือกคอลัมน์ `Actual Target` ผิด** | `test_severe_negative_r2_triggers_scale_mismatch_warning()` | ส่ง `y_true` ที่เป็นจำนวนห้องนอน `(1..5)` เทียบกับ `y_pred` ที่เป็นราคาบ้าน `(1M..10M)` | เมื่อ $R^2 < -1.0$ ระบบแสดงแถบเตือน `Scale Mismatch Warning` บน UI |
| **6** | **Multiclass ที่มีจำนวนคลาสมากเกินไป ($> 10$ คลาส)** | `test_multiclass_over_10_classes_switches_to_top_confused_pairs()` | จำลองผลทำนายที่มี 15 คลาส | ระบบส่งรายการ `Top 15 Most Confused Class Pairs` แทนการวาดตารางจัตุรัสขนาดใหญ่ |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Evaluation Latency SLA**: การคำนวณ Global Metrics และจัดอันดับ Error Slicer บนชุดข้อมูล `100,000 แถว` ต้องใช้เวลา `< 200 ms`
- [ ] **1-Click Queue Sync**: การคลิกปุ่ม Slicer หรือช่องใน Confusion Matrix ต้องอัปเดต `useTrajectoryQueueStore.selectedRowIndices` ทันทีใน `< 16 ms` (1 เฟรมหน้าจอ)
- [ ] **Unlabeled Resilience**: หากผู้ใช้ไม่ได้เลือกคอลัมน์ `Actual Target` ระบบยังคงแสดงผล `OOD Outliers (> 3 SD)` และการกระจายของค่าทำนายได้โดยไม่เกิด Error
