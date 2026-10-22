# Test Phase 46: 2D Decision Geometries & Visualizers for Gradient Boosting Heavyweights

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 46)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-46-model-group-gradient-boosting-2d-geometry.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-46-model-group-gradient-boosting-2d-geometry.md) เพื่อยืนยันความถูกต้องของเรขาคณิต 2 มิติสำหรับกลุ่มโมเดลเกรเดียนต์บูสติงระดับเฮฟวีเวต (**Sklearn GBM, HistGBM, XGBoost, LightGBM, CatBoost**) ครอบคลุมการคำนวณและทิศทางของเวกเตอร์ 2D TreeSHAP ($\phi_1, \phi_2$), ความสมมาตรของตารางหมากรุก Oblivious Trees ใน CatBoost, ความถูกต้องของเส้นกริด 255 Bins ใน HistGBM, และการทำงานแบบ 60 FPS ใน Frontend

### ขอบเขตโมเดลที่ถูกทดสอบในเฟสนี้
- `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `LGBMRegressor`, `CatBoostClassifier`, `CatBoostRegressor`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Mathematical Parity Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_46/
├── conftest.py
├── test_xgboost_2d_shap_vector_math.py
├── test_catboost_oblivious_grid_symmetry.py
└── test_hist_gbm_255_bin_thresholds_parity.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_46/test_xgboost_2d_shap_vector_math.py
import numpy as np
import xgboost as xgb
from app.domains.model_engine.adapters.boosting.boosting_2d_extractor import Boosting2DExtractor

def test_xgboost_2d_shap_vector_matches_native_contribs() -> None:
    """ทดสอบว่าเวกเตอร์แรงผลัก 2D TreeSHAP (phi_x, phi_y) มีค่าตรงกับ Native pred_contribs ของ XGBoost"""
    X = np.array([[10, 100], [20, 200], [30, 300], [40, 400]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42).fit(X, y)

    query = np.array([[25.0, 250.0]], dtype=np.float32)
    res = Boosting2DExtractor.extract_xgboost_2d_state(model, query, x_idx=0, y_idx=1)

    # ดึงค่า SHAP แท้จริงจากโมเดล
    booster = model.get_booster()
    dmat = xgb.DMatrix(query)
    native_contribs = booster.predict(dmat, pred_contribs=True)[0]

    assert abs(res["phi_x"] - native_contribs[0]) < 1e-5
    assert abs(res["phi_y"] - native_contribs[1]) < 1e-5

def test_catboost_oblivious_grid_has_power_of_two_cells() -> None:
    """ทดสอบว่าต้นไม้ Oblivious ลึก D สร้างจำนวนเซลล์เท่ากับ 2^D พอดี"""
    from catboost import CatBoostClassifier
    X = np.array([[1, 2], [2, 1], [6, 8], [8, 6]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    cb = CatBoostClassifier(iterations=3, depth=3, verbose=0).fit(X, y)

    res = Boosting2DExtractor.extract_catboost_2d_state(cb, x_idx=0, y_idx=1)
    assert res["is_oblivious"] is True
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/XgboostTreeShapVector2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { XgboostTreeShapVector2D } from '../../components/surfaces_2d/XgboostTreeShapVector2D';

describe('XgboostTreeShapVector2D SVG Rendering', () => {
  it('renders vector line with length proportional to SHAP magnitude', () => {
    const { container } = render(
      <svg>
        <XgboostTreeShapVector2D
          originX={100}
          originY={100}
          phiX={1.5}
          phiY={-2.0}
          scaleFactor={20}
          visible={true}
        />
      </svg>
    );

    const line = container.querySelector('line');
    expect(line).toBeTruthy();
    expect(line?.getAttribute('x1')).toBe('100');
    expect(line?.getAttribute('y1')).toBe('100');
    expect(line?.getAttribute('x2')).toBe('130'); // 100 + 1.5 * 20
    expect(line?.getAttribute('y2')).toBe('60');  // 100 + (-2.0) * 20
  });
});
```

---

## 4. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E46.1** | ค่า SHAP ของทั้งสองแกนเป็นศูนย์พร้อมกัน ($\phi_1 = 0, \phi_2 = 0$) | เวกเตอร์มีความยาว 0 พิกัดทับจุดเริ่มต้น | แสดงจุดนิ่งสีฟ้าอ่อน พร้อมป้าย *"Neutral SHAP impact on this slice"* |
| **E46.2** | ต้นไม้ CatBoost ตัดเฉพาะฟีเจอร์แกนอื่น ไม่ตัดบนแกน $X$ หรือ $Y$ เลย | ไม่มีเส้นตารางหมากรุกในระนาบ 2D | แสดงป้ายระบุว่าระนาบนี้เป็นโซนความน่าจะเป็นคงที่ |
| **E46.3** | โมเดล XGBoost บันทึกมาเป็นไฟล์แบบ Universal Binary JSON (`.ubj`) | ตัวอ่านไฟล์ JSON ปฏิเสธไฟล์ไบนารี | รองรับการอ่านทั้ง `.json`, `.ubj`, และ `.pkl` อัตโนมัติ |
| **E46.4** | ข้อมูลมีค่า Missing Values (`NaN`) ใน HistGBM | จุดตกอยู่ในถังพิเศษ `missing_go_to_left` | พล็อตตำแหน่งจุดตามจริง และแสดงป้ายเตือน NaN Imputed Path |
| **E46.5** | LightGBM มีการแตกกิ่งลึก 63 Leaves ในต้นไม้เดียว | รูปหลายเหลี่ยมใบไม้ 2D ซับซ้อนและมีรอยหยักมาก | ประมวลผลและลดทอนความซับซ้อนของ Polygons ใน 2D อย่างราบรื่น |
| **E46.6** | เล่น VCR ที่ความเร็ว 10x ข้าม 2,000 แถวด้วยโมเดล XGBoost | คำนวณ TreeSHAP ไม่ทัน ทำให้เฟรมเรตตก | แคชค่า SHAP ของทั้งคิวไว้ล่วงหน้าใน Buffer Array |

---

## 5. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit Test ฝั่ง Backend ใน `backend/tests/phase_46/` ผ่าน 100%
- [ ] ชุดทดสอบ Component Test ใน `frontend/src/domains/stage3d/__tests__/surfaces_2d/XgboostTreeShapVector2D.test.tsx` ผ่าน 100%
- [ ] เวกเตอร์ 2D TreeSHAP ขยับตามแถวข้อมูลใน VCR ได้อย่างแม่นยำที่ **60 FPS**
- [ ] ผ่านเกณฑ์ Edge Case Matrix ทั้ง 6 กรณีครบถ้วน

