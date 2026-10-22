# Test Phase 44: 2D Decision Geometries & Visualizers for Probabilistic & Discriminant Models

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 44)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-44-model-group-probabilistic-discriminant-2d-geometry.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-44-model-group-probabilistic-discriminant-2d-geometry.md) เพื่อยืนยันความถูกต้องทางคณิตศาสตร์สถิติ 100% สำหรับกลุ่มโมเดล Probabilistic & Discriminant (**GaussianNB, MultinomialNB, BernoulliNB, LDA, QDA**) ครอบคลุมการคำนวณจุดศูนย์กลางค่าเฉลี่ย $(\mu_1, \mu_2)$, ขนาดรัศมีกึ่งแกนของวงรี $1\sigma, 2\sigma, 3\sigma$, ความตรงของเส้นแบ่งเขต LDA, ความโค้งภาคตัดกรวยของ QDA จากเมทริกซ์ $\boldsymbol{\Sigma}_c$, และการวาด SVG คอมโพเนนต์ใน Frontend

### ขอบเขตโมเดลที่ถูกทดสอบในเฟสนี้
- `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Mathematical Parity Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_44/
├── conftest.py
├── test_gnb_bivariate_ellipses_math.py
├── test_lda_linear_line_parity.py
└── test_qda_conic_section_curvature.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_44/test_gnb_bivariate_ellipses_math.py
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from app.domains.model_engine.adapters.probabilistic.probabilistic_2d_extractor import Probabilistic2DExtractor

def test_gnb_ellipses_radii_match_feature_variances() -> None:
    """ทดสอบว่ารัศมีกึ่งแกน rx และ ry ของวงรี GNB เท่ากับ sqrt(var_[c, j]) เป๊ะ"""
    X = np.array([
        [1.0, 10.0],
        [2.0, 12.0],
        [3.0, 14.0],
        [8.0, 50.0],
        [9.0, 52.0]
    ], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1], dtype=np.int32)

    gnb = GaussianNB()
    gnb.fit(X, y)

    res = Probabilistic2DExtractor.extract_gnb_ellipses(gnb, x_idx=0, y_idx=1)
    classes = res["classes"]
    assert len(classes) == 2

    for cls in classes:
        c_idx = cls["class_index"]
        expected_std_x = np.sqrt(gnb.var_[c_idx, 0])
        expected_std_y = np.sqrt(gnb.var_[c_idx, 1])
        assert abs(cls["std_x"] - expected_std_x) < 1e-5
        assert abs(cls["std_y"] - expected_std_y) < 1e-5

def test_lda_produces_linear_boundary_and_qda_produces_curve() -> None:
    """ทดสอบว่าความชันของเส้นแบ่งเขต LDA สม่ำเสมอคงที่ (เส้นตรง) ขณะที่ QDA มีพจน์กำลังสอง"""
    X = np.array([[1, 2], [2, 1], [6, 8], [8, 6]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    lda = LinearDiscriminantAnalysis().fit(X, y)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)

    lda_res = Probabilistic2DExtractor.extract_discriminant_2d(lda, x_idx=0, y_idx=1)
    qda_res = Probabilistic2DExtractor.extract_discriminant_2d(qda, x_idx=0, y_idx=1)

    assert lda_res["is_linear"] is True
    assert qda_res["is_linear"] is False
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/GaussianEllipses2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { GaussianEllipses2D } from '../../components/surfaces_2d/GaussianEllipses2D';

describe('GaussianEllipses2D SVG Rendering', () => {
  it('renders 3 concentric ellipses per class', () => {
    const classData = [
      { classIndex: 0, meanX: 5.0, meanY: 3.0, stdX: 0.5, stdY: 1.0, color: '#38bdf8' },
      { classIndex: 1, meanX: 7.0, meanY: 4.0, stdX: 0.8, stdY: 1.2, color: '#fb923c' },
    ];

    const { container } = render(
      <svg>
        <GaussianEllipses2D classes={classData} visible={true} />
      </svg>
    );

    // 2 คลาส x 3 วงรี = 6 วงรี
    const ellipses = container.querySelectorAll('ellipse');
    expect(ellipses.length).toBe(6);
  });
});
```

---

## 4. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E44.1** | ฟีเจอร์ในคลาสหนึ่งมีค่าเท่ากันหมด ($\sigma^2 = 0$) | เกิด Division by Zero ใน Gaussian Density ($1/\sigma$) | ใช้ `var_smoothing` ($10^{-9} \times \max(\text{var})$) ตาม Scikit-Learn |
| **E44.2** | เมทริกซ์ความแปรปรวนร่วม $\boldsymbol{\Sigma}_c$ ใน QDA เป็น Singular Matrix | ไม่สามารถหาอินเวิร์ส $\boldsymbol{\Sigma}_c^{-1}$ ได้ | เสริม Ridge Diagonal Regularization $\boldsymbol{\Sigma}_c + 10^{-6}\mathbf{I}$ |
| **E44.3** | คลาสมีค่าเฉลี่ย $(\mu_1, \mu_2)$ อยู่ที่จุดเดียวกันเป๊ะ แต่ขนาด $\boldsymbol{\Sigma}$ ต่างกัน | ขอบเขต QDA เป็นวงกลม/วงรีปิดรอบจุดศูนย์กลาง | วาดวงรีซ้อนวงรีและสกัดเส้นแบ่งเขตแบบวงปิดได้อย่างถูกต้อง |
| **E44.4** | ข้อมูลมีมากกว่า 10 คลาสในโมเดล GaussianNB | วงรี 30 วงซ้อนทับกันจนภาพรกสายตา | มีตัวเลือก "[x] Show Active Class Ellipses Only" ให้เลือกดูเฉพาะคลาสที่สนใจ |
| **E44.5** | ข้อมูล Multinomial NB มีคำที่ไม่เคยพบใน Training Set ($x_j = 0$) | คะแนน Log-Likelihood ติดลบอนันต์ ($-\infty$) | ใช้ Laplace Smoothing ($+1$) ในการแสดงผลค่าคะแนน |
| **E44.6** | สลับดูระนาบ 2D ของคู่ฟีเจอร์ที่ไม่มีความสัมพันธ์กับเป้าหมาย | วงรีของทุกคลาสทับซ้อนกันสนิท | แสดงค่า Overlap Index แจ้งเตือนว่าคู่ฟีเจอร์นี้แยกคลาสไม่ได้ |

---

## 5. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit Test ฝั่ง Backend ใน `backend/tests/phase_44/` ผ่าน 100%
- [ ] ชุดทดสอบ Component Test ใน `frontend/src/domains/stage3d/__tests__/surfaces_2d/GaussianEllipses2D.test.tsx` ผ่าน 100%
- [ ] วงรีความน่าจะเป็นและเส้นแบ่งเขต LDA/QDA แสดงผลคมชัดและถูกต้องตามหลักสถิติ
- [ ] ผ่านเกณฑ์ Edge Case Matrix ทั้ง 6 กรณีครบถ้วน

