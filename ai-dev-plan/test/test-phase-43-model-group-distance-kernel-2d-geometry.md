# Test Phase 43: 2D Decision Geometries & Visualizers for Distance & Kernel Models

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 43)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-43-model-group-distance-kernel-2d-geometry.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-43-model-group-distance-kernel-2d-geometry.md) เพื่อยืนยันความถูกต้องทางคณิตศาสตร์ 100% สำหรับกลุ่มโมเดล Distance-Based & Kernel-Based Learning (**KNN & SVM**) ครอบคลุมการคำนวณรัศมีวงกลมเรดาร์ 2D ($R_{t, \text{2D}} = \max d_{\text{2D}}$), พิกัดและการลากเส้นใยแสงเลเซอร์เชื่อมเพื่อนบ้าน $K$ ตัว, ความต่อเนื่องของเส้นโค้ง RBF Decision Boundary, แถบขอบเขตมาร์จิน ($f = \pm 1$), และการตรวจจับจุดที่เป็น Support Vectors ในอวกาศข้อมูล

### ขอบเขตโมเดลที่ถูกทดสอบในเฟสนี้
- `KNeighborsClassifier`, `KNeighborsRegressor`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`, `SVC`, `SVR`, `LinearSVC`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Mathematical Parity Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_43/
├── conftest.py
├── test_knn_radar_circle_math.py
├── test_knn_spider_lines_coordinates.py
├── test_svm_rbf_boundary_contour_math.py
└── test_svm_support_vector_halos_extraction.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_43/test_knn_radar_circle_math.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from app.domains.model_engine.adapters.distance.distance_2d_extractor import Distance2DExtractor

def test_knn_2d_radar_radius_contains_all_k_neighbors() -> None:
    """ทดสอบว่าวงกลมเรดาร์ 2D มีรัศมีที่ครอบคลุมเพื่อนบ้านทั้ง K ตัวบนระนาบ 2D อย่างสมบูรณ์"""
    X_train = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [3.0, 4.0, 1.0],
        [5.0, 5.0, 1.0]
    ], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1, 1], dtype=np.int32)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    query = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    res = Distance2DExtractor.extract_knn_2d_state(knn, query, x_idx=0, y_idx=1)

    radar_radius = res["radar_radius_2d"]
    neighbors_2d = res["neighbors_2d"]

    assert len(neighbors_2d) == 3
    # ทุกจุดเพื่อนบ้านต้องอยู่ในระยะรัศมีวงกลม (ระยะทาง 2D <= radar_radius + tolerance)
    for nb in neighbors_2d:
        dist_2d = np.hypot(nb["x"] - query[0, 0], nb["y"] - query[0, 1])
        assert dist_2d <= radar_radius + 1e-5

def test_svm_support_vectors_count_and_weights_parity() -> None:
    """ทดสอบว่าจำนวน Support Vectors และค่าน้ำหนัก dual_coef_ ถูกสกัดออกมาตรงกับโมเดล 100%"""
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y)

    res = Distance2DExtractor.extract_svm_2d_state(svm, x_idx=0, y_idx=1)
    assert len(res["support_vectors_2d"]) == len(svm.support_vectors_)
    assert len(res["dual_coefs"]) == len(svm.dual_coef_[0])
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/KnnRadar2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { KnnRadar2D } from '../../components/surfaces_2d/KnnRadar2D';

describe('KnnRadar2D SVG Geometry Rendering', () => {
  it('renders radar circle and K lines connecting to neighbors', () => {
    const neighbors = [
      { refIndex: 1, x: 10, y: 0, distance2D: 10, voteWeight: 0.6, targetClass: 0 },
      { refIndex: 2, x: 0, y: 15, distance2D: 15, voteWeight: 0.4, targetClass: 1 },
    ];

    const { container } = render(
      <svg>
        <KnnRadar2D
          centerX={0}
          centerY={0}
          radarRadius2D={15}
          neighbors={neighbors}
          visible={true}
        />
      </svg>
    );

    // วงกลมเรดาร์ 1 วง + วงกลมจุดเพื่อนบ้าน 2 วง = 3 วง
    const circles = container.querySelectorAll('circle');
    expect(circles.length).toBe(3);

    // เส้นใยเลเซอร์ 2 เส้น
    const lines = container.querySelectorAll('line');
    expect(lines.length).toBe(2);
  });
});
```

---

## 4. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E43.1** | จุดข้อมูลอยู่ทับกับจุดอ้างอิงเพื่อนบ้านเป๊ะ ($d = 0$) | การคำนวณน้ำหนัก $1/d$ เกิด Division by Zero | ป้องกันด้วยตัวแปรป้องกัน $\epsilon = 10^{-12}$ |
| **E43.2** | ค่า $K = 1$ ในโมเดล KNN | มีเพื่อนบ้านเพียงตัวเดียว วงกลมมีขนาดเท่าระยะเพื่อนบ้านตัวนั้น | วาดเส้นเลเซอร์เส้นเดียว และแสดงผลโหวต 100% ชัดเจน |
| **E43.3** | โมเดล SVC มีค่า Gamma สูงมากจนเกิด Overfitting รุนแรง | ขอบเขต RBF แตกเป็นเกาะย่อยๆ เล็กๆ ทั่วระนาบ | Marching Squares ตรวจจับและสกัดเกาะรอบจุดได้อย่างอิสระ |
| **E43.4** | จุดเพื่อนบ้าน $K$ ตัวอยู่ในระนาบมิติอื่น แต่ในระนาบ 2D ทับกัน | เส้นเลเซอร์ใน 2D ทับกันมองไม่เห็น | จัดการลำดับ Z-Index และกระจายป้ายข้อความไม่ให้ทับกัน |
| **E43.5** | โมเดล Support Vector Machines มี Support Vectors มากกว่า 5,000 จุด | การวาดวงแหวนเพชร 5,000 วงทำให้ SVG กระตุก | สลับไปวาดด้วย `THREE.InstancedMesh` ใน 1 Draw Call อัตโนมัติ |
| **E43.6** | ใช้เคอร์เนล `poly` ดีกรีสูง ($d=5$) | การคำนวณเคอร์เนลตัวเลขพุ่งสูงทะลุ Frustum | ปรับสเกลให้อยู่ในช่วงมาตรฐาน `[-5, +5]` |

---

## 5. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit Test ฝั่ง Backend ใน `backend/tests/phase_43/` ผ่าน 100%
- [ ] ชุดทดสอบ Component Test ใน `frontend/src/domains/stage3d/__tests__/surfaces_2d/KnnRadar2D.test.tsx` ผ่าน 100%
- [ ] วงกลมเรดาร์ 2D และเส้นใยแมงมุมขยับตามการเล่นวิดีโอได้อย่างลื่นไหลที่ **60 FPS**
- [ ] ผ่านเกณฑ์ Edge Case Matrix ทั้ง 6 กรณีครบถ้วน

