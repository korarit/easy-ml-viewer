# Test Phase 45: 2D Decision Geometries & Visualizers for Decision Trees & Bagging Ensembles

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 45)
แผนการทดสอบนี้สร้างขึ้นคู่ขนานแบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-45-model-group-trees-ensembles-2d-geometry.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-45-model-group-trees-ensembles-2d-geometry.md) เพื่อยืนยันความถูกต้องของการตัดแบ่งพื้นที่ตั้งฉากแกนพิกัด (**2D Axis-Aligned Partitions**), ความถูกต้องของพิกัดกล่องใบไม้ (**Active Leaf Bounding Box**), การรวมคะแนนเสียงโหวตของต้นไม้ทั้งป่าใน Random Forest & Extra Trees, และการทำงานของสไลเดอร์ AdaBoost Cascade พร้อมการเรนเดอร์คอมโพเนนต์ SVG ใน Frontend ที่ 60 FPS

### ขอบเขตโมเดลที่ถูกทดสอบในเฟสนี้
- `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor`

---

## 2. รายการทดสอบฝั่ง Backend Unit & Mathematical Parity Tests (`pytest`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_45/
├── conftest.py
├── test_tree_split_nodes_extraction.py
├── test_leaf_bounding_box_geometry.py
└── test_rf_consensus_aggregation_parity.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest`)
```python
# backend/tests/phase_45/test_leaf_bounding_box_geometry.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor

def test_active_leaf_bounding_box_contains_query_point() -> None:
    """ทดสอบว่าพิกัดของกล่อง Active Leaf Bounding Box ครอบคลุมพิกัดของจุด Query เสมอ"""
    X = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
        [40.0, 400.0]
    ], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
    query_point = np.array([15.0, 150.0], dtype=np.float32)

    leaf_box = Tree2DExtractor.compute_active_leaf_box(
        tree, query_point, x_idx=0, y_idx=1, x_range=(0.0, 50.0), y_range=(50.0, 500.0)
    )

    assert leaf_box["x_min"] <= query_point[0] <= leaf_box["x_max"]
    assert leaf_box["y_min"] <= query_point[1] <= leaf_box["y_max"]
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component Tests (`vitest` & RTL)

```tsx
// frontend/src/domains/stage3d/__tests__/surfaces_2d/TreePartitions2D.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { TreePartitions2D } from '../../components/surfaces_2d/TreePartitions2D';

describe('TreePartitions2D Component Rendering', () => {
  it('renders dashed lines for relevant axis splits and highlights active leaf rect', () => {
    const splits = [
      { featureIndex: 0, threshold: 25.0, relation: '<=' as const },
      { featureIndex: 1, threshold: 140.0, relation: '<=' as const },
    ];

    const { container } = render(
      <svg>
        <TreePartitions2D
          splitRules={splits}
          xFeatureIndex={0}
          yFeatureIndex={1}
          xRange={[0, 100]}
          yRange={[0, 200]}
          activeLeafBox={{ xMin: 0, xMax: 25, yMin: 0, yMax: 140 }}
          visible={true}
        />
      </svg>
    );

    expect(container.querySelectorAll('line').length).toBe(2);
    const rect = container.querySelector('rect');
    expect(rect).toBeTruthy();
    expect(rect?.getAttribute('width')).toBe('25');
    expect(rect?.getAttribute('height')).toBe('140');
  });
});
```

---

## 4. Adversarial & Edge Case Test Matrix (ตารางทดสอบกรณีขอบ 6 เคส)

| รหัสเคส | สถานการณ์กรณีขอบ (Edge Case Scenario) | ความเสี่ยงที่อาจเกิดขึ้น | พฤติกรรมที่ถูกต้องของระบบ (Expected Handling) |
| :---: | :--- | :--- | :--- |
| **E45.1** | ต้นไม้ลึกมาก (Depth > 20) ทำให้มีเส้นตัดหลายพันเส้นใน 2D | SVG ล้นจอ หรือบราวเซอร์ Render กระตุก | รวบรวมเส้นผ่าน Polyline Batching หรือแสดงเฉพาะ Top-20 โหนดหลัก |
| **E45.2** | โหนดแรกตัดข้อมูลออกเป็น Leaf ทันที (Tree ลึก 1 ชั้น - Stump) | มีเส้นตัดเพียงเส้นเดียวบนจอ | เรนเดอร์เส้นเดียวอย่างคมชัด และแบ่งครึ่งจอออกเป็น 2 โซนชัดเจน |
| **E45.3** | Random Forest มีต้นไม้ 500 ต้น | คำนวณเสียงโหวตช้าเกินพิกัด | ใช้ Batch Vectorized Inference คำนวณเสร็จภายใน $< 50\text{ ms}$ |
| **E45.4** | จุด Query อยู่ตรงเส้นแบ่ง Threshold พอดีเป๊ะ ($x_j = \theta$) | เกิดความกำกวมในการเลือกกล่อง Leaf | ใช้กฎ `<=` ตามสัญญาของ Scikit-Learn (เลือกกิ่งซ้าย) อย่างเคร่งครัด |
| **E45.5** | ต้นไม้ไม่มีการตัดบนแกน $X$ หรือ $Y$ เลย (ตัดบนแกนอื่นหมด) | ไม่มีเส้นตัดใดๆ ปรากฏในระนาบ 2D | แสดงป้าย *"Splits occur on non-visualized axes; uniform prediction on this slice"* |
| **E45.6** | สไลเดอร์ AdaBoost ถูกลากไปที่ $T=0$ | ไม่มี Weak Learner ให้แสดงผล | แสดงขอบเขตเริ่มต้น Base Prediction ($P = 0.5$) อย่างปลอดภัย |

---

## 5. เกณฑ์การผ่านการทดสอบระดับเฟส (Phase Acceptance Criteria & Test Gates)
- [ ] ชุดทดสอบ Unit Test ฝั่ง Backend ใน `backend/tests/phase_45/` ผ่าน 100%
- [ ] ชุดทดสอบ Component Test ใน `frontend/src/domains/stage3d/__tests__/surfaces_2d/TreePartitions2D.test.tsx` ผ่าน 100%
- [ ] พิกัดกล่อง Active Leaf Bounding Box ครอบคลุมจุด Query ได้อย่างถูกต้องทุกเฟรม
- [ ] ผ่านเกณฑ์ Edge Case Matrix ทั้ง 6 กรณีครบถ้วน

