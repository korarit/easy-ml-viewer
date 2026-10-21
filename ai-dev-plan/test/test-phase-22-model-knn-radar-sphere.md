# Test Blueprint — Phase 22: K-Nearest Neighbors (KNN) Adapter & 3D Expanding Radar Sphere + Spider Lines

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-22-model-knn-radar-sphere.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-22-model-knn-radar-sphere.md) เพื่อยืนยันความถูกต้องของคลาส `KNNAdapter` (`backend/app/domains/model_engine/adapters/distance/knn.py`), คอมโพเนนต์เรดาร์และเส้นใยแมงมุม `KnnRadarActor3D` (`frontend/src/domains/stage3d/components/actors/KnnRadarActor3D.tsx`), และตารางผลโหวต `KnnNeighborTable` (`frontend/src/domains/hud/components/inspectors/KnnNeighborTable.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทุกตัว**:
  - `sklearn.neighbors.KNeighborsClassifier` (`weights='uniform'` และ `weights='distance'`)
  - `sklearn.neighbors.KNeighborsRegressor`
  - `sklearn.neighbors.RadiusNeighborsClassifier` (รวมถึงการตั้งค่า `outlier_label`)
  - `sklearn.neighbors.RadiusNeighborsRegressor`
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์และอัลกอริทึม**:
  - การดึงระยะห่างจริงและดัชนีเพื่อนบ้านผ่าน `model.kneighbors(X_batch, return_distance=True)`
  - รัศมีเรดาร์ $R_t = \max_{k \in \{1\dots K\}} d_{(k)}$
  - อัลกอริทึม **Mandatory Union + Stratified Downsampling (`MAX_BACKGROUND_REF_POINTS = 5000`)** ที่รับประกันว่าจุดเพื่อนบ้านทุกตัวที่ถูกเส้น Spider Lines โยงไปหาจะมีตัวตนอยู่ในฉาก 3 มิติครบ 100% เสมอ!
  - การเปลี่ยนรูปทรงเรดาร์ตาม Metric: `sphere` ($L_2$), `octahedron` ($L_1$ Manhattan), `cube` ($L_\infty$ Chebyshev)

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **100% Mandatory Neighbor Integrity in Reference Cloud**:
   - สำหรับทุกแถว $t$ ใน `X_batch` และทุกเพื่อนบ้าน $i \in \text{neighbor\_original\_indices}[t]$: ดัชนี $i$ **ต้องปรากฏอยู่ใน `reference_cloud["original_indices"]` และ `reference_cloud["index_map"]` ครบ 100%** แม้ว่า `model._fit_X` จะมีขนาดใหญ่กว่า `5,000` แถวก็ตาม!
2. **Radar Radius & Distance Parity**:
   - `extra_telemetry["radar_radii"][t]` ต้องเท่ากับ `np.max(neighbor_distances[t])` (สำหรับ `KNeighbors*`) หรือเท่ากับ `model.radius` (สำหรับ `RadiusNeighbors*`)
3. **Metric-Driven Radar Geometry Mapping**:
   - `metric='euclidean'` ($p=2$) $\to$ `"sphere"`
   - `metric='manhattan'` หรือ `p=1` $\to$ `"octahedron"`
   - `metric='chebyshev'` $\to$ `"cube"`

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_knn_adapter.py`)
```python
import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor
)
from app.domains.model_engine.adapters.distance.knn import KNNAdapter

FEATURE_NAMES = ["f0", "f1", "f2", "f3"]

@pytest.mark.parametrize("metric,p_val,expected_shape", [
    ("euclidean", 2, "sphere"),
    ("manhattan", 1, "octahedron"),
    ("minkowski", 1, "octahedron"),
    ("chebyshev", 2, "cube"),
])
def test_knn_metric_to_radar_geometry_mapping(metric, p_val, expected_shape):
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5, metric=metric, p=p_val).fit(X, y)
    adapter = KNNAdapter(model, FEATURE_NAMES)
    meta = adapter.extract_metadata()
    assert meta["radar_geometry_shape"] == expected_shape

def test_mandatory_union_downsampling_guarantees_100_percent_neighbor_presence():
    # Create dataset larger than MAX_BACKGROUND_REF_POINTS (8,000 rows > 5,000)
    rng = np.random.default_rng(42)
    X_large = rng.normal(size=(8000, 4)).astype(np.float32)
    y_large = rng.integers(0, 3, size=8000)

    model = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(X_large, y_large)
    adapter = KNNAdapter(model, FEATURE_NAMES)

    X_query = X_large[:50] + 0.05
    flow = adapter.trace_internal_flow(X_query)
    ref_cloud = flow.extra_telemetry["reference_cloud"]
    neigh_indices = np.asarray(flow.extra_telemetry["neighbor_original_indices"])

    # 1. Total reference cloud size is strictly capped at MAX_BACKGROUND_REF_POINTS (5000)
    assert len(ref_cloud["original_indices"]) == 5000
    assert len(ref_cloud["coordinates"]) == 5000

    # 2. Every single neighbor referenced by any frame in X_query MUST exist in index_map
    index_map = ref_cloud["index_map"]
    for row_neighbors in neigh_indices:
        for orig_idx in row_neighbors:
            assert int(orig_idx) in index_map
            loc_idx = index_map[int(orig_idx)]
            np.testing.assert_allclose(
                ref_cloud["coordinates"][loc_idx], X_large[orig_idx], atol=1e-5
            )

def test_knn_regressor_and_radius_neighbors_flow():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    knn_reg = KNeighborsRegressor(n_neighbors=4, weights="distance").fit(X, y)
    adapter = KNNAdapter(knn_reg, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:10])
    np.testing.assert_allclose(flow.predictions, knn_reg.predict(X[:10]), atol=1e-4)
    dists = np.asarray(flow.extra_telemetry["neighbor_distances"])
    radii = np.asarray(flow.extra_telemetry["radar_radii"])
    np.testing.assert_allclose(radii, np.max(dists, axis=1), atol=1e-5)

    # Test RadiusNeighborsRegressor
    rad_reg = RadiusNeighborsRegressor(radius=2.5).fit(X, y)
    rad_adapter = KNNAdapter(rad_reg, FEATURE_NAMES)
    rad_flow = rad_adapter.trace_internal_flow(X[:5])
    assert all(r == 2.5 for r in rad_flow.extra_telemetry["radar_radii"])
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `KnnRadarActor3D` และ `KnnNeighborTable`
1. **Dynamic Radar Shell Scaling & Shape Switching**:
   - เมื่อ `radar_geometry_shape === "octahedron"` คอมโพเนนต์ `KnnRadarActor3D` ต้องเรนเดอร์ `<octahedronGeometry>` แทน `<sphereGeometry>` และตั้งค่า `scale` ของโดมเรดาร์ให้เท่ากับรัศมี `radarRadius` ของเฟรมปัจจุบัน
2. **Spider Web Laser Lines & Distance-Weighted Opacity**:
   - เส้นเลเซอร์ทั้ง $K$ เส้นต้องลากจากพิกัดจุดปัจจุบัน $(x_t, y_t, z_t)$ ไปยังพิกัดของเพื่อนบ้านใน `reference_cloud.coordinates[index_map[orig_idx]]`
   - เมื่อ `weights === "distance"` ความหนาและความสว่างของเส้นเลเซอร์อันดับ `#1` ต้องมากกว่าอันดับ `#K` ตามสัดส่วน $w_{t,k} \propto \frac{1}{\max(d_k, 10^{-12})}$
3. **What-If Sandbox Real-Time Neighbor Re-binding**:
   - เมื่อลากลูกบอลเงา (Ghost Sphere) ในโหมด What-If เส้น Spider Lines ทั้ง $K$ เส้นต้องสลับไปจับเพื่อนบ้านชุดใหม่ที่อยู่ใกล้ตำแหน่งลูกบอลเงามากที่สุดทันที

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/actors/__tests__/KnnRadarActor3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { KnnNeighborTable, computeNeighborVoteWeights } from "@/domains/hud/components/inspectors/KnnNeighborTable";

describe("Phase 22: KnnNeighborTable & Distance Vote Weighting", () => {
  it("computes normalized inverse-distance weights safely even when distance is 0.0 (exact match)", () => {
    // First neighbor has distance 0.0 (exact match with training point)
    const dists = [0.0, 0.5, 1.0, 2.0, 4.0];
    const weights = computeNeighborVoteWeights(dists, "distance");
    expect(weights).toHaveLength(5);
    expect(weights.every((w) => Number.isFinite(w))).toBe(true);
    const sum = weights.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
    // Exact match neighbor must dominate (~100% weight)
    expect(weights[0]).toBeGreaterThan(0.999);
  });

  it("renders Neighbor Vote Tally Board with both 3D Projected Dist and Full D-Dim Dist", () => {
    render(
      <KnnNeighborTable
        frameIndex={58}
        radarRadius={1.428}
        weightsMode="distance"
        neighborIds={[412, 108, 889]}
        neighborLabels={[1, 1, 0]}
        fullDistances={[0.52, 0.78, 0.79]}
        projected3DDistances={[0.41, 0.62, 0.71]}
      />
    );

    expect(screen.getByText(/Row #412/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.520/)).toBeInTheDocument();
    expect(screen.getByText(/0\.410/)).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **`model._fit_X` Memory Cap Test (500,000 Rows Dataset)**:
   - ทดสอบจำลองโมเดล KNN ที่เทรนด้วยข้อมูลขนาดใหญ่ $500,000 \times 10$ แถว
   - ยืนยันว่า `get_downsampled_reference_cloud` คัดกรองจุดอ้างอิงลงเหลือ `5,000` จุดภายในเวลา `< 50 ms` และขนาด JSON ของ `reference_cloud` มีขนาดไม่เกิน `600 KB`
2. **Single Draw Call `THREE.InstancedMesh` Verification**:
   - ตรวจสอบใน `KnnRadarActor3D.tsx` ว่าจุดอ้างอิงทั้ง 5,000 จุดถูกเรนเดอร์ผ่าน `<instancedMesh args={[undefined, undefined, 5000]}>` เพียง 1 ตัว (`renderer.info.render.calls` สำหรับ Reference Cloud เท่ากับ `1`) และไฮไลต์เพื่อนบ้าน $K$ ตัวด้วยการอัปเดต `setColorAt` / `setMatrixAt` เฉพาะดัชนีที่เปลี่ยน

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **ระยะห่างใน $D$ มิติ vs ระยะห่างที่ฉายลงบน 3 มิติไม่เท่ากัน** | ทดสอบโมเดล 10 มิติโดยฉายลงแกน $(f_0, f_1)$ | ตารางใน HUD แสดงทั้ง `3D Dist` และ `Full D-Dist` ชัดเจนเพื่ออธิบายความต่างของโปรเจกชัน |
| 2 | **จุดทดสอบทับจุดเทรนสนิท ($d = 0.0$) เมื่อใช้ `weights='distance'`** | ส่ง `X_batch = model._fit_X[:5]` เข้าคำนวณน้ำหนักโหวต | ใช้ `1.0 / np.maximum(dists, 1e-12)` ไม่เกิด `ZeroDivisionError` และให้น้ำหนัก `1.0` แก่จุดที่ทับกันสนิท |
| 3 | **`RadiusNeighborsClassifier` ไม่พบเพื่อนบ้านเลยในรัศมี $R$** | ส่งจุด Outlier ที่อยู่ห่างออกไป `100.0` หน่วยเข้า `RadiusNeighborsClassifier(radius=1.0, outlier_label=-1)` | ไม่แครช คืนค่า `neighbor_original_indices = []` สำหรับแถวนั้น และแสดงป้ายเตือน `"OUTLIER: 0 Neighbors found within Radius R"` |
| 4 | **`model._fit_X` ถูกเก็บเป็น `scipy.sparse.csr_matrix`** | เทรน `KNeighborsClassifier(algorithm='brute')` บน `csr_matrix` | `KNNAdapter.__init__` ตรวจพบ `hasattr(model._fit_X, "toarray")` และแปลงอย่างปลอดภัย |
| 5 | **Metric เป็น `manhattan` ($L_1$) หรือ `chebyshev` ($L_\infty$)** | เทรนโมเดลด้วย `metric='manhattan'` และ `'chebyshev'` | `extract_metadata()["radar_geometry_shape"]` คืนค่า `"octahedron"` และ `"cube"` ตามลำดับ |
| 6 | **จำนวน $K$ สูงมาก (เช่น `n_neighbors = 150`)** | เทรน `KNeighborsClassifier(n_neighbors=150)` | ฉาก 3D จำกัดการวาดเส้นเลเซอร์ Spider Lines เฉพาะ `Top-15 Closest Neighbors` เพื่อไม่ให้ฉากรกและรักษาความลื่นไหล 60FPS |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **100% Neighbor Integrity Gate**: ดัชนีเพื่อนบ้านทุกตัวใน `neighbor_original_indices` ต้องมีพิกัดอยู่ใน `reference_cloud` ครบ 100% เสมอ ไม่ว่า `_fit_X` จะใหญ่แค่ไหน
- [ ] **WebGL Draw Call SLA**: การเรนเดอร์จุดอ้างอิง 5,000 จุดต้องใช้ `InstancedMesh` เพียง `1 Draw Call` และรักษาเฟรมเรต `60 FPS`
- [ ] **Zero-Division Gate**: กรณี $d = 0.0$ ใน `weights='distance'` ต้องไม่เกิด `NaN` หรือ `Infinity`
- [ ] **Coverage Gate**: Code Coverage ของ `knn.py`, `KnnRadarActor3D.tsx`, และ `KnnNeighborTable.tsx` ไม่ต่ำกว่า `95%`
