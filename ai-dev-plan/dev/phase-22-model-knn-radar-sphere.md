# Phase 22: K-Nearest Neighbors (KNN) Adapter & 3D Expanding Radar Sphere + Spider Lines

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลตระกูล **Instance-Based / Distance-Based Learning** อย่าง **K-Nearest Neighbors (KNN)** ซึ่งไม่ได้ใช้สมการน้ำหนักตายตัว แต่ตัดสินใจจากการค้นหาเพื่อนบ้านที่อยู่ใกล้ที่สุด $K$ จุดในอวกาศข้อมูล โดยแสดงผลเป็น **ทรงกลมเรดาร์ 3 มิติที่ขยายตัวไปแตะเพื่อนบ้านตัวที่ $K$ (3D Expanding Radar Sphere / Octahedron)** พร้อม **เส้นใยแสงเลเซอร์ (Spider Web Laser Lines)** ที่พุ่งไปเชื่อมกับจุดเพื่อนบ้านทั้ง $K$ ตัวในทุกๆ เฟรมที่เล่นผ่าน

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.neighbors.KNeighborsClassifier`
  - `sklearn.neighbors.KNeighborsRegressor`
  - `sklearn.neighbors.RadiusNeighborsClassifier`
  - `sklearn.neighbors.RadiusNeighborsRegressor`
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary & Multiclass Classification**: โหวตเสียงข้างมากแบบ `weights='uniform'` หรือถ่วงน้ำหนักด้วยระยะทาง `weights='distance'`
  - **Single-target & Multi-output Regression**: เฉลี่ยค่าของเพื่อนบ้าน $K$ ตัว

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการระยะทาง (Minkowski / Custom Metrics) และการลงคะแนนของเพื่อนบ้าน
กำหนดให้ชุดข้อมูลอ้างอิงที่โมเดลจดจำไว้คือ $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_{(1)}, y_{(1)}), \dots, (\mathbf{x}_{(M)}, y_{(M)})\}$ โดยเก็บอยู่ใน `model._fit_X` ขนาด $(M, D)$:

1. **การคำนวณระยะทางในอวกาศ $D$ มิติ**:
   - สำหรับจุดข้อมูลปัจจุบัน $\mathbf{x}_t \in \mathbb{R}^D$ และจุดอ้างอิง $\mathbf{x}_{(m)} \in \mathbb{R}^D$:
     $$d_p(\mathbf{x}_t, \mathbf{x}_{(m)}) = \left( \sum_{j=1}^{D} |x_{t,j} - x_{(m),j}|^p \right)^{1/p}$$
     - $p = 2$ (`euclidean` / `l2`): ทรงกลมรัศมีเท่ากันใน 3 มิติมีรูปร่างเป็น **Sphere (ทรงกลม)**
     - $p = 1$ (`manhattan` / `cityblock` / `l1`): ทรงกลมรัศมีเท่ากันใน 3 มิติมีรูปร่างเป็น **Regular Octahedron (ทรงแปดหน้า / เพชร)**
     - $p = \infty$ (`chebyshev`): รูปร่างเป็น **Cube (ลูกบาศก์)**
2. **เซตเพื่อนบ้าน $K$ อันดับแรก $\mathcal{N}_K(\mathbf{x}_t)$ และรัศมีเรดาร์ $R_t$**:
   - เรียงลำดับระยะทาง $d_{(1)} \le d_{(2)} \le \dots \le d_{(K)}$ พร้อมดัชนีจุดอ้างอิง $[i_1, i_2, \dots, i_K]$
   - **รัศมีทรงกลมเรดาร์ของเฟรม $t$**: $R_t = d_{(K)} = \max_{m \in \mathcal{N}_K(\mathbf{x}_t)} d(\mathbf{x}_t, \mathbf{x}_{(m)})$ (หรือ $R_t = \text{model.radius}$ สำหรับ `RadiusNeighbors*`)
3. **น้ำหนักการโหวตของเพื่อนบ้านแต่ละตัว ($w_{t,k}$)**:
   - กรณี `weights='uniform'`: $w_{t,k} = \frac{1}{K}$
   - กรณี `weights='distance'`: $w_{t,k} = \frac{1 / \max(d_{(k)}, \epsilon)}{\sum_{r=1}^{K} 1 / \max(d_{(r)}, \epsilon)}$ (โดย $\epsilon = 10^{-12}$ ป้องกันการหารด้วยศูนย์เมื่อจุดทับกันสนิท)

### 2.2 โครงสร้าง Array Shape และการคำนวณ Local Feature Contribution (Distance Waterfall)
- `kneighbors(X_batch, return_distance=True)` คืนค่า:
  - `distances`: ขนาด $(N, K)$
  - `indices`: ขนาด $(N, K)$ อ้างอิงแถวใน `model._fit_X`
- **การคำนวณ Local Feature Contribution (Per-Feature Squared Distance Breakdown)**:
  - ในโมเดล KNN คำถามสำคัญคือ *"ฟีเจอร์ตัวไหนที่ทำให้จุดปัจจุบันอยู่ใกล้หรือไกลจากกลุ่มเพื่อนบ้านที่ชนะ?"*
  - สำหรับแถว $t$ คำนวณค่าเฉลี่ยจุดศูนย์ถ่วงของเพื่อนบ้านทั้ง $K$ ตัว: $\bar{\mathbf{x}}_{\mathcal{N}_K} = \sum_{k=1}^{K} w_{t,k} \mathbf{x}_{(i_k)}$
  - ความห่างรายฟีเจอร์จากจุดศูนย์ถ่วงเพื่อนบ้าน: $\phi_{t,j} = -|x_{t,j} - \bar{x}_{\mathcal{N}_K, j}|$ (ยิ่งใกล้ศูนย์ แปลว่าฟีเจอร์ $j$ ยิ่งตรงกับคุณลักษณะของกลุ่มเพื่อนบ้านที่ชนะ!)

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **Training Reference Point Cloud (`THREE.InstancedMesh`)**:
   - แสดงกลุ่มจุดข้อมูลอ้างอิงจาก `_fit_X` ลอยอยู่ในอวกาศ 3D โดยแบ่งสีตามคลาสหรือค่าเป้าหมายจริง (`opacity: 0.35` ในสภาวะปกติเพื่อไม่ให้บดบังสายตา)
2. **3D Expanding Radar Bubble (`SphereGeometry` / `OctahedronGeometry` / `BoxGeometry`)**:
   - ล้อมรอบลูกบอลข้อมูลของเฟรมปัจจุบัน $t$ ด้วยโดมแก้วโปร่งแสง (`transmission: 0.85, roughness: 0.1, wireframe overlay`)
   - รูปทรงของโดมเรดาร์เปลี่ยนไปตาม `effective_metric_` ของโมเดลอัตโนมัติ:
     - Euclidean ($L_2$) $\to$ ทรงกลมเรดาร์ (`SphereGeometry`)
     - Manhattan ($L_1$) $\to$ ทรงเพชรแปดหน้า (`OctahedronGeometry`)
     - Chebyshev ($L_\infty$) $\to$ กล่องลูกบาศก์เรดาร์ (`BoxGeometry`)
   - รัศมีของโดมขยายหรือหดตัวแบบเรียลไทม์ให้ไปแตะเพื่อนบ้านตัวที่ $K$ พอดี!
3. **Spider Web Laser Lines & Active Neighbor Halo**:
   - ยิงเส้นแสงเลเซอร์ (`drei/Line` หรือ `THREE.LineSegments`) จากจุดข้อมูลปัจจุบัน $\mathbf{x}_t$ พุ่งไปยังพิกัดของเพื่อนบ้านทั้ง $K$ ตัวโดยตรง
   - ความหนาและความสว่างของเส้นเลเซอร์แปรผันตรงกับน้ำหนักการโหวต $w_{t,k}$ (เพื่อนบ้านที่อยู่ใกล้กว่าจะมีเส้นเลเซอร์สว่างและหนากว่าเมื่อตั้งค่า `weights='distance'`)
   - จุดเพื่อนบ้านทั้ง $K$ ตัวที่ถูกเลือกจะสว่างวาบขึ้น (`opacity: 1.0`, ขยายขนาด $1.6\times$) พร้อมป้ายลำดับ `#1`, `#2`, ..., `#K`

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Neighbor Vote Tally Board**: ตารางสรุปผลโหวตของเพื่อนบ้านทั้ง $K$ ตัว เช่น `Class A: 4 votes (82.4% weight) vs Class B: 1 vote (17.6% weight)`
- **Radar Shell & Metric Badge**: แสดงค่า `n_neighbors (K)`, `weights` (`uniform` / `distance`), `algorithm` (`kd_tree`, `ball_tree`, `brute`), และ `effective_metric_`
- **Background Reference Cloud Opacity Slider**: ปรับความโปร่งใสของจุดอ้างอิงพื้นหลัง

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - เมื่อลูกบอลเคลื่อนที่ผ่านบริเวณที่มีจุดอ้างอิงหนาแน่น (Dense Cluster) ทรงกลมเรดาร์จะหดตัวเล็กลงเพราะหาเพื่อนบ้านครบ $K$ ตัวได้ในระยะใกล้
   - เมื่อลูกบอลวิ่งออกไปในโซนเบาบาง (Outlier / Sparse Region) ทรงกลมเรดาร์จะขยายขนาดใหญ่ขึ้นอย่างชัดเจนเพื่อควานหาเพื่อนบ้านให้ครบ $K$ ตัว!
2. **เมื่อลาก Scrubber ข้ามจุดเปลี่ยนคำตัดสิน**:
   - เส้นใยแมงมุม (Spider Lines) จะสลับการยึดเกาะจากกลุ่มจุดสีฟ้าไปเกาะกลุ่มจุดสีส้มอย่างรวดเร็ว ทำให้เห็นกลไกการเปลี่ยนเสียงข้างมากด้วยตาเปล่า
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - ขณะลากลูกบอลเงา (Ghost Sphere) ไปในฉาก 3D เส้นใยแมงมุมทั้ง $K$ เส้นจะคำนวณระยะทางใหม่แบบสดๆ ปล่อยมือจากเพื่อนบ้านตัวเก่าที่อยู่ไกลออกไป แล้วพุ่งไปจับเพื่อนบ้านตัวใหม่ที่อยู่ใกล้กว่าทันที!

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล KNN
```text
+-------------------------------------------------------------------------+
| [KNN RADAR HUD]  K = 5 Neighbors | Weights: distance | Metric: l2 (p=2) |
| Search Index: kd_tree (leaf_size=30) | Ref Cloud: 1,500 pts (100% Exact)|
+-------------------------------------------------------------------------+
| Radar Visuals: [x] 3D Radar Bubble   [x] Spider Laser Lines   Opacity:50%|
+-------------------------------------------------------------------------+
| FRAME #0058 K-NEIGHBORS VOTE TALLY (Radar Radius R_t = 1.428 units):    |
|  Winning Verdict : CLASS 1 (Weighted Vote: 78.4% | Raw Count: 4 / 5)    |
|  [███████████████████████████████░░░░░░░░░] 4 Class 1  vs  1 Class 0    |
+-------------------------------------------------------------------------+
| ACTIVE K=5 NEIGHBORS CONNECTED BY SPIDER LINES:                         |
|  Rank | Ref ID  | Class   | 3D Dist | Full D-Dist | Vote Weight Share   |
|  #1   | Row #412| Class 1 |  0.410  |    0.520    |  31.5% [==========] |
|  #2   | Row #108| Class 1 |  0.620  |    0.780    |  21.0% [=======   ] |
|  #3   | Row #889| Class 0 |  0.710  |    0.790    |  20.7% [=======   ] |
|  #4   | Row #204| Class 1 |  0.980  |    1.150    |  14.3% [=====     ] |
|  #5   | Row #519| Class 1 |  1.290  |    1.428    |  12.5% [====      ] |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/distance/knn.py`
- `frontend/src/domains/stage3d/components/actors/KnnRadarActor3D.tsx`
- `frontend/src/domains/hud/components/inspectors/KnnNeighborTable.tsx`

### 5.2 โค้ดต้นแบบ `KNNAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor
)
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class KNNAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (
        KNeighborsClassifier, KNeighborsRegressor,
        RadiusNeighborsClassifier, RadiusNeighborsRegressor
    )
    MAX_BACKGROUND_REF_POINTS = 5000

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.is_classifier = hasattr(model, "classes_")
        self.is_radius = isinstance(model, (RadiusNeighborsClassifier, RadiusNeighborsRegressor))
        self.fit_X: np.ndarray = np.asarray(
            model._fit_X.toarray() if hasattr(model._fit_X, "toarray") else model._fit_X,
            dtype=np.float32
        )
        self.fit_y: np.ndarray = np.asarray(model._y)
        if self.is_classifier and hasattr(model, "classes_") and self.fit_y.ndim == 1:
            self.fit_y_labels = np.asarray(model.classes_)[self.fit_y]
        else:
            self.fit_y_labels = self.fit_y

    def extract_metadata(self) -> Dict[str, Any]:
        metric = getattr(self.model, "effective_metric_", getattr(self.model, "metric", "euclidean"))
        p_val = getattr(self.model, "p", 2)
        radar_shape = "octahedron" if (metric in ("manhattan", "cityblock", "l1") or p_val == 1) else (
            "cube" if metric == "chebyshev" else "sphere"
        )
        return {
            "model_family": "distance_knn",
            "algorithm": self.model.__class__.__name__,
            "task_type": "classification" if self.is_classifier else "regression",
            "n_neighbors": getattr(self.model, "n_neighbors", None),
            "radius": getattr(self.model, "radius", None),
            "weights": getattr(self.model, "weights", "uniform"),
            "effective_metric": str(metric),
            "radar_geometry_shape": radar_shape,
            "total_reference_points": int(self.fit_X.shape[0]),
        }

    def get_downsampled_reference_cloud(self, mandatory_indices: np.ndarray) -> Dict[str, Any]:
        """รวมจุดเพื่อนบ้านที่ถูกอ้างอิงจริง 100% เข้ากับจุดสุ่มพื้นหลังสูงสุดไม่เกิน 5,000 จุด"""
        n_total = self.fit_X.shape[0]
        mandatory_uniq = np.unique(mandatory_indices.ravel())
        if n_total <= self.MAX_BACKGROUND_REF_POINTS:
            selected_idx = np.arange(n_total)
        else:
            rem_budget = max(0, self.MAX_BACKGROUND_REF_POINTS - len(mandatory_uniq))
            all_idx = np.setdiff1d(np.arange(n_total), mandatory_uniq, assume_unique=True)
            rng = np.random.default_rng(seed=42)
            bg_idx = rng.choice(all_idx, size=min(rem_budget, len(all_idx)), replace=False)
            selected_idx = np.sort(np.concatenate([mandatory_uniq, bg_idx]))

        # สร้างแผนที่แปลง Original Index ใน _fit_X -> Local Index ใน Cloud ที่ส่งให้ Frontend
        idx_map = {int(orig): int(loc) for loc, orig in enumerate(selected_idx)}
        return {
            "original_indices": selected_idx.tolist(),
            "coordinates": self.fit_X[selected_idx].tolist(),
            "targets": self.fit_y_labels[selected_idx].tolist(),
            "index_map": idx_map,
        }

    def trace_internal_flow(self, X_batch: np.ndarray) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float32)
        preds = self.model.predict(X_arr)
        probas = self.model.predict_proba(X_arr).astype(np.float32).tolist() if (
            self.is_classifier and hasattr(self.model, "predict_proba")
        ) else None

        if not self.is_radius:
            dists, neigh_idx = self.model.kneighbors(X_arr, return_distance=True)
            max_radii = np.max(dists, axis=1).astype(np.float32).tolist()
            neighbor_centroids = np.mean(self.fit_X[neigh_idx], axis=1)  # (N, D)
            contributions = -np.abs(X_arr - neighbor_centroids)
            dists_list = dists.astype(np.float32).tolist()
            idx_list = neigh_idx.tolist()
            ref_cloud = self.get_downsampled_reference_cloud(neigh_idx)
        else:
            dists_obj, idx_obj = self.model.radius_neighbors(X_arr, return_distance=True)
            max_radii = [float(self.model.radius)] * len(X_arr)
            dists_list = [d.astype(np.float32).tolist()[:20] for d in dists_obj]
            idx_list = [i.tolist()[:20] for i in idx_obj]
            all_ref = np.concatenate(idx_obj) if len(idx_obj) > 0 else np.array([], dtype=int)
            ref_cloud = self.get_downsampled_reference_cloud(all_ref)
            contributions = np.zeros_like(X_arr)

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas,
            decision_scores= max_radii,
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=[0.0],
            extra_telemetry={
                "neighbor_distances": dists_list,
                "neighbor_original_indices": idx_list,
                "neighbor_labels": [self.fit_y_labels[i].tolist() for i in idx_list],
                "radar_radii": max_radii,
                "reference_cloud": ref_cloud,
            },
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 45, target_class_idx: int = 1
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float32)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        grid_X = np.tile(fixed_features.astype(np.float32), (grid_resolution * grid_resolution, 1))
        grid_X[:, axis_x_idx] = xx.ravel()
        grid_X[:, axis_y_idx] = yy.ravel()

        if self.is_classifier and hasattr(self.model, "predict_proba"):
            z_flat = self.model.predict_proba(grid_X)[:, min(target_class_idx, len(self.model.classes_) - 1)]
        else:
            z_flat = self.model.predict(grid_X)

        return SurfaceMesh3D(
            x_grid=xs.tolist(),
            y_grid=ys.tolist(),
            z_values=z_flat.reshape(grid_resolution, grid_resolution).astype(np.float32).tolist(),
            auxiliary_grids={},
            boundary_threshold=0.5 if self.is_classifier else None,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ส่วน `extra_telemetry` ของ KNN)
```json
{
  "neighbor_distances": [[0.520, 0.780, 0.790, 1.150, 1.428]],
  "neighbor_original_indices": [[412, 108, 889, 204, 519]],
  "neighbor_labels": [[1, 1, 0, 1, 1]],
  "radar_radii": [1.428],
  "reference_cloud": {
    "original_indices": [108, 204, 412, 519, 889],
    "coordinates": [[1.2, 0.5], [1.5, 0.8], [1.1, 0.4], [1.9, 1.1], [0.9, -0.2]],
    "targets": [1, 1, 1, 1, 0]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **Mandatory Union + Stratified Downsampling สำหรับ `model._fit_X`**:
   - โมเดล KNN เก็บชุดข้อมูลเทรนทั้งหมดไว้ใน `model._fit_X` ซึ่งอาจมีถึง $500,000$ แถว หากส่งทั้งหมดไปที่เบราว์เซอร์จะทำให้ JSON มีขนาดหลายร้อย MB และทำให้ WebGL ค้างทันที
   - อัลกอริทึม `get_downsampled_reference_cloud()` ของเราแก้ปัญหานี้อย่างเด็ดขาดโดย:
     1. ดึง **Union ของดัชนีเพื่อนบ้านทุกตัวที่ถูกอ้างอิงจริงในคิวการเล่น (`np.unique(neigh_idx)`)** มาใส่ไว้ในชุดข้อมูล 100% ก่อนเสมอ (รับประกันว่าเส้นใยแมงมุม Spider Lines ทุกเส้นในทุกเฟรมจะมีจุดปลายทางอยู่ในฉาก 3D เสมอ ไม่เคยชี้ไปในอากาศว่างเปล่า!)
     2. เติมจุดพื้นหลังจาก `_fit_X` ด้วยการสุ่มตัวอย่างแบบกำหนด Seed คงที่จนครบเพดานสูงสุด `5,000` จุด
2. **การเรนเดอร์จุดอ้างอิง 5,000 จุดด้วย `THREE.InstancedMesh` เดียว**:
   - ห้ามสร้าง `<mesh>` แยก 5,000 ตัวเด็ดขาด ให้ใช้ `<instancedMesh args={[sphereGeo, mat, count]}>` เพียง 1 Draw Call และไฮไลต์เพื่อนบ้าน $K$ ตัวที่ชนะด้วยการอัปเดต `instanceMatrix` และ `instanceColor` เฉพาะดัชนีที่เปลี่ยนในเฟรมนั้น

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **ระยะห่างใน $D$ มิติ vs ระยะห่างที่ตาเห็นในโปรเจกชัน 3 มิติ** | จุดเพื่อนบ้านที่อยู่ใกล้ที่สุดในอวกาศ 20 มิติ อาจดูอยู่ไกลกว่าจุดอื่นเมื่อถูกฉายลงบนแกน $(X, Y)$ เพียง 2 แกน | แสดงคอลัมน์เปรียบเทียบใน HUD ทั้ง `3D Projected Dist` และ `Full D-Dim Dist` พร้อมวาดรัศมีทรงกลมเรดาร์ในฉาก 3D ให้ครอบคลุมระยะ 3D ของเพื่อนบ้านตัวที่ $K$ พอดี |
| 2 | **`weights='distance'` เมื่อจุดทดสอบทับจุดเทรนสนิท ($d = 0.0$)** | เมื่อ $d_{(1)} = 0.0$ การคำนวณ $1 / d_{(1)}$ จะเกิด `ZeroDivisionError` | ใช้ `1.0 / np.maximum(dists, 1e-12)` แล้ว Normalize ให้ผลรวมน้ำหนักเท่ากับ `1.0` เสมอ |
| 3 | **`RadiusNeighborsClassifier` ไม่พบเพื่อนบ้านเลยในรัศมีที่กำหนด** | จุดที่เป็น Outlier อยู่ห่างเกิน `model.radius` ทำให้ `radius_neighbors` คืนค่า Array ว่าง `[]` และเกิด `ValueError` ตอน `predict` (หากไม่ได้ตั้ง `outlier_label`) | ดักจับ `ValueError` ใน `predict()` และตรวจสอบเมื่อ `len(neigh_idx) == 0` ให้แสดงป้ายเตือน **"OUTLIER: 0 Neighbors found within Radius R"** บนลูกบอลเรดาร์ |
| 4 | **`model._fit_X` ถูกเก็บเป็น Sparse Matrix (`csr_matrix`)** | เมื่อเทรนด้วย `algorithm='brute'` กับข้อมูล Sparse ตัว `model._fit_X` จะไม่ใช่ `np.ndarray` | ตรวจสอบ `hasattr(model._fit_X, "toarray")` และแปลงเฉพาะแถวที่ผ่านการ Downsample แล้วเท่านั้น |
| 5 | **Metric ที่ไม่ใช่ Euclidean (`manhattan`, `chebyshev`, `cosine`)** | การวาดทรงกลมกลมเกลี้ยง (`SphereGeometry`) สำหรับ Manhattan Distance ($L_1$) จะสื่อความหมายทางคณิตศาสตร์ผิด | ตรวจสอบ `effective_metric_` แล้วสลับ Geometry ของเรดาร์เป็น `OctahedronGeometry` สำหรับ $L_1$ และ `BoxGeometry` สำหรับ Chebyshev ($L_\infty$) อัตโนมัติ |
| 6 | **จำนวน $K$ สูงมาก (เช่น $K = 200$)** | การวาดเส้น Spider Lines 200 เส้นพร้อมป้ายชื่อจะทำให้ฉาก 3D รกทึบ | จำกัดการวาดเส้น Spider Lines ที่สว่างชัดเจนสูงสุด 15 อันดับแรกที่ใกล้ที่สุด (`Top-15 Closest Neighbors`) และแสดงผลรวมทั้งหมดในตาราง HUD |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `KNNAdapter` รองรับครบทั้ง `KNeighborsClassifier`, `KNeighborsRegressor`, `RadiusNeighborsClassifier`, และ `RadiusNeighborsRegressor`
- [ ] ระบบ `get_downsampled_reference_cloud` รับประกันว่าจุดเพื่อนบ้านทุกตัวที่ถูกชี้โดย `kneighbors` มีพิกัดอยู่ในฉาก 3D ครบ 100% โดยจำกัดจำนวนจุดรวมไม่เกิน 5,000 จุด
- [ ] ฉาก 3 มิติแสดงกลุ่มจุดอ้างอิงด้วย `InstancedMesh` (1 Draw Call) พร้อมทรงกลมเรดาร์ที่เปลี่ยนรูปทรงตาม Metric (`Sphere` / `Octahedron` / `Cube`)
- [ ] เส้นแสงเลเซอร์ **Spider Web Lines** โยงจากลูกบอลปัจจุบันไปยังเพื่อนบ้านทั้ง $K$ จุด พร้อมปรับความสว่างตามน้ำหนักการโหวตจริง
- [ ] โหมด What-If Sandbox คำนวณการเปลี่ยนจุดเกาะของเส้นใยแมงมุมเมื่อลากลูกบอลเงาไปในฉาก 3D ได้อย่างแม่นยำ
