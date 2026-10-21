# Phase 26: LDA & QDA Adapters & 3D Covariance Ellipsoid Visualizer

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลการวิเคราะห์จำแนกประเภทเชิงสถิติพหุตัวแปร ได้แก่ **`LinearDiscriminantAnalysis` (LDA)** และ **`QuadraticDiscriminantAnalysis` (QDA)** ซึ่งขยายขีดความสามารถต่อจาก Gaussian Naive Bayes ด้วยการนำ **เมทริกซ์ความแปรปรวนร่วมระหว่างฟีเจอร์ (Full Covariance Matrix $\mathbf{\Sigma}$)** เข้ามาคำนวณระยะห่าง **Mahalanobis Distance** โดยแสดงผลเป็น **แคปซูลทรงรีความแปรปรวนร่วม 3 มิติที่หมุนเอียงตามแกน Eigenvectors จริง (3D Tilted Covariance Ellipsoids)** พร้อมเปรียบเทียบพื้นผิวแบ่งเขตแบบระนาบเส้นตรง (LDA Shared Covariance) กับพื้นผิวโค้งควอดราติก (QDA Per-Class Covariance)

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (รองรับทุก Solver: `svd`, `lsqr`, `eigen`)
  - `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary & Multiclass Classification**
  - **Supervised Dimensionality Reduction Projection** (ผ่าน `lda.transform(X)` สำหรับ LDA)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของระยะห่าง Mahalanobis และ LDA vs QDA
กำหนดให้มีคลาส $C$ คลาส ($k \in \{0, \dots, C-1\}$) จุดศูนย์กลางคลาสคือ $\boldsymbol{\mu}_k \in \mathbb{R}^D$ (`model.means_`) และความน่าจะเป็นก่อนหน้าคือ $\pi_k$ (`model.priors_`):

1. **ระยะห่างกำลังสองแบบ Mahalanobis (Squared Mahalanobis Distance)**:
   $$D_M^2(\mathbf{x}_t, \boldsymbol{\mu}_k; \mathbf{\Sigma}_k) = (\mathbf{x}_t - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_k)$$
2. **กรณี `LinearDiscriminantAnalysis` (LDA — สมมติฐาน $\mathbf{\Sigma}_k = \mathbf{\Sigma}$ ร่วมกันทุกคลาส)**:
   - เมื่อเมทริกซ์ความแปรปรวนร่วมเท่ากันทุกคลาส เทอมกำลังสอง $\mathbf{x}_t^T \mathbf{\Sigma}^{-1} \mathbf{x}_t$ จะหักล้างกันหมด เหลือสมการจำแนกเชิงเส้นตรง:
     $$\delta_k(\mathbf{x}_t) = \mathbf{w}_k^T \mathbf{x}_t + b_k, \quad \text{โดยที่ } \mathbf{w}_k = \mathbf{\Sigma}^{-1}\boldsymbol{\mu}_k, \quad b_k = -\frac{1}{2}\boldsymbol{\mu}_k^T \mathbf{\Sigma}^{-1}\boldsymbol{\mu}_k + \log \pi_k$$
   - รูปทรงแคปซูลทรงรี 3 มิติของทุกคลาสจะมี **ขนาดและมุมเอียง (Orientation) เหมือนกันทุกประการ** ต่างกันเพียงจุดศูนย์กลาง $\boldsymbol{\mu}_k$!
3. **กรณี `QuadraticDiscriminantAnalysis` (QDA — แต่ละคลาสมี $\mathbf{\Sigma}_k$ ของตนเอง)**:
   - สมการคะแนนจำแนกเป็นฟังก์ชันกำลังสอง (Quadratic Form):
     $$\delta_k(\mathbf{x}_t) = -\frac{1}{2}\log |\mathbf{\Sigma}_k| - \frac{1}{2}(\mathbf{x}_t - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_k) + \log \pi_k$$
   - แคปซูลทรงรี 3 มิติของแต่ละคลาสจะมี **ขนาดและมุมหมุนเอียงเป็นอิสระต่อกัน** และเส้นแบ่งเขตระหว่างคลาสจะเป็นเส้นโค้งภาคตัดกรวย (Parabola / Hyperbola / Ellipse)!

### 2.2 การแยกตัวประกอบ Eigendecomposition สำหรับเรนเดอร์ทรงรี 3 มิติ และ Waterfall
- **การคำนวณพารามิเตอร์ทรงรีจาก Sub-matrix $\mathbf{\Sigma}_k^{(xyz)}$**:
  - ดึงเมทริกซ์ย่อยขนาด $2 \times 2$ (หรือ $3 \times 3$) ของแกนที่เลือก แล้วคำนวณ Eigendecomposition ด้วย `np.linalg.eigh`:
    $$\mathbf{\Sigma}_k^{\text{sub}} = \mathbf{V}_k \mathbf{\Lambda}_k \mathbf{V}_k^T, \quad \mathbf{\Lambda}_k = \text{diag}(\lambda_{k,1}, \lambda_{k,2})$$
  - รัศมีของทรงรีที่ระดับ $m$-Sigma ($m \in \{1, 2\}$): $r_{k,i} = m \sqrt{\max(\lambda_{k,i}, \epsilon)}$
  - เมทริกซ์การหมุนแกน $\mathbf{V}_k$ (ปรับให้ $\det(\mathbf{V}_k) = +1$ เสมอเพื่อให้เป็น Right-handed Rotation Matrix) แปลงเป็นมุมหมุน $\theta_k = \text{atan2}(V_{1,0}, V_{0,0})$ หรือ Quaternion สำหรับ Three.js
- **การคำนวณ Local Feature Contribution (Waterfall)**:
  - **สำหรับ LDA**: ใช้สมการเชิงเส้นตรง $\phi_{t,j}^{(k)} = w_{k,j} x_{t,j}$ จาก `model.coef_`
  - **สำหรับ QDA**: แยกองค์ประกอบของเทอม Quadratic $(\mathbf{x}_t - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_k)$ รายฟีเจอร์โดยคำนวณเวกเตอร์ความแม่นยำ $\mathbf{p}_{t,k} = \mathbf{\Sigma}_k^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_k)$ แล้วกำหนดให้:
    $$\phi_{t,j}^{(k)} = -\frac{1}{2} (x_{t,j} - \mu_{k,j}) \cdot p_{t,k,j}$$
    ซึ่งมีผลรวม $\sum_{j=1}^{D} \phi_{t,j}^{(k)} = -\frac{1}{2} D_M^2(\mathbf{x}_t, \boldsymbol{\mu}_k; \mathbf{\Sigma}_k)$ ตรงตามสมการ QDA 100%!

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **3D Tilted Covariance Ellipsoids (`SphereGeometry` Scaled & Rotated by Eigenvectors)**:
   - ล้อมรอบจุดศูนย์กลาง $\boldsymbol{\mu}_k$ ของแต่ละคลาสด้วยแคปซูลทรงรีโปร่งแสง 2 ชั้น ($1\sigma$ Inner Core ความทึบ `0.45` และ $2\sigma$ Outer Shell แบบเส้นตาราง Wireframe ความทึบ `0.25`)
   - ภายในทรงรีแต่ละลูกมี **แกน Eigenvector Principal Axes 3 มิติ** (เส้นกากบาทแกนหลัก-แกนรองที่หมุนเอียงตามทิศทางสหสัมพันธ์ของฟีเจอร์) แสดงให้เห็นความแตกต่างจาก GaussianNB ที่แกนไม่เคยหมุนเอียง!
2. **Mahalanobis Distance Laser Tethers vs Euclidean Comparison**:
   - เมื่อลูกบอลข้อมูลปัจจุบัน $\mathbf{x}_t$ เคลื่อนที่ไปในฉาก 3D จะมีเส้นเลเซอร์พุ่งจากลูกบอลไปยังจุดศูนย์กลาง $\boldsymbol{\mu}_k$ ของทุกคลาส
   - แสดงตัวเลขเปรียบเทียบบนเส้นเลเซอร์ทั้ง **Mahalanobis Distance ($D_M$)** และ **Euclidean Distance ($D_E$)** เพื่อให้ผู้ใช้เข้าใจกรณีที่จุดข้อมูลดูเหมือนอยู่ใกล้กลุ่ม $A$ ตามระยะทางไม้บรรทัดปกติ ($D_E$) แต่จริงๆ แล้วอยู่ใกล้กลุ่ม $B$ มากกว่าเมื่อคิดตามแนวการยืดตัวของทรงรี ($D_M$)!
3. **Linear vs Quadratic Decision Boundary Surface**:
   - วาดพื้นผิวความน่าจะเป็น $P(y = k \mid \mathbf{x})$ พร้อมเส้นแบ่งเขตที่รอยต่อซึ่งเป็นเส้นตรงเป๊ะสำหรับ LDA และเป็นเส้นโค้งโอบล้อมสำหรับ QDA

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Ellipsoid Sigma Shell Selector**: ปุ่มเลือกแสดงขอบเขตทรงรีที่ระดับ `1σ (68%)`, `2σ (95%)`, หรือ `3σ (99.7%)`
- **Principal Eigenvector Axes Toggle**: เปิด-ปิดการแสดงแกนหลักของ Covariance Matrix ภายในทรงรี
- **Mahalanobis vs Euclidean Paradox Indicator**: ป้ายไฟแจ้งเตือนพิเศษที่จะสว่างขึ้นในเฟรมที่ **คลาสที่อยู่ใกล้ที่สุดตามระยะ Mahalanobis ไม่ตรงกับคลาสที่อยู่ใกล้ที่สุดตามระยะ Euclidean**!

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลวิ่งผ่านกลุ่มแคปซูลทรงรี 3 มิติ โดยมีเส้นวัดระยะ Mahalanobis ยืด-หดตามตำแหน่ง
   - หากเฟรมใดเกิดปรากฏการณ์ *Mahalanobis vs Euclidean Paradox* (เช่น จุดอยู่ตามแนวแกนยาวของทรงรีคลาส 1 ทำให้ $D_M$ น้อยกว่าคลาส 0 ทั้งที่ระยะทางตรง $D_E$ ไกลกว่า) ระบบจะแสดงไฮไลต์สีทองอธิบายเหตุผลทางสถิติทันที
2. **เมื่อลาก Scrubber ข้ามเส้นแบ่งเขต**:
   - แสดงการเปลี่ยนผ่านของค่าความน่าจะเป็นและลำดับชนะของระยะห่าง Mahalanobis
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - เมื่อผู้ใช้ลากลูกบอลเงา (Ghost Sphere) เคลื่อนที่ไปตามแนวแกนยาว (Major Eigenvector) ของทรงรี จะเห็นว่าค่า Mahalanobis Distance เพิ่มขึ้นช้ามาก แต่หากลากตัดขวางแนวแกนสั้น (Minor Eigenvector) เพียงนิดเดียว ค่า Mahalanobis Distance จะพุ่งสูงขึ้นอย่างรวดเร็ว!

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล LDA & QDA
```text
+-------------------------------------------------------------------------+
| [DISCRIMINANT ANALYSIS HUD]  Model: QDA | Covariance: Per-Class Full Σk |
| reg_param: 0.000 | store_covariance: True (Exact Eigen-Ellipsoids)      |
+-------------------------------------------------------------------------+
| Ellipsoid Shells: [x] 1σ Core  [x] 2σ Wireframe  [x] Eigenvector Axes   |
| Paradox Alert   : ⚡ ACTIVE! Euclidean favors Class 0, Mahalanobis -> 1 |
+-------------------------------------------------------------------------+
| FRAME #0182 MAHALANOBIS vs EUCLIDEAN DISTANCE COMPARISON:               |
|  Class | Prior | Euclidean Dist | Mahalanobis (D_M) | Posterior P(C|x)  |
|  *#1*  | 0.500 |  2.450 (Far)   |  *1.120 (CLOSER)* |  *84.6% (WINNER)* |
|   #0   | 0.500 | *1.890 (Near)* |   2.310 (Farther) |   15.4%           |
+-------------------------------------------------------------------------+
| QUADRATIC PRECISION WATERFALL (-0.5 * (x - μ_k)_j * [Σ^-1 (x - μ_k)]_j):|
|  Prior & Log-Det Base (-0.5 log|Σ1| + log π1) : -1.240                  |
|  feature_length (X) (Aligned with Major Axis) : -0.310 [     <=|      ] |
|  feature_width  (Y) (Cross-correlation bonus) : +0.180 [       |=>    ] |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/probabilistic/discriminant.py`
- `frontend/src/domains/stage3d/components/actors/CovarianceEllipsoids3D.tsx`
- `frontend/src/domains/hud/components/inspectors/DiscriminantInspector.tsx`

### 5.2 โค้ดต้นแบบ `DiscriminantAnalysisAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class DiscriminantAnalysisAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.is_qda = isinstance(model, QuadraticDiscriminantAnalysis)
        self.classes_: np.ndarray = np.asarray(model.classes_)
        self.n_classes: int = len(self.classes_)
        self.means_: np.ndarray = np.asarray(model.means_, dtype=np.float64)  # (C, D)
        self.priors_: np.ndarray = np.asarray(model.priors_, dtype=np.float64)
        self.covariances_per_class, self.is_cov_approximated = self._resolve_covariances()
        self.precisions_per_class = [
            np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-9) for cov in self.covariances_per_class
        ]

    def _resolve_covariances(self) -> Tuple[List[np.ndarray], bool]:
        """ดึงเมทริกซ์ความแปรปรวนร่วม หรือสร้างจาก scalings_ /rotations_ หาก store_covariance=False"""
        D = self.means_.shape[1]
        if self.is_qda:
            if hasattr(self.model, "covariance_") and self.model.covariance_ is not None:
                return [np.asarray(c, dtype=np.float64) for c in self.model.covariance_], False
            if hasattr(self.model, "rotations_") and hasattr(self.model, "scalings_"):
                covs = []
                for R, s in zip(self.model.rotations_, self.model.scalings_):
                    # In sklearn QDA: scalings_ holds variances (eigenvalues of Σ_k)
                    cov_k = (R * s[np.newaxis, :]) @ R.T
                    covs.append(np.asarray(cov_k, dtype=np.float64))
                return covs, False
        else:
            if hasattr(self.model, "covariance_") and self.model.covariance_ is not None:
                shared = np.asarray(self.model.covariance_, dtype=np.float64)
                return [shared for _ in range(self.n_classes)], False
            if hasattr(self.model, "scalings_") and self.model.scalings_ is not None:
                S = np.asarray(self.model.scalings_, dtype=np.float64)
                # Reconstruct approximate within-class covariance from pseudo-inverse of scalings
                S_pinv = np.linalg.pinv(S)
                approx_cov = S_pinv.T @ S_pinv
                return [approx_cov for _ in range(self.n_classes)], True

        return [np.eye(D, dtype=np.float64) for _ in range(self.n_classes)], True

    def compute_ellipsoid_2d_specs(self, axis_x_idx: int, axis_y_idx: int) -> List[Dict[str, Any]]:
        specs = []
        idx_pair = [axis_x_idx, axis_y_idx]
        for k in range(self.n_classes):
            sub_cov = self.covariances_per_class[k][np.ix_(idx_pair, idx_pair)]
            eigvals, eigvecs = np.linalg.eigh(sub_cov)
            eigvals = np.maximum(eigvals, 1e-9)
            if np.linalg.det(eigvecs) < 0:
                eigvecs[:, 0] *= -1.0
            angle_rad = float(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            specs.append({
                "class_idx": k,
                "center_xy": self.means_[k, idx_pair].tolist(),
                "radii_1sigma": np.sqrt(eigvals).tolist(),
                "rotation_angle_rad": angle_rad,
                "eigenvectors_2d": eigvecs.tolist(),
            })
        return specs

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "model_family": "discriminant_analysis",
            "algorithm": "QDA" if self.is_qda else "LDA",
            "task_type": "binary_classification" if self.n_classes == 2 else "multiclass_classification",
            "classes": self.classes_.tolist(),
            "priors": self.priors_.tolist(),
            "means": self.means_.tolist(),
            "is_covariance_approximated": self.is_cov_approximated,
            "solver": getattr(self.model, "solver", "svd"),
            "reg_param": float(getattr(self.model, "reg_param", 0.0)),
        }

    def trace_internal_flow(self, X_batch: np.ndarray) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        preds = self.model.predict(X_arr)
        probas = self.model.predict_proba(X_arr)
        dec_scores = self.model.decision_function(X_arr)
        scores_2d = dec_scores.reshape(-1, 1) if dec_scores.ndim == 1 else dec_scores

        N, D = X_arr.shape
        mahal_dists = np.zeros((N, self.n_classes), dtype=np.float64)
        euclid_dists = np.zeros((N, self.n_classes), dtype=np.float64)
        for k in range(self.n_classes):
            diff = X_arr - self.means_[k][np.newaxis, :]  # (N, D)
            euclid_dists[:, k] = np.linalg.norm(diff, axis=1)
            prec_diff = diff @ self.precisions_per_class[k]  # (N, D)
            mahal_sq = np.maximum(np.sum(diff * prec_diff, axis=1), 0.0)
            mahal_dists[:, k] = np.sqrt(mahal_sq)

        win_idx = np.argmax(probas, axis=1)
        if not self.is_qda and hasattr(self.model, "coef_"):
            coef_2d = np.atleast_2d(self.model.coef_)
            w = coef_2d[0] if self.n_classes == 2 else coef_2d[win_idx]
            contributions = X_arr * w
            intercept_list = np.atleast_1d(self.model.intercept_).astype(float).tolist()
        else:
            contributions = np.zeros((N, D), dtype=np.float64)
            for n in range(N):
                c = win_idx[n]
                diff_n = X_arr[n] - self.means_[c]
                contributions[n] = -0.5 * diff_n * (self.precisions_per_class[c] @ diff_n)
            intercept_list = np.log(np.maximum(self.priors_, 1e-15)).tolist()

        paradox_mask = (np.argmin(mahal_dists, axis=1) != np.argmin(euclid_dists, axis=1)).tolist()

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas.astype(np.float32).tolist(),
            decision_scores=scores_2d.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=intercept_list,
            extra_telemetry={
                "mahalanobis_distances": mahal_dists.astype(np.float32).tolist(),
                "euclidean_distances": euclid_dists.astype(np.float32).tolist(),
                "paradox_detected": paradox_mask,
            },
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 60, target_class_idx: int = 1
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        grid_X = np.tile(fixed_features.astype(np.float64), (grid_resolution * grid_resolution, 1))
        grid_X[:, axis_x_idx] = xx.ravel()
        grid_X[:, axis_y_idx] = yy.ravel()

        probas = self.model.predict_proba(grid_X)
        c_idx = min(target_class_idx, self.n_classes - 1)
        z_surface = probas[:, c_idx].reshape(grid_resolution, grid_resolution)

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=z_surface.astype(np.float32).tolist(),
            auxiliary_grids={
                "ellipsoids_2d": self.compute_ellipsoid_2d_specs(axis_x_idx, axis_y_idx)
            },
            boundary_threshold=0.5,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [1, 0],
    "probabilities": [[0.154, 0.846], [0.910, 0.090]],
    "decision_scores": [[1.703], [-2.314]],
    "feature_contributions": [[-0.310, 0.180], [-0.120, -0.090]],
    "base_intercepts": [-0.693, -0.693],
    "extra_telemetry": {
      "mahalanobis_distances": [[2.310, 1.120], [0.850, 2.740]],
      "euclidean_distances": [[1.890, 2.450], [0.920, 2.610]],
      "paradox_detected": [true, false]
    }
  },
  "SurfaceMesh3D": {
    "x_grid": [-3.0, 3.0],
    "y_grid": [-3.0, 3.0],
    "z_values": [[0.02, 0.85], [0.14, 0.96]],
    "auxiliary_grids": {
      "ellipsoids_2d": [
        {
          "class_idx": 0,
          "center_xy": [-1.2, -0.8],
          "radii_1sigma": [0.45, 1.35],
          "rotation_angle_rad": 0.68,
          "eigenvectors_2d": [[0.77, -0.63], [0.63, 0.77]]
        }
      ]
    },
    "boundary_threshold": 0.5
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **Sub-Matrix $2 \times 2$ Eigendecomposition แทนการคำนวณเต็ม $D \times D$**:
   - เมื่อผู้ใช้สลับแกน $(X, Y)$ ในฉาก 3 มิติ เราไม่จำเป็นต้องคำนวณ `np.linalg.eigh` บนเมทริกซ์ขนาดใหญ่ $(D, D)$ ซ้ำทุกครั้ง แต่ตัดเฉพาะ Sub-matrix ขนาด $2 \times 2$ ด้วย `cov[np.ix_([axis_x_idx, axis_y_idx], [axis_x_idx, axis_y_idx])]` ซึ่งใช้เวลาคำนวณในระดับไมโครวินาที (`< 0.01 ms`)
2. **Pre-computed Precision Matrices ($\mathbf{\Sigma}_k^{-1}$) ตอนโหลดโมเดล**:
   - การหาอินเวอร์สของเมทริกซ์ (`np.linalg.pinv`) มีความซับซ้อน $O(D^3)$ เราจึงคำนวณเก็บไว้ใน `self.precisions_per_class` เพียงครั้งเดียวใน `__init__` แล้วใช้การคูณเมทริกซ์ `diff @ self.precisions_per_class[k]` แบบ Vectorized ใน `trace_internal_flow`

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`store_covariance=False` เป็นค่าเริ่มต้นของทั้ง LDA และ QDA!** | ใน Scikit-learn หากผู้ใช้ไม่ได้ระบุ `store_covariance=True` ตอนสร้างโมเดล แอตทริบิวต์ `model.covariance_` จะไม่มีอยู่เลย! | ใน QDA ให้สร้าง $\mathbf{\Sigma}_k$ คืนกลับมาอย่างแม่นยำ 100% จาก `(R * s) @ R.T` โดยใช้ `model.rotations_` และ `model.scalings_`! ส่วนใน LDA หากไม่มี `covariance_` ให้ประมาณค่าจาก `pinv(model.scalings_)` พร้อมแจ้งสถานะ `is_covariance_approximated: true` |
| 2 | **Singular Covariance Matrix (Collinear Features)** | เมื่อมีฟีเจอร์ซ้ำซ้อนหรือจำนวนตัวอย่างในคลาสน้อยกว่าจำนวนฟีเจอร์ ($N_k < D$) การใช้ `np.linalg.inv` จะเกิด `LinAlgError: Singular matrix` | ห้ามใช้ `np.linalg.inv` เด็ดขาด ให้ใช้ `np.linalg.pinv(cov + np.eye(D) * 1e-9)` (Moore-Penrose Pseudo-Inverse พร้อม Ridge Jitter) เสมอ |
| 3 | **Improper Rotation Matrix ($\det(\mathbf{V}) = -1$) จาก `np.linalg.eigh`** | เวกเตอร์ลักษณะเฉพาะจาก `eigh` อาจเรียงตัวในระบบมือซ้าย (Reflection) ซึ่งเมื่อส่งไปแปลงเป็น Quaternion ใน Three.js จะทำให้ทรงรีพลิกกลับด้านผิดเพี้ยน | ตรวจสอบ `if np.linalg.det(eigvecs) < 0: eigvecs[:, 0] *= -1.0` เสมอเพื่อรับประกันว่าเป็น Right-handed Rotation Matrix แท้ |
| 4 | **`solver='lsqr'` หรือ `'eigen'` ใน LDA ไม่มี `scalings_` แบบ SVD** | เมื่อเทรน LDA ด้วย `solver='lsqr'` โมเดลจะคำนวณ `covariance_` โดยตรงแต่โครงสร้าง `scalings_` จะต่างจาก `'svd'` | ลำดับการดึงใน `_resolve_covariances` จะเช็ก `hasattr(model, "covariance_")` ก่อนเสมอ จึงทำงานกับ `solver='lsqr'` ได้อย่างสมบูรณ์ |
| 5 | **ความหมายของ `scalings_` ใน LDA vs QDA ไม่เหมือนกัน!** | ใน QDA `model.scalings_` คือค่า **Eigenvalues (ความแปรปรวน $\lambda$)** ของ $\mathbf{\Sigma}_k$ แต่ใน LDA `model.scalings_` คือเมทริกซ์โปรเจกชันขนาด `(D, n_components)` | แยกบล็อกโค้ด `if self.is_qda:` กับ `else:` อย่างชัดเจนใน `_resolve_covariances` ห้ามใช้สูตรเดียวกันเด็ดขาด |
| 6 | **Numerical Overflow ใน `predict_proba` ของ QDA เมื่อจุดอยู่ไกลมาก** | ในบางกรณีที่ความแปรปรวนแคบมาก คะแนน Quadratic อาจติดลบสูงแต่ความต่างทำให้คลาสที่มีความแปรปรวนกว้างกว่าชนะในระยะไกล (Quadratic Wrap-around Effect) | แสดงเส้นแบ่งเขตโค้งของ QDA เต็มพื้นที่ Grid เพื่อให้ผู้ใช้เห็นพฤติกรรมธรรมชาติของ QDA ที่คลาสที่มี $\sigma$ กว้างกว่าจะโอบล้อมคลาสที่มี $\sigma$ แคบกว่าในระยะไกล |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `DiscriminantAnalysisAdapter` รองรับครบทั้ง `LinearDiscriminantAnalysis` และ `QuadraticDiscriminantAnalysis`
- [ ] สามารถกู้คืนรูปทรงทรงรี 3 มิติได้แม้ผู้ใช้เทรนโมเดลด้วยค่าเริ่มต้น `store_covariance=False` (ผ่าน `rotations_` และ `scalings_`)
- [ ] คำนวณ Eigendecomposition ด้วย `np.linalg.eigh` พร้อมบังคับ Right-handed Determinant ($\det(\mathbf{V}) = +1$) สำหรับหมุนทรงรีใน Three.js ได้อย่างถูกต้องแม่นยำ
- [ ] ฉาก 3 มิติแสดงแคปซูลทรงรีความแปรปรวนร่วม ($1\sigma, 2\sigma$) ที่หมุนเอียงตามสหสัมพันธ์จริง พร้อมเส้นวัดระยะ Mahalanobis เทียบกับ Euclidean
- [ ] แผง HUD แจ้งเตือน **Mahalanobis vs Euclidean Paradox** อัตโนมัติในเฟรมที่ระยะทางทั้งสองให้ผู้ชนะต่างคลาสกัน
