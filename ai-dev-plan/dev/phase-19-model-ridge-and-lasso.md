# Phase 19: Ridge & Lasso Regression Adapters & 3D Regularized Weight Vectors

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลเชิงเส้นที่มีการควบคุมความซับซ้อนด้วย Regularization สองตระกูลหลัก ได้แก่ **Ridge ($L_2$ Penalty — การย่อขนาดน้ำหนัก)** และ **Lasso ($L_1$ Penalty — การคัดเลือกฟีเจอร์และตัดน้ำหนักเป็นศูนย์)** เพื่อให้ผู้ใช้งานมองเห็นความแตกต่างทางเรขาคณิตระหว่างการบีบลดความชันแบบนุ่มนวลของ Ridge กับการตัดแกนตัวแปรที่ไม่มีนัยสำคัญทิ้งจนแบนราบสนิท (`Zeroed-out Features`) ของ Lasso ในอวกาศ 3 มิติ

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - **กลุ่ม $L_2$ (Ridge)**: `Ridge`, `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`
  - **กลุ่ม $L_1$ (Lasso)**: `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Regression**: `Ridge`, `Lasso`, `LassoLars` (และรุ่น `CV` ทั้งหมด)
  - **Multi-output Regression**: รองรับเมื่อ `coef_` มีขนาด `(n_targets, n_features)`
  - **Binary & Multiclass Classification**: รองรับผ่าน `RidgeClassifier` และ `RidgeClassifierCV`

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของการทำนายและฟังก์ชันลงโทษ (Penalty Formulation)
กำหนดให้ข้อมูลนำเข้ามีขนาด $(N, D)$ เวกเตอร์น้ำหนัก $\mathbf{w} \in \mathbb{R}^D$ และจุดตัดแกน $b \in \mathbb{R}$:

1. **สมการทำนายเชิงเส้น (Prediction Equation)**:
   $$\hat{y}_t = \mathbf{w}^T \mathbf{x}_t + b = \sum_{j=1}^{D} w_j x_{t,j} + b$$
   - สำหรับ `RidgeClassifier`: โมเดลแปลงเป้าหมายเป็น $\{-1, +1\}$ และใช้ $\text{sign}(\hat{y}_t)$ สำหรับ Binary หรือ $\arg\max_k (\mathbf{w}_k^T \mathbf{x}_t + b_k)$ สำหรับ Multiclass
2. **ฟังก์ชันวัตถุประสงค์และผลกระทบต่อเวกเตอร์ $\mathbf{w}$**:
   - **Ridge ($L_2$ Regularization)**:
     $$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2N}\|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|_2^2 + \alpha \|\mathbf{w}\|_2^2 \implies \|\mathbf{w}\|_2 = \sqrt{\sum_{j=1}^{D} w_j^2}$$
     ทำให้น้ำหนัก $w_j$ ถูกย่อขนาดเข้าใกล้ศูนย์ตามสัดส่วน แต่ไม่เป็นศูนย์สนิท ($w_j \neq 0$)
   - **Lasso ($L_1$ Regularization)**:
     $$\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \frac{1}{2N}\|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|_2^2 + \alpha \|\mathbf{w}\|_1 \implies \|\mathbf{w}\|_1 = \sum_{j=1}^{D} |w_j|$$
     ผลจากมุมแหลมของ $L_1$ Diamond Constraint ทำให้เกิด **Sparsity**: ฟีเจอร์ที่มีความสำคัญต่ำกว่าเกณฑ์ $\alpha$ จะถูกตัดเป็น $w_j = 0.0$ ทันที!

### 2.2 โครงสร้าง Array Shape และ Local Feature Contribution (Waterfall)
- **Active vs Zeroed Feature Partitioning**:
  - กำหนดเกณฑ์ความแม่นยำเชิงตัวเลข $\epsilon = 10^{-12}$:
  - เซตฟีเจอร์ที่ยังมีผล (Active Set): $\mathcal{A} = \{j \in \{1,\dots,D\} \mid |w_j| > \epsilon\}$
  - เซตฟีเจอร์ที่ถูกตัดทิ้ง (Pruned/Zeroed Set): $\mathcal{Z} = \{j \in \{1,\dots,D\} \mid |w_j| \le \epsilon\}$
  - อัตราความโปร่งบาง (Sparsity Ratio): $S = \frac{|\mathcal{Z}|}{D} \in [0, 1]$
- **การคำนวณ Local Feature Contribution (Waterfall)**:
  - สำหรับแถว $t$: แรงผลักของฟีเจอร์ $j$ คือ $\phi_{t,j} = w_j x_{t,j}$
  - ฟีเจอร์ในเซต $\mathcal{Z}$ จะมี $\phi_{t,j} = 0.0$ เสมอในทุกแถวข้อมูล และถูกจัดกลุ่มใน Waterfall ภายใต้หมวด **"L1 Eliminated Features (0 Impact)"**

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **3D Tilted Hyperplane & Residual Error Pillars**:
   - แผ่นระนาบทำนาย 3 มิติ (`THREE.Mesh` โปร่งแสง `opacity: 0.65`) ที่เอียงตามความชัน $w_x$ และ $w_y$
   - สำหรับงาน Regression: วาดเสาพลังงานแนวตั้ง (Residual Pillars) เชื่อมระหว่างความสูงทำนาย $\hat{y}_t$ บนแผ่นระนาบ กับค่าจริง $y_t$ (สีเขียวหากคลาดเคลื่อนน้อย สีแดงหากคลาดเคลื่อนสูง)
2. **3D Force Vector Arrows (`THREE.ArrowHelper`)**:
   - บนจุดกึ่งกลางของระนาบ 3D มีลูกศรเวกเตอร์แยก 2 แกน:
     - ลูกศรสีฟ้า (`#38bdf8`) ชี้ตามแนวแกน $X$ ความยาวแปรผันตรงกับ $|w_x|$
     - ลูกศรสีม่วง (`#c084fc`) ชี้ตามแนวแกน $Y$ ความยาวแปรผันตรงกับ $|w_y|$
     - ลูกศรสีทอง (`#facc15`) แสดงเวกเตอร์ลัพธ์ (Resultant Gradient Vector $\nabla_{xy} \hat{y} = [w_x, w_y]^T$)
3. **Lasso Sparse Axis Dimming & Flatline Indicator**:
   - หากผู้ใช้เลือกแกน $X$ หรือ $Y$ เป็นฟีเจอร์ที่อยู่ในเซต $\mathcal{Z}$ ($w = 0.0$ จากการตัดของ Lasso):
     - เส้นกริดของแกนนั้นจะเปลี่ยนเป็นสีเทามืดแบบเส้นประ (`#475569`)
     - ป้าย hologram 3 มิติ (`@react-three/drei <Html>`) จะลอยขึ้นเหนือแกนนั้นพร้อมข้อความ **"L1 PRUNED: Weight = 0.000 (No Effect on Output)"**
     - ระนาบ 3D ในแนวแกนนั้นจะแบนราบสนิท (ความชัน 0 องศา)

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Regularization Norm Badge & Penalty Meter**: แสดงค่า $\alpha$ (หรือ `alpha_` ที่ได้จาก CV), ค่า $\|w\|_1$, ค่า $\|w\|_2^2$ และสัดส่วน Sparsity (เช่น `14 / 20 Features Active (30% Pruned)`)
- **Active / Zeroed Feature Filter Pills**: รายการปุ่มฟีเจอร์ที่แบ่งสีชัดเจนระหว่างฟีเจอร์ที่รอดชีวิต (สีเขียว/ฟ้า) กับฟีเจอร์ที่ถูก Lasso ตัดทิ้ง (สีเทาขีดฆ่า) คลิกเพื่อสลับแกน $X, Y$ ในฉาก 3D ได้ทันที
- **Target / Class Selector**: สำหรับ Multi-output Regression หรือ Multiclass `RidgeClassifier`

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลวิ่งไปตามพิกัด $(x_t, y_t)$ บนแผ่นระนาบเอียง พร้อมแสดงความยาวของ Residual Pillar (ระยะห่างระหว่าง $\hat{y}_t$ กับ $y_t$)
   - ในกรณี `RidgeClassifier` จะแสดงเส้นแบ่งเขต $w^T x + b = 0$ บนระนาบ และเปลี่ยนสีลูกบอลเมื่อข้ามฝั่งบวก/ลบ
2. **เมื่อลาก Scrubber ข้ามแถวที่มีค่า Outliers หรือเปลี่ยนคำตัดสิน**:
   - ลูกศรแรงผลักรายฟีเจอร์ ($w_x x_t$ และ $w_y y_t$) ยืด-หดตามค่าของข้อมูลแถวปัจจุบันอย่างต่อเนื่อง
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - **กรณีขยับฟีเจอร์ที่ Active ($w_j \neq 0$)**: ลูกบอลเงา (Ghost Sphere) ไต่ขึ้นลงตามความชันของแผ่นระนาบ 3D ทันที
   - **กรณีขยับฟีเจอร์ที่ถูก Lasso ตัดเป็นศูนย์ ($w_j = 0$)**: เมื่อผู้ใช้ลากสไลเดอร์ของฟีเจอร์ที่ถูกตัดทิ้ง ลูกบอลเงาจะเคลื่อนที่ไปตามแนวแกนนั้นแต่ **ระดับความสูง $\hat{y}$ จะไม่เปลี่ยนแปลงเลยแม้แต่ทศนิยมเดียว!** พร้อมไฟกระพริบเตือนที่ป้าย `L1 PRUNED (w = 0)` ทำให้ผู้ใช้ตระหนักเห็นภาพทันทีว่า Lasso ได้ตัดตัวแปรนี้ออกจากสมการอย่างสมบูรณ์แล้ว

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล Ridge & Lasso
```text
+-------------------------------------------------------------------------+
| [REGULARIZED LINEAR HUD]  Model: LassoCV | Penalty: L1 | alpha*: 0.042  |
+-------------------------------------------------------------------------+
| Sparsity Gauge: [████████████░░░░░░] 12 Active / 6 Zeroed (33.3% Sparse)|
| Norm Telemetry: ||w||_1 = 4.821   |   ||w||_2 = 1.940                   |
+-------------------------------------------------------------------------+
| FEATURE STATUS & 3D AXIS QUICK-ASSIGN:                                  |
|  [X] area_sqft (w=+0.84)  [Y] rooms (w=+0.31)  [Pruned] noise_1 (w=0.0) |
|  [ ] age_years (w=-0.42)  [Pruned] zip_rnd (w=0.0)                      |
+-------------------------------------------------------------------------+
| FRAME #0089 PREDICTION & RESIDUAL TRACE:                                |
|  Predicted (y_hat) :  245.80   | Actual (y_true) :  252.00              |
|  Residual Error    :   -6.20   | Squared Error   :   38.44              |
+-------------------------------------------------------------------------+
| FORCE VECTOR WATERFALL (y_hat = b + sum(w_i * x_i)):                    |
|  Intercept (b)     : +180.00  [==========|               ]              |
|  area_sqft (X)     :  +84.20  [          |========>      ]              |
|  age_years         :  -18.40  [       <==|               ]              |
|  noise_1 (L1=0)    :    0.00  [          | (L1 PRUNED)   ]              |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/linear/regularized.py`
- `frontend/src/domains/stage3d/components/surfaces/RegularizedPlane3D.tsx`
- `frontend/src/domains/stage3d/components/actors/ForceVectorArrows3D.tsx`

### 5.2 โค้ดต้นแบบ `RegularizedLinearAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import (
    Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV,
    Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
)
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class RegularizedLinearAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (
        Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV,
        Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,
    )
    ZERO_TOL = 1e-12

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.is_classifier = isinstance(model, (RidgeClassifier, RidgeClassifierCV))
        self.is_l1 = isinstance(model, (Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC))
        raw_coef = np.asarray(model.coef_, dtype=np.float64)
        self.coef_2d = np.atleast_2d(raw_coef)  # Shape (K, D)
        self.intercept_1d = np.atleast_1d(np.asarray(model.intercept_, dtype=np.float64))

    def extract_metadata(self) -> Dict[str, Any]:
        alpha_val = getattr(self.model, "alpha_", getattr(self.model, "alpha", 1.0))
        w_flat = self.coef_2d[0]
        active_indices = np.where(np.abs(w_flat) > self.ZERO_TOL)[0].tolist()
        zeroed_indices = np.where(np.abs(w_flat) <= self.ZERO_TOL)[0].tolist()

        return {
            "model_family": "regularized_linear",
            "algorithm": self.model.__class__.__name__,
            "penalty_type": "L1 (Lasso)" if self.is_l1 else "L2 (Ridge)",
            "task_type": "classification" if self.is_classifier else "regression",
            "alpha": float(np.asarray(alpha_val).flat[0]),
            "l1_norm": float(np.sum(np.abs(w_flat))),
            "l2_norm": float(np.linalg.norm(w_flat)),
            "active_features": [self.feature_names[i] for i in active_indices],
            "zeroed_features": [self.feature_names[i] for i in zeroed_indices],
            "sparsity_ratio": len(zeroed_indices) / max(1, len(self.feature_names)),
            "coef": self.coef_2d.tolist(),
            "intercept": self.intercept_1d.tolist(),
        }

    def trace_internal_flow(self, X_batch: np.ndarray, target_idx: int = 0) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        preds = self.model.predict(X_arr)
        w = self.coef_2d[min(target_idx, len(self.coef_2d) - 1)]
        b = float(self.intercept_1d[min(target_idx, len(self.intercept_1d) - 1)])
        contributions = X_arr * w[np.newaxis, :]  # (N, D)

        probas = None
        scores = (X_arr @ self.coef_2d.T) + self.intercept_1d[np.newaxis, :]
        if self.is_classifier:
            raw_dec = self.model.decision_function(X_arr)
            if raw_dec.ndim == 1:
                p1 = expit(raw_dec)
                probas = np.column_stack([1.0 - p1, p1]).astype(np.float32).tolist()
                scores = raw_dec.reshape(-1, 1)
            else:
                probas = softmax(raw_dec, axis=1).astype(np.float32).tolist()
                scores = raw_dec

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas,
            decision_scores=scores.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=[b],
            extra_telemetry={
                "is_pseudo_proba": self.is_classifier,
                "active_weight_count": int(np.sum(np.abs(w) > self.ZERO_TOL)),
            },
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 40, target_idx: int = 0
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        w = self.coef_2d[min(target_idx, len(self.coef_2d) - 1)]
        b = float(self.intercept_1d[min(target_idx, len(self.intercept_1d) - 1)])

        # Fast vectorized analytical plane evaluation without full matrix tiling
        off_axis_sum = float(np.dot(fixed_features, w) - w[axis_x_idx]*fixed_features[axis_x_idx] - w[axis_y_idx]*fixed_features[axis_y_idx] + b)
        zz = w[axis_x_idx] * xx + w[axis_y_idx] * yy + off_axis_sum

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=zz.astype(np.float32).tolist(),
            auxiliary_grids={
                "wx": float(w[axis_x_idx]),
                "wy": float(w[axis_y_idx]),
                "is_x_pruned": bool(abs(w[axis_x_idx]) <= self.ZERO_TOL),
                "is_y_pruned": bool(abs(w[axis_y_idx]) <= self.ZERO_TOL),
            },
            boundary_threshold=0.0 if self.is_classifier else None,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [245.80, 198.15],
    "probabilities": null,
    "decision_scores": [[245.80], [198.15]],
    "feature_contributions": [[84.20, -18.40, 0.0], [42.10, -3.95, 0.0]],
    "base_intercepts": [180.00],
    "extra_telemetry": { "is_pseudo_proba": false, "active_weight_count": 12 }
  },
  "SurfaceMesh3D": {
    "x_grid": [0.0, 10.0],
    "y_grid": [0.0, 5.0],
    "z_values": [[180.0, 188.4], [181.55, 189.95]],
    "auxiliary_grids": { "wx": 0.84, "wy": 0.31, "is_x_pruned": false, "is_y_pruned": false },
    "boundary_threshold": null
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **Analytical Plane Calculation แทนการสร้าง Dense Meshgrid `(N_grid, D)`**:
   - เนื่องจากสมการระนาบของ Ridge และ Lasso เป็นเชิงเส้นแท้ 100% เราไม่จำเป็นต้อง `np.tile` สร้างตารางขนาด $(1600, D)$ เพื่อโยนเข้า `model.predict()` แต่คำนวณเพียง `off_axis_sum` ครั้งเดียวด้วย `np.dot(fixed_features, w)` แล้วคำนวณ $Z = w_x X + w_y Y + \text{off\_axis\_sum}$ ลดการใช้ RAM และเวลาประมวลผลลงกว่า 95%
2. **ความละเอียดของ Grid สำหรับระนาบเส้นตรง**:
   - สำหรับแผ่นระนาบเชิงเส้น ใช้ Grid Resolution เพียง $20 \times 20$ (หรือ $2 \times 2$ สำหรับแผ่นระนาบเรียบ) ก็เพียงพอที่จะแสดงระนาบได้สมบูรณ์แบบ ช่วยประหยัด Vertex Buffer บนการ์ดจอ
3. **การจัดการ `ArrowHelper` ใน Three.js**:
   - ใช้ `arrowRef.current.setDirection(dirVec)` และ `arrowRef.current.setLength(len)` แทนการ Unmount/Mount `<arrowHelper>` ใหม่เมื่อผู้ใช้เปลี่ยนแกน $X, Y$

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`RidgeClassifier` ไม่มีเมธอด `predict_proba`** | `RidgeClassifier` คำนวณคะแนนผ่าน Least Squares บนเป้าหมาย $\{-1, +1\}$ จึงมีเฉพาะ `decision_function(X)` ไม่มี `predict_proba` | แปลงคะแนน `decision_function` ผ่าน `expit` (Binary) หรือ `softmax` (Multiclass) เพื่อแสดงความมั่นใจสัมพัทธ์ พร้อมแนบ Flag `is_pseudo_proba: true` และแสดงป้ายกำกับ `" Pseudo-Probability (from Ridge Score)"` ใน UI |
| 2 | **`Lasso` ตัดน้ำหนักฟีเจอร์ทุกตัวเป็น `0.0` ทั้งหมด** | หากตั้งค่า `alpha` สูงเกินไป โมเดล Lasso จะตัด `coef_` ทุกตัวเป็น `0.0` เหลือเพียงค่าเฉลี่ย `intercept_` | ตรวจจับเมื่อ `active_weight_count == 0` ให้แสดง Banner แจ้งเตือน **"All Features Pruned by High L1 Alpha ($\alpha$)"** และวาดระนาบขนานพื้นราบที่ความสูง $Z = b$ |
| 3 | **Multi-output Regression (`coef_` เป็น 2D `(n_targets, D)`)** | หากเทรน `Ridge` หรือ `Lasso` ทำนายหลายตัวแปรตามพร้อมกัน `coef_` จะมีขนาด `(M, D)` แทนที่จะเป็น 1D `(D,)` | บังคับ `np.atleast_2d(model.coef_)` เสมอใน `__init__` และเพิ่ม Dropdown เลือก Target Output index ในแผง HUD |
| 4 | **`RidgeCV` / `LassoCV` เก็บค่า Alpha ใน `alpha_` ไม่ใช่ `alpha`** | คลาสตระกูล `*CV` จะเก็บพารามิเตอร์ที่เลือกแล้วไว้ใน attribute `alpha_` (มีขีดล่าง) ในขณะที่คลาสปกติเก็บใน `alpha` | ใช้ `getattr(model, "alpha_", getattr(model, "alpha", 1.0))` เสมอเพื่อป้องกัน `AttributeError` |
| 5 | **Floating-Point Noise ใน Coordinate Descent ของ Lasso** | บางครั้ง Solver ของ Lasso หยุดที่ค่าใกล้ศูนย์มาก เช่น `1e-16` แทนที่จะเป็น `0.0` เป๊ะๆ | ใช้เกณฑ์ Threshold `abs(w_j) <= 1e-12` (`ZERO_TOL`) ในการตัดสินว่าฟีเจอร์นั้นถูกตัดเป็นศูนย์หรือไม่ |
| 6 | **`LassoLarsIC` ใช้เกณฑ์ AIC/BIC** | ผู้ใช้ที่อัปโหลด `LassoLarsIC` มักต้องการทราบค่า `criterion_` (`aic` หรือ `bic`) ที่ทำให้เลือก `alpha_` นั้น | ดึง `getattr(model, "criterion", None)` มาแสดงในส่วน Metadata ของ HUD เพิ่มเติม |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `RegularizedLinearAdapter` รองรับครบทั้ง `Ridge`, `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`, `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`
- [ ] สกัดรายการ Active Features และ Zeroed Features (`|w| <= 1e-12`) พร้อมคำนวณค่า $\|w\|_1$, $\|w\|_2$ และ Sparsity Ratio ได้อย่างถูกต้อง
- [ ] ฉาก 3 มิติแสดงลูกศรเวกเตอร์ความชัน (`ForceVectorArrows3D`) แยกตามแกน $X, Y$ และเวกเตอร์ลัพธ์
- [ ] เมื่อเลือกฟีเจอร์ที่ถูก Lasso ตัดเป็นศูนย์ (`w = 0`) บนแกน 3D จะแสดงเอฟเฟกต์ **Sparse Axis Dimming** พร้อมป้ายเตือน `L1 PRUNED (Weight = 0)`
- [ ] เมื่อขยับสไลเดอร์ของฟีเจอร์ที่ถูกตัดเป็นศูนย์ในโหมด What-If Sandbox ความสูงทำนาย $\hat{y}$ คงที่ 100% พร้อมเอฟเฟกต์แจ้งเตือนผู้ใช้
- [ ] รองรับ `RidgeClassifier` โดยไม่เกิด `AttributeError: predict_proba` พร้อมแสดงป้ายกำกับ `Decision Score / Pseudo-Proba` ชัดเจน
