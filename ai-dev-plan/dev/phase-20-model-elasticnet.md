# Phase 20: ElasticNet Adapter & L1/L2 Penalty Ratio 3D Inspector

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลตระกูล **ElasticNet** ซึ่งผสานจุดเด่นของ $L_1$ Regularization (การคัดเลือกฟีเจอร์แบบ Lasso) และ $L_2$ Regularization (การรักษากลุ่มฟีเจอร์ที่มีสหสัมพันธ์สูงแบบ Ridge) เข้าด้วยกันผ่านพารามิเตอร์สัดส่วน `l1_ratio` ($\rho$) โดยแสดงทั้งแผ่นระนาบทำนาย 3 มิติ วงแหวนแสดงรูปทรงขอบเขตข้อจำกัดลูกผสม (L1-L2 Rounded Diamond Constraint Ring) และมาตรวัดสัดส่วนแรงดึงดูดใน HUD

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.linear_model.ElasticNet`
  - `sklearn.linear_model.ElasticNetCV`
  - `sklearn.linear_model.MultiTaskElasticNet`
  - `sklearn.linear_model.MultiTaskElasticNetCV`
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Single-target Regression**: `ElasticNet`, `ElasticNetCV`
  - **Multi-task / Multi-output Regression**: `MultiTaskElasticNet`, `MultiTaskElasticNetCV` (ใช้ $L_{1,2}$ Mixed-Norm Penalty เพื่อคัดเลือกฟีเจอร์ชุดเดียวกันข้ามทุก Task)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของ ElasticNet และสัดส่วนโทษปรับ ($L_1 / L_2$ Decomposition)
กำหนดให้ข้อมูลนำเข้ามีขนาด $(N, D)$ พารามิเตอร์ความเข้มข้นรวมคือ $\alpha > 0$ และสัดส่วนผสมคือ $\rho = \text{l1\_ratio} \in [0, 1]$:

1. **ฟังก์ชันวัตถุประสงค์ของ `ElasticNet` (Single-target)**:
   $$\mathcal{L}_{\text{ElasticNet}}(\mathbf{w}) = \frac{1}{2N}\|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|_2^2 + \underbrace{\alpha \rho \|\mathbf{w}\|_1}_{\text{L1 Penalty } (P_{L1})} + \underbrace{\frac{1}{2}\alpha (1 - \rho) \|\mathbf{w}\|_2^2}_{\text{L2 Penalty } (P_{L2})}$$
   - สัมประสิทธิ์โทษปรับจริง: $\lambda_1 = \alpha \rho$ และ $\lambda_2 = \frac{1}{2}\alpha(1 - \rho)$
   - ค่าการทำนายรายแถว $t$: $\hat{y}_t = \sum_{j=1}^{D} w_j x_{t,j} + b$

2. **ฟังก์ชันวัตถุประสงค์ของ `MultiTaskElasticNet` ($M$ Tasks, $\mathbf{W} \in \mathbb{R}^{M \times D}$)**:
   $$\mathcal{L}_{\text{MT}}(\mathbf{W}) = \frac{1}{2N}\|\mathbf{Y} - \mathbf{X}\mathbf{W}^T - \mathbf{b}\|_F^2 + \alpha \rho \|\mathbf{W}\|_{2,1} + \frac{1}{2}\alpha (1 - \rho) \|\mathbf{W}\|_F^2$$
   โดยที่ $\|\mathbf{W}\|_{2,1} = \sum_{j=1}^{D} \sqrt{\sum_{m=1}^{M} w_{m,j}^2}$ ซึ่งบังคับให้ฟีเจอร์ที่ $j$ ถูกตัดเป็นศูนย์หรือคงไว้ **พร้อมกันในทุกๆ Task (Joint Feature Selection)**!

### 2.2 โครงสร้าง Array Shape และ Local Feature Contribution (Waterfall)
- **Weight Matrix (`coef_`)**:
  - สำหรับ `ElasticNet`: ขนาด $(D,)$ แปลงเป็น 2D ขนาด $(1, D)$
  - สำหรับ `MultiTaskElasticNet`: ขนาด $(M, D)$ โดย $M$ คือจำนวน Tasks
- **การคำนวณ Local Feature Contribution (Waterfall)**:
  - สำหรับแถว $t$ และ Task เป้าหมาย $m$: $\phi_{t,j}^{(m)} = w_{m,j} \cdot x_{t,j}$
  - คำนวณ **Effective Penalty Share** ของแต่ละฟีเจอร์ $j$:
    $$\text{Penalty}_{j} = \alpha \rho |w_{m,j}| + \frac{1}{2}\alpha(1 - \rho) w_{m,j}^2$$
    เพื่อให้ผู้ใช้เห็นว่าฟีเจอร์ตัวไหนแบกรับภาระ Regularization สูงสุดในสมการ

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **3D ElasticNet Hyperplane & Shrinkage Grid**:
   - แผ่นระนาบทำนาย 3 มิติที่แสดงความชันตาม $(w_x, w_y)$ พร้อมเส้นตารางเรืองแสงที่เปลี่ยนสีตามค่า `l1_ratio` (จากสีฟ้าครามเมื่อใกล้ 0 แบบ Ridge ไปเป็นสีม่วงอมชมพูเมื่อใกล้ 1 แบบ Lasso)
2. **3D L1-L2 Superellipse Constraint Aura (วงแหวนรูปทรงลูกผสมรอบจุดกำเนิดระนาบ)**:
   - วาดเส้นขอบเขตเรขาคณิตบนระนาบฐานตามสมการตัวแปรคู่:
     $$\rho (|u| + |v|) + (1 - \rho)(u^2 + v^2) = r^2$$
   - เมื่อ `l1_ratio = 0.0` วงแหวนจะเป็น **วงกลมสมบูรณ์แบบ (Ridge $L_2$ Ball)**
   - เมื่อ `0 < l1_ratio < 1` วงแหวนจะเป็น **สี่เหลี่ยมข้าวหลามตัดขอบโค้งมน (ElasticNet Rounded Diamond)**
   - เมื่อ `l1_ratio = 1.0` วงแหวนจะเป็น **สี่เหลี่ยมข้าวหลามตัดมุมแหลม (Lasso $L_1$ Diamond)**
   - ช่วยให้ผู้เรียนและวิศวกรเห็นเหตุผลทางเรขาคณิตทันทีว่าทำไมมุมแหลมของ `l1_ratio` จึงสามารถตัดน้ำหนักให้เป็นศูนย์ได้ในขณะที่ความโค้งของ $L_2$ ช่วยประคองกลุ่มตัวแปรที่มีสหสัมพันธ์กัน!
3. **Correlated Feature Grouping Arcs**:
   - หากทั้งแกน $X$ และแกน $Y$ มีน้ำหนักไม่เป็นศูนย์และเคลื่อนที่ไปในทิศทางส่งเสริมกัน จะมีเส้นโค้งเชื่อมระหว่างเวกเตอร์ $w_x$ และ $w_y$ แสดงเอฟเฟกต์ Grouping ของ ElasticNet

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **L1 / L2 Balance Gauge (Tug-of-War Bar)**: มาตรวัดแนวนอนแสดงสัดส่วน `l1_ratio` ($\rho$) เทียบกับ `1 - l1_ratio` พร้อมตัวเลขสัมประสิทธิ์จริง $\lambda_1 = \alpha\rho$ และ $\lambda_2 = \frac{1}{2}\alpha(1-\rho)$
- **Multi-Task Output Switcher**: ปุ่มสลับดูแต่ละ Task ($m = 0 \dots M-1$) สำหรับ `MultiTaskElasticNet` พร้อมตารางเปรียบเทียบน้ำหนัก $w_{m,j}$ ข้ามทุก Task
- **Convergence Telemetry Badge**: แสดงจำนวนรอบการคำนวณ Coordinate Descent ที่ใช้จริง (`model.n_iter_`) เทียบกับ `max_iter`

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลวิ่งไปตามระนาบ 3D พร้อมเสาวัดระยะคลาดเคลื่อน (Residual Pillar) เทียบกับค่าจริง $y_t$
   - แผง HUD อัปเดตกราฟแท่ง Waterfall รายแถว โดยแยกสีระหว่างฟีเจอร์ที่ยังมีน้ำหนัก (Active) กับฟีเจอร์ที่ถูกส่วนผสม $L_1$ ตัดทิ้งเป็นศูนย์
2. **เมื่อลาก Scrubber หรือสลับ Task ใน `MultiTaskElasticNet`**:
   - เมื่อสลับจาก Task 0 ไป Task 1 ความชันของระนาบ 3D จะหมุนเปลี่ยนมุมอย่างนุ่มนวล (Smooth Lerp) แต่ผู้ใช้จะสังเกตเห็นว่าฟีเจอร์ที่ถูกตัดเป็นศูนย์ (`L1/L2 Joint Pruned`) จะยังคงแบนราบเป็นศูนย์ในทุกๆ Task
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - ลูกบอลเงา (Ghost Sphere) เคลื่อนที่บนระนาบพร้อมแสดงการเปลี่ยนแปลงของ $\Delta \hat{y}$ สำหรับทุก Task พร้อมกันในตาราง Multi-Task Preview

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล ElasticNet
```text
+-------------------------------------------------------------------------+
| [ELASTICNET INSPECTOR HUD]  Model: ElasticNetCV | n_iter_: 148 / 1000   |
+-------------------------------------------------------------------------+
| Total Penalty (alpha*): 0.120   |   L1 Ratio (ρ*): 0.70 (70% L1 / 30% L2)|
| [ L2 Ridge (30%) <==========|=======================> (70%) L1 Lasso ]  |
| Effective λ1 (L1): 0.0840       |   Effective λ2 (L2): 0.0180           |
| Penalty Burden   : L1 = 0.312   |   L2 = 0.094   | Total Reg = 0.406    |
+-------------------------------------------------------------------------+
| TASK SELECTOR (MultiTask): [ *Task 0 (Price)* ] [ Task 1 (Rent Yield) ] |
| Sparsity Status          : 9 Active Features  | 5 Pruned Features (0.0) |
+-------------------------------------------------------------------------+
| FRAME #0210 PREDICTION & WATERFALL BREAKDOWN:                           |
|  Predicted (y_hat) :   41.25    |   Actual (y_true) :   39.80           |
|  Residual (y - ^y) :   -1.45    |   Abs Error       :    1.45           |
+-------------------------------------------------------------------------+
| FEATURE FORCE & REGULARIZATION PENALTY SHARE:                           |
|  Intercept (b)     : +25.00  [========|               ] (Reg: N/A   )   |
|  feature_rm (X)    : +12.40  [        |=======>       ] (Reg: 0.182 )   |
|  feature_lstat (Y) :  +3.85  [        |==>            ] (Reg: 0.064 )   |
|  feature_tax       :   0.00  [        | (PRUNED w=0)  ] (Reg: 0.000 )   |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/linear/elasticnet.py`
- `frontend/src/domains/stage3d/components/actors/ElasticNetConstraintRing3D.tsx`
- `frontend/src/domains/hud/components/inspectors/ElasticNetGaugePanel.tsx`

### 5.2 โค้ดต้นแบบ `ElasticNetAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
)
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class ElasticNetAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (
        ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
    )
    ZERO_TOL = 1e-12

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.is_multitask = isinstance(model, (MultiTaskElasticNet, MultiTaskElasticNetCV))
        self.coef_2d = np.atleast_2d(np.asarray(model.coef_, dtype=np.float64))  # (M, D)
        self.intercept_1d = np.atleast_1d(np.asarray(model.intercept_, dtype=np.float64))  # (M,)
        self.n_tasks = self.coef_2d.shape[0]

    def _get_alpha_and_l1_ratio(self) -> Tuple[float, float]:
        alpha = getattr(self.model, "alpha_", getattr(self.model, "alpha", 1.0))
        l1_ratio = getattr(self.model, "l1_ratio_", getattr(self.model, "l1_ratio", 0.5))
        return float(np.asarray(alpha).flat[0]), float(np.asarray(l1_ratio).flat[0])

    def extract_metadata(self) -> Dict[str, Any]:
        alpha, l1_ratio = self._get_alpha_and_l1_ratio()
        lambda_1 = alpha * l1_ratio
        lambda_2 = 0.5 * alpha * (1.0 - l1_ratio)

        # Joint norm across tasks for MultiTask or L1 norm for single task
        col_norms = np.linalg.norm(self.coef_2d, axis=0)
        active_idx = np.where(col_norms > self.ZERO_TOL)[0].tolist()
        zeroed_idx = np.where(col_norms <= self.ZERO_TOL)[0].tolist()

        l1_term = float(np.sum(col_norms)) if self.is_multitask else float(np.sum(np.abs(self.coef_2d[0])))
        l2_term = float(np.sum(self.coef_2d ** 2))
        n_iter = getattr(self.model, "n_iter_", 0)
        max_iter = getattr(self.model, "max_iter", 1000)

        return {
            "model_family": "regularized_linear",
            "algorithm": self.model.__class__.__name__,
            "task_type": "multitask_regression" if self.is_multitask else "regression",
            "n_tasks": self.n_tasks,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "l1_penalty_burden": lambda_1 * l1_term,
            "l2_penalty_burden": lambda_2 * l2_term,
            "n_iter": int(np.asarray(n_iter).max()) if n_iter is not None else 0,
            "max_iter": int(max_iter),
            "active_features": [self.feature_names[i] for i in active_idx],
            "zeroed_features": [self.feature_names[i] for i in zeroed_idx],
            "coef": self.coef_2d.tolist(),
            "intercept": self.intercept_1d.tolist(),
        }

    def trace_internal_flow(self, X_batch: np.ndarray, task_idx: int = 0) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        preds = np.asarray(self.model.predict(X_arr), dtype=np.float64)
        if preds.ndim == 1:
            preds_2d = preds.reshape(-1, 1)
        else:
            preds_2d = preds  # (N, M)

        t_idx = min(task_idx, self.n_tasks - 1)
        w = self.coef_2d[t_idx]
        b = float(self.intercept_1d[t_idx])
        contributions = X_arr * w[np.newaxis, :]  # (N, D)

        return InternalFlowBatch(
            predictions=preds_2d[:, t_idx].tolist(),
            probabilities=None,
            decision_scores=preds_2d.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=self.intercept_1d.tolist(),
            extra_telemetry={"active_task_idx": t_idx, "n_tasks": self.n_tasks},
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 30, task_idx: int = 0
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        t_idx = min(task_idx, self.n_tasks - 1)
        w = self.coef_2d[t_idx]
        b = float(self.intercept_1d[t_idx])

        off_axis = float(np.dot(fixed_features, w) - w[axis_x_idx]*fixed_features[axis_x_idx] - w[axis_y_idx]*fixed_features[axis_y_idx] + b)
        zz = w[axis_x_idx] * xx + w[axis_y_idx] * yy + off_axis
        _, l1_ratio = self._get_alpha_and_l1_ratio()

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=zz.astype(np.float32).tolist(),
            auxiliary_grids={
                "l1_ratio": l1_ratio,
                "wx": float(w[axis_x_idx]),
                "wy": float(w[axis_y_idx]),
            },
            boundary_threshold=None,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [41.25, 36.10],
    "probabilities": null,
    "decision_scores": [[41.25, 5.12], [36.10, 4.85]],
    "feature_contributions": [[12.40, 3.85, 0.0], [8.10, 3.00, 0.0]],
    "base_intercepts": [25.00, 3.20],
    "extra_telemetry": { "active_task_idx": 0, "n_tasks": 2 }
  },
  "SurfaceMesh3D": {
    "x_grid": [-2.0, 2.0],
    "y_grid": [-2.0, 2.0],
    "z_values": [[1.5, 26.3], [9.2, 34.0]],
    "auxiliary_grids": { "l1_ratio": 0.70, "wx": 6.20, "wy": 1.925 },
    "boundary_threshold": null
  }
}
```

### 5.4 โครงสร้าง Component Three.js (`ElasticNetConstraintRing3D.tsx`)
- คำนวณพิกัดเส้นรอบรูป (256 จุด) รอบมุม $\theta \in [0, 2\pi)$ โดยแก้สมการกำลังสองตามทิศทาง $(\cos\theta, \sin\theta)$:
  $$a r^2 + b r - c = 0, \quad a = (1 - \rho), \quad b = \rho(|\cos\theta| + |\sin\theta|)$$
- วาดด้วย `<Line points={ringPoints} color="#e879f9" lineWidth={2.5} />` บนระนาบฐานเพื่อให้เห็นรูปทรง Rounded Diamond ที่เปลี่ยนความโค้งมนตามค่า `l1_ratio` จริงของโมเดล

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **Multi-Task Prediction Caching**:
   - ในกรณี `MultiTaskElasticNet` ที่มีจำนวน Task สูง (เช่น $M = 20$) การคำนวณ `X_arr @ self.coef_2d.T` จะทำเพียงครั้งเดียวแล้วส่ง `decision_scores` ขนาด $(N, M)$ ไปให้ Frontend เพื่อให้เมื่อผู้ใช้คลิกสลับ Task ใน HUD สามารถคำนวณความสูงแกน $Z$ ใหม่บนฝั่ง Client ได้ทันทีโดยไม่ต้องยิง HTTP Request ซ้ำ
2. **Pre-allocated Constraint Ring Buffer**:
   - เส้นวงแหวน Superellipse (`ElasticNetConstraintRing3D`) ใช้จุดเพียง 256 Vertices คำนวณใน `useMemo([l1_ratio])` เพียงครั้งเดียวตอนโหลดโมเดล จึงไม่กินทรัพยากร CPU/GPU ระหว่างเล่นแอนิเมชัน VCR Playback

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`ElasticNetCV` รับ `l1_ratio` เป็น List/Array ตอนเทรน** | ผู้ใช้สามารถส่ง `l1_ratio=[0.1, 0.5, 0.9]` ให้ `ElasticNetCV` ทำให้ `model.l1_ratio` เป็น List ในขณะที่ค่าที่ชนะจริงอยู่ที่ `model.l1_ratio_` | ต้องอ่านจาก `getattr(model, "l1_ratio_", getattr(model, "l1_ratio", 0.5))` เสมอ และดึง `.flat[0]` หากเป็น Array |
| 2 | **`MultiTaskElasticNet` มี `coef_` ขนาด `(n_tasks, n_features)`** | ต่างจาก `ElasticNet` ปกติที่ `coef_` เป็น 1D `(n_features,)` หากไม่แปลงเป็น 2D เสมอจะทำให้เกิด `IndexError` | ใช้ `np.atleast_2d(model.coef_)` และ `np.atleast_1d(model.intercept_)` ตั้งแต่ใน `__init__` |
| 3 | **ค่าสุดโต่ง `l1_ratio == 0.0` หรือ `l1_ratio == 1.0`** | ในสมการสร้างวงแหวน Constraint Ring หาก $\rho = 1.0$ เทอมกำลังสอง $a = (1-\rho)$ จะกลายเป็น $0$ ทำให้หารด้วยศูนย์ในสูตรสมการกำลังสอง | แยกเงื่อนไขชัดเจน: ถ้า `abs(1.0 - rho) < 1e-6` ให้ใช้สมการเส้นตรง $r = \frac{c}{|\cos\theta| + |\sin\theta|}$ แทนสมการกำลังสอง |
| 4 | **Coordinate Descent ไม่ลู่เข้า (`n_iter_ == max_iter`)** | เมื่อฟีเจอร์มี Collinearity สูงและ $\alpha$ น้อย โมเดลอาจหยุดที่ `max_iter` โดยยังไม่ลู่เข้า | ตรวจสอบ `n_iter >= max_iter` ใน Metadata หากพบให้แสดงป้ายเตือนสีเหลืองใน HUD ว่า **"Convergence Warning: Reached max_iter"** |
| 5 | **Constraint `positive=True` ใน ElasticNet** | Scikit-learn รองรับ `ElasticNet(positive=True)` ซึ่งบังคับให้ $w_j \ge 0$ ทุกตัว | หาก `getattr(model, "positive", False)` เป็น `True` ให้แสดงป้ายกำกับ `"Non-negative Weights Constraint Active (w >= 0)"` และวาดวงแหวน Constraint เฉพาะจตุภาคที่ 1 ($u \ge 0, v \ge 0$) |
| 6 | **`MultiTaskElasticNet` ไม่รองรับ Sparse Input ในบางเวอร์ชัน** | `MultiTaskElasticNet` บังคับให้ข้อมูล `X` ต้องเป็น Dense Array | แปลง `X_batch.toarray()` อัตโนมัติหากตรวจพบ `scipy.sparse.issparse(X_batch)` ก่อนส่งเข้า `model.predict` |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `ElasticNetAdapter` รองรับครบทั้ง `ElasticNet`, `ElasticNetCV`, `MultiTaskElasticNet`, และ `MultiTaskElasticNetCV`
- [ ] สกัดค่า `alpha_`, `l1_ratio_`, $\lambda_1 = \alpha\rho$, $\lambda_2 = \frac{1}{2}\alpha(1-\rho)$ และภาระโทษปรับจริงของ $L_1$ เทียบกับ $L_2$ ได้ถูกต้อง 100%
- [ ] แผง HUD แสดงมาตรวัด **L1 / L2 Balance Gauge** พร้อมตัวบ่งชี้สถานะการลู่เข้า (`n_iter_ / max_iter`)
- [ ] ฉาก 3 มิติแสดงวงแหวน **3D L1-L2 Superellipse Constraint Aura** ที่เปลี่ยนรูปทรงตั้งแต่ Circle ($\rho=0$) $\to$ Rounded Diamond ($0<\rho<1$) $\to$ Sharp Diamond ($\rho=1$) อย่างถูกต้องโดยไม่มีบั๊กหารด้วยศูนย์
- [ ] รองรับการสลับดูแต่ละ Task ใน `MultiTaskElasticNet` ทั้งบนพื้นผิว 3D และในกราฟ Waterfall
