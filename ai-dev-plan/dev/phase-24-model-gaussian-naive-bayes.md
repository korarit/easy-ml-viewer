# Phase 24: Gaussian Naive Bayes Adapter & 3D Bivariate Bell Hills Visualizer

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลความน่าจะเป็นแบบต่อเนื่อง **`GaussianNB` (Gaussian Naive Bayes)** ซึ่งตั้งอยู่บนสมมติฐานว่าแต่ละฟีเจอร์มีการแจกแจงแบบปกติ (Gaussian Distribution) ที่เป็นอิสระต่อกันแบบมีเงื่อนไขภายใต้คลาสเดียวกัน โดยสร้างภาพจำลอง **ภูเขาระฆังคว่ำ 3 มิติแยกตามคลาส (3D Bivariate Gaussian Bell Hills)** ซ้อนทับกันในฉากเดียว เพื่อให้ผู้ใช้งานเห็นด้วยตาเปล่าว่าจุดข้อมูลแถวปัจจุบันตกลงบนไหล่เขาของคลาสใดสูงกว่ากัน

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.naive_bayes.GaussianNB` (รองรับทั้งการเทรนปกติด้วย `fit()` และการเทรนแบบออนไลน์ด้วย `partial_fit()`)
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: เปรียบเทียบภูเขาระฆังคว่ำ 2 ลูก พร้อมเส้นแบ่งเขตที่รอยตัดของภูเขาทั้งสอง
  - **Multiclass Classification ($C \ge 3$)**: แสดงหมู่ภูเขาระฆังคว่ำหลายลูกแยกสีตามคลาส

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของ Gaussian Likelihood และ Bayes' Theorem
กำหนดให้มีคลาส $C$ คลาส ($k \in \{0, \dots, C-1\}$) และข้อมูลนำเข้าขนาด $(N, D)$ โดยพารามิเตอร์ภายในของ `GaussianNB` คือ:
- ความน่าจะเป็นก่อนหน้า (Class Prior): $\pi_k = P(y = k)$ เก็บใน `model.class_prior_` ขนาด $(C,)$
- ค่าเฉลี่ยของแต่ละคลาสและฟีเจอร์: $\mu_{k,j}$ เก็บใน `model.theta_` ขนาด $(C, D)$
- ความแปรปรวนที่ผ่านการปรับเสถียรภาพแล้ว: $\sigma_{k,j}^2$ เก็บใน `model.var_` (หรือ `model.sigma_` ในเวอร์ชันเก่า) ขนาด $(C, D)$

1. **ความหนาแน่นความน่าจะเป็นแบบ Gaussian รายฟีเจอร์ (Univariate Feature Density)**:
   $$p(x_{t,j} \mid y = k) = \frac{1}{\sqrt{2\pi \sigma_{k,j}^2}} \exp\left( -\frac{(x_{t,j} - \mu_{k,j})^2}{2\sigma_{k,j}^2} \right)$$
2. **คะแนน Log-Likelihood รายฟีเจอร์ ($\ell_{t,k,j}$) และ Joint Log-Likelihood ($J_{t,k}$)**:
   $$\ell_{t,k,j} = \log p(x_{t,j} \mid y = k) = -\frac{1}{2}\log(2\pi \sigma_{k,j}^2) - \frac{(x_{t,j} - \mu_{k,j})^2}{2\sigma_{k,j}^2}$$
   $$J_{t,k} = \log P(y = k, \mathbf{x}_t) = \log \pi_k + \sum_{j=1}^{D} \ell_{t,k,j}$$
   (ค่า $J_{t,k}$ ตรงกับผลลัพธ์ของ `model._joint_log_likelihood(X)` 100%)
3. **ความน่าจะเป็นภายหลัง (Posterior Probability) ผ่าน Log-Sum-Exp**:
   $$P(y_t = k \mid \mathbf{x}_t) = \frac{\exp(J_{t,k})}{\sum_{c=0}^{C-1} \exp(J_{t,c})}$$

### 2.2 โครงสร้าง Array Shape และ Local Feature Contribution (Log-Bayes Factor Waterfall)
- `theta_` และ `var_`: ขนาด $(C, D)$
- `joint_log_likelihood`: ขนาด $(N, C)$
- `probabilities`: ขนาด $(N, C)$
- **การคำนวณ Local Feature Contribution (Waterfall)**:
  - สำหรับแถว $t$ และคลาสที่ชนะ (หรือคลาสเป้าหมาย $k$):
  - **โหมด Single-Class Log-Likelihood**: ค่าตั้งต้นคือ $\phi_0^{(k)} = \log \pi_k$ และแรงผลักของฟีเจอร์ $j$ คือ $\phi_{t,j}^{(k)} = \ell_{t,k,j}$ (ซึ่งมีค่าติดลบเสมอ ยิ่งติดลบน้อยคืออยู่ใกล้ยอดเขา $\mu_{k,j}$ มาก!)
  - **โหมด Binary Log-Odds Ratio (เปรียบเทียบคลาส $1$ เทียบกับคลาส $0$)**:
    $$\phi_0 = \log \frac{\pi_1}{\pi_0}, \quad \phi_{t,j} = \ell_{t,1,j} - \ell_{t,0,j}$$
    ทำให้เห็นทันทีว่าฟีเจอร์ $j$ สนับสนุนคลาส 1 (ค่าบวก) หรือสนับสนุนคลาส 0 (ค่าลบ) มากกว่ากัน!

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **Multi-Class 3D Bivariate Gaussian Bell Hills**:
   - สำหรับแต่ละคลาส $k \in \{0, \dots, C-1\}$ วาดพื้นผิวภูเขาระฆังคว่ำ 3 มิติตามสมการความหนาแน่น 2 ตัวแปรบนแกน $(f_x, f_y)$:
     $$Z_k(x, y) = \pi_k \cdot \exp\left( -\frac{(x - \mu_{k,x})^2}{2\sigma_{k,x}^2} - \frac{(y - \mu_{k,y})^2}{2\sigma_{k,y}^2} \right)$$
   - เนื่องจากสมมติฐานความเป็นอิสระแบบมีเงื่อนไข (Conditional Independence) ของ Naive Bayes ฐานของภูเขาระฆังคว่ำทุกใบจะมีแกนหลัก **ขนานกับแกนพิกัด $X$ และ $Y$ เสมอ (Axis-Aligned Elliptical Footprints)** ไม่มีการบิดเฉียงทแยงมุม!
   - ใช้สีประจำคลาสแบบกึ่งโปร่งแสง (`opacity: 0.55`, เปิด `depthWrite: false` สำหรับชั้นโปร่งใส) พร้อมเส้นวงแหวนความสูงเท่ากัน ($1\sigma, 2\sigma, 3\sigma$ Iso-density Contour Rings)
2. **Intersection Seam Line (เส้นรอยตัดแบ่งเขตระหว่างภูเขา)**:
   - บริเวณที่ไหล่เขาของคลาส $A$ และคลาส $B$ ตัดกัน ($Z_A(x,y) = Z_B(x,y)$) จะเกิดเส้นขอบเขตการตัดสินใจรูปภาคตัดกรวย (Parabola / Ellipse / Hyperbola) วาดด้วยเส้นท่อเรืองแสงสีขาวบนพื้นระนาบ
3. **Vertical Density Drop-Pins (เข็มวัดความสูงไหล่เขาเทียบทุกคลาส)**:
   - ณ พิกัด $(x_t, y_t)$ ของเฟรมปัจจุบัน จะมีเสาเข็มแนวตั้งพุ่งจากพื้น $Z=0$ ขึ้นไปแตะผิวภูเขาของแต่ละคลาส พร้อมจุดไฟสว่างที่ยอดเข็มแต่ละคลาส เพื่อให้เห็นชัดเจนว่าที่ตำแหน่งนี้ ภูเขาสีไหนมีความสูง ($P(\mathbf{x}_t, C_k)$) เหนือกว่ากัน

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Surface Mode Switcher**: สลับการแสดงผลฉาก 3D ระหว่าง:
  - **Generative Bell Hills ($P(x, y \mid C_k)\pi_k$)**: แสดงภูเขาระฆังคว่ำของทุกคลาสซ้อนกัน
  - **Posterior Probability Surface ($P(C_k \mid x, y)$)**: แสดงพื้นผิวความน่าจะเป็นสุทธิหลังหารด้วยผลรวมทุกคลาสแล้ว
- **Class Hill Visibility Checkboxes**: เปิด-ปิดการแสดงผลภูเขาของแต่ละคลาสเพื่อลดความซับซ้อนเมื่อมีหลายคลาส
- **Z-Score Distance Inspector**: ตารางแสดงระยะห่างเป็นจำนวนเท่าของส่วนเบี่ยงเบนมาตรฐาน ($z_{k,j} = \frac{|x_{t,j} - \mu_{k,j}|}{\sigma_{k,j}}$)

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - เสาเข็มวัดความสูง (Vertical Density Pins) เคลื่อนที่ไปตามพิกัด $(x_t, y_t)$ ของแต่ละเฟรม
   - ความสูงของเข็มแต่ละคลาสยืด-หดตามระดับความลาดชันของภูเขาแต่ละลูก พร้อมอัปเดตกราฟ Log-Bayes Factor Waterfall ใน HUD
2. **เมื่อลาก Scrubber ข้ามรอยตัดระหว่างภูเขาสองลูก**:
   - ผู้ใช้จะเห็นจังหวะที่เข็มของคลาส 0 ลดต่ำลงสวนทางกับเข็มของคลาส 1 ที่พุ่งสูงขึ้นแซงหน้า ณ จุดรอยตัดของภูเขาทั้งสองลูกพอดี
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - เมื่อลากลูกบอลเงาเข้าหายอดเขา $(\mu_{k,x}, \mu_{k,y})$ ของคลาสใด ค่า Z-Score จะลดเข้าใกล้ $0.0\sigma$ และความน่าจะเป็นของคลาสนั้นจะพุ่งขึ้นสู่จุดสูงสุดทันที

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล GaussianNB
```text
+-------------------------------------------------------------------------+
| [GAUSSIAN NAIVE BAYES HUD]  Classes: 3 | var_smoothing: 1e-09           |
+-------------------------------------------------------------------------+
| 3D Mode: [ *Generative Bell Hills (Joint Density)* ] [ Posterior P(C|x) ]|
| Visible Hills: [x] Class 0 (Blue)  [x] Class 1 (Orange)  [x] Class 2    |
+-------------------------------------------------------------------------+
| FRAME #0074 BAYESIAN POSTERIOR & BELL HILL HEIGHTS:                     |
|  Class 1 (Winner) : Posterior = 88.4% | Joint Log-Prob = -2.140         |
|  Class 0          : Posterior = 11.2% | Joint Log-Prob = -4.205         |
|  Class 2          : Posterior =  0.4% | Joint Log-Prob = -7.530         |
+-------------------------------------------------------------------------+
| FEATURE Z-SCORE & LOG-LIKELIHOOD BREAKDOWN (Target: Class 1 vs Class 0):|
|  Prior Log-Ratio log(π1/π0) : +0.000 (Equal Priors: 0.333 each)         |
|  petal_length (X=4.20)      : μ1=4.26, σ1=0.47 (0.13σ) -> LLR: +3.850   |
|  petal_width  (Y=1.30)      : μ1=1.33, σ1=0.20 (0.15σ) -> LLR: +2.410   |
|  sepal_length  (5.90)       : μ1=5.94, σ1=0.52 (0.08σ) -> LLR: +0.625   |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/probabilistic/gaussian_nb.py`
- `frontend/src/domains/stage3d/components/surfaces/GaussianBellHills3D.tsx`
- `frontend/src/domains/hud/components/inspectors/GaussianNBInspector.tsx`

### 5.2 โค้ดต้นแบบ `GaussianNBAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.naive_bayes import GaussianNB
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class GaussianNBAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (GaussianNB,)

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.classes_: np.ndarray = np.asarray(model.classes_)
        self.n_classes: int = len(self.classes_)
        self.theta_: np.ndarray = np.asarray(model.theta_, dtype=np.float64)  # (C, D)
        raw_var = getattr(model, "var_", getattr(model, "sigma_", None))
        self.var_: np.ndarray = np.maximum(np.asarray(raw_var, dtype=np.float64), 1e-12)  # (C, D)
        self.class_prior_: np.ndarray = np.asarray(model.class_prior_, dtype=np.float64)
        self.log_priors: np.ndarray = np.log(np.maximum(self.class_prior_, 1e-15))

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "model_family": "probabilistic_naive_bayes",
            "algorithm": "GaussianNB",
            "task_type": "binary_classification" if self.n_classes == 2 else "multiclass_classification",
            "classes": self.classes_.tolist(),
            "class_priors": self.class_prior_.tolist(),
            "theta_means": self.theta_.tolist(),
            "variances": self.var_.tolist(),
            "std_devs": np.sqrt(self.var_).tolist(),
            "var_smoothing": float(getattr(self.model, "var_smoothing", 1e-9)),
            "epsilon": float(getattr(self.model, "epsilon_", 0.0)),
        }

    def _per_feature_log_likelihood(self, X_arr: np.ndarray) -> np.ndarray:
        """คืนค่าเทนเซอร์ขนาด (N, C, D) ของ Log-Likelihood รายฟีเจอร์"""
        # X_arr: (N, 1, D), theta_: (1, C, D), var_: (1, C, D)
        diff_sq = (X_arr[:, np.newaxis, :] - self.theta_[np.newaxis, :, :]) ** 2
        log_2pi_var = np.log(2.0 * np.pi * self.var_[np.newaxis, :, :])
        return -0.5 * (log_2pi_var + (diff_sq / self.var_[np.newaxis, :, :]))

    def trace_internal_flow(self, X_batch: np.ndarray) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        preds = self.model.predict(X_arr)
        probas = self.model.predict_proba(X_arr)
        joint_log_prob = self.model.predict_joint_log_proba(X_arr) if hasattr(
            self.model, "predict_joint_log_proba"
        ) else self.model._joint_log_likelihood(X_arr)

        feat_ll = self._per_feature_log_likelihood(X_arr)  # (N, C, D)
        if self.n_classes == 2:
            # Log-Likelihood Ratio (Class 1 vs Class 0) per feature
            contributions = feat_ll[:, 1, :] - feat_ll[:, 0, :]
            base_intercepts = [float(self.log_priors[1] - self.log_priors[0])]
        else:
            win_idx = np.argmax(probas, axis=1)
            contributions = feat_ll[np.arange(len(X_arr)), win_idx, :]
            base_intercepts = self.log_priors.tolist()

        # คำนวณ Z-score ของคลาสที่ชนะสำหรับแสดงใน HUD
        win_c = np.argmax(probas, axis=1)
        z_scores = np.abs(X_arr - self.theta_[win_c]) / np.sqrt(self.var_[win_c])

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas.astype(np.float32).tolist(),
            decision_scores=joint_log_prob.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=base_intercepts,
            extra_telemetry={
                "winning_class_z_scores": z_scores.astype(np.float32).tolist(),
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

        # คำนวณความสูงของภูเขาระฆังคว่ำ 2 มิติของแต่ละคลาส (Normalized Peak Height scaled by Prior)
        bell_hills_per_class = []
        for k in range(self.n_classes):
            zx = ((xx - self.theta_[k, axis_x_idx]) ** 2) / (2.0 * self.var_[k, axis_x_idx])
            zy = ((yy - self.theta_[k, axis_y_idx]) ** 2) / (2.0 * self.var_[k, axis_y_idx])
            hill_k = self.class_prior_[k] * np.exp(-(zx + zy))
            bell_hills_per_class.append(hill_k.astype(np.float32).tolist())

        c_idx = min(target_class_idx, self.n_classes - 1)
        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=bell_hills_per_class[c_idx],
            auxiliary_grids={
                "bell_hills_per_class": bell_hills_per_class,
                "peaks_2d": self.theta_[:, [axis_x_idx, axis_y_idx]].tolist(),
                "stds_2d": np.sqrt(self.var_[:, [axis_x_idx, axis_y_idx]]).tolist(),
            },
            boundary_threshold=0.5,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [1, 0],
    "probabilities": [[0.112, 0.884, 0.004], [0.991, 0.009, 0.000]],
    "decision_scores": [[-4.205, -2.140, -7.530], [-0.850, -5.560, -14.200]],
    "feature_contributions": [[-0.410, -0.320, -1.410], [-0.120, -0.080, -0.650]],
    "base_intercepts": [-1.0986, -1.0986, -1.0986],
    "extra_telemetry": { "winning_class_z_scores": [[0.13, 0.15, 0.08], [0.05, 0.11, 0.20]] }
  },
  "SurfaceMesh3D": {
    "x_grid": [1.0, 7.0],
    "y_grid": [0.1, 2.5],
    "z_values": [[0.001, 0.312], [0.002, 0.285]],
    "auxiliary_grids": {
      "bell_hills_per_class": [[[0.32, 0.01]], [[0.01, 0.31]]],
      "peaks_2d": [[1.46, 0.24], [4.26, 1.33], [5.55, 2.03]],
      "stds_2d": [[0.17, 0.10], [0.47, 0.20], [0.55, 0.27]]
    },
    "boundary_threshold": 0.5
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **Analytical GPU Vertex Shader สำหรับภูเขาระฆังคว่ำ (`GaussianBellHills3D.tsx`)**:
   - เนื่องจากสมการภูเขาระฆังคว่ำ 2 มิติของแต่ละคลาสต้องการเพียงพารามิเตอร์ 5 ตัวคือ $(\mu_{k,x}, \mu_{k,y}, \sigma_{k,x}, \sigma_{k,y}, \pi_k)$ เราสามารถส่งพารามิเตอร์เหล่านี้เป็น Uniforms เข้าสู่ Vertex Shader ของ Three.js เพื่อคำนวณความสูงแกน $Z$ บน GPU ได้โดยตรง ช่วยลดเวลาเปลี่ยนแกน $(X, Y)$ ให้เป็น 0ms!
2. **การจำกัดจำนวนชั้นของภูเขาที่เรนเดอร์พร้อมกัน**:
   - หากมีจำนวนคลาสมากกว่า 8 คลาสขึ้นไป จะเรนเดอร์เฉพาะภูเขาของ Top-5 Classes ที่มีความน่าจะเป็นสูงสุดเพื่อป้องกันปัญหา Transparency Overdraw บน GPU

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`var_` vs `sigma_` ใน Scikit-learn ต่างเวอร์ชัน** | ใน Scikit-learn `< 1.0` ใช้ชื่อแอตทริบิวต์ `sigma_` แต่ตั้งแต่ `1.0+` เปลี่ยนเป็น `var_` และลบ `sigma_` ออกใน `1.2+` | ใช้ `getattr(model, "var_", getattr(model, "sigma_", None))` เสมอเพื่อรองรับโมเดล `.pkl` ทุกยุคสมัย |
| 2 | **Numerical Underflow เมื่อคูณความหนาแน่นหลายฟีเจอร์เข้าด้วยกัน** | เมื่อจำนวนฟีเจอร์ $D$ สูง (เช่น $D = 100$) ค่าความหนาแน่น $\prod_{j=1}^D p(x_j \mid C_k)$ จะเล็กกว่า `1e-308` จนกลายเป็น `0.0` | คำนวณทุกอย่างในสเกล Log (`_per_feature_log_likelihood`) เสมอ และสำหรับความสูงภูเขาในฉาก 3D ให้ Normalize ยอดเขาด้วย $\pi_k \exp(-z_x - z_y)$ เพื่อให้ยอดเขามีความสูงในช่วง $[0, \pi_k]$ เสมอ |
| 3 | **ฟีเจอร์มีค่าคงที่ในบางคลาส ($\sigma_{k,j}^2 \approx 0$)** | หากในคลาส $k$ ฟีเจอร์ $j$ มีค่าเท่ากันทุกแถว ความแปรปรวนดิบจะเป็น 0 ทำให้ภูเขาแคบเหมือนเข็มและเกิดการหารด้วยศูนย์ | Scikit-learn บวกค่า `epsilon_` (`var_smoothing`) ไว้แล้ว และใน Adapter ต้องบังคับ `np.maximum(raw_var, 1e-12)` ซ้ำอีกครั้งเสมอ |
| 4 | **จุดข้อมูล Outlier อยู่ห่างจากยอดเขาทุกคลาสเกิน $10\sigma$** | เมื่อจุดอยู่ไกลมาก ค่า `predict_proba` ของคลาสที่ชนะจะพุ่งเป็น `1.000` เป๊ะๆ แม้ว่าจุดนั้นจะอยู่บนพื้นราบห่างไกลจากทุกภูเขาก็ตาม | แสดงค่า `Winning Class Z-Score` ใน HUD หาก $\min_k \|z_k\| > 4.0\sigma$ ให้แสดงป้ายเตือน **"LOW DENSITY OUTLIER: Far from all Gaussian centers (> 4σ)"** |
| 5 | **ความสูงของ PDF มีค่ามากกว่า 1.0 เมื่อ $\sigma < \frac{1}{\sqrt{2\pi}} \approx 0.3989$** | ความหนาแน่นความน่าจะเป็น (Probability Density) ไม่ใช่ความน่าจะเป็น จึงมีค่าเกิน 1.0 ได้เมื่อส่วนเบี่ยงเบนมาตรฐานแคบมาก | ในโหมด `Generative Bell Hills` ให้แสดงสเกลความสูงสัมพัทธ์ $\pi_k \exp(-\frac{1}{2}\Delta^2)$ ในช่วง $[0, 1]$ พร้อมแสดงค่า Peak Density จริงใน Tooltip |
| 6 | **โมเดลเทรนด้วย `priors` ที่กำหนดเอง (`model.priors is not None`)** | ผู้ใช้อาจตั้งค่า `GaussianNB(priors=[0.1, 0.9])` ซึ่งไม่ตรงกับสัดส่วนจำนวนข้อมูลใน `class_count_` | อ่านค่า Prior ที่ใช้คำนวณจริงจาก `model.class_prior_` เสมอ และแสดงเปรียบเทียบกับ `model.class_count_` ในหน้าต่าง Metadata |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `GaussianNBAdapter` รองรับโมเดล `GaussianNB` ทั้งเวอร์ชันที่ใช้ `var_` และ `sigma_`
- [ ] ผลรวมของ `base_intercepts` ($\log \pi_k$) และ `feature_contributions` ($\sum_j \ell_{t,k,j}$) ตรงกับค่า `predict_joint_log_proba(X)` ในระดับความแม่นยำ `1e-6`
- [ ] ฉาก 3 มิติแสดงภูเขาระฆังคว่ำ **3D Bivariate Gaussian Bell Hills** แยกสีตามคลาส พร้อมเส้นวงแหวนส่วนเบี่ยงเบนมาตรฐาน ($1\sigma, 2\sigma$)
- [ ] มีเสาเข็มแนวตั้ง (Vertical Density Drop-Pins) เปรียบเทียบความสูงของไหล่เขาทุกคลาส ณ พิกัดของเฟรมปัจจุบัน
- [ ] แจ้งเตือนสถานะ `Low Density Outlier (> 4σ)` เมื่อจุดข้อมูลอยู่ห่างจากจุดศูนย์กลางของทุกคลาส
