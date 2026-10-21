# Phase 18: Logistic Regression Adapter & 3D Sigmoid S-Surface Visualizer

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลจำแนกประเภทเชิงเส้นพื้นฐานที่สำคัญที่สุดอย่าง **Logistic Regression** โดยแปลงสมการ Log-Odds เชิงเส้นและการบีบอัดผ่านฟังก์ชัน Sigmoid / Softmax ให้กลายเป็น **พื้นผิวคลื่นรูปตัว S 3 มิติ (3D Sigmoid S-Surface)** พร้อมเส้นขอบเขตการตัดสินใจเรืองแสง ($P = 0.5$ หรือขอบเขตชนะสูงสุดในกรณี Multiclass) เพื่อให้ผู้ใช้งานเข้าใจกลไกการเปลี่ยนผ่านความน่าจะเป็นอย่างเป็นธรรมชาติ

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.linear_model.LogisticRegression` (รองรับทุก Solver: `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga`)
  - `sklearn.linear_model.LogisticRegressionCV` (ดึงค่า Hyperparameter `C_` ที่ดีที่สุดจาก Cross-Validation มาแสดงอัตโนมัติ)
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และรองรับการ Export เป็น `.mlv` Self-contained Bundle)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: คำนวณผ่านฟังก์ชัน Logistic Sigmoid $\sigma(z)$ มาตรฐาน
  - **Multiclass Classification (One-vs-Rest / Multinomial Softmax)**: รองรับทั้ง `multi_class='ovr'` และ `multi_class='multinomial'`
  - **หมายเหตุ**: ไม่รองรับงาน Regression และ Multi-output Classification โดยตรง (หากพบจะแจ้งเตือนให้แยกตัวแปรตามเป็นรายโมเดล)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของการทำนาย
กำหนดให้ข้อมูลนำเข้ามีขนาด $(N, D)$ โดยที่ $N$ คือจำนวนแถวข้อมูล และ $D$ คือจำนวนฟีเจอร์ทั้งหมด และจำนวนคลาสทั้งหมดคือ $C$:

1. **กรณี Binary Classification ($C = 2$)**:
   - พารามิเตอร์ภายในโมเดล: `coef_` มีขนาด $(1, D)$ แทนด้วยเวกเตอร์น้ำหนัก $\mathbf{w} \in \mathbb{R}^D$ และ `intercept_` มีขนาด $(1,)$ แทนด้วย $b \in \mathbb{R}$
   - **คะแนน Log-Odds (Logit Score)** สำหรับแถวที่ $t \in \{0, \dots, N-1\}$:
     $$z_t = \mathbf{w}^T \mathbf{x}_t + b = \sum_{j=1}^{D} w_j x_{t,j} + b$$
   - **ความน่าจะเป็นของคลาสบวก ($y = 1$)**:
     $$P(y_t = 1 \mid \mathbf{x}_t) = \sigma(z_t) = \frac{1}{1 + e^{-z_t}}, \quad P(y_t = 0 \mid \mathbf{x}_t) = 1 - \sigma(z_t)$$

2. **กรณี Multiclass Classification ($C > 2$, Multinomial Softmax)**:
   - พารามิเตอร์ภายในโมเดล: `coef_` มีขนาด $(C, D)$ แทนด้วย $\mathbf{W} \in \mathbb{R}^{C \times D}$ และ `intercept_` มีขนาด $(C,)$ แทนด้วย $\mathbf{b} \in \mathbb{R}^C$
   - **เวกเตอร์ Logit รายคลาส**: $\mathbf{z}_t = \mathbf{W} \mathbf{x}_t + \mathbf{b} \in \mathbb{R}^C$
   - **ความน่าจะเป็นผ่าน Softmax**:
     $$P(y_t = k \mid \mathbf{x}_t) = \frac{\exp(z_{t,k} - \max_m z_{t,m})}{\sum_{c=1}^{C} \exp(z_{t,c} - \max_m z_{t,m})}$$

### 2.2 โครงสร้าง Array Shape และ Local Feature Contribution (Waterfall)
- **Input Tensor (`X_batch`)**: `float64` หรือ `float32` ขนาด $(N, D)$
- **Decision Logits (`logits`)**: ขนาด $(N, 1)$ สำหรับ Binary หรือ $(N, C)$ สำหรับ Multiclass
- **Probabilities (`probas`)**: ขนาด $(N, C)$ เสมอ (ผลรวมแต่ละแถวเท่ากับ $1.0$)
- **การคำนวณ Local Feature Contribution (Log-Odds Waterfall)**:
  - เพื่อให้ผลรวมของ Waterfall แม่นยำตรงตามหลักการ Additive Explanation บนสเกล Log-Odds:
  - ค่าเริ่มต้น (Base Value) ของคลาส $k$: $\phi_0^{(k)} = b_k + \sum_{j=1}^{D} w_{k,j} \bar{x}_j$ (โดย $\bar{x}_j$ คือค่าเฉลี่ยหรือจุดอ้างอิงของฟีเจอร์ $j$ หากไม่ได้ระบุจะใช้ $\phi_0^{(k)} = b_k$)
  - แรงผลักของฟีเจอร์ที่ $j$ ในแถว $t$: $\phi_{t,j}^{(k)} = w_{k,j} \cdot x_{t,j}$
  - ผลรวมตรวจสอบ: $b_k + \sum_{j=1}^{D} \phi_{t,j}^{(k)} = z_{t,k}$ (ตรงกับ `decision_function` 100%)

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **3D Sigmoid S-Surface Mesh**:
   - แกน $X$ และแกน $Y$ แทนคู่ฟีเจอร์ที่ผู้ใช้เลือก ($f_x, f_y$) ส่วนความสูงแกน $Z \in [0.0, 1.0]$ แทนค่าความน่าจะเป็น $P(y = k_{\text{target}} \mid \mathbf{x})$
   - ใช้ `THREE.BufferGeometry` ขนาด Grid มาตรฐาน $60 \times 60$ จุด ($3,600$ Vertices) เพื่อความโค้งมนของช่วงไหล่คลื่น Sigmoid
   - **Custom Diverging Probability Shader**: ระบายสีตามระดับความสูง $Z$:
     - $Z \to 0.0$: สีน้ำเงินเข้ม (`#1e3a8a`) แทนฝั่งคลาส 0
     - $Z = 0.5$: สีขาวอมฟ้าเรืองแสง (`#f8fafc`) แทนโซนความไม่แน่นอนสูงสุด (Maximum Entropy Zone)
     - $Z \to 1.0$: สีส้มแดงสด (`#ea580c`) แทนฝั่งคลาสเป้าหมาย
2. **Glowing Iso-Probability Contour Line ($P = 0.5$ Decision Boundary)**:
   - วาดเส้นท่อเรืองแสง (`THREE.TubeGeometry` หรือ `drei/Line` ความหนา `3.5px` สีเขียวมรกต `#10b981`) ลากผ่านจุดบนพื้นผิวที่ $z_t = 0$ ($P = 0.5$)
   - เส้นท่อมีแอนิเมชันวิ่งเป็นจังหวะ (Pulse Shader) เพื่อเน้นแนวเขตแดนเปลี่ยนคำตัดสิน
3. **Log-Odds Tangent Force Vector (`ArrowHelper`)**:
   - ลูกศร 3 มิติสีทองพุ่งออกจากจุดกึ่งกลางขอบเขตการตัดสินใจในทิศทางของเวกเตอร์ความชัน $(\frac{w_x}{\|\mathbf{w}_{xy}\|}, \frac{w_y}{\|\mathbf{w}_{xy}\|})$ แสดงทิศทางที่ทำให้ค่า Log-Odds เพิ่มขึ้นเร็วที่สุด
4. **Data Point Drop-Line & Probability Ripple**:
   - ลูกบอลข้อมูลของเฟรมปัจจุบัน $t$ จะลอยอยู่บนผิว S-Surface ณ ความสูง $Z_t = P(y = k \mid \mathbf{x}_t)$ พร้อมเส้นประแนวตั้งพุ่งลงไปยังพื้นระนาบ $Z = 0$ และวงแหวนคลื่นน้ำ (Ripple Ring) กระจายตัวเมื่อเกิดการเปลี่ยนคลาส

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Target Class Surface Selector**: ปุ่มสลับดูพื้นผิวความน่าจะเป็นของแต่ละคลาส (`Class 0`, `Class 1`, ..., `Class K`) หรือโหมด **Winning Class Envelope** ($\max_k P(y=k)$)
- **Z-Axis Metric Toggle**: สวิตช์สลับความสูงแกน $Z$ ระหว่าง **Probability Space** ($\sigma(z) \in [0, 1]$ รูปตัว S) และ **Log-Odds Linear Space** ($z \in (-\infty, +\infty)$ รูปแผ่นระนาบเอียง)
- **Decision Threshold Slider**: สไลเดอร์ปรับเกณฑ์ตัดสินใจจากค่าเริ่มต้น `0.50` เป็นค่าใดๆ ในช่วง `[0.05, 0.95]` โดยเส้นขอบเขตเรืองแสงบนเนิน 3D จะเลื่อนไหลขึ้นลงตามระดับความสูงที่เลือกแบบเรียลไทม์

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลหลักเคลื่อนที่ไต่ระดับไปตามความโค้งของเนิน Sigmoid S-Surface
   - แผง HUD ด้านขวาแสดงหลอดวัด Log-Odds ($z_t$) ที่วิ่งผ่านสมการ $\sigma(z_t)$ ไปเป็นเปอร์เซ็นต์ความมั่นใจ พร้อมกราฟ Waterfall แยกส่วนประกอบ $w_j x_{t,j}$
2. **เมื่อลาก Scrubber ข้ามจุดเปลี่ยนคำตัดสิน (Decision Boundary Crossing)**:
   - ทันทีที่ลูกบอลวิ่งข้ามเส้นเรืองแสง $P = 0.5$ (ค่า $z_t$ เปลี่ยนเครื่องหมายระหว่างบวกและลบ) เส้นขอบเขตจะสว่างวาบ และป้ายสถานะจะแสดง Badge **"BOUNDARY CROSSED: Class 0 → Class 1"**
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - ลูกบอลเงา (Ghost Sphere) จะไต่ขึ้นหรือไหลลงตามความลาดชันของคลื่น S-Surface ทันที พร้อมเส้นโค้ง (Geodesic Trajectory) เชื่อมระหว่างจุดจริงกับจุดจำลอง
   - หากผู้ใช้ปรับค่าฟีเจอร์ที่ไม่ได้อยู่บนแกน $X, Y$ (Off-axis feature $f_m$) ค่า $\sum_{m \notin \{x,y\}} w_m x_m$ จะเปลี่ยนไป ส่งผลให้ **คลื่น S-Surface ทั้งแผ่นเลื่อนตัวไปข้างหน้าหรือถอยหลัง (Phase Shift)** แบบเรียลไทม์ 60FPS!

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล Logistic Regression
```text
+-------------------------------------------------------------------------+
| [LOGISTIC REGRESSION HUD]  Solver: lbfgs | Penalty: l2 | C: 1.00        |
+-------------------------------------------------------------------------+
| Surface View: [ (o) Probability S-Curve ]  [ ( ) Linear Log-Odds (z) ]  |
| Target Class: [ Class 0 ] [ *Class 1 (Positive)* ] [ Winning Envelope ] |
| Threshold τ : [=======(0.50)=======] (Logit Cutoff z = 0.00)            |
+-------------------------------------------------------------------------+
| FRAME #0142 INTERNAL FLOW TRACE:                                        |
|  1. Linear Log-Odds (z_t) :  +1.734  [==========|=====>    ] (+Logit)   |
|  2. Sigmoid Transform σ(z):   85.0%  [=================>   ] ( > 50% )  |
|  3. Final Verdict         :  CLASS 1 (Confidence: High | Margin: +35%)  |
+-------------------------------------------------------------------------+
| TOP LOG-ODDS CONTRIBUTORS (w_i * x_i + b):                              |
|  Intercept (b)       : -0.450  [      <===|            ]                |
|  feature_income (X)  : +1.620  [          |=========>  ]                |
|  feature_age (Y)     : +0.810  [          |====>       ]                |
|  feature_debt        : -0.246  [        <=|            ]                |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/linear/logistic_regression.py`
- `frontend/src/domains/stage3d/components/surfaces/SigmoidSurface3D.tsx`
- `frontend/src/domains/hud/components/inspectors/LogisticInspectorPanel.tsx`

### 5.2 โค้ดต้นแบบ `LogisticRegressionAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class LogisticRegressionAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (LogisticRegression, LogisticRegressionCV)

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.classes_: np.ndarray = np.asarray(model.classes_)
        self.is_binary: bool = len(self.classes_) == 2
        self.coef_: np.ndarray = np.asarray(model.coef_, dtype=np.float64)
        self.intercept_: np.ndarray = np.asarray(model.intercept_, dtype=np.float64)

    def extract_metadata(self) -> Dict[str, Any]:
        c_val = getattr(self.model, "C_", getattr(self.model, "C", 1.0))
        if isinstance(c_val, np.ndarray):
            c_val = float(c_val.flat[0])
        return {
            "model_family": "linear_classification",
            "algorithm": self.model.__class__.__name__,
            "task_type": "binary_classification" if self.is_binary else "multiclass_classification",
            "classes": self.classes_.tolist(),
            "hyperparameters": {
                "penalty": getattr(self.model, "penalty", "l2"),
                "C": float(c_val),
                "solver": getattr(self.model, "solver", "lbfgs"),
            },
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_.tolist(),
        }

    def trace_internal_flow(self, X_batch: np.ndarray) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        logits = self.model.decision_function(X_arr)
        probas = self.model.predict_proba(X_arr)
        preds = self.model.predict(X_arr)

        if self.is_binary:
            # Shape (N,) -> (N, 1) and per-feature contributions (N, D)
            logits_2d = logits.reshape(-1, 1)
            contributions = X_arr * self.coef_[0:1, :]  # (N, D)
            intercept_list = [float(self.intercept_[0])]
        else:
            logits_2d = logits  # (N, C)
            # For Winning Class or Class 0 default waterfall: shape (N, C, D) compressed to active class
            winning_idx = np.argmax(probas, axis=1)
            contributions = X_arr * self.coef_[winning_idx, :]  # (N, D)
            intercept_list = self.intercept_.tolist()

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas.astype(np.float32).tolist(),
            decision_scores=logits_2d.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=intercept_list,
            extra_telemetry={"is_binary": self.is_binary},
        )

    def compute_surface_mesh(
        self,
        axis_x_idx: int,
        axis_y_idx: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        fixed_features: np.ndarray,
        grid_resolution: int = 60,
        target_class_idx: int = 1,
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        n_points = grid_resolution * grid_resolution

        grid_X = np.tile(fixed_features.astype(np.float64), (n_points, 1))
        grid_X[:, axis_x_idx] = xx.ravel()
        grid_X[:, axis_y_idx] = yy.ravel()

        probas = self.model.predict_proba(grid_X)
        logits = self.model.decision_function(grid_X)
        z_surface = probas[:, target_class_idx].reshape(grid_resolution, grid_resolution)

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=z_surface.astype(np.float32).tolist(),
            auxiliary_grids={
                "logits": logits.reshape(grid_resolution, grid_resolution, -1).astype(np.float32).tolist()
            },
            boundary_threshold=0.5,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [1, 0],
    "probabilities": [[0.149, 0.851], [0.782, 0.218]],
    "decision_scores": [[1.734], [-1.277]],
    "feature_contributions": [[1.620, 0.810, -0.246], [-0.540, -0.287, 0.0]],
    "base_intercepts": [-0.450],
    "extra_telemetry": { "is_binary": true }
  },
  "SurfaceMesh3D": {
    "x_grid": [-3.0, -2.89, 3.0],
    "y_grid": [-3.0, -2.89, 3.0],
    "z_values": [[0.012, 0.015], [0.981, 0.988]],
    "auxiliary_grids": { "logits": [[[-4.39], [-4.18]], [[3.94], [4.41]]] },
    "boundary_threshold": 0.5
  }
}
```

### 5.4 โครงสร้าง Component Three.js (`SigmoidSurface3D.tsx`)
- รับ Props `surfaceMesh: SurfaceMesh3D`, `coef2D: [number, number]`, `interceptOffset: number`, `threshold: number`
- ใช้ `useMemo` สร้าง `THREE.PlaneGeometry(width, height, res - 1, res - 1)` และอัปเดต `position.array` แกน $Z$ ใน `useEffect` หรือผ่าน Vertex Shader โดยส่ง `w_x, w_y, b_offset` เป็น Uniforms เพื่อให้การเลื่อน What-If Slider ของฟีเจอร์แกนอื่นๆ คำนวณ $\sigma(w_x x + w_y y + b_{\text{rest}})$ บน GPU ได้ทันทีในระดับ 60FPS โดยไม่ต้องเรียก API ซ้ำ!

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การคำนวณ Sigmoid บน GPU Shader สำหรับ What-If Sandbox**:
   - เนื่องจากสมการของ Logistic Regression ใน 2 มิติคือ $\sigma(w_x x + w_y y + c_{\text{off-axis}})$ เราจึงส่งเพียงเวกเตอร์ `coef_` และ `intercept_` ไปยัง Frontend ครั้งเดียวตอนโหลดโมเดล เมื่อผู้ใช้ลากสไลเดอร์ในโหมด What-If ตัว Vertex Shader จะคำนวณความสูง $Z$ ใหม่บน GPU โดยตรง ลดภาระ CPU Backend เป็น 0% และใช้ RAM เพียงไม่กี่กิโลไบต์
2. **Vectorized BLAS Execution บน Backend**:
   - การคำนวณ `feature_contributions` สำหรับคิวข้อมูลขนาดใหญ่ (เช่น $10,000$ แถว) ใช้ Broadcasting `X_arr * self.coef_[0:1, :]` แทนการใช้ Python `for` loop และแปลงผลลัพธ์เป็น `float32` ก่อน Serialize ลดขนาด Payload ลง 50%
3. **การป้องกัน WebGL BufferGeometry Memory Leak**:
   - ห้ามสร้าง `new THREE.BufferGeometry()` ใหม่ทุกเฟรมเมื่อสลับคลาสหรือปรับ Threshold ให้จอง `Float32Array(60 * 60 * 3)` ไว้ล่วงหน้าใน `useRef` แล้วอัปเดตค่าภายใน Array เดิมพร้อมตั้งค่า `geometry.attributes.position.needsUpdate = true` และเรียก `geometry.dispose()` ใน Cleanup function ของ `useEffect` เสมอ

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **Binary vs Multiclass `coef_` Shape ไม่เหมือนกัน** | ใน Scikit-learn กรณี $C=2$ `coef_` มีขนาด `(1, D)` และ `decision_function` คืนค่า 1D `(N,)` แต่กรณี $C>2$ `coef_` มีขนาด `(C, D)` และคืนค่า 2D `(N, C)` | ตรวจสอบ `len(model.classes_) == 2` เสมอ หากเป็น Binary ให้ `.reshape(-1, 1)` เพื่อให้ Schema ของ `decision_scores` เป็น 2D `(N, 1)` สม่ำเสมอ |
| 2 | **Overflow ในฟังก์ชัน Exponential ($\exp(-z)$)** | หากข้อมูลแถวใดมีค่า Outliers สูงมากทำให้ $z_t < -710$ การคำนวณ `np.exp(-z)` แบบเขียนเองจะเกิด `RuntimeWarning: overflow` | ห้ามเขียน `1 / (1 + np.exp(-z))` ด้วยตนเอง ให้ใช้ `scipy.special.expit(z)` และ `scipy.special.softmax(z, axis=-1)` ที่มีระบบป้องกัน Overflow เสมอ |
| 3 | **โมเดลตั้งค่า `fit_intercept=False`** | เมื่อผู้ใช้เทรนโดยปิด Intercept ค่า `model.intercept_` อาจเป็น `0.0` (Scalar) หรือ Array `[0.]` | แปลงด้วย `np.atleast_1d(model.intercept_)` เสมอ และแสดงป้ายใน HUD ว่า `Intercept: 0.0 (Disabled)` |
| 4 | **`penalty=None` หรือ `'l1'` ทำให้บางฟีเจอร์มีน้ำหนักเป็น 0** | เมื่อใช้ `penalty='l1'` (`solver='liblinear'` หรือ `'saga'`) น้ำหนักของทั้งแกน $X$ และ $Y$ ที่เลือกอาจเป็น `0.0` พร้อมกัน | หาก $w_x = 0$ และ $w_y = 0$ เส้นขอบเขต $P=0.5$ จะไม่มีอยู่ในระนาบ 2D ให้ซ่อนเส้น Boundary Tube และแสดงป้ายเตือนบนฉาก 3D ว่าทั้งสองแกนไม่มีอิทธิพลต่อโมเดล |
| 5 | **`LogisticRegressionCV` มีพารามิเตอร์ `C_` เป็น Array** | ใน `LogisticRegressionCV` แอตทริบิวต์ชื่อ `C_` (มีขีดล่าง) และเป็น `ndarray` ขนาด `(1,)` หรือ `(n_classes,)` ไม่ใช่ `float` เดี่ยว | ดึงค่าด้วย `getattr(model, "C_", getattr(model, "C", 1.0))` และแปลงเป็น `float` หรือแสดงค่า `C_` แยกตามคลาสใน HUD |
| 6 | **Sparse Matrix Input (`scipy.sparse`)** | โมเดลที่เทรนกับข้อมูลข้อความหรือ One-Hot ขนาดใหญ่อาจส่ง `X_batch` เข้ามาเป็น `csr_matrix` | ใน `trace_internal_flow` หาก `scipy.sparse.issparse(X_batch)` ให้ใช้ `X_batch.multiply(self.coef_[0])` ก่อนแปลงเฉพาะ Top-K Contributors เพื่อไม่ให้ RAM พุ่ง |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `LogisticRegressionAdapter` รองรับทั้ง `LogisticRegression` และ `LogisticRegressionCV` ทั้งแบบ Binary และ Multiclass (OvR & Multinomial) โดยผ่าน Unit Tests ครบทุกกรณี
- [ ] สมการแยกส่วนประกอบ Log-Odds Waterfall ($\sum w_j x_{t,j} + b$) มีผลรวมตรงกับค่า `model.decision_function(X)` ในระดับความแม่นยำ `1e-5`
- [ ] ฉาก 3 มิติแสดงพื้นผิวโค้ง **3D Sigmoid S-Surface** พร้อมเส้นขอบเขตเรืองแสงที่ $P = \tau$ (ปรับค่า Threshold $\tau$ ได้ลื่นไหล 60FPS)
- [ ] รองรับการสลับแกน $Z$ ระหว่างโหมด **Probability S-Curve ($\sigma(z)$)** และ **Linear Log-Odds Plane ($z$)**
- [ ] โหมด What-If Sandbox สามารถคำนวณการเลื่อนเฟส (Phase Shift) ของคลื่น S-Surface บน GPU Shader ได้ทันทีเมื่อขยับสไลเดอร์ฟีเจอร์ใดๆ
- [ ] ผ่านการทดสอบ Memory Profiling โดยการสลับคลาสและเลื่อน Scrubber ต่อเนื่อง 1,000 เฟรมโดยไม่มี WebGL Geometry/Material รั่วไหล (0 Memory Leak)
