# Phase 21: SGD Classifier & Regressor Adapter & 3D Margin Boundary Plane

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลตระกูล **Stochastic Gradient Descent (`SGDClassifier` และ `SGDRegressor`)** ซึ่งเป็นหัวใจสำคัญของการเรียนรู้เชิงเส้นขนาดใหญ่ (Large-scale / Online Linear Learning) ที่สามารถเปลี่ยนพฤติกรรมของสมการและพื้นผิวการตัดสินใจไปตามชนิดของ **Loss Function** (`hinge`, `log_loss`, `modified_huber`, `perceptron`, `squared_error`, `huber`, `epsilon_insensitive`) โดยแสดง **ช่องทางเดินขอบเขตปลอดภัย 3 มิติ (3D Linear Margin Corridor $\pm 1$)** และระยะห่างทางเรขาคณิตจริงจากระนาบแบ่งเขต

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.linear_model.SGDClassifier`
  - `sklearn.linear_model.SGDRegressor`
  - `sklearn.linear_model.PassiveAggressiveClassifier` (โครงสร้างเดียวกัน)
  - `sklearn.linear_model.PassiveAggressiveRegressor`
  - `sklearn.linear_model.Perceptron`
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: รองรับทุก Loss Function ของ Classifier
  - **Multiclass Classification (One-vs-Rest)**: รองรับเมื่อ $C > 2$
  - **Single-target Regression**: รองรับทุก Loss Function ของ Regressor (`squared_error`, `huber`, `epsilon_insensitive`, `squared_epsilon_insensitive`)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของระยะห่างเรขาคณิตและ Loss Regime
กำหนดให้ข้อมูลนำเข้ามีขนาด $(N, D)$ เวกเตอร์น้ำหนัก $\mathbf{w} \in \mathbb{R}^D$ และจุดตัดแกน $b \in \mathbb{R}$:

1. **คะแนนการตัดสินใจดิบ (Raw Decision Function)**:
   $$f(\mathbf{x}_t) = \mathbf{w}^T \mathbf{x}_t + b = \sum_{j=1}^{D} w_j x_{t,j} + b$$
2. **ระยะห่างทางเรขาคณิตแบบมีเครื่องหมายจากระนาบแบ่งเขต (Signed Geometric Margin Distance)**:
   $$d_{\text{geom}}(\mathbf{x}_t) = \frac{f(\mathbf{x}_t)}{\|\mathbf{w}\|_2} = \frac{\mathbf{w}^T \mathbf{x}_t + b}{\sqrt{\sum_{j=1}^{D} w_j^2}}$$
   - ขอบเขต Margin มาตรฐาน ($\mathbf{w}^T \mathbf{x} + b = \pm 1$) จะอยู่ห่างจากระนาบตัดสินใจเป็นระยะทางเรขาคณิต $\pm \frac{1}{\|\mathbf{w}\|_2}$
   - หาก $|f(\mathbf{x}_t)| < 1.0$ แปลว่าจุดข้อมูลแถว $t$ ตกอยู่ใน **Margin Violation / High-Risk Corridor**!
3. **การแปลงความน่าจะเป็นตามชนิดของ `model.loss` (ใน `SGDClassifier`)**:
   - **กรณี `loss="log_loss"`**: $P(y_t=1 \mid \mathbf{x}_t) = \sigma(f(\mathbf{x}_t)) = \frac{1}{1 + e^{-f(\mathbf{x}_t)}}$
   - **กรณี `loss="modified_huber"`**:
     $$P(y_t=1 \mid \mathbf{x}_t) = \frac{\min(1, \max(-1, f(\mathbf{x}_t))) + 1}{2}$$
   - **กรณี `loss in ("hinge", "squared_hinge", "perceptron")`**: โมเดลไม่มี `predict_proba` ตามทฤษฎี ต้องใช้คะแนน Margin $f(\mathbf{x}_t)$ เป็นตัววัดหลัก พร้อมคำนวณ Calibrated Pseudo-Confidence $\sigma(f(\mathbf{x}_t))$ สำหรับแสดงสีเปรียบเทียบเท่านั้น
4. **โซนไม่คิดค่าความคลาดเคลื่อนใน `SGDRegressor` ($\epsilon$-Insensitive Tube)**:
   - สำหรับ `loss="epsilon_insensitive"`: หาก $|\hat{y}_t - y_t| \le \epsilon$ ค่า Loss ของแถวนั้นจะเท่ากับ $0.0$

### 2.2 โครงสร้าง Array Shape และ Local Feature Contribution (Waterfall)
- **Input (`X_batch`)**: ขนาด $(N, D)$
- **Decision Scores (`f_x`)**: ขนาด $(N, 1)$ สำหรับ Binary/Regression หรือ $(N, C)$ สำหรับ Multiclass
- **Geometric Distances (`d_geom`)**: ขนาด $(N, 1)$ หรือ $(N, C)$ คำนวณจาก `f_x / np.linalg.norm(w, axis=-1)`
- **Local Feature Contribution (Waterfall)**:
  - แรงผลักต่อคะแนนดิบของฟีเจอร์ $j$ ในแถว $t$: $\phi_{t,j} = w_j x_{t,j}$
  - แรงผลักต่อระยะห่างเรขาคณิตจริง: $\tilde{\phi}_{t,j} = \frac{w_j}{\|\mathbf{w}\|_2} x_{t,j}$

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **3D Linear Decision Plane & Parallel Margin Corridor ($\pm 1$ Planes)**:
   - **แผ่นระนาบหลัก ($f(\mathbf{x}) = 0$)**: แผ่นกระจกเรืองแสงสีขาวอมฟ้าแสดงเส้นแบ่งเขตการตัดสินใจ
   - **แผ่นระนาบขอบเขตคู่ขนาน 2 ฝั่ง ($f(\mathbf{x}) = +1$ และ $f(\mathbf{x}) = -1$)**: แผ่นโปร่งแสงแบบเส้นตาราง (`wireframe / dashed shader`) สีเขียวอ่อนฝั่ง $+1$ และสีส้มอ่อนฝั่ง $-1$ ขนาบข้างระนาบหลัก
   - สำหรับ `SGDRegressor` ที่ใช้ `huber` หรือ `epsilon_insensitive`: แผ่นคู่ขนานจะแสดงเป็น **อุโมงค์ความคลาดเคลื่อนที่ยอมรับได้ ($\hat{y} \pm \epsilon$ Tube)**
2. **Orthogonal Margin Distance Laser Line**:
   - เมื่อลูกบอลข้อมูลของแถวปัจจุบัน $t$ เคลื่อนที่ไปในฉาก 3D จะมีเส้นเลเซอร์ตั้งฉากพุ่งจากลูกบอลไปยังจุดที่ใกล้ที่สุดบนระนาบตัดสินใจ ($f(\mathbf{x}) = 0$) พร้อมป้ายแสดงระยะห่าง $d_{\text{geom}}$
3. **Margin Zone Warning Aura**:
   - หากจุดข้อมูลตกอยู่ในช่องแคบระหว่างระนาบ $-1$ และ $+1$ ($|f(\mathbf{x}_t)| < 1$) ลูกบอลจะมีวงแหวนเตือนภัยสีเหลืองอำพันกระพริบ แสดงว่าจุดนี้อยู่ในเขต **Inside Margin Zone (Uncertain / Margin Violator)**

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Active Loss & Penalty Inspector Badge**: แสดงชนิดของ `loss` (เช่น `hinge`, `log_loss`, `huber`), `penalty` (`l2`, `l1`, `elasticnet`), `alpha`, `learning_rate` และจำนวนข้อมูลทั้งหมดที่โมเดลเห็นระหว่างเทรน (`model.t_`)
- **Margin Corridor Visibility Toggle**: สวิตช์เปิด-ปิดแผ่นระนาบคู่ขนาน $\pm 1$ (หรือ $\pm \epsilon$) ในฉาก 3 มิติ
- **Score Scale Switcher**: สลับการแสดงผลระหว่าง **Raw Score ($f(x)$)**, **Geometric Distance ($f(x)/\|w\|$)** และ **Probability / Pseudo-Confidence**

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลวิ่งผ่านอวกาศ 3D โดยมีเส้นเลเซอร์ตั้งฉากวัดระยะห่างไปยังระนาบตัดสินใจตลอดเวลา
   - แผง HUD แสดงมาตรวัด **Margin Safety Bar** ว่าแถวปัจจุบันอยู่นอกเขต Margin อย่างปลอดภัย ($|f(\mathbf{x}_t)| \ge 1$) หรือล้ำเข้ามาในเขตเสี่ยง ($|f(\mathbf{x}_t)| < 1$)
2. **เมื่อลาก Scrubber ข้ามเขต Margin Corridor**:
   - เมื่อลูกบอลเคลื่อนจาก $f(\mathbf{x}) = +1.2$ เข้าสู่ $+0.4$ (ผ่านระนาบ $+1$) ป้ายสถานะจะเปลี่ยนจาก `SAFE MARGIN (> +1)` เป็น `INSIDE MARGIN CORRIDOR` และเมื่อข้าม $0.0$ จะเปลี่ยนเป็น `CLASS FLIPPED (< 0)`
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - ผู้ใช้สามารถทดลองเลื่อนสไลเดอร์เพื่อดูว่าต้องปรับค่าฟีเจอร์ใดเป็นปริมาณเท่าไร จึงจะสามารถผลักลูกบอลเงา (Ghost Sphere) ให้หลุดพ้นจากเขตเสี่ยง ($|f(\mathbf{x})| < 1$) ออกไปยังเขตปลอดภัยได้

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล SGD Classifier / Regressor
```text
+-------------------------------------------------------------------------+
| [SGD ENGINE HUD]  Model: SGDClassifier | Loss: hinge (Linear SVM Mode)  |
| Penalty: elasticnet (α=0.0001, l1_ratio=0.15) | Steps (t_): 45,000      |
+-------------------------------------------------------------------------+
| 3D Corridor View: [x] Show ±1 Margin Planes   [x] Orthogonal Laser Line |
| Metric Mode     : [ *Raw Margin f(x)* ] [ Geometric Dist ] [ Pseudo-P ] |
+-------------------------------------------------------------------------+
| FRAME #0315 MARGIN CORRIDOR TELEMETRY:                                  |
|  Raw Score f(x_t)       :  +0.640  (Status: ⚠️ INSIDE MARGIN CORRIDOR)  |
|  Geometric Dist f/||w|| :  +0.285 units from Decision Plane             |
|  Margin Safety Gauge    :  [   -1.0 |=====(0.64)| +1.0 Safe Zone     ] |
|  Native predict_proba   :  N/A (Hinge Loss) | Pseudo-Conf: 65.5%        |
+-------------------------------------------------------------------------+
| FEATURE MARGIN FORCE BREAKDOWN (w_i * x_i + b):                         |
|  Intercept (b)          :  -0.300  [       <==|              ]          |
|  feature_velocity (X)   :  +1.120  [          |=======>      ]          |
|  feature_pressure (Y)   :  -0.180  [        <=|              ]          |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/linear/sgd.py`
- `frontend/src/domains/stage3d/components/surfaces/MarginCorridorPlanes3D.tsx`
- `frontend/src/domains/hud/components/inspectors/SgdMarginInspector.tsx`

### 5.2 โค้ดต้นแบบ `SGDAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import (
    SGDClassifier, SGDRegressor,
    PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron
)
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class SGDAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (
        SGDClassifier, SGDRegressor,
        PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron
    )
    PROBA_LOSSES = {"log_loss", "modified_huber", "log"}

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.is_classifier = hasattr(model, "classes_")
        self.loss_name = getattr(model, "loss", "perceptron")
        self.has_native_proba = self.is_classifier and (self.loss_name in self.PROBA_LOSSES) and hasattr(model, "predict_proba")
        self.coef_2d = np.atleast_2d(np.asarray(model.coef_, dtype=np.float64))
        self.intercept_1d = np.atleast_1d(np.asarray(model.intercept_, dtype=np.float64))
        self.w_norms = np.maximum(np.linalg.norm(self.coef_2d, axis=1), 1e-12)

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "model_family": "stochastic_gradient_linear",
            "algorithm": self.model.__class__.__name__,
            "task_type": "classification" if self.is_classifier else "regression",
            "loss": self.loss_name,
            "has_native_proba": self.has_native_proba,
            "penalty": getattr(self.model, "penalty", "l2"),
            "alpha": float(getattr(self.model, "alpha", 0.0001)),
            "epsilon": float(getattr(self.model, "epsilon", 0.1)),
            "n_iter": int(getattr(self.model, "n_iter_", 0)),
            "total_weight_updates_t": int(getattr(self.model, "t_", 0)),
            "weight_l2_norms": self.w_norms.tolist(),
            "coef": self.coef_2d.tolist(),
            "intercept": self.intercept_1d.tolist(),
        }

    def trace_internal_flow(self, X_batch: np.ndarray, class_idx: int = 0) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        preds = self.model.predict(X_arr)
        c_idx = min(class_idx, self.coef_2d.shape[0] - 1)
        w = self.coef_2d[c_idx]
        b = float(self.intercept_1d[c_idx])
        contributions = X_arr * w[np.newaxis, :]  # (N, D)

        if self.is_classifier:
            raw_scores = self.model.decision_function(X_arr)
            scores_2d = raw_scores.reshape(-1, 1) if raw_scores.ndim == 1 else raw_scores
            geom_dist = scores_2d / self.w_norms[np.newaxis, :]
            if self.has_native_proba:
                probas = self.model.predict_proba(X_arr).astype(np.float32).tolist()
            else:
                if scores_2d.shape[1] == 1:
                    p1 = expit(scores_2d[:, 0])
                    probas = np.column_stack([1.0 - p1, p1]).astype(np.float32).tolist()
                else:
                    probas = softmax(scores_2d, axis=1).astype(np.float32).tolist()
        else:
            scores_2d = preds.reshape(-1, 1)
            geom_dist = scores_2d / self.w_norms[0]
            probas = None

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas,
            decision_scores=scores_2d.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=self.intercept_1d.tolist(),
            extra_telemetry={
                "has_native_proba": self.has_native_proba,
                "geometric_distances": geom_dist.astype(np.float32).tolist(),
                "inside_margin_mask": (np.abs(scores_2d[:, c_idx]) < 1.0).tolist() if self.is_classifier else [],
            },
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 40, class_idx: int = 0
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        c_idx = min(class_idx, self.coef_2d.shape[0] - 1)
        w = self.coef_2d[c_idx]
        b = float(self.intercept_1d[c_idx])

        off_axis = float(np.dot(fixed_features, w) - w[axis_x_idx]*fixed_features[axis_x_idx] - w[axis_y_idx]*fixed_features[axis_y_idx] + b)
        zz_main = w[axis_x_idx] * xx + w[axis_y_idx] * yy + off_axis
        margin_width = 1.0 if self.is_classifier else float(getattr(self.model, "epsilon", 0.1))

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=zz_main.astype(np.float32).tolist(),
            auxiliary_grids={
                "z_plus_margin": (zz_main + margin_width).astype(np.float32).tolist(),
                "z_minus_margin": (zz_main - margin_width).astype(np.float32).tolist(),
                "margin_width": margin_width,
                "w_norm": float(self.w_norms[c_idx]),
            },
            boundary_threshold=0.0 if self.is_classifier else None,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [1, 0],
    "probabilities": [[0.345, 0.655], [0.891, 0.109]],
    "decision_scores": [[0.640], [-2.100]],
    "feature_contributions": [[1.120, -0.180], [-1.450, -0.350]],
    "base_intercepts": [-0.300],
    "extra_telemetry": {
      "has_native_proba": false,
      "geometric_distances": [[0.285], [-0.936]],
      "inside_margin_mask": [true, false]
    }
  },
  "SurfaceMesh3D": {
    "x_grid": [-2.0, 2.0],
    "y_grid": [-2.0, 2.0],
    "z_values": [[-2.18, 2.30], [-2.90, 1.58]],
    "auxiliary_grids": {
      "z_plus_margin": [[-1.18, 3.30], [-1.90, 2.58]],
      "z_minus_margin": [[-3.18, 1.30], [-3.90, 0.58]],
      "margin_width": 1.0,
      "w_norm": 2.245
    },
    "boundary_threshold": 0.0
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **Single-Geometry Instancing / Offset Shader สำหรับระนาบคู่ขนาน 3 ชั้น**:
   - แทนที่จะส่งตาราง 2D ขนาดใหญ่ 3 ชุด (`z_values`, `z_plus_margin`, `z_minus_margin`) เมื่อความละเอียดสูง ในฝั่ง Three.js (`MarginCorridorPlanes3D.tsx`) สามารถใช้ `BufferGeometry` ชุดเดียวกันแล้วขยับตำแหน่งแกน $Z$ ของ Mesh ขอบบนและขอบล่างด้วย `<mesh position={[0, 0, +marginWidth]}>` และ `<mesh position={[0, 0, -marginWidth]}>` ได้โดยตรง ลดการใช้ GPU Vertex Memory ลง 66%!
2. **การป้องกันการหารด้วยศูนย์เมื่อ $\|\mathbf{w}\|_2 = 0$**:
   - ใช้ `np.maximum(np.linalg.norm(self.coef_2d, axis=1), 1e-12)` เสมอเพื่อไม่ให้เกิด `inf` หรือ `NaN` ใน `geometric_distances` ซึ่งจะทำให้ JSON Serializer ของ FastAPI ล้มเหลว

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`AttributeError: predict_proba is not available when loss='hinge'`** | ค่าเริ่มต้นของ `SGDClassifier` คือ `loss='hinge'` ซึ่งใช้ `@available_if` ซ่อนเมธอด `predict_proba` ไว้ | ตรวจเช็กทั้ง `self.loss_name in {"log_loss", "modified_huber", "log"}` และ `hasattr(model, "predict_proba")` ก่อนเรียกเสมอ หากไม่มีให้ใช้ `decision_function` เป็นหลัก |
| 2 | **`modified_huber` คืนค่า `predict_proba` ที่อิ่มตัวเป็น `0.0` หรือ `1.0` ทันทีเมื่อ $|f(x)| \ge 1$** | สมการของ `modified_huber` ตัดขอบความน่าจะเป็นเป็นเส้นตรงในช่วง $[-1, +1]$ ทำให้จุดที่อยู่นอก Margin มีความน่าจะเป็น `1.000` เป๊ะๆ | แสดงคำอธิบายใน HUD ว่า `Modified Huber Piecewise Linear Probability (Saturated outside [-1, +1])` เพื่อไม่ให้ผู้ใช้เข้าใจผิดว่าโมเดล Overconfident ผิดปกติ |
| 3 | **การเปลี่ยนชื่อ `loss='log'` เป็น `'log_loss'` ใน Scikit-learn 1.1+** | โมเดลที่ถูกบันทึกจาก Scikit-learn เวอร์ชันเก่าอาจมี `model.loss == 'log'` | รองรับทั้ง `'log'` และ `'log_loss'` ในเซต `PROBA_LOSSES` |
| 4 | **ฟีเจอร์ไม่ได้ทำ Feature Scaling ก่อนเทรน SGD** | SGD อ่อนไหวต่อสเกลของข้อมูลมาก หากไม่ได้ทำ StandardScaler ค่า `coef_` อาจแกว่งตัวสูงระดับ `1e6` ทำให้ระนาบ 3D ชันเกือบตั้งฉาก 90 องศา | ตรวจสอบหาก `max(abs(z_values)) > 1e4` ให้ทำการ Auto-Normalize สเกลแกน $Z$ ในฉาก 3D พร้อมแสดงป้ายเตือน **"Unscaled Features Detected: Large Gradient Magnitude"** |
| 5 | **`Perceptron` และ `PassiveAggressive` ไม่มีพารามิเตอร์ `loss` มาตรฐาน** | `Perceptron` ใช้ `loss='perceptron'` เสมอ และ `PassiveAggressive` ใช้ `hinge` / `squared_hinge` | ใช้ `getattr(model, "loss", "perceptron")` เพื่อให้ Adapter เดียวรองรับได้ทั้งตระกูล Online Linear |
| 6 | **`SGDRegressor` เป็น 1D `coef_` ขนาด `(D,)` แทนที่จะเป็น `(1, D)`** | ใน Scikit-learn `SGDRegressor.coef_` มีมิติเดียว `(D,)` ต่างจาก `SGDClassifier` ที่มี 2 มิติ `(1, D)` หรือ `(C, D)` | บังคับ `np.atleast_2d(model.coef_)` เสมอใน `__init__` |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `SGDAdapter` รองรับครบทั้ง `SGDClassifier`, `SGDRegressor`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, และ `Perceptron`
- [ ] รองรับทุก `loss` (`hinge`, `log_loss`, `modified_huber`, `squared_hinge`, `perceptron`, `squared_error`, `huber`, `epsilon_insensitive`) โดยไม่เกิด `AttributeError`
- [ ] คำนวณระยะห่างเรขาคณิต $d_{\text{geom}} = f(\mathbf{x}) / \|\mathbf{w}\|_2$ และระบุสถานะ `Inside Margin Corridor` ($|f(\mathbf{x})| < 1$) ได้อย่างถูกต้อง
- [ ] ฉาก 3 มิติแสดงแผ่นระนาบหลักพร้อมระนาบคู่ขนาน $\pm 1$ (หรือ $\pm \epsilon$) และเส้นเลเซอร์วัดระยะตั้งฉากไปยังระนาบ
- [ ] โหมด What-If Sandbox แสดงการเคลื่อนที่ของลูกบอลเงาเข้า-ออกเขต Margin Corridor แบบเรียลไทม์ 60FPS
