# Phase 23: Support Vector Machines (SVC/SVR) Adapter & 3D Kernel Saddle + Support Vector Pillars

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลตระกูล **Support Vector Machines (SVM)** ทั้งแบบเชิงเส้นและแบบไม่เชิงเส้นผ่าน **Kernel Trick** (`rbf`, `poly`, `sigmoid`, `linear`) โดยถ่ายทอดสมการ Dual Representation ออกมาเป็น **พื้นผิวอานม้าและภูเขาโค้งตามสมการ Kernel 3 มิติ (3D Kernel Saddle & Decision Surface)** พร้อม **เสาแสงและวงแหวนพลังงานรอบจุด Support Vectors (Support Vector Pillars)** และเส้นสายฟ้าแสดงอิทธิพลของ Kernel ($K(\mathbf{x}_t, \mathbf{x}_{\text{sv}})$) ที่ดึงดูดหรือผลักลูกบอลข้อมูลในแต่ละเฟรม

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - **Kernel-based SVM**: `sklearn.svm.SVC`, `sklearn.svm.SVR`, `sklearn.svm.NuSVC`, `sklearn.svm.NuSVR`, `sklearn.svm.OneClassSVM`
  - **Linear-optimized SVM**: `sklearn.svm.LinearSVC`, `sklearn.svm.LinearSVR`
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary & Multiclass Classification**: รองรับทั้งกลยุทธ์ One-vs-One (`ovo`) และ One-vs-Rest (`ovr`)
  - **Regression (`SVR`, `NuSVR`, `LinearSVR`)**: แสดงพื้นผิวทำนายพร้อมอุโมงค์ $\epsilon$-Tube
  - **Outlier / Novelty Detection (`OneClassSVM`)**: แสดงขอบเขตห่อหุ้มกลุ่มข้อมูลปกติ ($f(\mathbf{x}) \ge 0$)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์ของ Dual Formulation และ Kernel Trick
กำหนดให้โมเดลมีเซตของจุด Support Vectors จำนวน $N_{\text{sv}}$ จุด เก็บใน `model.support_vectors_` ขนาด $(N_{\text{sv}}, D)$ พร้อมสัมประสิทธิ์คู่ควบ `model.dual_coef_` แทนด้วย $\tilde{\alpha}_s = y_s \alpha_s$ และจุดตัดแกน $b$:

1. **สมการการตัดสินใจในรูป Kernel Expansion**:
   $$f(\mathbf{x}_t) = \sum_{s=1}^{N_{\text{sv}}} \tilde{\alpha}_s K(\mathbf{x}_s, \mathbf{x}_t) + b$$
2. **นิยามฟังก์ชัน Kernel มาตรฐานทั้ง 4 ชนิด**:
   - **Linear Kernel**: $K(\mathbf{x}_s, \mathbf{x}_t) = \langle \mathbf{x}_s, \mathbf{x}_t \rangle$
   - **RBF (Radial Basis Function / Gaussian) Kernel**:
     $$K(\mathbf{x}_s, \mathbf{x}_t) = \exp\left(-\gamma \|\mathbf{x}_s - \mathbf{x}_t\|_2^2\right), \quad \gamma = \text{model.\_gamma}$$
   - **Polynomial Kernel**:
     $$K(\mathbf{x}_s, \mathbf{x}_t) = \left(\gamma \langle \mathbf{x}_s, \mathbf{x}_t \rangle + r\right)^d, \quad r = \text{coef0}, \; d = \text{degree}$$
   - **Sigmoid Kernel**:
     $$K(\mathbf{x}_s, \mathbf{x}_t) = \tanh\left(\gamma \langle \mathbf{x}_s, \mathbf{x}_t \rangle + r\right)$$

### 2.2 การแกะค่าอิทธิพลราย Support Vector และ Local Feature Contribution (Waterfall)
- **Kernel Influence Score ต่อจุด Support Vector ที่ $s$**:
  - สำหรับแถว $t$: อิทธิพลสุทธิของ Support Vector ตัวที่ $s$ คือ $I_{t,s} = \tilde{\alpha}_s K(\mathbf{x}_s, \mathbf{x}_t)$
  - ผลรวมตรวจสอบ (สำหรับ Binary Classification / SVR): $\sum_{s=1}^{N_{\text{sv}}} I_{t,s} + b = f(\mathbf{x}_t)$ ตรงกับ `decision_function` 100%!
- **การคำนวณ Local Feature Contribution (Feature Waterfall)**:
  - **กรณี `kernel='linear'` หรือ `LinearSVC`/`LinearSVR`**: คำนวณตรงจาก $\phi_{t,j} = w_j x_{t,j}$
  - **กรณี Non-linear Kernel (`rbf`, `poly`, `sigmoid`)**: ใช้ **Local Kernel Gradient Sensitivity (First-order Taylor Attribution around Support Vector Centroid)**:
    - สำหรับ RBF Kernel: อนุพันธ์ย่อยเทียบกับฟีเจอร์ $j$ ณ จุด $\mathbf{x}_t$ คือ:
      $$g_{t,j} = \frac{\partial f(\mathbf{x}_t)}{\partial x_{t,j}} = \sum_{s=1}^{N_{\text{sv}}} \tilde{\alpha}_s \cdot 2\gamma (x_{s,j} - x_{t,j}) \exp\left(-\gamma \|\mathbf{x}_s - \mathbf{x}_t\|_2^2\right)$$
    - แรงผลักเฉพาะจุดของฟีเจอร์ $j$: $\phi_{t,j} = g_{t,j} \cdot (x_{t,j} - \bar{x}_{\text{sv}, j})$

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **3D Curved Kernel Saddle & Mountain Surface**:
   - เรนเดอร์พื้นผิว 3 มิติความละเอียด $60 \times 60$ ที่โค้งเว้าเป็นเนินเขาและแอ่งหุบเขาตามค่า $f(\mathbf{x})$ (หรือความน่าจะเป็น Platt Scaling)
   - วาดเส้นขอบเขต 3 เส้นบนพื้นผิวโค้ง:
     - **เส้นทึบเรืองแสงสีขาว/เขียว ($f(\mathbf{x}) = 0$)**: เส้นแบ่งเขตการตัดสินใจหลัก (Decision Boundary Contour)
     - **เส้นประสีฟ้าและสีส้ม ($f(\mathbf{x}) = +1$ และ $f(\mathbf{x}) = -1$)**: เส้นขอบเขต Margin บนพื้นผิวโค้ง!
2. **Support Vector Glowing Rings & Energy Pillars (`InstancedMesh`)**:
   - จุดที่เป็น **Support Vectors** (`model.support_vectors_`) จะถูกเน้นด้วย **วงแหวนพลังงานหมุนวน (Rotating Torus Ring)** และเสาแสงแนวตั้งที่มีความสูงแปรผันตามค่าสัมบูรณ์ของ $|\tilde{\alpha}_s|$ (Support Vector ตัวไหนมีค่า $|\alpha_s| = C$ คือเป็น Bounded Support Vector ที่แบกรับแรงกดดันสูงสุด จะมีวงแหวนสีทองเข้มพิเศษ)
3. **Top-5 Support Vector Force Tethers (เส้นสายฟ้าเชื่อมแรงผลัก Kernel)**:
   - เมื่อลูกบอลข้อมูลปัจจุบัน $t$ เคลื่อนที่ไปในฉาก 3D จะมีเส้นสายฟ้าพลังงาน 5 เส้นพุ่งจาก Support Vectors 5 ตัวที่มีค่า $|I_{t,s}| = |\tilde{\alpha}_s K(\mathbf{x}_s, \mathbf{x}_t)|$ สูงที่สุดมายังลูกบอล แสดงให้เห็นว่า Support Vector ตัวไหนกำลังดึงหรือผลักลูกบอลอยู่ในเฟรมนั้น!

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **Kernel Formula & Hyperparameter Banner**: แสดงชนิดของ Kernel (`RBF`, `Poly`, `Linear`, `Sigmoid`), ค่า $\gamma$ (`_gamma`), $C$, `degree`, `coef0` และจำนวน Support Vectors แยกตามคลาส (`model.n_support_`)
- **Support Vector Pillar Filter**: สวิตช์เลือกแสดงเฉพาะ Top-100 Support Vectors ที่มี $|\alpha_s|$ สูงสุด หรือเปิด-ปิดเส้น Force Tethers
- **Platt Probability vs Raw Margin Toggle**: สลับการแสดงผลระหว่างค่า `decision_function` (พร้อมป้ายเตือนหาก `probability=False`) และค่า `predict_proba`

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลไต่ไปตามภูมิประเทศโค้งเว้าของ Kernel Surface
   - เมื่อลูกบอลเข้าใกล้เสา Support Vector ต้นใด ค่าความคล้ายคลึง $K(\mathbf{x}_s, \mathbf{x}_t)$ จะพุ่งเข้าใกล้ $1.0$ เส้นสายฟ้าจากเสาต้นนั้นจะสว่างจ้าขึ้นทันที พร้อมดันลูกบอลให้พุ่งขึ้นยอดเขา ($+1$) หรือจมลงหุบเขา ($-1$)
2. **เมื่อลาก Scrubber ข้ามสันเขา $f(\mathbf{x}) = 0$**:
   - เส้น Contour ที่ $f(\mathbf{x}) = 0$ กระพริบเตือนการเปลี่ยนคลาส พร้อมแสดงตารางเปรียบเทียบใน HUD ว่าแรงดึงจาก Support Vectors ฝั่งบวกชนะแรงดึงจากฝั่งลบไปกี่คะแนน
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - เมื่อลากลูกบอลเงา (Ghost Sphere) ออกห่างจากกลุ่มข้อมูลในโมเดล RBF Kernel ผู้ใช้จะเห็นปรากฏการณ์สำคัญของ RBF SVM: เมื่อออกห่างจาก Support Vectors ทุกตัว ($K(\mathbf{x}_s, \mathbf{x}_{\text{ghost}}) \to 0$) ความสูงของพื้นผิวจะลู่เข้าหาค่าคงที่ $b$ (`intercept_`) เสมอ!

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล SVM
```text
+-------------------------------------------------------------------------+
| [SVM KERNEL HUD]  Model: SVC | Kernel: rbf (γ=0.450, C=1.0)             |
| Support Vectors : 68 Total (Class 0: 32 SVs | Class 1: 36 SVs)          |
| Platt Proba Mode: Disabled (probability=False -> Using Decision Score)  |
+-------------------------------------------------------------------------+
| 3D Visuals: [x] Support Vector Pillars  [x] Top-5 Kernel Force Tethers  |
| Contours  : [x] Decision f(x)=0         [x] Margin Contours f(x)=±1     |
+-------------------------------------------------------------------------+
| FRAME #0119 KERNEL DECISION TRACE:                                      |
|  Decision Score f(x_t) :  +1.420  [==========|=======>   ] (Outside +1) |
|  Platt / Sigmoid Est.  :   80.5%  (Calibrated Score Approximation)      |
|  Intercept Bias (b)    :  -0.180                                        |
+-------------------------------------------------------------------------+
| TOP SUPPORT VECTOR FORCES (α_s * y_s * K(x_s, x_t)):                    |
|  SV #14 (Class 1, α=+1.00) : K=0.892 -> Force: +0.892 [=========>     ] |
|  SV #41 (Class 1, α=+0.85) : K=0.740 -> Force: +0.629 [======>        ] |
|  SV #03 (Class 0, α=-1.00) : K=0.281 -> Force: -0.281 [   <===        ] |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/distance/svm.py`
- `frontend/src/domains/stage3d/components/surfaces/KernelSaddleSurface3D.tsx`
- `frontend/src/domains/stage3d/components/actors/SupportVectorPillars3D.tsx`

### 5.2 โค้ดต้นแบบ `SVMAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit, softmax
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM, LinearSVC, LinearSVR
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class SVMAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (SVC, SVR, NuSVC, NuSVR, OneClassSVM, LinearSVC, LinearSVR)
    MAX_DISPLAY_SV = 1000

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.is_linear_class = isinstance(model, (LinearSVC, LinearSVR))
        self.is_classifier = isinstance(model, (SVC, NuSVC, LinearSVC))
        self.kernel = "linear" if self.is_linear_class else getattr(model, "kernel", "rbf")
        self.has_native_proba = bool(
            self.is_classifier and getattr(model, "probability", False) and hasattr(model, "predict_proba")
        )

    def extract_metadata(self) -> Dict[str, Any]:
        gamma_val = getattr(self.model, "_gamma", getattr(self.model, "gamma", "scale"))
        n_support = getattr(self.model, "n_support_", np.array([])).tolist()
        sv_payload = self._extract_top_support_vectors()
        return {
            "model_family": "support_vector_machine",
            "algorithm": self.model.__class__.__name__,
            "task_type": "classification" if self.is_classifier else "regression",
            "kernel": self.kernel,
            "gamma": float(gamma_val) if isinstance(gamma_val, (int, float, np.number)) else str(gamma_val),
            "C": float(getattr(self.model, "C", 1.0)),
            "degree": int(getattr(self.model, "degree", 3)),
            "coef0": float(getattr(self.model, "coef0", 0.0)),
            "has_native_proba": self.has_native_proba,
            "n_support_per_class": n_support,
            "total_support_vectors": int(getattr(self.model, "support_vectors_", np.empty((0, 0))).shape[0]),
            "support_vectors_3d_data": sv_payload,
        }

    def _extract_top_support_vectors(self) -> Dict[str, Any]:
        if self.is_linear_class or not hasattr(self.model, "support_vectors_"):
            return {"coordinates": [], "dual_weights": [], "indices": []}
        sv = np.asarray(self.model.support_vectors_, dtype=np.float32)
        dual = np.asarray(self.model.dual_coef_, dtype=np.float32)
        importance = np.linalg.norm(dual, axis=0)  # (N_sv,)
        if len(sv) > self.MAX_DISPLAY_SV:
            top_idx = np.argsort(importance)[::-1][:self.MAX_DISPLAY_SV]
            sv = sv[top_idx]
            dual_main = dual[0, top_idx]
        else:
            dual_main = dual[0] if dual.shape[0] > 0 else np.zeros(len(sv))
        return {
            "coordinates": sv.tolist(),
            "dual_weights": dual_main.tolist(),
        }

    def _compute_local_contributions(self, X_arr: np.ndarray) -> np.ndarray:
        if self.kernel == "linear" and hasattr(self.model, "coef_"):
            w = np.atleast_2d(np.asarray(self.model.coef_, dtype=np.float64))[0]
            return X_arr * w[np.newaxis, :]
        if hasattr(self.model, "support_vectors_") and len(self.model.support_vectors_) > 0:
            sv = np.asarray(self.model.support_vectors_, dtype=np.float64)
            dual = np.asarray(self.model.dual_coef_, dtype=np.float64)[0]  # (N_sv,)
            gamma = float(getattr(self.model, "_gamma", 1.0 / max(1, X_arr.shape[1])))
            sq_dists = cdist(X_arr, sv, metric="sqeuclidean")  # (N, N_sv)
            K_mat = np.exp(-gamma * sq_dists) * dual[np.newaxis, :]  # (N, N_sv)
            # First-order directional contribution per feature
            sv_mean = np.mean(sv, axis=0, keepdims=True)
            weights_sum = np.sum(K_mat, axis=1, keepdims=True)
            weighted_sv = (K_mat @ sv) / np.where(np.abs(weights_sum) > 1e-9, weights_sum, 1.0)
            return -np.abs(X_arr - weighted_sv) * np.sign(weights_sum)
        return np.zeros_like(X_arr)

    def trace_internal_flow(self, X_batch: np.ndarray) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch, dtype=np.float64)
        preds = self.model.predict(X_arr)
        scores = self.model.decision_function(X_arr) if hasattr(self.model, "decision_function") else preds
        scores_2d = scores.reshape(-1, 1) if scores.ndim == 1 else scores

        if self.has_native_proba:
            probas = self.model.predict_proba(X_arr).astype(np.float32).tolist()
        elif self.is_classifier:
            if scores_2d.shape[1] == 1:
                p1 = expit(scores_2d[:, 0])
                probas = np.column_stack([1.0 - p1, p1]).astype(np.float32).tolist()
            else:
                probas = softmax(scores_2d, axis=1).astype(np.float32).tolist()
        else:
            probas = None

        contributions = self._compute_local_contributions(X_arr)
        intercept_list = np.atleast_1d(getattr(self.model, "intercept_", [0.0])).astype(float).tolist()

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas,
            decision_scores=scores_2d.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=intercept_list,
            extra_telemetry={"has_native_proba": self.has_native_proba, "kernel": self.kernel},
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 50, target_idx: int = 0
    ) -> SurfaceMesh3D:
        xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        grid_X = np.tile(fixed_features.astype(np.float64), (grid_resolution * grid_resolution, 1))
        grid_X[:, axis_x_idx] = xx.ravel()
        grid_X[:, axis_y_idx] = yy.ravel()

        raw = self.model.decision_function(grid_X) if hasattr(self.model, "decision_function") else self.model.predict(grid_X)
        z_flat = raw if raw.ndim == 1 else raw[:, min(target_idx, raw.shape[1] - 1)]
        z_mat = z_flat.reshape(grid_resolution, grid_resolution)

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=z_mat.astype(np.float32).tolist(),
            auxiliary_grids={"margin_levels": [-1.0, 0.0, 1.0]},
            boundary_threshold=0.0 if self.is_classifier else None,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [1, 0],
    "probabilities": [[0.195, 0.805], [0.760, 0.240]],
    "decision_scores": [[1.420], [-1.150]],
    "feature_contributions": [[0.892, 0.629, -0.101], [-0.540, -0.430, 0.0]],
    "base_intercepts": [-0.180],
    "extra_telemetry": { "has_native_proba": false, "kernel": "rbf" }
  },
  "SurfaceMesh3D": {
    "x_grid": [-3.0, 3.0],
    "y_grid": [-3.0, 3.0],
    "z_values": [[-0.95, 1.42], [-1.20, 0.85]],
    "auxiliary_grids": { "margin_levels": [-1.0, 0.0, 1.0] },
    "boundary_threshold": 0.0
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การจำกัดจำนวน `support_vectors_` ที่ส่งไปเรนเดอร์ (`MAX_DISPLAY_SV = 1000`)**:
   - ในชุดข้อมูลขนาดใหญ่ที่มี Noise สูง โมเดล `SVC` อาจมี Support Vectors มากถึง $20,000$ จุด ซึ่งจะทำให้การคำนวณ Pairwise Distance และ WebGL Pillars หน่วง
   - เราจึงจัดลำดับความสำคัญด้วย `np.linalg.norm(model.dual_coef_, axis=0)` และคัดเลือกเฉพาะ **Top 1,000 Bounded/High-Alpha Support Vectors** ที่มีอิทธิพลต่อรูปทรงขอบเขตสูงสุดส่งไปยังฉาก 3D
2. **Vectorized Pairwise Kernel Distance ด้วย `scipy.spatial.distance.cdist`**:
   - ห้ามคำนวณระยะห่างระหว่าง `X_batch` กับ `support_vectors_` ด้วย Python loop เด็ดขาด ให้ใช้ `cdist(X_arr, sv, metric='sqeuclidean')` ที่ทำงานบน C-level BLAS เสมอ

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`SVC(probability=False)` เป็นค่า Default ของ Scikit-learn** | หากไม่ได้ระบุ `probability=True` ตอนเทรน การเรียก `model.predict_proba(X)` จะเกิด `AttributeError` ทันที | เช็ก `getattr(model, "probability", False)` ก่อนเสมอ หากเป็น `False` ให้ใช้ `decision_function(X)` เป็นแกนหลัก และคำนวณ Pseudo-Probability ผ่าน `expit`/`softmax` พร้อมแสดงป้ายแจ้งเตือนใน HUD |
| 2 | **`model.coef_` มีเฉพาะใน `kernel='linear'` เท่านั้น** | หากเรียก `model.coef_` บนโมเดล `SVC(kernel='rbf')` จะเกิด `AttributeError: coef_ is only available when using a linear kernel` | ห้ามเข้าถึง `model.coef_` จนกว่าจะตรวจยืนยันว่า `self.kernel == "linear"` เท่านั้น! |
| 3 | **`LinearSVC` และ `LinearSVR` ไม่มี `support_vectors_` เก็บไว้** | `LinearSVC` ใช้ไลบรารี `liblinear` ซึ่งเก็บเฉพาะเวกเตอร์น้ำหนักรวม `coef_` และไม่มีแอตทริบิวต์ `support_vectors_` | ตรวจสอบ `hasattr(model, "support_vectors_")` เสมอ หากเป็น `LinearSVC`/`LinearSVR` ให้สลับไปเรนเดอร์โหมดแผ่นระนาบ Margin Corridor เหมือนใน Phase 21 |
| 4 | **One-vs-One (`decision_function_shape='ovo'`) ใน Multiclass SVC** | หากตั้งค่าเป็น `'ovo'` ขนาดของ `decision_function` จะเป็น `(N, C*(C-1)/2)` แทนที่จะเป็น `(N, C)` | ตรวจสอบ `getattr(model, "decision_function_shape", "ovr")` หากเป็น `'ovo'` และมีมากกว่า 2 คลาส ให้แสดงป้ายเตือน Pairwise OvO Score หรือแปลงผ่าน `_ovr_decision_function` |
| 5 | **ความไม่สอดคล้องกันระหว่าง `predict()` กับ `argmax(predict_proba())` ใน Platt Scaling** | เนื่องจาก `predict_proba` ของ `SVC` ใช้ 5-fold Cross-Validation Platt Scaling แยกต่างหาก บางครั้งจุดที่อยู่ใกล้ขอบเขต $f(\mathbf{x}) \approx 0$ อาจให้ค่า `predict()` เป็น Class 1 แต่ `predict_proba` ได้ `0.48` | ยึดผลลัพธ์จาก `model.predict(X)` เป็นคำตอบหลักเสมอ และหากพบความขัดแย้งบริเวณใกล้เส้นแบ่งเขต ให้แสดง Tooltip อธิบายกลไก **Platt Calibration Offset** |
| 6 | **`OneClassSVM` คืนค่าทำนายเป็น `{-1, +1}` (Outlier vs Inlier)** | `OneClassSVM` ไม่มี `classes_` และใช้ `0.0` เป็นเส้นแบ่งเขตระหว่าง Inlier ($+1$) กับ Outlier ($-1$) | รองรับ `OneClassSVM` โดยแมป `-1 -> Outlier (Anomaly)` และ `+1 -> Inlier (Normal)` |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `SVMAdapter` รองรับครบทั้ง `SVC`, `SVR`, `NuSVC`, `NuSVR`, `OneClassSVM`, `LinearSVC`, และ `LinearSVR`
- [ ] รองรับทั้ง 4 Kernel มาตรฐาน (`rbf`, `poly`, `sigmoid`, `linear`) โดยไม่เกิด `AttributeError` เรื่อง `coef_` หรือ `predict_proba`
- [ ] ฉาก 3 มิติแสดงพื้นผิวโค้ง **3D Kernel Saddle Surface** พร้อมเส้น Contour 3 ระดับ ($f(\mathbf{x}) \in \{-1, 0, +1\}$)
- [ ] แสดงจุด **Support Vectors** ด้วยเสาแสงและวงแหวนพลังงาน (`SupportVectorPillars3D`) พร้อมเส้นสายฟ้าเชื่อม Top-5 Support Vectors ที่มีอิทธิพลสูงสุดต่อจุดปัจจุบัน
- [ ] จำกัดจำนวน Support Vectors ที่ส่งไปเรนเดอร์ไม่เกิน 1,000 จุดที่สำคัญที่สุด เพื่อรักษาความเร็ว 60FPS
