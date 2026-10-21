# Phase 25: Multinomial & Bernoulli Naive Bayes Adapters & 3D Log-Likelihood Towers

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เฟสนี้มุ่งเน้นการพัฒนา Adapter และระบบแสดงผล 3 มิติสำหรับโมเดลความน่าจะเป็นเชิงไม่ต่อเนื่องและข้อมูลนับความถี่ (**Discrete / Count / Binary Probabilistic Models**) ได้แก่ **`MultinomialNB`**, **`BernoulliNB`**, **`ComplementNB`**, และ **`CategoricalNB`** ซึ่งเป็นโมเดลหลักในการจำแนกข้อความ (NLP Bag-of-Words / TF-IDF) และข้อมูลแบบตารางที่มีฟีเจอร์สถานะเปิด-ปิด โดยแสดงผลเป็น **พื้นผิวความน่าจะเป็น 3 มิติในโดเมนไม่ติดลบ ($X, Y \ge 0$)** ร่วมกับ **หอคอยพลังงาน Log-Likelihood เปรียบเทียบรายคลาส (3D Log-Likelihood & Probability Towers)**

### ขอบเขตคลาสและรูปแบบโมเดลที่รองรับ
- **คลาสของ Scikit-learn ที่รองรับ**:
  - `sklearn.naive_bayes.MultinomialNB` (สำหรับข้อมูลนับความถี่ / TF-IDF)
  - `sklearn.naive_bayes.BernoulliNB` (สำหรับข้อมูล Binary $0/1$ ที่คิดโทษของการไม่ปรากฏฟีเจอร์ด้วย)
  - `sklearn.naive_bayes.ComplementNB` (ออกแบบมาเพื่อแก้ปัญหา Imbalanced Text Datasets)
  - `sklearn.naive_bayes.CategoricalNB` (สำหรับข้อมูลหมวดหมู่ที่เข้ารหัสเป็นเลขจำนวนเต็ม $0, 1, \dots, K_j-1$)
- **นามสกุลไฟล์โมเดลที่รองรับ**: `.pkl`, `.joblib`, `.skops` (และ `.mlv`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary & Multiclass Classification**

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Flow Tracing)

### 2.1 สมการคณิตศาสตร์แยกตามชนิดของ Discrete Naive Bayes
กำหนดให้มีคลาส $C$ คลาส ($k \in \{0, \dots, C-1\}$) และข้อมูลนำเข้าขนาด $(N, D)$ ที่มีค่าไม่ติดลบ ($x_{t,j} \ge 0$):

1. **`MultinomialNB` (Linear in Log-Space)**:
   - พารามิเตอร์ภายใน: `class_log_prior_` ($\log \pi_k$) ขนาด $(C,)$ และ `feature_log_prob_` ($\log \theta_{k,j}$) ขนาด $(C, D)$
   - สมการ Joint Log-Likelihood ของแถว $t$:
     $$J_{t,k} = \log \pi_k + \sum_{j=1}^{D} x_{t,j} \log \theta_{k,j}$$
     สังเกตว่าในสเกล Log สมการนี้เป็น **สมการเชิงเส้นตรงเป๊ะ** โดยมีน้ำหนัก $w_{k,j} = \log \theta_{k,j}$ และจุดตัดแกน $b_k = \log \pi_k$!

2. **`BernoulliNB` (Explicit Absence Penalty)**:
   - ข้อมูลถูกแปลงเป็นค่าทวิภาค $b_{t,j} = \mathbb{I}(x_{t,j} > \text{binarize}) \in \{0, 1\}$
   - สมการ Joint Log-Likelihood คิดทั้งฟีเจอร์ที่ปรากฏ ($b_{t,j}=1$) และ **ฟีเจอร์ที่ไม่ปรากฏ ($b_{t,j}=0$)**:
     $$J_{t,k} = \log \pi_k + \sum_{j=1}^{D} \left[ b_{t,j} \log \theta_{k,j} + (1 - b_{t,j}) \log(1 - \theta_{k,j}) \right]$$
   - จัดรูปใหม่เป็นสมการเชิงเส้น:
     $$J_{t,k} = \underbrace{\left(\log \pi_k + \sum_{j=1}^{D} \log(1 - \theta_{k,j})\right)}_{\text{Effective Intercept } \tilde{b}_k \text{ (All-Zero Baseline)}} + \sum_{j=1}^{D} b_{t,j} \underbrace{\left(\log \theta_{k,j} - \log(1 - \theta_{k,j})\right)}_{\text{Feature Presence Shift } \tilde{w}_{k,j}}$$

3. **`ComplementNB`**:
   - คำนวณน้ำหนักจากคลาสอื่นๆ ที่ไม่ใช่ $k$ (`feature_log_prob_` เก็บค่า $\hat{w}_{k,j}$ ที่ผ่านการ Normalize แล้ว):
     $$J_{t,k} = -\sum_{j=1}^{D} x_{t,j} \hat{w}_{k,j}$$

### 2.2 โครงสร้าง Array Shape และ Local Feature Contribution (Waterfall)
- **สำหรับ `MultinomialNB`**:
  - Base Intercept: $\phi_0^{(k)} = \log \pi_k$
  - Feature Contribution: $\phi_{t,j}^{(k)} = x_{t,j} \cdot \log \theta_{k,j}$
- **สำหรับ `BernoulliNB`**:
  - เพื่อให้ผู้ใช้วิเคราะห์ได้ชัดเจนที่สุดว่ารายการใดเกิดจาก *"การมีฟีเจอร์ ($1$)"* และรายการใดเกิดจาก *"การไม่มีฟีเจอร์ ($0$)"*:
  - $\phi_{t,j}^{(k)} = b_{t,j} \log \theta_{k,j} + (1 - b_{t,j}) \log(1 - \theta_{k,j})$
  - ผลรวมตรวจสอบ: $\log \pi_k + \sum_{j=1}^{D} \phi_{t,j}^{(k)} = J_{t,k}$ ตรงกับ `model.predict_joint_log_proba(X)` 100%!

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 รายละเอียดองค์ประกอบในฉาก 3 มิติ (3D Visual Elements)
1. **Non-Negative Quadrant 3D Probability Surface ($X \ge 0, Y \ge 0$)**:
   - พื้นผิวความน่าจะเป็น 3 มิติที่ถูกจำกัดขอบเขตให้อยู่ในจตุภาคบวก ($x \ge 0, y \ge 0$) เสมอ พร้อมแสดงกำแพงกั้นโปร่งแสงที่แกน $X = 0$ และ $Y = 0$ พร้อมป้ายกำกับ **"Non-Negative Count Domain ($X \ge 0$)"**
   - สำหรับ `BernoulliNB`: แสดงหมุดพลังงานพิเศษที่มุมทั้ง 4 ของตารางลูกบาศก์ทวิภาค $(0,0), (1,0), (0,1), (1,1)$ พร้อมเส้น阈值 `binarize`
2. **3D Dynamic Log-Likelihood & Posterior Towers (หอคอยพลังงานเปรียบเทียบรายคลาส)**:
   - ด้านข้างของเวที 3 มิติมีฐานแท่น **3D Probability Towers** (เสาคริสตัล 3 มิติ 1 ต้นต่อ 1 คลาส)
   - ทุกครั้งที่เฟรม $t$ เปลี่ยนไป ความสูงของเสาคริสตัลแต่ละคลาสจะพุ่งขึ้น-ลงตามค่า Posterior Probability $P(C_k \mid \mathbf{x}_t)$ แบบ Smooth Spring Animation โดยเสาของคลาสที่ชนะจะส่องแสงเรืองรองและมียอดมงกุฎลอยอยู่ด้านบน
3. **Active Token / Feature Presence Beams**:
   - ยิงลำแสงจากฟีเจอร์ที่มีค่า $x_{t,j} > 0$ เข้าสู่หอคอยของคลาสที่ฟีเจอร์นั้นสนับสนุนมากที่สุด

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (Model-Specific HUD Controls)
- **NB Variant & Smoothing Badge**: แสดงชนิดโมเดล (`MultinomialNB`, `BernoulliNB`, `ComplementNB`), ค่า Laplace Smoothing `alpha`, และเกณฑ์ `binarize` (สำหรับ BernoulliNB)
- **Presence vs Absence Contribution Filter (เฉพาะ `BernoulliNB`)**: สวิตช์เลือกดูเฉพาะฟีเจอร์ที่ปรากฏ ($b_{t,j}=1$) หรือดูรวมผลกระทบจากฟีเจอร์ที่ไม่ปรากฏ ($b_{t,j}=0$)
- **Top Discriminative Features Ranking**: จัดอันดับฟีเจอร์ที่มีอัตราส่วน $\max_k \log \theta_{k,j} - \min_k \log \theta_{k,j}$ สูงที่สุด

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับประสบการณ์ผู้ใช้ (User Interaction Flow)
1. **ขณะกด Play ในโหมด VCR Playback**:
   - ลูกบอลข้อมูลวิ่งไปตามพื้นผิวความน่าจะเป็นในแดนบวก พร้อมกับที่หอคอย 3D Probability Towers ด้านข้างเต้นขึ้น-ลงตามคะแนนของแต่ละคลาสในเฟรมนั้น
2. **เมื่อลาก Scrubber ข้ามจุดเปลี่ยนคำตัดสิน**:
   - หอคอยของคลาสใหม่จะพุ่งแซงความสูงของหอคอยคลาสเดิม พร้อมแสดงรายการฟีเจอร์/คำศัพท์สำคัญ (Top Active Tokens) ที่เป็นตัวพลิกผลลัพธ์ใน HUD
3. **เมื่อขยับสไลเดอร์ใน What-If Sandbox**:
   - สไลเดอร์ทุกตัวของ `MultinomialNB` และ `CategoricalNB` จะถูกล็อกค่าต่ำสุดไว้ที่ `min = 0.0` โดยอัตโนมัติ (ป้องกันการส่งค่าติดลบ)
   - สำหรับ `BernoulliNB` เมื่อผู้ใช้เลื่อนสไลเดอร์ข้ามค่า `binarize` (เช่น ข้าม `0.0` หรือ `0.5`) สถานะของฟีเจอร์จะดีดตัวแบบ Step Function จาก `ABSENT (0)` เป็น `PRESENT (1)` และหอคอยความน่าจะเป็นจะเปลี่ยนความสูงแบบก้าวกระโดดทันที!

### 4.2 ASCII Wireframe ของแผง HUD เฉพาะโมเดล Multinomial / Bernoulli NB
```text
+-------------------------------------------------------------------------+
| [DISCRETE NAIVE BAYES HUD]  Model: BernoulliNB | alpha: 1.0 | bin: 0.0  |
+-------------------------------------------------------------------------+
| Waterfall Mode: [ *Show Active (x>0) + Absent (x=0)* ] [ Active Only ]  |
| Target Class  : [ Class 0 (Ham) ]  [ *Class 1 (Spam)* ]                 |
+-------------------------------------------------------------------------+
| FRAME #0104 3D PROBABILITY TOWERS & LOG-LIKELIHOOD TRACE:               |
|  Class 1 (Spam) : [██████████████████░░] 91.2% (Joint Log-P: -8.420)    |
|  Class 0 (Ham)  : [██░░░░░░░░░░░░░░░░░░]  8.8% (Joint Log-P: -10.761)   |
+-------------------------------------------------------------------------+
| BERNOULLI PRESENCE (1) & ABSENCE (0) LOG-ODDS CONTRIBUTIONS:            |
|  Prior Log-Ratio log(π1/π0)     : -0.405  [      <==|            ]      |
|  word_free = 1 (PRESENT)        : +2.850  [         |==========> ]      |
|  word_win  = 1 (PRESENT)        : +1.920  [         |=======>    ]      |
|  word_meeting = 0 (ABSENT)      : +0.640  [         |==>         ]      |
|  word_report  = 0 (ABSENT)      : -2.664  [ <=======|            ]      |
+-------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain-Driven Design
- `backend/app/domains/model_engine/adapters/probabilistic/discrete_nb.py`
- `frontend/src/domains/stage3d/components/actors/ProbabilityTowers3D.tsx`
- `frontend/src/domains/hud/components/inspectors/DiscreteNBInspector.tsx`

### 5.2 โค้ดต้นแบบ `DiscreteNBAdapter` (Backend Python)
```python
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from app.domains.model_engine.adapters.base import BaseModelAdapter
from app.domains.model_engine.schemas import InternalFlowBatch, SurfaceMesh3D

class DiscreteNBAdapter(BaseModelAdapter):
    SUPPORTED_CLASSES = (MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB)

    def __init__(self, model: Any, feature_names: List[str]):
        super().__init__(model, feature_names)
        self.classes_: np.ndarray = np.asarray(model.classes_)
        self.n_classes: int = len(self.classes_)
        self.is_bernoulli = isinstance(model, BernoulliNB)
        self.is_complement = isinstance(model, ComplementNB)
        self.is_categorical = isinstance(model, CategoricalNB)
        self.class_log_prior_: np.ndarray = np.asarray(
            getattr(model, "class_log_prior_", np.zeros(self.n_classes)), dtype=np.float64
        )
        if not self.is_categorical:
            self.feature_log_prob_: np.ndarray = np.asarray(model.feature_log_prob_, dtype=np.float64)

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "model_family": "probabilistic_discrete_nb",
            "algorithm": self.model.__class__.__name__,
            "task_type": "binary_classification" if self.n_classes == 2 else "multiclass_classification",
            "classes": self.classes_.tolist(),
            "alpha": float(getattr(self.model, "alpha", 1.0)),
            "binarize": getattr(self.model, "binarize", None),
            "requires_non_negative_input": not self.is_bernoulli,
            "class_log_prior": self.class_log_prior_.tolist(),
        }

    def _compute_contributions(self, X_arr: np.ndarray, winning_idx: np.ndarray) -> np.ndarray:
        if self.is_categorical:
            # CategoricalNB stores list of (C, n_categories_j) arrays in feature_log_prob_
            N, D = X_arr.shape
            contribs = np.zeros((N, D), dtype=np.float64)
            X_int = np.clip(np.rint(X_arr).astype(int), 0, None)
            for j in range(D):
                log_p_j = self.model.feature_log_prob_[j]  # (C, K_j)
                safe_vals = np.clip(X_int[:, j], 0, log_p_j.shape[1] - 1)
                contribs[:, j] = log_p_j[winning_idx, safe_vals]
            return contribs

        if self.is_bernoulli:
            thresh = getattr(self.model, "binarize", 0.0)
            X_bin = (X_arr > thresh).astype(np.float64) if thresh is not None else X_arr
            neg_prob = np.log(np.maximum(1.0 - np.exp(self.feature_log_prob_), 1e-15))
            if self.n_classes == 2:
                pos_diff = self.feature_log_prob_[1] - self.feature_log_prob_[0]
                neg_diff = neg_prob[1] - neg_prob[0]
                return X_bin * pos_diff[np.newaxis, :] + (1.0 - X_bin) * neg_diff[np.newaxis, :]
            return X_bin * self.feature_log_prob_[winning_idx] + (1.0 - X_bin) * neg_prob[winning_idx]

        if self.is_complement:
            return -X_arr * self.feature_log_prob_[winning_idx]

        # MultinomialNB
        if self.n_classes == 2:
            w_diff = self.feature_log_prob_[1] - self.feature_log_prob_[0]
            return X_arr * w_diff[np.newaxis, :]
        return X_arr * self.feature_log_prob_[winning_idx]

    def trace_internal_flow(self, X_batch: np.ndarray) -> InternalFlowBatch:
        X_arr = np.asarray(X_batch.toarray() if hasattr(X_batch, "toarray") else X_batch, dtype=np.float64)
        if not self.is_bernoulli:
            X_arr = np.maximum(X_arr, 0.0)

        preds = self.model.predict(X_arr)
        probas = self.model.predict_proba(X_arr)
        jll = self.model.predict_joint_log_proba(X_arr) if hasattr(
            self.model, "predict_joint_log_proba"
        ) else self.model._joint_log_likelihood(X_arr)

        win_idx = np.argmax(probas, axis=1)
        contributions = self._compute_contributions(X_arr, win_idx)
        base_b = [float(self.class_log_prior_[1] - self.class_log_prior_[0])] if self.n_classes == 2 else self.class_log_prior_.tolist()

        return InternalFlowBatch(
            predictions=preds.tolist(),
            probabilities=probas.astype(np.float32).tolist(),
            decision_scores=jll.astype(np.float32).tolist(),
            feature_contributions=contributions.astype(np.float32).tolist(),
            base_intercepts=base_b,
            extra_telemetry={"tower_heights": probas.astype(np.float32).tolist()},
        )

    def compute_surface_mesh(
        self, axis_x_idx: int, axis_y_idx: int,
        x_range: Tuple[float, float], y_range: Tuple[float, float],
        fixed_features: np.ndarray, grid_resolution: int = 50, target_class_idx: int = 1
    ) -> SurfaceMesh3D:
        # บังคับ Non-negative Grid Domain สำหรับ MultinomialNB, ComplementNB, CategoricalNB
        safe_x_min = max(0.0, x_range[0]) if not self.is_bernoulli else x_range[0]
        safe_y_min = max(0.0, y_range[0]) if not self.is_bernoulli else y_range[0]
        safe_x_max = max(safe_x_min + 1.0, x_range[1])
        safe_y_max = max(safe_y_min + 1.0, y_range[1])

        xs = np.linspace(safe_x_min, safe_x_max, grid_resolution, dtype=np.float64)
        ys = np.linspace(safe_y_min, safe_y_max, grid_resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)

        base_vec = np.maximum(fixed_features.astype(np.float64), 0.0) if not self.is_bernoulli else fixed_features.astype(np.float64)
        grid_X = np.tile(base_vec, (grid_resolution * grid_resolution, 1))
        grid_X[:, axis_x_idx] = xx.ravel()
        grid_X[:, axis_y_idx] = yy.ravel()

        if self.is_categorical:
            grid_X = np.clip(np.rint(grid_X).astype(int), 0, None)

        probas = self.model.predict_proba(grid_X)
        c_idx = min(target_class_idx, self.n_classes - 1)
        z_surface = probas[:, c_idx].reshape(grid_resolution, grid_resolution)

        return SurfaceMesh3D(
            x_grid=xs.astype(np.float32).tolist(),
            y_grid=ys.astype(np.float32).tolist(),
            z_values=z_surface.astype(np.float32).tolist(),
            auxiliary_grids={"clamped_non_negative": not self.is_bernoulli},
            boundary_threshold=0.5,
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` และ `SurfaceMesh3D`
```json
{
  "InternalFlowBatch": {
    "predictions": [1, 0],
    "probabilities": [[0.088, 0.912], [0.845, 0.155]],
    "decision_scores": [[-10.761, -8.420], [-6.120, -7.815]],
    "feature_contributions": [[2.850, 1.920, -2.664], [-1.120, -0.850, 0.410]],
    "base_intercepts": [-0.405],
    "extra_telemetry": { "tower_heights": [[0.088, 0.912], [0.845, 0.155]] }
  },
  "SurfaceMesh3D": {
    "x_grid": [0.0, 10.0],
    "y_grid": [0.0, 10.0],
    "z_values": [[0.400, 0.912], [0.150, 0.780]],
    "auxiliary_grids": { "clamped_non_negative": true },
    "boundary_threshold": 0.5
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การจัดการ High-Dimensional Sparse Vocabulary (เช่น 50,000 คำใน `MultinomialNB`)**:
   - โมเดล NLP ที่ใช้ `MultinomialNB` มักมีจำนวนฟีเจอร์ $D > 10,000$ โดยที่ในแต่ละแถวมีคำที่ปรากฏจริง ($x_{t,j} > 0$) ไม่ถึง 50 คำ
   - ในกรณีที่ $D > 500$ ให้ส่งเฉพาะ **Top-30 Positive & Negative Feature Contributors** ของแต่ละแถวไปใน `feature_contributions` แทนการส่ง Array ขนาด $(N, 50000)$ เต็มรูปแบบ ช่วยลดขนาด JSON Payload ลงกว่า 99%!
2. **การReuse Mesh สำหรับ `ProbabilityTowers3D`**:
   - หอคอยความน่าจะเป็นใช้ `THREE.BoxGeometry(1, 1, 1)` ที่เลื่อนจุดหมุนฐานมาอยู่ที่ $Z = 0$ (`geometry.translate(0, 0, 0.5)`) แล้วปรับเฉพาะ `meshRef.current.scale.z = Math.max(0.01, prob)` ใน `useFrame` โดยไม่สร้าง Geometry ใหม่

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| ลำดับ | กรณีปัญหา (Edge Case) | สาเหตุทางเทคนิค (Root Cause) | แนวทางแก้ไขและ Fallback ใน Adapter / UI |
| :---: | :--- | :--- | :--- |
| 1 | **`ValueError: Negative values in data passed to MultinomialNB`** | ตอนสร้างขอบเขต Grid 3D ระบบทั่วไปอาจขยาย Padding ไปฝั่งลบ (เช่น `[-1.0, +5.0]`) ซึ่งทำให้ `MultinomialNB` และ `ComplementNB` โยน Exception ทันที | บังคับ `safe_x_min = max(0.0, x_range[0])` และ `np.maximum(X_arr, 0.0)` เสมอสำหรับทุกคลาสยกเว้น `BernoulliNB` |
| 2 | **`BernoulliNB` คิดคะแนนจากฟีเจอร์ที่เป็น `0` ด้วย** | หากคำนวณ Waterfall เฉพาะ `X * feature_log_prob_` เหมือน `MultinomialNB` ผลรวมจะไม่ตรงกับ `predict_log_proba` ของ `BernoulliNB` | ต้องบวกเทอม $(1 - b_{t,j}) \log(1 - \theta_{k,j})$ เข้าไปใน `_compute_contributions` ของ `BernoulliNB` เสมอ |
| 3 | **`CategoricalNB` เกิด `IndexError` เมื่อเจอหมวดหมู่เกินกว่าตอนเทรน** | หาก Grid 3D หรือข้อมูลทดสอบมีค่าจำนวนเต็มสูงกว่าจำนวนหมวดหมู่สูงสุด $K_j - 1$ ที่โมเดลเคยเห็นตอน `fit` จะเกิด `IndexError` | ใช้ `np.clip(np.rint(X), 0, n_categories_j - 1)` ก่อนส่งเข้า `CategoricalNB` เสมอ พร้อมแสดงป้ายเตือน `Unseen Category Clamped` |
| 4 | **`BernoulliNB(binarize=None)` เมื่อข้อมูลถูกแปลงเป็น Binary มาแล้ว** | ผู้ใช้อาจตั้ง `binarize=None` ตอนเทรน ทำให้ `model.binarize` เป็น `None` และการเปรียบเทียบ `X_arr > model.binarize` จะเกิด `TypeError` | ตรวจเช็ก `if thresh is not None:` ก่อนเปรียบเทียบเสมอ หากเป็น `None` ให้ถือว่าข้อมูลอยู่ในรูป `[0, 1]` แล้ว |
| 5 | **`ComplementNB` ใช้เครื่องหมายลบหน้าผลรวมน้ำหนัก** | ใน `ComplementNB` ค่า `feature_log_prob_` คือน้ำหนักความผิดปกติของคลาสตรงข้าม ยิ่งค่าน้อยแปลว่ายิ่งมีโอกาสเป็นคลาส $k$ มาก | คูณ `-1.0` หน้า `X_arr * self.feature_log_prob_` เพื่อให้ทิศทางบวก/ลบในกราฟ Waterfall สื่อความหมายตรงกับคลาสที่ชนะ |
| 6 | **`alpha=0` (ปิด Laplace Smoothing) ทำให้เกิด `-inf` ใน `feature_log_prob_`** | เมื่อตั้ง `alpha=0` ความน่าจะเป็นศูนย์จะกลายเป็น $\log(0) = -\infty$ ทำให้เกิด `NaN` ตอนคูณกับ `0` (`0 * -inf = nan`) | ใช้ `np.nan_to_num(contributions, nan=0.0, neginf=-30.0)` ก่อนแปลงเป็น JSON เสมอ |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] `DiscreteNBAdapter` รองรับครบทั้ง `MultinomialNB`, `BernoulliNB`, `ComplementNB`, และ `CategoricalNB`
- [ ] ระบบสร้าง Grid 3D และ What-If Sliders ป้องกันค่าติดลบ ($X < 0$) โดยอัตโนมัติสำหรับ `MultinomialNB`, `ComplementNB`, และ `CategoricalNB`
- [ ] สมการแยกส่วนประกอบของ `BernoulliNB` รวมทั้งเทอม Presence ($1$) และ Absence ($0$) จนผลรวมตรงกับ `predict_joint_log_proba` 100%
- [ ] ฉาก 3 มิติแสดงพื้นผิวความน่าจะเป็นในแดนบวกพร้อมหอคอย **3D Probability Towers** ที่ขยับขึ้นลงตามเฟรมอย่างลื่นไหล 60FPS
