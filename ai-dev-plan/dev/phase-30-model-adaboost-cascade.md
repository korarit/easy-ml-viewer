# Phase 30: AdaBoost Adapter & 3D Sequential Weak-Learner Cascade

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter สำหรับโมเดลต้นแบบแห่งตระกูล Boosting คือ **AdaBoost (Adaptive Boosting)** เพื่อจำลองกลไกการทำงานแบบลำดับขั้น (Sequential Weak-Learner Cascade) ที่ซึ่งตัวเรียนแบบอ่อน (Weak Learners / Decision Stumps) แต่ละตัว $m = 1, 2, \dots, M$ ออกแรงดึงค่าการทำนายสะสมตามน้ำหนักความน่าเชื่อถือของตัวเอง ($\alpha_m$ หรือ `estimator_weights_`) ออกมาเป็น **3D Sequential Cascade Ladder (บันไดการไต่ระดับของเหล่า Weak Learners)** ในฉาก 3 มิติ

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `sklearn.ensemble.AdaBoostClassifier` (รองรับทั้งอัลกอริทึม `SAMME` และ `SAMME.R`)
  - `sklearn.ensemble.AdaBoostRegressor` (อัลกอริทึม `AdaBoost.R2` ทั้ง `linear`, `square`, `exponential` loss)
  - รองรับทั้งกรณีที่ Base Estimator เป็น Decision Tree Stump (`max_depth=1`), ต้นไม้ลึก (`max_depth > 1`), หรือ Estimator ชนิดอื่น
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.pkl`, `.pickle`, `.joblib`
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: ติดตามการเปลี่ยนแปลงของความน่าจะเป็นสะสม $P_m(y=1 \mid \mathbf{x})$ และคะแนนตัดสินใจสะสม $F_m(\mathbf{x})$ ทีละขั้น
  - **Multiclass Classification**: ติดตามความน่าจะเป็นสะสมของคลาสเป้าหมายในแต่ละ Stage $m = 1 \dots M$
  - **Regression**: ติดตามค่ากลางถ่วงน้ำหนัก (Weighted Median / Weighted Average) ที่ค่อยๆ ขยับเข้าหาคำตอบสุดท้าย

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 สมการของ AdaBoost (`SAMME` / `SAMME.R` / `AdaBoost.R2`)
AdaBoost สร้างลำดับของ Weak Learners $h_1, h_2, \dots, h_M$ (`model.estimators_`) พร้อมค่าน้ำหนักความสำคัญ $\alpha_1, \alpha_2, \dots, \alpha_M$ (`model.estimator_weights_`) และค่าความผิดพลาดตอนเทรน $\epsilon_1, \dots, \epsilon_M$ (`model.estimator_errors_`):
- **การสะสมคะแนนทีละขั้น (Staged Cumulative Progression)**:
  ใน Stage ที่ $m \in \{1, \dots, M\}$ คำตอบสะสมเมื่อใช้ Weak Learners เพียง $m$ ตัวแรกคือ:
  $$F_m(\mathbf{x}) = \frac{\sum_{t=1}^{m} \alpha_t h_t(\mathbf{x})}{\sum_{t=1}^{m} \alpha_t}$$
- **การแกะรอยด้วย `staged_predict_proba` และ `staged_predict`**:
  Scikit-learn มี Generator Method ในตัวที่คำนวณคำตอบสะสมทีละ Stage โดยไม่ต้องเขียนสมการแยกตาม `SAMME` หรือ `R2` เอง:
  - สำหรับ `AdaBoostClassifier`: `list(model.staged_predict_proba(X_chunk))` คืนลิสต์ความยาว $M$ ที่แต่ละสมาชิกมีขนาด `(n_samples, n_classes)`
  - สำหรับ `AdaBoostRegressor`: `list(model.staged_predict(X_chunk))` คืนลิสต์ความยาว $M$ ที่แต่ละสมาชิกมีขนาด `(n_samples,)`

### 2.2 การสกัด Marginal Step Jump ($\Delta S_m$) และ Feature Attribution
- กำหนดให้ $S_m(\mathbf{x})$ คือค่าการทำนายสะสม ณ Stage ที่ $m$ (ความน่าจะเป็นของคลาสเป้าหมาย หรือค่าทำนาย Regression):
  - ค่าเริ่มต้นก่อนตัวเรียนตัวแรก: $S_0 = \text{Base Value}$ (ค่าเฉลี่ยของ $S_1$ ทั้งชุดข้อมูล หรือ $1/C$ สำหรับ Classification)
  - แรงดึงสุทธิของตัวเรียนตัวที่ $m$ (Marginal Stage Delta):
    $$\Delta S_m(\mathbf{x}) = S_m(\mathbf{x}) - S_{m-1}(\mathbf{x})$$
- **การผูกแรงดึง $\Delta S_m$ เข้ากับฟีเจอร์ (Feature Attribution)**:
  - กรณีที่ `estimators_[m]` เป็น **Decision Stump (`node_count == 3`, แบ่งแค่ครั้งเดียวที่ Root)**: ฟีเจอร์ที่ถูกใช้ที่ Root (`est.tree_.feature[0]`) คือผู้รับผิดชอบ $\Delta S_m(\mathbf{x})$ เต็ม $100\%$!
  - กรณีที่ `estimators_[m]` เป็นต้นไม้หลายชั้น (`max_depth > 1`): กระจายค่า $\Delta S_m(\mathbf{x})$ ให้แก่ฟีเจอร์ที่อยู่ใน `decision_path` ของต้นไม้ต้นที่ $m$ ตามสัดส่วนขนาดการเปลี่ยนแปลงภายในต้นไม้นั้น
  - กรณีที่ `estimators_[m]` ไม่ใช่ต้นไม้ (เช่น LogisticRegression): ใช้ผลคูณความชันสัมพัทธ์ในการกระจายค่า $\Delta S_m(\mathbf{x})$
- **Array Shape ของ Contributions**:
  - `feature_contributions`: `(n_samples, n_features)` (ชนิด `float32`)
  - `staged_trajectory`: `(n_samples, n_sampled_stages)` เก็บเส้นทางการขยับของคำตอบตั้งแต่วินาทีแรกจนถึงตัวเรียนสุดท้าย

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Sequential Cascade Ladder (`AdaBoostCascadeLadder.tsx`)**:
   - ที่ตำแหน่ง $(x_t, y_t)$ ของลูกบอลข้อมูลในเฟรมปัจจุบัน จะมี **เสาบันไดแสงแนวดิ่ง (Vertical Boosting Cascade)** ตั้งขึ้นมาจากระดับ `base_value` ไปสู่ความสูงสุดท้าย `prediction`
   - ขั้นบันไดแต่ละขั้นแทนตัวเรียน $h_m$ ที่มีอิทธิพลสูงสุด โดยความหนาของวงแหวนแสงแทนค่าน้ำหนัก `estimator_weights_[m]` ($\alpha_m$) และสีของขั้นบันไดแทนทิศทางการดึง (สีเขียว/ฟ้า = ดึงขึ้น $\Delta S_m > 0$, สีแดง/ส้ม = กดลง $\Delta S_m < 0$)
2. **Stump Split Curtain Planes (`StumpSplitCurtains.tsx`)**:
   - เนื่องจาก AdaBoost มักใช้ Decision Stumps ที่มีเส้นแบ่งพาดขวางแกน $X$ หรือ $Y$ เพียงเส้นเดียวต่อตัวเรียน ผู้ใช้สามารถเปิดดูม่านแสงของ Stump ตัวที่ $m$ ที่เลือก เพื่อดูว่าตัวเรียนตัวที่ $m$ แบ่งครึ่งโลกตรงพิกัดใด
3. **Stepped Plaid Terrain (`SteppedVoxelTerrain.tsx`)**:
   - พื้นผิวขั้นบันได 3 มิติของ AdaBoost ที่เกิดจากการบวกซ้อนทับกันของ Stump แนวตั้งและแนวนอนจนกลายเป็นลายตารางหมากรุก 3 มิติ (3D Plaid / Tartan Terraces) อันเป็นเอกลักษณ์เฉพาะของ Boosting Stumps

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`AdaBoostHudPanel.tsx`)
- **Staged Boosting Curve Chart**: กราฟเส้นขนาดกะทัดรัดใน HUD ที่พล็อตค่าคำตอบสะสม $S_m(\mathbf{x}_t)$ ตามแกนแนวนอน $m = 1 \dots M$ พร้อมเส้นประแสดงขีดแบ่ง Decision Threshold ($0.5$) เพื่อให้เห็นว่าคำตอบข้ามเส้นแบ่งที่ Stage ที่เท่าไหร่
- **Weak Learner Weight & Error Inspector**: ตาราง/แถบเลื่อนแสดงตัวเรียน $h_m$ พร้อมค่า `estimator_weights_[m]` ($\alpha_m$), `estimator_errors_[m]` ($\epsilon_m$), และกฎการแบ่งของ Stump ตัวนั้น
- **Stage Scrubbing Slider**: เลื่อนดูว่าหากหยุด Boosting ที่ Stage $m < M$ (Early Stopping Simulation) ลูกบอลข้อมูลและคำตอบจะอยู่ที่ระดับความสูงเท่าใด

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - ลูกบอลข้อมูลเคลื่อนที่ไปบนภูมิประเทศตารางหมากรุก 3D พร้อมกับเสาบันไดแสง **Sequential Cascade Ladder** ที่ยืด-หดตามจังหวะการดึงของเหล่า Weak Learners
   - กราฟ **Staged Boosting Curve** ใน HUD จะวาดเส้นกราฟการลู่เข้าของคำตอบใหม่ในทุกเฟรมแบบ Real-time
2. เมื่ออยู่ในโหมด **What-If Sandbox**:
   - เมื่อผู้ใช้ลากลูกบอลข้ามเส้นแบ่งของ Stump ที่มีน้ำหนัก $\alpha_m$ สูง ขั้นบันไดแสงขั้นที่ $m$ บนเสาแนวดิ่งจะกลับทิศทางจากบวกเป็นลบทันที และกราฟ Staged Curve จะหักหัวลงให้เห็นจะๆ ว่าตัวเรียนตัวไหนที่เปลี่ยนใจ!

### 4.2 ASCII Wireframe ของหน้าจอ AdaBoost HUD & Sequential Cascade
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: AdaBoostClassifier (n_estimators=50, base=Stump)    [ Stage View: 50/50 ]|
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [STAGED BOOSTING CURVE (t=19)|
|            3D PLAID TERRACES & SEQUENTIAL CASCADE LADDER          |  1.0 |          .---* (0.82) |
|                                                                   |      |      .--'             |
|                        ((o))  <-- Final Prediction: 0.82          |  0.5 |----/----------------- |
|                          ^                                        |      |  .'  <-- Crossed @ m=8|
|                          | [Stage #12: Age <= 29.5, Δ=+0.14]      |  0.0 +---------------------- |
|                          o                                        |      1      10     25     50 |
|                          ^                                        +------------------------------+
|                          | [Stage #04: Debt > 12k,  Δ=-0.09]      | [TOP INFLUENTIAL LEARNERS]   |
|                          o                                        | #01 (w=1.42, err=0.11):      |
|                          ^                                        |   Income <= 65.0 ➔ Δ = +0.24 |
|                          | [Stage #01: Income <= 65, Δ=+0.24]     | #12 (w=0.95, err=0.21):      |
|       +------------------+------------------+                     |   Age <= 29.5    ➔ Δ = +0.14 |
|       |          Base Score: 0.50           |                     | #04 (w=0.88, err=0.23):      |
|       +-------------------------------------+                     |   Debt > 12.0    ➔ Δ = -0.09 |
+-------------------------------------------------------------------+------------------------------+
| [STAGE DELTAS] Base: 0.50 | Income: +0.26 | Age: +0.15 | Debt: -0.09 ===> Final Pred: 0.82       |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0019 / 1000  [==(*)------------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/trees/adaboost.py` (`AdaBoostAdapter`)
- `frontend/src/domains/stage3d/actors/AdaBoostCascadeLadder.tsx`
- `frontend/src/domains/hud/components/AdaBoostHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `AdaBoostAdapter`
```python
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.tree import BaseDecisionTree
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

MAX_TRAJECTORY_POINTS = 50

class AdaBoostAdapter(BaseModelAdapter):
    """Adapter สำหรับ AdaBoostClassifier และ AdaBoostRegressor."""

    def __init__(self, model: Any, feature_names: list[str], target_class_idx: int = 1) -> None:
        super().__init__(model, feature_names)
        self.estimators: list[Any] = list(model.estimators_)
        self.n_estimators: int = len(self.estimators)
        self.estimator_weights: np.ndarray = np.asarray(model.estimator_weights_, dtype=np.float32)
        self.estimator_errors: np.ndarray = np.asarray(model.estimator_errors_, dtype=np.float32)
        self.is_classifier: bool = hasattr(model, "staged_predict_proba")
        self.target_class_idx: int = target_class_idx

    def _compute_staged_matrix(self, X_f32: np.ndarray) -> np.ndarray:
        """คืนเมทริกซ์ค่าทำนายสะสมขนาด (n_samples, n_estimators) จาก staged_predict(_proba)."""
        if self.is_classifier:
            staged_list = []
            for proba in self.model.staged_predict_proba(X_f32):
                idx = min(self.target_class_idx, proba.shape[1] - 1)
                staged_list.append(proba[:, idx].astype(np.float32))
            return np.column_stack(staged_list)
        else:
            staged_list = [pred.astype(np.float32) for pred in self.model.staged_predict(X_f32)]
            return np.column_stack(staged_list)

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        X_f32 = np.asarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        staged_mat = self._compute_staged_matrix(X_f32)  # shape: (n_samples, M)

        base_val = 0.5 if self.is_classifier else float(np.mean(staged_mat[:, 0]))
        prev_col = np.full((n_samples, 1), base_val, dtype=np.float32)
        stage_deltas = np.diff(np.hstack([prev_col, staged_mat]), axis=1)  # shape: (n_samples, M)

        contribs = np.zeros((n_samples, n_features), dtype=np.float32)
        for m, est in enumerate(self.estimators):
            delta_m = stage_deltas[:, m]
            if isinstance(est, BaseDecisionTree) and est.tree_.node_count > 1:
                if est.tree_.max_depth == 1:
                    f_idx = int(est.tree_.feature[0])
                    if f_idx >= 0:
                        contribs[:, f_idx] += delta_m
                else:
                    csr = est.decision_path(X_f32)
                    feats = est.tree_.feature
                    for i in range(n_samples):
                        path = csr.indices[csr.indptr[i] : csr.indptr[i + 1]]
                        internal_feats = [int(feats[u]) for u in path[:-1] if feats[u] >= 0]
                        if internal_feats:
                            share = delta_m[i] / len(internal_feats)
                            for f_idx in internal_feats:
                                contribs[i, f_idx] += share

        # ย่อความยาว staged_trajectory ไม่ให้เกิน 50 จุดสำคัญ
        step_indices = np.unique(np.linspace(0, self.n_estimators - 1, min(self.n_estimators, MAX_TRAJECTORY_POINTS), dtype=int))
        sampled_trajectory = staged_mat[:, step_indices]

        # ดึง Top 8 ตัวเรียนที่ออกแรงดึงสูงสุดของแต่ละแถว
        top_learners_per_sample: list[list[dict[str, Any]]] = []
        k_top = min(8, self.n_estimators)
        for i in range(n_samples):
            top_m_idx = np.argsort(np.abs(stage_deltas[i]))[::-1][:k_top]
            top_m_idx_sorted = np.sort(top_m_idx)
            row_steps = []
            for m_idx in top_m_idx_sorted:
                est = self.estimators[int(m_idx)]
                f_idx = int(est.tree_.feature[0]) if isinstance(est, BaseDecisionTree) and est.tree_.node_count > 1 else -1
                th = float(est.tree_.threshold[0]) if f_idx >= 0 else 0.0
                row_steps.append({
                    "stage": int(m_idx),
                    "weight": round(float(self.estimator_weights[m_idx]), 4),
                    "error": round(float(self.estimator_errors[m_idx]), 4),
                    "feature_idx": f_idx,
                    "threshold": round(th, 4),
                    "delta": round(float(stage_deltas[i, m_idx]), 5),
                    "cumulative": round(float(staged_mat[i, m_idx]), 5),
                })
            top_learners_per_sample.append(row_steps)

        return InternalFlowBatch(
            base_value=base_val,
            feature_contributions=contribs,
            extra_telemetry={
                "model_family": "adaboost",
                "sampled_stage_indices": step_indices.tolist(),
                "staged_trajectory": np.round(sampled_trajectory, 4).tolist(),
                "top_influential_learners": top_learners_per_sample,
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 19,
  "prediction": 0.82,
  "base_value": 0.50,
  "feature_contributions": [0.26, 0.15, -0.09, 0.0],
  "telemetry": {
    "model_family": "adaboost",
    "sampled_stage_indices": [0, 1, 2, 3, 4, 12, 49],
    "staged_trajectory": [0.74, 0.71, 0.73, 0.65, 0.56, 0.70, 0.82],
    "top_influential_learners": [
      {"stage": 0, "weight": 1.42, "error": 0.11, "feature_idx": 0, "threshold": 65.0, "delta": 0.24, "cumulative": 0.74},
      {"stage": 3, "weight": 0.88, "error": 0.23, "feature_idx": 2, "threshold": 12.0, "delta": -0.09, "cumulative": 0.56},
      {"stage": 11, "weight": 0.95, "error": 0.21, "feature_idx": 1, "threshold": 29.5, "delta": 0.14, "cumulative": 0.70}
    ]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การบริโภค Generator `staged_predict_proba` แบบประหยัดหน่วยความจำ**:
   - `model.staged_predict_proba(X_f32)` เป็น Generator ที่คำนวณสะสมในระดับ Python/Cython ต่อเนื่องโดยไม่ต้องรันตัวเรียนตัวที่ $1 \dots m$ ซ้ำตั้งแต่ต้นในทุก Stage (ความซับซ้อนเชิงเวลา $\mathcal{O}(M)$ แทนที่จะเป็น $\mathcal{O}(M^2)$)
   - ดึงเฉพาะคอลัมน์ `proba[:, target_class_idx]` ทันทีในแต่ละรอบของ Generator เพื่อลดการใช้ RAM สำหรับงาน Multiclass ลง $C$ เท่า
2. **การบีบอัด `staged_trajectory` ไม่เกิน 50 จุด (`MAX_TRAJECTORY_POINTS = 50`)**:
   - หากผู้ใช้ตั้ง `n_estimators = 500` การเก็บทั้ง 500 จุดในทุกเฟรมจะทำให้ JSON Payload ใหญ่เกินจำเป็น เราจึงใช้ `np.linspace(0, M-1, 50, dtype=int)` สุ่มจุดตัวแทนแบบสม่ำเสมอ พร้อมคัดแยก `top_influential_learners` 8 อันดับแรกที่มี $|\Delta S_m|$ สูงสุดมาแสดงเป็นขั้นบันได 3D
3. **การแยกกรณี Fast Path สำหรับ Decision Stumps (`max_depth == 1`)**:
   - เนื่องจาก 95% ของโมเดล AdaBoost ใช้ Decision Stump ที่มีโหนดแบ่งเพียงโหนดเดียว (`tree_.feature[0]`) โค้ดจึงเช็ก `est.tree_.max_depth == 1` เพื่อบวก `delta_m` เข้า `contribs[:, f_idx]` แบบ Vectorized ทั้ง Chunk ในคำสั่งเดียวโดยไม่ต้องเรียก `decision_path`!

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **Early Termination ตอนเทรน (`len(estimators_) < n_estimators`)** | หาก AdaBoost ได้ Perfect Fit หรือ Error $\ge 0.5$ ตั้งแต่ Stage ต้นๆ อาเรย์ `estimator_weights_` จะมีค่า `0` ในช่องท้ายๆ และ `len(estimators_)` จะน้อยกว่าพารามิเตอร์ `n_estimators` | อ้างอิงจำนวนตัวเรียนจริงจาก `len(model.estimators_)` เสมอ ห้ามใช้ `model.n_estimators` เป็นตัวนับลูปโดยเด็ดขาด |
| 2 | **Base Estimator ไม่ใช่ Decision Tree** | ผู้ใช้อาจส่ง `AdaBoostClassifier(estimator=LogisticRegression())` เข้ามา ซึ่งไม่มีแอตทริบิวต์ `.tree_` | ตรวจสอบ `isinstance(est, BaseDecisionTree)` ก่อนเข้าถึง `.tree_` เสมอ หากไม่ใช่ต้นไม้แต่มี `.coef_` ให้กระจาย `delta_m` ตามสัดส่วน `|coef_ * x|` |
| 3 | **การถอด `SAMME.R` ใน Scikit-learn >= 1.4+** | Scikit-learn เวอร์ชันใหม่ใช้ `SAMME` เป็นค่าเริ่มต้น ซึ่งทำให้ `predict_proba` มีสเกลใกล้เคียงกันและมีความก้าวกระโดดเป็นขั้นชัดเจนกว่า `SAMME.R` | ใช้ `staged_predict_proba` ซึ่งรองรับทั้งโมเดลเก่าที่เซฟด้วย `SAMME.R` และโมเดลใหม่ที่ใช้ `SAMME` โดยอัตโนมัติ |
| 4 | **`AdaBoostRegressor` ใช้ Weighted Median** | ใน `AdaBoost.R2` ค่าทำนายสะสมคือ Weighted Median ของต้นไม้ย่อย ทำให้ในบาง Stage ค่า `staged_predict` อาจไม่ขยับเลย ($\Delta S_m = 0$) แล้วค่อยกระโดดทีเดียวใน Stage ถัดไป | คัดกรอง `top_influential_learners` จากค่า `abs(stage_deltas)` เพื่อให้แสดงเฉพาะ Stage ที่ทำให้ค่า Weighted Median ขยับจริง |
| 5 | **ตัวเรียนตัวแรกสุด (`m = 0`) ชี้ขาดคำตอบไปไกลมาก** | หาก Stump ตัวแรกแม่นยำมาก ค่า `staged_mat[:, 0]` จะกระโดดจาก `base_val` ทันที | กำหนด `prev_col = base_val` ก่อนหา `np.diff` เพื่อให้แรงดึงของตัวเรียนตัวที่ 1 ($\Delta S_1 = S_1 - S_0$) ถูกบันทึกครบถ้วน |
| 6 | **กรณีที่มีตัวเรียนเพียง 1 ตัว (`len(estimators_) == 1`)** | กราฟ Staged Curve มีเพียงจุดเดียว | ใน `AdaBoostHudPanel.tsx` ให้พล็อตจุดเริ่มต้นที่ Stage $0$ (`base_value`) เสมอ เพื่อให้กราฟมีอย่างน้อย 2 จุดเชื่อมกันเป็นเส้นตรง |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `AdaBoostAdapter` รองรับทั้ง `AdaBoostClassifier` และ `AdaBoostRegressor` รวมถึงทั้งกรณี Stump (`max_depth=1`), Deep Tree, และ Non-Tree Base Estimator
- [ ] สกัด `staged_trajectory` และ `top_influential_learners` ผ่าน `staged_predict_proba` / `staged_predict` โดยอ้างอิงความยาวจริงจาก `len(model.estimators_)`
- [ ] ฉาก 3 มิติแสดง **3D Sequential Cascade Ladder** บนลูกบอลข้อมูลที่แสดงขั้นการดึงขึ้น/ลงของ Weak Learners สำคัญอย่างถูกต้อง
- [ ] แผง HUD แสดงกราฟ **Staged Boosting Curve** พร้อมจุดตัด Decision Threshold และตารางน้ำหนัก `estimator_weights_` ($\alpha_m$)
- [ ] โหมด What-If Sandbox อัปเดตขั้นบันไดแสงและกราฟ Staged Curve ทันทีเมื่อลากลูกบอลข้ามเส้นแบ่งของ Stump
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุม `SAMME`, `AdaBoostRegressor`, Early-stopped AdaBoost, และ Deep Base Estimator
