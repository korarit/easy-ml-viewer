# Phase 31: Scikit-Learn Gradient Boosting (GBM) Adapter & Staged Residual Trajectory

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter สำหรับโมเดล **Scikit-learn Gradient Boosting Machines (`GradientBoostingClassifier` และ `GradientBoostingRegressor`)** เพื่อจำลองกระบวนการไต่ระดับลดความคลาดเคลื่อนทีละขั้น (Additive Functional Gradient Descent on Residuals) ออกมาเป็น **3D Residual Spiral Ladder (บันไดเกลียวสะสมค่า Residual ในแนวดิ่ง)** เหนือลูกบอลข้อมูลในฉาก 3 มิติ พร้อมแกะรอย `staged_decision_function` และโครงสร้างต้นไม้ `DecisionTreeRegressor` ภายในทุก Stage

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `sklearn.ensemble.GradientBoostingClassifier` (Loss: `log_loss`, `exponential`)
  - `sklearn.ensemble.GradientBoostingRegressor` (Loss: `squared_error`, `absolute_error`, `huber`, `quantile`)
  - โมเดล GBM ที่อยู่ภายใน `Pipeline` หรือ `GridSearchCV`
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.pkl`, `.pickle`, `.joblib`
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: ติดตามการบวกสะสมค่า Log-odds ($F_m(\mathbf{x})$) แล้วแปลงผ่านฟังก์ชัน Sigmoid $\sigma(F_m(\mathbf{x}))$ เป็นความสูงความน่าจะเป็น
  - **Multiclass Classification**: รองรับ $C \ge 3$ คลาส ซึ่งใช้ต้นไม้ $C$ ต้นต่อ 1 Stage (`estimators_.shape == (n_estimators_, C)`) แล้วแปลงผ่าน Softmax
  - **Regression**: ติดตามการบวกสะสมค่า Residual ($\eta \cdot h_m(\mathbf{x})$) จากค่าเริ่มต้น `init_` ไปสู่ค่าทำนายสุดท้าย

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 สมการ Additive Gradient Boosting และการสะสม Residual
Gradient Boosting เริ่มต้นด้วยตัวทำนายตั้งต้น $F_0(\mathbf{x})$ จาก `model.init_` (เช่น ค่าเฉลี่ย $\bar{y}$ สำหรับ Regression หรือ $\log\left(\frac{p}{1-p}\right)$ สำหรับ Binary Classification) แล้วบวกเพิ่มด้วยผลลัพธ์ของต้นไม้ `DecisionTreeRegressor` ที่ถูกย่อขนาดด้วยอัตราการเรียนรู้ $\eta = \text{model.learning\_rate}$:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{t=1}^{m} \eta \cdot h_t(\mathbf{x})$$
- **ข้อเท็จจริงสำคัญของ `model.estimators_`**:
  ไม่ว่าจะเป็น `GradientBoostingRegressor` หรือ `GradientBoostingClassifier` ต้นไม้ย่อยทุกตัวใน `model.estimators_[m, c]` **เป็นคลาส `DecisionTreeRegressor` เสมอ!** เพราะมันทำหน้าที่ทำนายค่า Negative Gradient (Pseudo-residuals) ในปริภูมิ Raw Margin / Log-odds:
  - **Regression & Binary Classification**: `model.estimators_` มี Shape `(n_estimators_, 1)`
  - **Multiclass Classification ($C$ คลาส)**: `model.estimators_` มี Shape `(n_estimators_, C)`

### 2.2 การดึง Leaf Index (`model.apply`) และ `staged_decision_function`
- **การดึง Leaf Index**:
  เรียก `leaf_tensor = model.apply(X_chunk)`:
  - คืนค่า `np.ndarray` ขนาด `(n_samples, n_estimators_, 1)` สำหรับ Binary/Regression
  - คืนค่า `np.ndarray` ขนาด `(n_samples, n_estimators_, n_classes_)` สำหรับ Multiclass
- **การดึงเส้นทางการไต่ระดับทีละ Stage**:
  - สำหรับ `GradientBoostingClassifier`: ใช้ `model.staged_predict_proba(X_chunk)` เพื่อดึงความน่าจะเป็นสะสม หรือ `model.staged_decision_function(X_chunk)` เพื่อดึงค่า Log-odds สะสม $F_m(\mathbf{x})$
  - สำหรับ `GradientBoostingRegressor`: ใช้ `model.staged_predict(X_chunk)` เพื่อดึงค่าทำนายสะสม $F_m(\mathbf{x})$

### 2.3 การคำนวณ Exact Tree Path Feature Contributions (Scaled by Learning Rate $\eta$)
เนื่องจากแต่ละต้นไม้ $h_{m, c}$ คือ `DecisionTreeRegressor` ที่ถูกคูณด้วย $\eta = \text{model.learning\_rate}$ ค่าอิทธิพลของฟีเจอร์ $j$ ในต้นไม้ต้นที่ $m$ จึงคำนวณได้จากผลต่างค่าคาดหวังระหว่างโหนดพ่อ $u \to \text{ลูก } v$ คูณด้วย $\eta$:
$$\Delta_{m, u \to v} = \eta \cdot \left(E_{m, v} - E_{m, u}\right), \qquad \Phi_j(\mathbf{x}) = \sum_{m=1}^{M} \sum_{(u \to v) \in \text{path}_m(\mathbf{x}), f_u = j} \Delta_{m, u \to v}$$
สมการนี้รับประกันคุณสมบัติ Exact Additivity ในปริภูมิ Raw Score / Margin ทุกประการ:
$$F_0 + \sum_{j=1}^{d} \Phi_j(\mathbf{x}) = F_M(\mathbf{x})$$
- **Array Shape ของ Contributions**:
  - **Binary / Regression**: `(n_samples, n_features)` (ในหน่วย Raw Margin สำหรับ Classifier หรือหน่วย Target สำหรับ Regressor)
  - **Multiclass**: `(n_samples, n_classes, n_features)`

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Residual Spiral Ladder (`ResidualSpiralLadder.tsx`)**:
   - เหนือหรือใต้ลูกบอลข้อมูลของแถวปัจจุบัน แสดง **เส้นเกลียวสปริงเรืองแสงแนวดิ่ง (Residual Spiral Ladder)** ที่เริ่มต้นจากระดับแผ่นฐาน `Base Score (F_0)` แล้วหมุนไต่ระดับขึ้นหรือลงทีละขั้นตามค่า $\eta \cdot h_m(\mathbf{x}_t)$ จนถึงจุดที่ลูกบอลลอยอยู่
   - ขั้นเกลียวที่ดันค่าขึ้น ($\eta h_m > 0$) จะเรืองแสงสีฟ้านีออน ส่วนขั้นเกลียวที่กดค่าลง ($\eta h_m < 0$) จะเรืองแสงสีส้มแดง
2. **Additive Boosted Voxel Terrain (`SteppedVoxelTerrain.tsx`)**:
   - พื้นผิวขั้นบันได 3 มิติที่มีความซับซ้อนจากการบวกซ้อนทับกันของต้นไม้ตื้นๆ (ปกติ `max_depth=3`) จำนวนหลายร้อยต้น
   - รองรับการเลื่อน **Stage Progression Slider** เพื่อดูภูมิประเทศ 3D ค่อยๆ ก่อตัวขึ้นจากแผ่นราบเรียบที่ Stage $0$ กลายเป็นภูเขาขั้นบันไดที่ละเอียดขึ้นเรื่อยๆ เมื่อ $m \to M$!
3. **Active Stage Tree Branch Overlay**:
   - เมื่อเลือกดู Stage ที่ $m$ ใน HUD จะแสดงผังกิ่งไม้ขนาดกะทัดรัด (Depth 3) ของ `estimators_[m, c]` พร้อมไฮไลต์กิ่งที่แถวปัจจุบันวิ่งผ่าน

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`SklearnGbmHudPanel.tsx`)
- **Stage Evolution Scrubber (`Stage m / M`)**: แถบเลื่อนดูวิวัฒนาการของการ Boosting ตั้งแต่ต้นไม้ต้นที่ $1$ ถึง $M$ (`n_estimators_`)
- **Staged Residual & Loss Convergence Curve**: กราฟแสดงการไต่ระดับของ $F_m(\mathbf{x}_t)$ และความน่าจะเป็น $P_m(\mathbf{x}_t)$ ราย Stage พร้อมจุดไฮไลต์ Stage ที่สร้างการเปลี่ยนแปลง (Residual Jump) ใหญ่ที่สุด
- **Subsample / OOB Improvement Chart**: หากโมเดลเทรนด้วย `subsample < 1.0` จะมี `model.oob_improvement_` ให้พล็อตกราฟแสดงว่าหลังจากต้นไม้ต้นที่เท่าไหร่ที่โมเดลเริ่ม Overfit

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - ลูกบอลข้อมูลวิ่งไปตามภูมิประเทศขั้นบันได พร้อมกับเกลียวสปริง **Residual Spiral Ladder** ที่ยืด-หดแสดงประวัติการก่อร่างสร้างคำตอบจาก `Base Score` มายังค่าทำนายปัจจุบัน
2. เมื่อผู้ใช้ทดลองเลื่อน **Stage Evolution Scrubber** ใน HUD จาก Stage $1 \to 100$:
   - พื้นผิว 3 มิติและลูกบอลข้อมูลจะขยับตามการบวกสะสมของต้นไม้ทีละต้น ให้เห็นกับตาว่า Gradient Boosting ค่อยๆ "แกะสลัก" พื้นผิวจากแผ่นราบให้กลายเป็นเนินเขาตามกลุ่มข้อมูลอย่างไร
3. เมื่ออยู่ในโหมด **What-If Sandbox**:
   - ทันทีที่ลากลูกบอลข้ามขอบเขตการแบ่งของต้นไม้ใน Stage สำคัญ เกลียวสปริงขั้นนั้นจะเปลี่ยนความสูงทันที พร้อมอัปเดตกราฟ Staged Curve แบบ Real-time

### 4.2 ASCII Wireframe ของหน้าจอ Scikit-Learn GBM HUD & Residual Spiral
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: GradientBoostingClassifier (M=100, lr=0.1, depth=3)   [ Stage: 100/100 ] |
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [STAGED RESIDUAL CURVE (t=5)]|
|          3D BOOSTED TERRAIN & RESIDUAL SPIRAL LADDER              | Prob |        .----* (0.89)  |
|                                                                   | 0.8  |     .-'               |
|                        ((o))  <-- Final Prob: 0.89 (LogOdds=+2.1) | 0.5  |---/------------------ |
|                         /                                         | 0.2  | .' (Init Prior: 0.35) |
|                        o  <-- Stage #42: +0.35 log-odds (Leaf #6) |      +---------------------- |
|                         \                                         |      0     25     50    100  |
|                          o <-- Stage #15: -0.18 log-odds (Leaf #3)+------------------------------+
|                         /                                         | [TOP RESIDUAL STAGES]        |
|                        o  <-- Stage #03: +0.62 log-odds (Leaf #7) | Stage #01: +0.74 (Leaf #4)   |
|                       /                                           | Stage #03: +0.62 (Leaf #7)   |
|                      o  <-- Stage #01: +0.74 log-odds (Leaf #4)   | Stage #42: +0.35 (Leaf #6)   |
|      +---------------+---------------+                            | Stage #15: -0.18 (Leaf #3)   |
|      |  Init Base Score (F_0 = -0.62) |                            | Learning Rate (eta): 0.100   |
+-------------------------------------------------------------------+------------------------------+
| [EXACT PATH DELTAS (Log-Odds)] Init: -0.62 | f1: +1.85 | f0: +1.12 | f3: -0.25 ===> Sum: +2.10   |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0005 / 1000  [(*)--------------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/boosting/sklearn_gbm.py` (`SklearnGBMAdapter`)
- `frontend/src/domains/stage3d/actors/ResidualSpiralLadder.tsx`
- `frontend/src/domains/hud/components/SklearnGbmHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `SklearnGBMAdapter`
```python
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

MAX_TRAJECTORY_STAGES = 50
MAX_TRACED_STAGES = 100

class SklearnGBMAdapter(BaseModelAdapter):
    """Adapter สำหรับ GradientBoostingClassifier และ GradientBoostingRegressor."""

    def __init__(self, model: Any, feature_names: list[str], target_class_idx: int = 1) -> None:
        super().__init__(model, feature_names)
        self.estimators_: np.ndarray = model.estimators_  # shape: (n_estimators_, n_trees_per_stage)
        self.n_stages: int = self.estimators_.shape[0]
        self.n_cols: int = self.estimators_.shape[1]
        self.learning_rate: float = float(model.learning_rate)
        self.is_classifier: bool = isinstance(model, GradientBoostingClassifier)
        self.target_class_idx: int = target_class_idx
        self.tree_col: int = 0 if self.n_cols == 1 else min(target_class_idx, self.n_cols - 1)

    def _get_base_raw_score(self, X_sample: np.ndarray) -> float:
        """ดึงค่าเริ่มต้น F_0 (Raw Margin / Init Score) จาก model._raw_predict_init."""
        if hasattr(self.model, "_raw_predict_init"):
            raw_init = self.model._raw_predict_init(X_sample[:1])
            return float(raw_init[0, self.tree_col])
        return 0.0

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        X_f32 = np.asarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        base_raw = self._get_base_raw_score(X_f32)

        # ดึง Leaf Index ของทุก Stage: shape (n_samples, n_stages, n_cols)
        leaf_tensor = self.model.apply(X_f32)
        if leaf_tensor.ndim == 2:
            leaf_matrix = leaf_tensor
        else:
            leaf_matrix = leaf_tensor[:, :, self.tree_col]

        # ดึง Staged Predictions (Probability สำหรับ Classifier, Value สำหรับ Regressor)
        if self.is_classifier:
            staged_preds = np.column_stack([
                proba[:, min(self.target_class_idx, proba.shape[1] - 1)].astype(np.float32)
                for proba in self.model.staged_predict_proba(X_f32)
            ])
        else:
            staged_preds = np.column_stack([
                pred.astype(np.float32) for pred in self.model.staged_predict(X_f32)
            ])

        # คำนวณ Raw Residual Jump ของแต่ละ Stage และ Exact Feature Contributions
        contribs = np.zeros((n_samples, n_features), dtype=np.float32)
        stage_raw_deltas = np.zeros((n_samples, self.n_stages), dtype=np.float32)
        traced_stages = min(self.n_stages, MAX_TRACED_STAGES)

        for m in range(self.n_stages):
            reg_tree = self.estimators_[m, self.tree_col]
            node_vals = reg_tree.tree_.value[:, 0, 0].astype(np.float32)
            leaves_m = leaf_matrix[:, m]
            stage_raw_deltas[:, m] = self.learning_rate * node_vals[leaves_m]

            if m < traced_stages:
                csr = reg_tree.decision_path(X_f32)
                feats = reg_tree.tree_.feature
                for i in range(n_samples):
                    path = csr.indices[csr.indptr[i] : csr.indptr[i + 1]]
                    for s in range(len(path) - 1):
                        u, v = path[s], path[s + 1]
                        f_idx = feats[u]
                        if f_idx >= 0:
                            contribs[i, f_idx] += self.learning_rate * (node_vals[v] - node_vals[u])

        step_idx = np.unique(np.linspace(0, self.n_stages - 1, min(self.n_stages, MAX_TRAJECTORY_STAGES), dtype=int))
        k_export = min(self.n_stages, MAX_TRACED_STAGES)

        return InternalFlowBatch(
            base_value=base_raw,
            feature_contributions=contribs,
            extra_telemetry={
                "model_family": "sklearn_gbm",
                "learning_rate": self.learning_rate,
                "n_stages_total": self.n_stages,
                "sampled_stage_indices": step_idx.tolist(),
                "staged_predictions": np.round(staged_preds[:, step_idx], 4).tolist(),
                "stage_raw_deltas_sample": np.round(stage_raw_deltas[:, :k_export], 4).tolist(),
                "leaf_indices_per_stage": leaf_matrix[:, :k_export].astype(np.int32).tolist(),
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 5,
  "prediction": 0.891,
  "base_value": -0.619,
  "feature_contributions": [1.12, 1.85, 0.0, -0.25],
  "telemetry": {
    "model_family": "sklearn_gbm",
    "learning_rate": 0.1,
    "n_stages_total": 100,
    "sampled_stage_indices": [0, 2, 14, 41, 99],
    "staged_predictions": [0.47, 0.62, 0.58, 0.81, 0.891],
    "stage_raw_deltas_sample": [0.74, 0.12, 0.62, -0.05],
    "leaf_indices_per_stage": [4, 2, 7, 1]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การดึง `stage_raw_deltas` ผ่าน `model.apply(X_f32)` โดยไม่ต้องรัน `predict` รายต้นไม้**:
   - เมื่อได้ `leaf_matrix = model.apply(X_f32)` แล้ว ค่าทำนายของต้นไม้ต้นที่ $m$ คือ `reg_tree.tree_.value[leaf_matrix[:, m], 0, 0]` คูณด้วย `learning_rate` ทันที! ไม่ต้องเรียก `reg_tree.predict(X_f32)` ซ้ำ ทำให้ประหยัดเวลา CPU ลงกว่า $70\%$
2. **การจำกัดจำนวน Stage ที่แกะ `decision_path` (`MAX_TRACED_STAGES = 100`)**:
   - ใน Gradient Boosting ต้นไม้ช่วงต้นๆ ($m = 1 \dots 100$) คือตัวสร้างโครงสร้างหลักของพื้นผิว ในขณะที่ต้นไม้ช่วงท้ายๆ ($m > 100$) มีค่า Residual เล็กมากเนื่องจากถูกกดด้วย Shrinkage การจำกัดการเดิน `decision_path` ไว้ที่ 100 Stage แรกช่วยคุมเวลา Pre-computation ให้คงที่แม้โมเดลจะมี 1,000 ต้น
3. **การเรนเดอร์ `ResidualSpiralLadder.tsx` ด้วย `THREE.BufferGeometry` เส้นเดียว**:
   - คำนวณพิกัดจุดยอดของเกลียวสปริงใน `useMemo` จาก `stage_raw_deltas_sample` และวาดด้วย `<Line>` ของ `@react-three/drei` เพียง 1 Draw Call ต่อเฟรม

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **`model.estimators_` เป็น `DecisionTreeRegressor` เสมอแม้ใน Classifier** | หากโค้ดเข้าใจผิดคิดว่า `GradientBoostingClassifier` เก็บ `DecisionTreeClassifier` แล้วพยายามหารผลรวมคลาสใน `tree_.value` จะได้ค่าเป็น `1.0` ตลอดเวลา! | อ่านค่า `reg_tree.tree_.value[:, 0, 0]` ตรงๆ เสมอ และคูณด้วย `model.learning_rate` ทุกครั้ง |
| 2 | **Binary vs Multiclass Shape ของ `estimators_` และ `apply()`** | ใน Binary Classification `estimators_.shape` คือ `(M, 1)` (มีต้นไม้เดียวต่อ Stage) แต่ใน Multiclass $C$ คลาส จะมีขนาด `(M, C)` | ตรวจสอบ `self.estimators_.shape[1]` (`self.n_cols`): ถ้าเท่ากับ `1` ให้ใช้ดัชนีคอลัมน์ `0` เสมอ ห้ามใช้ `target_class_idx = 1` ไป Index คอลัมน์ของ Binary GBM เพราะจะเกิด `IndexError: index 1 is out of bounds for axis 1 with size 1`! |
| 3 | **Early Stopping (`n_iter_no_change` เปิดใช้งาน)** | จำนวนต้นไม้ที่เทรนจริง `model.n_estimators_` จะน้อยกว่าพารามิเตอร์ `model.n_estimators` ที่ตั้งไว้ | ใช้ `self.estimators_.shape[0]` หรือ `model.n_estimators_` เป็นจำนวน Stage จริงเสมอ |
| 4 | **ผู้ใช้ตั้ง `init='zero'` หรือ Custom Estimator ใน `init`** | `model.init_` อาจเป็นสตริง `'zero'`, `DummyClassifier`, `DummyRegressor` หรือ Estimator อื่น | เรียกใช้เมธอดภายใน `model._raw_predict_init(X_f32[:1])` ของ Scikit-learn ซึ่งรองรับ `init_` ทุกประเภทอย่างถูกต้อง |
| 5 | **Binary Classification สำหรับ `target_class_idx == 0`** | ใน Binary GBM ต้นไม้คอลัมน์ `0` ทำนาย Log-odds ของคลาส `1` เสมอ ($\log(P(y=1)/P(y=0))$) | หากผู้ใช้เลือกดูมุมมองของคลาส `0` ให้กลับเครื่องหมาย `base_raw` และ `contribs` เป็นค่าลบ (`-1.0 * val`) |
| 6 | **Loss แบบ `exponential` ใน `GradientBoostingClassifier`** | ค่า Raw Score จะเป็น $\frac{1}{2} \log\left(\frac{P(y=1)}{P(y=0)}\right)$ แทนที่จะเป็น Log-odds ปกติ | ใช้ `model.staged_predict_proba(X_f32)` ในการดึงความน่าจะเป็นราย Stage เพื่อให้ Scikit-learn จัดการ Link Function ของแต่ละ Loss ให้โดยอัตโนมัติ |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `SklearnGBMAdapter` รองรับทั้ง `GradientBoostingClassifier` และ `GradientBoostingRegressor` โดยจัดการความแตกต่างของ `estimators_.shape` ระหว่าง Binary `(M, 1)` และ Multiclass `(M, C)` ได้ถูกต้อง
- [ ] คำนวณ Exact Path Contributions ที่คูณด้วย `model.learning_rate` พร้อมดึง `base_value` จาก `_raw_predict_init`
- [ ] ฉาก 3 มิติแสดง **3D Residual Spiral Ladder** ที่หมุนไต่ระดับจาก `Base Score` ไปยังความสูงสุดท้ายของลูกบอลข้อมูล
- [ ] แผง HUD มี **Stage Evolution Scrubber** และกราฟ **Staged Residual Curve** ที่แสดงการลู่เข้าของคำตอบทีละ Stage
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุม Binary (`n_cols=1`), Multiclass (`n_cols=C`), Regression, `init='zero'`, และ Early Stopping
