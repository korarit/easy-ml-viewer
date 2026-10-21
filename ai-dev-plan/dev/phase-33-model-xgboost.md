# Phase 33: XGBoost Adapter (`.pkl`, `.json`, `.ubj`), Tree Dump Parser & 3D Boosted Leaf Flow

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter ระดับ Production สำหรับราชาแห่งข้อมูลตาราง **XGBoost (eXtreme Gradient Boosting)** รองรับทั้งไฟล์โมเดลที่บันทึกผ่าน Scikit-learn Wrapper และไฟล์ Native Booster มาตรฐานของ XGBoost พร้อมปลดล็อกขุมพลังระดับ C++ ในตัวของ XGBoost ได้แก่ **Exact TreeSHAP (`pred_contribs=True`)**, **Leaf Index Matrix (`pred_leaf=True`)**, และการแกะโครงสร้างต้นไม้จาก **`booster.get_dump(dump_format="json")`** เพื่อเรนเดอร์ภูมิประเทศขั้นบันไดความละเอียดสูงและเวกเตอร์แรงผลัก TreeSHAP ในโลก 3 มิติ

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `xgboost.XGBClassifier` (Scikit-learn API)
  - `xgboost.XGBRegressor` (Scikit-learn API)
  - `xgboost.XGBRFClassifier` และ `xgboost.XGBRFRegressor` (Random Forest mode ใน XGBoost)
  - `xgboost.Booster` (Native C++ Booster Object ที่โหลดจากไฟล์โดยตรง)
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.json` (XGBoost Portable JSON Format)
  - `.ubj` (Universal Binary JSON — ฟอร์แมตมาตรฐานใหม่ของ XGBoost 2.0+)
  - `.model`, `.xgb` (Legacy/Binary XGBoost Booster Format)
  - `.pkl`, `.pickle`, `.joblib` (Serialized Python Wrapper)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification** (`binary:logistic`, `binary:logitraw`, `binary:hinge`)
  - **Multiclass Classification** (`multi:softprob`, `multi:softmax`)
  - **Regression & Ranking** (`reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`, `reg:gamma`, `reg:tweedie`, `count:poisson`)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 สมการ Second-Order Taylor Expansion ของ XGBoost
XGBoost สร้างต้นไม้ $f_m \in \mathcal{F}$ ในรอบที่ $m$ เพื่อลด Objective Function ที่ใช้ทั้ง First-order Gradient ($g_i$) และ Second-order Hessian ($h_i$) พร้อมพจน์ Regularization ($\gamma T + \frac{1}{2}\lambda \sum w_j^2$):
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^{n} \left[ g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2, \qquad w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
ค่าทำนายในปริภูมิ Raw Margin คือผลรวมของน้ำหนักใบไม้ $w_{q(\mathbf{x})}$ บวกกับ `base_score` ($F_0$):
$$\hat{y}_{\text{margin}}(\mathbf{x}) = F_0 + \sum_{m=1}^{M} f_m(\mathbf{x})$$

### 2.2 การดึง Exact TreeSHAP ระดับ C++ (`pred_contribs=True`) และ Array Shapes
จุดเด่นที่สุดของ `xgboost.Booster` คือการฝังอัลกอริทึม **Exact TreeSHAP (Lundberg et al.)** ไว้ใน C++ Engine โดยตรง สามารถเรียกผ่าน `contribs = booster.predict(dmatrix, pred_contribs=True)` ได้ทันทีโดยไม่ต้องติดตั้งไลบรารี `shap` ภายนอก:
- **กรณี Regression และ Binary Classification (`ndim == 2`)**:
  - คืนอาเรย์ `float32` ขนาด **`(n_samples, n_features + 1)`**
  - คอลัมน์ที่ `0` ถึง `n_features - 1` คือค่า Exact TreeSHAP $\phi_1, \dots, \phi_d$ ของแต่ละฟีเจอร์
  - **คอลัมน์สุดท้าย (`[:, -1]`) คือค่า `bias` (`base_value` ในหน่วย Raw Margin)**!
  - คุณสมบัติ Exact Additivity: $\sum_{j=0}^{d-1} \text{contribs}[i, j] + \text{contribs}[i, -1] = \hat{y}_{\text{margin}}(\mathbf{x}_i)$
- **กรณี Multiclass Classification (`ndim == 3`)**:
  - คืนอาเรย์ `float32` ขนาด **`(n_samples, n_classes, n_features + 1)`**
  - สำหรับคลาสเป้าหมาย $c$: ค่าอิทธิพลฟีเจอร์คือ `contribs[:, c, :-1]` และค่า `bias` ของคลาส $c$ คือ `contribs[:, c, -1]`

### 2.3 การดึง Leaf Index (`pred_leaf=True`) และการแกะโครงสร้าง `get_dump(dump_format="json")`
1. **การดึงหมายเลขใบไม้ของทุกต้นไม้ (`pred_leaf=True`)**:
   - เรียก `leaf_indices = booster.predict(dmatrix, pred_leaf=True)`
   - สำหรับ Binary/Regression: คืนอาเรย์ขนาด `(n_samples, n_boost_rounds)`
   - สำหรับ Multiclass ($C$ คลาส) หรือ Parallel Trees (`num_parallel_tree > 1`): คืนอาเรย์ขนาด `(n_samples, n_boost_rounds, n_classes)` หรือ `(n_samples, n_boost_rounds * n_classes)` (สามารถส่ง `strict_shape=True` ใน XGBoost >= 1.6 เพื่อบังคับให้คืนเป็น 4D Tensor ขนาด `(n_samples, n_rounds, n_classes, n_parallel_trees)` ที่แน่นอนเสมอ!)
2. **การแกะ JSON Tree Dump (`booster.get_dump(dump_format="json", with_stats=True)`)**:
   - คืนค่า `list[str]` ที่แต่ละสตริงคือ JSON ของต้นไม้ต้นที่ $k$ มีโครงสร้าง Recursive Dict:
     `{"nodeid": 0, "depth": 0, "split": "f0", "split_condition": 42.5, "yes": 1, "no": 2, "missing": 1, "gain": 18.4, "cover": 250.0, "children": [...]}`
   - โหนดใบไม้จะมีรูปแบบ `{"nodeid": 3, "leaf": 0.1452, "cover": 85.0}`
   - เราแปลง Recursive JSON นี้เป็น Flat Parent-Lookup Dict เพียงครั้งเดียวตอนโหลดโมเดล เมื่อรู้ `leaf_id` จาก `pred_leaf=True` จึงสามารถย้อนรอยจาก Leaf กลับขึ้นไปหา Root เพื่อไฮไลต์ **Active Tree Branch Glow** ได้ในเวลา $\mathcal{O}(\text{depth})$!

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D XGBoost High-Definition Stepped Terrain (`SteppedVoxelTerrain.tsx`)**:
   - เรนเดอร์พื้นผิวขั้นบันไดความละเอียดสูงของ XGBoost พร้อมเส้นขอบรอยตัด (Split Lines) ที่กรองตามค่า `gain` สูงสุดจาก JSON Tree Dump
2. **3D Exact TreeSHAP Force Vectors (`TreeShapForceArrows.tsx`)**:
   - ที่ลูกบอลข้อมูลของแถวปัจจุบัน แสดงลูกศรเวกเตอร์ 3 มิติพุ่งออกจากลูกบอลตามแกน $X$ และแกน $Y$ โดยความยาวลูกศรแปรผันตรงกับค่า Exact TreeSHAP (`pred_contribs`) ของฟีเจอร์ทั้งสองในเฟรมนั้น
3. **Active Booster Tree Explorer & Missing Route Glow**:
   - แสดงแผนผังต้นไม้ต้นที่เลือก (`Tree #k`) พร้อมไฮไลต์เส้นทางจาก Root ลงมายัง `leaf_id` ที่ได้จาก `pred_leaf=True` และแสดงค่า `gain` (ความคมของการแบ่ง) กับ `cover` (Hessian sum) ของแต่ละโหนด

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`XgboostHudPanel.tsx`)
- **Exact TreeSHAP Waterfall Chart**: กราฟน้ำตกแสดงการบวกสะสมจาก `Bias (contribs[-1])` ผ่านฟีเจอร์ที่มีค่า $|\phi_j|$ สูงสุดไปจนถึงค่า `Raw Margin` และ `Final Probability / Prediction`
- **Booster Round Scrubber (`iteration_range = (0, m)`)**: เลื่อนดูผลการทำนายเมื่อจำกัดจำนวนรอบ Boosting ตั้งแต่รอบที่ $1$ ถึง $M$ (`best_iteration`)
- **Importance Metric Switcher (`gain` | `weight` | `cover` | `total_gain`)**: สลับดูความสำคัญของฟีเจอร์ทั้ง 4 มาตรวัดจาก `booster.get_score(importance_type=...)`

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - ลูกบอลข้อมูลวิ่งไปตามขั้นบันได 3 มิติ พร้อมลูกศร **TreeSHAP Force Vectors** และกราฟ **Exact TreeSHAP Waterfall** ใน HUD ที่เปลี่ยนค่าแบบเฟรมต่อเฟรมโดยไม่มีอาการหน่วง
2. เมื่ออยู่ในโหมด **What-If Sandbox**:
   - เมื่อผู้ใช้ลากลูกบอลในฉาก 3 มิติ Backend (หรือ Precomputed Grid) จะคำนวณ `predict(pred_contribs=True)` ในระดับ C++ ใช้เวลาไม่ถึง $1 \text{ ms}$ ทำให้เห็นทั้งการกระโดดข้ามขั้นบันไดและการเปลี่ยนขนาดของแท่ง TreeSHAP Waterfall แบบ Real-time!

### 4.2 ASCII Wireframe ของหน้าจอ XGBoost HUD & Exact TreeSHAP Waterfall
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: XGBClassifier (trees=250, depth=6, fmt=.ubj)    [ Tree #001 | Metric:Gain]|
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [EXACT C++ TREESHAP (t=215)] |
|          3D XGBOOST TERRAIN & TREESHAP FORCE VECTORS              | Bias (Base Margin) : -0.350  |
|                                                                   |  f2 (Income=85k)   : +1.420  |
|                        +---------------+                          |  f0 (Age=44)       : +0.680  |
|                        | Leaf #19      |                          |  f5 (Debt=32k)     : -0.490  |
|                +-------+  P(y=1)=0.78  +-------+                  |  Other 12 features : +0.020  |
|                |       |  ==========>  |       |                  | ---------------------------- |
|                |       |  (o) [SHAP X] |       |                  | Raw Margin Sum     : +1.280  |
|                |       |   |           |       |                  | Sigmoid P(y=1)     :  0.7824 |
|        +-------+-------+   v [SHAP Y]  +-------+-------+          +------------------------------+
|        |               +---------------+               |          | [TREE #001 JSON DUMP PATH]   |
|        |   XGBoost Second-Order Regularized Terraces   |          | Node 0: [f2 < 68.5] (no->2)  |
|        +-----------------------------------------------+          |   Gain: 42.8 | Cover: 412.0  |
|                                                                   | Node 2: [f0 < 39.0] (no->6)  |
|                                                                   | Leaf 6: weight = +0.1842     |
+-------------------------------------------------------------------+------------------------------+
| [ACTIVE LEAVES ACROSS TREES] T#0: Leaf #6 (+0.18) | T#1: Leaf #11 (+0.14) | T#2: Leaf #4 (-0.06) |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0215 / 1000  [=====(*)----------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/boosting/xgboost_adapter.py` (`XGBoostAdapter`)
- `frontend/src/domains/stage3d/actors/TreeShapForceArrows.tsx`
- `frontend/src/domains/hud/components/XgboostHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `XGBoostAdapter`
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import xgboost as xgb
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

MAX_EXPORTED_TREES = 100
MAX_TRAJECTORY_POINTS = 50

class XGBoostAdapter(BaseModelAdapter):
    """Adapter สำหรับ XGBClassifier, XGBRegressor และ Native xgboost.Booster (.json, .ubj, .model, .pkl)."""

    def __init__(self, model_or_booster: Any, feature_names: list[str] | None = None, target_class_idx: int = 1) -> None:
        if isinstance(model_or_booster, (str, Path)):
            booster = xgb.Booster()
            booster.load_model(str(model_or_booster))
            self.wrapper = None
            self.booster: xgb.Booster = booster
        elif hasattr(model_or_booster, "get_booster"):
            self.wrapper = model_or_booster
            self.booster = model_or_booster.get_booster()
        elif isinstance(model_or_booster, xgb.Booster):
            self.wrapper = None
            self.booster = model_or_booster
        else:
            raise TypeError(f"Unsupported XGBoost model type: {type(model_or_booster)}")

        resolved_names = feature_names or self.booster.feature_names or []
        super().__init__(model_or_booster, list(resolved_names))
        self.target_class_idx = target_class_idx
        # บังคับ nthread=2 เพื่อไม่ให้แย่ง CPU กับ Worker อื่นใน ComputePool
        self.booster.set_param({"nthread": 2})
        self.n_boost_rounds: int = self.booster.num_boosted_rounds()

    def parse_trees_topology(self, max_trees: int = MAX_EXPORTED_TREES) -> list[dict[str, Any]]:
        """แกะโครงสร้างต้นไม้จาก get_dump(dump_format='json', with_stats=True)."""
        raw_dumps = self.booster.get_dump(dump_format="json", with_stats=True)[:max_trees]
        return [json.loads(dump_str) for dump_str in raw_dumps]

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        """สร้าง DMatrix เพียงครั้งเดียวต่อ Chunk แล้วรันทั้ง pred_contribs, pred_leaf และ staged trajectory."""
        X_f32 = np.ascontiguousarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        dmat = xgb.DMatrix(X_f32, feature_names=self.feature_names if self.feature_names else None)

        # 1. Exact C++ TreeSHAP (pred_contribs=True)
        raw_contribs = self.booster.predict(dmat, pred_contribs=True, strict_shape=True)
        # strict_shape=True คืนขนาด (n_samples, n_groups, n_features + 1) เสมอ
        if raw_contribs.ndim == 3:
            grp_idx = min(self.target_class_idx, raw_contribs.shape[1] - 1) if raw_contribs.shape[1] > 1 else 0
            contribs = raw_contribs[:, grp_idx, :-1].astype(np.float32)
            base_val = float(raw_contribs[0, grp_idx, -1])
        else:
            contribs = raw_contribs[:, :-1].astype(np.float32)
            base_val = float(raw_contribs[0, -1])

        # 2. Leaf Index Extraction (pred_leaf=True)
        raw_leaves = self.booster.predict(dmat, pred_leaf=True, strict_shape=True)
        # strict_shape=True คืนขนาด (n_samples, n_rounds, n_groups, n_parallel_trees)
        if raw_leaves.ndim == 4:
            grp_idx = min(self.target_class_idx, raw_leaves.shape[2] - 1) if raw_leaves.shape[2] > 1 else 0
            leaf_matrix = raw_leaves[:, :, grp_idx, 0].astype(np.int32)
        else:
            leaf_matrix = raw_leaves.astype(np.int32)

        # 3. Staged Progression Trajectory via iteration_range
        n_rounds = max(1, self.n_boost_rounds)
        step_rounds = np.unique(np.linspace(1, n_rounds, min(n_rounds, MAX_TRAJECTORY_POINTS), dtype=int))
        staged_cols = []
        for r in step_rounds:
            pred_r = self.booster.predict(dmat, iteration_range=(0, int(r)))
            if pred_r.ndim == 2:
                c_idx = min(self.target_class_idx, pred_r.shape[1] - 1)
                staged_cols.append(pred_r[:, c_idx].astype(np.float32))
            else:
                staged_cols.append(pred_r.astype(np.float32))
        staged_trajectory = np.column_stack(staged_cols)

        k_export = min(leaf_matrix.shape[1], MAX_EXPORTED_TREES)
        return InternalFlowBatch(
            base_value=base_val,
            feature_contributions=contribs,
            extra_telemetry={
                "model_family": "xgboost",
                "n_boost_rounds": n_rounds,
                "sampled_round_indices": (step_rounds - 1).tolist(),
                "staged_trajectory": np.round(staged_trajectory, 4).tolist(),
                "leaf_indices_sample": leaf_matrix[:, :k_export].tolist(),
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 215,
  "prediction": 0.7824,
  "base_value": -0.350,
  "feature_contributions": [0.680, 0.0, 1.420, 0.0, 0.0, -0.490],
  "telemetry": {
    "model_family": "xgboost",
    "n_boost_rounds": 250,
    "sampled_round_indices": [0, 10, 50, 120, 249],
    "staged_trajectory": [0.46, 0.58, 0.69, 0.75, 0.7824],
    "leaf_indices_sample": [6, 11, 4, 19, 8]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การสร้าง `xgb.DMatrix` เพียงครั้งเดียวต่อ Chunk (Zero Redundant Buffer Copy)**:
   - การสร้าง `xgb.DMatrix` ต้องแปลง NumPy Array เข้าสู่โครงสร้าง CSR/Page ย่อยในหน่วยความจำ C++ ของ XGBoost ดังนั้นใน `trace_internal_flow` ต้องสร้าง `dmat = xgb.DMatrix(X_f32, ...)` เพียง **1 ครั้งต่อ Chunk** แล้วส่ง `dmat` ตัวเดิมเข้าทั้ง `predict()`, `predict(pred_contribs=True)`, และ `predict(pred_leaf=True)`
2. **การควบคุมจำนวน Threads ใน C++ OpenMP (`nthread = 2`)**:
   - โดยค่าเริ่มต้น XGBoost จะจอง OpenMP Threads เท่ากับจำนวนคอร์ทั้งหมดของ CPU (`nthread = os.cpu_count()`) เมื่อรันอยู่ใน Worker ของ `ComputePool` จะเกิดการแย่งคอร์กันเอง (Thread Thrashing) จึงต้องสั่ง `self.booster.set_param({"nthread": 2})` ทันทีที่โหลดโมเดล
3. **การป้องกันการเรียก `get_dump` ซ้ำๆ**:
   - `booster.get_dump(dump_format="json", with_stats=True)` แปลงต้นไม้ C++ เป็น JSON String ซึ่งหากเรียกทุก Chunk จะช้ามาก ต้องเรียกเพียง **1 ครั้งเดียว** ตอนสร้าง Header ของไฟล์ `.mlv` และจำกัดไว้ที่ `MAX_EXPORTED_TREES = 100` ต้นแรก

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **Multiclass `pred_contribs` Shape (2D vs 3D)** | ใน Binary/Regression อาเรย์ `pred_contribs` มีขนาด 2 มิติ `(n_samples, n_features + 1)` แต่ใน Multiclass มีขนาด 3 มิติ `(n_samples, n_classes, n_features + 1)` | ใช้พารามิเตอร์ `strict_shape=True` ใน `booster.predict(..., pred_contribs=True, strict_shape=True)` ซึ่งบังคับให้คืนค่า 3 มิติ `(n_samples, n_groups, n_features + 1)` เสมอ ทำให้ไม่ต้องเดา Shape! |
| 2 | **Native Booster ไม่มี `feature_names` (แสดงเป็น `f0, f1, ...`)** | เมื่อเทรนด้วย NumPy Array หรือโหลดจากไฟล์ `.model`/`.ubj` บางครั้ง `booster.feature_names` จะเป็น `None` และใน JSON Dump จะเขียนว่า `"split": "f0"` | หาก `booster.feature_names is None` ให้แมป `"f0" -> feature_names[0]` จาก Dataset ที่ผู้ใช้ส่งเข้ามา และส่ง `feature_names=None` ตอนสร้าง `DMatrix` เพื่อไม่ให้ XGBoost ฟ้อง `ValueError: feature_names mismatch` |
| 3 | **Early Stopping (`best_iteration`) ในโมเดลที่เซฟไว้** | โมเดลที่เทรนพร้อม `early_stopping_rounds` อาจมีต้นไม้ต่อท้ายเกิน `best_iteration` อยู่ใน Booster | ตรวจสอบ `getattr(self.wrapper, "best_iteration", None)` หรือ `booster.attr("best_iteration")` แล้วใช้ `iteration_range=(0, best_iteration + 1)` ในการทำนายเสมอ |
| 4 | **`gblinear` Booster แทนที่จะเป็น `gbtree` / `dart`** | ผู้ใช้อาจตั้ง `booster="gblinear"` ใน XGBoost ซึ่งไม่มีต้นไม้และเรียก `pred_leaf=True` ไม่ได้ | ตรวจสอบ `json.loads(booster.save_config())["learner"]["gradient_booster"]["name"]` หากเป็น `"gblinear"` ให้ใช้เฉพาะ `pred_contribs=True` และข้าม `pred_leaf=True` |
| 5 | **Categorical Features (`enable_categorical=True`) ใน XGBoost** | โหนดที่แบ่งด้วย Categorical Feature ใน JSON Dump จะมีคีย์ `"split_condition": [0, 3, 5]` เป็น `list` ของหมวดหมู่แทนที่จะเป็นตัวเลข `float` ตัวเดียว | ใน JSON Tree Parser ต้องตรวจสอบ `isinstance(node.get("split_condition"), list)` เพื่อแสดงเป็นเงื่อนไข `in {cat_0, cat_3}` แทน `< threshold` |
| 6 | **โมเดลชนิด `DART` (`booster="dart"`)** | ใน DART มีการสุ่มตัดต้นไม้และถ่วงน้ำหนักซ้ำ ทำให้การรัน `predict(iteration_range=...)` มีความซับซ้อนกว่า `gbtree` | ลดจำนวนจุด `MAX_TRAJECTORY_POINTS` จาก 50 เหลือ 20 จุดเมื่อตรวจพบว่า `booster_type == "dart"` เพื่อรักษาความเร็ว |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `XGBoostAdapter` โหลดและทำงานได้กับทั้ง `XGBClassifier`, `XGBRegressor`, และ `xgb.Booster` จากไฟล์ `.pkl`, `.joblib`, `.json`, `.ubj`, และ `.model`
- [ ] ใช้ `DMatrix` เดียวต่อ Chunk ในการดึงทั้ง `pred_contribs=True` (Exact C++ TreeSHAP) และ `pred_leaf=True` พร้อมเปิด `strict_shape=True` รองรับทั้ง Binary, Regression, และ Multiclass
- [ ] แกะโครงสร้างต้นไม้จาก `booster.get_dump(dump_format="json", with_stats=True)` พร้อมดึงค่า `gain` และ `cover` รายโหนด
- [ ] ฉาก 3 มิติแสดงพื้นผิวขั้นบันไดของ XGBoost, ลูกศร **TreeSHAP Force Vectors**, และการไฮไลต์กิ่งไม้ที่นำไปสู่ `leaf_id` ของแถวปัจจุบัน
- [ ] แผง HUD แสดงกราฟน้ำตก **Exact TreeSHAP Waterfall** ที่ผลรวม `Bias + sum(SHAP)` เท่ากับ Raw Margin ของ XGBoost (`atol=1e-4`)
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุมไฟล์ `.json`/`.ubj`, Binary, Multiclass, Regression, และกรณี `booster.feature_names is None`
