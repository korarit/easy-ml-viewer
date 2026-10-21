# Phase 32: HistGradientBoosting Adapter & 3D Quantized Histogram Bin Terraces

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter สำหรับโมเดล Gradient Boosting ยุคใหม่ความเร็วสูงของ Scikit-learn คือ **`HistGradientBoostingClassifier`** และ **`HistGradientBoostingRegressor`** (ซึ่งได้รับแรงบันดาลใจจาก LightGBM) โดยสกัดขอบเขตถังข้อมูล 255 ถัง (`_bin_mapper.bin_thresholds_`) และโครงสร้างต้นไม้ระดับ Cython (`_predictors` / `TreePredictor.nodes`) ออกมาแสดงผลเป็น **3D Quantized Histogram Bin Terraces (ภูมิประเทศขั้นบันไดตามถังฮิสโตแกรม 255 ถัง)** พร้อมตัวบ่งชี้ทิศทางเมื่อข้อมูลเป็นค่าว่าง (`missing_go_to_left`)

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `sklearn.ensemble.HistGradientBoostingClassifier`
  - `sklearn.ensemble.HistGradientBoostingRegressor`
  - โมเดล HistGBM ที่รองรับ Native Categorical Features (`categorical_features`) และ Monotonic Constraints (`monotonic_cst`)
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.pkl`, `.pickle`, `.joblib`
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: แสดงความน่าจะเป็นสะสมจาก `staged_predict_proba` และขั้นบันไดฮิสโตแกรม
  - **Multiclass Classification**: รองรับ $C \ge 3$ คลาส (`_predictors[stage][class_idx]`)
  - **Regression**: รองรับ Loss ทุกชนิด (`squared_error`, `absolute_error`, `gamma`, `poisson`, `quantile`)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 การจัดกลุ่มข้อมูลลงถังฮิสโตแกรม (255-Bin Quantization) และ Missing Value Routing
ก่อนสร้างต้นไม้ `HistGradientBoosting` จะแปลงฟีเจอร์ต่อเนื่องแต่ละคอลัมน์ $j$ ให้กลายเป็นเลขจำนวนเต็ม `uint8` ในช่วง $\{0, 1, \dots, B_j - 1\}$ (โดยปกติ $B_j \le 255$) และสำรองถังสุดท้าย (`bin = 255`) ไว้สำหรับค่าว่าง (`NaN` / `missing_values_bin_idx_`):
- ขอบเขตของถังเก็บอยู่ใน `model._bin_mapper.bin_thresholds_[j]` ซึ่งเป็นอาเรย์ของค่าขีดแบ่งจริง $t_{j, 0} < t_{j, 1} < \dots < t_{j, B_j - 2}$
- ในโหนดภายใน $u$ ของ `TreePredictor` การตัดสินใจเดินซ้ายหรือขวาทำได้ทั้งบนค่าจริง (`num_threshold`) หรือบนหมายเลขถัง (`bin_threshold`) พร้อมแฟล็กพิเศษสำหรับค่า `NaN`:
  $$\text{Go Left}(u) = \begin{cases} \text{bool}(\text{missing\_go\_to\_left}[u]) & \text{if } x_{f_u} \text{ is NaN} \\ \text{BitsetLookup}(x_{f_u}) & \text{if } \text{is\_categorical}[u] = 1 \\ x_{f_u} \le \text{num\_threshold}[u] & \text{otherwise} \end{cases}$$

### 2.2 การแกะโครงสร้าง `model._predictors` และ `TreePredictor.nodes`
แตกต่างจาก `DecisionTree` ปกติ โมเดล `HistGradientBoosting` **ไม่มี `.estimators_` และไม่มีเมธอด `.decision_path()` หรือ `.apply()`** แต่เก็บต้นไม้ทั้งหมดไว้ใน `model._predictors`:
- `model._predictors` เป็น `list[list[TreePredictor]]` ขนาด `(n_iter_, n_trees_per_iter)` (ซึ่ง `n_trees_per_iter = 1` สำหรับ Binary/Regression และ `= C` สำหรับ Multiclass)
- ในแต่ละ `predictor: TreePredictor` มีแอตทริบิวต์ `predictor.nodes` ซึ่งเป็น **1D Structured NumPy Array (`NODE_DTYPE`)** ที่มีฟิลด์สำคัญดังนี้:
  - `feature_idx` (`intp`): ดัชนีฟีเจอร์ที่ใช้แบ่งที่โหนดนี้
  - `num_threshold` (`float64`): ค่าขีดแบ่งในหน่วยจริงของฟีเจอร์
  - `bin_threshold` (`uint8`): หมายเลขถังฮิสโตแกรมที่ใช้แบ่ง
  - `left` (`uint32`): ดัชนีโหนดลูกซ้าย
  - `right` (`uint32`): ดัชนีโหนดลูกขวา
  - `value` (`float64`): ค่าทำนาย Raw Margin / Residual ณ โหนดนั้น (ใน HistGBM ค่า `value` ของใบไม้ถูกคูณด้วย `learning_rate` มาเรียบร้อยแล้วตั้งแต่ตอนสร้างต้นไม้!)
  - `gain` (`float64`): ค่า Loss Gain ที่ลดลงจากการแบ่งโหนดนี้
  - `count` (`uint32`): จำนวนตัวอย่างตอนเทรนที่ตกมาที่โหนดนี้
  - `is_leaf` (`uint8`): `1` ถ้าเป็นโหนดใบไม้, `0` ถ้าเป็นโหนดภายใน
  - `is_categorical` (`uint8`): `1` ถ้าเป็นการแบ่งแบบ Categorical Bitset
  - `missing_go_to_left` (`uint8`): `1` ถ้าค่า `NaN` ต้องเดินไปทางซ้าย, `0` ถ้าไปทางขวา

### 2.3 การเขียน Vectorized NumPy Tree Walker เพื่อดึง Leaf Index, Path Contributions และ Gain Importance
1. **การคำนวณ Global Feature Importance จาก `gain` (แก้ปัญหาไม่มี `feature_importances_`)**:
   เนื่องจาก `HistGradientBoosting` ไม่มีแอตทริบิวต์ `model.feature_importances_` เราสามารถวนอ่าน `predictor.nodes` ของทุก Stage แล้วรวมค่า `nodes['gain']` แยกตาม `nodes['feature_idx']` ของโหนดที่ไม่ใช่ใบไม้ (`nodes['is_leaf'] == 0`) ได้เสร็จภายในไม่ถึง $1 \text{ ms}$!
2. **การแกะรอยเส้นทางและคำนวณ Exact Tree Path Contributions**:
   เนื่องจาก `nodes['value']` ใน `TreePredictor` เก็บค่าที่คูณ `learning_rate` ไว้แล้วทั้งที่โหนดภายในและโหนดใบไม้ เมื่อข้อมูลแถวที่ $i$ วิ่งจากโหนดพ่อ $u$ ไปยังโหนดลูก $v$ ค่าอิทธิพลที่ฟีเจอร์ $f_u$ สร้างขึ้นคือ:
   $$\Delta_{u \to v} = \text{nodes}[v][\text{'value'}] - \text{nodes}[u][\text{'value'}]$$
   รวมค่า $\Delta_{u \to v}$ ตลอดทุก Stage จะได้ `feature_contributions` ขนาด `(n_samples, n_features)` ที่เมื่อบวกกับ `model._baseline_prediction` จะเท่ากับ Raw Prediction สุดท้ายพอดี!

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Quantized Histogram Bin Grid (`HistogramBinGridOverlay.tsx`)**:
   - บนพื้นเวที 3 มิติ วาดเส้นตารางเรืองแสงตามขอบเขตถังจริง `bin_thresholds_` ของแกน $X$ และแกน $Y$ (สูงสุด 255 ช่อง)
   - แสดงให้เห็นว่าภายในถังฮิสโตแกรมเดียวกัน (`Bin #b`) พื้นผิว 3D จะเป็นแผ่นระนาบความสูงคงที่เสมอ เพราะโมเดลมองเห็นข้อมูลในถังเดียวกันเป็นค่า quantized เดียวกันทั้งหมด!
2. **Missing Value Routing Portal (`MissingValuePortalActor.tsx`)**:
   - หากข้อมูลในเฟรมปัจจุบันมีค่าว่าง (`NaN`) ในฟีเจอร์ใด จะแสดงสัญลักษณ์วงแหวนวาร์ปสีม่วงทอง (**NaN Routing Indicator**) บนกิ่งไม้และบนลูกบอลข้อมูล พร้อมลูกศรชี้ว่าโหนดนั้นส่งค่า `NaN` ไปทางซ้าย (`missing_go_to_left = 1`) หรือทางขวา (`= 0`)
3. **Residual Spiral Ladder & Stepped Bin Terraces**:
   - แสดงเกลียวสปริงแนวดิ่งสะสมค่าจาก `_baseline_prediction` ไปยังคำตอบสุดท้าย เหมือนใน Phase 31

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`HistGbmHudPanel.tsx`)
- **Active Histogram Bin Badge**: แสดงว่าค่าฟีเจอร์แกน $X, Y$ ของแถวปัจจุบันตกอยู่ในถังฮิสโตแกรมหมายเลขใด เช่น `Feature X: 42.3 ➔ Bin #118 / 255 (Range: [41.80, 43.10])`
- **Computed Total Gain Importance Chart**: แสดงกราฟความสำคัญของฟีเจอร์ที่สกัดสดจากผลรวม `gain` ใน `_predictors`
- **NaN Routing & Monotonic Constraint Inspector**: แสดงสถานะว่าแต่ละฟีเจอร์ถูกตั้ง `monotonic_cst` (`+1` เพิ่มขึ้นเท่านั้น, `-1` ลดลงเท่านั้น, `0` อิสระ) หรือไม่ และแสดงพฤติกรรมเมื่อเจอค่า `NaN`

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - เมื่อเลื่อนผ่านแถวที่มีค่า `NaN` แผง HUD จะสว่างขึ้นพร้อมแจ้งเตือน `[NaN Detected in Feature 'Income' ➔ Routed via missing_go_to_left]` เพื่อให้เห็นความสามารถของ `HistGradientBoosting` ในการรับมือค่าว่างโดยไม่ต้องทำ Imputation
2. เมื่ออยู่ในโหมด **What-If Sandbox**:
   - ผู้ใช้สามารถกดปุ่ม **`[Inject NaN]`** บนฟีเจอร์แกน $X$ หรือ $Y$ เพื่อจำลองว่าถ้าฟีเจอร์นี้หายไป (`NaN`) ลูกบอลข้อมูลจะถูกโหนดต่างๆ ส่งไปที่ใบไม้ไหนและได้คำตอบเท่าไหร่!

### 4.2 ASCII Wireframe ของหน้าจอ HistGradientBoosting HUD & Bin Grid
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: HistGradientBoostingClassifier (iters=100, max_bins=255)  [Inject NaN:OFF]|
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [HISTOGRAM BIN TELEMETRY]    |
|          3D QUANTIZED BIN TERRACES & NaN ROUTING PORTAL           | X (Age=34.2)  ➔ Bin #089/255 |
|                                                                   |   Bin Bounds : [33.5, 35.0]  |
|                  +-----------+-----------+                        | Y (Inc=NaN!)  ➔ Bin #255(NaN)|
|                  | Bin (89,4)| Bin (90,4)|                        |   NaN Route  : 82% Go Left   |
|          +-------+-----------+-----------+                        +------------------------------+
|          |       |    (o)    |           | <-- Flat plateau inside| [GAIN FEATURE IMPORTANCE]    |
|          |       | P = 0.76  |           |     same Histogram Bin!| Income : ████████████ 48.2%  |
|  +-------+-------+-----------+-----------+                        | Age    : ████████     31.5%  |
|  |   :   |   :   |   :   |   :   |   :   | <-- Quantized Bin Grid | Credit : █████        20.3%  |
|  +---+---+---+---+---+---+---+---+---+---+     (bin_thresholds_)  +------------------------------+
|   28.0  31.2  33.5  35.0  38.1  42.0                              | [STAGED PROGRESSION]         |
|                                                                   | Baseline Log-Odds : -0.405   |
|                                                                   | Final Prob P(y=1) :  0.762   |
+-------------------------------------------------------------------+------------------------------+
| [EXACT PATH DELTAS] Base: -0.41 | Income(NaN->L): +0.68 | Age(Bin#89): +0.89 ===> LogOdds: +1.16 |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0064 / 1000  [===(*)------------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/boosting/hist_gbm.py` (`HistGBMAdapter`)
- `frontend/src/domains/stage3d/actors/HistogramBinGridOverlay.tsx`
- `frontend/src/domains/hud/components/HistGbmHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `HistGBMAdapter`
```python
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

MAX_TRACED_ITERS = 100
MAX_TRAJECTORY_POINTS = 50

class HistGBMAdapter(BaseModelAdapter):
    """Adapter สำหรับ HistGradientBoostingClassifier และ HistGradientBoostingRegressor."""

    def __init__(self, model: Any, feature_names: list[str], target_class_idx: int = 1) -> None:
        super().__init__(model, feature_names)
        self.predictors: list[list[Any]] = model._predictors
        self.n_iter: int = len(self.predictors)
        self.n_trees_per_iter: int = len(self.predictors[0]) if self.n_iter > 0 else 1
        self.is_classifier: bool = isinstance(model, HistGradientBoostingClassifier)
        self.target_class_idx: int = target_class_idx
        self.tree_col: int = 0 if self.n_trees_per_iter == 1 else min(target_class_idx, self.n_trees_per_iter - 1)

    def compute_gain_feature_importances(self) -> np.ndarray:
        """คำนวณ Feature Importance จากผลรวม gain ของทุกโหนดใน _predictors (แทน feature_importances_ ที่ไม่มี)."""
        n_features = len(self.feature_names)
        gains = np.zeros(n_features, dtype=np.float64)
        for iter_preds in self.predictors:
            for pred in iter_preds:
                nodes = pred.nodes
                internal_mask = nodes["is_leaf"] == 0
                for f_idx, g in zip(nodes["feature_idx"][internal_mask], nodes["gain"][internal_mask]):
                    if 0 <= int(f_idx) < n_features:
                        gains[int(f_idx)] += float(g)
        total = np.sum(gains)
        return (gains / total if total > 0 else gains).astype(np.float32)

    def extract_bin_thresholds(self, max_bins_export: int = 128) -> dict[int, list[float]]:
        """สกัดขอบเขตถังฮิสโตแกรมจาก model._bin_mapper.bin_thresholds_."""
        mapper = getattr(self.model, "_bin_mapper", None)
        thresholds_list = getattr(mapper, "bin_thresholds_", None)
        if not thresholds_list:
            return {}
        result: dict[int, list[float]] = {}
        for f_idx, th_arr in enumerate(thresholds_list):
            if th_arr is not None and len(th_arr) > 0:
                step = max(1, len(th_arr) // max_bins_export)
                result[f_idx] = np.round(th_arr[::step], 5).tolist()
        return result

    def _walk_single_predictor(
        self, predictor: Any, X_f32: np.ndarray, contribs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """เดินตามโครงสร้าง predictor.nodes สำหรับข้อมูลทั้ง Chunk เพื่อเก็บ Leaf ID, Leaf Value และ Path Delta."""
        nodes = predictor.nodes
        n_samples = X_f32.shape[0]
        leaf_ids = np.zeros(n_samples, dtype=np.int32)
        leaf_vals = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            u = 0
            while nodes[u]["is_leaf"] == 0:
                f_idx = int(nodes[u]["feature_idx"])
                x_val = X_f32[i, f_idx]
                if np.isnan(x_val):
                    go_left = bool(nodes[u]["missing_go_to_left"])
                elif nodes[u]["is_categorical"]:
                    go_left = bool(x_val <= nodes[u]["num_threshold"])
                else:
                    go_left = bool(x_val <= nodes[u]["num_threshold"])
                v = int(nodes[u]["left"]) if go_left else int(nodes[u]["right"])
                contribs[i, f_idx] += float(nodes[v]["value"] - nodes[u]["value"])
                u = v
            leaf_ids[i] = u
            leaf_vals[i] = float(nodes[u]["value"])
        return leaf_ids, leaf_vals

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        X_f32 = np.asarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        base_raw = float(np.ravel(self.model._baseline_prediction)[self.tree_col])

        if self.is_classifier:
            staged_preds = np.column_stack([
                proba[:, min(self.target_class_idx, proba.shape[1] - 1)].astype(np.float32)
                for proba in self.model.staged_predict_proba(X_f32)
            ])
        else:
            staged_preds = np.column_stack([
                pred.astype(np.float32) for pred in self.model.staged_predict(X_f32)
            ])

        traced_iters = min(self.n_iter, MAX_TRACED_ITERS)
        contribs = np.zeros((n_samples, n_features), dtype=np.float32)
        leaf_matrix = np.zeros((n_samples, traced_iters), dtype=np.int32)
        stage_deltas = np.zeros((n_samples, traced_iters), dtype=np.float32)

        for m in range(traced_iters):
            pred_m = self.predictors[m][self.tree_col]
            l_ids, l_vals = self._walk_single_predictor(pred_m, X_f32, contribs)
            leaf_matrix[:, m] = l_ids
            stage_deltas[:, m] = l_vals

        step_idx = np.unique(np.linspace(0, self.n_iter - 1, min(self.n_iter, MAX_TRAJECTORY_POINTS), dtype=int))
        nan_mask = np.isnan(X_f32).any(axis=1).tolist()

        return InternalFlowBatch(
            base_value=base_raw,
            feature_contributions=contribs,
            extra_telemetry={
                "model_family": "hist_gbm",
                "n_iter_total": self.n_iter,
                "has_missing_values": nan_mask,
                "sampled_stage_indices": step_idx.tolist(),
                "staged_predictions": np.round(staged_preds[:, step_idx], 4).tolist(),
                "stage_raw_deltas_sample": np.round(stage_deltas, 4).tolist(),
                "leaf_indices_per_stage": leaf_matrix.tolist(),
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 64,
  "prediction": 0.762,
  "base_value": -0.405,
  "feature_contributions": [0.89, 0.68, 0.0],
  "telemetry": {
    "model_family": "hist_gbm",
    "n_iter_total": 100,
    "has_missing_values": true,
    "sampled_stage_indices": [0, 10, 50, 99],
    "staged_predictions": [0.42, 0.59, 0.71, 0.762],
    "stage_raw_deltas_sample": [0.18, 0.14, -0.05, 0.09],
    "leaf_indices_per_stage": [12, 9, 15, 7]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การเดินต้นไม้ผ่าน `predictor.nodes` (Structured Array इन-memory)**:
   - ต้นไม้ใน `HistGradientBoosting` ถูกจำกัดด้วย `max_leaf_nodes=31` โดยค่าเริ่มต้น (ความลึกเฉลี่ยเพียง 5–6 ชั้น) ทำให้ลูป `while nodes[u]["is_leaf"] == 0:` วิ่งเพียงไม่เกิน 6 รอบต่อแถวต่อต้นไม้
   - หากต้องการเร่งความเร็วเพิ่มเติม สามารถใช้ Vectorized Indexing บน `nodes[u_vec]` ทั้ง Chunk พร้อมกันได้โดยไม่ต้องสร้าง Object ใหม่
2. **การสกัด `feature_importances_` จาก `gain` แทนการใช้ `permutation_importance`**:
   - การรัน `sklearn.inspection.permutation_importance` ต้องทำนายซ้ำ $d \times n_{\text{repeats}}$ ครั้งซึ่งกินเวลาหลายวินาที แต่การรวมค่า `nodes["gain"]` จาก `model._predictors` ใช้เวลาไม่ถึง $0.5 \text{ ms}$ และไม่กิน RAM เพิ่มเลย
3. **การควบคุมจำนวนเส้นตาราง `HistogramBinGridOverlay.tsx`**:
   - แม้ฮิสโตแกรมจะมีได้ถึง 255 ถังต่อแกน แต่การวาดตาราง $255 \times 255$ เส้นพร้อมกันจะทำให้ลายตาและเปลือง Vertex Buffer จึงคัดแสดงเฉพาะเส้นขอบถังที่ถูกใช้เป็นจุดตัดจริง (`active_bin_thresholds`) ในต้นไม้ พร้อมเส้นตารางหลักไม่เกิน 64 เส้นต่อแกน

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **`HistGradientBoosting` ไม่มี `feature_importances_`** | หากโค้ดเรียก `model.feature_importances_` เหมือนต้นไม้ตัวอื่น จะเกิด `AttributeError: 'HistGradientBoostingClassifier' object has no attribute 'feature_importances_'` | ใช้เมธอด `compute_gain_feature_importances()` ที่รวมค่า `gain` จาก `_predictors` แทนเสมอ |
| 2 | **ข้อมูลมีค่า `NaN` (`np.isnan(x_val)`)** | การเปรียบเทียบ `np.nan <= threshold` ใน Python/NumPy จะคืนค่า `False` เสมอ ทำให้เดินผิดกิ่งหากโหนดนั้นตั้ง `missing_go_to_left == 1` | ต้องเช็ก `if np.isnan(x_val): go_left = bool(nodes[u]["missing_go_to_left"])` เป็นเงื่อนไขแรกสุดในทุกโหนดเสมอ! |
| 3 | **Native Categorical Split (`is_categorical == 1`)** | เมื่อโหนดแบ่งฟีเจอร์ประเภทหมวดหมู่ `HistGBM` จะใช้ Bitset (`_raw_left_cat_bitsets`) แทนการใช้ `<= num_threshold` | หาก `nodes[u]["is_categorical"] == 1` ให้เรียกใช้ฟังก์ชันเช็ก Bitset จาก `predictor.raw_left_cat_bitsets` หรือใช้ความต่างของ `staged_decision_function` ในการสอบทาน |
| 4 | **`nodes["value"]` คูณ `learning_rate` มาให้แล้ว** | ต่างจาก `GradientBoostingClassifier` ใน Phase 31 ที่ต้องคูณ `learning_rate` เอง ใน `HistGBM` ค่า `nodes["value"]` ถูกลดทอนด้วย `learning_rate` เรียบร้อยแล้ว | ห้ามนำ `model.learning_rate` ไปคูณซ้ำกับ `nodes["value"]` โดยเด็ดขาด เพราะจะทำให้ค่า Contribution เล็กลงเป็น $\eta^2$ |
| 5 | **Early Stopping ทำให้ `_predictors` สั้นกว่า `max_iter`** | `HistGradientBoosting` เปิด `early_stopping='auto'` เป็นค่าเริ่มต้นเมื่อข้อมูลเกิน 10,000 แถว | ใช้ `len(model._predictors)` แทน `model.max_iter` เสมอ และตรวจสอบกรณี `n_iter == 0` |
| 6 | **`_baseline_prediction` มี Shape ต่างกันใน Binary vs Multiclass** | ใน Binary/Regression `_baseline_prediction` เป็นสเกลาร์หรืออาเรย์ `(1,)` แต่ใน Multiclass เป็น `(1, n_classes)` | ใช้ `np.ravel(self.model._baseline_prediction)[self.tree_col]` เพื่อดึงค่าสเกลาร์อย่างปลอดภัยทุกกรณี |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `HistGBMAdapter` รองรับทั้ง `HistGradientBoostingClassifier` และ `HistGradientBoostingRegressor` โดยไม่พึ่งพา `.estimators_` หรือ `.feature_importances_`
- [ ] คำนวณ Feature Importance จากผลรวม `gain` ใน `_predictors` ได้อย่างถูกต้องรวดเร็ว
- [ ] รองรับการเดินต้นไม้เมื่อข้อมูลมีค่าว่าง (`NaN`) ตามแฟล็ก `missing_go_to_left` ของแต่ละโหนดได้อย่างแม่นยำ 100%
- [ ] ฉาก 3 มิติแสดงเส้นตาราง **3D Quantized Histogram Bin Grid** จาก `_bin_mapper.bin_thresholds_` และแสดงสัญลักษณ์ **NaN Routing Portal** เมื่อข้อมูลมีค่าว่าง
- [ ] โหมด What-If Sandbox มีปุ่มทดลอง **`[Inject NaN]`** เพื่อให้ผู้ใช้ทดสอบพฤติกรรมการรับมือค่าว่างของโมเดลแบบโต้ตอบได้
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุม Binary, Multiclass, Regression, ข้อมูลที่มีค่า `NaN` จำนวนมาก, และ Early Stopping
