# Phase 34: LightGBM Adapter (`.pkl`, `.txt`, `.model`) & 3D Leaf-Wise Split Hierarchy

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter ระดับ Production สำหรับ **LightGBM (Light Gradient Boosting Machine)** ซึ่งโดดเด่นด้วยกลยุทธ์การแตกกิ่งแบบ **Leaf-wise (Best-first) Tree Growth** ร่วมกับอัลกอริทึม GOSS และ EFB โดยรองรับทั้ง Scikit-learn Wrapper และไฟล์ Native Booster (`.txt`, `.model`) พร้อมดึงค่า **Exact TreeSHAP (`pred_contrib=True`)**, **Leaf Index (`pred_leaf=True`)**, และโครงสร้างต้นไม้จาก **`booster.dump_model()`** เพื่อเรนเดอร์ภูมิประเทศขั้นบันไดแบบไม่สมมาตร (Asymmetric Leaf-wise Terrain) และแผนผังการแตกกิ่งลึกเฉพาะจุดในโลก 3 มิติ

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `lightgbm.LGBMClassifier` (Scikit-learn API)
  - `lightgbm.LGBMRegressor` (Scikit-learn API)
  - `lightgbm.LGBMRanker` (Learning-to-Rank API)
  - `lightgbm.Booster` (Native C++ Booster โหลดจากไฟล์ข้อความ/ไบนารีโดยตรง)
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.txt`, `.model`, `.lgb` (LightGBM Native Text/Model Format)
  - `.pkl`, `.pickle`, `.joblib` (Serialized Python Wrapper / Booster)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification** (`binary`)
  - **Multiclass Classification** (`multiclass`, `multiclassova`)
  - **Regression & Ranking** (`regression`, `regression_l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`, `lambdarank`)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 คณิตศาสตร์ของการแตกกิ่งแบบ Leaf-wise (Best-First) เทียบกับ Level-wise
ในขณะที่ XGBoost/Scikit-learn GBM แตกกิ่งทีละชั้นความลึกพร้อมกัน (Level-wise / Depth-wise) โมเดล LightGBM จะเลือกใบไม้ใบเดียวจากใบไม้ทั้งหมดในปัจจุบันที่ให้ค่า **Maximum Split Gain ($\Delta \mathcal{L}_{\max}$)** สูงสุดมาแตกกิ่งต่อ:
$$l^* = \arg\max_{l \in \mathcal{L}_{\text{current}}} \Delta \mathcal{L}_{\text{split}}(l), \qquad \text{จนกว่า } |\mathcal{L}_{\text{current}}| = \text{num\_leaves}$$
- **ผลลัพธ์เชิงเรขาคณิต 3 มิติ**:
  พื้นผิว 3D ของ LightGBM จะ **ไม่สมมาตร (Asymmetric Resolution)** กล่าวคือ โซนของปริภูมิฟีเจอร์ที่ข้อมูลมีความชันหรือความซับซ้อนสูงจะถูกซอยย่อยเป็นขั้นบันไดขนาดจิ๋วที่ลึกและละเอียดมาก ในขณะที่โซนที่ข้อมูลเรียบง่ายจะถูกปล่อยเป็นแผ่นขั้นบันไดขนาดใหญ่ตั้งแต่ความลึกชั้นที่ 1 หรือ 2!

### 2.2 การดึง Exact TreeSHAP (`pred_contrib=True`) และความแตกต่างของ Multiclass Array Shape
LightGBM มี C++ TreeSHAP Engine ในตัว เรียกผ่าน `contribs = booster.predict(X_chunk, pred_contrib=True)` (สังเกตชื่อพารามิเตอร์คือ **`pred_contrib` ไม่มีตัว `s`** ต่างจาก XGBoost ที่ใช้ `pred_contribs`):
- **กรณี Binary Classification และ Regression**:
  - คืนค่า `np.ndarray` (หรือ `scipy.sparse` ถ้าอินพุตเป็น Sparse) ขนาด **`(n_samples, n_features + 1)`**
  - คอลัมน์ `0 .. n_features - 1` คือค่า Exact TreeSHAP $\phi_j$ ของแต่ละฟีเจอร์
  - คอลัมน์สุดท้าย `[:, -1]` คือค่า Expected Value / Bias ($F_0$) ในหน่วย Raw Margin
- **กรณี Multiclass Classification ($C$ คลาส) — จุดที่ต้องระวังเป็นพิเศษ!**:
  - แตกต่างจาก XGBoost ที่คืนอาเรย์ 3 มิติ โมเดล LightGBM จะคืนอาเรย์ **2 มิติที่ถูก攤แบน (Flattened 2D Array)** ขนาด:
    $$\text{Shape}_{\text{raw}} = \left(\text{n\_samples}, \; C \times (\text{n\_features} + 1)\right)$$
  - ดังนั้นต้องทำการ Reshape ทันทีด้วยคำสั่ง:
    ```python
    contribs_3d = raw_contribs.reshape(n_samples, n_classes, n_features + 1)
    ```
    จึงจะได้ค่า TreeSHAP ของคลาส $c$ ที่ `contribs_3d[:, c, :-1]` และค่า Bias ของคลาส $c$ ที่ `contribs_3d[:, c, -1]`!

### 2.3 การดึง Leaf Index (`pred_leaf=True`) และการแกะ `booster.dump_model()`
1. **การดึงหมายเลขใบไม้ (`pred_leaf=True`)**:
   - เรียก `leaves = booster.predict(X_chunk, pred_leaf=True)`
   - สำหรับ Binary/Regression: คืนอาเรย์ `int32` ขนาด `(n_samples, n_trees)` โดยค่าข้างในคือ `leaf_index` ($0 \dots \text{num\_leaves} - 1$)
   - สำหรับ Multiclass ($C$ คลาส): คืนอาเรย์ขนาด `(n_samples, n_trees * C)` ซึ่งต้อง `.reshape(n_samples, n_trees, C)` เพื่อดึงใบไม้ของคลาสเป้าหมาย `leaves_3d[:, :, target_class_idx]`
2. **การแกะโครงสร้างต้นไม้จาก `booster.dump_model()`**:
   - คืนค่า Python `dict` ที่มีคีย์สำคัญคือ:
     - `"feature_names"`: รายชื่อฟีเจอร์ทั้งหมด
     - `"pandas_categorical"`: ลิสต์ของหมวดหมู่จริงสำหรับ Categorical Features
     - `"tree_info"`: ลิสต์ของต้นไม้ทั้งหมด โดยแต่ละต้นมี `"tree_index"`, `"num_leaves"`, `"shrinkage"`, และ `"tree_structure"`
   - `"tree_structure"` เป็น Recursive Dict ที่แยกคีย์ชัดเจนระหว่าง:
     - **Internal Node**: `{"split_index": 0, "split_feature": 2, "split_gain": 34.5, "threshold": 18.2, "decision_type": "<=", "default_left": True, "missing_type": "NaN", "internal_value": -0.12, "internal_count": 400, "left_child": {...}, "right_child": {...}}`
     - **Leaf Node**: `{"leaf_index": 4, "leaf_value": 0.085, "leaf_count": 42}`

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Asymmetric Leaf-Wise Terrain (`SteppedVoxelTerrain.tsx`)**:
   - เรนเดอร์พื้นผิวขั้นบันได 3 มิติของ LightGBM พร้อมโหมดไฮไลต์ **Leaf Density Heatmap** ที่ระบายสีขอบขั้นบันไดตามความลึกของใบไม้ (`leaf_depth`) เพื่อให้เห็นโซนที่ LightGBM ทุ่มเทแตกกิ่งแบบ Leaf-wise ลึกเป็นพิเศษ
2. **3D Leaf-Wise Asymmetric Tree Hierarchy (`LeafWiseTreeVisualizer.tsx`)**:
   - แสดงแผนผังต้นไม้ที่สะท้อนรูปร่างแบบอสมมาตร (กิ่งหนึ่งสั้นแค่ชั้นเดียว แต่อีกกิ่งหนึ่งแตกลึกเป็นสายยาวลงมา 8 ชั้น) พร้อมเส้นแสง **Active Branch Glow** ที่วิ่งจาก Root ไปยัง `leaf_index` ของแถวปัจจุบัน
3. **3D Exact TreeSHAP Force Vectors (`TreeShapForceArrows.tsx`)**:
   - ใช้คอมโพเนนต์เดียวกับ `XGBoostAdapter` (Shared 100%) เพื่อวาดเวกเตอร์แรงผลักจากค่า `pred_contrib=True`

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`LightgbmHudPanel.tsx`)
- **Leaf-Wise Asymmetry & Depth Gauge**: แสดงความลึกของใบไม้ที่แถวปัจจุบันตกลงไป เทียบกับความลึกสูงสุดและต่ำสุดของต้นไม้นั้น (เช่น `Active Leaf #14 is at Depth 9 (Min Leaf Depth: 1, Max: 11, num_leaves: 31)`)
- **Exact TreeSHAP Waterfall Chart**: แสดงการบวกสะสมของค่า `pred_contrib` จาก Bias ไปสู่ Raw Margin
- **Categorical Bitset & `default_left` Inspector**: แสดงเงื่อนไขการแบ่งของโหนดประเภท Categorical (`decision_type == "=="`) เช่น `City in {'Bangkok', 'ChiangMai'}` ที่แปลกลับจาก `pandas_categorical`

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - เมื่อลูกบอลข้อมูลเคลื่อนที่จากพื้นที่ราบเรียบเข้าสู่พื้นที่ที่มีความซับซ้อนสูง ผู้ใช้จะเห็นแผนผัง **Leaf-Wise Tree Hierarchy** เปลี่ยนจากการวิ่งลงใบไม้ตื้นๆ (Depth 2) ไปเป็นกิ่งที่แตกลึก (Depth 9) ทันที!
2. เมื่ออยู่ในโหมด **What-If Sandbox**:
   - การลากลูกบอลในโซนที่แตกกิ่งตื้นจะทำให้ค่าทำนายคงที่เป็นบริเวณกว้าง แต่พอขยับเข้าสู่โซนที่แตกกิ่งแบบ Leaf-wise หนาแน่น ลูกบอลจะกระโดดขึ้นลงตามขั้นบันไดขนาดเล็กอย่างรวดเร็ว พร้อมอัปเดตกราฟ TreeSHAP Waterfall ทันที

### 4.2 ASCII Wireframe ของหน้าจอ LightGBM HUD & Leaf-Wise Hierarchy
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: LGBMClassifier (trees=150, num_leaves=31, fmt=.txt)    [ Tree #001 / 150]|
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [LEAF-WISE TREE #001 (t=88)] |
|         3D ASYMMETRIC LEAF-WISE TERRAIN & TREESHAP VECTORS        |           (Split #0)         |
|                                                                   |           /        \         |
|                      +---+---+---+                                |     [Leaf #0]    (Split #1)  |
|                      |L#9|L#8|L#7| <-- Deep Leaf-wise Micro-Steps |     (Depth 1)     /      \   |
|                  +---+---+---+---+     (High Complexity Zone)     |              [Leaf#1] (Sp#2) |
|                  |    (o)    |   |                                |                       /    \ |
|  +---------------+ P = 0.91  +---+                                |                  (Sp#3) [L#2]|
|  |                               |                                |                   /          |
|  |   Broad Flat Plateau (Leaf #0)| <-- Shallow Leaf (Depth 1)     |            [[ LEAF #9 ]]     |
|  |         P(y=1) = 0.08         |     (Low Complexity Zone)      | Active Leaf Depth : 5 (Asym) |
|  +-------------------------------+                                | Leaf Value (Margin): +0.214  |
+-------------------------------------------------------------------+------------------------------+
| [EXACT C++ TREESHAP] Bias: -1.10 | f1: +1.95 | f3(Cat=='A'): +1.62 | f0: -0.15 ===> Prob: 0.910  |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0088 / 1000  [==(*)------------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/boosting/lightgbm_adapter.py` (`LightGBMAdapter`)
- `frontend/src/domains/stage3d/actors/LeafWiseTreeVisualizer.tsx`
- `frontend/src/domains/hud/components/LightgbmHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `LightGBMAdapter`
```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import lightgbm as lgb
import numpy as np
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

MAX_EXPORTED_TREES = 100
MAX_TRAJECTORY_POINTS = 50

class LightGBMAdapter(BaseModelAdapter):
    """Adapter สำหรับ LGBMClassifier, LGBMRegressor และ Native lightgbm.Booster (.txt, .model, .pkl)."""

    def __init__(self, model_or_booster: Any, feature_names: list[str] | None = None, target_class_idx: int = 1) -> None:
        if isinstance(model_or_booster, (str, Path)):
            self.wrapper = None
            self.booster: lgb.Booster = lgb.Booster(model_file=str(model_or_booster))
        elif hasattr(model_or_booster, "booster_"):
            self.wrapper = model_or_booster
            self.booster = model_or_booster.booster_
        elif isinstance(model_or_booster, lgb.Booster):
            self.wrapper = None
            self.booster = model_or_booster
        else:
            raise TypeError(f"Unsupported LightGBM model type: {type(model_or_booster)}")

        resolved_names = feature_names or self.booster.feature_name() or []
        super().__init__(model_or_booster, list(resolved_names))
        self.target_class_idx = target_class_idx
        self.n_classes_internal: int = int(self.booster.num_model_per_iteration())
        self.n_trees_total: int = int(self.booster.num_trees())
        self.n_iterations: int = max(1, self.n_trees_total // max(1, self.n_classes_internal))

    def extract_model_dump_summary(self, max_trees: int = MAX_EXPORTED_TREES) -> dict[str, Any]:
        """ดึงโครงสร้างต้นไม้แบบ Leaf-wise และข้อมูล Categorical จาก booster.dump_model()."""
        full_dump = self.booster.dump_model(num_iteration=max_trees)
        return {
            "feature_names": full_dump.get("feature_names", self.feature_names),
            "pandas_categorical": full_dump.get("pandas_categorical", []),
            "tree_info": full_dump.get("tree_info", [])[:max_trees],
        }

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        """ดึง Exact TreeSHAP (pred_contrib=True), Leaf Index (pred_leaf=True) และ Staged Trajectory."""
        X_f32 = np.ascontiguousarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape

        # 1. Exact C++ TreeSHAP (pred_contrib=True) — กำหนด num_threads=2
        raw_contrib = self.booster.predict(X_f32, pred_contrib=True, num_threads=2)
        if hasattr(raw_contrib, "toarray"):
            raw_contrib = raw_contrib.toarray()
        raw_contrib = np.asarray(raw_contrib, dtype=np.float32)

        # จัดการกรณี Multiclass ที่ LightGBM คืน Flattened 2D Array: (n_samples, n_classes * (n_features + 1))
        if self.n_classes_internal > 1 and raw_contrib.shape[1] == self.n_classes_internal * (n_features + 1):
            contrib_3d = raw_contrib.reshape(n_samples, self.n_classes_internal, n_features + 1)
            c_idx = min(self.target_class_idx, self.n_classes_internal - 1)
            contribs = contrib_3d[:, c_idx, :-1]
            base_val = float(contrib_3d[0, c_idx, -1])
        else:
            contribs = raw_contrib[:, :-1]
            base_val = float(raw_contrib[0, -1])

        # 2. Leaf Index Extraction (pred_leaf=True)
        raw_leaves = np.asarray(self.booster.predict(X_f32, pred_leaf=True, num_threads=2), dtype=np.int32)
        if raw_leaves.ndim == 1:
            raw_leaves = raw_leaves.reshape(n_samples, -1)
        if self.n_classes_internal > 1 and raw_leaves.shape[1] == self.n_iterations * self.n_classes_internal:
            leaves_3d = raw_leaves.reshape(n_samples, self.n_iterations, self.n_classes_internal)
            c_idx = min(self.target_class_idx, self.n_classes_internal - 1)
            leaf_matrix = leaves_3d[:, :, c_idx]
        else:
            leaf_matrix = raw_leaves

        # 3. Staged Trajectory via num_iteration
        step_iters = np.unique(np.linspace(1, self.n_iterations, min(self.n_iterations, MAX_TRAJECTORY_POINTS), dtype=int))
        staged_cols = []
        for it in step_iters:
            pred_it = np.asarray(self.booster.predict(X_f32, num_iteration=int(it), num_threads=2), dtype=np.float32)
            if pred_it.ndim == 2:
                c_idx = min(self.target_class_idx, pred_it.shape[1] - 1)
                staged_cols.append(pred_it[:, c_idx])
            else:
                staged_cols.append(pred_it)
        staged_trajectory = np.column_stack(staged_cols)

        k_export = min(leaf_matrix.shape[1], MAX_EXPORTED_TREES)
        return InternalFlowBatch(
            base_value=base_val,
            feature_contributions=contribs,
            extra_telemetry={
                "model_family": "lightgbm",
                "n_iterations": self.n_iterations,
                "sampled_iteration_indices": (step_iters - 1).tolist(),
                "staged_trajectory": np.round(staged_trajectory, 4).tolist(),
                "leaf_indices_sample": leaf_matrix[:, :k_export].tolist(),
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 88,
  "prediction": 0.910,
  "base_value": -1.100,
  "feature_contributions": [-0.15, 1.95, 0.0, 1.62],
  "telemetry": {
    "model_family": "lightgbm",
    "n_iterations": 150,
    "sampled_iteration_indices": [0, 10, 45, 95, 149],
    "staged_trajectory": [0.35, 0.61, 0.79, 0.88, 0.910],
    "leaf_indices_sample": [9, 14, 0, 22, 18]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การจำกัด `num_iteration` ใน `booster.dump_model(num_iteration=100)`**:
   - เมธอด `booster.dump_model()` ของ LightGBM สร้าง Python Dict ซ้อนกันลึกตามจำนวนต้นไม้และจำนวนใบไม้ หากโมเดลมี $5,000$ ต้น การเรียก `dump_model()` ทั้งหมดจะสร้าง JSON ขนาดเกือบ $100 \text{ MB}$!
   - การระบุ `num_iteration=MAX_EXPORTED_TREES` ($100$) ช่วยให้ LightGBM ดัมพ์เฉพาะ 100 รอบแรกออกมาในระดับ C++ ประหยัดทั้งเวลาและ RAM ลงกว่า $95\%$
2. **การตั้งค่า `num_threads=2` ในทุกการเรียก `booster.predict(...)`**:
   - ป้องกันไม่ให้ OpenMP ของ LightGBM แย่งคอร์ CPU ทั้งหมดเมื่อทำงานขนานกับ Worker อื่นใน `ComputePool`
3. **การแปลง `X_chunk` เป็น Contiguous `np.float32`**:
   - ส่ง `np.ascontiguousarray(X_chunk, dtype=np.float32)` เข้า `booster.predict` โดยตรงแทน `pd.DataFrame` เพื่อข้ามขั้นตอนการแปลงประเภทคอลัมน์ของ Pandas ในทุก Chunk

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **Multiclass `pred_contrib=True` คืนค่าเป็น 2D Flattened Array** | ต่างจาก XGBoost ที่คืน 3D Array ใน LightGBM ค่า `pred_contrib=True` สำหรับ Multiclass จะมีขนาด `(n_samples, n_classes * (n_features + 1))` หากดึง `[:, :-1]` ตรงๆ จะได้จำนวนคอลัมน์ผิดและพังทันที! | ตรวจสอบ `self.n_classes_internal > 1` แล้วทำ `.reshape(n_samples, self.n_classes_internal, n_features + 1)` ก่อนดึง `[:, c_idx, :-1]` เสมอ |
| 2 | **ชื่อพารามิเตอร์ `pred_contrib` (ไม่มี `s`) vs `pred_contribs` ของ XGBoost** | หากส่ง `pred_contribs=True` เข้า LightGBM จะไม่ถูกตีความว่าเป็น TreeSHAP หรือเกิด Warning เรื่อง Unknown Parameter | ใช้ `pred_contrib=True` (เอกพจน์) สำหรับ LightGBM เสมอ |
| 3 | **เมื่อมีเพียง 1 ต้นไม้ (`n_trees == 1`) `pred_leaf=True` คืน 1D Array** | ใน LightGBM หากโมเดลมีเพียง 1 ต้นไม้ `booster.predict(X, pred_leaf=True)` จะคืนอาเรย์ 1 มิติขนาด `(n_samples,)` แทนที่จะเป็น `(n_samples, 1)` | เช็ก `if raw_leaves.ndim == 1: raw_leaves = raw_leaves.reshape(n_samples, -1)` เสมอ |
| 4 | **Categorical Splits ใน `dump_model()` ใช้รูปแบบ `"0||3||5"`** | โหนดที่แบ่งฟีเจอร์หมวดหมู่จะมี `"decision_type": "=="` และค่า `"threshold"` เป็นสตริงของดัชนีที่คั่นด้วย `||` เช่น `"0||3||5"` ไม่ใช่ตัวเลข `float` | ใน Tree Parser ให้ตรวจเช็ก `node.get("decision_type") == "=="` แล้วแยกสตริงด้วย `.split("||")` แทนการแปลงเป็น `float(node["threshold"])` |
| 5 | **อินพุตเป็น Sparse Matrix ทำให้ `pred_contrib=True` คืน `scipy.sparse`** | หากส่ง Sparse Matrix เข้าไป LightGBM จะคืน `pred_contrib` เป็น `scipy.sparse.csr_matrix` | ใช้ `np.ascontiguousarray(X_chunk, dtype=np.float32)` และดัก `if hasattr(raw_contrib, "toarray"): raw_contrib = raw_contrib.toarray()` |
| 6 | **ชื่อฟีเจอร์มีอักขระพิเศษ (`[`, `]`, `:`, `,`) หรือช่องว่าง** | LightGBM แทนที่ช่องว่างด้วย `_` ใน `booster.feature_name()` ทำให้ชื่อในโมเดลอาจไม่ตรงกับหัวคอลัมน์ใน CSV | อ้างอิงตามลำดับดัชนีคอลัมน์ (`feature_idx: 0 .. d-1`) เมื่อส่ง `np.ndarray` เข้า `booster.predict` และเก็บ Mapping ระหว่างชื่อคอลัมน์จริงกับ `booster.feature_name()` |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `LightGBMAdapter` รองรับทั้ง `LGBMClassifier`, `LGBMRegressor`, และ `lgb.Booster` จากไฟล์ `.pkl`, `.joblib`, `.txt`, และ `.model`
- [ ] ดึงค่า Exact C++ TreeSHAP ด้วย `pred_contrib=True` และทำการ `.reshape(n_samples, n_classes, n_features + 1)` สำหรับงาน Multiclass ได้อย่างถูกต้องแม่นยำ
- [ ] ดึง `pred_leaf=True` พร้อมป้องกันกรณี 1D Array เมื่อมีต้นไม้เดียว และแกะโครงสร้าง Leaf-wise จาก `booster.dump_model(num_iteration=100)`
- [ ] ฉาก 3 มิติแสดงภูมิประเทศ **3D Asymmetric Leaf-Wise Terrain** และแผนผังต้นไม้อสมมาตรพร้อมเส้นแสง **Active Branch Glow**
- [ ] แผง HUD แสดงความลึกของใบไม้ปัจจุบัน (`Active Leaf Depth`) เทียบกับความลึกตื้นสุด/ลึกสุดของต้นไม้ และแสดงกราฟ **Exact TreeSHAP Waterfall** ร่วมกับคอมโพเนนต์ของ Phase 33 ได้ 100%
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุมไฟล์ `.txt`/`.model`, Binary, Multiclass (3D Reshape), Regression, Single-tree (`n_estimators=1`), และ Categorical Splits (`"0||2"`)
