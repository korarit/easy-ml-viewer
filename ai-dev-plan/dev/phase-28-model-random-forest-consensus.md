# Phase 28: Random Forest Adapter & 3D Multi-Tree Voting Arena + Active Branch Glow

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter สำหรับโมเดลกลุ่มป่าสุ่ม (**Random Forest**) เพื่อเปิดเผยกระบวนการลงคะแนนเสียงของต้นไม้ตัดสินใจจำนวนมากในป่า (`estimators_`) ออกมาเป็น **3D Multi-Tree Voting Arena (ลานโหวตคริสตัลของป่า 3 มิติ)** พร้อมคำนวณฉันทามติของป่า (Ensemble Consensus Ratio) และส่วนเบี่ยงเบนระหว่างต้นไม้ในทุกเฟรม

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `sklearn.ensemble.RandomForestClassifier`
  - `sklearn.ensemble.RandomForestRegressor`
  - โมเดล Random Forest ที่อยู่ภายใน `sklearn.pipeline.Pipeline` หรือ `GridSearchCV.best_estimator_`
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.pkl`, `.pickle`, `.joblib`
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: เฉลี่ยความน่าจะเป็นจากต้นไม้ทุกต้นในป่า (Soft Voting ตามมาตรฐาน Scikit-learn) และนับสัดส่วน Hard Vote รายต้นไม้
  - **Multiclass Classification**: เฉลี่ยเวกเตอร์ความน่าจะเป็นขนาด $C$ คลาส across `n_estimators`
  - **Regression**: เฉลี่ยค่าทำนายต่อเนื่องจากต้นไม้ทุกต้น พร้อมคำนวณช่วงความเชื่อมั่นจากความแปรปรวนระหว่างต้นไม้

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 สมการการรวมผลโหวตแบบ Bagging Ensemble
Random Forest สร้างต้นไม้ตัดสินใจอิสระจำนวน $K$ ต้น ($\{T_1, T_2, \dots, T_K\}$) จาก Bootstrap Samples และการสุ่มซับเซตของฟีเจอร์ในแต่ละจุดตัด:
- **กรณี Regression**:
  $$\hat{y}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^{K} T_k(\mathbf{x}), \qquad \sigma_{\text{forest}}(\mathbf{x}) = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \left(T_k(\mathbf{x}) - \hat{y}(\mathbf{x})\right)^2}$$
- **กรณี Classification (Scikit-learn Soft Voting)**:
  ให้ $\mathbf{p}_k(\mathbf{x}) \in [0, 1]^C$ คือเวกเตอร์ความน่าจะเป็นที่ใบไม้ของต้นไม้ต้นที่ $k$:
  $$\mathbf{P}_{\text{forest}}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^{K} \mathbf{p}_k(\mathbf{x}), \qquad \hat{c} = \arg\max_{c} P_{\text{forest}, c}(\mathbf{x})$$
- **อัตราฉันทามติของป่า (Ensemble Consensus / Agreement Ratio)**:
  ให้ $\hat{c}_k(\mathbf{x}) = \arg\max_c p_{k,c}(\mathbf{x})$ คือคลาสที่ต้นไม้ต้นที่ $k$ โหวตให้ชนะ อัตราฉันทามติของป่าสำหรับแถว $\mathbf{x}$ คือ:
  $$\text{Consensus}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\left(\hat{c}_k(\mathbf{x}) = \hat{c}\right) \in \left[\frac{1}{C}, 1.0\right]$$

### 2.2 การแกะโครงสร้างต้นไม้ย่อยและดึง Leaf Index (`model.estimators_` & `model.apply`)
- **รายการต้นไม้ในป่า (`model.estimators_`)**: เป็น Python `list` ของ `DecisionTreeClassifier` หรือ `DecisionTreeRegressor` จำนวน $K$ ต้น โดยแต่ละต้นมี `estimator.tree_` เหมือนใน Phase 27 ทุกประการ
- **การดึง Leaf Index ของทั้งป่า**:
  เรียก `leaf_matrix = model.apply(X_chunk)` จะได้ `np.ndarray` ชนิด `int64` ขนาด `(n_samples, n_estimators)` ในครั้งเดียว (รันแบบขนานด้วย OpenMP/Joblib ภายใน Scikit-learn)
- **การดึงผลทำนายรายต้นไม้แบบเร็วพิเศษ (Fast Leaf Value Lookup)**:
  แทนที่จะวนลูปเรียก `tree.predict(X_chunk)` ทีละต้น เราสามารถใช้ `leaf_matrix[:, k]` ไปเปิดตาราง (Fancy Indexing) จากอาเรย์ `node_expectations[k][leaf_matrix[:, k]]` ที่คำนวณเตรียมไว้ล่วงหน้าของแต่ละต้นไม้ได้ทันที! วิธีนี้เร็วกว่าการ traverse ต้นไม้ซ้ำถึง $5\times$

### 2.3 การคำนวณ Ensemble Tree Path Feature Contributions
ค่าอิทธิพลของแต่ละฟีเจอร์ของทั้งป่า ($\Phi_j(\mathbf{x})$) คือค่าเฉลี่ยของค่าอิทธิพลตามเส้นทาง (`decision_path`) ของต้นไม้แต่ละต้น:
$$\Phi_j(\mathbf{x}) = \frac{1}{K_{\text{traced}}} \sum_{k=1}^{K_{\text{traced}}} \phi_{k, j}(\mathbf{x}), \qquad \text{Base}_{\text{forest}} = \frac{1}{K_{\text{traced}}} \sum_{k=1}^{K_{\text{traced}}} E_{k, \text{root}}$$
- **Array Shape ของ Contributions**:
  - **Binary / Regression**: `(n_samples, n_features)` (ชนิด `float32`)
  - **Multiclass**: `(n_samples, n_classes, n_features)` (ชนิด `float32`)

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Forest Voting Ring (`ForestVotingArena.tsx`)**:
   - รอบขอบเวที 3 มิติ จัดวาง **แท่งเสาคริสตัลเรืองแสง (Voting Pillars)** เรียงเป็นวงแหวน จำนวนสูงสุด 100 แท่ง (แทนต้นไม้ $T_1 \dots T_{100}$)
   - ในแต่ละเฟรม ความสูงและสีของเสาคริสตัลต้นที่ $k$ จะเปลี่ยนตามค่าทำนาย $T_k(\mathbf{x}_t)$ (เช่น สีฟ้านีออน = โหวต Class 1, สีส้มแอมเบอร์ = โหวต Class 0)
   - มีเส้นลำแสงเลเซอร์บางๆ (Voting Beams) พุ่งจากยอดเสาคริสตัลที่โหวตชนะเข้ามายังลูกบอลข้อมูลตรงกลาง
2. **Ensemble Multi-Step Voxel Terrain (`SteppedVoxelTerrain.tsx`)**:
   - พื้นผิว 3 มิติที่เกิดจากการซ้อนทับขั้นบันไดของต้นไม้หลายสิบต้น ทำให้เกิดขั้นบันไดขนาดเล็กที่ละเอียดและซับซ้อนกว่าต้นไม้เดี่ยว (Fine-grained Terraces)
3. **Single-Tree Spotlight & Active Branch Glow**:
   - เมื่อผู้ใช้คลิกเลือกเสาคริสตัลต้นที่ $k$ ในวงแหวน (หรือเลือกจาก Tree Selector ใน HUD) พื้นผิว 3D จะสลับไปแสดงเฉพาะขั้นบันไดของต้นไม้ต้นที่ $k$ พร้อมแสดง **Active Tree Branch Glow** ของต้นไม้นั้น

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`RandomForestHudPanel.tsx`)
- **Tree Selector (`Tree #k Explorer`)**: ดรอปดาวน์และปุ่มเลื่อน `< Tree #12 / 100 >` เพื่อส่องดูโครงสร้างกิ่งไม้และ `active_path` ของต้นไม้รายต้นเทียบกับค่าเฉลี่ยของทั้งป่า
- **Consensus & Vote Distribution Bar**: แถบแสดงสัดส่วนผลโหวตของแถวปัจจุบัน เช่น `Class 1: 86 trees (86%) | Class 0: 14 trees (14%)` หรือฮิสโตแกรมการกระจายตัวของคำตอบต้นไม้ในงาน Regression
- **Global Gini Importance vs Local Frame Contribution Toggle**: สลับดูความสำคัญฟีเจอร์ภาพรวม (`model.feature_importances_`) เทียบกับแรงผลักฟีเจอร์เฉพาะของแถวปัจจุบัน

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - ลูกบอลข้อมูลวิ่งไปตามภูมิประเทศขั้นบันไดรวมของป่า
   - วงแหวนเสาคริสตัล **3D Forest Voting Ring** รอบเวทีจะกระเพื่อมขึ้นลงเป็นคลื่น (Mexican Wave / Equalizer Effect) ตามมติของต้นไม้ทั้ง 100 ต้นในแต่ละเฟรม
2. เมื่ออยู่ในโหมด **What-If Sandbox** แล้วลากลูกบอลเข้าสู่บริเวณรอยต่อระหว่างคลาส (Decision Boundary):
   - ผู้ใช้จะเห็นเสาคริสตัลรอบเวทีค่อยๆ เปลี่ยนสีทีละต้นสองต้น (เช่น จากชนะขาด 95:5 ค่อยๆ กลายเป็นก้ำกึ่ง 52:48) สะท้อนให้เห็นว่าทำไมความน่าจะเป็นของ Random Forest จึงเปลี่ยนแปลงอย่างนุ่มนวลกว่า Decision Tree เดี่ยว

### 4.2 ASCII Wireframe ของหน้าจอ Random Forest HUD & Voting Arena
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: RandomForestClassifier (n_estimators=200)   [ View: Ensemble | Tree #k ] |
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [FOREST CONSENSUS (t=108)]   |
|               3D MULTI-TREE VOTING ARENA & TERRAIN                |                              |
|                                                                   | Agreement: 84.0% (High)      |
|             |  |  |  |  |  |  |  |  |  <-- 100 Crystal Pillars    | [████████████████░░░] 168/32 |
|          |  #  #  .  #  #  #  .  #  #  |   (#=Class 1, .=Class 0) |                              |
|        |  #   \     |     /     /    #  |                         | Mean Prob P(y=1) : 0.812     |
|       |  .     \    v    /     /      # |                         | Inter-Tree Std   : 0.145     |
|       |  #    +-------------------+   . |                         +------------------------------+
|       |  #    | Ensemble Terraces |   # |                         | [TREE INSPECTOR: Tree #07]   |
|        |  #   |       (o) <-------+---# | <-- Voting Laser Beams  | Tree #07 Pred    : 0.940     |
|          |    |  P(y=1) = 0.81    |   |                           | Leaf ID          : #23       |
|               +-------------------+                               | Active Path (Depth 4):       |
|                                                                   |  0: f2 <= 1.85 (True, +0.21) |
|                                                                   |  1: f0 > 45.0  (True, +0.23) |
+-------------------------------------------------------------------+------------------------------+
| [ENSEMBLE DELTAS] Base: 0.50 | f2: +0.19 | f0: +0.15 | f4: -0.03 ===> Final Pred: 0.81           |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0108 / 1000  [====(*)-----------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/trees/random_forest.py` (`BaseForestAdapter`, `RandomForestAdapter`)
- `frontend/src/domains/stage3d/actors/ForestVotingArena.tsx`
- `frontend/src/domains/hud/components/RandomForestHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `BaseForestAdapter` และ `RandomForestAdapter`
```python
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

MAX_EXPORTED_TREES = 100

class BaseForestAdapter(BaseModelAdapter):
    """Base Adapter สำหรับตระกูล Bagging Forest (RandomForest และ ExtraTrees)."""

    def __init__(self, model: Any, feature_names: list[str], target_class_idx: int = 1) -> None:
        super().__init__(model, feature_names)
        self.estimators: list[Any] = list(model.estimators_)
        self.n_estimators: int = len(self.estimators)
        self.is_classifier: bool = hasattr(model, "predict_proba")
        self.target_class_idx: int = target_class_idx
        self._node_exp_per_tree: list[np.ndarray] = [
            self._extract_node_expectations(est.tree_) for est in self.estimators
        ]

    def _extract_node_expectations(self, tree_struct: Any) -> np.ndarray:
        vals = tree_struct.value
        if not self.is_classifier:
            return vals[:, 0, 0].astype(np.float32)
        totals = np.sum(vals[:, 0, :], axis=1, keepdims=True)
        probs = np.divide(vals[:, 0, :], totals, out=np.zeros_like(vals[:, 0, :]), where=totals > 0)
        idx = min(self.target_class_idx, probs.shape[1] - 1)
        return probs[:, idx].astype(np.float32)

    def predict_all_trees_fast(self, X_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ดึงผลทำนายของต้นไม้ทุกต้นแบบเร็วพิเศษผ่าน model.apply(X) + Lookup Table."""
        leaf_matrix = self.model.apply(X_f32)  # shape: (n_samples, n_estimators)
        n_samples = X_f32.shape[0]
        tree_preds = np.empty((n_samples, self.n_estimators), dtype=np.float32)
        for k, node_exp in enumerate(self._node_exp_per_tree):
            tree_preds[:, k] = node_exp[leaf_matrix[:, k]]
        return tree_preds, leaf_matrix

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        X_f32 = np.asarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        tree_preds, leaf_matrix = self.predict_all_trees_fast(X_f32)

        k_export = min(self.n_estimators, MAX_EXPORTED_TREES)
        inter_tree_mean = np.mean(tree_preds, axis=1)
        inter_tree_std = np.std(tree_preds, axis=1, ddof=0)

        if self.is_classifier:
            votes_pos = np.mean(tree_preds >= 0.5, axis=1)
            consensus_ratio = np.maximum(votes_pos, 1.0 - votes_pos)
        else:
            consensus_ratio = 1.0 / (1.0 + inter_tree_std)

        # คำนวณค่าเฉลี่ย Feature Contributions จาก k_export ต้นแรก
        contribs = np.zeros((n_samples, n_features), dtype=np.float32)
        base_val = float(np.mean([exp[0] for exp in self._node_exp_per_tree[:k_export]]))

        for k in range(k_export):
            est = self.estimators[k]
            csr = est.decision_path(X_f32)
            feats = est.tree_.feature
            exp_k = self._node_exp_per_tree[k]
            for i in range(n_samples):
                path = csr.indices[csr.indptr[i] : csr.indptr[i + 1]]
                for s in range(len(path) - 1):
                    u, v = path[s], path[s + 1]
                    f_idx = feats[u]
                    if f_idx >= 0:
                        contribs[i, f_idx] += (exp_k[v] - exp_k[u]) / k_export

        return InternalFlowBatch(
            base_value=base_val,
            feature_contributions=contribs,
            extra_telemetry={
                "tree_predictions_sample": np.round(tree_preds[:, :k_export], 4).tolist(),
                "leaf_indices_sample": leaf_matrix[:, :k_export].astype(np.int32).tolist(),
                "inter_tree_mean": np.round(inter_tree_mean, 5).tolist(),
                "inter_tree_std": np.round(inter_tree_std, 5).tolist(),
                "consensus_ratio": np.round(consensus_ratio, 4).tolist(),
            },
        )

class RandomForestAdapter(BaseForestAdapter):
    """Adapter สำหรับ RandomForestClassifier และ RandomForestRegressor."""
    pass
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 108,
  "prediction": 0.812,
  "base_value": 0.500,
  "feature_contributions": [0.15, 0.0, 0.19, 0.0, -0.028],
  "telemetry": {
    "model_family": "random_forest",
    "n_estimators_total": 200,
    "n_estimators_exported": 100,
    "consensus_ratio": 0.84,
    "inter_tree_std": 0.145,
    "tree_predictions_sample": [0.92, 0.88, 0.15, 0.94, 0.79],
    "leaf_indices_sample": [14, 22, 7, 23, 19]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การส่ง `np.ndarray` (`float32`) แทน `pd.DataFrame` เข้า `estimators_`**:
   - ต้นไม้ย่อย `model.estimators_[k]` ของ Scikit-learn ไม่มีแอตทริบิวต์ `feature_names_in_` หากส่ง `DataFrame` เข้าไปตรงๆ นอกจากจะช้าลงกว่า $10\times$ จากการสร้าง Index Slice แล้ว ยังอาจเกิด Overhead มหาศาลเมื่อคูณด้วยจำนวนต้นไม้ $K$ ต้น ดังนั้นต้องแปลงเป็น `np.asarray(X_chunk, dtype=np.float32)` ตั้งแต่วินาทีแรก
2. **เทคนิค `model.apply(X)` + Node Expectation Lookup**:
   - การเรียก `[tree.predict(X) for tree in model.estimators_]` ใน Python Loop มี Overhead ของการเข้า-ออก Cython $K$ ครั้ง แต่การเรียก `model.apply(X_f32)` เพียงครั้งเดียวจะใช้ Cython Parallel Loop ของ Scikit-learn คืนค่า Leaf ID ทั้งหมดทันที แล้วใช้ Numpy Fancy Indexing ดึงค่าทำนายของทุกต้นไม้ในเวลาไม่ถึง $2 \text{ ms}$ ต่อ $1,000$ แถว!
3. **การจำกัดจำนวนต้นไม้ที่เก็บละเอียดในไฟล์ `.mlv` (`MAX_EXPORTED_TREES = 100`)**:
   - หากผู้ใช้เทรน Random Forest ด้วย `n_estimators = 2,000` การเก็บผลโหวตและ Leaf ID ทั้ง 2,000 ต้นในทุกเฟรมจะทำให้ไฟล์ `.mlv` ใหญ่ขึ้น $20\times$
   - เราจึงเก็บอาเรย์รายต้น (`tree_predictions_sample`, `leaf_indices_sample`) สูงสุดเพียง **100 ต้นแรก** สำหรับแสดงใน 3D Voting Ring แต่คำนวณค่าสถิติรวม (`inter_tree_mean`, `inter_tree_std`, `consensus_ratio`) จาก **ต้นไม้ทั้งหมด 2,000 ต้นครบถ้วน**
4. **การวาดเสาคริสตัล 100 ต้นด้วย `THREE.InstancedMesh` เดียว**:
   - ใน `ForestVotingArena.tsx` ใช้ `InstancedMesh` ขนาด `count = min(n_estimators, 100)` อัปเดตความสูง (`scale.y`) และสี (`setColorAt`) ใน `useFrame` โดยไม่สร้าง Mesh แยกชิ้น ทำให้กิน GPU Draw Call เพียง 1 ครั้ง

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **`estimators_[k]` ไม่มี `feature_names_in_`** | หากส่ง DataFrame เข้า `estimators_[k]` หรือส่ง ndarray เข้าโมเดลหลักที่เทรนด้วย DataFrame จะเกิด `UserWarning: X has feature names...` นับพันครั้ง | ใช้ `np.asarray(X, dtype=np.float32)` สำหรับ `estimators_[k]` เสมอ และใช้ `warnings.catch_warnings()` บล็อกเฉพาะ `UserWarning` เรื่องชื่อฟีเจอร์ในโมเดลหลัก |
| 2 | **ป่าขนาดใหญ่มาก (`n_estimators >= 1000`)** | การวนลูป `est.decision_path(X_f32)` ครบ 1,000 ต้นในทุก Chunk อาจใช้เวลานานเกินงบเวลา Pre-computation | จำกัดการแกะ `decision_path` เฉพาะ `k_export = min(n_estimators, 100)` ต้นแรก ซึ่งเพียงพอที่จะให้ค่าเฉลี่ย Feature Contribution ลู่เข้าสู่ค่าจริงด้วยความคลาดเคลื่อนต่ำมาก |
| 3 | **โมเดลถูกเทรนด้วย `n_jobs=-1` ค้างไว้ในไฟล์ `.pkl`** | เมื่อรันภายใน Worker Process ของ `ComputePool` การที่โมเดลพยายามแตก Thread ย่อยซ้ำซ้อน (Oversubscription) จะทำให้ CPU Context Switch หนักและช้าลง | ก่อนรัน Pre-computation ให้ตั้งค่า `model.n_jobs = 1` หรือครอบด้วย `joblib.parallel_backend("threading", n_jobs=2)` ภายใน Worker |
| 4 | **ต้นไม้บางต้นในป่ามีเพียงโหนดเดียว (`node_count == 1`)** | เกิดขึ้นเมื่อ `max_samples` เล็กมากหรือสุ่มเจอฟีเจอร์คงที่ ทำให้ `decision_path` มีความยาว 1 โหนด | ลูป `for s in range(len(path) - 1)` จะข้ามต้นไม้นั้นไปอย่างปลอดภัยโดยไม่เกิดการหารด้วยศูนย์ |
| 5 | **Multiclass ที่ต้นไม้ย่อยบางต้นเห็นคลาสไม่ครบตอน Bootstrap** | ใน Bootstrap Sample เล็กๆ ต้นไม้บางต้นอาจมี `tree_.n_classes_[0] < model.n_classes_` | ตรวจสอบ `est.classes_` ของแต่ละต้นไม้แล้วแมปกลับเข้าสู่ดัชนีของ `model.classes_` ระดับป่าเสมอ |
| 6 | **`oob_score_` มีหรือไม่มีในโมเดล** | ผู้ใช้บางคนไม่ได้เปิด `oob_score=True` ตอนเทรน หากโค้ดเรียก `model.oob_score_` ตรงๆ จะเกิด `AttributeError` | ใช้ `getattr(model, "oob_score_", None)` เพื่อแสดงใน HUD เฉพาะเมื่อมีค่าเท่านั้น |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `RandomForestAdapter` รองรับทั้ง `RandomForestClassifier` และ `RandomForestRegressor` พร้อมดึงผลโหวตรายต้นผ่าน `model.apply()` + Lookup Table
- [ ] คำนวณ `consensus_ratio`, `inter_tree_mean`, และ `inter_tree_std` จากต้นไม้ทุกต้นในป่า และส่งออกรายละเอียดรายต้นสูงสุด 100 ต้นแรก
- [ ] ฉาก 3 มิติแสดง **3D Forest Voting Ring** ด้วย `InstancedMesh` (1 Draw Call) ที่เปลี่ยนความสูงและสีตามมติของต้นไม้แต่ละต้นในเฟรมปัจจุบัน
- [ ] ผู้ใช้สามารถคลิกเลือกต้นไม้ต้นที่ $k$ ใน HUD (`Tree Selector`) เพื่อสลับดูพื้นผิวขั้นบันไดและเส้นทางกิ่งเรืองแสงเฉพาะของต้นไม้นั้นได้ทันที
- [ ] ไม่มี `UserWarning` เรื่อง `feature_names_in_` หลุดออกมาใน Log และความเร็วในการสกัดข้อมูล 1,000 แถว (100 ต้น) เร็วกว่า 500 มิลลิวินาที
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุม Binary, Multiclass, Regression และกรณี `n_estimators > 100`
