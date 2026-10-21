# Phase 27: Decision Tree Adapter, `decision_path` Tracer & 3D Stepped Voxel Terraces

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter สำหรับโมเดลตระกูลต้นไม้ตัดสินใจเดี่ยว (Single Decision Trees) เพื่อถอดรหัสโครงสร้างการแบ่งพื้นที่แบบ Orthogonal Axis-Aligned Splits ออกมาเป็นภูมิประเทศ **3D Stepped Voxel Terraces (ภูเขาขั้นบันได 3 มิติ)** พร้อมระบบแกะรอยเส้นทางการตัดสินใจ (`decision_path`) และคำนวณค่าอิทธิพลของแต่ละโหนดแบบ Exact Tree Path Contribution ในทุกเฟรมของ VCR Playback

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `sklearn.tree.DecisionTreeClassifier`
  - `sklearn.tree.DecisionTreeRegressor`
  - `sklearn.tree.ExtraTreeClassifier` (Single Extremely Randomized Tree)
  - `sklearn.tree.ExtraTreeRegressor`
  - โมเดลที่ถูกห่ออยู่ใน `sklearn.pipeline.Pipeline` หรือ `CalibratedClassifierCV` (ดึง Estimator ชั้นในสุด)
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.pkl`, `.pickle`, `.joblib` (ผ่าน Sandboxed Loader ใน `infrastructure/loaders/pickle_loader.py`)
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: แสดงความน่าจะเป็นของคลาสเป้าหมาย $P(y=1 \mid \mathbf{x}) \in [0, 1]$ เป็นความสูงของแท่งขั้นบันไดแกน $Z$
  - **Multiclass Classification**: รองรับ $C \ge 3$ คลาส โดยแสดงความสูงตามความน่าจะเป็นของคลาสที่เลือกดูหรือคลาสที่ชนะ ($\max_c P(y=c \mid \mathbf{x})$) พร้อมระบายสีพื้นผิวตาม `predicted_class`
  - **Regression**: แสดงค่าทำนายต่อเนื่อง $\hat{y}(\mathbf{x}) \in \mathbb{R}$ เป็นความสูงของขั้นบันได พร้อม Normalization เข้าสู่สเกลของฉาก 3 มิติ

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 สมการการแบ่งพื้นที่แบบ Axis-Aligned Orthogonal Splits
ต้นไม้ตัดสินใจแบ่งปริภูมิฟีเจอร์ $\mathbb{R}^d$ ออกเป็นกล่องสี่เหลี่ยมมุมฉากที่ไม่ทับซ้อนกัน (Disjoint Hyper-rectangles) $R_1, R_2, \dots, R_L$ โดยที่แต่ละโหนดภายใน $m$ จะตรวจสอบเงื่อนไขตั้งฉากกับแกนฟีเจอร์ $f_m$ ด้วยค่าขีดแบ่ง $t_m$:
$$\text{Go Left}(m): x_{f_m} \le t_m \quad \Longrightarrow \quad \text{child}_{\text{left}}(m), \qquad \text{Go Right}(m): x_{f_m} > t_m \quad \Longrightarrow \quad \text{child}_{\text{right}}(m)$$
ฟังก์ชันทำนายของต้นไม้จึงอยู่ในรูปผลรวมของฟังก์ชันบ่งชี้ (Indicator Function) บนใบไม้ $l \in \mathcal{L}$:
$$\hat{y}(\mathbf{x}) = \sum_{l \in \mathcal{L}} v_l \cdot \mathbb{I}(\mathbf{x} \in R_l)$$
ซึ่งทำให้พื้นผิว 3 มิติบนระนาบ 2 ฟีเจอร์ใดๆ มีลักษณะเป็น **ขั้นบันไดแนวราบสลับหน้าผาตั้งฉาก (Piecewise Constant Stepped Surface)**

### 2.2 การแกะโครงสร้างต้นไม้จาก `model.tree_` (Cython `Tree` Struct)
อ็อบเจกต์ `model.tree_` ใน Scikit-learn เก็บอาเรย์ C-level ขนาดเท่ากับ `node_count` ($N_{\text{nodes}}$):
- `children_left`: `ndarray` ขนาด `(N_nodes,)` เก็บดัชนีโหนดลูกซ้าย (หากเป็นใบไม้จะมีค่าเท่ากับ `TREE_LEAF = -1`)
- `children_right`: `ndarray` ขนาด `(N_nodes,)` เก็บดัชนีโหนดลูกขวา (`-1` สำหรับใบไม้)
- `feature`: `ndarray` ขนาด `(N_nodes,)` เก็บดัชนีฟีเจอร์ที่ใช้แบ่ง (`-2` สำหรับใบไม้)
- `threshold`: `ndarray` ขนาด `(N_nodes,)` เก็บค่าเกณฑ์ $t_m$ (`-2.0` สำหรับใบไม้)
- `impurity`: `ndarray` ขนาด `(N_nodes,)` เก็บค่า Gini, Entropy หรือ MSE ณ โหนดนั้น
- `n_node_samples`: `ndarray` ขนาด `(N_nodes,)` และ `weighted_n_node_samples`
- `value`: `ndarray` ขนาด `(N_nodes, n_outputs, max_n_classes)`:
  - **Regression**: `value[:, 0, 0]` คือค่าเฉลี่ย $\bar{y}_m$ ณ โหนด $m$ (Shape: `(N_nodes, 1, 1)`)
  - **Binary Classification**: `value[:, 0, :]` เก็บจำนวนตัวอย่างสะสม `[count_class_0, count_class_1]` (Shape: `(N_nodes, 1, 2)`) คำนวณความน่าจะเป็นโหนด $m$ ได้จาก $p_m = \text{value}[m, 0, 1] / \sum_c \text{value}[m, 0, c]$
  - **Multiclass Classification**: `value[:, 0, :]` มี Shape `(N_nodes, 1, C)` คำนวณเวกเตอร์ความน่าจะเป็น $\mathbf{p}_m \in \mathbb{R}^C$ ได้จากการหารด้วยผลรวมแถว

### 2.3 การดึงเส้นทางและใบไม้ (`decision_path` & `apply`)
- **การดึง Leaf Index**: เรียก `leaf_ids = model.apply(X_chunk)` ได้อาเรย์ `int64` ขนาด `(n_samples,)` ที่ระบุหมายเลขโหนดใบไม้สุดท้ายของแต่ละแถว
- **การดึง Decision Path**: เรียก `csr = model.decision_path(X_chunk)` ได้ `scipy.sparse.csr_matrix` ขนาด `(n_samples, N_nodes)`
  - สำหรับแถวที่ $i$: รายการโหนดที่วิ่งผ่านเรียงจาก Root ($0$) ไปจนถึง Leaf คือ `path_i = csr.indices[csr.indptr[i] : csr.indptr[i+1]]`

### 2.4 การคำนวณ Exact Tree Path Feature Contribution (Per-Node Delta)
สำหรับต้นไม้เดี่ยว เราสามารถคำนวณค่าอิทธิพลของแต่ละฟีเจอร์ ($\phi_j$) ตามเส้นทางจริงจาก Root ลงสู่ Leaf $L$:
- กำหนดให้ค่าคาดหวัง ณ โหนด $m$ คือ $E_m$ (สำหรับ Regression คือ `value[m, 0, 0]`, สำหรับ Classification คือความน่าจะเป็นของคลาสเป้าหมาย ณ โหนด $m$)
- เมื่อข้อมูลไหลจากโหนดพ่อ $u$ ไปยังโหนดลูก $v$ ซึ่งแบ่งด้วยฟีเจอร์ $f_u$:
  $$\Delta_{u \to v} = E_v - E_u$$
- สะสมค่า $\Delta_{u \to v}$ เข้าไปในฟีเจอร์ $f_u$ ตลอดเส้นทาง `path_i` จะได้เวกเตอร์อิทธิพลฟีเจอร์ที่滿足คุณสมบัติ Local Accuracy ทุกประการ:
  $$E_{\text{root}} + \sum_{j=1}^{d} \phi_j = E_{\text{leaf}} = \hat{y}(\mathbf{x})$$
- **Array Shape ของ Contributions**:
  - **Regression / Binary**: `(n_samples, n_features)` พร้อม `base_value` ขนาด `(1,)`
  - **Multiclass**: `(n_samples, n_classes, n_features)` พร้อม `base_value` ขนาด `(n_classes,)`

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Stepped Voxel Terraces (`SteppedVoxelTerrain.tsx`)**:
   - สร้างพื้นผิวขั้นบันไดความละเอียด $80 \times 80$ เซลล์ โดยใช้ Flat-shaded Box/Extruded Voxel Geometry แทนการเกลี่ยสีแบบ Smooth Interpolation เพื่อให้เห็นขอบหน้าผาตั้งฉากชัดเจน
   - รองรับโหมดสลับระหว่าง **Solid Stepped Blocks (บล็อกทึบสไตล์ Voxel)** และ **Terrace Contour Wireframe (เส้นโครงขั้นบันไดโปร่งแสง)**
2. **Orthogonal Split Threshold Lines (`SplitBoundaryLines.tsx`)**:
   - วาดเส้นเรืองแสงแนวตั้งและแนวนอนบนพื้นผิวขั้นบันไดตรงตำแหน่ง $x_{f_m} = t_m$ ของโหนดที่เกี่ยวข้องกับแกน $X, Y$ ปัจจุบัน
   - เส้นแบ่งของโหนดระดับบน (Depth 1–2) จะมีความหนาและความสว่างสูงกว่าโหนดระดับลึก
3. **Active Tree Branch Glow (`ActiveTreeBranchOverlay.tsx`)**:
   - แสดงโครงสร้างต้นไม้ 3 มิติแบบ Floating Hierarchy เหนือฉาก (หรือในมุม HUD 3D Inset) โดยในแต่ละเฟรม เส้นเชื่อมระหว่างโหนดใน `active_path` จะมีพัลส์แสง (Neon Pulse) วิ่งจาก Root ลงมายัง Leaf ที่ลูกบอลข้อมูลยืนอยู่
4. **Active Leaf Highlight Platform**:
   - แผ่นพื้นที่สอดคล้องกับ Hyper-rectangle ของ `leaf_id` ปัจจุบันจะเรืองแสงขอบสีทอง/ฟ้าสว่างขึ้น เพื่อแสดงขอบเขตของใบไม้ที่แถวปัจจุบันตกลงไป

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`DecisionTreeHudPanel.tsx`)
- **Depth Clipping Slider (ตัวกรองความลึกของกิ่งไม้)**: เลื่อนดูโครงสร้างต้นไม้ตั้งแต่ความลึกที่ $1$ ถึง $\text{max\_depth}$ เพื่อศึกษาว่าหากตัดกิ่ง (Pruning) ที่ความลึกต่างๆ ภูมิประเทศ 3D จะเปลี่ยนไปอย่างไร
- **Active Path Rule Breadcrumb**: แถบแสดงกฎ If-Else ของแถวปัจจุบัน เช่น `[Root: Gini=0.48] ➔ [petal_length <= 2.45 (True, Δ=+0.38)] ➔ [Leaf #1: Class 0 (Prob=1.00, N=50)]`
- **Leaf Impurity & Sample Counter Badge**: แสดงค่า Impurity (Gini/Entropy/MSE) และสัดส่วนจำนวนตัวอย่างเมื่อตอนเทรน (`n_node_samples`) ที่อยู่ในใบไม้เดียวกันกับข้อมูลแถวปัจจุบัน

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. เมื่อผู้ใช้กด Play หรือเลื่อน Scrubber ไปยังเฟรมที่ $t$:
   - ลูกบอลข้อมูล (`DataPointActor`) เคลื่อนที่ไปยังพิกัด $(x_t, y_t, \hat{y}_t)$ บนแท่งขั้นบันได
   - แผง Active Path Breadcrumb อัปเดตเงื่อนไข If-Else ของแถวที่ $t$ ทันที พร้อมไฮไลต์กิ่งที่เรืองแสง
2. เมื่อผู้ใช้เปิดโหมด **What-If Sandbox** แล้วลากค่าฟีเจอร์แกน $X$ ค่อยๆ เพิ่มขึ้น:
   - ขณะที่ยังอยู่ในช่วง $x \le t_m$ ลูกบอลจะเคลื่อนที่ในแนวราบบนแท่งขั้นบันไดเดิม (ค่า $\hat{y}$ ไม่เปลี่ยนเลย)
   - ทันทีที่ลากข้ามเส้นเรืองแสง $x > t_m$ ลูกบอลจะ **"กระโดดข้ามหน้าผาขั้นบันได"** ขึ้นหรือลงไปยังระดับความสูงของใบไม้ใหม่อย่างฉับพลัน พร้อมเปลี่ยนเส้นทางกิ่งเรืองแสงในแผนผังต้นไม้แบบ Real-time!

### 4.2 ASCII Wireframe ของหน้าจอ Decision Tree HUD & 3D Voxel Stage
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: DecisionTreeClassifier (max_depth=6, leaves=28)    [ depth filter: 4/6 ] |
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [ACTIVE DECISION PATH (t=42)]|
|                     3D STEPPED VOXEL TERRACES                     |                              |
|                                                                   |      (( Node #0: Root ))     |
|                  +---------------+                                |        [Income <= 52.5k]     |
|                  | Leaf #14      |  <-- Split Line (Age = 41.0)   |          /           \       |
|                  | P(y=1) = 0.92 |                                |     (True)           (False) |
|          +-------+---------------+-------+                        |       /                 \    |
|          | Leaf #9       |      (o)      | <-- Active Ball (t=42) |  ((Node #1))         (Node#2)|
|          | P(y=1) = 0.65 |  Leaf #15     |     [Jumps cliff when  | [Age <= 41.0]          ...   |
|  +-------+---------------+  P(y=1) = 0.45|      crossing split!]  |    /      \                  |
|  | Leaf #4               +---------------+                        | (False)  (True)              |
|  | P(y=1) = 0.12                         |                        |   /                          |
|  +---------------------------------------+                        | [[ LEAF #15: P=0.45 ]]       |
|   X: Income (k$) ------------------------>                        | Impurity (Gini): 0.24        |
|                                                                   | Train Samples in Leaf: 38    |
+-------------------------------------------------------------------+------------------------------+
| [NODE DELTAS] Base: 0.50 | Income<=52.5k: -0.18 | Age>41.0: +0.13 ===> Final Pred: 0.45          |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0042 / 1000  [===========(*)-----------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/trees/decision_tree.py` (`DecisionTreeAdapter`)
- `frontend/src/domains/stage3d/actors/SteppedVoxelTerrain.tsx`
- `frontend/src/domains/stage3d/actors/ActiveTreeBranchOverlay.tsx`
- `frontend/src/domains/hud/components/DecisionTreeHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `DecisionTreeAdapter`
```python
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.tree import BaseDecisionTree
from domains.Precomputation.adapters.base import BaseModelAdapter, InternalFlowBatch

class DecisionTreeAdapter(BaseModelAdapter):
    """Adapter สำหรับ DecisionTreeClassifier/Regressor และ ExtraTreeClassifier/Regressor."""

    def __init__(self, model: BaseDecisionTree, feature_names: list[str], target_class_idx: int = 1) -> None:
        super().__init__(model, feature_names)
        self.tree_ = model.tree_
        self.is_classifier = hasattr(model, "predict_proba")
        self.target_class_idx = target_class_idx

    def _compute_node_expectations(self) -> np.ndarray:
        """คำนวณค่าคาดหวัง E_m ของทุกโหนดในต้นไม้ล่วงหน้า (Precomputed Vector)."""
        vals = self.tree_.value  # shape: (n_nodes, 1, max_n_classes)
        if not self.is_classifier:
            return vals[:, 0, 0].astype(np.float32)
        totals = np.sum(vals[:, 0, :], axis=1, keepdims=True)
        probs = np.divide(vals[:, 0, :], totals, out=np.zeros_like(vals[:, 0, :]), where=totals > 0)
        idx = min(self.target_class_idx, probs.shape[1] - 1)
        return probs[:, idx].astype(np.float32)

    def extract_tree_topology(self, max_export_nodes: int = 512) -> dict[str, Any]:
        """สกัดโครงสร้างต้นไม้แบบ BFS จำกัดจำนวนโหนดสูงสุดเพื่อป้องกันไฟล์ .mlv บวม."""
        node_exp = self._compute_node_expectations()
        n_export = min(int(self.tree_.node_count), max_export_nodes)
        return {
            "node_count": int(self.tree_.node_count),
            "max_depth": int(self.tree_.max_depth),
            "children_left": self.tree_.children_left[:n_export].tolist(),
            "children_right": self.tree_.children_right[:n_export].tolist(),
            "feature": self.tree_.feature[:n_export].tolist(),
            "threshold": np.round(self.tree_.threshold[:n_export], 5).tolist(),
            "impurity": np.round(self.tree_.impurity[:n_export], 4).tolist(),
            "n_node_samples": self.tree_.n_node_samples[:n_export].tolist(),
            "node_expectation": np.round(node_exp[:n_export], 5).tolist(),
        }

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        """สกัด decision_path, leaf_id และคำนวณ Exact Tree Path Feature Contributions."""
        X_f32 = np.asarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        node_exp = self._compute_node_expectations()

        csr: csr_matrix = self.model.decision_path(X_f32)
        leaf_ids: np.ndarray = self.model.apply(X_f32)
        features_arr = self.tree_.feature
        thresholds_arr = self.tree_.threshold

        contribs = np.zeros((n_samples, n_features), dtype=np.float32)
        active_paths: list[list[dict[str, Any]]] = []

        for i in range(n_samples):
            start, end = csr.indptr[i], csr.indptr[i + 1]
            path_nodes = csr.indices[start:end]
            sample_steps: list[dict[str, Any]] = []
            for step_idx in range(len(path_nodes) - 1):
                u = int(path_nodes[step_idx])
                v = int(path_nodes[step_idx + 1])
                feat_idx = int(features_arr[u])
                delta = float(node_exp[v] - node_exp[u])
                if feat_idx >= 0:
                    contribs[i, feat_idx] += delta
                if step_idx < 32:  # จำกัดความลึกของเส้นทางที่เก็บลงเฟรมไม่เกิน 32 ขั้น
                    sample_steps.append({
                        "node_id": u,
                        "next_node_id": v,
                        "feature_idx": feat_idx,
                        "threshold": round(float(thresholds_arr[u]), 4),
                        "went_left": bool(X_f32[i, feat_idx] <= thresholds_arr[u]),
                        "delta": round(delta, 5),
                    })
            active_paths.append(sample_steps)

        return InternalFlowBatch(
            base_value=float(node_exp[0]),
            feature_contributions=contribs,
            extra_telemetry={
                "leaf_ids": leaf_ids.astype(np.int32).tolist(),
                "active_paths": active_paths,
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 42,
  "prediction": 0.45,
  "base_value": 0.50,
  "feature_contributions": [-0.18, 0.13, 0.0, 0.0],
  "telemetry": {
    "model_family": "decision_tree",
    "leaf_id": 15,
    "leaf_impurity": 0.24,
    "leaf_n_samples": 38,
    "active_path": [
      {"node_id": 0, "next_node_id": 1, "feature_idx": 0, "threshold": 52.5, "went_left": true, "delta": -0.18},
      {"node_id": 1, "next_node_id": 15, "feature_idx": 1, "threshold": 41.0, "went_left": false, "delta": 0.13}
    ]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การใช้ CSR Sparse Matrix สำหรับ `decision_path`**:
   - ห้ามเรียก `.toarray()` บนผลลัพธ์ของ `model.decision_path(X_chunk)` โดยเด็ดขาด เพราะถ้าต้นไม้มี $50,000$ โหนด และ `X_chunk` มี $2,000$ แถว Dense Matrix จะกินหน่วยความจำถึง $2,000 \times 50,000 \times 8 \text{ bytes} \approx 800 \text{ MB}$ ต่อ Chunk!
   - ให้เข้าถึงผ่าน `csr.indices[csr.indptr[i] : csr.indptr[i+1]]` เท่านั้น ซึ่งใช้ RAM น้อยกว่า $1 \text{ MB}$ เสมอ
2. **การส่ง `np.ndarray` (`float32`) แทน `pd.DataFrame`**:
   - แปลง `X_chunk` เป็น Contiguous `np.float32` ก่อนส่งเข้า `decision_path` และ `apply` เพื่อลด Overhead การตรวจสอบชื่อคอลัมน์ของ Pandas และทำให้ Cython Kernel ของ Scikit-learn ทำงานเร็วขึ้นกว่า $8\text{--}12\times$
3. **การจำกัดขนาด Topology ของต้นไม้ในไฟล์ `.mlv`**:
   - ต้นไม้ที่ไม่ได้จำกัด `max_depth` อาจมีโหนดมากกว่า $20,000$ โหนด ในส่วน Header ของไฟล์ `.mlv` ให้บันทึกโครงสร้างโหนดแบบ BFS สูงสุดไม่เกิน `max_export_nodes = 512` โหนดแรก พร้อมทั้งเก็บเฉพาะ `active_path` ของแต่ละเฟรม (ตัดที่ความลึกไม่เกิน 32 ขั้นแรกที่มี `|delta|` สูงสุด)
4. **การจัดการ `InstancedMesh` ใน WebGL (`SteppedVoxelTerrain.tsx`)**:
   - พื้นผิวขั้นบันไดขนาด $80 \times 80 = 6,400$ แท่ง ต้องวาดด้วย `THREE.InstancedMesh` เพียง 1 ตัว (1 Draw Call) โดยอัปเดตเฉพาะ `instanceMatrix` และ `instanceColor` เมื่อผู้ใช้เปลี่ยนฟีเจอร์แกน $X, Y$ หรือเลื่อน Depth Slider

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **ต้นไม้ลึกมาก (`max_depth > 25`, โหนดหลักหมื่น)** | การวาดแผนผังต้นไม้ทั้งหมดบนหน้าจอจะทำให้ DOM/WebGL ค้าง และไฟล์ `.mlv` ใหญ่เกินจำเป็น | เรนเดอร์เฉพาะโหนดที่อยู่ใน `active_path` ของแถวปัจจุบัน พร้อมโหนดลูกพี่ลูกน้อง (Sibling Nodes) ในแต่ละชั้น รวมไม่เกิน 30–40 โหนดบนจอ |
| 2 | **ต้นไม้มีโหนดเดียว (Stump ที่ไม่แตกเลย `node_count == 1`)** | โมเดลที่ถูกตั้ง `max_depth=0` หรือข้อมูลทุกแถวเหมือนกันหมด จะมีแค่ Root ที่เป็น Leaf ในตัว (`feature[0] == -2`) | ตรวจสอบ `len(path_nodes) == 1` ในลูป `trace_internal_flow` ให้คืนค่า `active_path = []` และ `contribs = 0` โดยไม่เกิด IndexError |
| 3 | **Multiclass Classification ที่บางคลาสไม่มีตัวอย่างตอนเทรน** | ขนาด `tree_.value` จะอ้างอิงตาม `model.n_classes_` ซึ่งอาจไม่ตรงกับดัชนีคลาสที่ผู้ใช้ระบุ | ใช้ `min(self.target_class_idx, probs.shape[1] - 1)` และแมปชื่อคลาสผ่าน `model.classes_` เสมอ |
| 4 | **Multi-output Tree (`n_outputs_ > 1`)** | `tree_.value` จะมีมิติที่สองมากกว่า 1 (`shape: (n_nodes, n_outputs, max_n_classes)`) | ตรวจสอบ `self.tree_.n_outputs > 1` แล้วล็อกดัชนี Output เป้าหมายที่ `output_idx = 0` (หรือตามที่ผู้ใช้เลือกใน UI) พร้อมแจ้งเตือนใน Metadata |
| 5 | **Scikit-learn >= 1.3 รองรับค่า `NaN` ใน Decision Tree (`missing_go_to_left`)** | เมื่อข้อมูลแถวที่ $i$ มีค่า `NaN` การเปรียบเทียบ `X_f32[i, feat_idx] <= thresholds_arr[u]` จะได้ `False` เสมอ แม้จริงแล้วต้นไม้อาจส่งไปทางซ้าย | เช็กทิศทางจริงจากการเปรียบเทียบ `v == int(self.tree_.children_left[u])` แทนการใช้เครื่องหมาย `<=` บนค่าฟีเจอร์โดยตรง |
| 6 | **`feature_importances_` เป็นศูนย์ทั้งหมด** | ต้นไม้ที่มีโหนดเดียวหรือแบ่งด้วย Impurity ลดลงน้อยมาก อาจทำให้ `feature_importances_` เป็น `0` ทุกคอลัมน์ | หาก `np.sum(model.feature_importances_) == 0` ให้ Fallback ไปเลือก 2 ฟีเจอร์แรกที่มีความแปรปรวน (`np.nanvar`) สูงสุดเป็นแกน $X, Y$ เริ่มต้น |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `DecisionTreeAdapter` รองรับทั้ง `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, และ `ExtraTreeRegressor` ครบถ้วน
- [ ] สกัด `decision_path` ด้วย CSR Sparse Slice โดยไม่มีการใช้ `.toarray()` และผ่านการทดสอบกับต้นไม้ที่มีโหนดมากกว่า 10,000 โหนดโดยใช้ RAM ไม่เกิน 50 MB
- [ ] คำนวณ Exact Tree Path Feature Contributions ($\Delta_{u \to v}$) ได้ถูกต้อง โดยผลรวม `base_value + sum(contribs)` เท่ากับค่าทำนายของใบไม้ (`atol=1e-5`)
- [ ] ฉาก 3 มิติแสดง **3D Stepped Voxel Terraces** ด้วย `InstancedMesh` พร้อมเส้นเรืองแสง **Split Threshold Lines** และ **Active Tree Branch Glow** ที่วิ่งตามเฟรมปัจจุบันที่ 60 FPS
- [ ] โหมด What-If Sandbox แสดงพฤติกรรมลูกบอลกระโดดข้ามหน้าผาขั้นบันไดทันทีที่ลากค่าฟีเจอร์ข้าม `threshold`
- [ ] มี Unit Tests และ Integration Tests ครอบคลุมทั้ง Binary, Multiclass, Regression, Stump (`node_count=1`), และข้อมูลที่มีค่า `NaN`
