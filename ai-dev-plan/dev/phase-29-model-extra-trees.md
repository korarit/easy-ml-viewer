# Phase 29: Extra Trees Adapter & Randomized Split Variance 3D Visualizer

---

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)

พัฒนา Adapter สำหรับโมเดลตระกูล **Extra Trees (Extremely Randomized Trees)** ซึ่งแตกต่างจาก Random Forest ตรงที่ไม่ได้ค้นหาจุดตัดที่ดีที่สุด (Optimal Split Threshold) แต่ใช้การ **สุ่มค่าจุดตัด (Random Cut-points)** โดยตรงแล้วเลือกตัวที่ดีที่สุดจากกลุ่มที่สุ่มได้ ทำให้ความแปรปรวนระหว่างต้นไม้ (Inter-Tree Split Variance) สะท้อนถึงความไม่แน่นอนเชิงพื้นที่ (Epistemic Uncertainty) ได้อย่างยอดเยี่ยม จึงนำมาแสดงผลเป็น **3D Uncertainty Halo (วงรัศมีหมอกความไม่แน่นอนรอบลูกบอลข้อมูล)** และแผนผังการกระจายตัวของจุดตัดสุ่ม

### ขอบเขตคลาสของโมเดลและไฟล์ที่รองรับ
- **คลาสของโมเดลที่รองรับ (Supported Classes)**:
  - `sklearn.ensemble.ExtraTreesClassifier`
  - `sklearn.ensemble.ExtraTreesRegressor`
  - โมเดล Extra Trees ที่ห่ออยู่ใน `Pipeline` หรือ `TransformedTargetRegressor`
- **นามสกุลไฟล์โมเดลที่รองรับ (Supported Extensions)**:
  - `.pkl`, `.pickle`, `.joblib`
- **ประเภทงานที่รองรับ (Supported Tasks)**:
  - **Binary Classification**: ความน่าจะเป็นเฉลี่ย $P(y=1 \mid \mathbf{x})$ พร้อมค่าความแปรปรวนของความน่าจะเป็นระหว่างต้นไม้ $\sigma_P^2(\mathbf{x})$
  - **Multiclass Classification**: เวกเตอร์ความน่าจะเป็นเฉลี่ย $C$ คลาส พร้อมค่า Predictive Entropy และ Mutual Information (Epistemic Uncertainty)
  - **Regression**: ค่าทำนายเฉลี่ย $\hat{y}(\mathbf{x})$ พร้อมส่วนเบี่ยงเบนมาตรฐานระหว่างต้นไม้ $\sigma_{\hat{y}}(\mathbf{x})$

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน (Mathematical Formulation & Internal Tree/Boosting Tracing)

### 2.1 คณิตศาสตร์ของการสุ่มจุดตัด (Extremely Randomized Cut-Points) และความไม่แน่นอน
ในโหนด $m$ สำหรับฟีเจอร์ $f_m$ ที่มีค่าต่ำสุดและสูงสุดในโหนดคือ $[x_{\min}, x_{\max}]$ โมเดล Extra Trees จะสุ่มจุดตัด $t_m \sim \text{Uniform}(x_{\min}, x_{\max})$:
- ส่งผลให้ขั้นบันไดของต้นไม้แต่ละต้น $T_1, \dots, T_K$ มีตำแหน่งขอบหน้าผา ($t_{k,m}$) เหลื่อมกันอย่างต่อเนื่อง เมื่อนำมาเฉลี่ยกัน:
  $$\hat{y}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^{K} T_k(\mathbf{x})$$
  พื้นผิว 3 มิติของ Extra Trees จึงมีความลาดเอียงที่นุ่มนวล (Piecewise Smooth-like Stepped Ramp) กว่า Random Forest ที่มีขอบหน้าผาซ้อนทับกันที่จุดเดิมซ้ำๆ
- **การแยกองค์ประกอบความไม่แน่นอน (Uncertainty Decomposition)**:
  - **Inter-Tree Standard Deviation (`ddof=0`)**:
    $$\sigma_{\text{tree}}(\mathbf{x}) = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \left(T_k(\mathbf{x}) - \hat{y}(\mathbf{x})\right)^2}$$
  - **Predictive Dispersion Index (Normalized Uncertainty Radius $r_{\text{halo}} \in [0, 1]$)**:
    - สำหรับ Classification: $r_{\text{halo}}(\mathbf{x}) = 2 \cdot \sigma_{\text{tree}}(\mathbf{x})$ (เนื่องจากค่าความน่าจะเป็นอยู่ใน $[0, 1]$ ค่า $\sigma_{\max} = 0.5$)
    - สำหรับ Regression: ปรับสเกลด้วยช่วงพิสัยของเป้าหมาย $\Delta Y = \max(\hat{Y}) - \min(\hat{Y}) + \epsilon$:
      $$r_{\text{halo}}(\mathbf{x}) = \text{clip}\left(\frac{2 \cdot \sigma_{\text{tree}}(\mathbf{x})}{\Delta Y}, 0.0, 1.0\right)$$

### 2.2 การแกะโครงสร้างจุดตัดสุ่ม (`Randomized Split Threshold Distribution`)
- นอกจากการดึง `leaf_matrix = model.apply(X_chunk)` และ `decision_path` เหมือนใน `BaseForestAdapter` แล้ว `ExtraTreesAdapter` ยังสกัด **การกระจายตัวของค่า `threshold` รายฟีเจอร์** จาก `est.tree_.threshold` ของทุกต้นไม้ในป่า
- ช่วยให้ผู้ใช้เห็นว่าในฟีเจอร์แกน $X$ และแกน $Y$ จุดตัดของ Extra Trees กระจายตัวครอบคลุมทั่วทั้งโดเมนอย่างไร (เทียบกับ Random Forest ที่กระจุกตัวเฉพาะจุดที่มี Information Gain สูง)

### 2.3 การคำนวณ Exact Tree Path Contributions และ Variance Attribution
- คำนวณค่าเฉลี่ยอิทธิพลฟีเจอร์ $\Phi_j(\mathbf{x}) = \frac{1}{K'} \sum_{k=1}^{K'} \phi_{k,j}(\mathbf{x})$ และคำนวณ **ความแปรปรวนของอิทธิพลรายฟีเจอร์** $\text{Std}(\phi_{\cdot, j}(\mathbf{x}))$ เพื่อระบุว่าฟีเจอร์ใดคือสาเหตุหลักที่ทำให้ต้นไม้ในป่ามีความเห็นไม่ตรงกันในเฟรมนั้น!
- **Array Shape ของ Contributions**:
  - `feature_contributions`: `(n_samples, n_features)` สำหรับ Binary/Regression หรือ `(n_samples, n_classes, n_features)` สำหรับ Multiclass
  - `contribution_std`: `(n_samples, n_features)` เก็บส่วนเบี่ยงเบนของแรงผลักแต่ละฟีเจอร์ระหว่างต้นไม้

---

## 3. UI Feature List & 3D Visual Specification (รายการฟีเจอร์ UI และรายละเอียด 3D Visual เฉพาะโมเดล)

### 3.1 องค์ประกอบในฉาก 3 มิติ (`stage3d`)
1. **3D Uncertainty Halo (`UncertaintyHaloActor.tsx`)**:
   - รอบลูกบอลข้อมูล (`DataPointActor`) มีทรงกลมหมอกเรืองแสงโปร่งแสง (Custom Fresnel Glow Shader) ที่ยืด-หดรัศมีตามค่า `uncertainty_radius` ($r_{\text{halo}}$) ของเฟรมปัจจุบัน
   - เมื่อต้นไม้ทุกต้นเห็นตรงกัน ($r_{\text{halo}} \to 0$) วงรัศมีจะหดรัดแน่นเป็นแกนเพชรคมชัดสีเขียวมรกต แต่เมื่อต้นไม้เสียงแตก ($r_{\text{halo}} > 0.4$) วงรัศมีจะขยายกว้างและสั่นไหวเบาๆ เป็นสีม่วง/ชมพูอมส้มเพื่อเตือนว่าพื้นที่นี้มีความไม่แน่นอนสูง
2. **Randomized Split Density Carpet (`SplitDensityCarpet.tsx`)**:
   - บนขอบแกน $X$ และแกน $Y$ ของพื้นเวที 3 มิติ แสดงแถบ Barcode / Rug Plot เรืองแสงที่พล็อตตำแหน่งจุดตัดสุ่ม ($t_{k,m}$) ของต้นไม้ในป่า เพื่อให้เห็นความหนาแน่นของการสุ่มแบ่งพื้นที่
3. **3D Forest Voting Ring & Stepped Terrain**:
   - สืบทอดวงแหวนเสาคริสตัล 100 ต้น (`ForestVotingArena.tsx`) และภูมิประเทศขั้นบันได (`SteppedVoxelTerrain.tsx`) จาก Phase 28 พร้อมเพิ่มโหมดระบายสีพื้นผิวตามค่า **Uncertainty Map ($\sigma_{\text{tree}}(x, y)$)**

### 3.2 แผงควบคุมเฉพาะโมเดลใน HUD (`ExtraTreesHudPanel.tsx`)
- **Surface Mode Switcher**: สลับการแสดงผลความสูง/สีของพื้นผิว 3D ระหว่าง `Mean Prediction Surface` กับ `Epistemic Uncertainty Surface (Inter-Tree Std)`
- **Feature Contribution Error Bars**: ในกราฟแท่งแสดงแรงผลักฟีเจอร์ของเฟรมปัจจุบัน จะมีหนวดกุ้ง (Error Bars $\pm 1\text{ SD}$) แสดงช่วงความเห็นที่แตกต่างกันของต้นไม้ในป่าต่อฟีเจอร์นั้นๆ
- **Tree Selector & Cut-point Inspector**: เลือกดูต้นไม้ต้นที่ $k$ เพื่อเปรียบเทียบจุดตัดสุ่มของต้นไม้นั้นกับค่าเฉลี่ยของป่า

---

## 4. UX Flow & การโต้ตอบในโหมด VCR Playback + What-If Sandbox

### 4.1 ลำดับเหตุการณ์เมื่อเล่น VCR Playback หรือลาก What-If
1. ขณะเล่น VCR Playback:
   - ลูกบอลข้อมูลวิ่งผ่านภูมิประเทศขั้นบันไดละเอียด โดยที่ **Uncertainty Halo** รอบลูกบอลจะวูบวาบขยายใหญ่ขึ้นเมื่อวิ่งผ่านแถวที่เป็น Outlier หรืออยู่ใกล้ขอบเขตการตัดสินใจ และหดเล็กลงเมื่ออยู่ในโซนที่ข้อมูลชัดเจน
2. เมื่ออยู่ในโหมด **What-If Sandbox** แล้วลากลูกบอลออกไปนอกขอบเขตข้อมูลเทรน (Out-of-Distribution / Extrapolation Zone):
   - เนื่องจาก Extra Trees สุ่มจุดตัดกระจายทั่วช่วง $[x_{\min}, x_{\max}]$ เมื่อลากลูกบอลเข้าใกล้ขอบโดเมน ผู้ใช้จะเห็นค่า `uncertainty_radius` และหนวดกุ้ง Error Bars ขยายตัวขึ้นทันที ช่วยให้วิศวกร ML วินิจฉัยจุดอ่อนของโมเดลในพื้นที่ข้อมูลเบาบางได้ด้วยตาเปล่า

### 4.2 ASCII Wireframe ของหน้าจอ Extra Trees HUD & Uncertainty Halo
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer] Model: ExtraTreesRegressor (n_estimators=150)     [ Terrain: Pred | Uncertainty]|
+-------------------------------------------------------------------+------------------------------+
|                                                                   | [UNCERTAINTY TELEMETRY (t=7)]|
|            3D RANDOMIZED TERRACES & UNCERTAINTY HALO              |                              |
|                                                                   | Mean Prediction  : 42.85     |
|               . . . . . . .                                       | Inter-Tree Std   : ± 6.42    |
|            .    ---------    .   <-- Glowing Uncertainty Halo     | Halo Radius      : 0.68 (HI) |
|          .    /           \    .     (Expands with High Std!)     | 95% Tree Range   : [30.1, 55]|
|         .    |     (o)     |    .                                 +------------------------------+
|          .    \           /    . <-- Data Point Ball              | [CONTRIBUTION + STD BARS]    |
|            .    ---------    .                                    | Base Value       : 35.00     |
|     +---------. . . . . . .---------+                             | Temp   : +9.20 [===|===] ±3.1|
|     | Fine-grained ExtraTrees Steps |                             | Humid  : -2.15 [==|==]   ±0.8|
|  +--+-------------------------------+--+                          | Wind   : +0.80 [=|]      ±0.2|
|  | ||| | |||| || | ||| | || |||| | | | | <-- Random Split Rug     |                              |
+-------------------------------------------------------------------+------------------------------+
| [TREE #k SELECTOR] < Tree #014 / 150 > | Tree #14 Pred: 48.10 | Active Depth: 5 | Leaf ID: #39   |
+--------------------------------------------------------------------------------------------------+
| [|<] [<] [ || PAUSE ] [>] [>|]  Frame: 0007 / 1000  [=(*)-------------------------] Speed: 1x  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบของ Adapter (DDD Adapter Blueprint & Code Signatures)

### 5.1 โครงสร้างไฟล์ใน Domain & Infrastructure
- `backend/src/domains/Precomputation/adapters/trees/extra_trees.py` (`ExtraTreesAdapter`)
- `frontend/src/domains/stage3d/actors/UncertaintyHaloActor.tsx`
- `frontend/src/domains/stage3d/actors/SplitDensityCarpet.tsx`
- `frontend/src/domains/hud/components/ExtraTreesHudPanel.tsx`

### 5.2 โค้ดต้นแบบ `ExtraTreesAdapter`
```python
from __future__ import annotations
from typing import Any
import numpy as np
from domains.Precomputation.adapters.base import InternalFlowBatch
from domains.Precomputation.adapters.trees.random_forest import BaseForestAdapter, MAX_EXPORTED_TREES

class ExtraTreesAdapter(BaseForestAdapter):
    """Adapter สำหรับ ExtraTreesClassifier และ ExtraTreesRegressor พร้อมคำนวณ Split Variance."""

    def extract_split_threshold_distribution(self, max_splits_per_feature: int = 200) -> dict[int, list[float]]:
        """สกัดการกระจายตัวของจุดตัดสุ่ม (Random Cut-points) ของแต่ละฟีเจอร์เพื่อวาด Rug Plot 3D."""
        splits_by_feat: dict[int, list[float]] = {i: [] for i in range(len(self.feature_names))}
        for est in self.estimators[:MAX_EXPORTED_TREES]:
            feats = est.tree_.feature
            thresholds = est.tree_.threshold
            for f_idx, th in zip(feats, thresholds):
                if f_idx >= 0 and len(splits_by_feat[int(f_idx)]) < max_splits_per_feature:
                    splits_by_feat[int(f_idx)].append(round(float(th), 4))
        return splits_by_feat

    def trace_internal_flow(self, X_chunk: np.ndarray) -> InternalFlowBatch:
        X_f32 = np.asarray(X_chunk, dtype=np.float32)
        n_samples, n_features = X_f32.shape
        tree_preds, leaf_matrix = self.predict_all_trees_fast(X_f32)

        k_export = min(self.n_estimators, MAX_EXPORTED_TREES)
        # ใช้ ddof=0 เสมอเพื่อป้องกัน NaN กรณี n_estimators == 1
        inter_tree_mean = np.mean(tree_preds, axis=1)
        inter_tree_std = np.std(tree_preds, axis=1, ddof=0)
        q05 = np.percentile(tree_preds, 5, axis=1)
        q95 = np.percentile(tree_preds, 95, axis=1)

        if self.is_classifier:
            uncertainty_radius = np.clip(2.0 * inter_tree_std, 0.0, 1.0)
        else:
            pred_span = float(np.ptp(inter_tree_mean)) + 1e-6
            uncertainty_radius = np.clip((2.0 * inter_tree_std) / pred_span, 0.0, 1.0)

        per_tree_contribs = np.zeros((k_export, n_samples, n_features), dtype=np.float32)
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
                        per_tree_contribs[k, i, f_idx] += exp_k[v] - exp_k[u]

        mean_contribs = np.mean(per_tree_contribs, axis=0)
        std_contribs = np.std(per_tree_contribs, axis=0, ddof=0)

        return InternalFlowBatch(
            base_value=base_val,
            feature_contributions=mean_contribs,
            extra_telemetry={
                "model_family": "extra_trees",
                "tree_predictions_sample": np.round(tree_preds[:, :k_export], 4).tolist(),
                "leaf_indices_sample": leaf_matrix[:, :k_export].astype(np.int32).tolist(),
                "inter_tree_mean": np.round(inter_tree_mean, 5).tolist(),
                "inter_tree_std": np.round(inter_tree_std, 5).tolist(),
                "uncertainty_radius": np.round(uncertainty_radius, 4).tolist(),
                "prediction_p05": np.round(q05, 4).tolist(),
                "prediction_p95": np.round(q95, 4).tolist(),
                "contribution_std": np.round(std_contribs, 4).tolist(),
            },
        )
```

### 5.3 JSON Schema ของ `InternalFlowBatch` (ต่อ 1 เฟรมใน `.mlv`)
```json
{
  "frame_index": 7,
  "prediction": 42.85,
  "base_value": 35.00,
  "feature_contributions": [9.20, -2.15, 0.80],
  "telemetry": {
    "model_family": "extra_trees",
    "inter_tree_mean": 42.85,
    "inter_tree_std": 6.42,
    "uncertainty_radius": 0.68,
    "prediction_p05": 30.12,
    "prediction_p95": 55.04,
    "contribution_std": [3.10, 0.82, 0.21],
    "tree_predictions_sample": [48.10, 39.50, 44.20, 31.80],
    "leaf_indices_sample": [39, 18, 27, 12]
  }
}
```

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)

1. **การควบคุมขนาดอาเรย์ 3 มิติ `per_tree_contribs`**:
   - อาเรย์ `per_tree_contribs` มีขนาด `(k_export, n_samples, n_features)` ชนิด `float32` หาก `k_export = 100`, `n_samples = 2,000` (ต่อ Chunk), และ `n_features = 50` จะใช้ RAM เพียง $100 \times 2,000 \times 50 \times 4 \text{ bytes} = 40 \text{ MB}$ และถูกคืนหน่วยความจำทันทีที่จบ Chunk
   - ห้ามสร้างอาเรย์นี้กับข้อมูลทั้ง Dataset พร้อมกัน แต่ให้รันภายในลูป Chunk ขนาด $1,000\text{--}2,000$ แถวเสมอ
2. **การสุ่มเก็บ `split_threshold_distribution`**:
   - ต้นไม้ Extra Trees มักจะมีจำนวนโหนดมากกว่า Random Forest เพราะจุดตัดสุ่มทำให้ต้องแตกกิ่งลึกกว่าเดิมเพื่อลด Impurity ดังนั้นในการดึงจุดตัดไปแสดงเป็น Rug Plot บนขอบเวที 3D ให้เก็บไม่เกิน `max_splits_per_feature = 200` ค่าต่อฟีเจอร์
3. **การเรนเดอร์ `UncertaintyHaloActor.tsx` บน GPU**:
   - ใช้ `THREE.ShaderMaterial` แบบ Additive Blending บน Sphere Geometry ความละเอียดปานกลาง ($24 \times 24$ segments) ปรับขนาดด้วย `scale.setScalar(baseRadius + uncertaintyRadius * maxExpand)` ใน `useFrame` โดยไม่ต้องสร้าง Geometry ใหม่

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| # | กรณีศึกษา (Edge Case) | อาการหรือปัญหาที่อาจเกิดขึ้น | แนวทางป้องกันและแก้ไขในโค้ด (Mitigation) |
| :-: | :--- | :--- | :--- |
| 1 | **ป่าที่มีต้นไม้เพียงต้นเดียว (`n_estimators == 1`)** | หากคำนวณ `np.std(..., ddof=1)` จะเกิดการหารด้วยศูนย์และคืนค่า `NaN` ทำให้ทรงกลม Uncertainty Halo พังใน WebGL | บังคับใช้ `np.std(..., ddof=0)` เสมอ ซึ่งจะคืนค่า `0.0` อย่างปลอดภัยเมื่อ `n_estimators == 1` |
| 2 | **งาน Regression ที่ทุกแถวใน Chunk ทำนายได้ค่าเกือบเท่ากัน (`ptp == 0`)** | การหารด้วย `np.ptp(inter_tree_mean)` เพื่อคำนวณ `uncertainty_radius` อาจทำให้เกิดค่า `inf` หรือกระโดดผิดปกติ | บวกค่าคงที่กันเหนียว `pred_span = max(float(np.ptp(inter_tree_mean)), 1e-3)` และครอบด้วย `np.clip(..., 0.0, 1.0)` |
| 3 | **ต้นไม้ใน Extra Trees ลึกและใหญ่กว่า Random Forest มาก** | เนื่องจากไม่ได้ใช้ Bootstrap (`bootstrap=False` เป็นค่าเริ่มต้น) และสุ่มจุดตัด ต้นไม้ที่ไม่ได้จำกัด `max_depth` อาจมีโหนดมากกว่า Random Forest ถึง $2\text{--}3\times$ | ใช้ `model.apply(X_f32)` + Lookup Table และจำกัด `k_export = min(n_estimators, 100)` เพื่อรักษาความเร็วคงที่ |
| 4 | **`bootstrap=False` ทำให้ไม่มี `oob_score_` โดยค่าเริ่มต้น** | ใน Extra Trees ค่าเริ่มต้นคือ `bootstrap=False` จึงไม่มี Out-of-Bag Score | ตรวจสอบ `getattr(model, "bootstrap", False)` และ `getattr(model, "oob_score_", None)` ก่อนแสดงใน HUD |
| 5 | **ฟีเจอร์ที่เป็นค่าคงที่หรือ Binary (0/1)** | จุดตัดสุ่มของฟีเจอร์ Binary จะตกอยู่ในช่วง $(0, 1)$ ทั้งหมด แต่ให้ผลการแบ่งเหมือนเดิมทุกค่า | ใน `SplitDensityCarpet.tsx` ให้จัดกลุ่มจุดตัดลงใน 50 Histogram Bins เพื่อแสดงความหนาแน่นแทนการวาดเส้นทับกัน |
| 6 | **Multiclass Uncertainty Visualization** | ในกรณี $C \ge 3$ คลาส การดูแค่ `std` ของคลาสเดียวอาจไม่สะท้อนความไม่แน่นอนระหว่างคลาสที่ 2 และ 3 | คำนวณค่าเฉลี่ยของ `inter_tree_std` ของคลาสที่ชนะ ($\hat{c}$) ร่วมกับ Predictive Entropy ของความน่าจะเป็นเฉลี่ย |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)

- [ ] `ExtraTreesAdapter` สืบทอดจาก `BaseForestAdapter` และรองรับทั้ง `ExtraTreesClassifier` และ `ExtraTreesRegressor`
- [ ] คำนวณ `inter_tree_std`, `uncertainty_radius`, `prediction_p05/p95`, และ `contribution_std` โดยใช้ `ddof=0` ป้องกัน `NaN` 100%
- [ ] สกัด `split_threshold_distribution` เพื่อแสดงแถบความหนาแน่นของจุดตัดสุ่มบนขอบแกน $X, Y$ ในฉาก 3 มิติ
- [ ] ทรงกลมเรืองแสง **3D Uncertainty Halo** รอบลูกบอลข้อมูลขยาย/หดตัวและเปลี่ยนสีตามความขัดแย้งของต้นไม้ในป่าอย่างลื่นไหลที่ 60 FPS
- [ ] HUD แสดงหนวดกุ้ง Error Bars ($\pm 1\text{ SD}$) บนกราฟแรงผลักฟีเจอร์ พร้อมปุ่มสลับพื้นผิวระหว่าง `Mean Prediction` และ `Uncertainty Surface`
- [ ] มีชุดทดสอบ Unit & Integration Tests ครอบคลุมกรณี `n_estimators=1`, Binary, Multiclass, และ Regression
