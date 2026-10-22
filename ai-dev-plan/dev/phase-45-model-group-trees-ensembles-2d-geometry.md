# Phase 45: 2D Decision Geometries & Visualizers for Decision Trees & Bagging Ensembles (Decision Tree, Random Forest, Extra Trees, AdaBoost)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
พัฒนาชุดเรขาคณิตและการแสดงผล 2 มิติเฉพาะทางสำหรับกลุ่มโมเดล **ต้นไม้ตัดสินใจและกลุ่มรวมต้นไม้แบบแบ็กกิ้ง (Decision Trees & Bagging Ensembles)** ได้แก่ **Decision Tree**, **Random Forest**, **Extra Trees**, และ **AdaBoost** ซึ่งในโหมด 2 มิติจะทำให้ผู้ใช้เห็นธรรมชาติของการตัดแบ่งพื้นที่ของต้นไม้ได้อย่างเป็นรูปธรรมที่สุด:
- **เส้นตัดตั้งฉากตามแนวแกนพิกัด (2D Axis-Aligned Orthogonal Partitions)**: ที่แบ่งพื้นที่ระนาบ 2D ออกเป็นกล่องสี่เหลี่ยมผืนผ้า (Bounding Boxes) ตามกฎเงื่อนไขในแต่ละโหนด
- **การไฮไลต์กล่องใบไม้ของแถวปัจจุบัน (Active Leaf Bounding Box Glow)**: แสดงกล่องสี่เหลี่ยมที่ข้อมูลเฟรมปัจจุบันกำลังตกอยู่ พร้อมผลลัพธ์การทำนายของใบนั้น
- **แผนที่สีความเห็นพ้องของป่าทั้งป่า (Ensemble Vote Consensus Heatmap)**: การซ้อนทับของเส้นตัดจากต้นไม้หลายสิบต้นจนกลายเป็นขอบเขตขั้นบันไดที่นุ่มนวล
- **สไลเดอร์ดูวิวัฒนาการของ AdaBoost (Staged Weak Learner Cascade)**: ลากดูการดัดแปลงเส้นแบ่งเขตทีละสเต็ปของ Weak Learners

### ขอบเขตโมเดลและคลาสที่รองรับ (In-Scope Models)
1. **`DecisionTreeClassifier` & `DecisionTreeRegressor`**: ต้นไม้ตัดสินใจเดี่ยว
2. **`RandomForestClassifier` & `RandomForestRegressor`**: ป่าสุ่มแบบ Ensembles
3. **`ExtraTreesClassifier` & `ExtraTreesRegressor`**: ป่าต้นไม้แบบสุ่มจุดตัดอิสระ (Extremely Randomized Trees)
4. **`AdaBoostClassifier` & `AdaBoostRegressor`**: การบูสต์ต้นไม้แบบลูกโซ่

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน 2 มิติ (Mathematical Formulation & 2D Geometry)

### 2.1 เรขาคณิตการแบ่งพื้นที่ตั้งฉากของ Decision Tree (2D Orthogonal Partitions)
1. **โครงสร้างโหนดตัดและขอบเขตกล่อง (Node Splits & Bounding Boxes)**:
   ต้นไม้ตัดสินใจแยกข้อมูลด้วยเงื่อนไขแกนเดี่ยวเสมอ: $x_j \le \theta_k$ หรือ $x_j > \theta_k$
   - หากตัดบนแกน $X_1$: ก่อให้เกิดเส้นดิ่ง $x_1 = \theta_k$
   - หากตัดบนแกน $X_2$: ก่อให้เกิดเส้นนอน $x_2 = \theta_k$
   - หากตัดบนแกนอื่น ($X_k$ ที่ถูกตรึงไว้): จะส่งผลให้กิ่งบางกิ่งถูกตัดทิ้งสำหรับค่า Pin ปัจจุบัน
2. **การสกัดกล่องใบไม้ (Leaf Bounding Box Extraction via DFS)**:
   - เริ่มต้นที่ Root Node ด้วยขอบเขต $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$
   - เมื่อเดินลงโหนดซ้ายหรือขวา จะปรับปรุงขอบเขตตามค่า Threshold:
     $$\text{Left: } x_{\max} \leftarrow \min(x_{\max}, \theta_k), \quad \text{Right: } x_{\min} \leftarrow \max(x_{\min}, \theta_k)$$
   - เมื่อถึง Leaf Node แต่ละใบจะได้กล่องพิกัดสี่เหลี่ยมผืนผ้า $[\text{xmin}_l, \text{xmax}_l, \text{ymin}_l, \text{ymax}_l]$
3. **Active Leaf Highlight**:
   - แถวปัจจุบัน $t$ ตกอยู่ในโหนดใบไม้หมายเลข $L_t$ (จาก `model.apply(X_t)`):
   - ระบบจะเรนเดอร์กล่องสี่เหลี่ยมของใบ $L_t$ ด้วยเส้นขอบเรืองแสงสีฟ้า Cyan สว่างวาบ พร้อมป้ายลอยระบุ `Leaf #L (Samples: N, Pred: C, Impurity: G)`

### 2.2 เรขาคณิตของ Random Forest & Extra Trees Consensus
1. **การรวมคะแนนเสียงโหวตของต้นไม้ทั้งป่า (Ensemble Consensus Surface)**:
   สำหรับต้นไม้ย่อย $M$ ต้นใน `model.estimators_`:
   $$P_{\text{RF}}(y=1 | x_1, x_2) = \frac{1}{M} \sum_{m=1}^{M} h_m(x_1, x_2)$$
   - การซ้อนทับกันของเส้นตัดตั้งฉากนับร้อยเส้นทำให้เกิดลวดลายโมเสกสี่เหลี่ยมย่อยๆ (Tessellated Rectangular Mosaic)
   - สีของแต่ละเซลล์สะท้อนสัดส่วนเสียงโหวต: ยิ่งเขียวเข้มคือเสียงโหวตเป็นเอกฉันท์ ยิ่งเหลืองคือคะแนนสูสี 50-50
2. **Extra Trees Variance Band**:
   - ใน Extra Trees จุดตัด $\theta$ ถูกสุ่มอย่างอิสระ $\to$ พื้นผิวคะแนนโหวตจะมีความเรียบเนียนกว่า Random Forest ปกติ
   - คำนวณความแปรปรวนของคำตอบรายต้น: $\sigma^2(x_1, x_2) = \frac{1}{M} \sum_{m=1}^M (h_m - \bar{h})^2$ เพื่อแสดงบริเวณที่ต้นไม้แต่ละต้นทายขัดแย้งกัน

### 2.3 AdaBoost Staged Cascade Slider
1. **สมการผลรวมถ่วงน้ำหนักตามสเต็ป $T$**:
   $$F_T(x_1, x_2) = \sum_{t=1}^{T} \alpha_t h_t(x_1, x_2)$$
2. **Staged Boundary Playback**:
   - มีแถบสไลเดอร์ $T \in [1, \text{n\_estimators}]$ ให้ผู้ใช้ลากดูว่า Weak Learner ต้นแรกๆ เริ่มตัดเส้นตรงแบบหยาบๆ อย่างไร และต้นต่อๆ มาค่อยๆ ตัดเส้นแก้ข้อผิดพลาดจนขอบเขตครอบคลุมจุดยากได้อย่างไร

---

## 3. UI Feature List & 2D Overlays ของกลุ่มโมเดล Trees & Ensembles

```text
+--------------------------------------------------------------------------------------------------+
| 2D TREES & ENSEMBLES OVERLAYS MATRIX                                                             |
+----------------------+---------------------------------------------------------------------------+
| UI Component         | ความสามารถและองค์ประกอบที่แสดงในโหมด 2D Planar                             |
+----------------------+---------------------------------------------------------------------------+
| TreePartitions2D     | • เส้นประตัดตั้งฉากแนวนอนและแนวดิ่งตามกฎของต้นไม้                         |
|                      | • กล่องสี่เหลี่ยมเรืองแสง Active Leaf Box ของแถวปัจจุบัน                  |
|                      | • Tooltip ชี้ที่กล่องเพื่อดูกฎของใบไม้ เช่น "Age > 35 AND Income <= 50k"   |
+----------------------+---------------------------------------------------------------------------+
| EnsembleConsensus2D  | • แผนผังสีโมเสก Consensus Heatmap (% เสียงโหวตของต้นไม้ทั้งป่า)            |
|                      | • ตัวเลือกเปิด/ปิด: "[x] Superimpose Top-10 Tree Cut Lines"               |
+----------------------+---------------------------------------------------------------------------+
| AdaBoostCascade2D    | • แถบสไลเดอร์ "Weak Learner Stage: T"                                     |
|                      | • แอนิเมชันแสดงการดัดแปลงเส้นขอบเขตทีละขั้นตอน                             |
+----------------------+---------------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ 2D Trees Group

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: RandomForestClassifier (n_trees=100)          [● 2D Mode] [3D Mode] (V)|
+--------------------------------------------------------------------------------------------------+
| Y: Glucose Level (40 .. 200)                                     CONSENSUS: 84 / 100 Trees (84%) |
|   200 |              |                 |                                                         |
|       |   Zone 0     |   Zone 1 (95%)  |   Zone 1 (92%)                                          |
|   160 |   (Low Risk) |   (High Risk)   |   (High Risk)                                           |
|       |  ------------+-----------------+-------------------------- [Cut: Glucose = 140]          |
|   120 |              | +-------------+ |                                                         |
|       |   Zone 0     | | ACTIVE LEAF | |   Zone 0 (72%)                                          |
|    80 |   (88% Conf) | | ● Row #42   | |   (Moderate)                                            |
|       |              | +-------------+ |                                                         |
|    40 +--------------+-----------------+-------------------------> X: BMI (15 .. 50)             |
|       15             25                35                         50                             |
|                      [Cut: BMI = 25]   [Cut: BMI = 35]                                           |
|                                                                                                   |
| TREE CONTROLS: [x] Orthogonal Cut Lines   [x] Active Leaf Box   [x] Forest Consensus Heatmap     |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Signatures)

### 5.1 โครงสร้างไฟล์
- `frontend/src/domains/stage3d/components/surfaces_2d/TreePartitions2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/EnsembleConsensus2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/AdaBoostCascade2D.tsx`
- `backend/app/domains/model_engine/adapters/trees/tree_2d_extractor.py`

### 5.2 โค้ดต้นแบบ: `tree_2d_extractor.py`
```python
from typing import List, Dict, Any, Tuple
import numpy as np

class Tree2DExtractor:
    """สกัดกฎการตัดแบ่ง 2 มิติ และกล่อง Leaf Bounding Box ของต้นไม้ตัดสินใจ"""

    @staticmethod
    def extract_tree_partitions(
        tree_model: Any,
        x_idx: int,
        y_idx: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        tree_ = tree_model.tree_
        feature = tree_.feature
        threshold = tree_.threshold

        splits = []
        for node_idx in range(tree_.node_count):
            feat = feature[node_idx]
            if feat == x_idx:
                splits.append({"feature_index": x_idx, "threshold": float(threshold[node_idx]), "axis": "x"})
            elif feat == y_idx:
                splits.append({"feature_index": y_idx, "threshold": float(threshold[node_idx]), "axis": "y"})

        return {
            "splits": splits,
            "node_count": int(tree_.node_count),
            "leaf_count": int(tree_.n_leaves)
        }
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] เส้นตัดตั้งฉากของ Decision Tree แบ่งระนาบ $X-Y$ ถูกต้องตรงตามเงื่อนไขใน Node Split
- [ ] กล่อง Active Leaf Bounding Box เรืองแสงครอบคลุมตำแหน่งของจุดข้อมูลปัจจุบันได้อย่างแม่นยำ
- [ ] แผนผังเสียงโหวต Consensus Heatmap ของ Random Forest แสดงสัดส่วนคะแนนเสียงโหวตถูกต้องครบถ้วน
- [ ] สไลเดอร์ AdaBoost Cascade แสดงการดัดแปลงเส้นขอบเขตตามลำดับ Weak Learner ได้ลื่นไหลที่ **60 FPS**

