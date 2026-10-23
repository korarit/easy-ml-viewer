# Phase 53: 2D Point Intelligence Integration for Trees & Bagging Ensembles (Decision Trees, Random Forest, ExtraTrees, AdaBoost)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เชื่อมต่อระบบ **2D Point Intelligence (A, B, C, D)** เข้ากับกลุ่มโมเดลต้นไม้ตัดสินใจและกลุ่มรวมต้นไม้แบบแบ็กกิ้ง (**Decision Trees & Bagging Ensembles**) เพื่อเปิดเผยโครงสร้างการตัดขนานแกน (Axis-Aligned Splits), ความสอดคล้องของเสียงโหวตในป่าไม้ (Forest Voting Consensus & Variance), และการระบุจุดข้อมูลที่มีค่าน้ำหนักแฝงสูง (Sample Weights ใน AdaBoost) ในมุมมองระนาบ 2 มิติ

### ขอบเขตโมเดลที่รองรับในเฟสนี้
- **`DecisionTreeClassifier` & `DecisionTreeRegressor`**
- **`RandomForestClassifier` & `RandomForestRegressor`**
- **`ExtraTreesClassifier` & `ExtraTreesRegressor`**
- **`AdaBoostClassifier` & `AdaBoostRegressor`**

---

## 2. กลไกคณิตศาสตร์และการประยุกต์ใช้ Point Intelligence เชิงลึก (Mathematical & Geometric Integration)

### 2.1 กลไก A: Orthogonal Leaf-Hopping Trail (Comet Trail แบบก้าวบันได)
ต้นไม้ตัดสินใจแบ่งพื้นที่ข้อมูลด้วยเส้นตรงขนานแกน ($x_j \le \theta$):
- **Manhattan Step Path**: ในขณะที่จุด Active Point เคลื่อนที่ข้ามรอยต่อของกล่องใบไม้ (Leaf Bounding Boxes) เส้นหาง Comet Trail จะมีตัวเลือกในการแสดงผลเป็นเส้นตัดฉากแบบ Manhattan/Orthogonal Steps เพื่อเน้นย้ำจังหวะการเปลี่ยนกิ่ง (Branch Switching)
- **Leaf Transition Pulse**: เมื่อจุดข้อมูลก้าวข้ามเส้น Split Line กล่อง Leaf Box ปลายทางจะสว่างวาบขึ้นในระนาบ 2D

### 2.2 กลไก B: Forest Voting Variance Dual-Ring
1. **Random Forest Voting Consensus Ring**:
   - ในโมเดล Random Forest ที่มีต้นไม้ $T$ ต้น ให้ $v_{c}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\hat{y}_t = c)$ เป็นสัดส่วนเสียงโหวตของคลาส $c$:
   - คำนวณค่าความแปรปรวนของความเห็นพ้อง (Agreement Variance):
     $$\sigma_{\text{agreement}} = 1.0 - \max_c v_c(\mathbf{x})$$
   - วงแหวนรอบนอก (Outer Ring):
     - หากต้นไม้ทุกต้นเห็นตรงกัน ($\sigma = 0$): วงแหวนเป็นเส้นทึบบางเฉียบสีคมชัด
     - หากเกิดการโหวตแตกคอ เช่น 52/48 ($\sigma \approx 0.5$): วงแหวนจะหนาขึ้นเป็นสองเท่าและเป็นเส้นประคู่ (Double-Dashed Ring)
2. **AdaBoost Sample Weight Ring**:
   - ความหนาของวงแหวนรอบนอกแปรผันตรงกับค่าน้ำหนักของข้อมูล $\omega_i$:
     $$t_{\text{ring}} = t_{\text{base}} \cdot \sqrt{\frac{\omega_i}{\bar{\omega}}}$$
     จุดที่ AdaBoost ให้น้ำหนักสูง (จุดที่โมเดลก่อนหน้าทายผิดซ้ำๆ) จะมีวงแหวนขนาดใหญ่สะดุดตา

### 2.3 กลไก C: Leaf Impurity & Leaf-Center Residual Spokes
1. **Leaf Mixed-Sample Spokes**:
   - ภายในกล่อง Leaf ที่มีค่า Impurity (Gini $> 0$ หรือ Entropy $> 0$): จุดข้อมูลที่เป็นชนกลุ่มน้อย (Minority Class Samples) จะมีเส้นประลากชี้ไปยังจุดศูนย์กลางกล่อง Leaf เพื่อแสดงว่าจุดนี้เป็น Outlier ที่ตกค้างอยู่ในใบ
2. **Tree Stepwise Regression Residuals**:
   - ใน Decision Tree Regressor ค่าทำนายจะเป็นค่าคงที่ขั้นบันได $\hat{y}_{\text{leaf}}$ ภายในแต่ละกล่อง: ก้าน Residual Spoke จะลากจากจุดข้อมูลไปยังระนาบแนวนอนของความสูงประจำใบนั้น

### 2.4 กลไก D: Leaf-Partitioned Hexbin Aggregation
- ทำการแบ่งกลุ่ม Hexbin Grid ให้สอดคล้องกับเส้นขอบเขตของ Leaf กล่องสี่เหลี่ยมผืนผ้า ($[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$)
- แสดงสัดส่วนความบริสุทธิ์ของข้อมูล (Purity Score) ประจำแต่ละบล็อกรังผึ้งภายในกล่อง Leaf

---

## 3. สถาปัตยกรรมคอมโพเนนต์ (`frontend/src/domains/stage3d/components/surfaces_2d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D TREES & ENSEMBLES POINT INTELLIGENCE OVERLAYS                                                 |
+--------------------------+-----------------------------------------------------------------------+
| Component                | บทบาทเฉพาะตัวสำหรับ Trees & Ensemble Family                           |
+--------------------------+-----------------------------------------------------------------------+
| ForestConsensusRing2D    | วงแหวน Dual-Ring แสดงระดับความเห็นพ้องของต้นไม้ T ต้นใน Random Forest |
| AdaBoostWeightHalo2D     | วงแหวนขนาดแปรผันตามค่าน้ำหนักตัวอย่าง (Sample Weight omega_i)         |
| LeafMinoritySpoke2D      | ก้านเชื่อมจุดชนกลุ่มน้อยไปยังศูนย์กลางกล่อง Leaf ใน Decision Tree     |
| LeafBoundedHexbin2D      | กริดรังผึ้งจัดกลุ่มสอดคล้องกับพิกัดกล่องแบ่งขนานแกนของต้นไม้           |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: RandomForestClassifier (100 Trees) [Trails (T)] [DualRing (R)]        |
| Features: Feature_1 vs Feature_2                              [Errors (E)] [Hexbin (H)]          |
+--------------------------------------------------------------------------------------------------+
| Y: Feature_2                                                                                     |
|   1.0 +-----------------------+------------------------------------+                             |
|       | Leaf Box #1           | Leaf Box #2                        |                             |
|       |  ◎ [100% Tree Agree]  |                                    |                             |
|   0.7 |                       |   ☼ [Split Vote 52/48: Fuzzy Ring] |                             |
|       +-----------------------+------------------------------------+  Split: Y = 0.5             |
|   0.5 |                       |                                    |                             |
|       |                       |        ● [Minority Error Point]    |                             |
|       |                       |        :                           |                             |
|   0.2 |                       |        +-- Leaf Center Spoke       |                             |
|       |                       |                                    |                             |
|   0.0 +-----------------------+------------------------------------+                             |
|       0.0                    0.4                                  1.0 (X: 1)                     |
|                              Split: X = 0.4                                                      |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Definition of Done (DoD)
1. วงแหวนของ Random Forest ปรับเปลี่ยนเป็นเส้นประคู่เมื่อระดับความเห็นพ้องของต้นไม้ลดลงต่ำกว่า $60\%$
2. ใน AdaBoost ขนาดของวงแหวนภายนอกสอดคล้องกับเวกเตอร์น้ำหนัก $\omega_i$ อย่างถูกต้อง
3. จุดชนกลุ่มน้อยในกล่อง Leaf ที่ไม่บริสุทธิ์แสดงเส้น Leaf Center Spoke อย่างถูกต้อง
4. การทำงานร่วมกับกริด Leaf Bounded Hexbin คงความเร็วที่ $60$ FPS

