# Phase 50: 2D Point Intelligence Integration for Linear & Generalized Linear Models (OLS, Logistic, Ridge, Lasso, ElasticNet, SGD)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เชื่อมต่อระบบ **2D Point Intelligence (A, B, C, D)** เข้ากับกลุ่มโมเดลเชิงเส้นและโมเดลเชิงเส้นนัยทั่วไป (**Linear & Generalized Linear Models**) เพื่อยกระดับความสามารถในการอธิบายผล (Explainability) ในมุมมองระนาบ 2 มิติ ให้เห็นความสัมพันธ์ระหว่างจุดข้อมูลแต่ละจุดกับสมการระนาบตัดสินใจ ($z = \mathbf{w}^T \mathbf{x} + b$) อย่างแม่นยำ

### ขอบเขตโมเดลที่รองรับในเฟสนี้
- **`LinearRegression` (OLS)**
- **`LogisticRegression` (Binary & Multiclass)**
- **`Ridge` & `RidgeClassifier`**
- **`Lasso` & `LassoClassifier`**
- **`ElasticNet` & `ElasticNetCV`**
- **`SGDClassifier` & `SGDRegressor`**

---

## 2. กลไกคณิตศาสตร์และการประยุกต์ใช้ Point Intelligence เชิงลึก (Mathematical & Geometric Integration)

### 2.1 กลไก A: Trajectory Gradient Alignment (Comet Trail)
สำหรับโมเดลเชิงเส้น ค่า Log-Odds หรือค่าทำนายขึ้นกับผลคูณสเกลาร์ $\mathbf{w}^T \mathbf{x}$:
- **แนวคิด**: Comet Trail ใน Linear Model จะแสดงทิศทางการเคลื่อนที่ของข้อมูลเทียบกับเวกเตอร์น้ำหนัก $\mathbf{w}_{1,2} = (w_1, w_2)^T$
- **Directional Dot Product**:
  $$\Delta z_f = \mathbf{w}_{1,2}^T (\mathbf{p}_f - \mathbf{p}_{f-1})$$
  หาก $\Delta z_f > 0$ เส้นหาง Comet Tail จะเปล่งประกายสีเขียว (พุ่งเข้าหา Positive Class) และหาก $\Delta z_f < 0$ จะเปล่งประกายสีส้มแดง (พุ่งเข้าหา Negative Class)

### 2.2 กลไก B: Margin-Violating Dual-Ring (สำหรับ SGD & Logistic)
1. **SGD Margin Corridor Violation**:
   - สำหรับจุดที่ตกอยู่ภายในแถบ Margin Corridor ($|z_i| < 1.0$):
   - วงแหวนรอบนอก (Outer Ring) จะแสดงเป็น **ขอบประสีเหลืองอำพัน (Pulsing Amber Dashed Ring)** แสดงว่าเป็นจุดที่มีความเสี่ยงสูง (Uncertain / Low Margin)
2. **Logistic Probability Confidence Ring**:
   - ความหนาของวงแหวนรอบนอกแปรผกผันกับระยะห่างจากความไม่แน่นอน:
     $$t_{\text{ring}} = 0.02 + 0.06 \cdot (1.0 - 2 \cdot |P(y=1) - 0.5|)$$
     จุดที่อยู่ใกล้เส้นตัดสินใจ ($P \approx 0.5$) จะมีวงแหวนหนาและฟุ้งเด่นชัดที่สุด

### 2.3 กลไก C: Orthogonal Distance Spokes & Vertical Residuals
1. **Classification: Orthogonal Hyperplane Spoke**:
   - สำหรับจุดที่ถูกจัดประเภทผิด (Misclassified): ระบบจะลากเส้นตั้งฉากจากจุด $\mathbf{p}_i = (x_1, x_2)$ ไปยังเส้นตรง Decision Boundary ($w_1 x_1 + w_2 x_2 + b_{\text{eff}} = 0$):
     $$\mathbf{p}_{\text{projected}} = \mathbf{p}_i - \frac{w_1 x_1 + w_2 x_2 + b_{\text{eff}}}{w_1^2 + w_2^2} \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}$$
   - ระยะทางตั้งฉาก $d_{\perp} = \frac{|w_1 x_1 + w_2 x_2 + b_{\text{eff}}|}{\sqrt{w_1^2 + w_2^2}}$ จะถูกแสดงเป็นป้ายตัวเลขขนาดเล็กบนเส้นประ
2. **Regression: Exact Vertical Residual Stem**:
   - ลากเส้นตรงดิ่งตามแกน $Y$ (หรือแกน Residual) จากจุดข้อมูลจริงไปยังเส้นแนวโน้มการถดถอย พร้อมสัญลักษณ์หัวลูกศรบอก Over-predict (+) หรือ Under-predict (-)

### 2.4 กลไก D: Linear-Partitioned Hexbin Aggregation
- **Hexbin Boundary Bisector**: ช่องรังผึ้งที่เส้นตรง Decision Boundary พาดผ่านจะถูกติดธงเป็น `boundary_intersected: true`
- **Error Rate Gradient Along Weight Vector**: คำนวณความสัมพันธ์ระหว่างระยะทาง $d_{\perp}$ ของศูนย์กลางรังผึ้งกับ Error Rate ของจุดภายในรังผึ้ง เพื่อให้เห็นว่าจุดที่ห่างจากเส้นตรงมี Error ลดลงตามทฤษฎีหรือไม่

---

## 3. สถาปัตยกรรมคอมโพเนนต์และการปรับใช้ (`frontend/src/domains/stage3d/components/surfaces_2d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D LINEAR POINT INTELLIGENCE OVERLAYS                                                            |
+--------------------------+-----------------------------------------------------------------------+
| Component                | บทบาทเฉพาะตัวสำหรับ Linear Family                                     |
+--------------------------+-----------------------------------------------------------------------+
| LinearOrthogonalSpoke2D  | วาดเส้นตั้งฉาก Orthogonal Projection จากจุด Misclassified ไปยังเส้น w |
| LinearMarginDualRing2D   | เสริมวงแหวนประสีอำพันรอบจุดที่ตกอยู่ใน Margin Corridor                 |
| LinearResidualStem2D     | เสริมก้านดิ่ง Residual ลอยตัวสำหรับ LinearRegression                   |
| LinearHexbinAlignment2D  | ไฮไลต์ช่องรังผึ้ง 6 เหลี่ยมที่ผ่ากลางแนวเส้นตรง Decision Boundary      |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: SGDClassifier (Hinge Loss)         [Trails (T)] [DualRing (R)]        |
| Features: Feature_A vs Feature_B                              [Errors (E)] [Hexbin (H)]          |
+--------------------------------------------------------------------------------------------------+
| Y: Feature_B                                                                                     |
|   1.0 |                [ ZONE +1: POSITIVE ]                                                     |
|       |                                        \  Margin Bound (z = +1)                          |
|   0.8 |                    ◎                    \                                                |
|       |                                          \  Decision Boundary (z = 0)                    |
|   0.6 |             (Inside Corridor)             \                                              |
|       |                 ☼ [Pulsing Amber Ring]     \                                             |
|   0.4 |                    |                        \                                            |
|       |                    +--- Orthogonal Spoke --->\  Margin Bound (z = -1)                    |
|   0.2 |                     (d_perp = 0.28)           \                                          |
|       |                                                \                                         |
|   0.0 |                                                 \    [ ZONE -1: NEGATIVE ]               |
|       +------------------------------------------------------------------------------------------+
|       0.0             0.2             0.4             0.6             0.8            1.0 (X: A)  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Definition of Done (DoD)
1. เวกเตอร์ Orthogonal Projection Spoke สอดคล้องกับสูตร $d_{\perp}$ ทางคณิตศาสตร์ $100\%$
2. จุดใน Margin Corridor ($|z| \le 1$) แสดงวงแหวนประสีอำพันในโหมด Dual-Ring
3. เส้นดิ่ง Residual Stem ของ LinearRegression ลากสัมผัสกับระนาบทำนายอย่างแม่นยำ
4. ทดสอบความเร็วในการเรนเดอร์ร่วมกับ Hexbin บน dataset $5,000$ แถว คงที่ที่ $60$ FPS

