# Phase 42: 2D Decision Geometries & Visualizers for Linear & Generalized Linear Models (OLS, Logistic, Ridge, Lasso, ElasticNet, SGD)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
พัฒนาชุดเรขาคณิตและการแสดงผล 2 มิติเฉพาะทาง (**Specialized 2D Planar Geometries & Explainability Overlays**) สำหรับกลุ่มโมเดลเชิงเส้นและโมเดลเชิงเส้นนัยทั่วไป (**Linear & Generalized Linear Models**) ซึ่งเป็นรากฐานสำคัญที่สุดของ Machine Learning โดยในมุมมอง 2 มิติ ผู้ใช้จะสามารถเห็นสมการทางคณิตศาสตร์ที่แท้จริงได้ทันที เช่น **เส้นตรงแบ่งเขตการตัดสินใจ (Exact 2D Decision Line)**, **แถบทางเดินมาร์จิน (Margin Corridor Band $\pm 1$)**, **เวกเตอร์น้ำหนักความชัน (Weight Normal Vector Arrow)**, **การตรวจจับฟีเจอร์ที่เป็นศูนย์จากโทษปรับ $L_1$ (Zeroed-Out Axis Sparsity Indicator)**, และ **กราฟการตอบสนองเชิงฟังก์ชันพร้อมเส้นดิ่ง Residual 2 มิติ**

### ขอบเขตโมเดลและคลาสที่รองรับในเฟสนี้ (In-Scope Models)
1. **`LinearRegression` (OLS)**: โมเดลการถดถอยเชิงเส้นสามัญ (`sklearn.linear_model.LinearRegression`)
2. **`LogisticRegression`**: การจำแนกประเภททั้ง Binary (Sigmoid) และ Multiclass (Multinomial Softmax)
3. **`Ridge` & `RidgeClassifier`**: การลดทอนขนาดน้ำหนักด้วย $L_2$ Penalty
4. **`Lasso` & `LassoClassifier`**: การคัดเลือกฟีเจอร์ด้วย $L_1$ Penalty (Feature Sparsity)
5. **`ElasticNet` & `ElasticNetCV`**: การผสมผสานสัดส่วน $L_1/L_2$ Regularization
6. **`SGDClassifier` & `SGDRegressor`**: การเรียนรู้ด้วย Stochastic Gradient Descent (Hinge Loss, Log Loss, Modified Huber)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน 2 มิติ (Mathematical Formulation & 2D Geometry)

### 2.1 สมการเส้นตรงแบ่งเขตการตัดสินใจ 2 มิติ (Exact 2D Decision Hyperplane Line)
กำหนดให้โมเดลเชิงเส้นมีเวกเตอร์น้ำหนัก $\mathbf{w} \in \mathbb{R}^D$ และค่าไบแอส $b \in \mathbb{R}$ เมื่อผู้ใช้เลือกแกน $X_1$ (แกนนอน) และ $X_2$ (แกนตั้ง) โดยฟีเจอร์ที่เหลือถูกตรึงค่าไว้ที่ $\mathbf{x}_{\text{pinned}}$:
1. **ค่าไบแอสประสิทธิผล (Effective Bias $b_{\text{eff}}$)**:
   $$b_{\text{eff}} = b + \sum_{j \notin \{1, 2\}} w_j x_{\text{pinned}, j}$$
2. **สมการเส้นแบ่งเขตการตัดสินใจ ($z = 0$) ในระนาบ 2D**:
   $$w_1 x_1 + w_2 x_2 + b_{\text{eff}} = 0 \implies x_2 = -\frac{w_1}{w_2} x_1 - \frac{b_{\text{eff}}}{w_2} \quad (\text{กรณี } w_2 \ne 0)$$
   - หาก $w_2 = 0$ และ $w_1 \ne 0$: เส้นแบ่งเขตเป็นเส้นดิ่ง $x_1 = -b_{\text{eff}} / w_1$
3. **เวกเตอร์ตั้งฉากระนาบ (Weight Normal Vector Arrow)**:
   $$\mathbf{w}_{1,2} = (w_1, w_2)^T, \quad \hat{\mathbf{w}} = \frac{\mathbf{w}_{1,2}}{\sqrt{w_1^2 + w_2^2}}$$
   - วาดลูกศรเวกเตอร์ชี้ออกจากจุดกึ่งกลางของเส้นแบ่งเขต เพื่อแสดงทิศทางที่ค่าคะแนนพุ่งขึ้นเร็วที่สุด (Direction of Steepest Ascent)

### 2.2 เรขาคณิต 2 มิติของ Logistic Regression (Sigmoid & Softmax)
1. **Binary Classification**:
   $$P(y=1 | x_1, x_2) = \sigma(w_1 x_1 + w_2 x_2 + b_{\text{eff}}) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + b_{\text{eff}})}}$$
   - เส้นตัดสินใจหลักคือเส้นไอโซไลน์ $P = 0.5$ (ซึ่งตรงกับ $z = 0$)
   - เส้นชั้นความมั่นใจ $P \in \{0.7, 0.8, 0.9\}$ คำนวณได้โดยตรงจากสมการเส้นตรงที่เลื่อนขนานออกไป:
     $$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b_{\text{eff}} - \ln\left(\frac{P}{1-P}\right)}{w_2}$$
2. **Multiclass Softmax (พหุคลาส $C > 2$)**:
   - สำหรับแต่ละคลาส $c \in \{1, \dots, C\}$ จะมีเส้นตรง $z_c = \mathbf{w}_c^T \mathbf{x} + b_c$
   - ขอบเขตระหว่างคลาส $A$ และ $B$ คือเส้นตรงที่ $z_A = z_B$:
     $$(w_{A,1} - w_{B,1}) x_1 + (w_{A,2} - w_{B,2}) x_2 + (b_{A, \text{eff}} - b_{B, \text{eff}}) = 0$$
   - ระบายสีโซนคลาสในระนาบ 2D ตาม $\arg\max_c P(y=c)$ ก่อให้เกิดรูปหลายเหลี่ยมนูน (Convex Polygons)

### 2.3 เรขาคณิตของ Ridge, Lasso, ElasticNet และ Sparsity Indicator
1. **Lasso ($L_1$) Zeroed-Out Axis Detection**:
   - หาก $L_1$ Penalty บีบให้น้ำหนัก $w_1 = 0$: เส้นแบ่งเขตจะกลายเป็นเส้นแนวนอนขนานกับแกน $X_1$ อย่างสมบูรณ์ ($x_2 = -b_{\text{eff}}/w_2$) แสดงว่าแกน $X_1$ ไม่มีผลต่อการตัดสินใจ
   - หากทั้ง $w_1 = 0$ และ $w_2 = 0$: ระบบจะตรวจจับและแสดงป้ายเตือน $\emptyset \text{ Sparsity Warning: Both visualized features zeroed out}$
2. **ElasticNet $L_1/L_2$ Balance Dial**:
   - แสดงวงล้อมาตรวัดสัดส่วนผสม $\rho = \text{l1\_ratio} \in [0, 1]$ บนระนาบ 2D เพื่อแสดงว่าโมเดลกำลังให้น้ำหนักกับความเบาบาง (Sparsity) หรือเสถียรภาพของขนาดน้ำหนัก (Shrinkage)

### 2.4 เรขาคณิตของ SGD Margin Corridor (Hinge Loss)
สำหรับ `SGDClassifier(loss='hinge')` หรือ Linear SVM:
1. **Margin Boundaries ($z = \pm 1$)**:
   - เส้นขอบเขตบน ($z = +1$): $w_1 x_1 + w_2 x_2 + b_{\text{eff}} = 1$
   - เส้นขอบเขตล่าง ($z = -1$): $w_1 x_1 + w_2 x_2 + b_{\text{eff}} = -1$
2. **ความกว้างของทางเดินมาร์จิน (Corridor Width $W_{\text{2D}}$)**:
   $$W_{\text{2D}} = \frac{2}{\sqrt{w_1^2 + w_2^2}}$$
   - ระบายแถบสีโปร่งแสง (Shaded Corridor Band) ระหว่างเส้นทั้งสอง จุดข้อมูลที่ตกอยู่ในแถบนี้จะถูกไฮไลต์ว่าเป็นจุดที่อยู่ภายใน Margin หรือกำลังละเมิด Margin (Margin Violations)

---

## 3. UI Feature List & 2D Overlays ของกลุ่มโมเดล Linear

```text
+--------------------------------------------------------------------------------------------------+
| 2D LINEAR OVERLAYS MATRIX                                                                        |
+----------------------+---------------------------------------------------------------------------+
| UI Component         | ความสามารถและองค์ประกอบที่แสดงในโหมด 2D Planar                             |
+----------------------+---------------------------------------------------------------------------+
| LinearBoundary2D     | • วาดเส้นตรง Decision Boundary สีขาวทึบ (Stroke Width 2.5px)             |
|                      | • วาดลูกศรเวกเตอร์ Normal Vector w_1,2 ชี้ทิศทางความชัน                    |
|                      | • แสดงสมการตัวเลขจริงบนป้ายกำกับลอย เช่น "1.42*x1 - 0.85*x2 + 0.32 = 0"   |
+----------------------+---------------------------------------------------------------------------+
| SgdMarginCorridor2D  | • เส้นประคู่ขนานขอบเขต Margin (z = +1 และ z = -1)                         |
|                      | • แถบสีโปร่งแสงแสดงพื้นที่ Margin Corridor (ปรับความโปร่งใสได้ 10% - 80%)    |
|                      | • ไฮไลต์จุดข้อมูลที่ตกอยู่ใน Margin Corridor หรือทายผิดฝั่ง Margin         |
+----------------------+---------------------------------------------------------------------------+
| LassoSparsityBadge2D | • ตรวจจับน้ำหนัก w_j == 0.0 บนแกน X หรือ Y                              |
|                      | • แสดงป้ายไอคอนเตือน "Axis X Zeroed Out by L1 Penalty"                   |
+----------------------+---------------------------------------------------------------------------+
| LinearRegression2D   | • เส้นความชันสมการถดถอย 2 มิติ (2D Slope Line)                           |
|                      | • เส้นประเชื่อมค่าคลาดเคลื่อน Residual ระหว่างจุดจริงกับเส้นทำนาย         |
+----------------------+---------------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ 2D Linear Group

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: LogisticRegression (L2 Penalty, C=1.0)       [● 2D Mode] [3D Mode] (V)|
+--------------------------------------------------------------------------------------------------+
| Y: Debt Ratio (0.0 .. 1.0)                                     FORMULA: 2.15*x1 - 4.20*x2 + 1.10 = 0
|   0.90 | [ ZONE 0: DENIED ]                                                                      |
|        |                                              \  (P = 0.80 Isoline)                      |
|   0.75 |                                               \                                         |
|        |                                                \  (Decision Boundary: P = 0.50)         |
|   0.60 |                                                 \      ▲ w Normal Vector (Ascent)       |
|        |                                                  \    /                                 |
|   0.45 |      ● [Row #14]                                  \  /                                  |
|        |       (P_fraud = 12.4%)                            \/                                   |
|   0.30 |                                                     \                                   |
|        |                                                      \  [ ZONE 1: APPROVED ]            |
|   0.15 |                                                       \                                 |
|   0.00 +--------------------------------------------------------\-----> X: Annual Income ($k)   |
|        $20k        $40k           $60k           $80k          $100k                             |
|                                                                                                   |
| LINEAR CONTROLS: [x] Decision Line   [x] Normal Vector   [x] Confidence Bands   [x] Residual Bars|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Signatures)

### 5.1 โครงสร้างไฟล์
- `frontend/src/domains/stage3d/components/surfaces_2d/LinearBoundary2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/SgdMarginCorridor2D.tsx`
- `backend/app/domains/model_engine/adapters/linear/linear_2d_extractor.py`

### 5.2 โค้ดต้นแบบ: `LinearBoundary2D.tsx`
```tsx
import React from 'react';

export interface LinearBoundary2DProps {
  w1: number;
  w2: number;
  bEff: number;
  xRange: [number, number];
  yRange: [number, number];
  showNormalVector?: boolean;
  visible?: boolean;
}

export function LinearBoundary2D({
  w1,
  w2,
  bEff,
  xRange,
  yRange,
  showNormalVector = true,
  visible = true,
}: LinearBoundary2DProps) {
  if (!visible) return null;

  // คำนวณจุดตัดขอบจอระนาบ 2D
  let x1Start = xRange[0];
  let y1Start = w2 !== 0 ? (-w1 * x1Start - bEff) / w2 : yRange[0];
  let x1End = xRange[1];
  let y1End = w2 !== 0 ? (-w1 * x1End - bEff) / w2 : yRange[1];

  // จุดกึ่งกลางสำหรับวางลูกศร Normal Vector
  const midX = (x1Start + x1End) / 2;
  const midY = (y1Start + y1End) / 2;
  const normLen = Math.hypot(w1, w2) || 1;
  const arrowLen = 30; // pixels
  const arrowTargetX = midX + (w1 / normLen) * arrowLen;
  const arrowTargetY = midY + (w2 / normLen) * arrowLen;

  return (
    <g className="linear-boundary-2d pointer-events-none">
      {/* 1. เส้นตรง Decision Boundary */}
      <line
        x1={x1Start}
        y1={y1Start}
        x2={x1End}
        y2={y1End}
        stroke="#ffffff"
        strokeWidth={2.5}
        strokeLinecap="round"
      />
      {/* 2. ลูกศร Normal Vector */}
      {showNormalVector && (
        <line
          x1={midX}
          y1={midY}
          x2={arrowTargetX}
          y2={arrowTargetY}
          stroke="var(--ml-primary)"
          strokeWidth={2}
          markerEnd="url(#arrowhead-cyan)"
        />
      )}
    </g>
  );
}
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] เส้นตรง 2D Decision Boundary ถูกคำนวณและวาดตรงตามสมการ $w_1 x_1 + w_2 x_2 + b_{\text{eff}} = 0$ คลาดเคลื่อน $< 10^{-4}$
- [ ] แถบ Margin Corridor ของ SGD และ Margin Bounds ($z=\pm 1$) แสดงผลถูกต้องตามค่าความกว้าง $2/\|\mathbf{w}\|$
- [ ] ตรวจจับฟีเจอร์ที่เป็นศูนย์ ($w_j=0$) ของ Lasso/ElasticNet ได้แม่นยำ พร้อมวาดเส้นแบ่งเขตขนานแกนอย่างถูกต้อง
- [ ] เรนเดอร์เส้นความชันและเส้นประ Residual ของ Linear Regression ได้คมชัด
- [ ] เฟรมเรตคงที่ที่ **60 FPS** และไม่มี Memory Leak เมื่อสลับโหมดไปมา

