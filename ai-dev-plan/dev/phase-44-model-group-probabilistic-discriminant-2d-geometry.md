# Phase 44: 2D Decision Geometries & Visualizers for Probabilistic & Discriminant Models (Gaussian NB, Discrete NB, LDA, QDA)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
พัฒนาชุดเรขาคณิตและการแสดงผล 2 มิติเฉพาะทางสำหรับกลุ่มโมเดล **ความน่าจะเป็นและการวิเคราะห์การจำแนกเชิงความแปรปรวนร่วม (Probabilistic & Discriminant Models)** ได้แก่ **Gaussian Naive Bayes (GNB)**, **Multinomial & Bernoulli Naive Bayes**, **Linear Discriminant Analysis (LDA)**, และ **Quadratic Discriminant Analysis (QDA)** โดยในมุมมอง 2 มิติ ผู้ใช้จะได้เห็นรูปทรงเรขาคณิตทางสถิติที่สวยงามและเข้าใจง่ายที่สุดในวงการสถิติ ได้แก่:
- **วงรีความน่าจะเป็นแบบสองตัวแปร (Bivariate Gaussian Equiprobability Ellipses)** ที่ระดับ $1\sigma, 2\sigma, 3\sigma$
- การเปรียบเทียบเชิงเรขาคณิตระหว่าง **เส้นตรงแบ่งเขตของ LDA** (Common Covariance) กับ **เส้นโค้งภาคตัดกรวยของ QDA (Conic Section Hyperbola/Ellipse/Parabola)**
- **แผนผังความหนาแน่นความน่าจะเป็นสะสม 2 มิติ (2D Log-Likelihood Heatmap)**

### ขอบเขตโมเดลและคลาสที่รองรับ (In-Scope Models)
1. **`GaussianNB`**: โมเดล Naive Bayes แบบการแจกแจงปกติ
2. **`MultinomialNB` & `BernoulliNB`**: โมเดล Naive Bayes สำหรับข้อมูลนับและข้อมูลไบนารี
3. **`LinearDiscriminantAnalysis` (LDA)**: การจำแนกประเภทด้วยความแปรปรวนร่วมร่วมกัน (Shared Covariance)
4. **`QuadraticDiscriminantAnalysis` (QDA)**: การจำแนกประเภทด้วยความแปรปรวนร่วมแยกรายคลาส (Class-Specific Covariance)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน 2 มิติ (Mathematical Formulation & 2D Geometry)

### 2.1 เรขาคณิตวงรีความน่าจะเป็นของ Gaussian Naive Bayes (Bivariate Ellipses)
1. **การแจกแจงปกติร่วม 2 มิติบนแกน $X_1$ และ $X_2$**:
   เนื่องจากสมมติฐาน Naive Bayes กำหนดให้ฟีเจอร์เป็นอิสระต่อกันโดยมีเงื่อนไขตามคลาส เมทริกซ์ความแปรปรวนร่วมใน 2D จึงเป็นเมทริกซ์ทแยงมุม:
   $$\boldsymbol{\mu}_c = \begin{bmatrix} \mu_{c,1} \\ \mu_{c,2} \end{bmatrix}, \quad \boldsymbol{\Sigma}_c = \begin{bmatrix} \sigma_{c,1}^2 & 0 \\ 0 & \sigma_{c,2}^2 \end{bmatrix}$$
2. **สมการวงรีความน่าจะเป็นเท่ากัน (Equiprobability Ellipses)**:
   ระยะทางมาฮาลาโนบิสกำลังสองเท่ากับค่าคงที่ $k^2$:
   $$\frac{(x_1 - \mu_{c,1})^2}{\sigma_{c,1}^2} + \frac{(x_2 - \mu_{c,2})^2}{\sigma_{c,2}^2} = k^2$$
   - $k = 1$ ($1\sigma$): ครอบคลุมข้อมูลประมาณ $39.3\%$ ใน 2D
   - $k = 2$ ($2\sigma$): ครอบคลุมข้อมูลประมาณ $86.5\%$ ใน 2D
   - $k = 3$ ($3\sigma$): ครอบคลุมข้อมูลประมาณ $98.9\%$ ใน 2D
   - วาดเป็นวงรี 3 ชั้นซ้อนกันรอบจุดกึ่งกลาง $(\mu_{c,1}, \mu_{c,2})$ โดยแกนหลักของวงรีขนานกับแกน $X_1$ และ $X_2$ เป๊ะ

### 2.2 เรขาคณิตเปรียบเทียบระหว่าง LDA และ QDA (Linear Line vs Conic Section)
1. **Linear Discriminant Analysis (LDA)**:
   - ใช้เมทริกซ์ความแปรปรวนร่วมร่วมกันของทุกคลาส $\boldsymbol{\Sigma} = \sum_{c} \frac{N_c}{N} \boldsymbol{\Sigma}_c$
   - วงรีของทุกคลาสใน LDA จะมี **ขนาด รูปร่าง และการเอียงตัวเหมือนกันเป๊ะ**
   - ฟังก์ชันการตัดสินใจ $\delta_c(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c - \frac{1}{2} \boldsymbol{\mu}_c^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c + \ln \pi_c$
   - เส้นแบ่งเขตระหว่างคลาส $A$ และ $B$ ($\delta_A(\mathbf{x}) = \delta_B(\mathbf{x})$) ในระนาบ 2D จึงเป็น **เส้นตรงเสมอ (Straight Line)**
2. **Quadratic Discriminant Analysis (QDA)**:
   - แต่ละคลาสมีเมทริกซ์ความแปรปรวนร่วมของตัวเอง $\boldsymbol{\Sigma}_c$ ทำให้วงรีของแต่ละคลาสมีขนาดและมุมเอียงแตกต่างกันตาม Eigenvectors ของ $\boldsymbol{\Sigma}_c$
   - พจน์กำลังสองไม่ตัดกัน:
     $$\mathbf{x}^T \left( \boldsymbol{\Sigma}_A^{-1} - \boldsymbol{\Sigma}_B^{-1} \right) \mathbf{x} + \mathbf{w}^T \mathbf{x} + b = 0$$
   - เส้นแบ่งเขตในระนาบ 2D จึงเป็น **เส้นโค้งภาคตัดกรวย (Conic Section)**: ไฮเพอร์โบลา (Hyperbola), วงรี (Ellipse), หรือพาราโบลา (Parabola)

---

## 3. UI Feature List & 2D Overlays ของกลุ่มโมเดล Probabilistic & Discriminant

```text
+--------------------------------------------------------------------------------------------------+
| 2D PROBABILISTIC OVERLAYS MATRIX                                                                 |
+----------------------+---------------------------------------------------------------------------+
| UI Component         | ความสามารถและองค์ประกอบที่แสดงในโหมด 2D Planar                             |
+----------------------+---------------------------------------------------------------------------+
| GaussianEllipses2D   | • วงรี 3 ระดับ (1-sigma, 2-sigma, 3-sigma) แบ่งสีตามคลาส                   |
|                      | • เครื่องหมายกากบาทจุดศูนย์กลางค่าเฉลี่ยของคลาส (mu_c1, mu_c2)              |
|                      | • ป้ายกำกับตัวเลขแสดงค่าสถิติ Mean และ Std รายแกน                          |
+----------------------+---------------------------------------------------------------------------+
| DiscriminantCurve2D  | • เส้นตรงแบ่งเขต LDA สีขาวทึบ                                             |
|                      | • เส้นโค้งภาคตัดกรวย QDA (Hyperbola / Ellipse) คมชัดระดับเวกเตอร์         |
|                      | • วงรีแสดงความแปรปรวนร่วมที่หมุนเอียงตาม Eigenvectors ของ Sigma            |
+----------------------+---------------------------------------------------------------------------+
| DiscreteNbSteps2D    | • เส้นชั้นความสูง Log-Likelihood ของ Multinomial & Bernoulli NB          |
|                      | • แผนผังสีความน่าจะเป็นสะสมในรูปสเต็ปขั้นบันได 2 มิติ                      |
+----------------------+---------------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ 2D Probabilistic Group

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: QuadraticDiscriminantAnalysis (QDA)          [● 2D Mode] [3D Mode] (V)|
+--------------------------------------------------------------------------------------------------+
| Y: Sepal Width (2.0 .. 4.5)                                                                      |
|   4.5 |             / - - - - - - \                                                              |
|       |           /   / - - - \   \    [ CLASS 0: SETOSA ]                                       |
|   4.0 |          |   |  (★)   |   |    (mu_0 = [5.01, 3.43])                                     |
|       |           \   \ - - - /   /                                                              |
|   3.5 |             \ - - - - - - /                                                              |
|       |   =========================================== (QDA Hyperbolic Boundary Curve)            |
|   3.0 |                     / - - - - - - - - - \                                                |
|       |                   /   / - - - - - - \   \    [ CLASS 1: VERSICOLOR ]                     |
|   2.5 |                  |   |     (★)      |   |    (mu_1 = [5.94, 2.77])                       |
|       |                   \   \ - - - - - - /   /                                                |
|   2.0 +---------------------\ - - - - - - - - /----------------------> X: Sepal Length (4.0..8.0)|
|       4.0          5.0            6.0           7.0           8.0                                 |
|                                                                                                   |
| PROB CONTROLS: [x] Class Ellipses (1-3 sigma)   [x] Centroid Crosshairs   [x] Conic Boundary Curve|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Signatures)

### 5.1 โครงสร้างไฟล์
- `frontend/src/domains/stage3d/components/surfaces_2d/GaussianEllipses2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/DiscriminantBoundary2D.tsx`
- `backend/app/domains/model_engine/adapters/probabilistic/probabilistic_2d_extractor.py`

### 5.2 โค้ดต้นแบบ: `GaussianEllipses2D.tsx`
```tsx
import React from 'react';

export interface ClassEllipseData {
  classIndex: number;
  meanX: number;
  meanY: number;
  stdX: number;
  stdY: number;
  rotationRad?: number;
  color: string;
}

export function GaussianEllipses2D({
  classes,
  visible = true,
}: {
  classes: ClassEllipseData[];
  visible?: boolean;
}) {
  if (!visible) return null;

  return (
    <g className="gaussian-ellipses-2d pointer-events-none">
      {classes.map((cls) => {
        const rotDeg = ((cls.rotationRad || 0) * 180) / Math.PI;
        return (
          <g
            key={cls.classIndex}
            transform={`translate(${cls.meanX}, ${cls.meanY}) rotate(${rotDeg})`}
          >
            {/* วงรี 3 sigma (98.9%) */}
            <ellipse
              rx={cls.stdX * 3.0}
              ry={cls.stdY * 3.0}
              fill="none"
              stroke={cls.color}
              strokeWidth={1.0}
              strokeDasharray="4 4"
              opacity={0.35}
            />
            {/* วงรี 2 sigma (86.5%) */}
            <ellipse
              rx={cls.stdX * 2.0}
              ry={cls.stdY * 2.0}
              fill="none"
              stroke={cls.color}
              strokeWidth={1.5}
              strokeDasharray="6 3"
              opacity={0.65}
            />
            {/* วงรี 1 sigma (39.3%) */}
            <ellipse
              rx={cls.stdX * 1.0}
              ry={cls.stdY * 1.0}
              fill={cls.color}
              fillOpacity={0.12}
              stroke={cls.color}
              strokeWidth={2.0}
            />
            {/* จุดกึ่งกลาง Mean */}
            <line x1={-6} y1={0} x2={6} y2={0} stroke={cls.color} strokeWidth={2} />
            <line x1={0} y1={-6} x2={0} y2={6} stroke={cls.color} strokeWidth={2} />
          </g>
        );
      })}
    </g>
  );
}
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] จุดศูนย์กลางของวงรีตรงกับค่าเฉลี่ย $\mu_{c,1}$ และ $\mu_{c,2}$ ของโมเดล 100%
- [ ] ขนาดกึ่งแกนของวงรี $1\sigma, 2\sigma, 3\sigma$ สอดคล้องกับค่าส่วนเบี่ยงเบนมาตรฐาน $\sigma_1, \sigma_2$
- [ ] เส้นแบ่งเขตของ LDA แสดงเป็นเส้นตรง และเส้นแบ่งเขตของ QDA แสดงเป็นเส้นโค้งภาคตัดกรวยอย่างถูกต้อง
- [ ] เรนเดอร์บนจอ 2D ได้ที่ความเร็ว **60 FPS** โดยไม่มีอาการภาพกระตุก

