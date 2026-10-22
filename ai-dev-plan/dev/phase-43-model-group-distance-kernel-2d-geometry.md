# Phase 43: 2D Decision Geometries & Visualizers for Distance & Kernel Models (KNN & SVM)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
พัฒนาชุดเรขาคณิตและการแสดงผล 2 มิติเฉพาะทางสำหรับกลุ่มโมเดล **Distance-Based & Kernel-Based Learning** ได้แก่ **K-Nearest Neighbors (KNN)** และ **Support Vector Machines (SVM)** ซึ่งในโหมด 2 มิติจะทำให้ผู้ใช้เห็นภาพกลไกการตัดสินใจที่แท้จริงได้ชัดเจนอย่างยิ่ง เช่น:
- สำหรับ **KNN**: **วงกลมเรดาร์ 2 มิติ (2D Expanding Radar Circle)** ที่ขยายตัวแตะเพื่อนบ้านตัวที่ $K$ พอดี พร้อม **เส้นใยแมงมุม 2 มิติ (2D Spider Laser Lines)** ที่เชื่อมไปยังจุดอ้างอิงเพื่อนบ้าน และแผนภาพเซลล์โวโรนอย (Voronoi Boundary Approximation)
- สำหรับ **SVM**: **เส้นโค้งขอบเขตตัดสินใจไม่เป็นเชิงเส้น (2D Non-linear RBF Contour Curve)**, **แถบขอบเขตมาร์จิน (Margin Corridor Bounds $f(\mathbf{x}) = \pm 1$)**, และ **วงแหวนเพชรล้อมรอบ Support Vectors** ที่มีขนาดเส้นขอบแปรผันตามสัมประสิทธิ์ทวิภาค $|\alpha_i|$

### ขอบเขตโมเดลและคลาสที่รองรับ (In-Scope Models)
1. **`KNeighborsClassifier` & `KNeighborsRegressor`**: การจำแนกประเภทและการถดถอยด้วยเพื่อนบ้าน $K$ ตัว (Uniform และ Distance Weighting)
2. **`RadiusNeighborsClassifier` & `RadiusNeighborsRegressor`**: การจำแนกด้วยรัศมีคงที่
3. **`SVC` & `SVR`**: โมเดล SVM ทั้ง Classification และ Regression ด้วยเคอร์เนล `rbf`, `poly`, `linear`, และ `sigmoid`
4. **`LinearSVC`**: SVM เชิงเส้นความเร็วสูง

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน 2 มิติ (Mathematical Formulation & 2D Geometry)

### 2.1 เรขาคณิต 2 มิติของ K-Nearest Neighbors (KNN)
1. **วงกลมเรดาร์ 2 มิติ (2D Expanding Radar Circle)**:
   - กำหนดให้แถวข้อมูลปัจจุบันมีพิกัดบนระนาบเป็น $(x_{t,1}, x_{t,2})$
   - โมเดลคำนวณเพื่อนบ้าน $K$ ตัวอันดับแรก $\mathcal{N}_K(\mathbf{x}_t) = \{i_1, i_2, \dots, i_K\}$ ในอวกาศข้อมูล $D$ มิติ
   - **รัศมีเรดาร์ในระนาบ 2D ($R_{t, \text{2D}}$)**:
     $$R_{t, \text{2D}} = \max_{k \in \mathcal{N}_K} \sqrt{(x_{t,1} - x_{(k),1})^2 + (x_{t,2} - x_{(k),2})^2}$$
   - วาดวงกลมรัศมี $R_{t, \text{2D}}$ ล้อมรอบจุดปัจจุบัน ทำให้เห็นชัดเจนว่าในระนาบ 2 มิตินี้ เพื่อนบ้านตัวที่ไกลที่สุดของกลุ่ม $K$ ตัวอยู่ตรงจุดใด
2. **เส้นใยแสงเลเซอร์ 2 มิติ (2D Spider Web Lines)**:
   - วาดเส้นตรงเชื่อมระหว่าง $(x_{t,1}, x_{t,2})$ กับพิกัด 2D ของเพื่อนบ้านแต่ละตัว $(x_{(k),1}, x_{(k),2})$
   - ความหนาของเส้นแปรผันตามค่าน้ำหนักเสียงโหวต $w_{t,k}$:
     $$\text{stroke\_width}_k = 1.0 + 3.0 \times \frac{w_{t,k}}{\max_r w_{t,r}}$$
   - จุดเพื่อนบ้านทั้ง $K$ ตัวจะแสดงป้ายอันดับ `#1`, `#2`, ..., `#K` พร้อมสว่างวาบขึ้นในฉาก

### 2.2 เรขาคณิต 2 มิติของ Support Vector Machines (SVM)
1. **ฟังก์ชันการตัดสินใจเคอร์เนล RBF บนระนาบ 2D**:
   $$f(x_1, x_2) = \sum_{m \in SV} \alpha_m y_m \exp\left(-\gamma \|\mathbf{x}_m - \mathbf{x}\|^2\right) + b_{\text{eff}}$$
   - โดยค่าของฟีเจอร์นอกคู่แกนถูกตรึงไว้ที่ $\mathbf{x}_{\text{pinned}}$
   - เส้นแบ่งเขตตัดสินใจหลักคือเส้นไอโซไลน์ $f(x_1, x_2) = 0$
2. **แถบขอบเขตมาร์จิน (Margin Corridor Bands)**:
   - เส้นขอบมาร์จินฝั่งบวก: $f(x_1, x_2) = +1$
   - เส้นขอบมาร์จินฝั่งลบ: $f(x_1, x_2) = -1$
   - ระบายสีไฮไลต์ระหว่างเส้น $f = -1$ และ $f = +1$ เพื่อแสดงพื้นที่ของ **Soft Margin**
3. **การแสดงผลจุด Support Vectors (SV Halos & Diamond Rings)**:
   - ข้อมูลแถวที่เป็น Support Vector จริง (เก็บอยู่ใน `model.support_vectors_` และ `model.dual_coef_`) จะถูกล้อมรอบด้วยวงแหวนเพชร 2 มิติ
   - ขนาดรัศมีวงแหวนแปรผันตามขนาดของสัมประสิทธิ์ $\alpha_m$:
     $$r_{\text{halo}, m} = 4.0 + 6.0 \times \frac{|\alpha_m|}{C}$$
   - แยกประเภทจุด SV:
     - จุดที่แตะขอบ Margin พอดี ($|\alpha_m| < C$): วงแหวนสีฟ้าอ่อน
     - จุดที่ละเมิด Margin ตกเข้ามาข้างใน ($|\alpha_m| = C$): วงแหวนสีส้มเข้ม

---

## 3. UI Feature List & 2D Overlays ของกลุ่มโมเดล Distance & Kernel

```text
+--------------------------------------------------------------------------------------------------+
| 2D DISTANCE & KERNEL OVERLAYS MATRIX                                                             |
+----------------------+---------------------------------------------------------------------------+
| UI Component         | ความสามารถและองค์ประกอบที่แสดงในโหมด 2D Planar                             |
+----------------------+---------------------------------------------------------------------------+
| KnnRadar2D           | • วงกลมเรดาร์โปร่งแสง 2 มิติ รัศมี R_t ขยาย/หดตัวแบบเรียลไทม์              |
|                      | • เส้นใยแสงเลเซอร์เชื่อม K เพื่อนบ้าน พร้อมป้ายระบุระยะทางจริงและอันดับ      |
|                      | • แสดงตารางคะแนนโหวต (Vote Tally Bar) ของ K เพื่อนบ้าน                      |
+----------------------+---------------------------------------------------------------------------+
| SvmKernelContour2D   | • เส้นโค้งเรียบ RBF Decision Boundary (Marching Squares ที่ f=0)          |
|                      | • เส้นประคู่ขนานขอบเขต Margin f(x) = +1 และ f(x) = -1                     |
|                      | • แถบสีไฮไลต์ Margin Band (พื้นที่ Soft Margin)                            |
+----------------------+---------------------------------------------------------------------------+
| SupportVectorRings2D | • วงแหวนเพชร 2 มิติล้อมรอบจุด Support Vectors ทุกจุด                      |
|                      | • ปรับขนาดวงแหวนตามสัมประสิทธิ์แลงกรองจ์ |alpha_i|                         |
|                      | • ตัวเลือกเปิด/ปิด: "[x] Highlight Support Vectors Only"                  |
+----------------------+---------------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ 2D Distance & Kernel Group

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: KNeighborsClassifier (K=5, weights=distance) [● 2D Mode] [3D Mode] (V)|
+--------------------------------------------------------------------------------------------------+
| Y: Feature 2 (Petal Width)                                                                       |
|   2.5 |                                              . . . . . . . . . .                          |
|       |                                        . . . [ CLASS 1: VERSICOLOR ]                      |
|   2.0 |                                    . .                                                    |
|       |                              . . .      ◆ Ref #108 (Dist: 0.22)                           |
|   1.5 |                         . . .          /                                                  |
|       |                   . . .         (---  /  ---) <- 2D Radar Circle (R_t = 0.41)             |
|   1.0 |             . . .              /     ●-------◆ Ref #412 (Dist: 0.18)                      |
|       |       . . .                   (   [Active]   )                                            |
|   0.5 | . . .                          \      \     /                                             |
|       | [ CLASS 0: SETOSA ]             (---   \ ---)                                             |
|   0.0 +-----------------------------------------◆ Ref #889 (Dist: 0.41)-----> X: Petal Length   |
|       1.0          2.0            3.0           4.0           5.0           6.0                   |
|                                                                                                   |
| KNN HUD: K = 5 | Active Radar R_t = 0.41 | Votes: 4 Class 1 (82%) vs 1 Class 0 (18%)              |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Signatures)

### 5.1 โครงสร้างไฟล์
- `frontend/src/domains/stage3d/components/surfaces_2d/KnnRadar2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/SvmKernelContour2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/SupportVectorRings2D.tsx`
- `backend/app/domains/model_engine/adapters/distance/distance_2d_extractor.py`

### 5.2 โค้ดต้นแบบ: `KnnRadar2D.tsx`
```tsx
import React from 'react';

export interface KnnNeighbor2D {
  refIndex: number;
  x: number;
  y: number;
  distance2D: number;
  voteWeight: number;
  targetClass: number;
}

export interface KnnRadar2DProps {
  centerX: number;
  centerY: number;
  radarRadius2D: number;
  neighbors: KnnNeighbor2D[];
  visible?: boolean;
}

export function KnnRadar2D({
  centerX,
  centerY,
  radarRadius2D,
  neighbors,
  visible = true,
}: KnnRadar2DProps) {
  if (!visible) return null;

  return (
    <g className="knn-radar-2d pointer-events-none">
      {/* 1. วงกลมเรดาร์ 2 มิติ */}
      <circle
        cx={centerX}
        cy={centerY}
        r={radarRadius2D}
        fill="rgba(56, 189, 248, 0.12)"
        stroke="var(--ml-primary)"
        strokeWidth={1.5}
        strokeDasharray="4 2"
      />

      {/* 2. เส้นใยแมงมุมเชื่อมเพื่อนบ้านทั้ง K ตัว */}
      {neighbors.map((nb, idx) => (
        <g key={nb.refIndex}>
          <line
            x1={centerX}
            y1={centerY}
            x2={nb.x}
            y2={nb.y}
            stroke="var(--ml-primary)"
            strokeWidth={1.0 + nb.voteWeight * 3.0}
            strokeOpacity={0.85}
          />
          <circle
            cx={nb.x}
            cy={nb.y}
            r={5}
            fill="none"
            stroke="#ffffff"
            strokeWidth={1.5}
          />
          <text
            x={nb.x + 8}
            y={nb.y - 4}
            fill="#ffffff"
            fontSize={11}
            fontFamily="Inter"
          >
            #{idx + 1}
          </text>
        </g>
      ))}
    </g>
  );
}
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] รัศมีวงกลมเรดาร์ 2D ของ KNN สอดคล้องกับพิกัดระยะห่างของเพื่อนบ้านตัวที่ $K$ บนระนาบ 100%
- [ ] เส้นใยเลเซอร์ของ KNN ลากเชื่อมไปยังพิกัดเพื่อนบ้านถูกต้องทั้ง $K$ ตัวในทุกเฟรมที่เล่นผ่าน
- [ ] เส้นขอบเขต RBF ของ SVM และแถบ Margin ($f=\pm 1$) สอดคล้องกับค่าการคำนวณของเคอร์เนล
- [ ] จุด Support Vectors ถูกตรวจจับและแสดงวงแหวนเพชรล้อมรอบได้อย่างถูกต้องครบถ้วน
- [ ] รักษาอัตราการเรนเดอร์ในโหมด 2D ไว้ที่ **60 FPS** อย่างคงที่

