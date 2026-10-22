# Phase 46: 2D Decision Geometries & Visualizers for Gradient Boosting Heavyweights (GBM, HistGBM, XGBoost, LightGBM, CatBoost)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
พัฒนาชุดเรขาคณิตและการแสดงผล 2 มิติเฉพาะทางสำหรับกลุ่มโมเดล **เกรเดียนต์บูสติงระดับเฮฟวีเวต (Gradient Boosting Heavyweights)** ซึ่งเป็นโมเดลที่ได้รับความนิยมสูงสุดในงานข้อมูลตาราง (Tabular Data) ประกอบด้วย **Scikit-Learn GBM**, **HistGradientBoosting**, **XGBoost**, **LightGBM**, และ **CatBoost** โดยในโหมด 2 มิติ ผู้ใช้จะได้เห็นจุดเด่นทางโครงสร้างของแต่ละค่ายอย่างแท้จริง:
- **XGBoost 2D TreeSHAP Push Vectors**: ลูกศรแรงผลัก 2 มิติ $\mathbf{F}_{\text{SHAP}} = (\phi_1, \phi_2)$ พุ่งออกจากจุดข้อมูล แสดงว่าฟีเจอร์ $X$ และ $Y$ ดันคะแนน Log-odds ไปในทิศทางใด
- **CatBoost Symmetric Oblivious 2D Checkerboard Grid**: ตารางหมากรุกสมมาตร 2 มิติที่เป็นเอกลักษณ์เฉพาะของ Oblivious Trees ซึ่งมองเห็นได้ชัดเจนที่สุดในมุมมอง 2D
- **HistGBM 2D Quantized Bin Grid**: เส้นกริดแสดงขอบเขตถังข้อมูล 255 Bins ที่โมเดลแปลงค่าต่อเนื่องเป็นตัวเลขจำนวนเต็ม
- **LightGBM Asymmetric Leaf-Wise Polygons**: รูปหลายเหลี่ยมแบ่งเขตใบไม้ที่แตกกิ่งแบบไม่สมมาตร

### ขอบเขตโมเดลและไลบรารีที่รองรับ (In-Scope Libraries & Models)
1. **`GradientBoostingClassifier` & `GradientBoostingRegressor`** (`sklearn`)
2. **`HistGradientBoostingClassifier` & `HistGradientBoostingRegressor`** (`sklearn`)
3. **`XGBoost`** (`xgboost`: `.json`, `.ubj`, `.pkl`)
4. **`LightGBM`** (`lightgbm`: `.txt`, `.model`, `.pkl`)
5. **`CatBoost`** (`catboost`: `.cbm`, `.pkl`)

---

## 2. กลไกคณิตศาสตร์และการแกะ Flow การทำงานภายใน 2 มิติ (Mathematical Formulation & 2D Geometry)

### 2.1 XGBoost 2D Leaf Flow & 2D TreeSHAP Force Push Vectors
1. **สมการเวกเตอร์แรงผลัก TreeSHAP 2 มิติ**:
   จากการเรียกใช้ C++ Native Exact TreeSHAP (`pred_contribs=True`):
   $$\mathbf{x}_t \implies [\phi_{t,1}, \phi_{t,2}, \dots, \phi_{t,D}, \phi_{\text{bias}}]$$
   - ในระนาบ 2D ของคู่ฟีเจอร์ $(X_1, X_2)$ เวกเตอร์แรงผลักคือ:
     $$\mathbf{F}_{\text{SHAP}} = \begin{bmatrix} \phi_{t,1} \\ \phi_{t,2} \end{bmatrix}$$
   - หาก $\phi_{t,1} > 0$ ลูกศรจะพุ่งไปทางขวา (ดันคะแนนขึ้น), หาก $\phi_{t,2} < 0$ ลูกศรจะพุ่งลงด้านล่าง (ฉุดคะแนนลง)
   - ความยาวของลูกศรแปรผันตามขนาดผลกระทบ $\sqrt{\phi_{t,1}^2 + \phi_{t,2}^2}$ ทำให้ผู้ใช้เห็นแรงขับเคลื่อนของทั้งสองฟีเจอร์ในระนาบ 2D ทันที
2. **Leaf Partition Polygons**:
   - นำโครงสร้างต้นไม้จาก JSON Dump มาสร้างเป็นรูปหลายเหลี่ยมแบ่งเขตใบไม้ (Polygons) ในระนาบ 2D

### 2.2 CatBoost 2D Symmetric Oblivious Checkerboard Grid
1. **ธรรมชาติของ Oblivious Trees (Symmetric Trees)**:
   CatBoost ใช้ต้นไม้สมมาตร โดยในระดับความลึก $d$ ทุกโหนดจะตัดบนฟีเจอร์และ Threshold เดียวกัน:
   $$\text{Split Rule at Depth } d: \quad x_{f_d} > \theta_d$$
2. **เรขาคณิตตารางหมากรุก 2 มิติ (2D Checkerboard Tessellation)**:
   - หากในต้นไม้นั้น มีระดับชั้นที่ตัดบนแกน $X_1$ จำนวน $K_1$ ครั้ง และตัดบนแกน $X_2$ จำนวน $K_2$ ครั้ง
   - ระนาบ 2D จะถูกแบ่งอย่างสม่ำเสมอเป็นตารางหมากรุก $(K_1 + 1) \times (K_2 + 1)$ เซลล์เป๊ะ!
   - แต่ละเซลล์สี่เหลี่ยมจะมีดัชนีใบไม้แบบ Bitwise:
     $$\text{leaf\_index} = \sum_{d=0}^{\text{depth}-1} \mathbb{I}(x_{f_d} > \theta_d) \cdot 2^d$$
   - ในมุมมอง 2D ผู้ใช้จะเห็นตารางหมากรุกของ CatBoost สว่างขึ้นเฉพาะเซลล์ที่ข้อมูลกำลังตกอยู่

### 2.3 HistGradientBoosting 255-Bin Integer Grid Lines
1. **การแปลงข้อมูลเป็น Bins จำนวนเต็ม $0..255$**:
   ดึงค่า `_bin_mapper.bin_thresholds_` ของแกน $X_1$ และ $X_2$
2. **การพล็อตเส้นกริดถังข้อมูล 2 มิติ**:
   วาดเส้นกริดบางๆ สีเทาเข้ม (`#1e293b`) ตามตำแหน่ง Thresholds ของถังข้อมูล ตัดกับเส้น Decision Boundary ทำให้เห็นว่าโมเดลมองข้อมูลเป็นบล็อกขั้นบันไดดิจิทัลอย่างไร

---

## 3. UI Feature List & 2D Overlays ของกลุ่มโมเดล Boosting Heavyweights

```text
+--------------------------------------------------------------------------------------------------+
| 2D BOOSTING OVERLAYS MATRIX                                                                      |
+--------------------------+-----------------------------------------------------------------------+
| UI Component             | ความสามารถและองค์ประกอบที่แสดงในโหมด 2D Planar                         |
+--------------------------+-----------------------------------------------------------------------+
| XgboostTreeShapVector2D  | • ลูกศรแรงผลัก 2D TreeSHAP (phi_1, phi_2) พุ่งจากจุดข้อมูล            |
|                          | • ป้ายกำกับตัวเลขแสดงค่า SHAP รายฟีเจอร์แบบเรียลไทม์                   |
|                          | • เส้นขอบเขต Leaf Partition Polygons ของต้นไม้ต้นที่เลือก             |
+--------------------------+-----------------------------------------------------------------------+
| CatBoostObliviousGrid2D  | • ตารางหมากรุกสมมาตร 2D Checkerboard Grid เส้นตัดขนานแกนเป๊ะ          |
|                          | • ไฮไลต์เซลล์สี่เหลี่ยม Bitwise Leaf Cell ของแถวปัจจุบัน               |
|                          | • แสดง Bit Mask ของแต่ละแกน เช่น "[X > 45] AND [Y <= 1.2]"            |
+--------------------------+-----------------------------------------------------------------------+
| HistGbmQuantizedBins2D   | • เส้นกริดถังข้อมูล 255 Bins ตัดกับ Decision Boundary                  |
|                          | • ไฮไลต์ Bin พิกัด (bin_x, bin_y) ที่จุดปัจจุบันตกอยู่                 |
+--------------------------+-----------------------------------------------------------------------+
| LightgbmLeafWise2D       | • รูปหลายเหลี่ยมแบ่งเขตใบไม้แบบไม่สมมาตร (Asymmetric Leaf Polygons)   |
|                          | • ไฮไลต์ใบไม้ Best-First Leaf ที่ชนะคะแนน                             |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ 2D Boosting Group

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: XGBoost Classifier (Trees: 100, LR: 0.1)      [● 2D Mode] [3D Mode] (V)|
+--------------------------------------------------------------------------------------------------+
| Y: Debt Ratio (0.0 .. 1.0)                                     SHAP: phi_x = +0.84, phi_y = -1.22|
|   1.0 |                    |                                                                     |
|       |                    |        [ ZONE 0: DENIED ]                                           |
|   0.8 |                    |                                                                     |
|       |   -----------------+--------------------------------- (Boundary Polyline)                 |
|   0.6 |                    |            ▲                                                        |
|       |                    |           /                                                         |
|   0.4 |                    |   ● [Active Row]                                                    |
|       |                    |    \                                                                |
|   0.2 |   [ ZONE 1 ]       |     ▼ 2D TreeSHAP Force Vector (Pushes down towards Denied)         |
|       |   (Approved)       |                                                                     |
|   0.0 +--------------------+----------------------------------> X: Annual Income ($k)           |
|       $20k                 $65k [Leaf Split: Income = 65k]    $120k                              |
|                                                                                                   |
| BOOSTING OVERLAYS: [x] 2D TreeSHAP Vector   [x] Leaf Polygons   [x] Quantized Bins               |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Signatures)

### 5.1 โครงสร้างไฟล์
- `frontend/src/domains/stage3d/components/surfaces_2d/XgboostTreeShapVector2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/CatBoostObliviousGrid2D.tsx`
- `frontend/src/domains/stage3d/components/surfaces_2d/HistGbmQuantizedBins2D.tsx`
- `backend/app/domains/model_engine/adapters/boosting/boosting_2d_extractor.py`

### 5.2 โค้ดต้นแบบ: `XgboostTreeShapVector2D.tsx`
```tsx
import React from 'react';

export interface XgboostTreeShapVector2DProps {
  originX: number;
  originY: number;
  phiX: number;
  phiY: number;
  scaleFactor?: number;
  visible?: boolean;
}

export function XgboostTreeShapVector2D({
  originX,
  originY,
  phiX,
  phiY,
  scaleFactor = 40,
  visible = true,
}: XgboostTreeShapVector2DProps) {
  if (!visible) return null;

  const targetX = originX + phiX * scaleFactor;
  const targetY = originY + phiY * scaleFactor;
  const magnitude = Math.hypot(phiX, phiY);

  return (
    <g className="xgboost-shap-vector-2d pointer-events-none">
      {/* 1. เส้นลูกศรแรงผลัก 2D TreeSHAP */}
      <line
        x1={originX}
        y1={originY}
        x2={targetX}
        y2={targetY}
        stroke="#f43f5e"
        strokeWidth={2.5}
        markerEnd="url(#arrowhead-red)"
      />
      {/* 2. ป้ายตัวเลขกำกับขนาดแรงผลัก */}
      <text
        x={targetX + 6}
        y={targetY - 6}
        fill="#f43f5e"
        fontSize={11}
        fontFamily="JetBrains Mono"
      >
        SHAP: {magnitude >= 0 ? '+' : ''}{magnitude.toFixed(2)}
      </text>
    </g>
  );
}
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] เวกเตอร์ 2D TreeSHAP ของ XGBoost มีทิศทางและขนาดสอดคล้องกับค่า Native TreeSHAP จาก C++
- [ ] ตารางหมากรุกสมมาตรของ CatBoost แสดงเซลล์และ Bitwise Index ถูกต้องตามระดับชั้นของ Oblivious Trees
- [ ] เส้นกริดถังข้อมูล 255 Bins ของ HistGBM แสดงตำแหน่ง Thresholds ตรงกับโมเดล 100%
- [ ] เรนเดอร์บนหน้าจอ 2D ได้ที่ความเร็ว **60 FPS** ขณะเล่นวิดีโอ

