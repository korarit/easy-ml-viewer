# Phase 49: Universal 2D Point Intelligence Engine (Comet Trails, Dual-Ring Glyphs, Error Lens & Hexbin Density)

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
พัฒนาโครงสร้างพื้นฐานกลาง (**Foundational 2D Point Intelligence Engine**) สำหรับมุมมอง 2D Planar ใน Easy ML Viewer เพื่อแก้ปัญหาจุดข้อมูล (Scatter Dots) ซ้อนทับกัน (Overplotting), กลืนไปกับ Heatmap พื้นหลัง (Color Camouflage), และไม่สะท้อนมิติเวลาของเครื่องเล่น VCR โดยปฏิวัติการแสดงผลจุดบนระนาบ 2 มิติผ่าน 4 กลไกอัจฉริยะ (A, B, C, D):

1. **กลไก A — Ghost & Trajectory Comet Trail Engine (มิติเวลาและการเคลื่อนที่)**:
   - ลดความเด่นของจุดข้อมูลในอดีตและอนาคตเป็น **Ghost Dots** (ปรับ Opacity 10% - 25% หรือ Grayscale Mono)
   - วาด **Comet Tail (หางดาวตก)** ลากเชื่อมโยงจุดรอบเฟรมปัจจุบัน $\pm W$ เฟรม พร้อมฟังก์ชัน Smooth Spline Interpolation และ Gradient Fading แสดงทิศทางการเคลื่อนที่ของข้อมูลตามลำดับเวลาใน Trajectory Queue
2. **กลไก B — Dual-Ring & Glyph Shape Engine (แยกแยะค่าจริงและค่าทำนาย)**:
   - สถาปัตยกรรมจุดแบบ **Dual Concentric Rings**: วงกลมด้านในแสดงสีของค่าจริง (**Ground Truth Class** / Actual Value) ส่วนวงแหวนด้านนอกแสดงสีของค่าที่โมเดลทำนาย (**Model Prediction $\hat{y}$**)
   - มีเส้นขอบสีคอนทราสต์สูง (High-Contrast White/Black Outer Stroke) ป้องกันจุดกลืนกับผืน Heatmap 100%
   - รองรับ **Glyph Shapes**: วงกลม (`circle`), สี่เหลี่ยม (`rect`), สามเหลี่ยม (`polygon`) แยกตามคลาสจริง
3. **กลไก C — Error Lens & Residual Spoke System (วิเคราะห์ความผิดพลาดระดับจุด)**:
   - สวิตช์ **"Show Errors Only"**: กรองจุดที่โมเดลทายถูกออกทันที เหลือเฉพาะจุดที่เกิดความผิดพลาด (**Misclassification** สำหรับ Classification หรือ **$|\text{Residual}| > \epsilon$** สำหรับ Regression)
   - วาด **Residual Spoke (ก้านเวกเตอร์คลาดเคลื่อน 2D)**: ลากเส้นตรงเชื่อมระหว่างพิกัดจริงของจุดไปยังเส้นระนาบการทำนายเพื่อบอกทิศทางและขนาดความผิดพลาด
4. **กลไก D — Vectorized Hexbin Grid & Density Contours Generator (สรุปความหนาแน่นข้อมูลขนาดใหญ่)**:
   - ระบบคำนวณ **Hexagonal Binning (ตารางรังผึ้ง 6 เหลี่ยม)** บนระนาบ $X-Y$ เมื่อข้อมูลมีจำนวนมาก ($N \ge 200$)
   - ช่องรังผึ้งสามารถเลือกแสดงผลได้ 3 โหมด: **Sample Density (Count)**, **Error Concentration (% Misclassified)**, หรือ **Dominant Class Hue**
   - คำนวณเส้นชั้นความหนาแน่นของข้อมูล (**Empirical Data Density Contours**) ด้วย Fast Bivariate KDE เพื่อให้เห็นพื้นที่ Data Desert (พื้นที่ที่โมเดลฟันธงแต่ไม่มีข้อมูลจริง)

---

## 2. กลไกคณิตศาสตร์และอัลกอริทึม (Mathematical Formulation & Algorithms)

### 2.1 กลไก A: Exponential Decay Kernel สำหรับ Comet Tail
กำหนดให้เฟรมปัจจุบันคือ $t \in [0, T-1]$ และมีหน้าต่างหางแสดงผลขนาด $W$ เฟรม:
1. **ค่าน้ำหนักความโปร่งใส (Opacity Decay) สำหรับเฟรม $f \in [t - W, t]$**:
   $$\alpha(f) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \exp\left( -\frac{(t - f)^2}{2 \sigma_{\text{trail}}^2} \right)$$
   โดย $\alpha_{\max} = 1.0$, $\alpha_{\min} = 0.05$, และ $\sigma_{\text{trail}} = W / 2.5$
2. **ความหนาของเส้นทางเดิน (Trail Stroke Width)**:
   $$w_{\text{stroke}}(f) = w_{\text{base}} \cdot \left( 1.0 + 2.0 \cdot \frac{f - (t - W)}{W} \right)$$
3. **Ghost Dots ในฉากหลัง ($f \notin [t - W, t + W]$)**:
   $$\alpha_{\text{ghost}} = 0.15, \quad r_{\text{ghost}} = 0.06 \quad (\text{ลดขนาดจากปกติ } 0.10)$$

### 2.2 กลไก B: Dual-Ring Concentric Geometry & Contrast Stroke
ในระบบพิกัด World Coordinate $[-5.5, +5.5]$:
1. **Inner Core (Ground Truth)**:
   $$\text{Radius } r_{\text{inner}} = 0.08, \quad \text{Color } C_{\text{actual}} = \text{Palette}(y_{\text{true}})$$
2. **Outer Halo Ring (Prediction)**:
   $$\text{Radius } r_{\text{outer}} = 0.14, \quad \text{Thickness } t_{\text{ring}} = 0.04, \quad \text{Color } C_{\text{pred}} = \text{Palette}(\hat{y})$$
3. **Contrast Border (High-Legibility Perimeter)**:
   - ขอบนอกสุดล้อมด้วยเส้นรอบวงสีเข้ม/สว่าง:
     $$\text{stroke} = \begin{cases} \text{"#0b0f14"} & \text{ถ้าจุดสว่าง} \\ \text{"#ffffff"} & \text{ถ้าจุดมืด} \end{cases}, \quad \text{stroke\_width} = 0.025$$
   - เมื่อ $y_{\text{true}} \ne \hat{y}$ (Mispredicted): วงแหวนรอบนอกจะกระพริบ Pulse พร้อมเครื่องหมายกากบาทตัดกันที่กึ่งกลาง

### 2.3 กลไก C: Residual Spoke Vector (2D Error Spoke)
สำหรับจุดข้อมูลที่ $i$ พิกัด $\mathbf{p}_i = (x_i, y_i)$ และค่าความสูงเป้าหมาย $z_i$:
1. **ระยะคลาดเคลื่อน (Residual Error)**:
   $$e_i = z_{i, \text{actual}} - \hat{z}_{i, \text{pred}}$$
2. **Spoke Length & Angle**:
   - ในงาน Regression: ก้าน Spoke ถูกโปรเจกต์ตามทิศทาง Gradient หรือดิ่งสู่เส้น Contour $\hat{z} = z_{i, \text{actual}}$
   - สำหรับจุดที่ $|e_i| > \epsilon_{\text{threshold}}$: เส้น Spoke ถูกเรนเดอร์เป็นเส้นประสีแดง/ส้ม พร้อมลูกศรชี้ทิศทางการเบี่ยงเบน

### 2.4 กลไก D: Hexagonal Binning Tessellation Math
กำหนดขนาดรัศมีภายนอกของช่องรังผึ้ง $R_{\text{hex}}$:
1. **ระยะก้าวแกนแนวนอนและแนวตั้ง**:
   $$\Delta x = \sqrt{3} R_{\text{hex}}, \quad \Delta y = \frac{3}{2} R_{\text{hex}}$$
2. **พิกัดศูนย์กลางของรังผึ้ง $(u, v)$**:
   $$x_c(u, v) = u \cdot \Delta x + (v \bmod 2) \cdot \frac{\Delta x}{2}, \quad y_c(v) = v \cdot \Delta y$$
3. **การ Aggregation ภายในช่อง Hexagon $\mathcal{H}_{u, v}$**:
   $$\text{Count}(u, v) = |\mathcal{P}_{u, v}|, \quad \text{ErrorRate}(u, v) = \frac{1}{|\mathcal{P}_{u, v}|} \sum_{i \in \mathcal{P}_{u, v}} \mathbb{I}(y_i \ne \hat{y}_i)$$
4. **Normalized Opacity & Radius Scaling**:
   $$r_{\text{vis}}(u, v) = R_{\text{hex}} \cdot \left( 0.3 + 0.7 \cdot \sqrt{\frac{\text{Count}(u, v)}{\max_{u', v'} \text{Count}(u', v')}} \right)$$

---

## 3. สถาปัตยกรรมข้อมูลและ DTO Schemas (`backend/app/domains/spatial_projection/`)

```python
# backend/app/domains/spatial_projection/point_intelligence_entities.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class HexbinCell2D(BaseModel):
    center_x: float
    center_y: float
    radius: float
    count: int
    dominant_class: Optional[int] = None
    mean_residual: float = 0.0
    error_rate: float = 0.0

class DensityContourLevel2D(BaseModel):
    density_threshold: float
    segments: List[List[float]] # [[x1, y1, x2, y2], ...]

class PointIntelligenceResponse(BaseModel):
    hexbin_cells: List[HexbinCell2D]
    density_contours: List[DensityContourLevel2D]
    max_bin_count: int
    computation_time_ms: float
```

---

## 4. UI Feature Matrix & Component Map (`frontend/src/domains/stage3d/components/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D POINT INTELLIGENCE COMPONENT ARCHITECTURE                                                     |
+--------------------------+-----------------------------------------------------------------------+
| Component                | หน้าที่และความรับผิดชอบในระนาบ 2D                                     |
+--------------------------+-----------------------------------------------------------------------+
| PointIntelligenceLayer2D | Component รวม Root ควบคุมการแสดงผล A, B, C, D ทั้งหมด                |
| CometTrail2D             | เรนเดอร์ SVG Spline หางดาวตกตามเวลา พร้อม Gradient Decay              |
| DualRingPoint2D          | เรนเดอร์จุดวงแหวนคู่ วงใน Actual วงนอก Prediction พร้อม Stroke คมชัด  |
| ResidualSpoke2D          | ก้านเวกเตอร์แสดง Error Magnitude และทิศทางไปยัง Contour ทำนาย          |
| HexbinOverlay2D          | กริดรังผึ้ง 6 เหลี่ยมแบบ Interactive Hover เพื่อดู Count และ Error     |
| PointDisplayToolbar2D    | แถบปุ่มควบคุม Quick Toggles [Trails (T)] [DualRing (R)] [Errors (E)]  |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 5. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Feature: Income vs Credit_Score           [Trails: ON (T)] [Dual-Ring (R)]    |
| Mode: Dual-Ring + Comet Trail + Error Filter                  [Errors Only: OFF (E)] [Hexbin (H)] |
+--------------------------------------------------------------------------------------------------+
| Y: Credit Score                                                                                  |
|   800 |        (Ghost Dots: Alpha 15%)                                                           |
|       |         ·       ·          ·                                                             |
|   600 |            ·         ◎ [Dual-Ring: True=1(Blue), Pred=1(Blue)]                           |
|       |                     /                                                                    |
|   400 |           ~~~~~~~●-●-◉ [Active Frame #42: Comet Tail]                                   |
|       |                 /                                                                        |
|   200 |               ⓧ [Dual-Ring Error: True=1(Blue), Pred=0(Red) -> Pulsing Ring + Cross]     |
|       |              /                                                                           |
|     0 |             /  (Decision Boundary Line)                                                  |
|       +------------------------------------------------------------------------------------------+
|       0              20k             40k             60k             80k            100k (X: Inc)|
+--------------------------------------------------------------------------------------------------+
```

---

## 6. Definition of Done (DoD)
1. มีเอนจินคำนวณ Hexbin Tessellation ใน Backend ส่งผลลัพธ์ผ่าน API ในเวลา $< 15$ ms
2. มีคอมโพเนนต์ `CometTrail2D`, `DualRingPoint2D`, `ResidualSpoke2D`, และ `HexbinOverlay2D` บน Frontend
3. อัตราเฟรมเรตคงที่ที่ $60$ FPS ขณะทำการ Scrubbing แถบเวลา VCR
4. สวิตช์สลับโหมด A, B, C, D ทำงานแบบเรียลไทม์โดยไม่มี Re-render กระตุก
5. มี Unit Test ครอบคลุมทั้ง Backend และ Frontend 100%

