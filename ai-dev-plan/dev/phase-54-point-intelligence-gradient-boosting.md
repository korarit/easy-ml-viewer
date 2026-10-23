# Phase 54: 2D Point Intelligence Integration for Gradient Boosting Heavyweights (GBM, HistGBM, XGBoost, LightGBM, CatBoost)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เชื่อมต่อระบบ **2D Point Intelligence (A, B, C, D)** เข้ากับกลุ่มโมเดลเกรเดียนต์บูสติงระดับเรือธง (**Gradient Boosting Heavyweights**) ได้แก่ **Scikit-Learn GBM, HistGradientBoosting, XGBoost, LightGBM**, และ **CatBoost** เพื่อเปิดเผยกลไกการปรับแก้ Residual ทีละขั้น (Staged Residual Descent), การผลักดันตำแหน่งผ่านเวกเตอร์ TreeSHAP, และการจัดกลุ่มข้อมูลตามกริดการควอนไทซ์ฟีเจอร์ (Quantized Binning Grid) บนระนาบ 2 มิติ

### ขอบเขตโมเดลที่รองรับในเฟสนี้
- **`GradientBoostingClassifier` & `GradientBoostingRegressor`** (Sklearn)
- **`HistGradientBoostingClassifier` & `HistGradientBoostingRegressor`** (Sklearn)
- **`XGBClassifier` & `XGBRegressor`** (XGBoost)
- **`LGBMClassifier` & `LGBMRegressor`** (LightGBM)
- **`CatBoostClassifier` & `CatBoostRegressor`** (CatBoost)

---

## 2. กลไกคณิตศาสตร์และการประยุกต์ใช้ Point Intelligence เชิงลึก (Mathematical & Geometric Integration)

### 2.1 กลไก A: Staged Boosting & TreeSHAP Push Comet Trail
ในการคำนวณแบบ Boosting ฟังก์ชันผลลัพธ์คือผลรวมสะสม $F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^M \eta f_m(\mathbf{x})$:
- **TreeSHAP Cumulative Push Vector Trail**:
  - เมื่อผู้ใช้ทำการเลื่อนสเต็ปหรือ Scrubbing ต้นไม้ $m = 1 \dots M$: เส้นหาง Comet Trail จะลากต่อทอดเวกเตอร์แรงผลักของสองฟีเจอร์ที่แสดงผล $\vec{\phi}_{1,2}^{(m)} = (\phi_{m, 1}, \phi_{m, 2})$
  - ทำให้เห็นเส้นทางการเดินทางของจุดข้อมูลว่าถูกต้นไม้แต่ละต้น "ผลัก (Push)" ข้ามเขตการตัดสินใจอย่างไร
- **Stage Progression Fading**: จุดในอดีตของขั้นตอน Boosting ที่แล้วจะถูกเรนเดอร์เป็น Ghost Trace ซ้อนอยู่เบื้องหลัง

### 2.2 กลไก B: Functional Margin Dual-Ring ($y \cdot F(\mathbf{x})$)
1. **Functional Margin Indicator Ring**:
   - คำนวณค่า Functional Margin $M_i = y_i \cdot F(\mathbf{x}_i)$ (โดย $y_i \in \{-1, +1\}$):
   - **กรณี $M_i > 1.5$ (Safe Margin)**: วงแหวนรอบนอกเป็นเส้นทึบสีเดียวกับคลาสจริง (ทายถูกและมั่นใจมาก)
   - **กรณี $0 \le M_i \le 1.5$ (Critical Margin)**: วงแหวนรอบนอกเป็นสีส้มประ (Pulsing Orange) เตือนว่าจุดนี้อยู่ใกล้จุดเปลี่ยนคำตัดสินใจ
   - **กรณี $M_i < 0$ (Negative Margin / Misclassified)**: วงแหวนรอบนอกเป็นสีแดงจัด พร้อมสัญลักษณ์กากบาทเตือนว่าเป็นเป้าหมายที่ Boosting กำลังพยายามตามแก้ในรอบถัดไป
2. **CatBoost Oblivious Symmetry Marker**:
   - แสดงไอคอนกริดสมมาตร 2D รอบจุดที่ตกอยู่ในบิตคำสั่งของ Oblivious Tree

### 2.3 กลไก C: Pseudo-Residual Gradient Spokes
- คำนวณ Pseudo-Residual หรือ Negative Gradient ประจำแถว:
  $$r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F = F_{m-1}}$$
- **Negative Gradient Spoke**:
  - ก้าน Spoke จะถูกยืดออกจากจุดข้อมูลไปตามทิศทาง Gradient บนระนาบ $X-Y$ โดยความยาวของก้านแปรผันตรงกับขนาดของ $|r_{im}|$
  - แสดงให้ Data Scientist เห็นทันทีว่าจุดไหนเป็น "จุดยาก (Hard Sample)" ที่กำลังส่งแรงดึงดูดโมเดลมากที่สุดในรอบการเทรนนั้น

### 2.4 กลไก D: Histogram Quantized Bin Hexbin (HistGBM & LightGBM)
- แทนที่จะใช้ระยะทางแบบยูคลิดทั่วไป กริดรังผึ้งจะถูกจัดวางให้สอดคล้องกับพิกัด **Quantized Bins 256 ระดับ** ของ HistGBM และ LightGBM
- ช่องรังผึ้งจะสะท้อนสถิติของ Histogram Gradient สองมิติ ($\sum g_i, \sum h_i$) ประจำแต่ละช่อง

---

## 3. สถาปัตยกรรมคอมโพเนนต์ (`frontend/src/domains/stage3d/components/surfaces_2d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D GRADIENT BOOSTING POINT INTELLIGENCE OVERLAYS                                                 |
+--------------------------+-----------------------------------------------------------------------+
| Component                | บทบาทเฉพาะตัวสำหรับ Boosting Family                                   |
+--------------------------+-----------------------------------------------------------------------+
| BoostingShapPushTrail2D  | หาง Comet Trail แสดงผลรวมเวกเตอร์แรงผลัก TreeSHAP ทีละขั้นตอน Boosting |
| FunctionalMarginRing2D   | วงแหวน Dual-Ring สะท้อนสถานะ Functional Margin y * F(x)               |
| PseudoResidualSpoke2D    | ก้านแสดงขนาดและทิศทาง Negative Gradient (Pseudo-Residual)             |
| QuantizedBinHexbin2D     | กริดรังผึ้งสอดคล้องกับพิกัด 256 Bins ของ HistGBM / LightGBM            |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: XGBoost (Tree #34 of 100)          [Trails (T)] [DualRing (R)]        |
| Features: Feature_A vs Feature_B                              [Errors (E)] [Hexbin (H)]          |
+--------------------------------------------------------------------------------------------------+
| Y: Feature_B                                                                                     |
|   1.0 |                [ Negative Margin Zone: Red Ring ]                                        |
|       |                     ⓧ [Row #89: M = -0.42]                                               |
|   0.8 |                      \                                                                   |
|       |                       +-- Pseudo-Residual Spoke (Grad = 0.85)                            |
|   0.6 |                        \                                                                 |
|       |                         \  Decision Boundary (F(x) = 0)                                  |
|   0.4 |     ◎                    \                                                               |
|       |   [Safe: M = +2.1]        \                                                              |
|   0.2 |                            \    ~~~~~~->● [Active Point: TreeSHAP Push Vector Trail]     |
|       |                             \                                                            |
|   0.0 |                              \                                                           |
|       +------------------------------------------------------------------------------------------+
|       0.0             0.2             0.4             0.6             0.8            1.0 (X: A)  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Definition of Done (DoD)
1. หาง Comet Trail สามารถสลับแสดงผลลัพธ์เวกเตอร์ TreeSHAP ตามการเลื่อนแถบเวลา Boosting ได้
2. วงแหวน Functional Margin สะท้อนค่า $y_i F(\mathbf{x}_i)$ อย่างถูกต้องทั้ง 3 สถานะ (Safe, Critical, Negative)
3. ก้าน Pseudo-Residual Spoke คำนวณขนาดและทิศทางตรงตามสูตร Negative Gradient ของ Loss Function
4. รักษาอัตราเฟรมเรตที่ $60$ FPS ขณะทำการ Scrubbing ขั้นตอน Boosting

