# Phase 51: 2D Point Intelligence Integration for Distance & Kernel Models (KNN & SVM)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เชื่อมต่อระบบ **2D Point Intelligence (A, B, C, D)** เข้ากับกลุ่มโมเดลเชิงระยะทางและเคอร์เนล (**Distance-Based & Kernel Models**) โดยเฉพาะ **K-Nearest Neighbors (KNN)** และ **Support Vector Machines (SVM)** เพื่อให้การวิเคราะห์เพื่อนบ้านใกล้เคียง, ขอบเขตมาร์จินไม่เป็นเชิงเส้น (Non-linear Kernel Margins), และการคัดเลือกเวกเตอร์สนับสนุน (Support Vectors) มีความชัดเจน โปร่งใส และเข้าใจง่ายที่สุดในระนาบ 2 มิติ

### ขอบเขตโมเดลที่รองรับในเฟสนี้
- **`KNeighborsClassifier` & `KNeighborsRegressor`**
- **`RadiusNeighborsClassifier` & `RadiusNeighborsRegressor`**
- **`SVC` (Kernels: RBF, Poly, Sigmoid, Linear)**
- **`SVR` (Support Vector Regression with $\epsilon$-tube)**
- **`LinearSVC`**

---

## 2. กลไกคณิตศาสตร์และการประยุกต์ใช้ Point Intelligence เชิงลึก (Mathematical & Geometric Integration)

### 2.1 กลไก A: Neighbor-Anchored Comet Trail (KNN Dynamic Neighborhood)
ในขณะที่ผู้ใช้ทำการ Scrubbing แถบเวลา VCR จุด Active Point จะเคลื่อนที่ผ่านระนาบ 2 มิติ:
- **Neighbor Swap Indicator**: เมื่อหาง Comet Trail เคลื่อนที่ข้าม Voronoi Cell หรือสลับกลุ่ม $K$-Nearest Neighbors ระบบจะวาดวงกระเพื่อมระลอกน้ำ (Ripple Wave) ที่จุดเพื่อนบ้านตัวใหม่ที่เพิ่งถูกดึงเข้ามาในกลุ่ม $\mathcal{N}_K$
- **Trajectory Distance Elasticity**: เส้นหาง Comet Trail ถูกยึดโยง (Elastic Web) เข้ากับจุดศูนย์กลางมวลของกลุ่มเพื่อนบ้าน $K$ ตัว ($\bar{\mathbf{x}}_K = \frac{1}{K} \sum_{j \in \mathcal{N}_K} \mathbf{x}_j$)

### 2.2 กลไก B: Support Vector Halo Dual-Ring (SVM Support Points)
สำหรับโมเดล SVM ข้อมูลใน Trajectory ทุกแถวจะถูกตรวจสอบความสัมพันธ์กับ Support Vectors:
1. **Unbounded Support Vectors ($|\alpha_i| = C$)**:
   - จุดที่ละเมิดมาร์จินหรือตกอยู่ใน Soft Margin จะแสดง Dual-Ring ล้อมรอบด้วย **วงแหวนเพชรสีส้มเข้ม (Diamond Halo)** บ่งชี้ว่าเป็นตัวกำหนดจุดตัดมาร์จิน
2. **Bounded Support Vectors ($0 < |\alpha_i| < C$)**:
   - จุดที่อยู่บนขอบเขตมาร์จินพอดี ($f(\mathbf{x}) = \pm 1$) จะแสดง Dual-Ring ล้อมรอบด้วย **วงแหวนเพชรสีฟ้าคราม (Cyan Diamond Halo)**
3. **KNN Vote Breakdown Ring (สำหรับ KNN)**:
   - วงแหวนรอบนอกถูกแบ่งสัดส่วนเป็น Segmented Pie Arc ตามสัดส่วนคะแนนโหวต เช่น คลาส A 60% (3/5 เสียง) และคลาส B 40% (2/5 เสียง)

### 2.3 กลไก C: SVR $\epsilon$-Tube Slack Spokes & KNN Discordance Lines
1. **SVR $\epsilon$-Insensitive Violation Spokes**:
   - สำหรับ SVR จะไม่สนใจ Error ที่อยู่ภายในท่อ $\pm \epsilon$
   - ระบบ Error Lens จะ **วาด Spoke เฉพาะจุดที่ทะลุออกจากท่อ $\epsilon$** ($|y_i - \hat{y}_i| > \epsilon$) โดยความยาวของ Spoke สะท้อนค่า Slack Variable โดยตรง:
     $$\xi_i = \max(0, y_i - \hat{y}_i - \epsilon), \quad \xi_i^* = \max(0, \hat{y}_i - y_i - \epsilon)$$
2. **KNN Conflict Discordance Spokes**:
   - จุดที่ทายผิดใน KNN จะมีเส้นประสีแดงลากเชื่อมไปยังเพื่อนบ้านในกลุ่ม $K$ ตัวที่โหวตเสียงข้างมากค้านกับคลาสจริงของจุดนั้น

### 2.4 กลไก D: Local Manifold Density Hexbin
- คำนวณค่าประมาณความหนาแน่นเชิงระยะทาง (Distance-Based Density Metric) ภายในช่องรังผึ้ง:
  $$\rho_{\text{knn}}(u, v) = \frac{K}{\pi \cdot \bar{R}_K^2}$$
  ช่วยให้ระบุบริเวณที่เป็น "เกาะข้อมูลหนาแน่น" เทียบกับบริเวณที่เป็น "ชายแดนข้อมูลเบาบาง (Sparse Outliers)"

---

## 3. สถาปัตยกรรมคอมโพเนนต์ (`frontend/src/domains/stage3d/components/surfaces_2d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D DISTANCE & KERNEL POINT INTELLIGENCE OVERLAYS                                                 |
+--------------------------+-----------------------------------------------------------------------+
| Component                | บทบาทเฉพาะตัวสำหรับ Distance & Kernel Family                          |
+--------------------------+-----------------------------------------------------------------------+
| SvmSupportVectorHalo2D   | ซ้อนวงแหวนเพชร Dual-Ring ตามสัมประสิทธิ์แอลฟา (alpha_i)               |
| SvrSlackSpoke2D          | ก้านแสดง Slack Variables xi / xi* เฉพาะจุดที่หลุดนอกท่อ epsilon       |
| KnnVoteArcRing2D         | วงแหวนแบ่งสัดส่วน Pie Arc ตามเสียงโหวตจริงของเพื่อนบ้าน K ตัว         |
| KnnDensityHexbin2D       | กริดรังผึ้งความหนาแน่นถ่วงน้ำหนักด้วยรัศมีเพื่อนบ้าน K ตัว             |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: SVC (RBF Kernel, C=1.0)            [Trails (T)] [DualRing (R)]        |
| Features: Feature_1 vs Feature_2                              [Errors (E)] [Hexbin (H)]          |
+--------------------------------------------------------------------------------------------------+
| Y: Feature_2                                                                                     |
|   1.0 |                ◇ [Support Vector: |alpha|=C, Orange Diamond]                             |
|       |                 \                                                                        |
|   0.8 |                  \  Margin Line: f(x) = +1                                               |
|       |                   \                                                                      |
|   0.6 |                    \  RBF Decision Boundary: f(x) = 0                                    |
|       |                     \                                                                    |
|   0.4 |     ◎ [Normal Point] \    ◆ [Support Vector on Margin: 0<|alpha|<C, Cyan]                |
|       |                       \                                                                  |
|   0.2 |                        \  Margin Line: f(x) = -1                                         |
|       |                         \                                                                |
|   0.0 |                          \                                                               |
|       +------------------------------------------------------------------------------------------+
|       0.0             0.2             0.4             0.6             0.8            1.0 (X: 1)  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Definition of Done (DoD)
1. จุด Support Vectors แสดงสัญลักษณ์ Diamond Halo ร่วมกับ Dual-Ring อย่างถูกต้องตามค่า $\alpha_i$
2. ใน SVR มีการวาด Slack Spoke เฉพาะจุดที่อยู่นอกท่อ $\epsilon$-tube เท่านั้น
3. ใน KNN มีการแสดงสัดส่วน Vote Arc บนขอบ Dual-Ring อย่างถูกต้อง
4. คงอัตราเฟรม $60$ FPS ในขณะที่เรดาร์ KNN และหาง Comet Trail เคลื่อนที่พร้อมกัน

