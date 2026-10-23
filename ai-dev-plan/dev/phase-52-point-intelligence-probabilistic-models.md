# Phase 52: 2D Point Intelligence Integration for Probabilistic & Discriminant Models (Naive Bayes, LDA, QDA)

## 1. วัตถุประสงค์และขอบเขตของโมเดลในเฟสนี้ (Objective & Model Scope)
เชื่อมต่อระบบ **2D Point Intelligence (A, B, C, D)** เข้ากับกลุ่มโมเดลความน่าจะเป็นและตัวจำแนกเชิงจำแนก (**Probabilistic & Discriminant Models**) ได้แก่ **GaussianNB, MultinomialNB, BernoulliNB, Linear Discriminant Analysis (LDA)**, และ **Quadratic Discriminant Analysis (QDA)** เพื่อเปิดเผยกลไกความน่าจะเป็นภายหลัง (Posterior Probabilities), เอนโทรปีความไม่แน่นอน (Shannon Entropy), และการเบี่ยงเบนของสมมติฐานการแจกแจงแบบเกาส์เซียน (Gaussian Assumption Discrepancy) บนระนาบ 2 มิติ

### ขอบเขตโมเดลที่รองรับในเฟสนี้
- **`GaussianNB`**: Bivariate Gaussian Naive Bayes (สมมติฐานตัวแปรเป็นอิสระต่อกัน)
- **`MultinomialNB` & `BernoulliNB`**: ตัวจำแนกความถี่คำและตัวแปรทวิภาค
- **`LinearDiscriminantAnalysis` (LDA)**: Covariance ร่วม (Shared Covariance Matrix $\boldsymbol{\Sigma}$)
- **`QuadraticDiscriminantAnalysis` (QDA)**: Covariance แยกแต่ละคลาส ($\boldsymbol{\Sigma}_c$)

---

## 2. กลไกคณิตศาสตร์และการประยุกต์ใช้ Point Intelligence เชิงลึก (Mathematical & Geometric Integration)

### 2.1 กลไก A: Posterior Entropy Flow (Comet Trail)
- **Shannon Entropy of Belief**: ในแต่ละจุดเฟรม $f$ ของ Trajectory คำนวณค่าเอนโทรปีของความน่าจะเป็นภายหลัง:
  $$H_f = -\sum_{c=1}^C P(y=c | \mathbf{x}_f) \log_2 P(y=c | \mathbf{x}_f)$$
- **Adaptive Trail Hue**:
  - หาก $H_f \to 0$ (โมเดลมั่นใจสูง): เส้นหาง Comet Trail เป็นสีทึบคมชัดตามคลาสที่ทาย
  - หาก $H_f \to 1.0$ (จุดอยู่บนรอยต่อความน่าจะเป็น 50/50): เส้นหาง Comet Trail จะเปล่งแสงสีม่วงนีออน (Magenta Glow) เพื่อเตือนว่าข้อมูลกำลังเดินทางผ่าน "โซนความไม่แน่นอนสูงสุด (Ambiguity Zone)"

### 2.2 กลไก B: Posterior Confidence Dual-Ring
1. **วงแหวนความน่าจะเป็นแปรผัน (Posterior Scaled Ring)**:
   - รัศมีของวงแหวนรอบนอกสะท้อนความมั่นใจของโมเดล:
     $$r_{\text{outer}} = r_{\text{base}} \cdot \left(0.5 + 0.5 \cdot \max_c P(y=c | \mathbf{x})\right)$$
   - เมื่อโมเดลมีความมั่นใจต่ำ ($P \approx 0.5$) วงแหวนรอบนอกจะหดตัวบางลงและกระพริบช้าๆ
2. **Mahalanobis Deviation Contour Ring (สำหรับ LDA/QDA)**:
   - แสดงวงรีขนาดจิ๋ว (Mini Elliptical Halo) ล้อมรอบจุด เพื่อบอกทิศทางความแปรปรวนร่วมตามเมทริกซ์ $\boldsymbol{\Sigma}_c$

### 2.3 กลไก C: Centroid Attractor Spokes (Bayes Paradox Explainer)
บ่อยครั้งที่โมเดล Naive Bayes หรือ LDA ทายคลาสที่อยู่ไกลกว่าในเชิงยูคลิด เนื่องจากค่าน้ำหนัก Prior หรือความเอียงของ Covariance:
- **Dual Centroid Spokes**:
  - สำหรับจุดที่ถูกทายผิด: ระบบ Error Lens จะลากเส้นประ 2 เส้นจากจุด $\mathbf{x}_i$:
    1. เส้นสีเขียวไปยังจุดศูนย์กลางคลาสจริง $\boldsymbol{\mu}_{\text{true}}$ (ระยะห่างยูคลิด $d_{\text{true}}$)
    2. เส้นสีส้มไปยังจุดศูนย์กลางคลาสที่โมเดลทาย $\boldsymbol{\mu}_{\text{pred}}$ (ระยะห่างยูคลิด $d_{\text{pred}}$)
  - แสดงป้ายกำกับอัตราส่วน Log-Likelihood Ratio เพื่ออธิบายเหตุผลทางสถิติว่าทำไมโมเดลถึงเลือกคลาสนั้น

### 2.4 กลไก D: Empirical vs Gaussian Density Divergence Hexbin
- คำนวณความแตกต่างระหว่างความหนาแน่นเชิงประจักษ์ (Empirical Count ใน Hexbin) กับความหนาแน่นเชิงทฤษฎีตามระฆังคว่ำเกาส์เซียน:
  $$\Delta \rho(u, v) = \text{Count}_{\text{empirical}}(u, v) - N \cdot \mathcal{N}\left(\mathbf{x}_c(u, v); \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c\right) \cdot \text{Area}_{\text{hex}}$$
- ช่องรังผึ้งที่มีค่า $|\Delta \rho|$ สูงจะถูกระบายสีเตือนว่า "Data Distribution ไม่ตรงกับสมมติฐานเกาส์เซียนของโมเดล"

---

## 3. สถาปัตยกรรมคอมโพเนนต์ (`frontend/src/domains/stage3d/components/surfaces_2d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D PROBABILISTIC & DISCRIMINANT POINT INTELLIGENCE OVERLAYS                                      |
+--------------------------+-----------------------------------------------------------------------+
| Component                | บทบาทเฉพาะตัวสำหรับ Probabilistic Family                              |
+--------------------------+-----------------------------------------------------------------------+
| BayesEntropyCometTrail2D | หาง Comet Trail เปลี่ยนสีตามเอนโทรปีความน่าจะเป็น Shannon Entropy H   |
| CentroidAttractorSpoke2D | ก้านคู่เชื่อมจุดข้อมูลไปยัง Centroids ของคลาสจริงและคลาสทำนาย         |
| PosteriorConfidenceRing2D| วงแหวน Dual-Ring ขยาย/หดตัวตามค่าความน่าจะเป็นสูงสุด max P(y=c|x)    |
| GaussianDivergenceHexbin2D| กริดรังผึ้งตรวจสอบจุดที่ข้อมูลจริงเบี่ยงเบนจากสมมติฐานเกาส์เซียน       |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  Model: QDA (Quadratic Discriminant)       [Trails (T)] [DualRing (R)]        |
| Features: Feature_X vs Feature_Y                              [Errors (E)] [Hexbin (H)]          |
+--------------------------------------------------------------------------------------------------+
| Y: Feature_Y                                                                                     |
|   1.0 |                   (Class 0 Gaussian Ellipse)                                             |
|       |                     /-----------\                                                        |
|   0.8 |                    /      +      \  <- mu_0 (True Centroid)                              |
|       |                   |       :       |                                                      |
|   0.6 |                    \      :      /                                                       |
|       |                     \-----:-----/                                                        |
|   0.4 |                           : (d_euc = 0.35)                                               |
|       |                           ● [Misclassified Point]                                        |
|   0.2 |                          /  (H = 0.94 bits, Ambiguity Glow)                              |
|       |                         /                                                                |
|   0.0 |       + <--------------+ (Covariance Pull: mu_1)                                         |
|       +------------------------------------------------------------------------------------------+
|       0.0             0.2             0.4             0.6             0.8            1.0 (X: X)  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Definition of Done (DoD)
1. สีของเส้นหาง Comet Trail ปรับเปลี่ยนตามค่า Shannon Entropy $H$ ได้อย่างลื่นไหล
2. แสดงเส้นดึงดูดคู่ Centroid Spokes เมื่อเปิดโหมด Error Lens บนจุดที่ทายผิด
3. แผนที่ Hexbin สามารถแสดงความคลาดเคลื่อนเทียบกับฟังก์ชันเกาส์เซียน (Gaussian Divergence) ได้
4. การคำนวณ Entropy และ Likelihood ไม่ส่งผลกระทบต่ออัตราเฟรม 60 FPS

