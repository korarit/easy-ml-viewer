# Test Blueprint — Phase 24: Gaussian Naive Bayes Adapter & 3D Bivariate Bell Hills Visualizer

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-24-model-gaussian-naive-bayes.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-24-model-gaussian-naive-bayes.md) เพื่อยืนยันความถูกต้องของคลาส `GaussianNBAdapter` (`backend/app/domains/model_engine/adapters/probabilistic/gaussian_nb.py`), คอมโพเนนต์ภูเขาระฆังคว่ำ `GaussianBellHills3D` (`frontend/src/domains/stage3d/components/surfaces/GaussianBellHills3D.tsx`), และแผงวิเคราะห์ความน่าจะเป็น `GaussianNBInspector` (`frontend/src/domains/hud/components/inspectors/GaussianNBInspector.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn**:
  - `sklearn.naive_bayes.GaussianNB` (ทั้งการเทรนด้วย `fit()`, การเทรนแบบทยอยด้วย `partial_fit()`, และการกำหนด `priors` เอง)
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - สมการ Univariate Log-Likelihood รายฟีเจอร์: $\ell_{t,k,j} = -\frac{1}{2}\log(2\pi\sigma_{k,j}^2) - \frac{(x_{t,j} - \mu_{k,j})^2}{2\sigma_{k,j}^2}$
  - สมการ Joint Log-Likelihood: $J_{t,k} = \log \pi_k + \sum_{j=1}^{D} \ell_{t,k,j}$ ต้องตรงกับ `model.predict_joint_log_proba(X)` 100%
  - สำหรับ Binary Classification: ผลรวมของ Log-Likelihood Ratio Waterfall ($\log\frac{\pi_1}{\pi_0} + \sum_{j=1}^{D}(\ell_{t,1,j} - \ell_{t,0,j})$) ต้องเท่ากับ $J_{t,1} - J_{t,0}$ เป๊ะ
  - การคำนวณความสูงภูเขาระฆังคว่ำ 2 มิติ $Z_k(x, y) = \pi_k \exp\left(-\frac{(x-\mu_{k,x})^2}{2\sigma_{k,x}^2} - \frac{(y-\mu_{k,y})^2}{2\sigma_{k,y}^2}\right)$

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **Joint Log-Probability & Waterfall Additivity**:
   - กรณี Binary ($C=2$): `np.sum(flow.feature_contributions, axis=1) + flow.base_intercepts[0]` ต้องเท่ากับ `flow.decision_scores[:, 1] - flow.decision_scores[:, 0]` (`atol=1e-5`)
   - กรณี Multiclass ($C \ge 3$): สำหรับแต่ละแถว $t$ ที่คลาสชนะคือ $w_t = \arg\max_k P(C_k \mid \mathbf{x}_t)$ ผลรวม `np.sum(flow.feature_contributions[t]) + flow.base_intercepts[w_t]` ต้องเท่ากับ `flow.decision_scores[t, w_t]` (`atol=1e-5`)
2. **Bell Hills Peak & Axis-Aligned Symmetry Verification**:
   - ที่พิกัดยอดเขา $(x, y) = (\mu_{k,x}, \mu_{k,y})$ ความสูงของภูเขาคลาส $k$ ต้องมีค่าสูงสุดเท่ากับ $\pi_k$ พอดี

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_gaussian_nb_adapter.py`)
```python
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.naive_bayes import GaussianNB
from app.domains.model_engine.adapters.probabilistic.gaussian_nb import GaussianNBAdapter

def test_binary_gaussian_nb_log_bayes_factor_additivity():
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=42
    )
    model = GaussianNB(priors=[0.3, 0.7]).fit(X, y)
    adapter = GaussianNBAdapter(model, ["f0", "f1", "f2", "f3"])

    meta = adapter.extract_metadata()
    np.testing.assert_allclose(meta["class_priors"], [0.3, 0.7], atol=1e-6)
    np.testing.assert_allclose(meta["theta_means"], model.theta_, atol=1e-6)

    flow = adapter.trace_internal_flow(X[:25])
    jll = np.asarray(flow.decision_scores)
    contribs = np.asarray(flow.feature_contributions)
    prior_log_ratio = flow.base_intercepts[0]

    # 1. Joint log probability matches sklearn native predict_joint_log_proba
    expected_jll = model.predict_joint_log_proba(X[:25])
    np.testing.assert_allclose(jll, expected_jll, atol=1e-5)

    # 2. Binary Log-Bayes Factor additivity: sum(LLR_j) + log(pi_1 / pi_0) == J_1 - J_0
    reconstructed_llr = np.sum(contribs, axis=1) + prior_log_ratio
    expected_llr = expected_jll[:, 1] - expected_jll[:, 0]
    np.testing.assert_allclose(reconstructed_llr, expected_llr, atol=1e-5)

def test_multiclass_iris_gaussian_nb_and_bell_hill_surfaces():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = GaussianNB().fit(X, y)
    adapter = GaussianNBAdapter(model, list(iris.feature_names))

    flow = adapter.trace_internal_flow(X[:30])
    probas = np.asarray(flow.probabilities)
    jll = np.asarray(flow.decision_scores)
    contribs = np.asarray(flow.feature_contributions)
    log_priors = np.asarray(flow.base_intercepts)

    win_idx = np.argmax(probas, axis=1)
    expected_win_jll = jll[np.arange(30), win_idx]
    reconstructed_win_jll = np.sum(contribs, axis=1) + log_priors[win_idx]
    np.testing.assert_allclose(reconstructed_win_jll, expected_win_jll, atol=1e-4)

    # Verify 3D Bivariate Bell Hills Mesh
    mesh = adapter.compute_surface_mesh(
        axis_x_idx=2, axis_y_idx=3,
        x_range=(1.0, 7.0), y_range=(0.1, 2.5),
        fixed_features=np.mean(X, axis=0), grid_resolution=50, target_class_idx=1
    )
    hills = np.asarray(mesh.auxiliary_grids["bell_hills_per_class"])
    assert hills.shape == (3, 50, 50)
    # All normalized bell heights must be bounded within (0, max_prior]
    assert np.all((hills >= 0.0) & (hills <= np.max(model.class_prior_) + 1e-5))
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `GaussianBellHills3D` และ `GaussianNBInspector`
1. **Multi-Class Bell Hill Rendering & Vertical Density Drop-Pins**:
   - ตรวจสอบว่า `GaussianBellHills3D` วาดพื้นผิวภูเขาระฆังคว่ำครบตามจำนวนคลาสที่เปิดแสดงผลใน `Visible Hills Checkboxes`
   - ตรวจสอบการคำนวณความสูงของเข็มวัดความหนาแน่นแนวตั้ง (Vertical Density Drop-Pins) ณ พิกัด $(x_t, y_t)$ ของทุกคลาส เพื่อให้เห็นว่าเข็มของคลาสใดสูงที่สุด
2. **Low-Density Outlier Detection Badge (`> 4.0σ`)**:
   - เมื่อจุดข้อมูลในเฟรมปัจจุบันหรือลูกบอลเงาในโหมด What-If อยู่ห่างจากยอดเขาของคลาสที่ชนะเกิน `4.0σ` แผง `GaussianNBInspector` ต้องแสดงป้ายเตือน `"LOW DENSITY OUTLIER: Far from all Gaussian centers (> 4σ)"`

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/surfaces/__tests__/GaussianBellHills3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GaussianNBInspector } from "@/domains/hud/components/inspectors/GaussianNBInspector";
import { evaluateBivariateBellHeight } from "@/domains/stage3d/components/surfaces/GaussianBellHills3D";

describe("Phase 24: GaussianBellHills3D & Outlier Z-Score Inspector", () => {
  it("computes exact peak height equal to class prior at (mu_x, mu_y) and decays symmetrically", () => {
    const prior = 0.4;
    const mu: [number, number] = [2.5, -1.0];
    const std: [number, number] = [0.5, 1.2];

    const hPeak = evaluateBivariateBellHeight(2.5, -1.0, mu, std, prior);
    expect(hPeak).toBeCloseTo(0.4, 6);

    // At 1 sigma along X and 0 sigma along Y: height = prior * exp(-0.5)
    const h1SigmaX = evaluateBivariateBellHeight(3.0, -1.0, mu, std, prior);
    expect(h1SigmaX).toBeCloseTo(0.4 * Math.exp(-0.5), 6);
  });

  it("displays LOW DENSITY OUTLIER warning when winning class Z-score exceeds 4.0 sigma", () => {
    render(
      <GaussianNBInspector
        frameIndex={74}
        probabilities={[0.01, 0.99]}
        jointLogProbs={[-45.2, -39.8]}
        winningZScores={[5.4, 6.1]}
        featureNames={["petal_length", "petal_width"]}
        contributions={[1.2, 4.2]}
        baseIntercept={0.0}
      />
    );
    expect(screen.getByText(/LOW DENSITY OUTLIER: Far from all Gaussian centers/i)).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **Analytical GPU Vertex Shader Uniform Update Test**:
   - ยืนยันว่าเมื่อผู้ใช้สลับแกน $(X, Y)$ หรือขยับ What-If Slider คอมโพเนนต์ `GaussianBellHills3D` อัปเดตเฉพาะ Shader Uniforms `uMeans2D`, `uStds2D`, และ `uPriors` บน GPU โดยตรง ไม่ต้องสร้าง `BufferGeometry` ใหม่
2. **Many-Class Overdraw Limiter Test ($C > 8$)**:
   - จำลองโมเดลที่มี `20` คลาส ตรวจสอบว่า `GaussianBellHills3D` เรนเดอร์เฉพาะภูเขาของ **Top-5 Classes** ที่มีความน่าจะเป็นสูงสุดเพื่อป้องกัน GPU Transparency Overdraw

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **โมเดลจาก Scikit-learn เวอร์ชันเก่ามี `sigma_` แทน `var_`** | ลบ `model.var_` จำลองและตั้งค่า `model.sigma_ = ...` ใน Unit Test | `getattr(model, "var_", getattr(model, "sigma_", None))` ดึงค่าความแปรปรวนได้สำเร็จโดยไม่เกิด `AttributeError` |
| 2 | **Numerical Underflow เมื่อจำนวนฟีเจอร์สูงมาก ($D = 200$)** | เทรน `GaussianNB` บนข้อมูล 200 ฟีเจอร์ แล้วรัน `trace_internal_flow` | คำนวณผ่าน `_per_feature_log_likelihood` ในสเกล Log เสมอ ไม่เกิดค่า `0.0` จาก Underflow |
| 3 | **ฟีเจอร์มีค่าคงที่ในบางคลาส ($\sigma_{k,j}^2 = 0$)** | สร้างข้อมูลที่คอลัมน์ 0 มีค่า `5.0` เท่ากันทุกแถวในคลาส 0 | `np.maximum(raw_var, 1e-12)` ป้องกันการหารด้วยศูนย์ในสมการ Log-Likelihood และ Z-Score |
| 4 | **จุดข้อมูล Outlier อยู่ไกลเกิน $10\sigma$ แต่ `predict_proba` ให้ `1.000`** | ส่งจุด `[100.0, 100.0, 100.0, 100.0]` เข้า Adapter | `winning_class_z_scores` รายงานค่า `> 10.0` และ HUD แสดงป้ายเตือน `LOW DENSITY OUTLIER (> 4σ)` |
| 5 | **ส่วนเบี่ยงเบนมาตรฐานแคบมาก ($\sigma = 0.01$) ทำให้ค่า PDF ดิบเกิน `1.0`** | ตั้งค่า `var_ = [[0.0001, 0.0001]]` แล้วเรียก `compute_surface_mesh` | ยอดเขาใน `bell_hills_per_class` ถูก Normalize ด้วย $\pi_k \exp(-z_x - z_y)$ จึงมีความสูงอยู่ในช่วง $[0, \pi_k]$ เสมอ ไม่ทะลุเพดานฉาก 3D |
| 6 | **ผู้ใช้กำหนด `priors` เองตอนเทรน (`GaussianNB(priors=[0.1, 0.9])`)** | เทรนโมเดลด้วย `priors=[0.1, 0.9]` บนข้อมูลที่จำนวนคลาสเท่ากัน 50:50 | ใช้ `model.class_prior_` (`[0.1, 0.9]`) ในการคำนวณความสูงภูเขาและ `base_intercepts` อย่างถูกต้อง |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Log-Likelihood Parity SLA**: ผลรวมของ `_per_feature_log_likelihood` บวก `log_priors` ต้องตรงกับ `model.predict_joint_log_proba(X)` ทุกแถว (`atol=1e-5`)
- [ ] **GPU Shader SLA**: การเรนเดอร์ภูเขาระฆังคว่ำซ้อนกัน 5 คลาสที่ความละเอียด $60 \times 60$ ต้องรักษาเฟรมเรต `60 FPS`
- [ ] **Outlier Alert Gate**: จุดที่มี Z-Score เกิน `4.0σ` ต้องแสดงป้ายเตือน Outlier ใน HUD เสมอ
- [ ] **Coverage Gate**: Code Coverage ของ `gaussian_nb.py`, `GaussianBellHills3D.tsx`, และ `GaussianNBInspector.tsx` ไม่ต่ำกว่า `95%`
