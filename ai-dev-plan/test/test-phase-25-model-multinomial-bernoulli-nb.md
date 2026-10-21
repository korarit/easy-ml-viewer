# Test Blueprint — Phase 25: Multinomial & Bernoulli Naive Bayes Adapters & 3D Log-Likelihood Towers

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-25-model-multinomial-bernoulli-nb.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-25-model-multinomial-bernoulli-nb.md) เพื่อยืนยันความถูกต้องของคลาส `DiscreteNBAdapter` (`backend/app/domains/model_engine/adapters/probabilistic/discrete_nb.py`), คอมโพเนนต์หอคอยความน่าจะเป็น `ProbabilityTowers3D` (`frontend/src/domains/stage3d/components/actors/ProbabilityTowers3D.tsx`), และแผงวิเคราะห์ `DiscreteNBInspector` (`frontend/src/domains/hud/components/inspectors/DiscreteNBInspector.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทั้ง 4 คลาส**:
  - `sklearn.naive_bayes.MultinomialNB`
  - `sklearn.naive_bayes.BernoulliNB` (ทั้งกรณี `binarize=0.0`, `binarize=0.5`, และ `binarize=None`)
  - `sklearn.naive_bayes.ComplementNB`
  - `sklearn.naive_bayes.CategoricalNB`
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - สมการเชิงเส้นในสเกล Log ของ `MultinomialNB`: $J_{t,k} = \log \pi_k + \sum_{j=1}^{D} x_{t,j} \log \theta_{k,j}$
  - สมการรวมเทอมปรากฏ ($b_{t,j}=1$) และเทอมไม่ปรากฏ ($b_{t,j}=0$) ของ `BernoulliNB`:
    $$J_{t,k} = \log \pi_k + \sum_{j=1}^{D}\left[ b_{t,j}\log\theta_{k,j} + (1 - b_{t,j})\log(1 - \theta_{k,j}) \right]$$
  - การบังคับขอบเขตโดเมนไม่ติดลบ ($X \ge 0, Y \ge 0$) ใน `compute_surface_mesh` และ `trace_internal_flow` เพื่อป้องกัน `ValueError: Negative values in data passed to MultinomialNB`

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **BernoulliNB Exact Presence + Absence Additivity**:
   - ยืนยันว่าใน `BernoulliNB` ผลรวมของ `feature_contributions` (ซึ่งรวมทั้งเทอมที่ $b_{t,j}=1$ และ $b_{t,j}=0$) บวกกับ `base_intercepts` เท่ากับ `predict_joint_log_proba` (หรือผลต่าง $J_{t,1} - J_{t,0}$ ในกรณี Binary) ด้วยความแม่นยำ `1e-5`!
2. **Non-Negative Grid Clamping Safety**:
   - เมื่อเรียก `compute_surface_mesh(..., x_range=(-5.0, 10.0), y_range=(-3.0, 8.0))` บน `MultinomialNB`, `ComplementNB`, หรือ `CategoricalNB` ระบบต้องปรับ `x_grid[0] == 0.0` และ `y_grid[0] == 0.0` อัตโนมัติโดยไม่โยน `ValueError`
3. **CategoricalNB Out-of-Bounds Category Clamping**:
   - เมื่อส่งค่าหมวดหมู่ที่สูงกว่าตอนเทรน (เช่น `99`) เข้า `CategoricalNB` ระบบต้องใช้ `np.clip` ป้องกัน `IndexError` อย่างปลอดภัย

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_discrete_nb_adapter.py`)
```python
import numpy as np
import pytest
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from app.domains.model_engine.adapters.probabilistic.discrete_nb import DiscreteNBAdapter

FEATURE_NAMES = ["word_free", "word_win", "word_meeting", "word_report"]

@pytest.fixture
def count_data():
    rng = np.random.default_rng(42)
    X = rng.poisson(lam=2.0, size=(100, 4)).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(int)
    return X, y

def test_multinomial_nb_waterfall_and_negative_grid_clamping(count_data):
    X, y = count_data
    model = MultinomialNB(alpha=1.0).fit(X, y)
    adapter = DiscreteNBAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:20])
    jll = np.asarray(flow.decision_scores)
    contribs = np.asarray(flow.feature_contributions)
    prior_diff = flow.base_intercepts[0]

    expected_jll = model.predict_joint_log_proba(X[:20])
    np.testing.assert_allclose(jll, expected_jll, atol=1e-5)
    # Binary LLR additivity: sum(x_j * (log_theta_1j - log_theta_0j)) + log(pi_1/pi_0) == J_1 - J_0
    np.testing.assert_allclose(np.sum(contribs, axis=1) + prior_diff, expected_jll[:, 1] - expected_jll[:, 0], atol=1e-5)

    # Pass negative x_range and y_range -> must clamp to >= 0.0 without ValueError
    mesh = adapter.compute_surface_mesh(
        axis_x_idx=0, axis_y_idx=1,
        x_range=(-5.0, 10.0), y_range=(-3.0, 8.0),
        fixed_features=np.array([-1.0, 2.0, 1.0, 0.0]), grid_resolution=25
    )
    assert mesh.x_grid[0] == 0.0
    assert mesh.y_grid[0] == 0.0
    assert mesh.auxiliary_grids["clamped_non_negative"] is True

@pytest.mark.parametrize("binarize_val", [0.0, 1.5, None])
def test_bernoulli_nb_presence_and_absence_exact_additivity(count_data, binarize_val):
    X, y = count_data
    X_in = (X > 1.0).astype(np.float64) if binarize_val is None else X
    model = BernoulliNB(alpha=0.5, binarize=binarize_val).fit(X_in, y)
    adapter = DiscreteNBAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X_in[:25])
    jll = np.asarray(flow.decision_scores)
    contribs = np.asarray(flow.feature_contributions)
    prior_diff = flow.base_intercepts[0]

    expected_jll = model.predict_joint_log_proba(X_in[:25])
    np.testing.assert_allclose(jll, expected_jll, atol=1e-5)
    # Verify that including (1 - b_j)*log(1 - theta_j) makes sum match exact J_1 - J_0!
    reconstructed_llr = np.sum(contribs, axis=1) + prior_diff
    np.testing.assert_allclose(reconstructed_llr, expected_jll[:, 1] - expected_jll[:, 0], atol=1e-5)

def test_complement_nb_and_categorical_nb_safety(count_data):
    X, y = count_data
    comp_model = ComplementNB(alpha=1.0).fit(X, y)
    comp_adapter = DiscreteNBAdapter(comp_model, FEATURE_NAMES)
    comp_flow = comp_adapter.trace_internal_flow(X[:10])
    assert len(comp_flow.predictions) == 10

    # CategoricalNB with unseen out-of-range category values in test batch
    X_cat = np.clip(X.astype(int), 0, 4)
    cat_model = CategoricalNB(alpha=1.0).fit(X_cat, y)
    cat_adapter = DiscreteNBAdapter(cat_model, FEATURE_NAMES)
    X_unseen = X_cat[:10].copy()
    X_unseen[0, 0] = 999  # Unseen category > max category 4
    cat_flow = cat_adapter.trace_internal_flow(X_unseen.astype(float))
    assert len(cat_flow.predictions) == 10
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `ProbabilityTowers3D` และ `DiscreteNBInspector`
1. **3D Probability Towers Height Scaling (`ProbabilityTowers3D`)**:
   - ทดสอบว่าความสูงแกน $Z$ ของเสาคริสตัลแต่ละคลาสถูกตั้งค่าด้วย `Math.max(0.01, prob)` ตามค่า `tower_heights` ของเฟรมปัจจุบัน และเสาของคลาสที่มีความน่าจะเป็นสูงสุดมีสถานะ `isWinner = true`
2. **BernoulliNB Presence vs Absence Filter & Step-Function What-If**:
   - ทดสอบสวิตช์สลับโหมด `Show Active (x>0) + Absent (x=0)` กับ `Active Only` ใน `DiscreteNBInspector`
   - ทดสอบว่าในโหมด What-If Sandbox สำหรับ `MultinomialNB` สไลเดอร์ทุกตัวถูกล็อก `min={0}` และสำหรับ `BernoulliNB` เมื่อลากสไลเดอร์ข้ามเกณฑ์ `binarize` คะแนนจะเปลี่ยนแบบ Step Function ทันที

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/actors/__tests__/ProbabilityTowers3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DiscreteNBInspector } from "@/domains/hud/components/inspectors/DiscreteNBInspector";

describe("Phase 25: DiscreteNBInspector & Bernoulli Presence/Absence Filter", () => {
  const mockMeta = {
    algorithm: "BernoulliNB",
    alpha: 1.0,
    binarize: 0.0,
    requires_non_negative_input: false,
    classes: ["Ham", "Spam"],
    class_log_prior: [-0.51, -0.915],
  };

  const mockFrame = {
    frameIndex: 104,
    probabilities: [0.088, 0.912],
    jointLogProbs: [-10.761, -8.42],
    featureValues: [1.0, 1.0, 0.0, 0.0],
    featureContributions: [2.85, 1.92, 0.64, -2.664],
  };

  it("toggles between full Presence+Absence waterfall and Active-Only (x > binarize) mode", () => {
    render(
      <DiscreteNBInspector
        metadata={mockMeta}
        frame={mockFrame}
        featureNames={["word_free", "word_win", "word_meeting", "word_report"]}
      />
    );

    // Initially shows both PRESENT and ABSENT features
    expect(screen.getByText(/word_free = 1 \(PRESENT\)/i)).toBeInTheDocument();
    expect(screen.getByText(/word_report = 0 \(ABSENT\)/i)).toBeInTheDocument();

    // Switch to Active Only
    fireEvent.click(screen.getByRole("button", { name: /Active Only/i }));
    expect(screen.getByText(/word_free = 1 \(PRESENT\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/word_report = 0 \(ABSENT\)/i)).not.toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **High-Dimensional Sparse Vocabulary Compression Test ($D = 50,000$ Words)**:
   - จำลองโมเดล `MultinomialNB` ที่เทรนด้วยคลังคำศัพท์ขนาดใหญ่ $D = 50,000$ คำ บน `scipy.sparse.csr_matrix`
   - ยืนยันว่าการสกัด Top-K Feature Contributions ไม่ทำการขยายทั้งตารางเป็น Dense Array ขนาด $(1000 \times 50000)$ และใช้หน่วยความจำต่ำกว่า `50 MB`
2. **Reuse `BoxGeometry` สำหรับ `ProbabilityTowers3D`**:
   - ยืนยันว่าการปรับความสูงของหอคอยใช้ `meshRef.current.scale.z` ใน `useFrame` โดยไม่สร้าง `new THREE.BoxGeometry` ใหม่

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **`ValueError: Negative values in data passed to MultinomialNB`** | ส่ง `x_range=(-2.0, 5.0)` หรือแถวที่มีค่า `-0.5` เข้า `MultinomialNB` / `ComplementNB` | ระบบ Clamp ค่าให้ `>= 0.0` อัตโนมัติ ไม่เกิด `ValueError` และส่ง `clamped_non_negative: true` แจ้งเตือน UI |
| 2 | **`BernoulliNB` ต้องรวมผลของฟีเจอร์ที่เป็น `0` (Absence Penalty)** | ทดสอบแถวที่ทุกฟีเจอร์เป็น `0` ทั้งหมด (`X = [[0, 0, 0, 0]]`) | `feature_contributions` คืนค่า $\log(1 - \theta_{k,j})$ ของแต่ละฟีเจอร์ และผลรวมตรงกับ `predict_joint_log_proba` 100% |
| 3 | **`CategoricalNB` เจอค่าหมวดหมู่ใน Grid/Test สูงกว่าตอนเทรน** | ส่งค่า `X = [[15]]` เข้าโมเดลที่เทรนด้วยหมวดหมู่ `0..4` | `np.clip` บีบค่าให้อยู่ในช่วง `[0, n_categories - 1]` ไม่เกิด `IndexError` |
| 4 | **`BernoulliNB(binarize=None)`** | เทรน `BernoulliNB(binarize=None)` แล้วรัน `trace_internal_flow` | ตรวจเช็ก `if thresh is not None:` ก่อนเปรียบเทียบ จึงไม่เกิด `TypeError: '>' not supported between instances of 'ndarray' and 'NoneType'` |
| 5 | **`ComplementNB` ใช้เครื่องหมายลบหน้าผลรวมน้ำหนัก** | ตรวจสอบเครื่องหมายของ `feature_contributions` ใน `ComplementNB` | คูณ `-1.0` หน้า `X_arr * self.feature_log_prob_` ทำให้ฟีเจอร์ที่สนับสนุนคลาสที่ชนะมีทิศทางสอดคล้องกับคะแนนรวม |
| 6 | **`alpha=0` ทำให้เกิด `-inf` และ `0 * -inf = nan`** | จำลอง `feature_log_prob_` ที่มีค่า `-np.inf` แล้วคูณกับ `x = 0.0` | ใช้ `np.nan_to_num(contributions, nan=0.0, neginf=-30.0)` รับประกันว่า JSON ไม่มี `NaN` หรือ `-Infinity` |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **All 4 Discrete NB Classes Gate**: ผ่านชุดทดสอบ `pytest` ครบทั้ง `MultinomialNB`, `BernoulliNB`, `ComplementNB`, และ `CategoricalNB`
- [ ] **Bernoulli Exact Additivity SLA**: ผลรวม Waterfall ของ `BernoulliNB` ต้องตรงกับ `predict_joint_log_proba` ด้วยความคลาดเคลื่อนไม่เกิน `1e-5`
- [ ] **Non-Negative Safety Gate**: ไม่มีโอกาสเกิด `ValueError: Negative values` จากการสร้าง Grid 3D หรือการลากสไลเดอร์ใน What-If Sandbox
- [ ] **Coverage Gate**: Code Coverage ของ `discrete_nb.py`, `ProbabilityTowers3D.tsx`, และ `DiscreteNBInspector.tsx` ไม่ต่ำกว่า `95%`
