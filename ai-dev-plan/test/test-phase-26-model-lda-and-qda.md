# Test Blueprint — Phase 26: LDA & QDA Adapters & 3D Covariance Ellipsoid Visualizer

## 1. วัตถุประสงค์และขอบเขตการทดสอบของโมเดลในเฟสนี้ (Test Objective & Model Scope)
เอกสารฉบับนี้กำหนดแผนการทดสอบเชิงลึกแบบ 1-ต่อ-1 สำหรับ [phase-26-model-lda-and-qda.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-26-model-lda-and-qda.md) เพื่อยืนยันความถูกต้องของคลาส `DiscriminantAnalysisAdapter` (`backend/app/domains/model_engine/adapters/probabilistic/discriminant.py`), คอมโพเนนต์ทรงรีความแปรปรวนร่วม `CovarianceEllipsoids3D` (`frontend/src/domains/stage3d/components/actors/CovarianceEllipsoids3D.tsx`), และแผงวิเคราะห์ระยะห่าง `DiscriminantInspector` (`frontend/src/domains/hud/components/inspectors/DiscriminantInspector.tsx`)

### ขอบเขตคลาสและฟีเจอร์ที่ต้องผ่านการทดสอบ
- **คลาสของ Scikit-learn ที่ต้องทดสอบครบทุกโหมด**:
  - `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (ครบทุก Solver: `'svd'`, `'lsqr'`, `'eigen'`, และทั้ง `store_covariance=True` และ `store_covariance=False`)
  - `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` (ทั้ง `store_covariance=True` และ `store_covariance=False`, รวมถึงการตั้งค่า `reg_param`)
- **ขอบเขตความสอดคล้องทางคณิตศาสตร์**:
  - การคำนวณระยะห่างกำลังสองแบบ Mahalanobis: $D_M(\mathbf{x}_t, \boldsymbol{\mu}_k) = \sqrt{(\mathbf{x}_t - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_k)}$ เทียบกับระยะห่างแบบ Euclidean $D_E(\mathbf{x}_t, \boldsymbol{\mu}_k) = \|\mathbf{x}_t - \boldsymbol{\mu}_k\|_2$
  - การตรวจจับปรากฏการณ์ **Mahalanobis vs Euclidean Paradox** (`argmin(D_M) != argmin(D_E)`)
  - การแยกตัวประกอบ Eigendecomposition (`np.linalg.eigh`) ที่บังคับ Right-handed Rotation ($\det(\mathbf{V}) = +1$)
  - การกู้คืนเมทริกซ์ความแปรปรวนร่วม $\mathbf{\Sigma}_k = (\mathbf{R}_k \mathbf{s}_k) \mathbf{R}_k^T$ ใน QDA เมื่อ `store_covariance=False`

---

## 2. การทดสอบความถูกต้องทางคณิตศาสตร์และ Internal Flow Parity Tests (Backend `pytest`)

### 2.1 รายการตรวจสอบเชิงคณิตศาสตร์ (Mathematical Verification Points)
1. **Exact QDA Covariance Reconstruction When `store_covariance=False`**:
   - ทดสอบเทรน `QuadraticDiscriminantAnalysis(store_covariance=False)` และ `QuadraticDiscriminantAnalysis(store_covariance=True)` บนข้อมูลชุดเดียวกัน ยืนยันว่า `_resolve_covariances()` กู้คืนเมทริกซ์ความแปรปรวนร่วมจาก `rotations_` และ `scalings_` ได้ตรงกับ `model.covariance_` ทุกตำแหน่ง (`np.allclose(..., atol=1e-6)`)!
2. **Right-Handed Eigenvector Basis ($\det(\mathbf{V}) = +1$)**:
   - สำหรับทุกทรงรีใน `compute_ellipsoid_2d_specs` เมทริกซ์ `eigenvectors_2d` ต้องเป็น Orthogonal Matrix ที่มี $\det(\mathbf{V}) = +1.0 \pm 10^{-6}$ เสมอ
3. **QDA Quadratic Precision Waterfall Additivity**:
   - สำหรับแถว $t$ และคลาสที่ชนะ $c$: ผลรวมของ `feature_contributions[t]` (ซึ่งคำนวณจาก $-\frac{1}{2}(x_{t,j} - \mu_{c,j})[\mathbf{\Sigma}_c^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_c)]_j$) ต้องเท่ากับ $-\frac{1}{2} D_M^2(\mathbf{x}_t, \boldsymbol{\mu}_c)$ เป๊ะ (`atol=1e-5`)

### 2.2 โค้ดต้นแบบชุดทดสอบ `pytest` (`backend/tests/domains/model_engine/test_discriminant_adapter.py`)
```python
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from app.domains.model_engine.adapters.probabilistic.discriminant import DiscriminantAnalysisAdapter

FEATURE_NAMES = ["f_len", "f_wid", "f_den", "f_curv"]

@pytest.fixture
def correlated_data():
    X, y = make_classification(
        n_samples=160, n_features=4, n_informative=4, n_redundant=0,
        n_classes=2, random_state=42
    )
    # Induce strong correlation between feature 0 and feature 1
    X[:, 1] = 0.85 * X[:, 0] + 0.3 * X[:, 1]
    return X, y

def test_qda_exact_covariance_recovery_when_store_covariance_is_false(correlated_data):
    X, y = correlated_data
    qda_stored = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
    qda_default = QuadraticDiscriminantAnalysis(store_covariance=False).fit(X, y)

    adapter_stored = DiscriminantAnalysisAdapter(qda_stored, FEATURE_NAMES)
    adapter_default = DiscriminantAnalysisAdapter(qda_default, FEATURE_NAMES)

    assert adapter_default.is_cov_approximated is False
    for k in range(2):
        np.testing.assert_allclose(
            adapter_default.covariances_per_class[k],
            adapter_stored.covariances_per_class[k],
            atol=1e-6,
        )

@pytest.mark.parametrize("solver_name,store_cov", [
    ("svd", False),
    ("svd", True),
    ("lsqr", True),
    ("eigen", True),
])
def test_lda_all_solvers_and_waterfall_additivity(correlated_data, solver_name, store_cov):
    X, y = correlated_data
    model = LinearDiscriminantAnalysis(solver=solver_name, store_covariance=store_cov).fit(X, y)
    adapter = DiscriminantAnalysisAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:20])
    scores = np.asarray(flow.decision_scores)[:, 0]
    contribs = np.asarray(flow.feature_contributions)
    b = flow.base_intercepts[0]

    np.testing.assert_allclose(scores, model.decision_function(X[:20]), atol=1e-5)
    np.testing.assert_allclose(np.sum(contribs, axis=1) + b, scores, atol=1e-5)

def test_qda_quadratic_mahalanobis_waterfall_and_right_handed_ellipsoids(correlated_data):
    X, y = correlated_data
    model = QuadraticDiscriminantAnalysis().fit(X, y)
    adapter = DiscriminantAnalysisAdapter(model, FEATURE_NAMES)

    flow = adapter.trace_internal_flow(X[:25])
    contribs = np.asarray(flow.feature_contributions)
    mahal_dists = np.asarray(flow.extra_telemetry["mahalanobis_distances"])
    probas = np.asarray(flow.probabilities)
    win_idx = np.argmax(probas, axis=1)

    # Verify sum of quadratic feature contributions == -0.5 * D_M^2 for winning class
    win_mahal_sq = mahal_dists[np.arange(25), win_idx] ** 2
    np.testing.assert_allclose(np.sum(contribs, axis=1), -0.5 * win_mahal_sq, atol=1e-5)

    # Verify 2D ellipsoid eigendecomposition specs
    mesh = adapter.compute_surface_mesh(0, 1, (-3.0, 3.0), (-3.0, 3.0), np.mean(X, axis=0), grid_resolution=30)
    ellipsoids = mesh.auxiliary_grids["ellipsoids_2d"]
    assert len(ellipsoids) == 2
    for ell in ellipsoids:
        V = np.asarray(ell["eigenvectors_2d"])
        # Must be proper right-handed rotation matrix (det == +1)
        assert np.isclose(np.linalg.det(V), 1.0, atol=1e-5)
        assert all(r > 0 for r in ell["radii_1sigma"])
```

---

## 3. การทดสอบฝั่ง Frontend 3D Visual Components, HUD Controls & What-If UX Tests (`vitest`)

### 3.1 รายการทดสอบ `CovarianceEllipsoids3D` และ `DiscriminantInspector`
1. **3D Tilted Covariance Ellipsoids (`1σ` & `2σ` Shells + Eigenvector Axes)**:
   - ทดสอบว่าในโหมด LDA แคปซูลทรงรีของทุกคลาสมี `radii_1sigma` และมุมหมุน `rotation_angle_rad` เท่ากันทุกประการ (Shared Covariance) ในขณะที่โหมด QDA แต่ละคลาสมีขนาดและมุมหมุนอิสระต่อกัน
2. **Mahalanobis vs Euclidean Paradox Alert**:
   - เมื่อ `paradox_detected[frameIndex] === true` (เช่น ระยะ Euclidean อยู่ใกล้ Class 0 กว่า แต่ระยะ Mahalanobis อยู่ใกล้ Class 1 กว่าเพราะจุดอยู่ในแนวแกนยาวของทรงรี Class 1) แผง `DiscriminantInspector` ต้องแสดงป้ายไฟ `"⚡ ACTIVE! Euclidean favors Class 0, Mahalanobis -> 1"`
3. **Directional Sensitivity in What-If Sandbox**:
   - เมื่อเลื่อนลูกบอลเงาไปตามแนวแกนหลัก (Major Eigenvector) ระยะ Mahalanobis จะเพิ่มขึ้นช้ากว่าการเลื่อนในระยะเท่ากันไปตามแนวแกนรอง (Minor Eigenvector) ตามอัตราส่วน $\sqrt{\lambda_{\text{major}} / \lambda_{\text{minor}}}$

### 3.2 โค้ดต้นแบบชุดทดสอบ `vitest` (`frontend/src/domains/stage3d/components/actors/__tests__/CovarianceEllipsoids3D.test.tsx`)
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DiscriminantInspector } from "@/domains/hud/components/inspectors/DiscriminantInspector";

describe("Phase 26: DiscriminantInspector & Mahalanobis Paradox Alert", () => {
  it("triggers Paradox Alert banner when Euclidean winner differs from Mahalanobis winner", () => {
    const mockFrame = {
      frameIndex: 182,
      probabilities: [0.154, 0.846],
      euclideanDistances: [1.89, 2.45], // Class 0 is closer in Euclidean
      mahalanobisDistances: [2.31, 1.12], // Class 1 is closer in Mahalanobis!
      paradoxDetected: true,
      contributions: [-0.31, 0.18],
      baseIntercept: -1.24,
    };

    render(
      <DiscriminantInspector
        algorithm="QDA"
        priors={[0.5, 0.5]}
        frame={mockFrame}
        featureNames={["feature_length", "feature_width"]}
      />
    );

    expect(screen.getByText(/ACTIVE! Euclidean favors Class 0, Mahalanobis -> 1/i)).toBeInTheDocument();
    expect(screen.getByText(/1\.120 \(CLOSER\)/i)).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, Downsampling & WebGL Buffer Optimization Tests)

1. **$2 \times 2$ Sub-Matrix Eigendecomposition Speed Test**:
   - ทดสอบเรียก `compute_ellipsoid_2d_specs` บนโมเดลที่มีจำนวนฟีเจอร์สูง ($D = 500$ ฟีเจอร์)
   - ยืนยันว่าการตัด Sub-matrix ขนาด $2 \times 2$ ด้วย `cov[np.ix_([axis_x_idx, axis_y_idx], [axis_x_idx, axis_y_idx])]` ก่อนเรียก `np.linalg.eigh` ใช้เวลาต่ำกว่า `0.1 ms` (เร็วกว่าการทำ Eigendecomposition บนเมทริกซ์ $500 \times 500$ กว่า $1,000\times$)
2. **Pre-computed Precision Matrix (`self.precisions_per_class`) Verification**:
   - ยืนยันว่า `np.linalg.pinv` ถูกเรียกเพียงครั้งเดียวใน `__init__` และไม่ถูกเรียกซ้ำใน `trace_internal_flow`

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix)

| ลำดับ | กรณีปัญหา (Edge Case จากแผน Dev) | วิธีจำลองในการทดสอบ (Test Stimulus) | ผลลัพธ์ที่ต้องผ่านเกณฑ์ (Expected Verification Result) |
| :---: | :--- | :--- | :--- |
| 1 | **`store_covariance=False` เป็นค่าเริ่มต้นของ LDA และ QDA** | เทรนทั้ง `LDA()` และ `QDA()` โดยไม่ส่งพารามิเตอร์ใดๆ | ไม่เกิด `AttributeError: covariance_` โดย QDA กู้คืน $\mathbf{\Sigma}_k$ ได้ตรง 100% จาก `rotations_` และ `scalings_` และ LDA ประมาณค่าจาก `pinv(scalings_)` |
| 2 | **Singular Covariance Matrix (ฟีเจอร์ซ้ำซ้อน $f_1 = 2 f_0$)** | สร้างข้อมูลที่คอลัมน์ 1 เป็นผลคูณของคอลัมน์ 0 เป๊ะๆ แล้วรัน Adapter | `np.linalg.pinv(cov + np.eye(D) * 1e-9)` ทำงานได้โดยไม่เกิด `LinAlgError: Singular matrix` |
| 3 | **Improper Rotation ($\det(\mathbf{V}) = -1$) จาก `np.linalg.eigh`** | ทดสอบเมทริกซ์ $2 \times 2$ ที่ `eigh` คืนค่าเวกเตอร์สะท้อนกลับด้าน | บล็อก `if np.linalg.det(eigvecs) < 0: eigvecs[:, 0] *= -1.0` แก้ไขให้ $\det(\mathbf{V}) = +1.0$ เสมอ |
| 4 | **`LinearDiscriminantAnalysis(solver='lsqr')` ไม่มี `scalings_` แบบ SVD** | เทรน `LDA(solver='lsqr', store_covariance=True)` | ดึง `model.covariance_` ได้โดยตรงโดยไม่พึ่งพา `scalings_` |
| 5 | **ความหมายของ `scalings_` ใน LDA vs QDA ต่างกัน** | เปรียบเทียบผลลัพธ์ `_resolve_covariances` ระหว่าง LDA และ QDA | แยกบล็อก `if self.is_qda:` ชัดเจน โดยใน QDA คูณ `(R * s) @ R.T` ได้ค่าความแปรปรวนตรงตามสถิติจริง |
| 6 | **จุดข้อมูลเกิด Mahalanobis vs Euclidean Paradox** | สร้างจุดทดสอบที่อยู่ตามแนวแกนยาวของทรงรีคลาส 1 | `extra_telemetry["paradox_detected"]` คืนค่า `True` และ HUD แสดงป้ายแจ้งเตือน Paradox ทันที |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **Covariance Recovery Gate**: `QuadraticDiscriminantAnalysis(store_covariance=False)` ต้องกู้คืนเมทริกซ์ความแปรปรวนร่วมได้ตรงกับ `store_covariance=True` ในระดับ `atol=1e-6`
- [ ] **Proper Rotation SLA**: เมทริกซ์หมุนทรงรีทุกตัวต้องมี $\det(\mathbf{V}) = +1.0$ และรัศมีเป็นค่าบวกแท้
- [ ] **Quadratic Waterfall SLA**: ผลรวมของ `feature_contributions` ใน QDA ต้องเท่ากับ $-\frac{1}{2} D_M^2$ ด้วยความคลาดเคลื่อนไม่เกิน `1e-5`
- [ ] **Coverage Gate**: Code Coverage ของ `discriminant.py`, `CovarianceEllipsoids3D.tsx`, และ `DiscriminantInspector.tsx` ไม่ต่ำกว่า `95%`
