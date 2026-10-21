# Test Phase 05: Mapping Domain — Fuzzy Auto-Matcher, Type Casting Engine & Interactive Mapper UI

## 1. วัตถุประสงค์และขอบเขตการทดสอบของเฟส (Test Objective & Scope — Synced with Dev Phase 05)
แผนการทดสอบนี้จับคู่แบบ **1-ต่อ-1 (1:1 Synced)** กับเอกสาร [`phase-05-column-mapping-and-type-casting.md`](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-05-column-mapping-and-type-casting.md) เพื่อยืนยันความถูกต้องของ Bounded Context **`domains/mapping/`** และหน้าจอ **Interactive Column Mapper UI (Step 3)** ครอบคลุมอัลกอริทึมจับคู่คอลัมน์อัตโนมัติ 4 ระดับ (`Exact`, `Normalized`, `Fuzzy SequenceMatcher`, `Positional Fallback`), การแปลงสตริงตัวเลขมีลูกน้ำ/สกุลเงิน (`"฿1,250.50"` -> `1250.50`), การเติมค่าว่าง (`Zero`, `Mean`, `Median`, `Custom`, `Pass NaN`), การรับประกันลำดับ `feature_index` ของเมทริกซ์ `X_float32` 100%, และการตรวจสอบล่วงหน้ากับข้อมูล 100 แถวแรก (`DryRunValidationReport`)

### ขอบเขตโมดูลที่ถูกทดสอบ (Target Modules Under Test)
- **Backend Mapping Domain**:
  - `app.domains.mapping.value_objects` (`MatchTier`, `ImputationStrategy`, `FeatureMappingItem`, `DryRunValidationReport`)
  - `app.domains.mapping.matcher_service.FuzzyColumnMatcherService`
  - `app.domains.mapping.casting_service.TypeCastingEngine` (`build_float32_matrix`)
  - REST Endpoints: `POST /api/v1/mappings/auto-suggest`, `POST /api/v1/mappings/validate-and-save`
- **Frontend Mapping Domain**:
  - `src/domains/mapping/store/useMappingStore.ts`
  - `MappingHeaderBar.tsx`, `FeatureMappingRowCard.tsx`, `TargetColumnSelectorCard.tsx`, `DryRunValidationDrawer.tsx`

---

## 2. รายการทดสอบฝั่ง Backend Unit, Domain & API Integration Tests (`pytest` & `httpx`)

### 2.1 โครงสร้างไฟล์ทดสอบ Backend
```text
backend/tests/phase_05/
├── conftest.py
├── test_fuzzy_column_matcher_service.py
├── test_type_casting_and_feature_ordering.py
├── test_dry_run_matrix_validator.py
└── test_mapping_api_endpoints.py
```

### 2.2 โค้ดต้นแบบการทดสอบ Backend (`pytest` + `httpx.AsyncClient`)
```python
# backend/tests/phase_05/test_fuzzy_column_matcher_service.py
from app.domains.dataset.value_objects import ColumnProfile
from app.domains.model_engine.value_objects import FeatureRequirement
from app.domains.mapping.matcher_service import FuzzyColumnMatcherService
from app.domains.mapping.value_objects import MatchTier

def _make_col(idx: int, name: str, ltype: str = "numeric", median: float = 50.0) -> ColumnProfile:
    return ColumnProfile(
        index=idx,
        name=name,
        logical_type=ltype,  # type: ignore[arg-type]
        storage_dtype="float32",
        null_count=0,
        null_percentage=0.0,
        unique_count=10,
        median_value=median,
    )

def test_four_tier_auto_matching_accuracy() -> None:
    """ทดสอบการจับคู่อัตโนมัติครบทั้ง Exact, Normalized, Fuzzy (>=0.65) และ Positional Fallback"""
    features = [
        FeatureRequirement(index=0, name="area_sqm", is_auto_generated_name=False),
        FeatureRequirement(index=1, name="bed_rooms", is_auto_generated_name=False),
        FeatureRequirement(index=2, name="distance_bts_km", is_auto_generated_name=False),
        FeatureRequirement(index=3, name="feature_3", is_auto_generated_name=True),
    ]
    columns = [
        _make_col(0, "area_sqm", median=80.0),         # Exact Match
        _make_col(1, "BedRooms", median=2.0),          # Normalized Match
        _make_col(2, "dist_bts_km", median=3.5),       # Fuzzy Match
        _make_col(3, "building_age_yr", median=10.0),  # Positional Fallback for feature_3
    ]

    matcher = FuzzyColumnMatcherService()
    mapped = matcher.auto_match(features, columns)

    assert len(mapped) == 4
    assert mapped[0].match_tier == MatchTier.EXACT and mapped[0].dataset_column_name == "area_sqm"
    assert mapped[1].match_tier == MatchTier.NORMALIZED and mapped[1].dataset_column_name == "BedRooms"
    assert mapped[2].match_tier == MatchTier.FUZZY and mapped[2].dataset_column_name == "dist_bts_km"
    assert mapped[3].match_tier == MatchTier.POSITIONAL and mapped[3].dataset_column_name == "building_age_yr"
```

```python
# backend/tests/phase_05/test_type_casting_and_feature_ordering.py
import numpy as np
import pandas as pd
import pytest
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.mapping.value_objects import FeatureMappingItem, ImputationStrategy, MatchTier

def test_strict_feature_index_ordering_and_currency_string_cleaning() -> None:
    """
    ทดสอบกฎเหล็ก: แม้คอลัมน์ใน CSV จะเรียงสลับกัน [dist, bedrooms, area]
    และส่งรายการ items เข้ามาแบบสลับลำดับ แต่เมทริกซ์ผลลัพธ์ต้องเรียงตาม feature_index (0, 1, 2) เสมอ!
    พร้อมล้างสัญลักษณ์ ฿, ลูกน้ำ, แปลง bool -> 0.0/1.0 และเติมค่า NaN ตามกลยุทธ์ที่เลือก
    """
    df_csv = pd.DataFrame({
        "dist_col": [2.5, np.nan],
        "is_corner_bool": [True, False],
        "area_formatted_str": ["฿1,250.50", "800.00"],
    })

    # จงใจส่งรายการเข้ามาแบบไม่เรียงลำดับ index (ส่ง 2, 0, 1)
    unordered_items = [
        FeatureMappingItem(
            feature_index=2,
            model_feature_name="distance_bts",
            dataset_column_name="dist_col",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
            imputation_strategy=ImputationStrategy.MEDIAN,
            resolved_fill_value=4.2,
        ),
        FeatureMappingItem(
            feature_index=0,
            model_feature_name="area_sqm",
            dataset_column_name="area_formatted_str",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
            imputation_strategy=ImputationStrategy.ZERO,
        ),
        FeatureMappingItem(
            feature_index=1,
            model_feature_name="is_corner",
            dataset_column_name="is_corner_bool",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
            imputation_strategy=ImputationStrategy.ZERO,
        ),
    ]

    mat = TypeCastingEngine.build_float32_matrix(df_csv, unordered_items)
    assert mat.dtype == np.float32
    assert mat.shape == (2, 3)
    assert mat.flags["C_CONTIGUOUS"]

    # คอลัมน์ที่ 0 ต้องเป็น area_sqm (1250.50, 800.0)
    assert np.allclose(mat[:, 0], [1250.50, 800.0])
    # คอลัมน์ที่ 1 ต้องเป็น is_corner (1.0, 0.0)
    assert np.allclose(mat[:, 1], [1.0, 0.0])
    # คอลัมน์ที่ 2 ต้องเป็น distance_bts (2.5, และ NaN ถูกเติมด้วย Median = 4.2)
    assert np.allclose(mat[:, 2], [2.5, 4.2])
```

---

## 3. รายการทดสอบฝั่ง Frontend UI Component, Zustand Store & UX Flow Tests (`vitest` & `@testing-library/react`)

### 3.1 โครงสร้างไฟล์ทดสอบ Frontend
```text
frontend/src/domains/mapping/__tests__/
├── useMappingStore.test.ts
├── FeatureMappingRowCard.test.tsx
└── TargetColumnSelectorCard.test.tsx
```

### 3.2 โค้ดต้นแบบการทดสอบ Frontend (`vitest` + `@testing-library/react`)
```tsx
// frontend/src/domains/mapping/__tests__/FeatureMappingRowCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { FeatureMappingRowCard } from '../components/FeatureMappingRowCard';
import { useMappingStore } from '../store/useMappingStore';

describe('Step 3 Interactive Feature Mapping Card & Validation States', () => {
  beforeEach(() => {
    useMappingStore.setState({
      mappingId: 'map_01',
      isValid: false,
      duplicateColumns: ['area_sqm'],
      targetColumnName: 'selling_price',
      targetLabelMap: null,
      previewMatrixHead: [],
      items: [
        {
          feature_index: 0,
          model_feature_name: 'area_sqm',
          dataset_column_name: 'district_name',
          match_tier: 'manual',
          match_confidence: 1.0,
          imputation_strategy: 'median',
          custom_fill_value: 0,
          resolved_fill_value: null,
          is_type_compatible: false,
          cast_warning_or_error: 'Column "district_name" contains non-numeric text ("Chatuchak")',
        },
      ],
    });
  });

  it('แสดงป้าย Type Incompatible Error สีแดงเมื่อเลือกคอลัมน์ข้อความล้วน และล็อกปุ่มไปต่อ', () => {
    render(<FeatureMappingRowCard item={useMappingStore.getState().items[0]} availableColumns={[]} />);
    expect(screen.getByText(/Type Incompatible Error/i)).toBeInTheDocument();
    expect(screen.getByText(/Chatuchak/i)).toBeInTheDocument();
  });
});
```

---

## 4. การทดสอบหน่วยความจำและทรัพยากรเครื่อง (Memory Check, VRAM Disposal & Process Resource Optimization Tests)

### 4.1 โค้ดทดสอบการดึงเฉพาะคอลัมน์ที่ถูกเลือกจาก Parquet (`Column-Selective Projection`) (`pytest`)
```python
# backend/tests/phase_05/test_column_selective_parquet_projection.py
import numpy as np
import pandas as pd
from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository

def test_parquet_projection_reads_only_mapped_columns(tmp_path) -> None:
    """ทดสอบว่าแม้ไฟล์ Parquet จะมี 100 คอลัมน์ แต่ตอนทำ Dry-Run และสร้าง Matrix จะโหลดขึ้นมาเพียง 3 คอลัมน์ที่ Map ไว้"""
    repo = ParquetDatasetRepository(cache_dir=tmp_path)
    wide_data = {f"col_{i}": np.random.rand(1000).astype(np.float32) for i in range(100)}
    df_wide = pd.DataFrame(wide_data)
    repo.save_dataset_parquet("ds_wide", df_wide)

    projected_df = repo.read_columns_mmap("ds_wide", columns=["col_5", "col_42", "col_99"], offset=0, limit=100)
    assert list(projected_df.columns) == ["col_5", "col_42", "col_99"]
    assert projected_df.shape == (100, 3)
```

---

## 5. การทดสอบตาราง Edge Cases ทั้งหมด (Edge Case Verification Matrix — Synced with Dev Phase 05)

| ลำดับ | เคสปัญหาจากแผนพัฒนา (Dev Edge Case) | ชื่อฟังก์ชันทดสอบ (Test Case Function) | วิธีจำลองสถานการณ์ (Simulation Method) | ผลลัพธ์ที่ต้องผ่าน (Expected Assertion) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **ลำดับคอลัมน์สลับกันทำให้ผลทำนายเพี้ยนโดยไม่เกิด Error** | `test_strict_feature_index_ordering_and_currency_string_cleaning()` | ส่ง `items` ที่เรียง `feature_index` สลับกัน `[2, 0, 1]` เข้า `build_float32_matrix()` | เมทริกซ์ `X_float32` ถูกจัดเรียงตาม `feature_index` `[0, 1, 2]` อย่างถูกต้อง 100% |
| **2** | **ตัวเลขใน CSV มีลูกน้ำหรือสกุลเงิน (`"1,250,000"`, `"฿450"`)** | `test_strict_feature_index_ordering_and_currency_string_cleaning()` | ส่งคอลัมน์ที่มีค่า `"฿1,250.50"` และ `"85%"` เข้าแปลง | ล้างสัญลักษณ์ออกและแปลงเป็น `1250.50` (`float32`) ได้โดยไม่กลายเป็น `NaN` |
| **3** | **คอลัมน์ที่เป็นข้อความล้วนถูกเลือกแล้วกลายเป็น `NaN` ทั้งคอลัมน์หลัง `coerce`** | `test_pure_text_column_rejected_by_dry_run_validator()` | จับคู่ `area_sqm` เข้ากับคอลัมน์ชื่อเขต `"Chatuchak", "Bangrak"` | Dry-Run Validator ฟ้อง `is_valid=False` และระบุ `feature_errors[0]` ทันที |
| **4** | **ผู้ใช้เผลอเลือกคอลัมน์คำตอบ (`Target Column`) มาเป็นหนึ่งใน Input Features** | `test_target_leakage_warning_when_target_mapped_as_feature()` | เลือก `selling_price` เป็นทั้ง `Feature #0` และ `Actual Target Column` | ระบบคืนค่าคำเตือน `Target Leakage Warning` ให้แสดงแถบสีส้มบน UI |
| **5** | **โมเดลไม่รองรับค่า `NaN` แต่ผู้ใช้เลือก `Pass NaN directly`** | `test_disallow_pass_nan_for_non_nan_tolerant_models()` | เลือก `ImputationStrategy.PASS_NAN` กับโมเดล `LinearRegression` | Validator ปฏิเสธการตั้งค่าพร้อมแจ้งว่า `LinearRegression does not support raw NaN inputs` |
| **6** | **ผู้ใช้เลือกคอลัมน์ CSV เดียวกันให้กับ 2 Features ของโมเดล** | `test_detect_duplicate_dataset_columns_across_features()` | จับคู่ทั้ง `Feature #0` และ `Feature #1` ไปที่คอลัมน์ `area_sqm` เดียวกัน | `DryRunValidationReport.duplicate_columns_used` มีค่า `["area_sqm"]` เพื่อแสดงกรอบสีเหลืองเตือน |

---

## 6. เกณฑ์ผ่านการทดสอบ (Pass/Fail SLA & Gate Criteria)
- [ ] **100% Feature Index Alignment**: ทดสอบยืนยันว่าไม่ว่าลำดับคอลัมน์ใน CSV จะสลับอย่างไร คอลัมน์ที่ `i` ของ `X_float32` จะตรงกับ `FeatureRequirement.index == i` เสมอ
- [ ] **Dry-Run Validation Speed SLA**: การตรวจสอบ `validate-and-save` บนข้อมูล 100 แถวแรกต้องใช้เวลา `< 15 ms`
- [ ] **Contiguous Memory Guarantee**: เมทริกซ์ผลลัพธ์จาก `build_float32_matrix()` ต้องมี `dtype == np.float32` และ `flags['C_CONTIGUOUS'] == True` เสมอ
