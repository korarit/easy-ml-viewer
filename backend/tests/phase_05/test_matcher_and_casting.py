"""Phase 05 — matcher + casting unit tests (offline, no HTTP)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.domains.dataset.value_objects import ColumnProfile
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.mapping.matcher_service import FuzzyColumnMatcherService
from app.domains.mapping.value_objects import ImputationStrategy
from app.domains.model_engine.value_objects import FeatureRequirement


def _col(name: str, logical: str = "numeric", median: float | None = 1.0) -> ColumnProfile:
    return ColumnProfile(
        index=0,
        name=name,
        logical_type=logical,  # type: ignore[arg-type]
        storage_dtype="float32",
        null_count=0,
        null_percentage=0.0,
        unique_count=10,
        median_value=median,
        mean_value=median,
    )


def _feat(index: int, name: str, auto: bool = False) -> FeatureRequirement:
    return FeatureRequirement(index=index, name=name, is_auto_generated_name=auto)


def test_exact_match() -> None:
    svc = FuzzyColumnMatcherService()
    items = svc.auto_match(
        [_feat(0, "area_sqm")],
        [_col("area_sqm"), _col("bedrooms")],
    )
    assert items[0].match_tier == "exact"
    assert items[0].match_confidence == 1.0
    assert items[0].dataset_column_name == "area_sqm"


def test_normalized_match() -> None:
    svc = FuzzyColumnMatcherService()
    items = svc.auto_match([_feat(0, "area_sqm")], [_col("Area_SQM")])
    assert items[0].match_tier == "normalized"
    assert items[0].dataset_column_name == "Area_SQM"


def test_fuzzy_match_and_unmapped() -> None:
    svc = FuzzyColumnMatcherService()
    items = svc.auto_match(
        [_feat(0, "distance_bts_km")],
        [_col("dist_bts_km"), _col("unrelated_xyz")],
    )
    assert items[0].match_tier == "fuzzy"
    assert items[0].dataset_column_name == "dist_bts_km"
    items2 = svc.auto_match([_feat(0, "totally_different_feature_qqq")], [_col("aaa"), _col("bbb")])
    assert items2[0].dataset_column_name is None
    assert items2[0].match_tier == "unmapped"


def test_positional_fallback_for_auto_names() -> None:
    svc = FuzzyColumnMatcherService()
    items = svc.auto_match(
        [_feat(0, "feature_0", True), _feat(1, "feature_1", True)],
        [_col("a"), _col("b")],
    )
    assert [i.dataset_column_name for i in items] == ["a", "b"]
    assert all(i.match_tier == "positional" for i in items)


def test_map_by_order() -> None:
    svc = FuzzyColumnMatcherService()
    items = svc.map_by_order([_feat(0, "x"), _feat(1, "y")], [_col("c1"), _col("c2"), _col("c3")])
    assert [i.dataset_column_name for i in items] == ["c1", "c2"]


def test_build_float32_matrix_strict_order_and_bool_and_currency() -> None:
    from app.domains.mapping.value_objects import FeatureMappingItem, MatchTier

    df = pd.DataFrame(
        {
            # Intentionally out of order vs feature_index to prove strict ordering.
            "bedrooms": pd.Series([2, 3], dtype="int32"),
            "area_sqm": ["฿1,250.50", "2,000"],
            "flag": pd.Series([True, False], dtype="bool"),
        }
    )
    items = [
        FeatureMappingItem(
            feature_index=0,
            model_feature_name="area_sqm",
            dataset_column_name="area_sqm",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
            imputation_strategy=ImputationStrategy.MEDIAN,
            resolved_fill_value=0.0,
        ),
        FeatureMappingItem(
            feature_index=1,
            model_feature_name="bedrooms",
            dataset_column_name="bedrooms",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
            imputation_strategy=ImputationStrategy.MEDIAN,
            resolved_fill_value=0.0,
        ),
        FeatureMappingItem(
            feature_index=2,
            model_feature_name="flag",
            dataset_column_name="flag",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
            imputation_strategy=ImputationStrategy.MEDIAN,
            resolved_fill_value=0.0,
        ),
    ]
    m = TypeCastingEngine.build_float32_matrix(df, items)
    assert m.dtype == np.float32
    assert m.shape == (2, 3)
    assert abs(float(m[0, 0]) - 1250.50) < 1e-3
    assert float(m[1, 0]) == 2000.0
    assert float(m[0, 2]) == 1.0 and float(m[1, 2]) == 0.0


def test_dry_run_detects_text_column_and_duplicates_and_leakage() -> None:
    from app.domains.mapping.value_objects import FeatureMappingItem, MatchTier

    df = pd.DataFrame(
        {
            "area": [1.0, 2.0, 3.0, 4.0],
            "hood": ["Chatuchak", "Siam", "Thonglor", "Asoke"],
        }
    )
    items = [
        FeatureMappingItem(
            feature_index=0,
            model_feature_name="area_sqm",
            dataset_column_name="area",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
        ),
        FeatureMappingItem(
            feature_index=1,
            model_feature_name="bedrooms",
            dataset_column_name="hood",
            match_tier=MatchTier.MANUAL,
            match_confidence=1.0,
        ),
    ]
    report = TypeCastingEngine.dry_run_validate(df, items)
    assert not report.is_valid
    assert 1 in report.feature_errors

    dup_items = [
        FeatureMappingItem(
            feature_index=0, model_feature_name="a", dataset_column_name="area",
            match_tier=MatchTier.MANUAL, match_confidence=1.0,
        ),
        FeatureMappingItem(
            feature_index=1, model_feature_name="b", dataset_column_name="area",
            match_tier=MatchTier.MANUAL, match_confidence=1.0,
        ),
    ]
    dup_report = TypeCastingEngine.dry_run_validate(df, dup_items)
    assert dup_report.duplicate_columns_used == ["area"]

    leak_report = TypeCastingEngine.dry_run_validate(df, dup_items[:1], target_column_name="area")
    assert leak_report.target_leakage_warning is not None
