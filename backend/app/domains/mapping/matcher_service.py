"""3-tier fuzzy auto-matcher: exact -> normalized -> positional/fuzzy (Phase 05 §4.3)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.domains.dataset.value_objects import ColumnProfile
from app.domains.mapping.value_objects import (
    FeatureMappingItem,
    ImputationStrategy,
    MatchTier,
)
from app.domains.model_engine.value_objects import FeatureRequirement


class FuzzyColumnMatcherService:
    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower().strip())

    def _default_fill(
        self, col: ColumnProfile | None, strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    ) -> float | None:
        if col is None:
            return None
        if strategy == ImputationStrategy.MEAN and col.mean_value is not None:
            return float(col.mean_value)
        if col.median_value is not None:
            return float(col.median_value)
        if col.mean_value is not None:
            return float(col.mean_value)
        return 0.0

    def auto_match(
        self,
        features: list[FeatureRequirement],
        columns: list[ColumnProfile],
    ) -> list[FeatureMappingItem]:
        numeric_cols = [c for c in columns if c.logical_type in ("numeric", "boolean")]
        col_by_name = {c.name: c for c in columns}
        col_by_norm: dict[str, ColumnProfile] = {}
        for c in columns:
            col_by_norm.setdefault(self._normalize(c.name), c)

        # Positional fallback only makes sense when the model never saw real
        # names (feature_0..n) — otherwise positional guessing would silently
        # mis-wire semantically named models.
        items: list[FeatureMappingItem] = []
        for feat in sorted(features, key=lambda f: f.index):
            matched: ColumnProfile | None = None
            tier = MatchTier.UNMAPPED
            conf = 0.0

            if feat.name in col_by_name:
                matched = col_by_name[feat.name]
                tier, conf = MatchTier.EXACT, 1.0
            elif self._normalize(feat.name) in col_by_norm:
                matched = col_by_norm[self._normalize(feat.name)]
                tier, conf = MatchTier.NORMALIZED, 0.95
            elif feat.is_auto_generated_name and feat.index < len(numeric_cols):
                matched = numeric_cols[feat.index]
                tier, conf = MatchTier.POSITIONAL, 0.80
            else:
                best_ratio = 0.0
                best: ColumnProfile | None = None
                feat_norm = self._normalize(feat.name)
                for cand in columns:
                    ratio = SequenceMatcher(None, feat_norm, self._normalize(cand.name)).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best = cand
                if best is not None and best_ratio >= 0.65:
                    matched = best
                    tier, conf = MatchTier.FUZZY, round(float(best_ratio), 2)
                else:
                    matched = None

            fill_val = self._default_fill(matched)
            items.append(
                FeatureMappingItem(
                    feature_index=feat.index,
                    model_feature_name=feat.name,
                    dataset_column_name=matched.name if matched else None,
                    match_tier=tier,
                    match_confidence=conf,
                    imputation_strategy=ImputationStrategy.MEDIAN,
                    resolved_fill_value=fill_val,
                    is_type_compatible=matched.logical_type in ("numeric", "boolean")
                    if matched
                    else False,
                    cast_warning_or_error=None
                    if (matched and matched.logical_type in ("numeric", "boolean"))
                    else (
                        None
                        if matched is None
                        else f"Column '{matched.name}' is {matched.logical_type}; will attempt numeric coercion"
                    ),
                )
            )
        return items

    def map_by_order(
        self,
        features: list[FeatureRequirement],
        columns: list[ColumnProfile],
    ) -> list[FeatureMappingItem]:
        """Map by CSV column order 0..N (useful when neither side has names)."""
        items: list[FeatureMappingItem] = []
        for feat in sorted(features, key=lambda f: f.index):
            col = columns[feat.index] if feat.index < len(columns) else None
            fill_val = self._default_fill(col)
            items.append(
                FeatureMappingItem(
                    feature_index=feat.index,
                    model_feature_name=feat.name,
                    dataset_column_name=col.name if col else None,
                    match_tier=MatchTier.POSITIONAL if col else MatchTier.UNMAPPED,
                    match_confidence=0.80 if col else 0.0,
                    imputation_strategy=ImputationStrategy.MEDIAN,
                    resolved_fill_value=fill_val,
                    is_type_compatible=col.logical_type in ("numeric", "boolean") if col else False,
                )
            )
        return items
