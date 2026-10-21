"""Phase 14 — alias suggestion + normalization service (pure functions)."""

from __future__ import annotations

import re

from app.domains.studio_metadata.value_objects import ClassAliasVO, FeatureAliasVO

DEFAULT_CLASS_PALETTE = [
    "#10B981",  # emerald — Low Risk / Normal
    "#F43F5E",  # rose — High Risk / Fraud
    "#3B82F6",  # blue
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
    "#84CC16",  # lime
    "#EC4899",  # pink
]

_SPLIT_RE = re.compile(r"[_\-\s]+")


def prettify_feature_name(raw: str) -> str:
    """`rev_util_pct` -> `Rev Util Pct`; `x_01_inc` -> `X 01 Inc`."""
    cleaned = raw.strip().strip("_- ")
    if not cleaned:
        return raw
    parts = [p for p in _SPLIT_RE.split(cleaned) if p]
    titled = " ".join(p[:1].upper() + p[1:] if p else p for p in parts)
    return titled[:64] or raw


def guess_unit_suffix(raw: str) -> str:
    lowered = raw.lower()
    if any(k in lowered for k in ("pct", "percent", "rate", "ratio")):
        return "%"
    if any(k in lowered for k in ("price", "income", "amount", "cost", "balance")):
        return ""
    if any(k in lowered for k in ("age", "year")):
        return "yrs"
    return ""


def suggest_feature_alias(raw_feature_name: str) -> FeatureAliasVO:
    return FeatureAliasVO(
        raw_feature_name=raw_feature_name,
        display_name=prettify_feature_name(raw_feature_name),
        unit_suffix=guess_unit_suffix(raw_feature_name),
    )


def suggest_class_alias(raw_value: str | int, position: int = 0) -> ClassAliasVO:
    text = str(raw_value)
    color = DEFAULT_CLASS_PALETTE[position % len(DEFAULT_CLASS_PALETTE)]
    lowered = text.lower()
    if position == 0 or lowered in ("0", "normal", "negative", "no", "pass"):
        label = "Normal"
    elif position == 1 or lowered in ("1", "fraud", "positive", "yes", "fail"):
        label = "High Risk"
    else:
        label = f"Class {text}"
    return ClassAliasVO(raw_class_value=raw_value, display_label=label, hex_color=color)


def normalize_hex_color(value: str, fallback: str = "#3B82F6") -> str:
    candidate = (value or "").strip().upper()
    if len(candidate) == 7 and candidate.startswith("#"):
        try:
            int(candidate[1:], 16)
            return candidate
        except ValueError:
            return fallback
    return fallback
