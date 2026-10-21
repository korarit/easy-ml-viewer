"""Phase 14 — studio metadata value objects (class + feature aliasing)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ClassAliasVO(BaseModel):
    raw_class_value: str | int
    display_label: str = Field(max_length=64)
    hex_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")

    @field_validator("display_label")
    @classmethod
    def _strip_label(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("display_label must not be blank")
        return cleaned

    @field_validator("hex_color")
    @classmethod
    def _normalize_color(cls, v: str) -> str:
        return v.upper()


class FeatureAliasVO(BaseModel):
    raw_feature_name: str = Field(min_length=1)
    display_name: str = Field(max_length=64)
    unit_suffix: str = Field(default="", max_length=16)

    @field_validator("display_name")
    @classmethod
    def _strip_display(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("display_name must not be blank")
        return cleaned

    @field_validator("unit_suffix")
    @classmethod
    def _strip_unit(cls, v: str) -> str:
        return v.strip()
