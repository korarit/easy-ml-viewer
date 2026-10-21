"""Phase 14 — alias value objects / entities / autosuggest unit tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.application.alias_orchestrator import AliasOrchestrator, AliasUpsertRequest
from app.domains.studio_metadata.services import (
    guess_unit_suffix,
    prettify_feature_name,
    suggest_class_alias,
    suggest_feature_alias,
)


def test_class_alias_rejects_bad_color_and_blank_label():
    from app.domains.studio_metadata.value_objects import ClassAliasVO

    with pytest.raises(ValidationError):
        ClassAliasVO(raw_class_value="0", display_label="Normal", hex_color="red")
    with pytest.raises(ValidationError):
        ClassAliasVO(raw_class_value="0", display_label="   ", hex_color="#10B981")
    with pytest.raises(ValidationError):
        ClassAliasVO(raw_class_value="0", display_label="x" * 65, hex_color="#10B981")


def test_feature_alias_rejects_blank_and_long_display():
    with pytest.raises(ValidationError):
        suggest_feature_alias("").model_dump()
    from app.domains.studio_metadata.value_objects import FeatureAliasVO

    with pytest.raises(ValidationError):
        FeatureAliasVO(raw_feature_name="a", display_name="y" * 65)


def test_prettify_and_unit_guesses():
    assert prettify_feature_name("rev_util_pct") == "Rev Util Pct"
    assert prettify_feature_name("x_01_inc") == "X 01 Inc"
    assert suggest_feature_alias("rev_util_pct").unit_suffix == "%"
    assert guess_unit_suffix("annual_income") == ""


def test_hex_color_normalized_uppercase():
    from app.domains.studio_metadata.value_objects import ClassAliasVO

    vo = ClassAliasVO(raw_class_value=0, display_label="Normal", hex_color="#10b981")
    assert vo.hex_color == "#10B981"


def test_orchestrator_upsert_roundtrip_and_clear():
    AliasOrchestrator.clear()
    entity = AliasOrchestrator.upsert(
        AliasUpsertRequest(
            session_id="sess_1",
            class_aliases={"0": {"raw_class_value": "0", "display_label": "Normal", "hex_color": "#10B981"}},
            feature_aliases={
                "rev_util_pct": {
                    "raw_feature_name": "rev_util_pct",
                    "display_name": "Rev Util",
                    "unit_suffix": "%",
                }
            },
        )
    )
    assert entity.session_id == "sess_1"
    fetched = AliasOrchestrator.get("sess_1")
    assert fetched.class_aliases["0"].display_label == "Normal"
    assert fetched.feature_aliases["rev_util_pct"].unit_suffix == "%"
    assert AliasOrchestrator.clear() == 1
