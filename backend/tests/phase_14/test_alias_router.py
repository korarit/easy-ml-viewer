"""Phase 14 — alias router integration tests (spec §7 edge matrix)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.application.alias_orchestrator import AliasOrchestrator
from app.main import app


def _client() -> TestClient:
    AliasOrchestrator.clear()
    return TestClient(app)


def test_upsert_get_export_import_roundtrip():
    client = _client()
    put = client.put(
        "/api/v1/alias/sess_a",
        json={
            "session_id": "sess_a",
            "class_aliases": {
                "0": {"raw_class_value": "0", "display_label": "Normal", "hex_color": "#10B981"},
                "1": {"raw_class_value": "1", "display_label": "High Risk", "hex_color": "#F43F5E"},
            },
            "feature_aliases": {
                "rev_util_pct": {
                    "raw_feature_name": "rev_util_pct",
                    "display_name": "Rev Util",
                    "unit_suffix": "%",
                }
            },
        },
    )
    assert put.status_code == 200, put.text
    assert put.json()["session_id"] == "sess_a"

    got = client.get("/api/v1/alias/sess_a")
    assert got.status_code == 200
    assert got.json()["class_aliases"]["1"]["hex_color"] == "#F43F5E"

    exported = client.get("/api/v1/alias/sess_a/export")
    assert exported.status_code == 200
    blob = exported.json()["alias_json"]
    assert blob["feature_aliases"]["rev_util_pct"]["unit_suffix"] == "%"

    imported = client.post("/api/v1/alias/sess_b/import", json={"alias_json": blob})
    assert imported.status_code == 200, imported.text
    assert imported.json()["session_id"] == "sess_b"
    assert imported.json()["class_aliases"]["0"]["display_label"] == "Normal"


def test_autosuggest_prettifies_and_assigns_palette():
    client = _client()
    res = client.post(
        "/api/v1/alias/autosuggest/preview",
        json={"feature_names": ["rev_util_pct", "annual_income"], "class_values": [0, 1]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["feature_aliases"]["rev_util_pct"]["display_name"] == "Rev Util Pct"
    assert body["feature_aliases"]["rev_util_pct"]["unit_suffix"] == "%"
    assert body["class_aliases"]["0"]["hex_color"] == "#10B981"
    assert body["class_aliases"]["1"]["hex_color"] == "#F43F5E"


def test_guards_404_422_path_mismatch():
    client = _client()
    assert client.get("/api/v1/alias/missing_session_xyz").status_code == 404
    # Bad hex color -> 422 (spec edge #5 guard: label/color validated at API boundary)
    bad = client.put(
        "/api/v1/alias/sess_bad",
        json={
            "session_id": "sess_bad",
            "class_aliases": {
                "0": {"raw_class_value": "0", "display_label": "Normal", "hex_color": "not-a-color"}
            },
        },
    )
    assert bad.status_code == 422
    # Over-long label -> 422 (truncate/tr Tooltip path is frontend; API rejects)
    long_label = client.put(
        "/api/v1/alias/sess_bad",
        json={
            "session_id": "sess_bad",
            "class_aliases": {
                "0": {"raw_class_value": "0", "display_label": "x" * 100, "hex_color": "#10B981"}
            },
        },
    )
    assert long_label.status_code == 422
    # Path/body session mismatch is reconciled to the path (no silent cross-write)
    ok = client.put(
        "/api/v1/alias/sess_path",
        json={"session_id": "other_id", "class_aliases": {}, "feature_aliases": {}},
    )
    assert ok.status_code == 200
    assert ok.json()["session_id"] == "sess_path"
