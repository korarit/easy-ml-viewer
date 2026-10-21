"""Phase 05 — mapping REST endpoint tests (auto-suggest -> validate-and-save)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())

SAMPLE_CSV = "E:/easy-ml-viewer/samples/sample_houses.csv"
SAMPLE_MODEL = "E:/easy-ml-viewer/samples/sample_house_price_linear.joblib"


def _upload_dataset_and_model() -> tuple[str, str]:
    with open(SAMPLE_CSV, "rb") as fh:
        res = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("sample_houses.csv", fh, "text/csv")},
        )
    assert res.status_code == 200, res.text
    dataset_id = res.json()["dataset_id"]
    with open(SAMPLE_MODEL, "rb") as fh:
        res = client.post(
            "/api/v1/models/upload",
            files={"file": ("sample_house_price_linear.joblib", fh, "application/octet-stream")},
        )
    assert res.status_code == 200, res.text
    return dataset_id, res.json()["model_id"]


def test_auto_suggest_exact_and_dry_run_valid() -> None:
    dataset_id, model_id = _upload_dataset_and_model()
    res = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": dataset_id, "model_id": model_id},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["is_valid"] is True
    assert body["report"]["output_shape"] == [100, 3]
    tiers = {i["model_feature_name"]: i["match_tier"] for i in body["mapping"]["items"]}
    assert tiers == {"area_sqm": "exact", "bedrooms": "exact", "distance_bts_km": "exact"}
    assert len(body["report"]["preview_matrix_head"][0]) == 3
    assert body["supports_nan_input"] is False


def test_map_by_order_and_validate_and_save_with_target() -> None:
    dataset_id, model_id = _upload_dataset_and_model()
    res = client.post(
        "/api/v1/mappings/map-by-order", json={"dataset_id": dataset_id, "model_id": model_id}
    )
    assert res.status_code == 200, res.text
    items = res.json()["mapping"]["items"]

    preview = client.post(
        "/api/v1/mappings/validate-preview",
        json={"dataset_id": dataset_id, "model_id": model_id, "items": items},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["is_valid"] is True

    saved = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "items": items,
            "target_column_name": "selling_price",
        },
    )
    assert saved.status_code == 200, saved.text
    mapping_id = saved.json()["mapping"]["mapping_id"]
    assert saved.json()["mapping"]["target_column_name"] == "selling_price"

    fetched = client.get(f"/api/v1/mappings/{mapping_id}")
    assert fetched.status_code == 200
    assert fetched.json()["mapping_id"] == mapping_id


def test_validate_preview_flags_text_column() -> None:
    dataset_id, model_id = _upload_dataset_and_model()
    res = client.post(
        "/api/v1/mappings/auto-suggest", json={"dataset_id": dataset_id, "model_id": model_id}
    )
    items = res.json()["mapping"]["items"]
    # Point bedrooms at a formatted but valid column is fine; break one mapping
    # by pointing it at nothing, and duplicate another to check warnings.
    items[1]["dataset_column_name"] = None
    items[2]["dataset_column_name"] = items[0]["dataset_column_name"]
    preview = client.post(
        "/api/v1/mappings/validate-preview",
        json={"dataset_id": dataset_id, "model_id": model_id, "items": items},
    )
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["is_valid"] is False
    assert body["duplicate_columns_used"] == [items[0]["dataset_column_name"]]
