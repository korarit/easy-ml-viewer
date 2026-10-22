"""Phase 40 — dual `.mlv` comparison service + endpoint tests."""

import numpy as np
from fastapi.testclient import TestClient

from app.core.binary_codec import encode_mlv_package
from app.domains.recording_mlv.dual_compare_service import (
    compare_decoded_payloads,
    summary_to_response,
)
from app.domains.recording_mlv.entities import MlvHeaderSection
from app.domains.recording_mlv.services import (
    assemble_payload,
    build_columnar_frames,
    build_static_world_payload,
)
from app.main import app


def _payload(title: str, preds: list[float], confs: list[float], is_clf: bool = True):
    n = len(preds)
    header = MlvHeaderSection(
        title=title, notes="", created_at_iso="2026-01-01T00:00:00+00:00",
        model_type=title, model_family="test", is_classification=is_clf,
        feature_names=["income", "age"], total_frames=n,
    )
    static_world = build_static_world_payload(
        surface_vertices_xyz=[0.0] * 12, surface_resolution=10,
        surface_visual_type="flat", axis_x_feature="income", axis_y_feature="age",
    )
    frames = build_columnar_frames(
        list(range(n)),
        np.zeros((n, 3), dtype=np.float32),
        np.asarray(preds, dtype=np.float32),
        np.asarray(confs, dtype=np.float32),
    )
    return assemble_payload(header, static_world, frames)


def test_classification_disagreement_mask_and_divergence():
    pa = _payload("A", [0, 1, 1, 0], [0.9, 0.8, 0.6, 0.7])
    pb = _payload("B", [0, 0, 1, 1], [0.5, 0.9, 0.6, 0.4])
    summary = compare_decoded_payloads(pa, pb)
    assert summary.total_aligned_frames == 4
    assert summary.truncated is False
    assert summary.disagreement_indices.tolist() == [1, 3]
    assert summary.agreement_rate == 0.5
    # confidence-gap proxy: |0.8-0.9|=0.1 at frame 1, |0.7-0.4|=0.3 at frame 3
    assert abs(summary.probability_divergence.tolist()[1] - 0.1) < 1e-6
    assert summary.probability_divergence.dtype == np.float32
    assert summary.disagreement_indices.dtype == np.uint32


def test_frame_mismatch_truncates_to_nmin_with_notice():
    pa = _payload("A", [0, 1, 0], [0.9, 0.8, 0.7])
    pb = _payload("B", [0, 0], [0.9, 0.2])
    summary = compare_decoded_payloads(pa, pb)
    assert summary.total_aligned_frames == 2
    assert summary.truncated is True
    body = summary_to_response(summary)
    assert "truncating comparison to first 2 frames" in body["truncation_notice"]
    assert body["disagreement_indices"] == [1]


def test_regression_absolute_error_no_conflicts():
    pa = _payload("A", [1.0, 2.0, 3.0], [0.9, 0.9, 0.9], is_clf=False)
    pb = _payload("B", [1.5, 2.0, 2.0], [0.9, 0.9, 0.9], is_clf=False)
    summary = compare_decoded_payloads(pa, pb)
    assert summary.disagreement_indices.size == 0
    assert summary.agreement_rate == 1.0
    assert summary.divergence_kind == "absolute_error"
    assert summary.absolute_error.tolist() == [0.5, 0.0, 1.0]


def test_compare_endpoint_roundtrip_and_invalid_magic():
    client = TestClient(app)
    blob_a = encode_mlv_package(_payload("A", [0, 1, 1], [0.9, 0.8, 0.7]))
    blob_b = encode_mlv_package(_payload("B", [0, 0, 1], [0.9, 0.2, 0.7]))
    res = client.post(
        "/api/v1/recordings/compare",
        files={
            "file_a": ("a.mlv", blob_a, "application/octet-stream"),
            "file_b": ("b.mlv", blob_b, "application/octet-stream"),
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["total_aligned_frames"] == 3
    assert body["disagreement_indices"] == [1]
    assert body["file_a_name"] == "a.mlv"

    bad = client.post(
        "/api/v1/recordings/compare",
        files={
            "file_a": ("a.csv", b"col1,col2\n1,2\n", "text/csv"),
            "file_b": ("b.mlv", blob_b, "application/octet-stream"),
        },
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "MLV_INVALID_MAGIC"
