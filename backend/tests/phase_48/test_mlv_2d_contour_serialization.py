"""Phase 48 — `.mlv` 2D contour cache round-trip (spec §2.2, E48.3)."""

import numpy as np

from app.core.binary_codec import decode_mlv_package, encode_mlv_package
from app.domains.recording_mlv.contour_2d_baker import (
    bake_contour_2d_cache,
    contour_cache_estimate_bytes,
    restore_contour_2d_cache,
)
from app.domains.recording_mlv.entities import MlvHeaderSection
from app.domains.recording_mlv.services import (
    assemble_payload,
    build_columnar_frames,
    build_static_world_payload,
    split_payload,
)
def make_contour_dict(res: int = 24, shift: float = 0.0, task: str = "classification"):
    rng = np.random.default_rng(48)
    field = rng.uniform(0.0, 1.0, size=(res, res)).astype(np.float64)
    field = np.clip(field + shift, 0.0, 1.0)
    h_min, h_max = float(field.min()), float(field.max())
    span = max(h_max - h_min, 1e-9)
    quantized = (np.rint((field - h_min) / span * 255.0).astype(np.uint8)).ravel().tolist()
    return {
        "x_feature": "income",
        "y_feature": "debt_ratio",
        "grid_resolution": res,
        "prediction_task": task,
        "heatmap_min": h_min,
        "heatmap_max": h_max,
        "heatmap_quantized": quantized,
        "contours": [
            {"threshold": 0.5, "class_index": 1, "kind": "decision",
             "line_segments": [[-2.5, 1.0, -1.0, 1.5], [-1.0, 1.5, 0.5, 2.0]]},
        ],
    }


def test_bake_restore_preserves_heatmap_and_polylines():
    contour = make_contour_dict(res=24)
    block = bake_contour_2d_cache(contour)
    assert block["grid_resolution"] == 24
    assert block["heatmap_float32"].shape == (24, 24)
    assert block["heatmap_float32"].dtype == np.float32
    assert len(block["boundary_polylines"]) == 2

    restored = restore_contour_2d_cache({"contour_2d_cache": block})
    assert restored is not None
    assert restored["grid_resolution"] == 24
    assert len(restored["boundary_polylines"]) == 2
    assert np.allclose(restored["heatmap_float32"], block["heatmap_float32"], atol=1e-6)


def test_legacy_payload_without_cache_returns_none_for_fallback():
    assert restore_contour_2d_cache({"header": {}}) is None
    assert restore_contour_2d_cache(None) is None
    assert restore_contour_2d_cache({}) is None


def test_mlv_roundtrip_carries_contour_and_stays_compact():
    contour = make_contour_dict(res=120)
    block = bake_contour_2d_cache(contour)
    n = 8
    header = MlvHeaderSection(
        title="2D cached", notes="", created_at_iso="2026-01-01T00:00:00+00:00",
        model_type="RandomForest", model_family="trees", is_classification=True,
        feature_names=["income", "debt_ratio"], total_frames=n,
    )
    static_world = build_static_world_payload(
        surface_vertices_xyz=[0.0] * 12, surface_resolution=10,
        surface_visual_type="flat", axis_x_feature="income", axis_y_feature="debt_ratio",
    )
    frames = build_columnar_frames(
        list(range(n)), np.zeros((n, 3), dtype=np.float32),
        np.zeros((n,), dtype=np.float32), np.ones((n,), dtype=np.float32),
    )
    payload = assemble_payload(header, static_world, frames, None, block)
    blob = encode_mlv_package(payload)
    # 120×120 float32 heatmap compresses to roughly the ~45 KB spec budget.
    assert len(blob) < 120_000, f"unexpected .mlv size {len(blob)}"

    decoded = decode_mlv_package(blob)
    package = split_payload(decoded)
    assert package.contour_2d_cache is not None
    restored = restore_contour_2d_cache(decoded)
    assert restored is not None
    assert restored["grid_resolution"] == 120
    assert restored["x_feature"] == "income"
    # Instant-2D requirement: heatmap fully present, no recompute needed.
    assert restored["heatmap_float32"].shape == (120, 120)


def test_cache_estimate_matches_spec_budget():
    assert contour_cache_estimate_bytes(120, 40) < 60_000
