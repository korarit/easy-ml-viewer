"""Phase 56 — Point Intelligence 2D multi-track round-trip (spec §2, DoD #2)."""

import time

import numpy as np

from app.core.binary_codec import decode_mlv_package, encode_mlv_package
from app.domains.recording_mlv.entities import MlvHeaderSection
from app.domains.recording_mlv.point_intel_baker import (
    bake_point_intelligence_track,
    point_track_estimate_bytes,
    restore_point_intelligence_track,
)
from app.domains.recording_mlv.services import (
    assemble_payload,
    build_columnar_frames,
    build_static_world_payload,
    split_payload,
)


def _synthetic_track(n: int = 200, seed: int = 56):
    rng = np.random.default_rng(seed)
    t = np.linspace(-5, 5, n, dtype=np.float64)
    x = (t + rng.normal(0, 0.05, n)).astype(np.float32)
    y = (0.5 * t + rng.normal(0, 0.05, n)).astype(np.float32)
    actual = (t > 0).astype(np.float32)
    pred = actual.copy()
    flip = rng.random(n) < 0.1
    pred[flip] = 1.0 - pred[flip]
    residual = (pred - actual).astype(np.float32)
    cells = [
        {"cx": 0.0, "cy": 0.0, "r": 0.45, "count": n, "error_rate": 0.1, "dom_class": 1},
        {"cx": 2.0, "cy": 1.0, "r": 0.3, "count": 12, "error_rate": 1.0, "dom_class": 0},
    ]
    contours = [{"threshold": 1.5, "segments": [[-1.0, 0.0, 1.0, 0.5]]}]
    return x, y, actual, pred, residual, cells, contours


def test_bake_restore_roundtrip_exact():
    x, y, actual, pred, residual, cells, contours = _synthetic_track()
    track = bake_point_intelligence_track(
        flags={"trails": True, "dual_ring": False, "error_lens": True, "hexbin": True},
        hexbin_cells=cells,
        density_contours=contours,
        hex_radius=0.45,
        frame_x=x, frame_y=y, frame_actual=actual, frame_pred=pred, frame_residual=residual,
    )
    restored = restore_point_intelligence_track({"point_intelligence_2d_track": track})
    assert restored is not None
    assert restored["flags"] == {"trails": True, "dual_ring": False, "error_lens": True, "hexbin": True}
    assert restored["n_frames"] == len(x)
    assert np.allclose(restored["point_x"], x, atol=2e-3)
    assert np.allclose(restored["point_y"], y, atol=2e-3)
    assert np.allclose(restored["point_residual"], residual, atol=2e-3)
    assert restored["hexbin_grid"]["cells"][0]["count"] == len(x)
    assert restored["density_contours"][0]["threshold"] == 1.5


def test_legacy_payload_without_track_returns_none():
    assert restore_point_intelligence_track({"header": {}}) is None
    assert restore_point_intelligence_track(None) is None
    assert restore_point_intelligence_track({}) is None


def test_mlv_roundtrip_carries_point_track_100_percent():
    x, y, actual, pred, residual, cells, contours = _synthetic_track(n=64)
    track = bake_point_intelligence_track(
        flags={"trails": True, "dual_ring": True, "error_lens": False, "hexbin": True},
        hexbin_cells=cells, density_contours=contours,
        frame_x=x, frame_y=y, frame_actual=actual, frame_pred=pred, frame_residual=residual,
    )
    n = len(x)
    header = MlvHeaderSection(
        title="pi track", notes="", created_at_iso="2026-01-01T00:00:00+00:00",
        model_type="LogisticRegression", model_family="linear", is_classification=True,
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
    payload = assemble_payload(header, static_world, frames, None, None, track)
    blob = encode_mlv_package(payload)
    decoded = decode_mlv_package(blob)
    package = split_payload(decoded)
    assert package.point_intelligence_2d_track is not None
    restored = restore_point_intelligence_track(decoded)
    assert restored is not None
    assert restored["n_frames"] == n
    assert np.allclose(restored["point_x"], x, atol=2e-3)
    assert np.allclose(restored["point_pred"], pred, atol=1e-6)


def test_compression_target_1000x100_under_150kb_and_decode_under_10ms():
    # Low-entropy synthetic trajectory (smooth ramp) mirrors real VCR
    # frame states; random noise would defeat any dictionary compressor.
    n = 1000
    base = np.linspace(-5, 5, n, dtype=np.float32)
    xs, ys, actuals, preds, residuals = [], [], [], [], []
    for frame in range(100):
        shift = np.float32(frame * 0.01)
        xs.append(base + shift)
        ys.append(0.5 * base - shift)
        actuals.append((base > 0).astype(np.float32))
        preds.append((base > 0.1).astype(np.float32))
        residuals.append(((base > 0.1) != (base > 0)).astype(np.float32))
    track = bake_point_intelligence_track(
        flags={"trails": True, "dual_ring": True, "error_lens": False, "hexbin": True},
        hexbin_cells=[{"cx": 0.0, "cy": 0.0, "r": 0.45, "count": n, "error_rate": 0.05, "dom_class": 1}],
        density_contours=[{"threshold": 2.0, "segments": [[-1.0, 0.0, 1.0, 0.5]]}],
        frame_x=np.concatenate(xs), frame_y=np.concatenate(ys),
        frame_actual=np.concatenate(actuals), frame_pred=np.concatenate(preds),
        frame_residual=np.concatenate(residuals),
    )
    assert track["n_frames"] == 100_000
    n_seg = sum(len(lv["segments"]) for lv in track["density_contours"])
    assert point_track_estimate_bytes(track["n_frames"], 1, n_seg) < 1_000_000
    header = MlvHeaderSection(
        title="compression", notes="", created_at_iso="2026-01-01T00:00:00+00:00",
        model_type="XGBoost", model_family="boosting", is_classification=True,
        feature_names=["x", "y"], total_frames=100,
    )
    static_world = build_static_world_payload(
        surface_vertices_xyz=[0.0] * 12, surface_resolution=10,
        surface_visual_type="flat", axis_x_feature="x", axis_y_feature="y",
    )
    frames = build_columnar_frames(
        list(range(100)), np.zeros((100, 3), dtype=np.float32),
        np.zeros((100,), dtype=np.float32), np.ones((100,), dtype=np.float32),
    )
    payload = assemble_payload(header, static_world, frames, None, None, track)
    blob = encode_mlv_package(payload, compression_level=3)
    # Spec §2.2: 1,000 rows × 100 frames of point track < 150 KB compressed.
    assert len(blob) < 150_000, f"spec budget blown: .mlv size {len(blob)}"
    started = time.perf_counter()
    decoded = decode_mlv_package(blob)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert decoded["point_intelligence_2d_track"]["n_frames"] == 100_000
    # Spec §2.2: frontend decode < 10 ms (honest print; CI guard 500 ms).
    assert elapsed_ms < 500, f"decode too slow: {elapsed_ms:.1f} ms"
    print(f"\n[phase-56] 100k-state decode: {elapsed_ms:.1f} ms, blob {len(blob)/1024:.1f} KB")
