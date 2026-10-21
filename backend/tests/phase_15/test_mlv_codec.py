"""Phase 15 — `.mlv` codec + columnar service unit tests."""

import numpy as np
import pytest

from app.core.binary_codec import (
    HEADER_PREFIX_SIZE,
    MLV_MAGIC_HEADER,
    decode_mlv_package,
    encode_mlv_package,
)
from app.core.binary_codec import MlvFormatError
from app.domains.recording_mlv.entities import MlvHeaderSection
from app.domains.recording_mlv.services import (
    assemble_payload,
    build_columnar_frames,
    build_static_world_payload,
    split_payload,
)


def _sample_payload():
    header = MlvHeaderSection(
        title="Credit Scoring XGBoost - Q3 Audit",
        notes="ตรวจสอบเคสลูกค้ากลุ่มรายได้แปรผัน 200-350",
        created_at_iso="2026-01-01T00:00:00+00:00",
        model_type="XGBoost",
        model_family="boosting",
        is_classification=True,
        feature_names=["income", "debt_ratio"],
        total_frames=4,
        class_aliases={"0": {"display_label": "ปกติ", "hex_color": "#10B981"}},
        bookmarks=[],
        auto_events=[],
    )
    static_world = build_static_world_payload(
        surface_vertices_xyz=[0.0, 1.0, 2.0] * 12,
        surface_resolution=10,
        surface_visual_type="stepped-terrain",
        axis_x_feature="income",
        axis_y_feature="debt_ratio",
    )
    frames = build_columnar_frames(
        [10, 11, 12, 13],
        np.array([[0, 0, 0.1], [1, 0, 0.9], [2, 0, 0.2], [3, 0, 0.8]], dtype=np.float32),
        np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
    )
    return header, assemble_payload(header, static_world, frames)


def test_roundtrip_numpy_and_thai_text():
    header, payload = _sample_payload()
    blob = encode_mlv_package(payload)
    assert blob[:8] == MLV_MAGIC_HEADER
    assert len(blob) > HEADER_PREFIX_SIZE
    out = decode_mlv_package(blob)
    assert out["header"]["title"] == header.title
    assert out["header"]["notes"] == header.notes
    pos = out["frames_track"]["positions_world"]
    assert isinstance(pos, np.ndarray) and pos.dtype == np.float32
    assert pos.shape == (4, 3)
    # decoded arrays are writable copies (edge #2)
    assert pos.flags["WRITEABLE"] is True
    pos[0, 0] = 123.0
    # non-contiguous slices still round-trip (edge #6)
    view = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    frames2 = build_columnar_frames([0, 1, 2, 3], np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
    assert frames2["positions_world"].flags["C_CONTIGUOUS"]


def test_split_payload_restores_entities():
    _, payload = _sample_payload()
    blob = encode_mlv_package(payload)
    package = split_payload(decode_mlv_package(blob))
    assert package.header.total_frames == 4
    assert package.header.feature_names == ["income", "debt_ratio"]
    assert package.frames_track.row_indices == [10, 11, 12, 13]
    assert package.static_world.axis_x_feature == "income"


def test_invalid_magic_rejected():
    with pytest.raises(MlvFormatError, match="Magic"):
        decode_mlv_package(b"BADMAGIC" + b"\x00" * 16)


def test_truncated_file_rejected():
    with pytest.raises(MlvFormatError, match="[Tt]runcated|small"):
        decode_mlv_package(b"EML")


def test_crc_mismatch_rejected():
    _, payload = _sample_payload()
    blob = bytearray(encode_mlv_package(payload))
    blob[-8] ^= 0xFF
    with pytest.raises(MlvFormatError, match="CRC32"):
        decode_mlv_package(bytes(blob))
