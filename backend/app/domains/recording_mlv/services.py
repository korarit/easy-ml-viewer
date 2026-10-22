"""Phase 15 — `.mlv` package builder (columnar float32 packing).

Spec §5.1: ``frames_track`` is stored as Struct-of-Arrays — contiguous
``float32`` ND arrays (``positions_world (N,3)``, ``predictions (N,)``,
``confidences (N,)``) — never as ``list[dict]`` with repeated keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domains.recording_mlv.entities import (
    MlvFramesTrackSection,
    MlvHeaderSection,
    MlvPackageEntity,
    MlvStaticWorldSection,
)

MAX_MLV_FRAMES = 10_000
MAX_SURFACE_VERTICES = 100 * 100 * 3


def _as_f32_contiguous(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float32), dtype=np.float32)


def build_columnar_frames(
    row_indices: list[int],
    positions_world: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pack per-frame columns into contiguous float32 arrays."""
    n = len(row_indices)
    if n > MAX_MLV_FRAMES:
        raise ValueError(f"At most {MAX_MLV_FRAMES} frames per .mlv package (got {n})")
    pos = _as_f32_contiguous(np.asarray(positions_world, dtype=np.float32).reshape(n, 3)) if n else np.zeros((0, 3), dtype=np.float32)
    preds = _as_f32_contiguous(np.asarray(predictions, dtype=np.float32).reshape(n)) if n else np.zeros((0,), dtype=np.float32)
    confs = _as_f32_contiguous(np.asarray(confidences, dtype=np.float32).reshape(n)) if n else np.zeros((0,), dtype=np.float32)
    rows = np.asarray(row_indices, dtype=np.int32).reshape(n)
    if pos.shape != (n, 3) or preds.shape != (n,) or confs.shape != (n,):
        raise ValueError("Columnar frame arrays have inconsistent shapes")
    return {"row_indices": rows, "positions_world": pos, "predictions": preds, "confidences": confs}


def build_static_world_payload(
    surface_vertices_xyz: list[float],
    surface_resolution: int,
    surface_visual_type: str,
    axis_x_feature: str,
    axis_y_feature: str,
) -> dict[str, Any]:
    verts = _as_f32_contiguous(np.asarray(surface_vertices_xyz, dtype=np.float32).ravel())
    if verts.size > MAX_SURFACE_VERTICES:
        raise ValueError("Surface mesh exceeds 100x100 LOD budget")
    return {
        "surface_resolution": int(surface_resolution),
        "surface_visual_type": surface_visual_type,
        "surface_world_vertices_xyz": verts,
        "axis_x_feature": axis_x_feature,
        "axis_y_feature": axis_y_feature,
    }


def assemble_payload(
    header: MlvHeaderSection,
    static_world: dict[str, Any],
    frames_columns: dict[str, np.ndarray],
    top_k_contribs: np.ndarray | None = None,
    contour_2d_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the msgpack-ready payload dict (ndarrays handled by the codec)."""
    contribs = (
        _as_f32_contiguous(np.asarray(top_k_contribs, dtype=np.float32))
        if top_k_contribs is not None and np.asarray(top_k_contribs).size
        else np.zeros((0, 0), dtype=np.float32)
    )
    payload: dict[str, Any] = {
        "schema_version": header.schema_version,
        "header": header.model_dump(mode="json"),
        "static_world": dict(static_world),
        "frames_track": dict(frames_columns),
        "top_k_contribs": contribs,
    }
    # Phase 48 — optional 2D contour block (absent on legacy / 3D-only bakes).
    if isinstance(contour_2d_cache, dict) and contour_2d_cache:
        payload["contour_2d_cache"] = dict(contour_2d_cache)
    return payload


def split_payload(payload: dict[str, Any]) -> MlvPackageEntity:
    """Restore a decoded payload back into typed entities for the API layer."""
    header_raw = dict(payload.get("header", {}))
    header_raw.setdefault("schema_version", payload.get("schema_version", "1.0"))
    header = MlvHeaderSection(**header_raw)
    sw_raw = dict(payload.get("static_world", {}))
    verts = sw_raw.get("surface_world_vertices_xyz", [])
    if isinstance(verts, np.ndarray):
        verts_list = [float(v) for v in np.asarray(verts, dtype=np.float32).ravel().tolist()]
    else:
        verts_list = [float(v) for v in (verts or [])]
    static_world = MlvStaticWorldSection(
        surface_resolution=int(sw_raw.get("surface_resolution", 60)),
        surface_visual_type=str(sw_raw.get("surface_visual_type", "")),
        surface_world_vertices_xyz=verts_list,
        axis_x_feature=str(sw_raw.get("axis_x_feature", "")),
        axis_y_feature=str(sw_raw.get("axis_y_feature", "")),
    )
    ft_raw = dict(payload.get("frames_track", {}))

    def _col(name: str) -> np.ndarray:
        v = ft_raw.get(name, [])
        return np.asarray(v).ravel() if not isinstance(v, np.ndarray) else v.ravel()

    rows = _col("row_indices")
    preds = _col("predictions")
    confs = _col("confidences")
    frames = MlvFramesTrackSection(
        row_indices=[int(v) for v in rows.tolist()],
        predictions=[float(v) for v in preds.tolist()],
        confidences=[float(v) for v in confs.tolist()],
    )
    # Phase 48 — keep the raw 2D cache block (if any) on the typed entity so
    # bake/load/rebake round-trips preserve instant-2D without recompute.
    contour_block = payload.get("contour_2d_cache")
    contour_out: dict[str, Any] | None = None
    if isinstance(contour_block, dict) and contour_block:
        try:
            from app.domains.recording_mlv.contour_2d_baker import restore_contour_2d_cache

            restored = restore_contour_2d_cache(payload)
            contour_out = restored if restored is not None else dict(contour_block)
        except Exception:
            contour_out = dict(contour_block)
    return MlvPackageEntity(
        header=header, static_world=static_world, frames_track=frames, contour_2d_cache=contour_out
    )


def estimate_compressed_bytes(n_frames: int, surface_resolution: int, include_explainability: bool) -> int:
    """Rough pre-bake size estimate: 20 B/frame + mesh + optional Top-10 track."""
    mesh_pts = int(surface_resolution) * int(surface_resolution) * 3 * 4
    per_frame = 20 + (10 * 4 if include_explainability else 0)
    raw = mesh_pts + n_frames * per_frame + 4_096
    return max(1_024, int(raw * 0.35))  # ~2.8x Zstd ratio on float32 columns
