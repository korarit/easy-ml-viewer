"""Phase 57 — Advanced Video Export Studio math (offline, pure-python).

Covers spec §2.1-§2.2:
- Resolution table (1080p / 720p / square_1080).
- Layout geometry: Split Studio (70/30) vs PiP vs Pure Stage.
- Color palette token matrix (dark / light / match_ui).
- Frame-range validation, duration + bitrate estimates.

Memory (§Master Plan): O(1) — only small dicts, never frame buffers.
"""

from __future__ import annotations

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "square_1080": (1080, 1080),
}

THEME_TOKENS: dict[str, dict[str, object]] = {
    "dark": {
        "background": "#0b1220",
        "grid": "#334155",
        "surface_opacity": 1.0,
        "ambient_boost": 1.0,
        "directional_boost": 1.0,
        "hud_text": "#e2e8f0",
        "hud_panel": "rgba(8,12,18,0.78)",
        "accent_loss": "#f43f5e",
    },
    "light": {
        # Spec §2.2 — paper/slide friendly Light Mode palette.
        "background": "#f8fafc",
        "grid": "#cbd5e1",
        "surface_opacity": 1.0,
        "ambient_boost": 1.4,
        "directional_boost": 1.2,
        "hud_text": "#0f172a",
        "hud_panel": "rgba(248,250,252,0.85)",
        "accent_loss": "#dc2626",
    },
    # Match UI resolves client-side; backend keeps dark tokens as safe default.
    "match_ui": {
        "background": "#0b1220",
        "grid": "#334155",
        "surface_opacity": 1.0,
        "ambient_boost": 1.0,
        "directional_boost": 1.0,
        "hud_text": "#e2e8f0",
        "hud_panel": "rgba(8,12,18,0.78)",
        "accent_loss": "#f43f5e",
    },
}

VALID_DIMENSIONS = ("3d", "2d")
VALID_LAYOUTS = ("split_studio", "pip", "pure_stage")
VALID_THEMES = ("match_ui", "dark", "light")
VALID_CAMERAS = ("isometric", "top_xy", "front_xz", "side_yz", "follow_ball")
VALID_FORMATS = ("mp4", "webm")
VALID_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
VALID_FPS = (30, 60)

MAX_EXPORT_FRAMES = 10_000


class VideoExportPlanError(ValueError):
    """Raised when option validation fails (maps to HTTP 400)."""


def resolution_size(resolution: str) -> tuple[int, int]:
    try:
        return RESOLUTIONS[resolution]
    except KeyError:
        raise VideoExportPlanError(f"Unknown resolution '{resolution}'") from None


def theme_tokens(theme: str) -> dict[str, object]:
    try:
        return dict(THEME_TOKENS[theme])
    except KeyError:
        raise VideoExportPlanError(f"Unknown theme '{theme}'") from None


def layout_rects(width: int, height: int, layout: str) -> dict[str, dict[str, int]]:
    """Pixel-exact compositor rectangles for the virtual canvas."""
    if layout == "split_studio":
        # 70% stage / 30% telemetry dock (spec §2.1 diagram: 1344/576 @1080p).
        stage_w = int(round(width * 0.70))
        return {
            "stage": {"x": 0, "y": 0, "w": stage_w, "h": height},
            "telemetry": {"x": stage_w, "y": 0, "w": width - stage_w, "h": height},
            "hud_bar": {"x": 0, "y": height - 36, "w": stage_w, "h": 36},
            "metrics_badge": {"x": stage_w, "y": height - 36, "w": width - stage_w, "h": 36},
        }
    if layout == "pip":
        pip_w = int(round(width * 0.27))
        pip_h = int(round(height * 0.33))
        margin = int(round(width * 0.02))
        return {
            "stage": {"x": 0, "y": 0, "w": width, "h": height},
            "telemetry": {
                "x": width - pip_w - margin,
                "y": height - pip_h - margin - 36,
                "w": pip_w,
                "h": pip_h,
            },
            "hud_bar": {"x": 0, "y": height - 36, "w": width, "h": 36},
            "metrics_badge": {"x": 0, "y": height - 36, "w": 0, "h": 0},
        }
    if layout == "pure_stage":
        return {
            "stage": {"x": 0, "y": 0, "w": width, "h": height},
            "telemetry": {"x": 0, "y": 0, "w": 0, "h": 0},
            "hud_bar": {"x": 0, "y": height - 36, "w": width, "h": 36},
            "metrics_badge": {"x": 0, "y": 0, "w": 0, "h": 0},
        }
    raise VideoExportPlanError(f"Unknown layout '{layout}'")


def validate_options(options: dict[str, object]) -> dict[str, object]:
    """Validate raw option dict; returns a normalized copy or raises."""
    dimension = str(options.get("dimension", "3d"))
    layout = str(options.get("layout", "split_studio"))
    theme = str(options.get("theme", "match_ui"))
    camera = str(options.get("camera_preset", options.get("cameraPreset", "isometric")))
    fmt = str(options.get("format", "mp4"))
    resolution = str(options.get("resolution", "1080p"))
    try:
        speed = float(options.get("speed_multiplier", options.get("speedMultiplier", 1.0)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise VideoExportPlanError("speed_multiplier must be numeric") from None
    try:
        fps = int(options.get("fps", 60))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise VideoExportPlanError("fps must be 30 or 60") from None
    try:
        bitrate = float(options.get("bitrate_mbps", options.get("bitrateMbps", 12.0)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise VideoExportPlanError("bitrate_mbps must be numeric") from None
    try:
        start = int(options.get("range_start", options.get("rangeStartFrame", 0)))  # type: ignore[arg-type]
        end = int(options.get("range_end", options.get("rangeEndFrame", 0)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise VideoExportPlanError("range_start/range_end must be integers") from None

    if dimension not in VALID_DIMENSIONS:
        raise VideoExportPlanError(f"dimension must be one of {VALID_DIMENSIONS}")
    if layout not in VALID_LAYOUTS:
        raise VideoExportPlanError(f"layout must be one of {VALID_LAYOUTS}")
    if theme not in VALID_THEMES:
        raise VideoExportPlanError(f"theme must be one of {VALID_THEMES}")
    if camera not in VALID_CAMERAS:
        raise VideoExportPlanError(f"camera_preset must be one of {VALID_CAMERAS}")
    if fmt not in VALID_FORMATS:
        raise VideoExportPlanError(f"format must be one of {VALID_FORMATS}")
    if resolution not in RESOLUTIONS:
        raise VideoExportPlanError(f"resolution must be one of {sorted(RESOLUTIONS)}")
    if speed not in VALID_SPEEDS:
        raise VideoExportPlanError(f"speed_multiplier must be one of {VALID_SPEEDS}")
    if fps not in VALID_FPS:
        raise VideoExportPlanError(f"fps must be one of {VALID_FPS}")
    if not (1.0 <= bitrate <= 40.0):
        raise VideoExportPlanError("bitrate_mbps must be within [1.0, 40.0]")
    if start < 0 or end < 0:
        raise VideoExportPlanError("range_start/range_end must be >= 0")
    if end < start:
        raise VideoExportPlanError("range_end must be >= range_start")
    n_frames = end - start + 1
    if n_frames > MAX_EXPORT_FRAMES:
        raise VideoExportPlanError(f"At most {MAX_EXPORT_FRAMES} frames per export (got {n_frames})")

    return {
        "dimension": dimension,
        "layout": layout,
        "include_why_panel": bool(options.get("include_why_panel", options.get("includeWhyPanel", True))),
        "include_what_if_panel": bool(
            options.get("include_what_if_panel", options.get("includeWhatIfPanel", True))
        ),
        "theme": theme,
        "camera_preset": camera,
        "speed_multiplier": speed,
        "fps": fps,
        "format": fmt,
        "resolution": resolution,
        "bitrate_mbps": bitrate,
        "range_start": start,
        "range_end": end,
    }


def build_video_plan(options: dict[str, object]) -> dict[str, object]:
    """Full deterministic export plan: rects + tokens + duration + estimates."""
    norm = validate_options(options)
    width, height = resolution_size(str(norm["resolution"]))
    rects = layout_rects(width, height, str(norm["layout"]))
    n_frames = int(norm["range_end"]) - int(norm["range_start"]) + 1
    fps = int(norm["fps"])
    duration_sec = n_frames / float(fps)
    # Effective playback time at the chosen speed multiplier.
    playback_sec = duration_sec / float(norm["speed_multiplier"])
    est_bytes = int(width * height * 3 * n_frames * 0.06) + 65_536
    return {
        **norm,
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration_sec": round(duration_sec, 3),
        "playback_sec": round(playback_sec, 3),
        "estimated_bytes": est_bytes,
        "layout_rects": rects,
        "theme_tokens": theme_tokens(str(norm["theme"])),
        "video_encoder_config": {
            "codec": "avc1.640028" if norm["format"] == "mp4" else "vp09.00.10.08",
            "container": norm["format"],
            "fps": fps,
            "bitrate_bps": int(float(norm["bitrate_mbps"]) * 1_000_000),
        },
    }


def av_available() -> bool:
    """True when the optional PyAV/FFmpeg backend is importable (offline)."""
    try:
        import av  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False
