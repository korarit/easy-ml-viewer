"""Phase 57 — compositor plan math (layout rects, theme, duration)."""

import pytest

from app.domains.export_reporting.video_export_service import (
    build_video_plan,
    layout_rects,
    theme_tokens,
    VideoExportPlanError,
)


def _opts(**over):
    base = {
        "dimension": "3d",
        "layout": "split_studio",
        "include_why_panel": True,
        "include_what_if_panel": True,
        "theme": "dark",
        "camera_preset": "isometric",
        "speed_multiplier": 1.0,
        "fps": 60,
        "format": "mp4",
        "resolution": "1080p",
        "bitrate_mbps": 12.0,
        "range_start": 0,
        "range_end": 199,
    }
    base.update(over)
    return base


def test_split_layout_matches_spec_1344_576_at_1080p():
    rects = layout_rects(1920, 1080, "split_studio")
    assert rects["stage"] == {"x": 0, "y": 0, "w": 1344, "h": 1080}
    assert rects["telemetry"] == {"x": 1344, "y": 0, "w": 576, "h": 1080}


def test_pip_layout_inside_fullscreen_stage():
    rects = layout_rects(1920, 1080, "pip")
    assert rects["stage"] == {"x": 0, "y": 0, "w": 1920, "h": 1080}
    pip = rects["telemetry"]
    assert pip["w"] > 0 and pip["h"] > 0
    assert pip["x"] + pip["w"] <= 1920 and pip["y"] + pip["h"] <= 1080


def test_pure_stage_has_no_telemetry():
    rects = layout_rects(1920, 1080, "pure_stage")
    assert rects["telemetry"]["w"] == 0


def test_light_theme_uses_paper_palette_and_boosted_lights():
    tokens = theme_tokens("light")
    assert tokens["background"] == "#f8fafc"
    assert tokens["grid"] == "#cbd5e1"
    assert tokens["ambient_boost"] == 1.4
    assert tokens["directional_boost"] == 1.2
    assert tokens["hud_text"] == "#0f172a"


def test_200_frames_at_60fps_plan_duration_and_codec():
    plan = build_video_plan(_opts())
    assert plan["n_frames"] == 200
    assert plan["duration_sec"] == pytest.approx(200 / 60.0, abs=1e-3)
    assert plan["width"] == 1920 and plan["height"] == 1080
    assert plan["video_encoder_config"]["codec"] == "avc1.640028"
    assert plan["video_encoder_config"]["container"] == "mp4"


def test_square_resolution_and_webm_codec():
    plan = build_video_plan(_opts(resolution="square_1080", format="webm"))
    assert (plan["width"], plan["height"]) == (1080, 1080)
    assert plan["video_encoder_config"]["codec"].startswith("vp09")


def test_invalid_options_raise_400_class_error():
    with pytest.raises(VideoExportPlanError):
        build_video_plan(_opts(fps=24))
    with pytest.raises(VideoExportPlanError):
        build_video_plan(_opts(range_start=10, range_end=5))
    with pytest.raises(VideoExportPlanError):
        build_video_plan(_opts(theme="neon"))
