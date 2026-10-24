"""Phase 57 — video-plan + transcode-mp4 endpoint contract."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _plan_payload(**over):
    base = {
        "dimension": "2d",
        "layout": "pip",
        "include_why_panel": True,
        "include_what_if_panel": True,
        "theme": "light",
        "camera_preset": "top_xy",
        "speed_multiplier": 1.0,
        "fps": 30,
        "format": "mp4",
        "resolution": "720p",
        "bitrate_mbps": 8.0,
        "range_start": 0,
        "range_end": 59,
    }
    base.update(over)
    return base


def test_video_plan_returns_layout_rects_and_duration():
    res = client.post("/api/v1/exports/video-plan", json=_plan_payload())
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["n_frames"] == 60
    assert body["width"] == 1280 and body["height"] == 720
    assert body["layout_rects"]["stage"]["w"] == 1280
    assert body["theme_tokens"]["background"] == "#f8fafc"
    assert body["video_encoder_config"]["container"] == "mp4"


def test_video_plan_rejects_bad_fps_with_400():
    res = client.post("/api/v1/exports/video-plan", json=_plan_payload(fps=24))
    assert res.status_code == 400
    assert res.json()["code"] == "VIDEO_PLAN_INVALID"


def test_transcode_empty_upload_returns_400():
    res = client.post(
        "/api/v1/exports/transcode-mp4",
        files={"file": ("clip.webm", b"", "video/webm")},
    )
    assert res.status_code == 400
    assert res.json()["code"] == "TRANSCODE_EMPTY_FILE"


def test_transcode_without_pyav_returns_501_or_mp4(monkeypatch):
    import app.presentation.endpoints.video_export_endpoints as ep

    monkeypatch.setattr(ep, "av_available", lambda: False)
    res = client.post(
        "/api/v1/exports/transcode-mp4",
        files={"file": ("clip.webm", b"\x1a\x45\xdf\xa3fake", "video/webm")},
    )
    assert res.status_code == 501
    assert res.json()["code"] == "TRANSCODE_BACKEND_MISSING"
