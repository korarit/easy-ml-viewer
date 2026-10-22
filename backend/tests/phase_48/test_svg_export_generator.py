"""Phase 48 — SVG export XML validity, escaping (E48.4) and scale SLA."""

import time
import xml.etree.ElementTree as ET

import pytest

from app.domains.export_reporting.svg_export_service import SvgExportGenerator


def test_svg_produces_well_formed_xml_with_expected_layers():
    svg_str = SvgExportGenerator.generate_svg(
        x_label="Feature X",
        y_label="Feature Y",
        boundary_segments=[[-1.0, 0.0, 1.0, 0.0]],
        data_points=[{"x": 0.0, "y": 0.5, "actual": 0}],
        width=800,
        height=600,
    )
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "800"
    assert root.attrib["height"] == "600"
    assert "<g id=\"decision-boundaries\"" in svg_str
    assert "<g id=\"trajectory-points\"" in svg_str
    assert "Feature X" in svg_str and "Feature Y" in svg_str


def test_svg_escapes_adversarial_column_names():
    svg_str = SvgExportGenerator.generate_svg(
        x_label='a<b>&"c"',
        y_label="y",
        boundary_segments=[],
        data_points=[],
    )
    # Must still parse, and raw specials must not appear unescaped in labels.
    ET.fromstring(svg_str)
    assert "a<b>" not in svg_str
    assert "&amp;" in svg_str and "&lt;" in svg_str


def test_svg_large_export_meets_time_and_size_sla():
    rng_points = [{"x": float(i % 100), "y": float(i % 50), "actual": i % 2} for i in range(20_000)]
    started = time.perf_counter()
    svg_str = SvgExportGenerator.generate_svg(
        x_label="income", y_label="debt_ratio",
        boundary_segments=[[0.0, 0.0, 1.0, 1.0]] * 200,
        data_points=rng_points,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ET.fromstring(svg_str)
    assert elapsed_ms < 2000.0, f"SVG build too slow: {elapsed_ms:.1f} ms"
    assert len(svg_str.encode("utf-8")) <= 4_500_000


def test_svg_import_shim_matches_test_plan_path():
    from app.domains.recording_mlv.svg_export_generator import SvgExportGenerator as Shim

    assert Shim is SvgExportGenerator


@pytest.mark.asyncio()
async def test_svg_export_endpoint_returns_svg_attachment():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    res = client.post("/api/v1/exports/svg-2d", json={
        "x_label": "income",
        "y_label": "debt_ratio",
        "boundary_segments": [[-1.0, 0.0, 1.0, 0.0]],
        "data_points": [{"x": 0.0, "y": 0.5, "actual": 0}],
        "width": 800,
        "height": 600,
    })
    assert res.status_code == 200, res.text
    assert "image/svg+xml" in res.headers["content-type"]
    root = ET.fromstring(res.text)
    assert root.tag.endswith("svg")
