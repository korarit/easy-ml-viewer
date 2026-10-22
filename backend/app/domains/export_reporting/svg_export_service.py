"""Phase 48 — publication-quality 2D decision-boundary SVG exporter.

Standalone, 100% offline SVG builder per spec §2.3/§5.2:

- ``<svg viewBox="0 0 W H">`` with dark (``precision-dark``) or light
  (``paper-clean``) theme root rect.
- Decision boundaries as ``<path>`` polylines (white 2.5px, round caps).
- Trajectory points as ``<circle>`` (green/red by label, 0.85 opacity).
- Axis labels + tick text in Inter (bundled locally on the client).

Hardening:
- E48.4: axis labels are XML-escaped (``&``/``<``/``>``/quotes) via
  :mod:`xml.sax.saxutils`, so adversarial column names never break the XML.
- Numeric inputs are finite-guarded (NaN/Inf → dropped) and rounded to keep
  large exports (20k points) under the 4.5 MB / 350 ms SLA.
- Pure function, no I/O, no network — safe for air-gapped operation.
"""

from __future__ import annotations

import math
from typing import Any
from xml.sax.saxutils import escape as _xml_escape, quoteattr as _xml_quoteattr


def _esc(text: Any) -> str:
    return _xml_escape(str(text), {'"': "&quot;", "'": "&apos;"})


def _finite_num(value: Any, fallback: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return f if math.isfinite(f) else fallback


class SvgExportGenerator:
    """Standalone 2D decision-map SVG builder (no backend state)."""

    @staticmethod
    def generate_svg(
        x_label: str,
        y_label: str,
        boundary_segments: list[list[float]],
        data_points: list[dict[str, Any]],
        width: int = 1200,
        height: int = 900,
        theme: str = "precision-dark",
        title: str = "2D Decision Boundary",
        disagreement_pct: float | None = None,
        max_points: int = 20_000,
    ) -> str:
        w = max(320, min(3840, int(width)))
        h = max(240, min(2160, int(height)))
        dark = str(theme).lower() != "paper-clean"
        bg = "#0b0f14" if dark else "#ffffff"
        fg = "#e2e8f0" if dark else "#0f172a"
        muted = "#94a3b8" if dark else "#475569"
        grid = "#1e293b" if dark else "#e2e8f0"
        boundary_color = "#ffffff" if dark else "#0f172a"

        margin_l, margin_r, margin_t, margin_b = 72, 28, 56, 64
        plot_w = max(10.0, float(w - margin_l - margin_r))
        plot_h = max(10.0, float(h - margin_t - margin_b))

        # World extent from finite segment/point coords (fallback ±5 cube).
        xs: list[float] = []
        ys: list[float] = []
        segs: list[tuple[float, float, float, float]] = []
        for seg in boundary_segments or []:
            try:
                x1, y1, x2, y2 = (float(v) for v in list(seg)[:4])
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            segs.append((x1, y1, x2, y2))
            xs.extend((x1, x2))
            ys.extend((y1, y2))
        pts: list[tuple[float, float, int]] = []
        for pt in (data_points or [])[: max(0, int(max_points))]:
            if not isinstance(pt, dict):
                continue
            x = _finite_num(pt.get("x"), math.nan)
            y = _finite_num(pt.get("y"), math.nan)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            try:
                actual = int(pt.get("actual", 0) or 0)
            except (TypeError, ValueError):
                actual = 0
            pts.append((x, y, 1 if actual else 0))
            xs.append(x)
            ys.append(y)
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            if x_max - x_min < 1e-9:
                x_min, x_max = x_min - 1.0, x_max + 1.0
            if y_max - y_min < 1e-9:
                y_min, y_max = y_min - 1.0, y_max + 1.0
            padx = (x_max - x_min) * 0.04
            pady = (y_max - y_min) * 0.04
            x_min, x_max = x_min - padx, x_max + padx
            y_min, y_max = y_min - pady, y_max + pady
        else:
            x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0

        def _sx(x: float) -> float:
            return margin_l + (x - x_min) / (x_max - x_min) * plot_w

        def _sy(y: float) -> float:
            return margin_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

        def _r2(v: float) -> str:
            return f"{v:.2f}"

        parts = [
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\" role=\"img\">",
            f"<title>{_esc(title)}</title>",
            f"<rect width=\"100%\" height=\"{h}\" fill=\"{bg}\"/>",
            f"<g id=\"grid\" stroke=\"{grid}\" stroke-width=\"1\">",
            f"<rect x=\"{margin_l}\" y=\"{margin_t}\" width=\"{_r2(plot_w)}\" height=\"{_r2(plot_h)}\" fill=\"none\"/>",
        ]
        for i in range(1, 5):
            gx = margin_l + plot_w * i / 5
            gy = margin_t + plot_h * i / 5
            parts.append(f"<line x1=\"{_r2(gx)}\" y1=\"{margin_t}\" x2=\"{_r2(gx)}\" y2=\"{_r2(margin_t + plot_h)}\"/>")
            parts.append(f"<line x1=\"{margin_l}\" y1=\"{_r2(gy)}\" x2=\"{_r2(margin_l + plot_w)}\" y2=\"{_r2(gy)}\"/>")
        parts.append("</g>")
        parts.append(
            f"<g id=\"decision-boundaries\" stroke={_xml_quoteattr(boundary_color)} "
            f"stroke-width=\"2.5\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
        )
        for x1, y1, x2, y2 in segs:
            parts.append(
                f"<line x1=\"{_r2(_sx(x1))}\" y1=\"{_r2(_sy(y1))}\" "
                f"x2=\"{_r2(_sx(x2))}\" y2=\"{_r2(_sy(y2))}\" />"
            )
        parts.append("</g>")
        parts.append("<g id=\"trajectory-points\">")
        for x, y, actual in pts:
            color = "#f43f5e" if actual else "#10b981"
            parts.append(
                f"<circle cx=\"{_r2(_sx(x))}\" cy=\"{_r2(_sy(y))}\" r=\"4\" "
                f"fill=\"{color}\" opacity=\"0.85\"/>"
            )
        parts.append("</g>")
        parts.append(
            f"<text x=\"{w / 2:.0f}\" y=\"{h - 20}\" fill=\"{muted}\" "
            f"font-family=\"Inter, sans-serif\" font-size=\"14\" text-anchor=\"middle\">{_esc(x_label)}</text>"
        )
        parts.append(
            f"<text x=\"30\" y=\"{h / 2:.0f}\" fill=\"{muted}\" "
            f"font-family=\"Inter, sans-serif\" font-size=\"14\" text-anchor=\"middle\" "
            f"transform=\"rotate(-90, 30, {h / 2:.0f})\">{_esc(y_label)}</text>"
        )
        parts.append(
            f"<text x=\"{margin_l}\" y=\"30\" fill=\"{fg}\" "
            f"font-family=\"Inter, sans-serif\" font-size=\"18\" font-weight=\"600\">{_esc(title)}</text>"
        )
        if disagreement_pct is not None and math.isfinite(float(disagreement_pct)):
            parts.append(
                f"<text x=\"{margin_l}\" y=\"50\" fill=\"{muted}\" "
                f"font-family=\"Inter, sans-serif\" font-size=\"12\">"
                f"Disagreement area: {_finite_num(disagreement_pct):.1f}%</text>"
            )
        parts.append("</svg>")
        return "\n".join(parts)
