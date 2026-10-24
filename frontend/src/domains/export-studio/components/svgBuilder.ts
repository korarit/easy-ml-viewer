/**
 * Phase 48 — 100% offline client-side 2D decision-map SVG builder.
 * Mirrors `backend/.../svg_export_service.py` so vector export works with
 * zero network requests (air-gapped SLA): escape labels, map world coords
 * into the plot rect, emit decision paths + trajectory circles + axes.
 */

export interface SvgPoint {
  x: number;
  y: number;
  actual: number;
}

export interface SvgHexCell {
  cx: number;
  cy: number;
  r: number;
  count: number;
  errorRate: number;
  domClass: number | null;
}

export interface SvgDualRing {
  x: number;
  y: number;
  actualColor: string;
  predColor: string;
  isError: boolean;
}

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function finite(v: unknown, fallback = 0): number {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function hexPolygonPoints(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let k = 0; k < 6; k += 1) {
    const a = (Math.PI / 180) * (60 * k - 30);
    pts.push(`${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(' ');
}

function hexFillFor(mode: 'count' | 'error' | 'class', cell: SvgHexCell, maxCount: number): string {
  if (mode === 'error') {
    const t = Math.min(1, Math.max(0, cell.errorRate));
    const r = Math.round(244 * t + 148 * (1 - t));
    const g = Math.round(63 * t + 163 * (1 - t));
    const b = Math.round(94 * t + 184 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }
  if (mode === 'class') {
    const palette = ['#38bdf8', '#f472b6', '#a3e635', '#fbbf24', '#c084fc', '#2dd4bf'];
    const idx = cell.domClass === null ? -1 : Math.abs(cell.domClass) % palette.length;
    return idx < 0 ? '#64748b' : palette[idx]!;
  }
  const t = maxCount > 0 ? Math.min(1, cell.count / maxCount) : 0;
  const alpha = 0.25 + 0.55 * Math.sqrt(t);
  const shade = Math.round(148 + 107 * t);
  return `rgba(56,189,248,${alpha.toFixed(2)})`;
}

export function buildDecisionSvg(options: {
  xLabel: string;
  yLabel: string;
  boundarySegments: number[][];
  dataPoints: SvgPoint[];
  width?: number;
  height?: number;
  theme?: 'precision-dark' | 'paper-clean';
  title?: string;
  disagreementPct?: number | null;
  maxPoints?: number;
  /** Phase 56 — true-vector Point Intelligence layers (all optional). */
  hexbinCells?: SvgHexCell[];
  hexbinMode?: 'count' | 'error' | 'class';
  dualRings?: SvgDualRing[];
  trailPolyline?: Array<[number, number]>;
  densitySegments?: number[][];
}): string {
  const {
    xLabel,
    yLabel,
    boundarySegments,
    dataPoints,
    width = 1200,
    height = 900,
    theme = 'precision-dark',
    title = '2D Decision Boundary',
    disagreementPct = null,
    maxPoints = 20000,
    hexbinCells = [],
    hexbinMode = 'count',
    dualRings = [],
    trailPolyline = [],
    densitySegments = [],
  } = options;
  const w = Math.max(320, Math.min(3840, Math.floor(width)));
  const h = Math.max(240, Math.min(2160, Math.floor(height)));
  const dark = theme !== 'paper-clean';
  const bg = dark ? '#0b0f14' : '#ffffff';
  const fg = dark ? '#e2e8f0' : '#0f172a';
  const muted = dark ? '#94a3b8' : '#475569';
  const grid = dark ? '#1e293b' : '#e2e8f0';
  const boundary = dark ? '#ffffff' : '#0f172a';
  const mL = 72;
  const mR = 28;
  const mT = 56;
  const mB = 64;
  const plotW = Math.max(10, w - mL - mR);
  const plotH = Math.max(10, h - mT - mB);

  const segs: Array<[number, number, number, number]> = [];
  const xs: number[] = [];
  const ys: number[] = [];
  for (const s of boundarySegments) {
    if (!Array.isArray(s) || s.length < 4) continue;
    const vals = [finite(s[0], NaN), finite(s[1], NaN), finite(s[2], NaN), finite(s[3], NaN)];
    if (vals.some((v) => !Number.isFinite(v))) continue;
    segs.push(vals as [number, number, number, number]);
    xs.push(vals[0], vals[2]);
    ys.push(vals[1], vals[3]);
  }
  const pts: Array<[number, number, number]> = [];
  for (const p of dataPoints.slice(0, Math.max(0, maxPoints))) {
    const x = finite(p.x, NaN);
    const y = finite(p.y, NaN);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    pts.push([x, y, p.actual ? 1 : 0]);
    xs.push(x);
    ys.push(y);
  }
  let xMin = -5;
  let xMax = 5;
  let yMin = -5;
  let yMax = 5;
  if (xs.length > 0 && ys.length > 0) {
    xMin = Math.min(...xs);
    xMax = Math.max(...xs);
    yMin = Math.min(...ys);
    yMax = Math.max(...ys);
    if (xMax - xMin < 1e-9) {
      xMin -= 1;
      xMax += 1;
    }
    if (yMax - yMin < 1e-9) {
      yMin -= 1;
      yMax += 1;
    }
    const padx = (xMax - xMin) * 0.04;
    const pady = (yMax - yMin) * 0.04;
    xMin -= padx;
    xMax += padx;
    yMin -= pady;
    yMax += pady;
  }
  const sx = (x: number) => mL + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y: number) => mT + (1 - (y - yMin) / (yMax - yMin)) * plotH;
  const r2 = (v: number) => v.toFixed(2);

  const out: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img">`,
    `<title>${esc(title)}</title>`,
    `<rect width="100%" height="${h}" fill="${bg}"/>`,
    `<g id="grid" stroke="${grid}" stroke-width="1">`,
    `<rect x="${mL}" y="${mT}" width="${r2(plotW)}" height="${r2(plotH)}" fill="none"/>`,
  ];
  for (let i = 1; i < 5; i++) {
    const gx = mL + (plotW * i) / 5;
    const gy = mT + (plotH * i) / 5;
    out.push(`<line x1="${r2(gx)}" y1="${mT}" x2="${r2(gx)}" y2="${r2(mT + plotH)}"/>`);
    out.push(`<line x1="${mL}" y1="${r2(gy)}" x2="${r2(mL + plotW)}" y2="${r2(gy)}"/>`);
  }
  out.push('</g>');
  out.push(
    `<g id="decision-boundaries" stroke="${boundary}" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">`,
  );
  for (const [x1, y1, x2, y2] of segs) {
    out.push(`<line x1="${r2(sx(x1))}" y1="${r2(sy(y1))}" x2="${r2(sx(x2))}" y2="${r2(sy(y2))}" />`);
  }
  out.push('</g>');
  // Phase 56 — empirical density isolines (KDE-lite contours) as dashed vectors.
  const dSegs: Array<[number, number, number, number]> = [];
  for (const s of densitySegments.slice(0, 5000)) {
    if (!Array.isArray(s) || s.length < 4) continue;
    const vals = [finite(s[0], NaN), finite(s[1], NaN), finite(s[2], NaN), finite(s[3], NaN)];
    if (vals.some((v) => !Number.isFinite(v))) continue;
    dSegs.push(vals as [number, number, number, number]);
  }
  if (dSegs.length > 0) {
    out.push('<g id="density-contours" stroke="#818cf8" stroke-width="1.2" fill="none" stroke-dasharray="6 3" opacity="0.75">');
    for (const [x1, y1, x2, y2] of dSegs) {
      out.push(`<line x1="${r2(sx(x1))}" y1="${r2(sy(y1))}" x2="${r2(sx(x2))}" y2="${r2(sy(y2))}" />`);
    }
    out.push('</g>');
  }
  // Phase 56 — hexbin tessellation grid as true <polygon> vectors.
  const hexes = hexbinCells.slice(0, 2000).filter((c) => Number.isFinite(c.cx) && Number.isFinite(c.cy) && c.r > 0 && c.count > 0);
  if (hexes.length > 0) {
    const maxCount = Math.max(...hexes.map((c) => c.count));
    const xScale = plotW / (xMax - xMin);
    out.push('<g id="hexbin-grid" stroke="#0f172a" stroke-width="0.6" opacity="0.9">');
    for (const c of hexes) {
      const rPx = Math.max(1, c.r * xScale);
      out.push(
        `<polygon points="${hexPolygonPoints(sx(c.cx), sy(c.cy), rPx)}" fill="${hexFillFor(hexbinMode, c, maxCount)}" fill-opacity="0.42"><title>n=${c.count} err=${(c.errorRate * 100).toFixed(0)}% cls=${c.domClass ?? '–'}</title></polygon>`,
      );
    }
    out.push('</g>');
  }
  // Phase 56 — comet trail trajectory as a single vector polyline.
  const trail = trailPolyline.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)).slice(-500);
  if (trail.length >= 2) {
    const d = trail.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${r2(sx(x))} ${r2(sy(y))}`).join('');
    out.push(`<g id="comet-trail" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" opacity="0.9"><path d="${d}"/></g>`);
  }
  out.push('<g id="trajectory-points">');
  for (const [x, y, actual] of pts) {
    const color = actual ? '#f43f5e' : '#10b981';
    out.push(`<circle cx="${r2(sx(x))}" cy="${r2(sy(y))}" r="4" fill="${color}" opacity="0.85"/>`);
  }
  out.push('</g>');
  // Phase 56 — dual-ring actual/prediction glyphs as concentric vector circles.
  const rings = dualRings.filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y)).slice(0, 2000);
  if (rings.length > 0) {
    out.push('<g id="dual-rings" fill="none">');
    for (const d of rings) {
      const cx = r2(sx(d.x));
      const cy = r2(sy(d.y));
      out.push(`<circle cx="${cx}" cy="${cy}" r="7" stroke="${esc(d.predColor || '#e2e8f0')}" stroke-width="2.5"/>`);
      out.push(`<circle cx="${cx}" cy="${cy}" r="3.5" fill="${esc(d.actualColor || '#38bdf8')}"${d.isError ? ' stroke="#ef4444" stroke-width="1.5"' : ''}/>`);
    }
    out.push('</g>');
  }
  out.push(
    `<text x="${(w / 2).toFixed(0)}" y="${h - 20}" fill="${muted}" font-family="Inter, sans-serif" font-size="14" text-anchor="middle">${esc(xLabel)}</text>`,
  );
  out.push(
    `<text x="30" y="${(h / 2).toFixed(0)}" fill="${muted}" font-family="Inter, sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90, 30, ${(h / 2).toFixed(0)})">${esc(yLabel)}</text>`,
  );
  out.push(
    `<text x="${mL}" y="30" fill="${fg}" font-family="Inter, sans-serif" font-size="18" font-weight="600">${esc(title)}</text>`,
  );
  if (disagreementPct !== null && Number.isFinite(disagreementPct)) {
    out.push(
      `<text x="${mL}" y="50" fill="${muted}" font-family="Inter, sans-serif" font-size="12">Disagreement area: ${finite(disagreementPct).toFixed(1)}%</text>`,
    );
  }
  out.push('</svg>');
  return out.join('\n');
}

export function downloadSvgFile(svg: string, filename: string): void {
  const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.svg') ? filename : `${filename}.svg`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5000);
}
