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
  out.push('<g id="trajectory-points">');
  for (const [x, y, actual] of pts) {
    const color = actual ? '#f43f5e' : '#10b981';
    out.push(`<circle cx="${r2(sx(x))}" cy="${r2(sy(y))}" r="4" fill="${color}" opacity="0.85"/>`);
  }
  out.push('</g>');
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
