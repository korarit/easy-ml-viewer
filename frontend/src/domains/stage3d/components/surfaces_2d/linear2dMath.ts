/**
 * Phase 42 — pure math helpers for exact linear 2D overlays.
 *
 * The backend ships the clipped analytic segments in world coords; this
 * module only derives what the backend cannot know: exact regression
 * elevation guides (parallel lines ŷ = t at the sampled contour's elevation
 * thresholds) mapped through the same raw→world transform.
 */

export type WorldSeg = [number, number, number, number];

const ZERO_TOL = 1e-12;

export function rawToWorld1D(v: number, min: number, max: number): number {
  const span = max - min;
  if (!Number.isFinite(v) || !Number.isFinite(min) || !Number.isFinite(max) || Math.abs(span) < 1e-12) return 0;
  const t = (v - min) / span;
  return -5 + Math.min(1, Math.max(0, t)) * 10;
}

/** Clip w1*x + w2*y + bEff = level to the raw rect; null when degenerate/missing. */
export function clipLineToRectRaw(
  w1: number,
  w2: number,
  bEff: number,
  level: number,
  xRange: [number, number],
  yRange: [number, number],
): [number, number, number, number] | null {
  const c = bEff - level;
  let [xMin, xMax] = xRange;
  let [yMin, yMax] = yRange;
  if (xMax < xMin) [xMin, xMax] = [xMax, xMin];
  if (yMax < yMin) [yMin, yMax] = [yMax, yMin];
  if (Math.abs(w1) <= ZERO_TOL && Math.abs(w2) <= ZERO_TOL) return null;
  if (Math.abs(w2) > ZERO_TOL && Math.abs(w1) <= ZERO_TOL) {
    const y = -c / w2;
    if (y < yMin || y > yMax) return null;
    return [xMin, y, xMax, y];
  }
  if (Math.abs(w1) > ZERO_TOL && Math.abs(w2) <= ZERO_TOL) {
    const x = -c / w1;
    if (x < xMin || x > xMax) return null;
    return [x, yMin, x, yMax];
  }
  const norm = Math.hypot(w1, w2);
  const dx = -w2 / norm;
  const dy = w1 / norm;
  const cx = (xMin + xMax) / 2;
  const cy = (yMin + yMax) / 2;
  const s0 = w1 * cx + w2 * cy + c;
  const px = cx - (s0 / (norm * norm)) * w1;
  const py = cy - (s0 / (norm * norm)) * w2;
  let tEnter = -Infinity;
  let tExit = Infinity;
  const axes: Array<[number, number, number, number]> = [
    [px, dx, xMin, xMax],
    [py, dy, yMin, yMax],
  ];
  for (const [p, d, lo, hi] of axes) {
    if (Math.abs(d) < 1e-15) {
      if (p < lo || p > hi) return null;
      continue;
    }
    let t0 = (lo - p) / d;
    let t1 = (hi - p) / d;
    if (t0 > t1) [t0, t1] = [t1, t0];
    tEnter = Math.max(tEnter, t0);
    tExit = Math.min(tExit, t1);
    if (tEnter > tExit) return null;
  }
  if (!Number.isFinite(tEnter) || !Number.isFinite(tExit) || tExit - tEnter < 1e-12) return null;
  return [px + tEnter * dx, py + tEnter * dy, px + tExit * dx, py + tExit * dy];
}

/**
 * Exact regression elevation guides: parallel lines ŷ(x1,x2) = t for each
 * sampled elevation threshold, in world coords. These coincide with the
 * Marching-Squares elevation isolines to < 1e-4 but stay razor-sharp.
 */
export function deriveExactElevationLines(
  w1: number,
  w2: number,
  bEff: number,
  thresholds: number[],
  xRangeRaw: [number, number],
  yRangeRaw: [number, number],
): Array<{ threshold: number; segment: WorldSeg }> {
  const out: Array<{ threshold: number; segment: WorldSeg }> = [];
  for (const t of thresholds) {
    if (!Number.isFinite(t)) continue;
    const raw = clipLineToRectRaw(w1, w2, bEff, t, xRangeRaw, yRangeRaw);
    if (!raw) continue;
    out.push({
      threshold: t,
      segment: [
        rawToWorld1D(raw[0], xRangeRaw[0], xRangeRaw[1]),
        rawToWorld1D(raw[1], yRangeRaw[0], yRangeRaw[1]),
        rawToWorld1D(raw[2], xRangeRaw[0], xRangeRaw[1]),
        rawToWorld1D(raw[3], yRangeRaw[0], yRangeRaw[1]),
      ],
    });
  }
  return out;
}
