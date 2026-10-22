/** Phase 47 — shared math/helpers for neural / pipeline / ONNX 2D overlays. */
import { wy } from '../DecisionBoundaryContour2D';
import type {
  MlpHyperplane2D,
  OnnxLatentPoint2D,
} from '../../../projection3d/types/projection.types';

/** Batch hyperplane segments into active vs dead SVG subpaths (≤2 draw calls). */
export function batchHyperplanePaths(hyperplanes: MlpHyperplane2D[]): {
  active: string | null;
  dead: string | null;
} {
  let active = '';
  let dead = '';
  for (const h of hyperplanes) {
    if (!h.segmentWorld) continue;
    const [x1, y1, x2, y2] = h.segmentWorld;
    const d = `M${x1} ${wy(y1)}L${x2} ${wy(y2)}`;
    if (h.isDead) dead += d;
    else active += d;
  }
  return {
    active: active ? active : null,
    dead: dead ? dead : null,
  };
}

/**
 * Fit raw SVD latent points into a square SVG inset of `size` units at
 * (`svgLeft`, `svgTop`). Centers on the sample mean and scales by the max
 * absolute deviation so clusters keep their shape; degenerate clouds
 * collapse to the inset center. PCA sign is arbitrary, so no Y-flip is
 * applied — coordinates are abstract latent axes, not world units.
 */
export function fitLatentToWorld(
  points: OnnxLatentPoint2D[],
  svgLeft: number,
  svgTop: number,
  size: number,
): Array<{ x: number; y: number }> {
  if (points.length === 0 || size <= 0) return [];
  let mx = 0;
  let my = 0;
  for (const [px, py] of points) {
    mx += px;
    my += py;
  }
  mx /= points.length;
  my /= points.length;
  let span = 0;
  for (const [px, py] of points) {
    span = Math.max(span, Math.abs(px - mx), Math.abs(py - my));
  }
  if (!(span > 1e-12)) {
    return points.map(() => ({ x: svgLeft + size / 2, y: svgTop + size / 2 }));
  }
  const half = size / 2 - 0.12;
  return points.map(([px, py]) => ({
    x: svgLeft + size / 2 + ((px - mx) / span) * half,
    y: svgTop + size / 2 + ((py - my) / span) * half,
  }));
}

/**
 * Map a world X/Y ruler position to its scaled-space label.
 * Exact affine map: scaled = (raw(w) - mu) / sigma.
 */
export function scaledRulerLabel(
  w: number,
  rawMin: number,
  rawMax: number,
  mu: number,
  sigma: number,
): string {
  if (!(sigma > 1e-12)) return '—';
  const raw = rawMin + ((w + 5) / 10) * (rawMax - rawMin);
  const v = (raw - mu) / sigma;
  const abs = Math.abs(v);
  if (abs !== 0 && (abs >= 10000 || abs < 0.001)) return v.toExponential(1);
  return String(Math.round(v * 10000) / 10000);
}
