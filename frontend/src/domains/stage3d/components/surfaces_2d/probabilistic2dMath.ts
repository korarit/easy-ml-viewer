/** Phase 44 — shared math/helpers for probabilistic 2D overlays. */

export const SIGMA_LEVELS = [1, 2, 3] as const;

/** Class palette (≤8 ellipses; dark-lab friendly, distinct from axis colors). */
const CLASS_COLORS = [
  '#38bdf8',
  '#fb7185',
  '#a3e635',
  '#fbbf24',
  '#c084fc',
  '#2dd4bf',
  '#f472b6',
  '#94a3b8',
] as const;

export function ellipseColor(classIndex: number): string {
  const idx = Math.abs(Math.floor(classIndex)) % CLASS_COLORS.length;
  return CLASS_COLORS[idx];
}

/** Coverage of a 2D Gaussian k-sigma ring (for tooltips). */
export function sigmaCoverage(k: number): string {
  if (k === 1) return '39.3%';
  if (k === 2) return '86.5%';
  return '98.9%';
}
