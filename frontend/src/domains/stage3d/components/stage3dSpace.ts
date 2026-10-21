/**
 * Phase 08 — world-space (X right, Y depth, Z up) <-> three-space (X right, Y up, Z depth) mapping.
 *
 * Backend `world_vertices_xyz` arrives as [wx, wy, wz, ...] in the normalized
 * [-5, +5] cube where Z is the vertical prediction axis. Three.js is Y-up,
 * so every geometry swizzles once at buffer-build time: three = (wx, wz, wy).
 * Camera presets and tick labels are all expressed through these helpers.
 */
export const GROUND_WORLD_Z = -5.0;

export function worldToThree(wx: number, wy: number, wz: number): [number, number, number] {
  return [wx, wz, wy];
}

export function formatTickLabel(raw: number): string {
  const abs = Math.abs(raw);
  if (abs !== 0 && (abs >= 1_000_000 || abs < 0.001)) return raw.toExponential(2);
  if (abs >= 1000) return raw.toLocaleString('en-US', { maximumFractionDigits: 1 });
  if (Number.isInteger(raw)) return String(raw);
  return String(Math.round(raw * 10000) / 10000);
}
