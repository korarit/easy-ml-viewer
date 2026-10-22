/**
 * Phase 25 — pure math helpers for the discrete-NB non-negative
 * probability surface + 3D log-likelihood/probability towers.
 *
 * Dependency-free (no React / Three imports) so tower heights, presence
 * flags and top-discriminative rankings stay unit-testable.
 *
 * Tower convention (blueprint §3.1):
 *   tower height of class k at frame t = P(C_k | x_t) (posterior).
 * Winner tower glows; the crown sits on the argmax tower.
 */

export type PresenceFilter = 'all' | 'present-only';

/** Clamp a tower height into the displayable posterior range. */
export function towerHeight(prob: number): number {
  if (!Number.isFinite(prob)) return 0;
  return Math.min(1, Math.max(0, prob));
}

/** Presence label of a raw feature value against the Bernoulli threshold. */
export function presenceLabel(value: number, binarize: number | null): 'present' | 'absent' {
  if (!Number.isFinite(value)) return 'absent';
  if (binarize === null || binarize === undefined) return value > 0 ? 'present' : 'absent';
  return value > binarize ? 'present' : 'absent';
}

/** Normalize tower heights so the winner always reaches full scale. */
export function normalizeTowerHeights(heights: number[]): number[] {
  const safe = heights.map((h) => (Number.isFinite(h) && h > 0 ? h : 0));
  const hMax = Math.max(...safe, 1e-12);
  if (!(hMax > 1e-12)) return safe.map(() => 0);
  return safe.map((h) => h / (hMax > 1 ? hMax : 1));
}

/** Rank features by absolute contribution (top discriminative first). */
export function topDiscriminative(
  contributions: Record<string, number>,
  topK: number,
): Array<{ name: string; value: number }> {
  return Object.entries(contributions)
    .map(([name, value]) => ({ name, value: Number.isFinite(value) ? value : 0 }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, Math.max(0, topK));
}

/** Filter contributions to present-only features (Bernoulli HUD switch). */
export function filterContributions(
  contributions: Record<string, number>,
  presence: Record<string, string>,
  mode: PresenceFilter,
  rawValues: number[],
  featureNames: string[],
  binarize: number | null,
): Record<string, number> {
  if (mode !== 'present-only') return contributions;
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(contributions)) {
    const known = presence[name];
    const flag =
      known === 'present' || known === 'absent'
        ? known
        : presenceLabel(rawValues[featureNames.indexOf(name)] ?? 0, binarize);
    if (flag === 'present') out[name] = value;
  }
  return out;
}

/** Softmax over joint log-probs (client-side posterior check). */
export function posteriorFromJointLog(joint: number[]): number[] {
  if (joint.length === 0) return [];
  const m = Math.max(...joint.map((v) => (Number.isFinite(v) ? v : -1e300)));
  const exps = joint.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0));
  const total = exps.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return joint.map(() => 0);
  return exps.map((e) => e / total);
}

/** Class color palette (tab10-ish, colorblind-safe order). */
export function towerColor(k: number): [number, number, number] {
  const palette: Array<[number, number, number]> = [
    [0x22 / 255, 0xd3 / 255, 0xee / 255],
    [0xfb / 255, 0x92 / 255, 0x3c / 255],
    [0x34 / 255, 0xd3 / 255, 0x99 / 255],
    [0xf5 / 255, 0x9e / 255, 0x0b / 255],
    [0xe8 / 255, 0x79 / 255, 0xf9 / 255],
    [0x81 / 255, 0x8c / 255, 0xf8 / 255],
    [0xf4 / 255, 0x3f / 255, 0x5e / 255],
    [0xa3 / 255, 0xe6 / 255, 0x35 / 255],
  ];
  return palette[Math.abs(k) % palette.length];
}
