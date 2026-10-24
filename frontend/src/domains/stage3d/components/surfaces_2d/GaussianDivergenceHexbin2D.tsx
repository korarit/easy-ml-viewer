import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath } from './pointIntelligence2dMath';
import { DIVERGENCE_OVER_COLOR, DIVERGENCE_UNDER_COLOR } from './probabilisticPointIntel2dMath';

export interface GaussianDivergenceHexbin2DProps {
  cells: HexbinCell2D[];
  /** Parallel Δρ = Count − N·p·Area values; NaN bins are skipped honestly. */
  divergences: Array<number | null>;
  visible?: boolean;
}

/**
 * Phase 52 (mechanism D, DoD #3) — empirical vs Gaussian divergence hexbin.
 *
 * Bins whose empirical count most violates the Gaussian assumption render
 * in two batched paths: red filled (model underestimates the local mass)
 * vs amber dashed (data desert the Gaussian expects but never observes).
 * NaN bins (no theory) never warn.
 */
export function GaussianDivergenceHexbin2D({ cells, divergences, visible = true }: GaussianDivergenceHexbin2DProps) {
  if (!visible || cells.length === 0 || divergences.length === 0) return null;
  const n = Math.min(cells.length, divergences.length);
  const finite = divergences
    .slice(0, n)
    .map((d, i) => ({ d, i }))
    .filter((e): e is { d: number; i: number } => typeof e.d === 'number' && Number.isFinite(e.d) && e.d !== 0);
  if (finite.length === 0) return null;
  const cut = Math.max(1, Math.floor(finite.length / 3));
  const ranked = [...finite].sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, cut);
  const over = ranked.filter((e) => e.d > 0);
  const under = ranked.filter((e) => e.d < 0);
  if (over.length === 0 && under.length === 0) return null;
  const pathOf = (idx: number[]): string =>
    idx.map((i) => hexPath(cells[i]!.centerX, cells[i]!.centerY, cells[i]!.radius)).join('');
  return (
    <g className="gauss-divergence-hexbin-2d pointer-events-none" aria-label="Gaussian divergence tiers">
      {over.length > 0 && (
        <path
          d={pathOf(over.map((e) => e.i))}
          fill={DIVERGENCE_OVER_COLOR}
          fillOpacity={0.35}
          stroke={DIVERGENCE_OVER_COLOR}
          strokeWidth={0.03}
        >
          <title>{`Over-dense vs Gaussian theory: ${over.length} bins (Δρ>0)`}</title>
        </path>
      )}
      {under.length > 0 && (
        <path
          d={pathOf(under.map((e) => e.i))}
          fill="none"
          stroke={DIVERGENCE_UNDER_COLOR}
          strokeWidth={0.035}
          strokeDasharray="0.2 0.12"
          opacity={0.9}
        >
          <title>{`Data deserts vs Gaussian theory: ${under.length} bins (Δρ<0)`}</title>
        </path>
      )}
    </g>
  );
}
