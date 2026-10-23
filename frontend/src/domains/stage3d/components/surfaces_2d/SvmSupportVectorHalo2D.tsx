import { wy } from '../DecisionBoundaryContour2D';
import type { SvmHalo2D } from '../../../projection3d/types/projection.types';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';
import { diamondPath, haloColor, haloDiamondRadius } from './distancePointIntel2dMath';

export interface SvmSupportVectorHalo2DProps {
  /** World-coord halos (mirrors the Phase-43 SvmState2D payload). */
  halos: SvmHalo2D[];
  visible?: boolean;
  /** Cap rendered dual-rings (diamonds always batch into ≤2 paths). */
  ringCap?: number;
}

/**
 * Phase 51 (mechanism B) — Support Vector Halo Dual-Ring.
 *
 * Each SV renders its Phase-43 diamond (sized by |α|/C, orange when the SV
 * violates the margin, cyan when it sits exactly on it — DoD #1) wrapped in
 * a concentric Dual-Ring whose stroke matches the same status color.
 * Diamonds batch into ≤2 paths by status; rings cap at `ringCap`.
 */
export function SvmSupportVectorHalo2D({ halos, visible = true, ringCap = 800 }: SvmSupportVectorHalo2DProps) {
  if (!visible || halos.length === 0) return null;
  const groups = new Map<SvmHalo2D['status'], SvmHalo2D[]>();
  for (const h of halos) {
    const list = groups.get(h.status) ?? [];
    list.push(h);
    groups.set(h.status, list);
  }
  const shown = halos.slice(0, Math.max(1, ringCap));
  return (
    <g className="sv-halo-dual-2d pointer-events-none" aria-label="Support vector dual-ring halos">
      {[...groups.entries()].map(([status, list]) => (
        <path
          key={`diamond-${status}`}
          d={list.map((h) => diamondPath(h.xWorld, h.yWorld, haloDiamondRadius(h.haloScale))).join('')}
          fill={haloColor(status)}
          fillOpacity={0.14}
          stroke={haloColor(status)}
          strokeWidth={0.04}
        >
          <title>
            {status === 'violates_margin'
              ? 'Unbounded SVs (|α| ≈ C, margin violators — orange diamond)'
              : 'Bounded SVs (0 < |α| < C, on the margin — cyan diamond)'}
          </title>
        </path>
      ))}
      {shown.map((h) => (
        <circle
          key={`ring-${h.trainIndex}`}
          cx={h.xWorld}
          cy={wy(h.yWorld)}
          r={RING_OUTER_RADIUS + 0.05}
          fill="none"
          stroke={haloColor(h.status)}
          strokeWidth={0.03}
          strokeDasharray={h.status === 'violates_margin' ? '0.16 0.1' : undefined}
          opacity={0.9}
        >
          <title>{`SV #${h.trainIndex} · α=${h.dualCoef.toPrecision(4)} · |α|/C=${h.haloScale.toFixed(2)}`}</title>
          {h.status === 'violates_margin' && (
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="1.6s" repeatCount="indefinite" />
          )}
        </circle>
      ))}
    </g>
  );
}
