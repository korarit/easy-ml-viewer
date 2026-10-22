import { wy } from '../DecisionBoundaryContour2D';
import type { SvmHalo2D } from '../../../projection3d/types/projection.types';

export interface SupportVectorRings2DProps {
  halos: SvmHalo2D[];
  visible?: boolean;
}

function haloRadiusWorld(scale: number): number {
  return 0.1 + 0.16 * Math.min(1, Math.max(0, scale));
}

function haloColor(status: SvmHalo2D['status']): string {
  // Free SVs on the margin edge: sky. Bound/violating (|α| ≈ C): deep orange.
  return status === 'violates_margin' ? '#fb923c' : '#7dd3fc';
}

/**
 * Phase 43 — diamond rings around support vectors, sized by |α|/C.
 *
 * Halos group by status into single SVG paths (≤3 draw calls no matter the
 * count), so even DTO-capped scenes (≤500) hold 60 FPS (E43.5). Larger SV
 * sets arrive pre-truncated by |dual| rank with an explicit flag + badge.
 */
export function SupportVectorRings2D({ halos, visible = true }: SupportVectorRings2DProps) {
  if (!visible || halos.length === 0) return null;
  const groups = new Map<SvmHalo2D['status'], SvmHalo2D[]>();
  for (const h of halos) {
    const list = groups.get(h.status) ?? [];
    list.push(h);
    groups.set(h.status, list);
  }
  return (
    <g className="sv-rings-2d pointer-events-none" aria-label="Support vector halos">
      {[...groups.entries()].map(([status, list]) => {
        const d = list
          .map((h) => {
            const r = haloRadiusWorld(h.haloScale);
            const x = h.xWorld;
            const y = wy(h.yWorld);
            return `M${(x - r).toFixed(3)} ${y.toFixed(3)}L${x.toFixed(3)} ${(y - r).toFixed(3)}L${(x + r).toFixed(3)} ${y.toFixed(3)}L${x.toFixed(3)} ${(y + r).toFixed(3)}Z`;
          })
          .join('');
        const color = haloColor(status);
        return (
          <path
            key={status}
            d={d}
            fill={color}
            fillOpacity={0.12}
            stroke={color}
            strokeWidth={0.04}
          >
            <title>
              {status === 'violates_margin'
                ? 'Bound support vectors (|α| ≈ C, margin violators)'
                : 'Free support vectors (on the margin edge)'}
            </title>
          </path>
        );
      })}
    </g>
  );
}
