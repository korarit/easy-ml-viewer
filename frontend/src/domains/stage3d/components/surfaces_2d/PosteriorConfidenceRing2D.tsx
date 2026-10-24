import { wy } from '../DecisionBoundaryContour2D';
import { confidenceRingRadius } from './probabilisticPointIntel2dMath';

export interface PosteriorRingDatum {
  x: number;
  y: number;
  /** max_c P(y=c|x) for this point. */
  maxProba: number;
  /** Slow pulse while inside the ambiguity zone. */
  ambiguous: boolean;
  /** Mini Σ halo (LDA/QDA): 1σ radii + negated data-space rotation. */
  halo?: { rx: number; ry: number; rotDeg: number };
  color: string;
  label: string;
}

export interface PosteriorConfidenceRing2DProps {
  rings: PosteriorRingDatum[];
  visible?: boolean;
  /** Cap rendered rings so dense queues stay legible (scales stay exact). */
  ringCap?: number;
}

/**
 * Phase 52 (mechanism B) — posterior confidence dual-ring.
 *
 * The outer ring breathes with the model's belief: full confidence renders
 * the base radius, a 50/50 coin flip shrinks it to half and pulses slowly
 * (DoD #1 companion). LDA/QDA points optionally carry a mini elliptical halo
 * echoing their class covariance orientation. Per-point circles (rare by
 * construction after the error filter upstream).
 */
export function PosteriorConfidenceRing2D({ rings, visible = true, ringCap = 500 }: PosteriorConfidenceRing2DProps) {
  if (!visible || rings.length === 0) return null;
  const shown = rings.slice(0, Math.max(1, ringCap));
  return (
    <g className="posterior-ring-2d pointer-events-none" aria-label="Posterior confidence rings">
      {shown.map((r, i) => (
        <g key={i}>
          {r.halo && (
            <ellipse
              cx={r.x}
              cy={wy(r.y)}
              rx={Math.max(r.halo.rx * 0.5, 1e-6)}
              ry={Math.max(r.halo.ry * 0.5, 1e-6)}
              transform={`rotate(${r.halo.rotDeg.toFixed(2)} ${r.x.toFixed(3)} ${wy(r.y).toFixed(3)})`}
              fill="none"
              stroke={r.color}
              strokeWidth={0.025}
              opacity={0.6}
            />
          )}
          <circle
            cx={r.x}
            cy={wy(r.y)}
            r={confidenceRingRadius(r.maxProba)}
            fill="none"
            stroke={r.color}
            strokeWidth={0.035}
            strokeDasharray={r.ambiguous ? '0.16 0.1' : undefined}
            opacity={0.9}
          >
            <title>{`${r.label} · max P=${r.maxProba.toFixed(3)}${r.ambiguous ? ' · ambiguity zone' : ''}`}</title>
            {r.ambiguous && <animate attributeName="opacity" values="0.9;0.35;0.9" dur="2.4s" repeatCount="indefinite" />}
          </circle>
        </g>
      ))}
    </g>
  );
}
