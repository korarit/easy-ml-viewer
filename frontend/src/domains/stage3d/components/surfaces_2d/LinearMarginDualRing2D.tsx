import { wy } from '../DecisionBoundaryContour2D';
import { MARGIN_RING_COLOR } from './linearPointIntel2dMath';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';

export interface LinearMarginRing {
  x: number;
  y: number;
  /** 'amber' = inside |z|≤1 corridor · 'logistic' = confidence-thickness ring. */
  kind: 'amber' | 'logistic';
  /** Stroke width for logistic rings (t_ring(P)); ignored for amber. */
  thickness?: number;
  color?: string;
}

export interface LinearMarginDualRing2DProps {
  rings: LinearMarginRing[];
  visible?: boolean;
  /** Cap rendered rings (worst z-scores first from the caller). */
  ringCap?: number;
}

/**
 * Phase 50 (mechanism B) — margin-violation & confidence rings.
 *
 * Corridor violators (`|z| ≤ 1`, SGD / RidgeClassifier) get a pulsing amber
 * dashed halo; logistic points get a confidence ring whose thickness grows
 * near `P = 0.5`. Per-point circles (violators are rare by construction).
 */
export function LinearMarginDualRing2D({ rings, visible = true, ringCap = 500 }: LinearMarginDualRing2DProps) {
  if (!visible || rings.length === 0) return null;
  const shown = rings.slice(0, Math.max(1, ringCap));
  return (
    <g className="linear-margin-ring-2d pointer-events-none" aria-label="Margin violation rings">
      {shown.map((r, i) =>
        r.kind === 'amber' ? (
          <circle
            key={i}
            cx={r.x}
            cy={wy(r.y)}
            r={RING_OUTER_RADIUS + 0.045}
            fill="none"
            stroke={MARGIN_RING_COLOR}
            strokeWidth={0.035}
            strokeDasharray="0.16 0.1"
            opacity={0.95}
          >
            <animate attributeName="opacity" values="0.95;0.35;0.95" dur="1.6s" repeatCount="indefinite" />
            <title>Inside margin corridor (|z| ≤ 1) — low-margin prediction</title>
          </circle>
        ) : (
          <circle
            key={i}
            cx={r.x}
            cy={wy(r.y)}
            r={RING_OUTER_RADIUS + 0.02 + (r.thickness ?? 0.02) / 2}
            fill="none"
            stroke={r.color ?? '#7dd3fc'}
            strokeWidth={r.thickness ?? 0.02}
            opacity={0.9}
          >
            <title>Logistic confidence ring (thick near P = 0.5)</title>
          </circle>
        ),
      )}
    </g>
  );
}
