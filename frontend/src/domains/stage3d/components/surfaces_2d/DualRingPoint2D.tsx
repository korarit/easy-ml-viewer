import { wy } from '../DecisionBoundaryContour2D';
import {
  RING_CONTRAST_WIDTH,
  RING_INNER_RADIUS,
  RING_OUTER_RADIUS,
  RING_THICKNESS,
  contrastStroke,
  type GlyphShape,
} from './pointIntelligence2dMath';

export interface DualRingPoint2DProps {
  x: number;
  y: number;
  /** Inner-core fill: ground-truth class color. */
  actualColor: string;
  /** Outer-halo stroke: model-prediction color. */
  predColor: string;
  glyph?: GlyphShape;
  mispredicted?: boolean;
  isActive?: boolean;
  opacity?: number;
  /** The outer halo is light → dark contrast perimeter (spec §2.2). */
  haloIsLight?: boolean;
}

/**
 * Phase 49 (mechanism B) — Dual concentric rings + glyph shapes.
 *
 * Inner core carries the ground-truth hue, the outer halo the prediction
 * hue, and a high-contrast perimeter keeps dots legible on any heatmap.
 * Mispredictions add a pulsing halo + center cross (pure SVG, no CSS).
 */
export function DualRingPoint2D({
  x,
  y,
  actualColor,
  predColor,
  glyph = 'circle',
  mispredicted = false,
  isActive = false,
  opacity = 0.95,
  haloIsLight = false,
}: DualRingPoint2DProps) {
  const sy = wy(y);
  const scale = isActive ? 1.35 : 1;
  const rIn = RING_INNER_RADIUS * scale;
  const rOut = RING_OUTER_RADIUS * scale;
  const core =
    glyph === 'rect' ? (
      <rect x={x - rIn} y={sy - rIn} width={rIn * 2} height={rIn * 2} fill={actualColor} opacity={opacity} />
    ) : glyph === 'polygon' ? (
      <polygon
        points={`${x},${(sy - rIn * 1.2).toFixed(3)} ${(x + rIn * 1.1).toFixed(3)},${(sy + rIn * 0.8).toFixed(3)} ${(x - rIn * 1.1).toFixed(3)},${(sy + rIn * 0.8).toFixed(3)}`}
        fill={actualColor}
        opacity={opacity}
      />
    ) : (
      <circle cx={x} cy={sy} r={rIn} fill={actualColor} opacity={opacity} />
    );
  return (
    <g className="dual-ring-point-2d" aria-label={mispredicted ? 'Mispredicted point' : 'Point'}>
      {/* Contrast perimeter (outer). */}
      <circle
        cx={x}
        cy={sy}
        r={rOut + RING_CONTRAST_WIDTH}
        fill="none"
        stroke={contrastStroke(haloIsLight)}
        strokeWidth={RING_CONTRAST_WIDTH}
        opacity={Math.min(1, opacity + 0.05)}
      />
      {/* Prediction halo. */}
      <circle
        cx={x}
        cy={sy}
        r={rOut}
        fill="none"
        stroke={predColor}
        strokeWidth={RING_THICKNESS}
        opacity={opacity}
      >
        {mispredicted && (
          <animate attributeName="opacity" values={`${opacity};0.35;${opacity}`} dur="1.2s" repeatCount="indefinite" />
        )}
      </circle>
      {core}
      {mispredicted && (
        <text x={x} y={sy} fontSize={0.22} fill="#ef4444" textAnchor="middle" dy="0.08">
          ×
        </text>
      )}
    </g>
  );
}
