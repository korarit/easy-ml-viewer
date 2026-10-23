import { wy } from '../DecisionBoundaryContour2D';

export interface LinearResidualStem {
  x: number;
  y: number;
  /** Normalized stem length in world units (peak residual → 1.5). */
  length: number;
  /** +1 over-predict (arrow up) · −1 under-predict (arrow down). */
  direction: 1 | -1;
}

export interface LinearResidualStem2DProps {
  stems: LinearResidualStem[];
  visible?: boolean;
}

/**
 * Phase 50 (mechanism C, regression) — exact vertical residual stems.
 *
 * Each stem rises/falls from the queued point with length ∝ |residual| and
 * an arrowhead marking over- vs under-prediction. Up/down batches share one
 * path each (≤ 4 draw calls total).
 */
export function LinearResidualStem2D({ stems, visible = true }: LinearResidualStem2DProps) {
  if (!visible || stems.length === 0) return null;
  const up = stems.filter((s) => s.direction > 0 && s.length > 1e-9);
  const down = stems.filter((s) => s.direction < 0 && s.length > 1e-9);
  const pathFor = (list: LinearResidualStem[], dir: 1 | -1): string =>
    list
      .map((s) => {
        const yEnd = dir > 0 ? s.y + s.length : s.y - s.length;
        return `M${s.x.toFixed(3)} ${wy(s.y).toFixed(3)}L${s.x.toFixed(3)} ${wy(yEnd).toFixed(3)}`;
      })
      .join('');
  return (
    <g className="linear-stem-2d pointer-events-none" aria-label="Residual stems">
      <defs>
        <marker id="li-stem-up" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill="#22c55e" />
        </marker>
        <marker id="li-stem-down" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill="#fb923c" />
        </marker>
      </defs>
      {up.length > 0 && (
        <path
          d={pathFor(up, 1)}
          fill="none"
          stroke="#22c55e"
          strokeWidth={0.035}
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#li-stem-up)"
        >
          <title>Over-predict stems (+residual, arrow up)</title>
        </path>
      )}
      {down.length > 0 && (
        <path
          d={pathFor(down, -1)}
          fill="none"
          stroke="#fb923c"
          strokeWidth={0.035}
          strokeDasharray="0.18 0.1"
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#li-stem-down)"
        >
          <title>Under-predict stems (−residual, arrow down)</title>
        </path>
      )}
    </g>
  );
}
