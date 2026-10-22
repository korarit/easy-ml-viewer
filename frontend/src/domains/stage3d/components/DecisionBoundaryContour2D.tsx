import type { ContourLevel2D } from '../../projection3d/types/projection.types';
import { useStage3DStore } from '../store/useStage3DStore';

/** World Y points up; SVG Y points down — flip once, everywhere. */
export function wy(y: number): number {
  return -y;
}

function levelStyle(kind: ContourLevel2D['kind']): {
  stroke: string;
  strokeWidth: number;
  dasharray?: string;
  opacity: number;
} {
  if (kind === 'decision') {
    return { stroke: '#ffffff', strokeWidth: 0.09, opacity: 0.95 };
  }
  if (kind === 'confidence') {
    return { stroke: '#94a3b8', strokeWidth: 0.035, dasharray: '0.22 0.14', opacity: 0.8 };
  }
  return { stroke: '#fbbf24', strokeWidth: 0.05, dasharray: '0.3 0.12', opacity: 0.85 };
}

/**
 * Phase 41 — crisp vector decision-boundary + isoline overlay.
 * Segments arrive in world coords ([-5, +5]); each level renders as one
 * SVG path so a 120² grid stays at a handful of draw calls.
 */
export function DecisionBoundaryContour2D({ contours }: { contours: ContourLevel2D[] }) {
  const showBoundaryLines = useStage3DStore((s) => s.showBoundaryLines);
  const showConfidenceIsolines = useStage3DStore((s) => s.showConfidenceIsolines);

  const visible = contours.filter((c) => {
    if (c.kind === 'decision' || c.kind === 'elevation') return showBoundaryLines;
    return showConfidenceIsolines;
  });

  return (
    <g aria-label="Decision boundary contours">
      {visible.map((c, idx) => {
        if (c.lineSegments.length === 0) return null;
        const st = levelStyle(c.kind);
        const d = c.lineSegments
          .map(([x1, y1, x2, y2]) => `M${x1.toFixed(3)} ${wy(y1).toFixed(3)}L${x2.toFixed(3)} ${wy(y2).toFixed(3)}`)
          .join('');
        return (
          <path
            key={`${c.kind}-${c.threshold}-${idx}`}
            d={d}
            fill="none"
            stroke={st.stroke}
            strokeWidth={st.strokeWidth}
            strokeDasharray={st.dasharray}
            opacity={st.opacity}
            strokeLinecap="round"
          >
            <title>
              {c.kind === 'decision'
                ? `Decision boundary (P = ${c.threshold})`
                : c.kind === 'confidence'
                  ? `Confidence isoline (P = ${c.threshold})`
                  : `Elevation isoline (ŷ = ${c.threshold})`}
            </title>
          </path>
        );
      })}
    </g>
  );
}
