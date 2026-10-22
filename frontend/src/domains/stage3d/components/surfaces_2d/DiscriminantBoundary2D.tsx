import { wy } from '../DecisionBoundaryContour2D';
import type {
  DiscriminantLine2D,
  QdaConic2D,
} from '../../../projection3d/types/projection.types';

export interface DiscriminantBoundary2DProps {
  line: DiscriminantLine2D | null;
  conic: QdaConic2D | null;
  showLine?: boolean;
  showConic?: boolean;
}

/**
 * Phase 44 — LDA straight separator vs QDA conic-section curve.
 *
 * - LDA: the exact analytic line (solid white, like the Phase-42 exact line)
 *   with the shared-covariance label in its tooltip.
 * - QDA: the analytic conic segments (solid orange) sampled from the exact
 *   score difference — vector-crisp even where predict_proba is flat.
 * Renders nothing when both payloads are absent (GNB / discrete-NB models).
 */
export function DiscriminantBoundary2D({
  line,
  conic,
  showLine = true,
  showConic = true,
}: DiscriminantBoundary2DProps) {
  const lineSeg = showLine ? line?.segmentWorld : null;
  const conicSegs = showConic ? (conic?.segmentsWorld ?? []) : [];
  if (!lineSeg && conicSegs.length === 0) return null;
  const conicPath = conicSegs
    .map(([x1, y1, x2, y2]) => `M${x1.toFixed(3)} ${wy(y1).toFixed(3)}L${x2.toFixed(3)} ${wy(y2).toFixed(3)}`)
    .join('');
  return (
    <g className="discriminant-boundary-2d pointer-events-none" aria-label="Discriminant boundary">
      {lineSeg && (
        <line
          x1={lineSeg[0]}
          y1={wy(lineSeg[1])}
          x2={lineSeg[2]}
          y2={wy(lineSeg[3])}
          stroke="#ffffff"
          strokeWidth={0.09}
          strokeLinecap="round"
          opacity={0.95}
        >
          <title>{`LDA straight boundary — ${line?.label ?? 'shared covariance'}`}</title>
        </line>
      )}
      {conicPath && (
        <path
          d={conicPath}
          fill="none"
          stroke="#fb923c"
          strokeWidth={0.09}
          strokeLinecap="round"
          opacity={0.95}
        >
          <title>{`QDA ${conic?.conicType ?? 'conic'} boundary — ${conic?.label ?? ''}`}</title>
        </path>
      )}
    </g>
  );
}
