import type { TrajectoryPoint3D } from '../store/useStage3DStore';
import { wy } from './DecisionBoundaryContour2D';

/**
 * Phase 41 — 2D active-frame reticle: double ring + crosshairs snapped to
 * the X/Y rulers with live raw-value labels, plus a 10-frame comet trail.
 */
export function ActiveDataPointReticle2D({
  points,
  activeFrameIndex,
}: {
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
}) {
  if (points.length === 0) return null;
  const active = points[Math.max(0, Math.min(activeFrameIndex, points.length - 1))];
  const ax = active.world_x;
  const ay = wy(active.world_y);

  const trailStart = Math.max(0, points.indexOf(active) - 10);
  const trail = points.slice(trailStart, points.indexOf(active) + 1);
  const trailPath =
    trail.length > 1
      ? trail.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.world_x.toFixed(3)} ${wy(p.world_y).toFixed(3)}`).join('')
      : null;

  return (
    <g aria-label="Active data point reticle" pointerEvents="none">
      {trailPath && (
        <path d={trailPath} fill="none" stroke="#fbbf24" strokeWidth={0.06} opacity={0.55} strokeLinecap="round" />
      )}
      {/* Crosshairs to the rulers */}
      <line x1={-5} y1={ay} x2={5} y2={ay} stroke="#fbbf24" strokeWidth={0.02} opacity={0.45} strokeDasharray="0.18 0.12" />
      <line x1={ax} y1={-5} x2={ax} y2={5} stroke="#fbbf24" strokeWidth={0.02} opacity={0.45} strokeDasharray="0.18 0.12" />
      {/* Double ring reticle */}
      <circle cx={ax} cy={ay} r={0.34} fill="none" stroke="#fbbf24" strokeWidth={0.07} opacity={0.95} />
      <circle cx={ax} cy={ay} r={0.2} fill="none" stroke="#fbbf24" strokeWidth={0.04} opacity={0.7} />
      <circle cx={ax} cy={ay} r={0.07} fill="#fbbf24" opacity={0.95} />
      <text x={ax + 0.42} y={ay - 0.3} fontSize={0.42} fill="#fbbf24" opacity={0.95}>
        {`#${active.original_row_index}`}
      </text>
    </g>
  );
}
