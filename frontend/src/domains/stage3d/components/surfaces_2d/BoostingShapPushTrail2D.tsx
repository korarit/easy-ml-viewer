import { wy } from '../DecisionBoundaryContour2D';
import { PUSH_TRAIL_COLOR, cumulativePush, pushWorldScale } from './boostingPointIntel2dMath';

export interface BoostingShapPushTrail2DProps {
  /** World-space origin (the active point). */
  origin: { x: number; y: number };
  /** Per-stage 2D TreeSHAP pushes φ_m = (φx, φy) (spec §2.1). */
  stages: Array<{ phiX: number; phiY: number }>;
  /** Replay stages [0..upto]; null replays the full cascade. */
  uptoStage?: number | null;
  tone?: string;
  visible?: boolean;
}

/**
 * Phase 54 (mechanism A, DoD #1) — staged TreeSHAP push trail.
 *
 * The walk P_m = origin + s·Σ_{k≤m} φ_k replays how each tree pushed the
 * point across the decision boundary (scrub `uptoStage` to time-travel).
 * Played stages render solid; the remaining cascade ghosts behind as a
 * faint trace (Stage Progression Fading, spec §2.1). Two draw calls.
 */
export function BoostingShapPushTrail2D({
  origin,
  stages,
  uptoStage = null,
  tone = PUSH_TRAIL_COLOR,
  visible = true,
}: BoostingShapPushTrail2DProps) {
  if (!visible || stages.length === 0) return null;
  const m = stages.length;
  const upto = uptoStage === null || uptoStage === undefined
    ? m
    : Math.max(0, Math.min(Math.round(uptoStage), m));
  const scale = pushWorldScale(stages);
  const walk = cumulativePush(stages).map((p) => ({
    x: origin.x + p.x * scale,
    y: origin.y + p.y * scale,
  }));
  const segPath = (pts: Array<{ x: number; y: number }>): string =>
    pts.slice(1).map((p, i) => {
      const a = pts[i]!;
      return `M${a.x.toFixed(3)} ${wy(a.y).toFixed(3)}L${p.x.toFixed(3)} ${wy(p.y).toFixed(3)}`;
    }).join('');
  const played = walk.slice(0, upto + 1);
  const rest = walk.slice(upto);
  const end = walk[upto] ?? walk[0]!;
  return (
    <g className="boost-push-trail-2d pointer-events-none" aria-label="Staged TreeSHAP push trail">
      {rest.length > 1 && (
        <path
          d={segPath(rest)}
          fill="none"
          stroke={tone}
          strokeWidth={0.03}
          strokeDasharray="0.14 0.12"
          opacity={0.35}
        >
          <title>{`Ghost cascade: stages ${upto}–${m} not yet replayed`}</title>
        </path>
      )}
      {played.length > 1 && (
        <path d={segPath(played)} fill="none" stroke={tone} strokeWidth={0.05} opacity={0.95} strokeLinecap="round">
          <title>{`TreeSHAP push walk: stages 0–${upto} of ${m} (played)`}</title>
        </path>
      )}
      <circle cx={end.x} cy={wy(end.y)} r={0.06} fill={tone} opacity={0.95}>
        <title>{`Stage ${upto}/${m} push position`}</title>
      </circle>
      <text x={end.x + 0.12} y={wy(end.y) - 0.12} fontSize={0.26} fill={tone} opacity={0.95}>
        {`T=${upto}/${m}`}
      </text>
    </g>
  );
}
