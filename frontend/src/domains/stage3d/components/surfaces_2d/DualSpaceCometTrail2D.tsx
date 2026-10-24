import { wy } from '../DecisionBoundaryContour2D';
import { DUAL_SPACE_TRAIL_COLOR, LATENT_TRAIL_COLOR } from './neuralPointIntel2dMath';

export interface DualSpaceCometTrail2DProps {
  /** World-space trail in the active frame (raw or pre-mapped scaled). */
  points: Array<{ x: number; y: number }>;
  activeFrame: number;
  /** Trail decay window (Phase-49 comet kernel companion). */
  window?: number;
  /** Deep-ONNX latent geodesic styling vs the pipeline dual-space tone. */
  geodesic?: boolean;
  /** Which space the trail currently replays (badge title honesty). */
  spaceLabel?: string;
  visible?: boolean;
}

/**
 * Phase 55 (mechanism A, DoD #1) — dual-space comet trail.
 *
 * The orchestrator pre-maps the queued points through the exact scaler
 * affine, so this layer replays the same walk in whichever space the
 * viewport toggle selects: raw units stretch/compress into Z-scored
 * geometry (metric distortion) while the 1-to-1 mapping keeps every
 * frame honest. Deep-ONNX latent geodesics reuse the path with the
 * violet latent tone. One batched path + head dot holds 60 FPS.
 */
export function DualSpaceCometTrail2D({
  points,
  activeFrame,
  window = 8,
  geodesic = false,
  spaceLabel = 'raw',
  visible = true,
}: DualSpaceCometTrail2DProps) {
  if (!visible || points.length === 0) return null;
  const tone = geodesic ? LATENT_TRAIL_COLOR : DUAL_SPACE_TRAIL_COLOR;
  const active = Math.max(0, Math.min(activeFrame, points.length - 1));
  const lo = Math.max(0, active - window);
  const seg = points.slice(lo, active + 1);
  if (seg.length < 2) return null;
  const d = seg
    .slice(1)
    .map((p, i) => {
      const a = seg[i]!;
      return `M${a.x.toFixed(3)} ${wy(a.y).toFixed(3)}L${p.x.toFixed(3)} ${wy(p.y).toFixed(3)}`;
    })
    .join('');
  const head = seg[seg.length - 1]!;
  return (
    <g className="dual-space-trail-2d pointer-events-none" aria-label="Dual-space comet trail">
      <path d={d} fill="none" stroke={tone} strokeWidth={0.05} strokeLinecap="round" opacity={0.95}>
        <title>{`Dual-space trail (${spaceLabel}, frames ${lo}–${active})`}</title>
      </path>
      <circle cx={head.x} cy={wy(head.y)} r={0.07} fill={tone} opacity={0.95}>
        <title>{`Active frame ${active} in ${spaceLabel} space`}</title>
      </circle>
    </g>
  );
}
