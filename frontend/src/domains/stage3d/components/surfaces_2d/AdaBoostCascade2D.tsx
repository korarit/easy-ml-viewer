import { wy } from '../DecisionBoundaryContour2D';
import type { AdaBoostStage2D } from '../../../projection3d/types/projection.types';

export interface AdaBoostCascade2DProps {
  stages: AdaBoostStage2D[];
  /** Null = all stages; 0 = base prediction only (E45.6). */
  stage: number | null;
  baseValue: number | null;
  visible?: boolean;
}

/**
 * Phase 45 — staged weak-learner cascade.
 *
 * Shows the first T on-slice stump cuts with alpha-proportional stroke
 * width (same vote-share rule as the KNN spider lines); off-slice stages
 * contribute no geometry. Pure function of (stages, stage) — the slider
 * only changes one store number, so scrubbing T stays at 60 FPS.
 */
export function AdaBoostCascade2D({ stages, stage, baseValue, visible = true }: AdaBoostCascade2DProps) {
  if (!visible) return null;
  const onSlice = stages.filter((s) => s.axis !== 'off' && s.segmentWorld);
  const shown = stage === null ? onSlice : onSlice.filter((s) => s.stage <= stage);
  const maxW = Math.max(1e-12, ...shown.map((s) => Math.abs(s.weight)));
  const shownCount = stage === null ? stages.length : Math.min(stage, stages.length);

  return (
    <g className="adaboost-cascade-2d pointer-events-none" aria-label="AdaBoost staged cascade">
      {shown.map((s) => {
        const [x1, y1, x2, y2] = s.segmentWorld!;
        return (
          <line
            key={`stage-${s.stage}`}
            x1={x1}
            y1={wy(y1)}
            x2={x2}
            y2={wy(y2)}
            stroke="#ffffff"
            strokeWidth={0.03 + 0.09 * (Math.abs(s.weight) / maxW)}
            strokeOpacity={0.9}
            strokeLinecap="round"
          >
            <title>
              {`Stage ${s.stage} · ${s.axis} = ${s.threshold} · α = ${s.weight.toPrecision(4)}`}
            </title>
          </line>
        );
      })}
      {stage === 0 && (
        <title>{`Base prediction only (P = ${baseValue ?? 0.5}) — no weak learners yet`}</title>
      )}
      {shownCount === 0 && stage !== 0 && (
        <title>No on-slice weak-learner cuts at this stage</title>
      )}
    </g>
  );
}
