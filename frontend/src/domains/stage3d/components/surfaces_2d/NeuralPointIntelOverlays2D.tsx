import { useStage3DStore, type TrajectoryPoint3D } from '../../store/useStage3DStore';
import type {
  HexbinCell2D,
  Neural2DPayload,
} from '../../../projection3d/types/projection.types';
import { DualSpaceCometTrail2D } from './DualSpaceCometTrail2D';
import { SoftmaxLogitGapRing2D, type LogitGapRingDatum } from './SoftmaxLogitGapRing2D';
import { LossGradientSpoke2D, type LossGradientSpoke } from './LossGradientSpoke2D';
import { DataDesertHexbin2D } from './DataDesertHexbin2D';
import {
  desertHazard,
  dualSpaceWorld,
  gradSpokeLengths,
  gradUnit,
  logitGap,
  logitGapStatus,
} from './neuralPointIntel2dMath';

export interface NeuralPointIntelOverlays2DProps {
  neural2d: Neural2DPayload | null;
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  hexbinCells: HexbinCell2D[];
  isClassification: boolean;
  classColor: (label: string | number | null) => string;
  /** Active viewport space (Phase-47 toggle); scaled replays the Z-frame. */
  pipelineSpace: 'raw' | 'scaled';
  xRangeRaw: [number, number];
  yRangeRaw: [number, number];
  /** Per-point pre-softmax logits — future endpoint; null stays dormant. */
  logits?: number[][] | null;
  /** Per-point input loss gradients (gx, gy) — future endpoint; null dormant. */
  gradients?: Array<{ gx: number; gy: number } | null> | null;
  /** Per-hexbin max class proba — future endpoint for desert tiers. */
  cellMaxProba?: Array<number | null> | null;
}

const NEURAL_FAMILIES = new Set(['mlp', 'pipeline', 'onnx_classical', 'onnx_deep']);

function isErrorPoint(p: TrajectoryPoint3D): boolean {
  const r = p.residual_error;
  if (r === null || r === undefined || !Number.isFinite(r)) return true;
  return Math.abs(r) > 0.5;
}

/**
 * Phase 55 — orchestrator for neural / pipeline / ONNX point intelligence.
 *
 * Dual-space trails remap through the exact scaler affine when the
 * viewport selects scaled space (1-to-1, DoD #1); logit-gap rings badge
 * saturated / tie / decisive points from pre-softmax Δz (DoD #2); loss
 * gradient spokes and desert hazard tiers ride optional per-point /
 * per-cell arrays and stay honestly dormant until the backend ships them
 * (same precedent as the Phase-51/53/54 dormant layers). Renders nothing
 * for non-neural models.
 */
export function NeuralPointIntelOverlays2D({
  neural2d,
  points,
  activeFrameIndex,
  hexbinCells,
  isClassification,
  classColor,
  pipelineSpace,
  xRangeRaw,
  yRangeRaw,
  logits = null,
  gradients = null,
  cellMaxProba = null,
}: NeuralPointIntelOverlays2DProps) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const showHexbins = useStage3DStore((s) => s.showHexbins);

  if (!neural2d || !NEURAL_FAMILIES.has(neural2d.family) || points.length === 0) {
    return null;
  }
  const active = Math.max(0, Math.min(activeFrameIndex, points.length - 1));
  const keep = (f: number): boolean => !showErrorsOnly || isErrorPoint(points[f]!) || f === active;

  // Mechanism A: dual-space trail (DoD #1).
  const pipe = neural2d.family === 'pipeline' ? neural2d.pipeline : null;
  const scaledLive =
    pipelineSpace === 'scaled' &&
    !!pipe?.hasToggle &&
    pipe.muX !== null &&
    pipe.sigmaX !== null &&
    pipe.muY !== null &&
    pipe.sigmaY !== null &&
    pipe.scaledXRange !== null &&
    pipe.scaledYRange !== null;
  const trailPts = points.map((p) => {
    if (!scaledLive) return { x: p.world_x, y: p.world_y };
    const sx = dualSpaceWorld(
      p.world_x, xRangeRaw[0], xRangeRaw[1], pipe!.muX!, pipe!.sigmaX!,
      pipe!.scaledXRange![0], pipe!.scaledXRange![1],
    );
    const sy = dualSpaceWorld(
      p.world_y, yRangeRaw[0], yRangeRaw[1], pipe!.muY!, pipe!.sigmaY!,
      pipe!.scaledYRange![0], pipe!.scaledYRange![1],
    );
    return {
      x: Number.isFinite(sx) ? sx : p.world_x,
      y: Number.isFinite(sy) ? sy : p.world_y,
    };
  });

  // Mechanism B: softmax logit-gap rings (DoD #2).
  const rings: LogitGapRingDatum[] = [];
  if (showDualRings && isClassification && logits && logits.length === points.length) {
    points.forEach((p, f) => {
      if (!keep(f)) return;
      const row = logits[f];
      if (!row) return;
      const gap = logitGap(row);
      const status = logitGapStatus(gap);
      if (status === 'unknown') return;
      rings.push({
        x: p.world_x,
        y: p.world_y,
        gap,
        status,
        color: classColor(p.raw_z_pred),
        label: `Frame ${f}`,
      });
    });
  }

  // Note: queued points carry predicted labels, not probas, so there is
  // no honest binary |logit| fallback — the full-logit path above is live
  // once the backend ships per-point pre-softmax outputs.

  // Mechanism C: input loss-gradient spokes.
  const spokes: LossGradientSpoke[] = [];
  if (showResidualSpokes && gradients && gradients.length === points.length) {
    const norms = gradients.map((g) =>
      g && Number.isFinite(g.gx) && Number.isFinite(g.gy) ? Math.hypot(g.gx, g.gy) : NaN,
    );
    const { lengths, live } = gradSpokeLengths(norms);
    points.forEach((p, f) => {
      if (!keep(f)) return;
      if (!live[f]) return;
      const g = gradients[f]!;
      const { ux, uy } = gradUnit(g.gx, g.gy);
      if (Math.hypot(ux, uy) <= 1e-9) return;
      spokes.push({
        x: p.world_x,
        y: p.world_y,
        ux,
        uy,
        length: lengths[f] as number,
        magnitude: norms[f],
      });
    });
  }

  // Mechanism D: data-desert hazard tiers (DoD #3).
  let hazard: Array<number | null> | null = null;
  if (showHexbins && hexbinCells.length > 0 && cellMaxProba && cellMaxProba.length >= hexbinCells.length) {
    hazard = hexbinCells.map((c, i) => {
      const mp = cellMaxProba[i];
      if (mp === null || mp === undefined || !Number.isFinite(mp)) return null;
      return desertHazard(c.count, mp).hazard;
    });
    if (hazard.every((h) => h === null)) hazard = null;
  }

  return (
    <>
      <g aria-label="Neural point intelligence overlays">
        {showCometTrails && (
          <DualSpaceCometTrail2D
            points={trailPts}
            activeFrame={active}
            geodesic={neural2d.family === 'onnx_deep'}
            spaceLabel={scaledLive ? 'scaled' : 'raw'}
            visible
          />
        )}
        <SoftmaxLogitGapRing2D rings={rings} visible={showDualRings} />
        <LossGradientSpoke2D spokes={spokes} visible={showResidualSpokes} />
        {hazard && <DataDesertHexbin2D cells={hexbinCells} hazard={hazard} visible={showHexbins} />}
      </g>
      {(rings.length > 0 || spokes.length > 0) && (
        <div
          className="font-mono-tabular pointer-events-none absolute right-1 bottom-14 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.72)', color: '#67e8f9' }}
          title="Neural point intelligence: dual-space trails · logit-gap rings · gradient spokes · desert detectors"
        >
          {`${scaledLive ? '◈ Scaled trail' : ''}${rings.length > 0 ? ` · ◈ ${rings.length} gaps` : ''}${spokes.length > 0 ? ` · ↯ ${spokes.length} grads` : ''}`}
        </div>
      )}
    </>
  );
}
