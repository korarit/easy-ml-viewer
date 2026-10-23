import { useStage3DStore, type TrajectoryPoint3D } from '../../store/useStage3DStore';
import type {
  HexbinCell2D,
  Linear2DPayload,
} from '../../../projection3d/types/projection.types';
import { LinearOrthogonalSpoke2D } from './LinearOrthogonalSpoke2D';
import { LinearMarginDualRing2D, type LinearMarginRing } from './LinearMarginDualRing2D';
import { LinearResidualStem2D, type LinearResidualStem } from './LinearResidualStem2D';
import { LinearHexbinAlignment2D } from './LinearHexbinAlignment2D';
import {
  errorDistanceCorrelation,
  expit,
  hexbinBoundaryFlags,
  isMarginViolator,
  logisticRingThickness,
  orthogonalSpoke,
  worldDistToLine,
  worldLine,
  zScoreRaw,
  type OrthogonalSpoke,
} from './linearPointIntel2dMath';

export interface LinearPointIntelOverlays2DProps {
  linear2d: Linear2DPayload | null;
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  hexbinCells: HexbinCell2D[];
  xRangeRaw: [number, number];
  yRangeRaw: [number, number];
}

function isErrorPoint(p: TrajectoryPoint3D): boolean {
  const r = p.residual_error;
  if (r === null || r === undefined || !Number.isFinite(r)) return true;
  return Math.abs(r) > 0.5;
}

/**
 * Phase 50 — orchestrator for linear-family point-intelligence overlays.
 *
 * Derives everything from the Phase-42 `linear2d` payload (no extra fetch):
 * ascent-colored comet signs flow into the shared layer via `trailWeights`,
 * orthogonal spokes / margin rings / residual stems render here, and
 * boundary-crossed hexbins get outlined when the Phase-49 hexbin layer is on.
 * Renders nothing for non-linear models (null payload) or degenerate axes.
 */
export function LinearPointIntelOverlays2D({
  linear2d,
  points,
  activeFrameIndex,
  hexbinCells,
  xRangeRaw,
  yRangeRaw,
}: LinearPointIntelOverlays2DProps) {
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showHexbins = useStage3DStore((s) => s.showHexbins);

  if (!linear2d || linear2d.bothZeroed || points.length === 0) return null;
  const line = worldLine(linear2d.w1, linear2d.w2, linear2d.bEff, xRangeRaw, yRangeRaw);
  if (line.degenerate) return null;

  const active = Math.max(0, Math.min(activeFrameIndex, points.length - 1));
  const isRegression = linear2d.task === 'regression';
  const isMulticlass = linear2d.isMulticlass;
  const isMarginFamily =
    !isRegression && !isMulticlass && (linear2d.family === 'sgd' || linear2d.family === 'ridge_classifier');
  const isLogistic = !isRegression && linear2d.family === 'logistic';

  // Mechanism C (classification): orthogonal spokes for mispredicted frames.
  const spokes: OrthogonalSpoke[] = [];
  if (!isRegression && !isMulticlass && showResidualSpokes) {
    points.forEach((p, f) => {
      if (showErrorsOnly && !isErrorPoint(p) && f !== active) return;
      if (!isErrorPoint(p)) return;
      const s = orthogonalSpoke(p.world_x, p.world_y, line);
      if (s) spokes.push(s);
    });
  }

  // Mechanism B: amber corridor rings + logistic confidence rings.
  const rings: LinearMarginRing[] = [];
  if (!isRegression && showDualRings && (isMarginFamily || isLogistic)) {
    points.forEach((p, f) => {
      if (showErrorsOnly && !isErrorPoint(p) && f !== active) return;
      const z = zScoreRaw(p.raw_x, p.raw_y, linear2d.w1, linear2d.w2, linear2d.bEff);
      if (!Number.isFinite(z)) return;
      if (isMarginFamily && isMarginViolator(z)) {
        rings.push({ x: p.world_x, y: p.world_y, kind: 'amber' });
      } else if (isLogistic && Math.abs(expit(z) - 0.5) < 0.15) {
        rings.push({
          x: p.world_x,
          y: p.world_y,
          kind: 'logistic',
          thickness: logisticRingThickness(expit(z)),
        });
      }
    });
  }

  // Mechanism C (regression): vertical residual stems, peak → 1.5 world.
  const stems: LinearResidualStem[] = [];
  if (isRegression && showResidualSpokes) {
    let peak = 0;
    const residuals = points.map((p) =>
      p.residual_error !== null && p.residual_error !== undefined && Number.isFinite(p.residual_error)
        ? p.residual_error
        : 0,
    );
    residuals.forEach((r) => {
      peak = Math.max(peak, Math.abs(r));
    });
    if (peak > 1e-12) {
      points.forEach((p, f) => {
        if (showErrorsOnly && !isErrorPoint(p) && f !== active) return;
        const r = residuals[f]!;
        if (Math.abs(r) <= 1e-12) return;
        stems.push({
          x: p.world_x,
          y: p.world_y,
          length: (Math.abs(r) / peak) * 1.5,
          direction: r > 0 ? 1 : -1,
        });
      });
    }
  }

  // Mechanism D: boundary-crossed hexbins + error/distance gradient badge.
  let errorCorr: number | null = null;
  let crossedCount = 0;
  if (showHexbins && hexbinCells.length > 0) {
    const flags = hexbinBoundaryFlags(hexbinCells, line);
    crossedCount = flags.filter(Boolean).length;
    const distances = hexbinCells.map((c) => worldDistToLine(c.centerX, c.centerY, line));
    errorCorr = errorDistanceCorrelation(
      distances,
      hexbinCells.map((c) => c.errorRate),
    );
    if (!Number.isFinite(errorCorr)) errorCorr = null;
  }

  return (
    <>
      <g aria-label="Linear point intelligence overlays">
        <LinearOrthogonalSpoke2D spokes={spokes} visible={showResidualSpokes} />
        <LinearMarginDualRing2D rings={rings} visible={showDualRings} />
        <LinearResidualStem2D stems={stems} visible={showResidualSpokes} />
        <LinearHexbinAlignment2D cells={hexbinCells} line={line} visible={showHexbins} />
      </g>
      {showHexbins && hexbinCells.length > 0 && (crossedCount > 0 || errorCorr !== null) && (
        <div
          className="font-mono-tabular pointer-events-none absolute right-1 bottom-14 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.72)', color: '#a5b4fc' }}
          title="Hexbins crossed by the exact decision line; r = Pearson(distance, error)"
        >
          {crossedCount > 0 && `⌀ ${crossedCount} bins × line`}
          {crossedCount > 0 && errorCorr !== null && ' · '}
          {errorCorr !== null && `r(d⊥,err)=${errorCorr.toFixed(2)}`}
        </div>
      )}
    </>
  );
}
