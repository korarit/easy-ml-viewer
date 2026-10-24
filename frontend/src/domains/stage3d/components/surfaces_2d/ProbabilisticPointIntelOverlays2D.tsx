import { useStage3DStore, type TrajectoryPoint3D } from '../../store/useStage3DStore';
import type {
  HexbinCell2D,
  Probabilistic2DPayload,
} from '../../../projection3d/types/projection.types';
import { BayesEntropyCometTrail2D } from './BayesEntropyCometTrail2D';
import { PosteriorConfidenceRing2D, type PosteriorRingDatum } from './PosteriorConfidenceRing2D';
import { CentroidAttractorSpoke2D, type CentroidSpoke } from './CentroidAttractorSpoke2D';
import { GaussianDivergenceHexbin2D } from './GaussianDivergenceHexbin2D';
import {
  entropyTrailColor,
  gaussianMixturePdf,
  gaussianPosterior2D,
  hexAreaRaw,
  isAmbiguous,
  shannonEntropy,
  worldToRaw,
  type ClassGaussian2D,
} from './probabilisticPointIntel2dMath';

export interface ProbabilisticPointIntelOverlays2DProps {
  prob2d: Probabilistic2DPayload | null;
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  hexbinCells: HexbinCell2D[];
  xRangeRaw: [number, number];
  yRangeRaw: [number, number];
  classColor: (label: string | number | null) => string;
}

const GAUSSIAN_FAMILIES = new Set(['gnb', 'lda', 'qda']);

function isErrorPoint(p: TrajectoryPoint3D): boolean {
  const r = p.residual_error;
  if (r === null || r === undefined || !Number.isFinite(r)) return true;
  return Math.abs(r) > 0.5;
}

/** LLR from posterior shares: ℓ = log P − log prior (shared const cancels). */
function llrFromPosterior(
  post: number[],
  priors: number[],
  trueIdx: number,
  predIdx: number,
): number {
  const pt = post[trueIdx];
  const pp = post[predIdx];
  const rt = priors[trueIdx];
  const rp = priors[predIdx];
  if (!pt || !pp || !rt || !rp || pt <= 0 || pp <= 0 || rt <= 0 || rp <= 0) return NaN;
  return Math.log(pt / rt) - Math.log(pp / rp);
}

/**
 * Phase 52 — orchestrator for probabilistic point-intelligence overlays.
 *
 * Derives everything from the Phase-44 `probabilistic2d` ellipses (no extra
 * fetch): 2D-slice posteriors → entropy trail colors, confidence rings,
 * centroid spokes and Gaussian divergence. Discrete-NB models carry no
 * Gaussian parameters, so the layer honestly renders nothing for them.
 * Renders nothing for non-probabilistic models (null payload).
 */
export function ProbabilisticPointIntelOverlays2D({
  prob2d,
  points,
  activeFrameIndex,
  hexbinCells,
  xRangeRaw,
  yRangeRaw,
  classColor,
}: ProbabilisticPointIntelOverlays2DProps) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const showHexbins = useStage3DStore((s) => s.showHexbins);

  if (!prob2d || !GAUSSIAN_FAMILIES.has(prob2d.family) || prob2d.ellipses.length === 0 || points.length === 0) {
    return null;
  }
  const nClasses = prob2d.ellipses.length;
  const gaussians: ClassGaussian2D[] = prob2d.ellipses.map((e) => ({
    meanRaw: e.meanRaw,
    stdRaw: e.stdRaw,
    rotationRad: e.rotationRad,
    prior: e.prior,
    label: e.label,
  }));
  const priors = gaussians.map((g) =>
    g.prior !== null && Number.isFinite(g.prior) && g.prior > 0 ? g.prior : 1 / nClasses,
  );
  const pSum = priors.reduce((s, p) => s + p, 0);
  const normPriors = priors.map((p) => p / pSum);

  // Per-point 2D-slice posterior on the visualized axis pair.
  const frames = points.map((p) => {
    const post = gaussianPosterior2D(p.raw_x, p.raw_y, gaussians);
    const finite = post.every((v) => Number.isFinite(v));
    const h = finite ? shannonEntropy(post) : NaN;
    let predIdx = -1;
    let maxP = NaN;
    if (finite) {
      predIdx = post.indexOf(Math.max(...post));
      maxP = post[predIdx]!;
    }
    return { p, post, h, predIdx, maxP };
  });

  const active = Math.max(0, Math.min(activeFrameIndex, points.length - 1));
  const keep = (f: number): boolean => !showErrorsOnly || isErrorPoint(points[f]!) || f === active;

  // Mechanism A: entropy-tinted trail colors (DoD #1).
  const trailColors = frames.map(({ h, predIdx }) => {
    const confident = predIdx >= 0 ? classColor(gaussians[predIdx]!.label) : '#94a3b8';
    return entropyTrailColor(h, nClasses, confident);
  });

  // Mechanism B: confidence rings + LDA/QDA mini Σ halos.
  const rings: PosteriorRingDatum[] = [];
  if (showDualRings) {
    frames.forEach(({ p, h, predIdx, maxP }, f) => {
      if (!keep(f) || predIdx < 0 || !Number.isFinite(maxP)) return;
      const e = prob2d.ellipses[predIdx]!;
      rings.push({
        x: p.world_x,
        y: p.world_y,
        maxProba: maxP,
        ambiguous: isAmbiguous(h, nClasses),
            halo:
              prob2d.family === 'lda' || prob2d.family === 'qda'
                ? {
                    // Same corrected mapping as GaussianEllipses2D: local-x
                    // lands on the major-axis direction, so rx = major.
                    rx: e.radiiWorld1Sigma[1],
                    ry: e.radiiWorld1Sigma[0],
                    rotDeg: (-e.rotationRad * 180) / Math.PI,
                  }
                : undefined,
        color: classColor(e.label),
        label: `Frame ${f} · class ${String(e.label)}`,
      });
    });
  }

  // Mechanism C: dual centroid spokes for mispredicted frames (DoD #2).
  const spokes: CentroidSpoke[] = [];
  if (showResidualSpokes) {
    frames.forEach(({ p, post, predIdx }, f) => {
      if (!isErrorPoint(p) || predIdx < 0 || post.some((v) => !Number.isFinite(v))) return;
      const trueIdx = prob2d.ellipses.findIndex(
        (e) => String(e.label) === String(p.raw_z_actual),
      );
      if (trueIdx < 0 || trueIdx === predIdx) return;
      const eTrue = prob2d.ellipses[trueIdx]!;
      const ePred = prob2d.ellipses[predIdx]!;
      spokes.push(
        { x1: p.world_x, y1: p.world_y, x2: eTrue.centerWorld[0], y2: eTrue.centerWorld[1], kind: 'true', llr: llrFromPosterior(post, normPriors, trueIdx, predIdx) },
        { x1: p.world_x, y1: p.world_y, x2: ePred.centerWorld[0], y2: ePred.centerWorld[1], kind: 'pred' },
      );
    });
  }

  // Mechanism D: empirical vs Gaussian divergence per hexbin (DoD #3).
  let divergences: Array<number | null> | null = null;
  let maxAbsDiv = 0;
  if (showHexbins && hexbinCells.length > 0) {
    const spanX = xRangeRaw[1] - xRangeRaw[0];
    const spanY = yRangeRaw[1] - yRangeRaw[0];
    const total = hexbinCells.reduce((s, c) => s + c.count, 0);
    const area = hexAreaRaw(hexbinCells[0]?.radius ?? NaN, spanX, spanY);
    if (total > 0 && Number.isFinite(area)) {
      divergences = hexbinCells.map((c) => {
        const rx = worldToRaw(c.centerX, xRangeRaw[0], xRangeRaw[1]);
        const ry = worldToRaw(c.centerY, xRangeRaw[0], xRangeRaw[1]);
        const pdf = gaussianMixturePdf(rx, ry, gaussians);
        if (!Number.isFinite(pdf)) return null;
        const d = c.count - total * pdf * area;
        if (Number.isFinite(d)) maxAbsDiv = Math.max(maxAbsDiv, Math.abs(d));
        return Number.isFinite(d) ? d : null;
      });
    }
  }

  return (
    <>
      <g aria-label="Probabilistic point intelligence overlays">
        {showCometTrails && (
          <BayesEntropyCometTrail2D
            points={points.map((p) => ({ x: p.world_x, y: p.world_y }))}
            colors={trailColors}
            activeFrame={active}
            visible
          />
        )}
        <PosteriorConfidenceRing2D rings={rings} visible={showDualRings} />
        <CentroidAttractorSpoke2D spokes={spokes} visible={showResidualSpokes} />
        {divergences && <GaussianDivergenceHexbin2D cells={hexbinCells} divergences={divergences} visible={showHexbins} />}
      </g>
      {divergences && maxAbsDiv > 0 && (
        <div
          className="font-mono-tabular pointer-events-none absolute right-1 bottom-14 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.72)', color: '#f0abfc' }}
          title="Max |Δρ| between empirical hexbin counts and Gaussian theory"
        >
          {`Δρ max ${maxAbsDiv.toPrecision(3)}${spokes.length > 0 ? ` · ${spokes.length / 2} paradox pts` : ''}`}
        </div>
      )}
    </>
  );
}
