import { useStage3DStore, type TrajectoryPoint3D } from '../../store/useStage3DStore';
import type {
  HexbinCell2D,
  Trees2DPayload,
} from '../../../projection3d/types/projection.types';
import { ManhattanStepTrail2D } from './ManhattanStepTrail2D';
import { ForestConsensusRing2D, type ConsensusRingDatum } from './ForestConsensusRing2D';
import { AdaBoostWeightHalo2D, type AdaBoostHaloDatum } from './AdaBoostWeightHalo2D';
import { LeafMinoritySpoke2D, type LeafMinoritySpoke } from './LeafMinoritySpoke2D';
import { LeafBoundedHexbin2D } from './LeafBoundedHexbin2D';
import { LinearResidualStem2D, type LinearResidualStem } from './LinearResidualStem2D';
import { findLeafAt } from './trees2dMath';
import {
  adaboostRingScale,
  forestAgreement,
  hexPurity,
  stepwiseResidual,
} from './treesPointIntel2dMath';

export interface TreesPointIntelOverlays2DProps {
  trees2d: Trees2DPayload | null;
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  hexbinCells: HexbinCell2D[];
  /** False for DecisionTree/RandomForest regressors (stems instead of spokes). */
  isClassification: boolean;
  classColor: (label: string | number | null) => string;
  /** Per-point per-tree votes (T labels each) — future frame endpoint; null stays dormant. */
  forestVotes?: Array<Array<string | number>> | null;
  /** Per-point AdaBoost sample weights ω_i — future endpoint; null stays dormant. */
  sampleWeights?: number[] | null;
}

const TREE_FAMILIES = new Set(['decision_tree', 'random_forest', 'extra_trees', 'adaboost']);

function isErrorPoint(p: TrajectoryPoint3D): boolean {
  const r = p.residual_error;
  if (r === null || r === undefined || !Number.isFinite(r)) return true;
  return Math.abs(r) > 0.5;
}

/**
 * Phase 53 — orchestrator for trees & ensemble point-intelligence overlays.
 *
 * Derives everything from the Phase-45 `trees2d` leaf boxes (no extra
 * fetch): Manhattan hop trails + leaf-transition pulses, impurity-backed
 * consensus rings, minority spokes to leaf centres (classifier) or
 * stepwise residual stems (regressor), and leaf-bounded hexbin tiers.
 * Forest vote rings and AdaBoost weight halos accept optional per-point
 * arrays and stay honestly dormant until the backend ships them (same
 * precedent as the Phase-51 ε-tube layers). Renders nothing for
 * non-tree models (null payload).
 */
export function TreesPointIntelOverlays2D({
  trees2d,
  points,
  activeFrameIndex,
  hexbinCells,
  isClassification,
  classColor,
  forestVotes = null,
  sampleWeights = null,
}: TreesPointIntelOverlays2DProps) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const showHexbins = useStage3DStore((s) => s.showHexbins);

  if (!trees2d || !TREE_FAMILIES.has(trees2d.family) || points.length === 0) {
    return null;
  }
  const active = Math.max(0, Math.min(activeFrameIndex, points.length - 1));
  const keep = (f: number): boolean => !showErrorsOnly || isErrorPoint(points[f]!) || f === active;
  const leaves = trees2d.leaves;

  // Mechanism A: orthogonal trail + pulse when the walk switches leaves.
  const trailPts = points.map((p) => ({ x: p.world_x, y: p.world_y }));
  let pulseBox: [number, number, number, number] | null = null;
  if (leaves.length > 0 && active > 0) {
    const prevLeaf = findLeafAt(leaves, points[active - 1]!.world_x, points[active - 1]!.world_y);
    const curLeaf = findLeafAt(leaves, points[active]!.world_x, points[active]!.world_y);
    if (prevLeaf && curLeaf && prevLeaf.leafId !== curLeaf.leafId) {
      pulseBox = curLeaf.boxWorld;
    }
  }

  // Mechanism B: forest consensus rings — exact votes when provided,
  // otherwise the leaf-impurity proxy (1 − Gini) for single-tree boxes.
  const rings: ConsensusRingDatum[] = [];
  if (showDualRings) {
    if (forestVotes && forestVotes.length === points.length) {
      const agreements = forestAgreement(forestVotes);
      points.forEach((p, f) => {
        if (!keep(f)) return;
        const share = agreements[f];
        if (share === undefined || !Number.isFinite(share)) return;
        rings.push({
          x: p.world_x,
          y: p.world_y,
          maxShare: share,
          color: classColor(p.raw_z_pred),
          label: `Frame ${f} · forest`,
        });
      });
    } else if (trees2d.family === 'decision_tree') {
      points.forEach((p, f) => {
        if (!keep(f)) return;
        const leaf = findLeafAt(leaves, p.world_x, p.world_y);
        if (!leaf || !Number.isFinite(leaf.impurity)) return;
        rings.push({
          x: p.world_x,
          y: p.world_y,
          maxShare: Math.min(1, Math.max(0, 1 - leaf.impurity)),
          color: classColor(leaf.prediction),
          label: `Frame ${f} · leaf ${leaf.leafId}`,
        });
      });
    }
  }

  // Mechanism B: AdaBoost weight halos (dormant without per-sample weights).
  const halos: AdaBoostHaloDatum[] = [];
  if (showDualRings && sampleWeights && sampleWeights.length === points.length) {
    const finite = sampleWeights.filter((w) => Number.isFinite(w) && w >= 0);
    const mean = finite.length > 0 ? finite.reduce((s, w) => s + (w as number), 0) / finite.length : NaN;
    points.forEach((p, f) => {
      if (!keep(f)) return;
      const w = sampleWeights[f];
      if (w === undefined || !Number.isFinite(w)) return;
      halos.push({
        x: p.world_x,
        y: p.world_y,
        scale: adaboostRingScale(w, mean),
        color: classColor(p.raw_z_pred),
        label: `Frame ${f} · ω=${Number(w).toPrecision(3)}`,
      });
    });
  }

  // Mechanism C: minority spokes (classifier) or stepwise stems (regressor).
  const spokes: LeafMinoritySpoke[] = [];
  const stems: LinearResidualStem[] = [];
  if (showResidualSpokes && leaves.length > 0) {
    if (isClassification) {
      points.forEach((p) => {
        if (!isErrorPoint(p)) return;
        const leaf = findLeafAt(leaves, p.world_x, p.world_y);
        if (!leaf || !(leaf.impurity > 0)) return;
        if (String(p.raw_z_actual) === String(leaf.prediction) && p.raw_z_actual !== null) return;
        const [xmin, xmax, ymin, ymax] = leaf.boxWorld;
        spokes.push({
          x1: p.world_x,
          y1: p.world_y,
          x2: (xmin + xmax) / 2,
          y2: (ymin + ymax) / 2,
          impurity: leaf.impurity,
        });
      });
    } else {
      const errs = points.map((p) => stepwiseResidual(p.raw_z_actual, p.raw_z_pred));
      const peak = errs.reduce((m, e) => (Number.isFinite(e) ? Math.max(m, Math.abs(e)) : m), 0);
      points.forEach((p, f) => {
        const e = errs[f];
        if (e === undefined || !Number.isFinite(e) || !(Math.abs(e) > 1e-12) || peak <= 1e-12) return;
        stems.push({
          x: p.world_x,
          y: p.world_y,
          length: (Math.abs(e) / peak) * 1.5,
          direction: e > 0 ? 1 : -1,
        });
      });
    }
  }

  // Mechanism D: leaf-bounded hexbin purity (DoD #4).
  let purities: Array<number | null> | null = null;
  if (showHexbins && hexbinCells.length > 0 && leaves.length > 0) {
    purities = hexbinCells.map((c) => {
      const leaf = findLeafAt(leaves, c.centerX, c.centerY);
      if (!leaf || !Number.isFinite(leaf.impurity)) return null;
      return hexPurity(1 - leaf.impurity, 1);
    });
  }

  return (
    <>
      <g aria-label="Trees and ensemble point intelligence overlays">
        {showCometTrails && (
          <ManhattanStepTrail2D points={trailPts} activeFrame={active} pulseBox={pulseBox} visible />
        )}
        <ForestConsensusRing2D rings={rings} visible={showDualRings} />
        <AdaBoostWeightHalo2D halos={halos} visible={showDualRings} />
        <LeafMinoritySpoke2D spokes={spokes} visible={showResidualSpokes} />
        {stems.length > 0 && <LinearResidualStem2D stems={stems} visible={showResidualSpokes} />}
        {purities && <LeafBoundedHexbin2D cells={hexbinCells} purities={purities} visible={showHexbins} />}
      </g>
      {(rings.length > 0 || halos.length > 0 || spokes.length > 0) && (
        <div
          className="font-mono-tabular pointer-events-none absolute right-1 bottom-14 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.72)', color: '#67e8f9' }}
          title="Trees point intelligence: consensus rings · weight halos · minority spokes"
        >
          {`${rings.length > 0 ? `▦ ${rings.length} rings` : ''}${halos.length > 0 ? ` · ◈ ${halos.length} halos` : ''}${spokes.length > 0 ? ` · ⚠ ${spokes.length} minority` : ''}`}
        </div>
      )}
    </>
  );
}
