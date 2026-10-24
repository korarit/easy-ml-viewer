import { wy } from '../DecisionBoundaryContour2D';
import { useStage3DStore, type TrajectoryPoint3D } from '../../store/useStage3DStore';
import type {
  BoostFrame2DPayload,
  Boosting2DPayload,
  HexbinCell2D,
} from '../../../projection3d/types/projection.types';
import { BoostingShapPushTrail2D } from './BoostingShapPushTrail2D';
import { FunctionalMarginRing2D, type MarginRingDatum } from './FunctionalMarginRing2D';
import { PseudoResidualSpoke2D, type PseudoResidualSpoke } from './PseudoResidualSpoke2D';
import { QuantizedBinHexbin2D } from './QuantizedBinHexbin2D';
import { findBoostLeafAt } from './boosting2dMath';
import {
  MARGIN_CRITICAL_COLOR,
  MARGIN_NEGATIVE_COLOR,
  PUSH_TRAIL_COLOR,
  functionalMargin,
  marginStatus,
  quantizeToBin,
  spokeLengths,
  toSignedLabel,
} from './boostingPointIntel2dMath';

export interface BoostingPointIntelOverlays2DProps {
  boosting2d: Boosting2DPayload | null;
  boostFrame: BoostFrame2DPayload | null;
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  hexbinCells: HexbinCell2D[];
  isClassification: boolean;
  /** Positive class for the signed margin map; null keeps rings dormant. */
  positiveLabel: string | number | null;
  classColor: (label: string | number | null) => string;
  /** Staged TreeSHAP pushes φ_m for the active point — future endpoint. */
  activeStagedPush?: Array<{ phiX: number; phiY: number }> | null;
  /** Per-point raw scores F(x) — future endpoint for margin rings. */
  margins?: number[] | null;
  /** Per-point pseudo-residuals r (classifier path) — future endpoint. */
  pseudoResiduals?: number[] | null;
  /** Per-hexbin gradient aggregates — future endpoint for hot tiers. */
  cellStats?: Array<{ sumG: number; sumH: number; count: number } | null> | null;
}

const BOOST_FAMILIES = new Set(['sklearn_gbm', 'hist_gbm', 'xgboost', 'lightgbm', 'catboost']);

const PUSH_TONE: Record<string, string> = {
  xgboost: '#f43d5e',
  lightgbm: '#f59e0b',
  catboost: '#8b5cf6',
  sklearn_gbm: '#34d399',
  hist_gbm: '#38bdf8',
};

function isErrorPoint(p: TrajectoryPoint3D): boolean {
  const r = p.residual_error;
  if (r === null || r === undefined || !Number.isFinite(r)) return true;
  return Math.abs(r) > 0.5;
}

/**
 * Phase 54 — orchestrator for boosting point-intelligence overlays.
 *
 * Staged SHAP pushes, margin rings and gradient stats ride optional
 * per-point arrays and stay honestly dormant until the backend ships
 * them (same precedent as the Phase-51/53 dormant layers); regressor
 * pseudo-residual spokes derive live from the signed `residual_error`
 * (r = y − ŷ exactly), and hist_gbm hexbins snap to the 256-bin lattice
 * with traffic tiers. Renders nothing for non-boosting models.
 */
export function BoostingPointIntelOverlays2D({
  boosting2d,
  boostFrame,
  points,
  activeFrameIndex,
  hexbinCells,
  isClassification,
  positiveLabel,
  classColor,
  activeStagedPush = null,
  margins = null,
  pseudoResiduals = null,
  cellStats = null,
}: BoostingPointIntelOverlays2DProps) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const showHexbins = useStage3DStore((s) => s.showHexbins);

  if (!boosting2d || !BOOST_FAMILIES.has(boosting2d.family) || points.length === 0) {
    return null;
  }
  void boostFrame;
  const active = Math.max(0, Math.min(activeFrameIndex, points.length - 1));
  const keep = (f: number): boolean => !showErrorsOnly || isErrorPoint(points[f]!) || f === active;
  const tone = PUSH_TONE[boosting2d.family] ?? PUSH_TRAIL_COLOR;
  const activePt = points[active]!;

  // Mechanism A: staged push walk from the active point (DoD #1).
  const showTrail = showCometTrails && activeStagedPush !== null && activeStagedPush.length > 0;

  // Mechanism B: functional-margin rings (DoD #2).
  const rings: MarginRingDatum[] = [];
  if (showDualRings && isClassification && positiveLabel !== null && margins && margins.length === points.length) {
    points.forEach((p, f) => {
      if (!keep(f)) return;
      const fx = margins[f];
      if (fx === undefined || !Number.isFinite(fx)) return;
      const ySigned = toSignedLabel(p.raw_z_actual, positiveLabel);
      const m = functionalMargin(ySigned, fx);
      const status = marginStatus(m);
      if (status === 'unknown') return;
      rings.push({
        x: p.world_x,
        y: p.world_y,
        margin: m,
        status,
        color: status === 'safe' ? classColor(p.raw_z_pred) : status === 'critical' ? MARGIN_CRITICAL_COLOR : MARGIN_NEGATIVE_COLOR,
        label: `Frame ${f}`,
      });
    });
  }

  // CatBoost oblivious symmetry marker (spec §2.2): mini grid on the cell.
  let symmetryCell: { cellId: number; bitmask: string } | null = null;
  if (
    showDualRings &&
    boosting2d.family === 'catboost' &&
    boosting2d.obliviousCells.length > 0
  ) {
    const hit = findBoostLeafAt(boosting2d.obliviousCells, activePt.world_x, activePt.world_y);
    if (hit) symmetryCell = { cellId: hit.cellId, bitmask: hit.bitmask };
  }

  // Mechanism C: pseudo-residual spokes (DoD #3).
  const spokes: PseudoResidualSpoke[] = [];
  if (showResidualSpokes) {
    const residuals: number[] = !isClassification
      ? points.map((p) => (p.residual_error === null || p.residual_error === undefined ? NaN : p.residual_error))
      : pseudoResiduals && pseudoResiduals.length === points.length
        ? pseudoResiduals.map((r) => (r === null || r === undefined ? NaN : r))
        : [];
    if (residuals.length === points.length) {
      const { lengths, sides } = spokeLengths(residuals);
      points.forEach((p, f) => {
        if (!keep(f)) return;
        const len = lengths[f];
        const side = sides[f];
        if (len === undefined || side === undefined || !(len > 1e-9) || side === 0) return;
        const raw = residuals[f]!;
        spokes.push({
          x: p.world_x,
          y: p.world_y,
          length: len,
          direction: side > 0 ? 1 : -1,
          magnitude: Number.isFinite(raw) ? Math.abs(raw) : undefined,
        });
      });
    }
  }

  // Mechanism D: quantized lattice hexbins, gradient-hot when stats ship (DoD #4).
  let heat: Array<number | null> | null = null;
  let heatMode: 'gradient' | 'traffic' = 'traffic';
  if (showHexbins && hexbinCells.length > 0 && boosting2d.family === 'hist_gbm') {
    const xs = [...boosting2d.binLinesXWorld].sort((a, b) => a - b);
    const ys = [...boosting2d.binLinesYWorld].sort((a, b) => a - b);
    if (xs.length > 0 && ys.length > 0) {
      const useStats = cellStats !== null && cellStats.length >= hexbinCells.length;
      heatMode = useStats ? 'gradient' : 'traffic';
      heat = hexbinCells.map((c, i) => {
        if (quantizeToBin(c.centerX, xs) < 0 || quantizeToBin(c.centerY, ys) < 0) return null;
        if (useStats) {
          const st = cellStats![i];
          if (!st || !Number.isFinite(st.sumG)) return null;
          return Math.abs(st.sumG);
        }
        return c.count > 0 ? c.count : null;
      });
      if (heat.every((h) => h === null)) heat = null;
    }
  }

  return (
    <>
      <g aria-label="Boosting point intelligence overlays">
        {showTrail && (
          <BoostingShapPushTrail2D
            origin={{ x: activePt.world_x, y: activePt.world_y }}
            stages={activeStagedPush ?? []}
            uptoStage={null}
            tone={tone}
            visible
          />
        )}
        <FunctionalMarginRing2D rings={rings} visible={showDualRings} />
        {symmetryCell && (
          <g
            className="oblivious-symmetry-2d pointer-events-none"
            aria-label="CatBoost oblivious symmetry marker"
            opacity={0.9}
          >
            {[-0.12, 0, 0.12].map((d) => (
              <g key={d}>
                <line
                  x1={activePt.world_x - 0.18}
                  y1={wy(activePt.world_y + d)}
                  x2={activePt.world_x + 0.18}
                  y2={wy(activePt.world_y + d)}
                  stroke={tone}
                  strokeWidth={0.02}
                />
                <line
                  x1={activePt.world_x + d}
                  y1={wy(activePt.world_y - 0.18)}
                  x2={activePt.world_x + d}
                  y2={wy(activePt.world_y + 0.18)}
                  stroke={tone}
                  strokeWidth={0.02}
                />
              </g>
            ))}
            <title>{`Oblivious cell ${symmetryCell.cellId} · bits ${symmetryCell.bitmask}`}</title>
          </g>
        )}
        <PseudoResidualSpoke2D spokes={spokes} visible={showResidualSpokes} />
        {heat && <QuantizedBinHexbin2D cells={hexbinCells} heat={heat} mode={heatMode} visible={showHexbins} />}
      </g>
      {(showTrail || rings.length > 0 || spokes.length > 0) && (
        <div
          className="font-mono-tabular pointer-events-none absolute right-1 bottom-14 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.72)', color: '#fda4af' }}
          title="Boosting point intelligence: SHAP push trail · margin rings · pseudo-residual spokes"
        >
          {`${showTrail ? `⚡ T=${(activeStagedPush ?? []).length}` : ''}${rings.length > 0 ? ` · ◈ ${rings.length} margins` : ''}${spokes.length > 0 ? ` · ↕ ${spokes.length} grads` : ''}`}
        </div>
      )}
    </>
  );
}
