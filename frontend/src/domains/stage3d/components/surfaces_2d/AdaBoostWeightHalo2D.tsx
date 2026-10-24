import { wy } from '../DecisionBoundaryContour2D';
import { ADABOOST_HALO_COLOR, adaboostHaloRadius } from './treesPointIntel2dMath';

export interface AdaBoostHaloDatum {
  x: number;
  y: number;
  /** Ring scale t_base·√(w/w̄) for this sample (DoD #2). */
  scale: number;
  color?: string;
  label: string;
}

export interface AdaBoostWeightHalo2DProps {
  halos: AdaBoostHaloDatum[];
  visible?: boolean;
  /** Cap rendered halos so dense queues stay legible (scales stay exact). */
  haloCap?: number;
}

/**
 * Phase 53 (mechanism B, DoD #2) — AdaBoost sample-weight halo.
 *
 * The outer ring grows with √(ω/ω̄): points AdaBoost reweighted heavily
 * (repeatedly mispredicted) stand out with a strikingly larger halo.
 * One batched path (≤ 2 draw calls with the weight readout).
 */
export function AdaBoostWeightHalo2D({ halos, visible = true, haloCap = 500 }: AdaBoostWeightHalo2DProps) {
  if (!visible || halos.length === 0) return null;
  const shown = halos.slice(0, Math.max(1, haloCap));
  const d = shown
    .map((h) => {
      const r = Math.min(adaboostHaloRadius(h.scale), 0.6);
      return `M${(h.x - r).toFixed(3)} ${wy(h.y).toFixed(3)}a${r.toFixed(3)} ${r.toFixed(3)} 0 1 0 ${(2 * r).toFixed(3)} 0a${r.toFixed(3)} ${r.toFixed(3)} 0 1 0 ${(-2 * r).toFixed(3)} 0`;
    })
    .join('');
  const peak = shown.reduce((m, h) => Math.max(m, Number.isFinite(h.scale) ? h.scale : 0), 0);
  return (
    <g className="adaboost-halo-2d pointer-events-none" aria-label="AdaBoost sample-weight halos">
      <path d={d} fill="none" stroke={ADABOOST_HALO_COLOR} strokeWidth={0.035} opacity={0.9}>
        <title>{`AdaBoost weight halos (${shown.length} pts · peak scale ×${peak.toFixed(2)})`}</title>
      </path>
    </g>
  );
}
