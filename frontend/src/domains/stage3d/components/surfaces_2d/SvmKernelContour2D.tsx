import type { ContourLevel2D } from '../../../projection3d/types/projection.types';

export interface SvmKernelContour2DProps {
  /** Contour levels with kind === 'margin' (f = ±1 parallels). */
  marginLevels: ContourLevel2D[];
  visible?: boolean;
}

/**
 * Phase 43 — SVM margin parallels (f = +1 / −1) from the score-field
 * Marching-Squares pass.
 *
 * Curved RBF margins are arbitrary polylines, so — unlike the straight
 * linear corridor — no band fill is drawn (filling between two sampled
 * curves would invent geometry). The soft-margin zone reads off the
 * heatmap gradient between the dashed parallels instead.
 */
export function SvmKernelContour2D({ marginLevels, visible = true }: SvmKernelContour2DProps) {
  if (!visible || marginLevels.length === 0) return null;
  return (
    <g className="svm-kernel-contour-2d pointer-events-none" aria-label="SVM margin parallels">
      {marginLevels.map((lv, idx) => {
        if (lv.lineSegments.length === 0) return null;
        const sign = lv.threshold >= 0 ? '+' : '−';
        const d = lv.lineSegments
          .map(([x1, y1, x2, y2]) => `M${x1.toFixed(3)} ${(-y1).toFixed(3)}L${x2.toFixed(3)} ${(-y2).toFixed(3)}`)
          .join('');
        return (
          <path
            key={`margin-${lv.threshold}-${idx}`}
            d={d}
            fill="none"
            stroke="#7dd3fc"
            strokeWidth={0.045}
            strokeDasharray="0.3 0.14"
            opacity={0.9}
            strokeLinecap="round"
          >
            <title>{`Margin boundary (f = ${sign}${Math.abs(lv.threshold)})`}</title>
          </path>
        );
      })}
    </g>
  );
}
