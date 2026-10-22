import { wy } from '../DecisionBoundaryContour2D';
import { binCellRect } from './boosting2dMath';

export interface HistGbmQuantizedBins2DProps {
  /** World x positions of the raw bin-threshold grid lines. */
  linesXWorld: number[];
  /** World y positions of the raw bin-threshold grid lines. */
  linesYWorld: number[];
  /** Active bin [binX, binY] of the query row; null hides the highlight. */
  activeBin: [number, number] | null;
  activeRowIndex: number | null;
  showBins?: boolean;
  showActiveBin?: boolean;
  visible?: boolean;
}

/**
 * Phase 46 — HistGradientBoosting 255-bin quantized grid.
 *
 * Thin slate lines at the exact `_bin_mapper.bin_thresholds_` positions cut
 * the dense decision boundary, showing how the model sees features as
 * digital step blocks. The active (bin_x, bin_y) cell glows so the pinned
 * row's quantized address reads at a glance. One path per axis keeps the
 * 255-line grid at 2 draw calls.
 */
export function HistGbmQuantizedBins2D({
  linesXWorld,
  linesYWorld,
  activeBin,
  activeRowIndex,
  showBins = true,
  showActiveBin = true,
  visible = true,
}: HistGbmQuantizedBins2DProps) {
  if (!visible) return null;

  const vPath =
    showBins && linesXWorld.length > 0
      ? linesXWorld.map((x) => `M${x.toFixed(3)} ${wy(-5).toFixed(3)}L${x.toFixed(3)} ${wy(5).toFixed(3)}`).join('')
      : '';
  const hPath =
    showBins && linesYWorld.length > 0
      ? linesYWorld.map((y) => `M${(-5).toFixed(3)} ${wy(y).toFixed(3)}L${(5).toFixed(3)} ${wy(y).toFixed(3)}`).join('')
      : '';
  const cell =
    showActiveBin && activeBin ? binCellRect(linesXWorld, linesYWorld, activeBin[0], activeBin[1]) : null;

  return (
    <g className="hist-gbm-bins-2d pointer-events-none" aria-label="HistGBM quantized bin grid">
      {vPath && (
        <path d={vPath} fill="none" stroke="#1e293b" strokeWidth={0.02} opacity={0.9} strokeLinecap="round">
          <title>{`${linesXWorld.length} bin edges on X (0..255 quantized)`}</title>
        </path>
      )}
      {hPath && (
        <path d={hPath} fill="none" stroke="#1e293b" strokeWidth={0.02} opacity={0.9} strokeLinecap="round">
          <title>{`${linesYWorld.length} bin edges on Y (0..255 quantized)`}</title>
        </path>
      )}
      {cell && activeBin && (
        <rect
          x={cell.x}
          y={cell.y}
          width={cell.width}
          height={cell.height}
          fill="#38bdf8"
          fillOpacity={0.1}
          stroke="#38bdf8"
          strokeWidth={0.05}
        >
          <title>
            {`ACTIVE BIN (x=${activeBin[0]}, y=${activeBin[1]})${activeRowIndex !== null ? ` · Row #${activeRowIndex}` : ''}`}
          </title>
        </rect>
      )}
    </g>
  );
}
