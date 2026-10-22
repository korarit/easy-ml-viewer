import { wy } from '../DecisionBoundaryContour2D';
import { cellToRect } from './boosting2dMath';
import type { ObliviousCell2D } from '../../../projection3d/types/projection.types';

export interface CatBoostObliviousGrid2DProps {
  /** World x positions of the symmetric oblivious cuts. */
  linesXWorld: number[];
  /** World y positions of the symmetric oblivious cuts. */
  linesYWorld: number[];
  cells: ObliviousCell2D[];
  obliviousRule: string;
  activeRowIndex: number | null;
  showGrid?: boolean;
  showActiveCell?: boolean;
  visible?: boolean;
}

function cellTitle(cell: ObliviousCell2D): string {
  return `leaf ${cell.leafIndex} · bitmask ${cell.bitmask}`;
}

/**
 * Phase 46 — CatBoost symmetric oblivious checkerboard.
 *
 * Oblivious trees cut every node at a depth on the same (feature, threshold),
 * so the 2D slice is a perfect axis-parallel checkerboard: (Kx+1)×(Ky+1)
 * cells, each carrying its bitwise leaf index. Grid lines batch into 2 SVG
 * paths; the active cell glows with its `[X > a] AND [Y <= b]` bit mask.
 */
export function CatBoostObliviousGrid2D({
  linesXWorld,
  linesYWorld,
  cells,
  obliviousRule,
  activeRowIndex,
  showGrid = true,
  showActiveCell = true,
  visible = true,
}: CatBoostObliviousGrid2DProps) {
  if (!visible) return null;

  const vPath =
    showGrid && linesXWorld.length > 0
      ? linesXWorld.map((x) => `M${x.toFixed(3)} ${wy(-5).toFixed(3)}L${x.toFixed(3)} ${wy(5).toFixed(3)}`).join('')
      : '';
  const hPath =
    showGrid && linesYWorld.length > 0
      ? linesYWorld.map((y) => `M${(-5).toFixed(3)} ${wy(y).toFixed(3)}L${(5).toFixed(3)} ${wy(y).toFixed(3)}`).join('')
      : '';
  const active = showActiveCell ? (cells.find((c) => c.isActive) ?? null) : null;

  return (
    <g className="catboost-oblivious-grid-2d pointer-events-none" aria-label="CatBoost oblivious checkerboard">
      {vPath && (
        <path d={vPath} fill="none" stroke="#8b5cf6" strokeWidth={0.035} opacity={0.75} strokeLinecap="round">
          <title>{`${linesXWorld.length} symmetric X cuts · ${obliviousRule}`}</title>
        </path>
      )}
      {hPath && (
        <path d={hPath} fill="none" stroke="#8b5cf6" strokeWidth={0.035} opacity={0.75} strokeLinecap="round">
          <title>{`${linesYWorld.length} symmetric Y cuts · ${obliviousRule}`}</title>
        </path>
      )}
      <g aria-label="Checkerboard cells">
        {cells.map((c) => {
          const r = cellToRect(c);
          return (
            <rect
              key={`oblivious-cell-${c.cellId}`}
              x={r.x}
              y={r.y}
              width={r.width}
              height={r.height}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth={0.02}
              opacity={0.5}
            >
              <title>{cellTitle(c)}</title>
            </rect>
          );
        })}
      </g>
      {active &&
        (() => {
          const r = cellToRect(active);
          return (
            <g aria-label="Active oblivious cell">
              <rect
                x={r.x}
                y={r.y}
                width={r.width}
                height={r.height}
                fill="#8b5cf6"
                fillOpacity={0.12}
                stroke="#a78bfa"
                strokeWidth={0.07}
                strokeLinecap="round"
              >
                <title>
                  {`ACTIVE ${cellTitle(active)}${activeRowIndex !== null ? ` · Row #${activeRowIndex}` : ''} · ${obliviousRule}`}
                </title>
              </rect>
            </g>
          );
        })()}
    </g>
  );
}
