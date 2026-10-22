import { wy } from '../DecisionBoundaryContour2D';
import { batchCutPaths } from './trees2dMath';
import { boostBoxToRect, findBoostLeafAt, findStrongestLeaf } from './boosting2dMath';
import type { BoostCutSegment2D, BoostLeafBox2D } from '../../../projection3d/types/projection.types';

export interface LightgbmLeafWise2DProps {
  segments: BoostCutSegment2D[];
  leaves: BoostLeafBox2D[];
  /** Active trajectory point in world coords; null hides the glow. */
  activePoint: { x: number; y: number } | null;
  activeRowIndex: number | null;
  /** Tree index label for the tooltip (always tree-0 polygons). */
  treeIndex?: number;
  showPolygons?: boolean;
  showActiveLeaf?: boolean;
  /** Highlight the best-first leaf (max |leaf value|). */
  showBestFirst?: boolean;
  visible?: boolean;
}

function leafTitle(leaf: BoostLeafBox2D): string {
  const rule = leaf.rule ? ` · ${leaf.rule}` : '';
  return `Leaf #${leaf.leafId} · value ${leaf.leafValue}${rule}`;
}

/**
 * Phase 46 — LightGBM (and XGBoost) asymmetric leaf-wise polygons.
 *
 * Leaf-wise growth splits the winning leaf first, so tree-0 polygons are
 * visibly lopsided compared with level-wise trees. Cuts batch by axis into
 * 2 SVG paths; each first-tree leaf box carries its raw leaf value; the
 * strongest leaf (best-first) gets an amber ring and the scrub-active leaf
 * glows cyan — both resolved client-side at 60 FPS.
 */
export function LightgbmLeafWise2D({
  segments,
  leaves,
  activePoint,
  activeRowIndex,
  treeIndex = 0,
  showPolygons = true,
  showActiveLeaf = true,
  showBestFirst = true,
  visible = true,
}: LightgbmLeafWise2DProps) {
  if (!visible) return null;

  const paths = showPolygons ? batchCutPaths(segments) : { vertical: '', horizontal: '' };
  const active =
    showActiveLeaf && activePoint ? findBoostLeafAt(leaves, activePoint.x, activePoint.y) : null;
  const strongest = showBestFirst ? findStrongestLeaf(leaves) : null;

  return (
    <g className="lightgbm-leafwise-2d pointer-events-none" aria-label="Leaf-wise polygons">
      {paths.vertical && (
        <path d={paths.vertical} fill="none" stroke="#f59e0b" strokeWidth={0.035} strokeDasharray="0.22 0.14" opacity={0.85} strokeLinecap="round">
          <title>{`tree-${treeIndex}: ${segments.filter((s) => s.axis === 'x').length} vertical leaf cuts`}</title>
        </path>
      )}
      {paths.horizontal && (
        <path d={paths.horizontal} fill="none" stroke="#f59e0b" strokeWidth={0.035} strokeDasharray="0.22 0.14" opacity={0.85} strokeLinecap="round">
          <title>{`tree-${treeIndex}: ${segments.filter((s) => s.axis === 'y').length} horizontal leaf cuts`}</title>
        </path>
      )}
      <g aria-label="Leaf boxes">
        {leaves.map((l) => {
          const r = boostBoxToRect(l.boxWorld);
          return (
            <rect
              key={`boost-leaf-${l.treeIndex}-${l.leafId}`}
              x={r.x}
              y={r.y}
              width={r.width}
              height={r.height}
              fill="none"
              stroke="#b45309"
              strokeWidth={0.025}
              opacity={0.55}
            >
              <title>{leafTitle(l)}</title>
            </rect>
          );
        })}
      </g>
      {strongest &&
        strongest !== active &&
        (() => {
          const r = boostBoxToRect(strongest.boxWorld);
          return (
            <rect
              x={r.x}
              y={r.y}
              width={r.width}
              height={r.height}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={0.06}
              strokeDasharray="0.3 0.12"
              opacity={0.9}
              aria-label="Best-first leaf"
            >
              <title>{`BEST-FIRST ${leafTitle(strongest)}`}</title>
            </rect>
          );
        })()}
      {active &&
        (() => {
          const r = boostBoxToRect(active.boxWorld);
          return (
            <g aria-label="Active leaf box">
              <rect
                x={r.x}
                y={r.y}
                width={r.width}
                height={r.height}
                fill="#22d3ee"
                fillOpacity={0.08}
                stroke="#22d3ee"
                strokeWidth={0.07}
                strokeLinecap="round"
              >
                <title>
                  {`ACTIVE LEAF #${active.leafId}${activeRowIndex !== null ? ` · Row #${activeRowIndex}` : ''} · ${leafTitle(active)}`}
                </title>
              </rect>
              <circle cx={activePoint!.x} cy={wy(activePoint!.y)} r={0.09} fill="#22d3ee">
                <title>{`Active query point${activeRowIndex !== null ? ` (Row #${activeRowIndex})` : ''}`}</title>
              </circle>
            </g>
          );
        })()}
    </g>
  );
}
