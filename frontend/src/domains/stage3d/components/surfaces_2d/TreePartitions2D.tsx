import { wy } from '../DecisionBoundaryContour2D';
import type { LeafBox2D, TreeCutSegment2D } from '../../../projection3d/types/projection.types';
import { batchCutPaths, boxToRect, findLeafAt } from './trees2dMath';

export interface TreePartitions2DProps {
  segments: TreeCutSegment2D[];
  leaves: LeafBox2D[];
  /** Active trajectory point in world coords; null hides the glow. */
  activePoint: { x: number; y: number } | null;
  activeRowIndex: number | null;
  showCuts?: boolean;
  showActiveLeaf?: boolean;
  visible?: boolean;
}

function leafTitle(leaf: LeafBox2D): string {
  const pred = leaf.prediction === null || leaf.prediction === undefined ? '?' : String(leaf.prediction);
  const rule = leaf.rule ? ` · ${leaf.rule}` : '';
  return `Leaf #${leaf.leafId} · pred ${pred} · n=${leaf.samples} · impurity=${leaf.impurity}${rule}`;
}

/**
 * Phase 45 — single-tree orthogonal partitions + active leaf glow.
 *
 * Cut segments batch by axis into 2 SVG paths (constant draw calls no
 * matter the depth); each leaf box is one rect carrying its rule tooltip.
 * The active leaf is resolved client-side (point-in-box over world boxes)
 * so the glow tracks scrubbing with zero round-trips at 60 FPS.
 */
export function TreePartitions2D({
  segments,
  leaves,
  activePoint,
  activeRowIndex,
  showCuts = true,
  showActiveLeaf = true,
  visible = true,
}: TreePartitions2DProps) {
  if (!visible) return null;
  const paths = showCuts ? batchCutPaths(segments) : { vertical: '', horizontal: '' };
  const active =
    showActiveLeaf && activePoint ? findLeafAt(leaves, activePoint.x, activePoint.y) : null;
  const cutStyle = {
    fill: 'none' as const,
    stroke: '#94a3b8',
    strokeWidth: 0.035,
    strokeDasharray: '0.22 0.14',
    opacity: 0.8,
    strokeLinecap: 'round' as const,
  };

  return (
    <g className="tree-partitions-2d pointer-events-none" aria-label="Decision tree partitions">
      {paths.vertical && (
        <path d={paths.vertical} {...cutStyle}>
          <title>{`${segments.filter((s) => s.axis === 'x').length} vertical cuts (x = θ)`}</title>
        </path>
      )}
      {paths.horizontal && (
        <path d={paths.horizontal} {...cutStyle}>
          <title>{`${segments.filter((s) => s.axis === 'y').length} horizontal cuts (y = θ)`}</title>
        </path>
      )}
      <g aria-label="Leaf boxes">
        {leaves.map((l) => {
          const r = boxToRect(l.boxWorld);
          return (
            <rect
              key={`leaf-${l.treeIndex}-${l.leafId}`}
              x={r.x}
              y={r.y}
              width={r.width}
              height={r.height}
              fill="none"
              stroke="#475569"
              strokeWidth={0.025}
              opacity={0.55}
            >
              <title>{leafTitle(l)}</title>
            </rect>
          );
        })}
      </g>
      {active &&
        (() => {
          const r = boxToRect(active.boxWorld);
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
