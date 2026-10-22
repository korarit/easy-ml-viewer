import { wy } from '../DecisionBoundaryContour2D';

export interface KnnNeighbor2D {
  refIndex: number;
  x: number;
  y: number;
  distance2D: number;
  voteWeight: number;
  targetClass: number | string | null;
  rank: number;
}

export interface KnnRadar2DProps {
  centerX: number;
  centerY: number;
  /** Ellipse radii (world units) — raw circles skew under anisotropic spans. */
  radiusX: number;
  radiusY: number;
  neighbors: KnnNeighbor2D[];
  visible?: boolean;
}

const NEIGHBOR_R = 0.12;

/**
 * Phase 43 — 2D expanding radar circle + spider laser lines for KNN.
 *
 * World coords (SVG Y flipped via `wy`). Stroke widths follow the blueprint
 * vote-share rule (scaled to world units); rank labels stagger alternately
 * so 2D-overlapping neighbors stay distinguishable (E43.4).
 */
export function KnnRadar2D({
  centerX,
  centerY,
  radiusX,
  radiusY,
  neighbors,
  visible = true,
}: KnnRadar2DProps) {
  if (!visible) return null;
  const maxW = Math.max(1e-12, ...neighbors.map((n) => n.voteWeight));

  return (
    <g className="knn-radar-2d pointer-events-none" aria-label="KNN radar">
      {/* 1. วงกลมเรดาร์ 2 มิติ */}
      <ellipse
        cx={centerX}
        cy={wy(centerY)}
        rx={Math.max(radiusX, 1e-6)}
        ry={Math.max(radiusY, 1e-6)}
        fill="rgba(56, 189, 248, 0.12)"
        stroke="var(--ml-primary)"
        strokeWidth={0.05}
        strokeDasharray="0.18 0.1"
      >
        <title>{`2D radar circle (R = ${Math.max(radiusX, radiusY).toPrecision(4)} world units)`}</title>
      </ellipse>

      {/* 2. เส้นใยแมงมุมเชื่อมเพื่อนบ้านทั้ง K ตัว */}
      {neighbors.map((nb) => (
        <g key={`${nb.refIndex}-${nb.rank}`}>
          <line
            x1={centerX}
            y1={wy(centerY)}
            x2={nb.x}
            y2={wy(nb.y)}
            stroke="var(--ml-primary)"
            strokeWidth={0.03 + 0.09 * (nb.voteWeight / maxW)}
            strokeOpacity={0.85}
            strokeLinecap="round"
          >
            <title>
              {`Ref #${nb.refIndex} (rank ${nb.rank}, 2D dist ${nb.distance2D.toPrecision(4)}, vote ${(nb.voteWeight * 100).toFixed(1)}%)`}
            </title>
          </line>
          <circle
            cx={nb.x}
            cy={wy(nb.y)}
            r={NEIGHBOR_R}
            fill="none"
            stroke="#ffffff"
            strokeWidth={0.045}
          />
          <text
            x={nb.x + (nb.rank % 2 === 0 ? 0.2 : -0.2)}
            y={wy(nb.y) - 0.18 - (nb.rank % 3) * 0.12}
            fill="#ffffff"
            fontSize={0.28}
            textAnchor={nb.rank % 2 === 0 ? 'start' : 'end'}
          >
            #{nb.rank}
          </text>
        </g>
      ))}
    </g>
  );
}
