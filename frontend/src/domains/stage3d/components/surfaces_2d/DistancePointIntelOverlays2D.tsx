import { useEffect, useRef } from 'react';
import { useStage3DStore, type TrajectoryPoint3D } from '../../store/useStage3DStore';
import type {
  HexbinCell2D,
  KnnFrame2DPayload,
  Svm2DPayload,
} from '../../../projection3d/types/projection.types';
import { wy } from '../DecisionBoundaryContour2D';
import { SvmSupportVectorHalo2D } from './SvmSupportVectorHalo2D';
import { SvrSlackSpoke2D, type SvrSlackSpokeDatum } from './SvrSlackSpoke2D';
import { KnnVoteArcRing2D } from './KnnVoteArcRing2D';
import { KnnDensityHexbin2D } from './KnnDensityHexbin2D';
import {
  detectNeighborSwaps,
  discordantRanks,
  knnCentroid2D,
  svrSlackSpoke,
  voteArcSegments,
} from './distancePointIntel2dMath';

export interface DistancePointIntelOverlays2DProps {
  svm2d: Svm2DPayload | null;
  knnFrame: KnnFrame2DPayload | null;
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  hexbinCells: HexbinCell2D[];
  /** Parallel ρ values; null when the backend ships no per-bin R̄_K. */
  hexbinDensities?: Array<number | null> | null;
  /** SVR ε; null hides slack spokes (the 2D payload ships no ε yet). */
  svrEpsilon?: number | null;
  classColor: (label: string | number | null) => string;
}

function isErrorPoint(p: TrajectoryPoint3D): boolean {
  const r = p.residual_error;
  if (r === null || r === undefined || !Number.isFinite(r)) return true;
  return Math.abs(r) > 0.5;
}

/**
 * Phase 51 — orchestrator for distance & kernel point-intelligence overlays.
 *
 * Derives everything from the Phase-43 `svm2d` / `knnFrame` payloads (no
 * extra fetch): α-accurate dual-ring halos, vote arcs, swap ripples +
 * elastic centroid tethers, discordance spokes, SVR slack spokes and density
 * tiers. Renders nothing for non-distance models (both payloads null).
 */
export function DistancePointIntelOverlays2D({
  svm2d,
  knnFrame,
  points,
  activeFrameIndex,
  hexbinCells,
  hexbinDensities = null,
  svrEpsilon = null,
  classColor,
}: DistancePointIntelOverlays2DProps) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const showHexbins = useStage3DStore((s) => s.showHexbins);

  // Mechanism A: swap detection needs the previous frame's neighbor ids.
  const prevIdsRef = useRef<number[]>([]);
  const curIds = knnFrame && !knnFrame.outlier ? knnFrame.neighbors.map((n) => n.trainIndex) : [];
  const swaps = detectNeighborSwaps(prevIdsRef.current, curIds);
  useEffect(() => {
    prevIdsRef.current = curIds;
  }, [knnFrame]);

  const hasSvmHalos = !!svm2d && !svm2d.isLinearNoSv && svm2d.halos.length > 0;
  const hasKnnFrame = !!knnFrame && !knnFrame.outlier && knnFrame.neighbors.length > 0;

  // Mechanism A: elastic-web anchor = vote-weighted centroid of K neighbors.
  const centroid =
    showCometTrails && hasKnnFrame
      ? knnCentroid2D(
          knnFrame!.neighbors.map((n) => ({ x: n.xWorld, y: n.yWorld, voteWeight: n.voteWeight })),
        )
      : null;
  const enteredById = new Map(
    (knnFrame?.neighbors ?? []).map((n) => [n.trainIndex, n] as const),
  );
  const ripples = showCometTrails
    ? swaps.entered
        .map((id) => enteredById.get(id))
        .filter((n) => n !== undefined)
    : [];

  // Mechanism B (KNN): vote-breakdown arcs around the active query (DoD #3).
  const segments =
    showDualRings && hasKnnFrame && Object.keys(knnFrame!.votes).length > 0
      ? voteArcSegments(knnFrame!.votes, (label) => classColor(label))
      : [];

  // Mechanism C (KNN): discordance spokes from a mispredicted active point
  // to the neighbors that voted against its true class.
  const active = points.length > 0 ? points[Math.max(0, Math.min(activeFrameIndex, points.length - 1))] : null;
  let discordance: Array<{ x1: number; y1: number; x2: number; y2: number }> = [];
  if (showResidualSpokes && hasKnnFrame && active && isErrorPoint(active)) {
    const ranks = new Set(
      discordantRanks(
        knnFrame!.neighbors.map((n) => ({ rank: n.rank, label: n.label })),
        active.raw_z_actual,
      ),
    );
    discordance = knnFrame!.neighbors
      .filter((n) => ranks.has(n.rank))
      .map((n) => ({ x1: knnFrame!.queryWorld[0], y1: knnFrame!.queryWorld[1], x2: n.xWorld, y2: n.yWorld }));
  }

  // Mechanism C (SVR): slack spokes only outside the ε-tube (DoD #2).
  const slackSpokes: SvrSlackSpokeDatum[] = [];
  if (showResidualSpokes && svrEpsilon !== null && svm2d?.isRegressor && points.length > 0) {
    const per = points.map((p, f) => {
      if (showErrorsOnly && !isErrorPoint(p) && f !== activeFrameIndex) return null;
      const s = svrSlackSpoke(p.raw_z_actual, p.raw_z_pred, svrEpsilon);
      return s.outside ? { p, s } : null;
    });
    const live = per.filter((e) => e !== null);
    const peak = live.reduce((m, e) => Math.max(m, e!.s.xi + e!.s.xiStar), 0);
    if (peak > 1e-12) {
      live.forEach((e) => {
        const mag = e!.s.xi + e!.s.xiStar;
        slackSpokes.push({
          x: e!.p.world_x,
          y: e!.p.world_y,
          length: (mag / peak) * 1.5,
          side: e!.s.xiStar > e!.s.xi ? 1 : -1,
          slack: mag,
        });
      });
    }
  }

  const showDensity = showHexbins && hexbinCells.length > 0 && !!hexbinDensities && hexbinDensities.length > 0;

  if (!hasSvmHalos && !hasKnnFrame && slackSpokes.length === 0 && !showDensity) return null;

  return (
    <>
      <g aria-label="Distance and kernel point intelligence overlays">
        {showDualRings && hasSvmHalos && <SvmSupportVectorHalo2D halos={svm2d!.halos} visible />}
        {showDualRings && hasKnnFrame && segments.length > 0 && (
          <KnnVoteArcRing2D x={knnFrame!.queryWorld[0]} y={knnFrame!.queryWorld[1]} segments={segments} visible />
        )}
        {showCometTrails && hasKnnFrame && centroid && (
          <line
            x1={knnFrame!.queryWorld[0]}
            y1={wy(knnFrame!.queryWorld[1])}
            x2={centroid.x}
            y2={wy(centroid.y)}
            stroke="#22d3ee"
            strokeWidth={0.03}
            strokeDasharray="0.14 0.1"
            opacity={0.8}
          >
            <title>Elastic web: active point tethered to the K-neighbor centroid</title>
          </line>
        )}
        {ripples.length > 0 &&
          ripples.map((n) => (
            <circle
              key={`ripple-${n!.trainIndex}`}
              cx={n!.xWorld}
              cy={wy(n!.yWorld)}
              r={0.3}
              fill="none"
              stroke="#22d3ee"
              strokeWidth={0.045}
              opacity={0.9}
            >
              <animate attributeName="r" values="0.14;0.42" dur="0.9s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.9;0.15" dur="0.9s" repeatCount="indefinite" />
              <title>{`Neighbor swap: ref #${n!.trainIndex} just entered the K-set`}</title>
            </circle>
          ))}
        {discordance.length > 0 && (
          <path
            d={discordance
              .map((s) => `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`)
              .join('')}
            fill="none"
            stroke="#ef4444"
            strokeWidth={0.035}
            strokeDasharray="0.18 0.1"
            opacity={0.9}
          >
            <title>KNN conflict: mispredicted point vs majority-vote neighbors</title>
          </path>
        )}
        <SvrSlackSpoke2D spokes={slackSpokes} visible={showResidualSpokes} />
        {showDensity && (
          <KnnDensityHexbin2D cells={hexbinCells} densities={hexbinDensities!} visible={showHexbins} />
        )}
      </g>
      {(ripples.length > 0 || slackSpokes.length > 0) && (
        <div
          className="font-mono-tabular pointer-events-none absolute right-1 bottom-8 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.72)', color: '#67e8f9' }}
          title="Phase-51 live point-intelligence readouts"
        >
          {ripples.length > 0 && `⇄ ${ripples.length} neighbor swap${ripples.length > 1 ? 's' : ''}`}
          {ripples.length > 0 && slackSpokes.length > 0 && ' · '}
          {slackSpokes.length > 0 && `ξ ${slackSpokes.length} outside ε-tube`}
        </div>
      )}
    </>
  );
}
