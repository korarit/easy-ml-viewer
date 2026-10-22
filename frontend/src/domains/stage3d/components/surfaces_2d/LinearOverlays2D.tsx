import { useStage3DStore } from '../../store/useStage3DStore';
import type { Linear2DPayload } from '../../../projection3d/types/projection.types';
import { LinearBoundary2D } from './LinearBoundary2D';
import { SgdMarginCorridor2D } from './SgdMarginCorridor2D';
import { LassoSparsityBadge2D } from './LassoSparsityBadge2D';
import { LinearRegression2D } from './LinearRegression2D';

/**
 * Phase 42 — orchestrator for exact linear 2D overlays.
 *
 * Renders on top of the sampled Phase-41 contour: the analytic decision
 * line (crisp white), margin corridor band, sparsity badges and regression
 * elevation guides. Renders nothing when the backend sent no `linear2d`
 * payload (non-linear / pipelined models) — the sampled contour remains.
 */
export function LinearOverlays2D({
  linear2d,
  elevationThresholds,
  xRangeRaw,
  yRangeRaw,
}: {
  linear2d: Linear2DPayload | null;
  elevationThresholds: number[];
  xRangeRaw: [number, number];
  yRangeRaw: [number, number];
}) {
  const showLinearExact = useStage3DStore((s) => s.showLinearExact);
  const showNormalVector = useStage3DStore((s) => s.showNormalVector);
  const showMarginCorridor = useStage3DStore((s) => s.showMarginCorridor);
  const showSparsityBadges = useStage3DStore((s) => s.showSparsityBadges);

  if (!linear2d) return null;
  const byKind = (kind: string) => linear2d.lines.filter((l) => l.kind === kind);
  const firstSeg = (kind: string) => byKind(kind).find((l) => l.segmentWorld)?.segmentWorld ?? null;

  const decisionSeg = firstSeg('decision');
  const pairwiseSegs = byKind('pairwise').filter((l) => l.segmentWorld);
  const confidenceSegs = byKind('confidence').filter((l) => l.segmentWorld);

  return (
    <g aria-label="Exact linear overlays">
      <defs>
        <marker
          id="arrowhead-cyan"
          markerWidth="6"
          markerHeight="6"
          refX="4.5"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L5,2.5 L0,5 Z" fill="var(--ml-primary)" />
        </marker>
      </defs>
      {/* Analytic decision line (+ multiclass pairwise lines). */}
      {showLinearExact && decisionSeg && (
        <LinearBoundary2D
          segment={decisionSeg}
          normal={linear2d.normalVector}
          label={`Exact boundary: ${linear2d.formula}`}
          showNormalVector={showNormalVector}
        />
      )}
      {showLinearExact &&
        pairwiseSegs.map((l, i) => (
          <LinearBoundary2D
            key={`pair-${l.classA}-${l.classB}-${i}`}
            segment={l.segmentWorld}
            normal={[0, 0]}
            label={`Exact pairwise boundary: ${l.label}`}
            showNormalVector={false}
          />
        ))}
      {/* Logistic confidence parallels (exact logit offsets). */}
      {showLinearExact &&
        confidenceSegs.map((l, i) => (
          <line
            key={`conf-${l.level}-${i}`}
            x1={l.segmentWorld![0]}
            y1={-l.segmentWorld![1]}
            x2={l.segmentWorld![2]}
            y2={-l.segmentWorld![3]}
            stroke="#94a3b8"
            strokeWidth={0.035}
            strokeDasharray="0.22 0.14"
            opacity={0.8}
            strokeLinecap="round"
          >
            <title>{`Exact confidence isoline (${l.label})`}</title>
          </line>
        ))}
      {/* Margin corridor band + parallels. */}
      {showMarginCorridor && (firstSeg('margin_pos') || firstSeg('margin_neg')) && (
        <SgdMarginCorridor2D
          posSegment={firstSeg('margin_pos')}
          negSegment={firstSeg('margin_neg')}
          corridorWidth={linear2d.corridorWidth}
          marginLevel={linear2d.marginLevel ?? 1}
        />
      )}
      {/* L1 sparsity badges. */}
      <LassoSparsityBadge2D
        isXZeroed={linear2d.isXZeroed}
        isYZeroed={linear2d.isYZeroed}
        bothZeroed={linear2d.bothZeroed}
        xFeature={linear2d.xFeature}
        yFeature={linear2d.yFeature}
        visible={showSparsityBadges}
      />
      {/* Regression elevation guides. */}
      {showLinearExact && linear2d.task === 'regression' && !linear2d.bothZeroed && (
        <LinearRegression2D
          w1={linear2d.w1}
          w2={linear2d.w2}
          bEff={linear2d.bEff}
          elevationThresholds={elevationThresholds}
          xRangeRaw={xRangeRaw}
          yRangeRaw={yRangeRaw}
          slopeW1={linear2d.slopeW1}
        />
      )}
    </g>
  );
}
