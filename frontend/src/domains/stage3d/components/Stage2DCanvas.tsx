import { useEffect, useRef, useState } from 'react';
import { useStage3DStore } from '../store/useStage3DStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { sampleColormap } from '../../projection3d/components/colormaps';
import { DecisionBoundaryContour2D, wy } from './DecisionBoundaryContour2D';
import { HexbinOverlay2D, type HexbinMode } from './surfaces_2d/HexbinOverlay2D';
import { PointDisplayToolbar2D } from './surfaces_2d/PointDisplayToolbar2D';
import { PointIntelligenceLayer2D } from './surfaces_2d/PointIntelligenceLayer2D';
import { LinearPointIntelOverlays2D } from './surfaces_2d/LinearPointIntelOverlays2D';
import { LinearOverlays2D } from './surfaces_2d/LinearOverlays2D';
import { DistanceOverlays2D } from './surfaces_2d/DistanceOverlays2D';
import { ProbabilisticOverlays2D } from './surfaces_2d/ProbabilisticOverlays2D';
import { TreesOverlays2D } from './surfaces_2d/TreesOverlays2D';
import { BoostingOverlays2D } from './surfaces_2d/BoostingOverlays2D';
import { NeuralOverlays2D } from './surfaces_2d/NeuralOverlays2D';
import { scaledRulerLabel } from './surfaces_2d/neural2dMath';
import { ActiveDataPointReticle2D } from './ActiveDataPointReticle2D';
import { WhatIfDisplacementVector2D } from './WhatIfDisplacementVector2D';

/**
 * Phase 41 — 2D planar viewport root.
 *
 * Orthographic-equivalent rendering: world X/Y map 1:1 onto a square SVG
 * stage (no perspective divide, no occlusion). Layers:
 *  - heatmap background on a <canvas> (uint8 field → colormap LUT),
 *  - vector boundary/isolines + trajectory points + reticle on SVG.
 * Click-to-scrub and hover tooltips mirror the 3D interactions at 60 FPS
 * (pointer math only — no React state per mousemove except hover frame).
 */

const WORLD_HALF = 5.5;

function pointFill(rawZPred: number, min: number, max: number, colormap: Parameters<typeof sampleColormap>[0]): string {
  const span = max - min;
  const t = span < 1e-12 ? 0.5 : (rawZPred - min) / span;
  const [r, g, b] = sampleColormap(colormap, t);
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

export function Stage2DCanvas() {
  const contourData = useProjectionStore((s) => s.contourData);
  const contourLoading = useProjectionStore((s) => s.contourLoading);
  const contourError = useProjectionStore((s) => s.contourError);
  const knnFrame = useProjectionStore((s) => s.knnFrame);
  const boostFrame = useProjectionStore((s) => s.boostFrame);
  const colormap = useProjectionStore((s) => s.colormap);
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);
  const adaBoostStage = useStage3DStore((s) => s.adaBoostStage);
  const showHeatmap = useStage3DStore((s) => s.showHeatmap);
  const showRulers = useStage3DStore((s) => s.showRulers);
  const showHexbins = useStage3DStore((s) => s.showHexbins);
  const pointIntel = useProjectionStore((s) => s.pointIntel);
  const pipelineSpace = useStage3DStore((s) => s.pipelineSpace);
  const [hexbinMode, setHexbinMode] = useState<HexbinMode>('count');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const squareRef = useRef<HTMLDivElement>(null);
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const [hoverXY, setHoverXY] = useState<{ xPct: number; yPct: number } | null>(null);
  const lastHoverAt = useRef(0);

  // Paint the quantized heatmap through the active colormap LUT.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !contourData || !showHeatmap) return;
    const res = contourData.gridResolution;
    const field = contourData.heatmapQuantized;
    if (field.length !== res * res) return;
    canvas.width = res;
    canvas.height = res;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(res, res);
    for (let j = 0; j < res; j += 1) {
      for (let i = 0; i < res; i += 1) {
        // Grid row 0 = y_min (bottom); canvas row 0 = top → flip vertically.
        const q = field[j * res + i] / 255;
        const [r, g, b] = sampleColormap(colormap, q);
        const dst = ((res - 1 - j) * res + i) * 4;
        img.data[dst] = Math.round(r * 255);
        img.data[dst + 1] = Math.round(g * 255);
        img.data[dst + 2] = Math.round(b * 255);
        img.data[dst + 3] = 235;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [contourData, colormap, showHeatmap]);

  const toWorld = (clientX: number, clientY: number): { wx: number; wy: number } | null => {
    const el = squareRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return null;
    const fx = (clientX - rect.left) / rect.width;
    const fy = (clientY - rect.top) / rect.height;
    return { wx: -WORLD_HALF + fx * WORLD_HALF * 2, wy: WORLD_HALF - fy * WORLD_HALF * 2 };
  };

  const nearestFrame = (wx: number, wyv: number): number => {
    let best = 0;
    let bestD = Infinity;
    for (let f = 0; f < points.length; f += 1) {
      const p = points[f];
      const d = (p.world_x - wx) ** 2 + (p.world_y - wyv) ** 2;
      if (d < bestD) {
        bestD = d;
        best = f;
      }
    }
    return best;
  };

  const handleClick = (e: React.MouseEvent) => {
    const w = toWorld(e.clientX, e.clientY);
    if (!w) return;
    const frame = nearestFrame(w.wx, w.wy);
    usePlaybackStore.getState().seekToFrame(frame, true);
    useStage3DStore.getState().setActiveFrameIndex(frame);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const now = performance.now();
    if (now - lastHoverAt.current < 50) return;
    lastHoverAt.current = now;
    const w = toWorld(e.clientX, e.clientY);
    const el = squareRef.current;
    if (!w || !el || points.length === 0) {
      setHoverFrame(null);
      return;
    }
    const rect = el.getBoundingClientRect();
    let best = -1;
    let bestPx = Infinity;
    for (let f = 0; f < points.length; f += 1) {
      const p = points[f];
      const px = ((p.world_x + WORLD_HALF) / (WORLD_HALF * 2)) * rect.width;
      const py = ((WORLD_HALF - p.world_y) / (WORLD_HALF * 2)) * rect.height;
      const d = Math.hypot(px - (e.clientX - rect.left), py - (e.clientY - rect.top));
      if (d < bestPx) {
        bestPx = d;
        best = f;
      }
    }
    if (best >= 0 && bestPx < 16) {
      const p = points[best];
      setHoverFrame(best);
      setHoverXY({
        xPct: ((p.world_x + WORLD_HALF) / (WORLD_HALF * 2)) * 100,
        yPct: ((WORLD_HALF - p.world_y) / (WORLD_HALF * 2)) * 100,
      });
    } else {
      setHoverFrame(null);
      setHoverXY(null);
    }
  };

  const isClassification = contourData?.predictionTask === 'classification';
  const hMin = contourData?.heatmapMin ?? 0;
  const hMax = contourData?.heatmapMax ?? 1;
  // Phase 42 — sampled elevation thresholds double as levels for the exact
  // analytic regression guides (parallel lines, razor-sharp).
  const elevationThresholds = contourData
    ? contourData.contours.filter((c) => c.kind === 'elevation').map((c) => c.threshold)
    : [];

  const rulerTicks = [-5, -2.5, 0, 2.5, 5];
  const rawLabel = (w: number, min: number, max: number): string => {
    const v = min + ((w + 5) / 10) * (max - min);
    const abs = Math.abs(v);
    if (abs !== 0 && (abs >= 10000 || abs < 0.001)) return v.toExponential(1);
    return String(Math.round(v * 10000) / 10000);
  };
  // Phase 47 — pipeline scaled-space rulers: exact (raw(w) - mu) / sigma labels.
  const neuralPipe =
    contourData?.neural2d?.family === 'pipeline' ? (contourData.neural2d.pipeline ?? null) : null;
  const useScaledRulers =
    pipelineSpace === 'scaled' &&
    !!neuralPipe?.hasToggle &&
    neuralPipe.muX !== null &&
    neuralPipe.sigmaX !== null &&
    neuralPipe.muY !== null &&
    neuralPipe.sigmaY !== null;
  const xRulerLabel = (w: number): string =>
    useScaledRulers
      ? scaledRulerLabel(w, contourData!.xRangeRaw[0], contourData!.xRangeRaw[1], neuralPipe!.muX!, neuralPipe!.sigmaX!)
      : rawLabel(w, contourData!.xRangeRaw[0], contourData!.xRangeRaw[1]);
  const yRulerLabel = (w: number): string =>
    useScaledRulers
      ? scaledRulerLabel(w, contourData!.yRangeRaw[0], contourData!.yRangeRaw[1], neuralPipe!.muY!, neuralPipe!.sigmaY!)
      : rawLabel(w, contourData!.yRangeRaw[0], contourData!.yRangeRaw[1]);

  return (
    <div className="relative flex h-full min-h-0 w-full items-center justify-center" data-stage2d-canvas="true">
      {contourLoading && !contourData && (
        <div className="absolute inset-0 flex items-center justify-center text-sm" style={{ color: 'var(--text-secondary)' }} role="status">
          Baking 2D contour grid…
        </div>
      )}
      {contourError && !contourData && (
        <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm" style={{ color: '#f43f5e' }} role="alert">
          2D contour failed: {contourError}
        </div>
      )}
      {contourData && (
        <div
          ref={squareRef}
          className="relative aspect-square max-h-full max-w-full"
          style={{ width: 'min(100%, calc(100vh - 340px))', minWidth: 0 }}
        >
          {/*
            Heatmap canvas is inset to the exact ±5 data rect (0.5/11 offset,
            10/11 size) so texels line up 1:1 with the vector overlay; the SVG
            owns the full ±5.5 stage including rulers.
          */}
          {showHeatmap && (
            <canvas
              ref={canvasRef}
              className="pointer-events-none absolute"
              style={{
                left: `${(0.5 / 11) * 100}%`,
                top: `${(0.5 / 11) * 100}%`,
                width: `${(10 / 11) * 100}%`,
                height: `${(10 / 11) * 100}%`,
                imageRendering: 'auto',
              }}
              aria-hidden="true"
            />
          )}
          <div
            className="absolute inset-0"
            onClick={handleClick}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => {
              setHoverFrame(null);
              setHoverXY(null);
            }}
            style={{ cursor: 'crosshair' }}
          >
            <svg
              viewBox={`${-WORLD_HALF} ${-WORLD_HALF} ${WORLD_HALF * 2} ${WORLD_HALF * 2}`}
              className="absolute inset-0 h-full w-full"
              role="img"
              aria-label={`2D planar decision map for ${contourData.xFeature} versus ${contourData.yFeature}`}
            >
              {/* Frame border of the [-5, +5] world cube */}
              <rect x={-5} y={-5} width={10} height={10} fill="none" stroke="#334155" strokeWidth={0.03} />
              <DecisionBoundaryContour2D contours={contourData.contours} />
              {/* Phase 42 — exact analytic linear overlays (no-op when linear2d is null). */}
              <LinearOverlays2D
                linear2d={contourData.linear2d}
                elevationThresholds={elevationThresholds}
                xRangeRaw={contourData.xRangeRaw}
                yRangeRaw={contourData.yRangeRaw}
              />
              {/* Phase 43 — KNN radar, SVM margins, SV halos. */}
              <DistanceOverlays2D contourData={contourData} knnFrame={knnFrame} />
              {/* Phase 44 — probability ellipses, LDA line / QDA conic, NB steps. */}
              <ProbabilisticOverlays2D contourData={contourData} />
              {/* Phase 45 — tree cuts, leaf boxes, forest mosaic, adaboost cascade. */}
              <TreesOverlays2D contourData={contourData} />
              {/* Phase 46 — boosting polygons, bin grids, oblivious checkerboard, SHAP vectors. */}
              <BoostingOverlays2D contourData={contourData} boostFrame={boostFrame} />
              {/* Phase 47 — MLP neuron cuts, pipeline space toggle, ONNX latent plane. */}
              <NeuralOverlays2D contourData={contourData} />
              {/* Phase 49 — hexbin density under the points (mechanism D). */}
              {showHexbins && pointIntel && (
                <HexbinOverlay2D
                  cells={pointIntel.hexbinCells}
                  maxBinCount={pointIntel.maxBinCount}
                  mode={hexbinMode}
                  classColor={(cls) => {
                    if (cls === null || !contourData) return '#94a3b8';
                    const idx = contourData.classes.findIndex((c) => c === cls || String(c) === String(cls));
                    const n = contourData.classes.length;
                    const t = n < 2 ? 0.5 : Math.max(0, idx) / (n - 1);
                    const [r, g, b] = sampleColormap(colormap, t);
                    return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
                  }}
                  visible
                />
              )}
              {/* Phase 49 — empirical density contours (KDE-lite isolines). */}
              {showHexbins && pointIntel && pointIntel.densityContours.length > 0 && (
                <g aria-label="Density contours" className="pointer-events-none">
                  {pointIntel.densityContours.map((level, li) => (
                    <path
                      key={`density-${level.densityThreshold}-${li}`}
                      d={level.segments
                        .map(([x1, y1, x2, y2]) => `M${x1.toFixed(3)} ${wy(y1).toFixed(3)}L${x2.toFixed(3)} ${wy(y2).toFixed(3)}`)
                        .join('')}
                      fill="none"
                      stroke="#818cf8"
                      strokeWidth={0.03}
                      strokeDasharray="0.3 0.14"
                      opacity={0.75}
                    >
                      <title>{`Empirical density ≥ ${level.densityThreshold}`}</title>
                    </path>
                  ))}
                </g>
              )}
              {/* Phase 49 — comet trails + dual-ring glyphs + error lens (A/B/C). */}
              <PointIntelligenceLayer2D
                points={points}
                activeFrameIndex={activeFrameIndex}
                isClassification={isClassification}
                classes={contourData?.classes ?? []}
                colormap={colormap}
                heatmapMin={hMin}
                heatmapMax={hMax}
                trailWeights={
                  contourData?.linear2d && !contourData.linear2d.bothZeroed
                    ? { w1: contourData.linear2d.w1, w2: contourData.linear2d.w2 }
                    : null
                }
              />
              {/* Phase 50 — linear-family spokes, margin rings, stems, hexbin alignment. */}
              {contourData?.linear2d && (
                <LinearPointIntelOverlays2D
                  linear2d={contourData.linear2d}
                  points={points}
                  activeFrameIndex={activeFrameIndex}
                  hexbinCells={pointIntel?.hexbinCells ?? []}
                  xRangeRaw={contourData.xRangeRaw}
                  yRangeRaw={contourData.yRangeRaw}
                />
              )}
              <WhatIfDisplacementVector2D />
              <ActiveDataPointReticle2D points={points} activeFrameIndex={activeFrameIndex} />
              {/* Precision rulers (raw values, inside the frame to avoid clipping) */}
              {showRulers && contourData && (
                <g fontSize={0.3} fill="#94a3b8" opacity={0.92} aria-label="Precision rulers">
                  {rulerTicks.map((w) => (
                    <g key={`x-${w}`}>
                      <line x1={w} y1={4.86} x2={w} y2={5} stroke="#475569" strokeWidth={0.03} />
                      <text x={w} y={4.72} textAnchor="middle">
                        {xRulerLabel(w)}
                      </text>
                    </g>
                  ))}
                  {rulerTicks.map((w) => (
                    <g key={`y-${w}`}>
                      <line x1={-5} y1={wy(w)} x2={-4.86} y2={wy(w)} stroke="#475569" strokeWidth={0.03} />
                      <text x={-4.78} y={wy(w) + 0.11} textAnchor="start">
                        {yRulerLabel(w)}
                      </text>
                    </g>
                  ))}
                </g>
              )}
            </svg>
            {/* Hover tooltip */}
            {hoverFrame !== null && hoverXY && points[hoverFrame] && (
              <div
                className="font-mono-tabular pointer-events-none absolute z-10 rounded px-2 py-1 text-[11px]"
                style={{
                  background: 'rgba(10,15,22,0.9)',
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--text-primary)',
                  left: `${Math.min(hoverXY.xPct, 72)}%`,
                  top: `${Math.max(hoverXY.yPct - 18, 0)}%`,
                }}
                role="status"
              >
                Frame {hoverFrame} · Row #{points[hoverFrame].original_row_index} · ŷ{' '}
                {Number(points[hoverFrame].raw_z_pred).toPrecision(4)}
              </div>
            )}
          </div>
            {/* Phase 49 — point display quick toggles [T][R][E][H]. */}
            <PointDisplayToolbar2D hexbinMode={hexbinMode} onHexbinMode={setHexbinMode} />
            {/* Phase 49 — hexbin bake badge (cells · max · ms). */}
            {showHexbins && pointIntel && (
              <div
                className="font-mono-tabular pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 rounded px-2 py-1 text-[11px]"
                style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-secondary)' }}
                title={`${pointIntel.hexbinCells.length} hexbin cells in world coords`}
              >
                ⬡ {pointIntel.hexbinCells.length} bins · max {pointIntel.maxBinCount} · {pointIntel.computationTimeMs.toFixed(1)} ms
              </div>
            )}
            {/* 2D status badge */}
          <div
            className="font-mono-tabular pointer-events-none absolute bottom-1 left-1 rounded px-2 py-1 text-[11px]"
            style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-secondary)' }}
          >
            2D Orthographic · {contourData.gridResolution}×{contourData.gridResolution} · X {contourData.xFeature} · Y{' '}
            {contourData.yFeature} · {contourData.computationTimeMs.toFixed(1)} ms
          </div>
          {/* Phase 43 — KNN HUD badge (K · radar radius · vote tally). */}
          {knnFrame && !knnFrame.outlier && (
            <div
              className="font-mono-tabular pointer-events-none absolute top-8 right-1 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
              style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-primary)' }}
              title={`metric=${knnFrame.metric} weights=${knnFrame.weightsMode} row=${knnFrame.rowIndex}`}
            >
              K = {knnFrame.k} · Rₜ = {knnFrame.radarRadiusRaw.toPrecision(4)} · Votes:{' '}
              {Object.entries(knnFrame.votes)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([lab, w]) => `${lab} ${(w * 100).toFixed(0)}%`)
                .join(' vs ')}
              {knnFrame.displayCapped ? ` (+${knnFrame.totalNeighbors - knnFrame.neighbors.length} more)` : ''}
            </div>
          )}
          {/* Phase 43 — SV truncation badge (top |dual|-ranked halves shown). */}
          {contourData.svm2d && contourData.svm2d.halosTruncated && (
            <div
              className="font-mono-tabular pointer-events-none absolute top-8 left-1 rounded px-2 py-1 text-[11px]"
              style={{ background: 'rgba(10,15,22,0.72)', color: '#7dd3fc' }}
              title={`${contourData.svm2d.nSupportTotal} support vectors (${contourData.svm2d.kernel} kernel); showing top ${contourData.svm2d.halos.length} by |dual|`}
            >
              ◆ {contourData.svm2d.halos.length}/{contourData.svm2d.nSupportTotal} SVs · top-|α|
            </div>
          )}
          {contourData.linear2d && (
            <div
              className="font-mono-tabular pointer-events-none absolute top-1 right-1 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
              style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-primary)' }}
              title={contourData.linear2d.formula}
            >
              {contourData.linear2d.formula}
              {contourData.linear2d.corridorWidth !== null &&
                contourData.linear2d.task === 'classification' &&
                ` · W=${contourData.linear2d.corridorWidth.toPrecision(4)}`}
              {contourData.linear2d.l1Ratio !== null && ` · ρ=${contourData.linear2d.l1Ratio.toFixed(2)}`}
            </div>
          )}
          {/* Phase 45 — tree/ensemble badge + AdaBoost cascade slider. */}
          {contourData.trees2d &&
            (() => {
              const t = contourData.trees2d!;
              const effStage =
                adaBoostStage === null ? t.nEstimators : Math.max(0, Math.min(adaBoostStage, t.nEstimators));
              const onSlice = t.stages.filter((s) => s.axis !== 'off').length;
              return (
                <div
                  className="font-mono-tabular pointer-events-auto absolute top-1 left-1 max-w-[70%] rounded px-2 py-1 text-[11px]"
                  style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-primary)' }}
                  title={t.formula}
                >
                  {t.family === 'decision_tree' &&
                    `▦ Tree ${t.segments.length} cuts · ${t.leaves.length} leaves${t.segmentsTruncated || t.leavesTruncated ? ' · capped' : ''}`}
                  {(t.family === 'random_forest' || t.family === 'extra_trees') &&
                    `▦ ${t.family === 'random_forest' ? 'RF' : 'ET'} top-${t.treesExported}/${t.nEstimators} cuts${t.segmentsTruncated ? ' · capped' : ''}`}
                  {t.family === 'adaboost' && (
                    <span className="inline-flex items-center gap-2">
                      <span title={t.formula}>
                        {`◈ Boost T=${effStage}/${t.nEstimators}${effStage === 0 ? ` · base P=${t.baseValue ?? 0.5}` : ''}${onSlice < t.stages.length ? ` · ${onSlice} on-slice` : ''}${t.stagesTruncated ? ' · capped' : ''}`}
                      </span>
                      <input
                        type="range"
                        min={0}
                        max={t.nEstimators}
                        value={effStage}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          useStage3DStore.getState().setAdaBoostStage(v >= t.nEstimators ? null : v);
                        }}
                        aria-label="Weak learner stage"
                        title="Drag to replay the staged cascade (T=0 shows the base prediction)"
                        className="h-1 w-24 cursor-pointer"
                      />
                    </span>
                  )}
                  {t.noAxisSplits &&
                    ' · Splits occur on non-visualized axes; uniform prediction on this slice'}
                  {t.multiOutput && ' · output 0'}
                </div>
              );
            })()}
          {/* Phase 44 — probabilistic badges (conic type / variant / truncation). */}
          {contourData.probabilistic2d && (
            <div
              className="font-mono-tabular pointer-events-none absolute bottom-8 left-1 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
              style={{ background: 'rgba(10,15,22,0.72)', color: '#a5b4fc' }}
              title={contourData.probabilistic2d.formula}
            >
              {contourData.probabilistic2d.family === 'gnb' &&
                `◯ GNB ${contourData.probabilistic2d.ellipses.length}×(1σ/2σ/3σ)`}
              {contourData.probabilistic2d.family === 'lda' && '━ LDA shared-Σ line'}
              {contourData.probabilistic2d.family === 'qda' &&
                `∿ QDA ${contourData.probabilistic2d.conic?.conicType ?? 'conic'}`}
              {contourData.probabilistic2d.family === 'discrete_nb' &&
                `▤ ${contourData.probabilistic2d.discrete?.variant ?? 'discrete'} NB steps`}
              {contourData.probabilistic2d.ellipsesTruncated && ' · top-8'}
              {contourData.probabilistic2d.isCovarianceApproximated && ' · Σ≈'}
            </div>
          )}
          {/* Phase 46 — boosting heavyweight badge (family summary + live SHAP readout). */}
          {contourData.boosting2d &&
            (() => {
              const b = contourData.boosting2d!;
              const live =
                boostFrame && boostFrame.family === b.family ? boostFrame.shap : b.shapVector;
              return (
                <div
                  className="font-mono-tabular pointer-events-none absolute right-1 bottom-1 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
                  style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-primary)' }}
                  title={b.formula}
                >
                  {(b.family === 'xgboost' || b.family === 'lightgbm') &&
                    `⚡ ${b.family === 'xgboost' ? 'XGB' : 'LGBM'} tree-0 polygons · T=${b.nEstimators}${b.segmentsTruncated || b.leavesTruncated ? ' · capped' : ''}`}
                  {b.family === 'sklearn_gbm' &&
                    `▦ GBM top-${b.treesExported}/${b.nEstimators} residual cuts${b.segmentsTruncated ? ' · capped' : ''}`}
                  {b.family === 'hist_gbm' &&
                    `▤ HistGBM ${b.binEdgesXRaw.length + 1}×${b.binEdgesYRaw.length + 1} bins${b.activeBin ? ` · bin (${b.activeBin[0]}, ${b.activeBin[1]})` : ''}`}
                  {b.family === 'catboost' &&
                    `▦ CatBoost ${b.obliviousXCutsRaw.length + 1}×${b.obliviousYCutsRaw.length + 1} checkerboard${b.obliviousTruncated ? ' · capped' : ''}`}
                  {live && ` · SHAP (${live.phiX >= 0 ? '+' : ''}${live.phiX.toFixed(2)}, ${live.phiY >= 0 ? '+' : ''}${live.phiY.toFixed(2)})`}
                  {b.noAxisSplits && ' · off-slice splits only'}
                </div>
              );
            })()}
          {/* Phase 47 — neural / pipeline / ONNX badge (MLP ratio, toggle state, latent summary). */}
          {contourData.neural2d &&
            (() => {
              const n = contourData.neural2d!;
              return (
                <div
                  className="font-mono-tabular pointer-events-none absolute bottom-14 left-1 max-w-[70%] truncate rounded px-2 py-1 text-[11px]"
                  style={{ background: 'rgba(10,15,22,0.72)', color: '#67e8f9' }}
                  title={n.formula}
                >
                  {n.family === 'mlp' &&
                    `◉ MLP ${n.activeCount}/${n.activeCount + n.deadCount} active${n.hyperplanesTruncated ? ' · capped' : ''}${n.activation ? ` · ${n.activation}` : ''}`}
                  {n.family === 'pipeline' &&
                    `${n.pipeline?.hasToggle ? (pipelineSpace === 'scaled' ? '◈ Scaled space' : '◈ Raw space') : '◈ Raw space only'}${n.pipeline ? ` → ${n.pipeline.innerFamily}` : ''}`}
                  {n.family === 'onnx_classical' &&
                    `⬡ ${n.onnx?.operator ?? 'ONNX'} · ${n.onnx?.nNodes ?? 0} nodes`}
                  {n.family === 'onnx_deep' &&
                    (n.onnx?.latentAvailable
                      ? `⬡ latent SVD 2D · ${n.onnx.latentPoints2d.length} pts`
                      : `⬡ deep graph · ${n.onnx?.nNodes ?? 0} nodes`)}
                  {n.noAxisSplits && n.family !== 'pipeline' && ' · off-slice splits only'}
                </div>
              );
            })()}
        </div>
      )}
    </div>
  );
}
