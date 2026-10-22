import { useEffect, useRef, useState } from 'react';
import { useStage3DStore } from '../store/useStage3DStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { sampleColormap } from '../../projection3d/components/colormaps';
import { DecisionBoundaryContour2D, wy } from './DecisionBoundaryContour2D';
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
  const colormap = useProjectionStore((s) => s.colormap);
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);
  const showHeatmap = useStage3DStore((s) => s.showHeatmap);
  const showRulers = useStage3DStore((s) => s.showRulers);

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

  const rulerTicks = [-5, -2.5, 0, 2.5, 5];
  const rawLabel = (w: number, min: number, max: number): string => {
    const v = min + ((w + 5) / 10) * (max - min);
    const abs = Math.abs(v);
    if (abs !== 0 && (abs >= 10000 || abs < 0.001)) return v.toExponential(1);
    return String(Math.round(v * 10000) / 10000);
  };

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
              {/* Trajectory points */}
              <g aria-label="Trajectory points">
                {points.map((p, f) => {
                  const mispredicted =
                    isClassification && p.residual_error !== null && Math.abs(p.residual_error) > 0.5;
                  const r = f === activeFrameIndex ? 0.14 : 0.1;
                  return (
                    <g key={`${p.original_row_index}-${f}`}>
                      <circle
                        cx={p.world_x}
                        cy={wy(p.world_y)}
                        r={r}
                        fill={pointFill(p.raw_z_pred, hMin, hMax, colormap)}
                        stroke={mispredicted ? '#ef4444' : '#0b0f14'}
                        strokeWidth={mispredicted ? 0.05 : 0.03}
                        opacity={f === activeFrameIndex ? 1 : 0.88}
                      />
                      {mispredicted && (
                        <text x={p.world_x} y={wy(p.world_y)} fontSize={0.22} fill="#ef4444" textAnchor="middle" dy="0.08">
                          ×
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
              <WhatIfDisplacementVector2D />
              <ActiveDataPointReticle2D points={points} activeFrameIndex={activeFrameIndex} />
              {/* Precision rulers (raw values, inside the frame to avoid clipping) */}
              {showRulers && contourData && (
                <g fontSize={0.3} fill="#94a3b8" opacity={0.92} aria-label="Precision rulers">
                  {rulerTicks.map((w) => (
                    <g key={`x-${w}`}>
                      <line x1={w} y1={4.86} x2={w} y2={5} stroke="#475569" strokeWidth={0.03} />
                      <text x={w} y={4.72} textAnchor="middle">
                        {rawLabel(w, contourData.xRangeRaw[0], contourData.xRangeRaw[1])}
                      </text>
                    </g>
                  ))}
                  {rulerTicks.map((w) => (
                    <g key={`y-${w}`}>
                      <line x1={-5} y1={wy(w)} x2={-4.86} y2={wy(w)} stroke="#475569" strokeWidth={0.03} />
                      <text x={-4.78} y={wy(w) + 0.11} textAnchor="start">
                        {rawLabel(w, contourData.yRangeRaw[0], contourData.yRangeRaw[1])}
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
          {/* 2D status badge */}
          <div
            className="font-mono-tabular pointer-events-none absolute bottom-1 left-1 rounded px-2 py-1 text-[11px]"
            style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-secondary)' }}
          >
            2D Orthographic · {contourData.gridResolution}×{contourData.gridResolution} · X {contourData.xFeature} · Y{' '}
            {contourData.yFeature} · {contourData.computationTimeMs.toFixed(1)} ms
          </div>
        </div>
      )}
    </div>
  );
}
