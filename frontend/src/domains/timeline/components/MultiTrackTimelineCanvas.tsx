import { useEffect, useMemo, useRef, useState } from 'react';
import { computeScreenPinClusters } from '../utils/pinClusterEngine';
import { useTimelineStore, resolveBookmarkFrame } from '../store/useTimelineStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import {
  BOOKMARK_COLOR_HEX,
  EVENT_TYPE_COLOR,
  type AutoEventPin,
  type BookmarkColor,
  type BookmarkPin,
  type MultiTrackTimelineData,
  type TimelineEventType,
  type UserBookmark,
} from '../types/timeline.types';

/** Layout (CSS px, top → bottom). */
const PAD = 8;
const MINIMAP_H = 26;
const TRACK1_H = 64;
const TRACK2_H = 20;
const TRACK3_H = 34;
const AXIS_H = 16;
const GAP = 4;
const TRACK3_Y = PAD + MINIMAP_H + GAP + TRACK1_H + GAP + TRACK2_H + GAP;
const TOTAL_H = TRACK3_Y + TRACK3_H + GAP + AXIS_H + PAD;
const MINIMAP_Y = PAD;
const TRACK1_Y = MINIMAP_Y + MINIMAP_H + GAP;
const TRACK2_Y = TRACK1_Y + TRACK1_H + GAP;

const PRED_COLOR = '#7dd3fc';
const ACTUAL_COLOR = '#34d399';

function confidenceColor(c: number): string {
  if (!Number.isFinite(c)) return '#64748b';
  if (c >= 0.75) return '#34d399';
  if (c >= 0.5) return '#fbbf24';
  return '#f43e5e';
}

function formatNum(v: number): string {
  if (!Number.isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs >= 1_000_000 || abs < 0.001)) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return String(Math.round(v * 10000) / 10000);
}

/** Bucket centers mapped to frame space (bucket k → frame k*bs + bs/2). */
function bucketFrameCenters(total: number, bucketSize: number, count: number): Float64Array {
  const out = new Float64Array(count);
  for (let k = 0; k < count; k += 1) {
    out[k] = Math.min(k * bucketSize + bucketSize / 2, Math.max(0, total - 1));
  }
  return out;
}

export interface HoverInfo {
  frame: number;
  x: number;
  pin: AutoEventPin | null;
}

/**
 * Phase 11 — dual-layer multi-track timeline canvas.
 *
 * Zero-DOM pins: tracks, heatmap, pins AND clusters all render into ONE
 * static `<canvas>` (redrawn only on data/zoom/filter/bookmark change).
 * A second `<canvas>` draws just the 60FPS playhead imperatively.
 * Exactly 2 `<canvas>` nodes no matter how many pins exist (§5.2).
 */
export function MultiTrackTimelineCanvas({
  data: dataProp,
  rowIndices = [],
  totalFrames: totalFramesProp,
  predictionCurve: predictionCurveProp,
  confidenceScores: confidenceScoresProp,
  autoEvents: autoEventsProp,
  bookmarks: bookmarksProp,
}: {
  /** Live store-backed payload (preferred). */
  data?: MultiTrackTimelineData | null;
  rowIndices?: number[];
  /** Flat test-harness props (spec §4.1 perf test contract). */
  totalFrames?: number;
  predictionCurve?: Float32Array | number[];
  confidenceScores?: Float32Array | number[];
  autoEvents?: AutoEventPin[];
  bookmarks?: UserBookmark[];
}) {
  // Prefer the store payload; synthesize one from flat props when the
  // component is rendered standalone (unit/perf tests).
  const data = useMemo<MultiTrackTimelineData | null>(() => {
    if (dataProp !== undefined) return dataProp;
    if (totalFramesProp === undefined) return null;
    return {
      session_id: 'standalone',
      totalFrames: totalFramesProp,
      isClassification: true,
      isLabeled: true,
      bucketSize: 1,
      predictionCurve: new Float32Array(predictionCurveProp ?? []),
      actualCurve: null,
      confidenceScores: new Float32Array(confidenceScoresProp ?? []),
      autoEvents: autoEventsProp ?? [],
      bookmarks: bookmarksProp ?? [],
    };
  }, [dataProp, totalFramesProp, predictionCurveProp, confidenceScoresProp, autoEventsProp, bookmarksProp]);
  const viewStart = useTimelineStore((s) => s.viewStart);
  const viewEnd = useTimelineStore((s) => s.viewEnd);
  const visible = useTimelineStore((s) => s.visible);
  const showBookmarks = useTimelineStore((s) => s.showBookmarks);
  const selectedBookmarkId = useTimelineStore((s) => s.selectedBookmarkId);

  const containerRef = useRef<HTMLDivElement>(null);
  const staticRef = useRef<HTMLCanvasElement>(null);
  const dynamicRef = useRef<HTMLCanvasElement>(null);
  const [widthPx, setWidthPx] = useState(0);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  // Measure container width (clusters depend on physical px).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      setWidthPx(Math.max(0, Math.floor(w)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const total = data?.totalFrames ?? 0;
  const span = Math.max(1, viewEnd - viewStart);
  const frameAtX = (x: number, w: number) =>
    total <= 0
      ? 0
      : Math.max(0, Math.min(total - 1, Math.round(viewStart + (x / Math.max(1, w)) * span)));

  // ---- static layer: minimap + 3 tracks + axis (rare redraws) ----
  useEffect(() => {
    const canvas = staticRef.current;
    if (!canvas || widthPx <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(widthPx * dpr);
    canvas.height = Math.floor(TOTAL_H * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthPx, TOTAL_H);
    if (!data || total <= 0) {
      // Empty state: dashed line + waiting copy (edge: no queue yet).
      ctx.strokeStyle = 'rgba(148,163,184,0.5)';
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(PAD, TRACK1_Y + TRACK1_H / 2);
      ctx.lineTo(widthPx - PAD, TRACK1_Y + TRACK1_H / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        'กำลังรอข้อมูลผลทำนายเพื่อสร้างกราฟ Multi-Track Timeline',
        widthPx / 2,
        TRACK1_Y + TRACK1_H / 2 - 10,
      );
      return;
    }

    const xOf = (frame: number) => ((frame - viewStart) / span) * widthPx;
    const nb = data.predictionCurve.length;
    const centers = bucketFrameCenters(total, data.bucketSize, nb);

    // Minimap: full-range confidence strip + viewport window.
    const mmY = MINIMAP_Y;
    ctx.fillStyle = 'rgba(148,163,184,0.12)';
    ctx.fillRect(PAD, mmY, widthPx - PAD * 2, MINIMAP_H);
    const mmW = widthPx - PAD * 2;
    for (let k = 0; k < nb; k += 1) {
      const bx = PAD + (k / Math.max(1, nb)) * mmW;
      const bw = Math.max(1, mmW / Math.max(1, nb));
      ctx.fillStyle = confidenceColor(data.confidenceScores[k]);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx, mmY + 2, bw, MINIMAP_H - 4);
    }
    ctx.globalAlpha = 1;
    const vx0 = PAD + (viewStart / Math.max(1, total - 1)) * mmW;
    const vx1 = PAD + (viewEnd / Math.max(1, total - 1)) * mmW;
    ctx.strokeStyle = '#7dd3fc';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(vx0, mmY, Math.max(2, vx1 - vx0), MINIMAP_H);
    ctx.lineWidth = 1;

    // Track 1: prediction curve (+ actual when labeled).
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < nb; k += 1) {
      const v = data.predictionCurve[k];
      if (Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    if (data.actualCurve) {
      for (let k = 0; k < data.actualCurve.length; k += 1) {
        const v = data.actualCurve[k];
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      lo = 0;
      hi = 1;
    }
    if (hi - lo < 1e-9) {
      hi = lo + 1;
    }
    const yOf = (v: number) => TRACK1_Y + TRACK1_H - ((v - lo) / (hi - lo)) * TRACK1_H;
    const strokeSeries = (curve: Float32Array, color: string, dashed: boolean) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.beginPath();
      let started = false;
      for (let k = 0; k < curve.length; k += 1) {
        const f = centers[k];
        if (f < viewStart - span * 0.02 || f > viewEnd + span * 0.02) {
          started = false;
          continue;
        }
        const X = xOf(f);
        const Y = yOf(curve[k]);
        if (!started) {
          ctx.moveTo(X, Y);
          started = true;
        } else {
          ctx.lineTo(X, Y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineWidth = 1;
    };
    strokeSeries(data.predictionCurve, PRED_COLOR, false);
    if (data.actualCurve) strokeSeries(data.actualCurve, ACTUAL_COLOR, true);

    // Track 2: confidence / residual heatmap strip.
    for (let k = 0; k < nb; k += 1) {
      const f0 = k * data.bucketSize;
      const f1 = Math.min((k + 1) * data.bucketSize, total) - 1;
      if (f1 < viewStart || f0 > viewEnd) continue;
      const X0 = xOf(Math.max(f0, viewStart));
      const X1 = xOf(Math.min(f1, viewEnd));
      ctx.fillStyle = confidenceColor(data.confidenceScores[k]);
      ctx.fillRect(X0, TRACK2_Y + 2, Math.max(1, X1 - X0), TRACK2_H - 4);
    }

    // Track 3: event pins + bookmarks (clustered < 14px).
    const pins = data.autoEvents.filter((p) => visible[p.event_type]);
    const clusters = computeScreenPinClusters(pins, viewStart, viewEnd, widthPx, 14);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const midY = TRACK3_Y + TRACK3_H / 2;
    for (const c of clusters) {
      if (c.pins.length === 1) {
        drawPinGlyph(ctx, c.pins[0].event_type, c.centerXPx, midY);
      } else {
        const dominant = c.pins.reduce((a, b) => (b.severity > a.severity ? b : a));
        const label = `${c.pins.length}`;
        const w = Math.max(30, 10 + label.length * 8);
        ctx.fillStyle = 'rgba(10,15,22,0.9)';
        ctx.strokeStyle = EVENT_TYPE_COLOR[dominant.event_type];
        ctx.lineWidth = 1.5;
        roundRect(ctx, c.centerXPx - w / 2, midY - 10, w, 20, 10);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 10px system-ui, sans-serif';
        ctx.fillText(label, c.centerXPx, midY + 0.5);
        ctx.lineWidth = 1;
      }
    }
    if (showBookmarks) {
      for (const b of data.bookmarks) {
        const f = resolveBookmarkFrame(b, rowIndices);
        if (f === null || f < viewStart || f > viewEnd) continue;
        const X = xOf(f);
        const color = BOOKMARK_COLOR_HEX[b.color];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(X, midY - 9);
        ctx.lineTo(X, midY + 9);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(X, midY - 11, 4, 0, Math.PI * 2);
        ctx.fill();
        if (b.bookmark_id === selectedBookmarkId) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(X, midY - 11, 7, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.lineWidth = 1;
      }
    }

    // Axis labels (canvas text keeps the DOM node budget flat).
    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(String(viewStart).padStart(4, '0'), PAD, TRACK3_Y + TRACK3_H + GAP + 10);
    ctx.textAlign = 'center';
    ctx.fillText(
      String(Math.round((viewStart + viewEnd) / 2)).padStart(4, '0'),
      widthPx / 2,
      TRACK3_Y + TRACK3_H + GAP + 10,
    );
    ctx.textAlign = 'right';
    ctx.fillText(String(viewEnd).padStart(4, '0'), widthPx - PAD, TRACK3_Y + TRACK3_H + GAP + 10);
  }, [data, viewStart, viewEnd, span, visible, showBookmarks, selectedBookmarkId, rowIndices, widthPx, total]);

  // ---- dynamic layer: 60FPS playhead only (zero static redraws) ----
  useEffect(() => {
    const canvas = dynamicRef.current;
    if (!canvas || widthPx <= 0) return;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.floor(widthPx * dpr) || canvas.height !== Math.floor(TOTAL_H * dpr)) {
        canvas.width = Math.floor(widthPx * dpr);
        canvas.height = Math.floor(TOTAL_H * dpr);
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, widthPx, TOTAL_H);
      const st = usePlaybackStore.getState();
      if (!data || st.totalFrames <= 1) return;
      const f = st.exactFrameFloat;
      if (f < viewStart || f > viewEnd) return;
      const X = ((f - viewStart) / Math.max(1, viewEnd - viewStart)) * widthPx;
      ctx.strokeStyle = '#7dd3fc';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#7dd3fc';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(X, PAD);
      ctx.lineTo(X, TRACK3_Y + TRACK3_H);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#7dd3fc';
      ctx.beginPath();
      ctx.moveTo(X - 5, PAD);
      ctx.lineTo(X + 5, PAD);
      ctx.lineTo(X, PAD + 7);
      ctx.closePath();
      ctx.fill();
      ctx.lineWidth = 1;
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [data, viewStart, viewEnd, widthPx]);

  // Ctrl + wheel zoom (non-passive so preventDefault works).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const w = Math.max(1, rect.width);
      const cursorFrame = frameAtX(e.clientX - rect.left, w);
      useTimelineStore.getState().zoomBy(e.deltaY > 0 ? 1.3 : 1 / 1.3, cursorFrame);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [total, viewStart, viewEnd, span]);

  const nearestPin = (frame: number): AutoEventPin | BookmarkPin | null => {
    if (!data) return null;
    let best: AutoEventPin | BookmarkPin | null = null;
    let bestDist = 3; // frames
    for (const p of data.autoEvents) {
      if (!visible[p.event_type]) continue;
      const d = Math.abs(p.frame_index - frame);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    if (showBookmarks) {
      for (const b of data.bookmarks) {
        const f = resolveBookmarkFrame(b, rowIndices);
        if (f === null) continue;
        const d = Math.abs(f - frame);
        if (d < bestDist) {
          bestDist = d;
          best = bookmarkPinOf(b, f);
        }
      }
    }
    return best;
  };

  const onClickStatic = (e: React.MouseEvent) => {
    if (!data || total <= 0) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const w = Math.max(1, rect.width);
    const st = useTimelineStore.getState();

    if (y >= MINIMAP_Y && y <= MINIMAP_Y + MINIMAP_H) {
      // Minimap: recenter the window on the clicked frame.
      const f = frameAtX(x, w);
      const half = (viewEnd - viewStart) / 2;
      st.setView(Math.round(f - half), Math.round(f + half));
      return;
    }
    if (y >= TRACK3_Y && y <= TRACK3_Y + TRACK3_H) {
      // Pin lane: hit-test clusters (click cluster → zoom, pin → seek).
      const pins = data.autoEvents.filter((p) => visible[p.event_type]);
      const clusters = computeScreenPinClusters(pins, viewStart, viewEnd, w, 14);
      let best = -1;
      let bestDist = 14;
      clusters.forEach((c, idx) => {
        const d = Math.abs(c.centerXPx - x);
        if (d < bestDist) {
          bestDist = d;
          best = idx;
        }
      });
      // Bookmarks take priority when closer.
      let bmFrame: number | null = null;
      if (showBookmarks) {
        for (const b of data.bookmarks) {
          const f = resolveBookmarkFrame(b, rowIndices);
          if (f === null || f < viewStart || f > viewEnd) continue;
          const bx = ((f - viewStart) / span) * w;
          if (Math.abs(bx - x) < bestDist) {
            bestDist = Math.abs(bx - x);
            bmFrame = f;
            st.setSelectedBookmark(b.bookmark_id);
          }
        }
      }
      if (bmFrame !== null) {
        usePlaybackStore.getState().seekToFrame(bmFrame, true);
        return;
      }
      if (best === -1) return;
      const c = clusters[best];
      if (c.pins.length === 1) {
        usePlaybackStore.getState().seekToFrame(c.pins[0].frame_index, true);
      } else {
        const pad = Math.max(2, Math.round((c.endFrame - c.startFrame) * 0.4));
        st.setView(c.startFrame - pad, c.endFrame + pad);
      }
      return;
    }
    // Curves / heatmap: click-to-seek.
    usePlaybackStore.getState().seekToFrame(frameAtX(x, w), true);
  };

  const onHover = (e: React.MouseEvent) => {
    if (!data || total <= 1) {
      setHover(null);
      return;
    }
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frame = frameAtX(x, Math.max(1, rect.width));
    setHover({ frame, x, pin: nearestPin(frame) });
  };

  const hoverPred =
    data && hover
      ? (data.predictionCurve[Math.floor(hover.frame / data.bucketSize)] ?? null)
      : null;
  const hoverConf =
    data && hover
      ? (data.confidenceScores[Math.floor(hover.frame / data.bucketSize)] ?? null)
      : null;

  const zeroAnomaly = !!data && total > 0 && data.autoEvents.length === 0;

  return (
    <div ref={containerRef} className="relative w-full" style={{ touchAction: 'pan-y' }}>
      {zeroAnomaly && (
        <div
          className="mb-1 inline-block rounded px-2 py-0.5 text-[11px]"
          style={{ background: 'rgba(52,211,153,0.14)', color: '#34d399' }}
          role="status"
        >
          ✨ 0 Misclassifications Detected
        </div>
      )}
      {data && !data.isLabeled && total > 0 && (
        <div
          className="font-mono-tabular mb-1 ml-2 inline-block rounded px-2 py-0.5 text-[11px]"
          style={{ background: 'rgba(148,163,184,0.14)', color: 'var(--text-secondary)' }}
          role="status"
        >
          Unlabeled Dataset
        </div>
      )}
      <div className="relative" style={{ height: TOTAL_H }}>
        <canvas
          ref={staticRef}
          aria-label="Multi-track timeline (prediction, confidence, events)"
          onClick={onClickStatic}
          onMouseMove={onHover}
          onMouseLeave={() => setHover(null)}
          style={{ position: 'absolute', inset: 0, width: '100%', height: TOTAL_H, cursor: 'crosshair' }}
        />
        <canvas
          ref={dynamicRef}
          aria-hidden
          style={{ position: 'absolute', inset: 0, width: '100%', height: TOTAL_H, pointerEvents: 'none' }}
        />
        {hover && (
          <div
            className="font-mono-tabular pointer-events-none absolute top-0 z-10 rounded px-2 py-1 text-[11px]"
            style={{
              left: Math.max(4, Math.min(widthPx - 4, hover.x)),
              transform: 'translate(-50%, 0)',
              background: 'rgba(10,15,22,0.92)',
              border: '1px solid var(--border-strong)',
              color: 'var(--text-primary)',
              whiteSpace: 'nowrap',
            }}
            role="status"
          >
            Frame #{hover.frame}
            {rowIndices[hover.frame] !== undefined && ` | Row #${rowIndices[hover.frame]}`}
            {hoverPred !== null && ` | Pred: ${formatNum(hoverPred)}`}
            {hoverConf !== null && ` | Conf: ${(hoverConf * 100).toFixed(1)}%`}
            {hover.pin && (
              <span style={{ color: '#fbbf24' }}>
                <br />◆ {hover.pin.title}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function drawPinGlyph(ctx: CanvasRenderingContext2D, type: TimelineEventType, x: number, y: number) {
  ctx.fillStyle = EVENT_TYPE_COLOR[type];
  ctx.beginPath();
  if (type === 'misclassified') {
    ctx.arc(x, y, 5, 0, Math.PI * 2);
  } else if (type === 'decision_flip') {
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x + 5.5, y + 4);
    ctx.lineTo(x - 5.5, y + 4);
    ctx.closePath();
  } else if (type === 'low_confidence') {
    ctx.rect(x - 4.5, y - 4.5, 9, 9);
  } else {
    // input_outlier: diamond.
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x + 5, y);
    ctx.lineTo(x, y + 6);
    ctx.lineTo(x - 5, y);
    ctx.closePath();
  }
  ctx.fill();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function bookmarkPinOf(b: UserBookmark, frame: number): BookmarkPin {
  return {
    frame_index: frame,
    dataset_row_index: b.dataset_row_index,
    event_type: 'misclassified',
    severity: 1,
    title: b.note_text || `Bookmark @ frame ${frame}`,
    detail: b.note_text,
    bookmarkId: b.bookmark_id,
    color: b.color,
  };
}
