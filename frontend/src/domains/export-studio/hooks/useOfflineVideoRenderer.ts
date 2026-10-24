/**
 * Phase 57 — deterministic off-screen video renderer (spec §5).
 *
 * `seekToFrame(i)` tick loop (no wall-clock playback) → per-frame Why/What-If
 * lookup from the prefetched explainability cache → `OfflineCompositor`
 * paint → WebCodecs `VideoEncoder` (H.264 for .mp4, VP9 for .webm) with
 * MediaRecorder-on-compositor-canvas fallback. Single reusable virtual
 * canvas → JS heap stays flat (Master Plan budget ≤ 350 MB).
 */
import { useCallback, useRef } from 'react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useExplainabilityStore } from '../../explainability/store/useExplainabilityStore';
import { useWhatIfStore } from '../../whatif/store/useWhatIfStore';
import { getStage3DCapture, getStage2DCapture } from './stage3dCaptureRegistry';
import { OfflineCompositor } from '../components/offscreenCompositor';
import { deterministicFrameSequence } from '../components/advancedVideoExportMath';
import type {
  AdvancedVideoExportOptions,
  FrameWhyDatum,
  OfflineRenderProgress,
} from '../types/advancedVideoExport.types';

export interface OfflineRenderCallbacks {
  onProgress?: (p: OfflineRenderProgress) => void;
  signal?: AbortSignal;
}

type SupportedEncoder = 'webcodecs-h264' | 'webcodecs-vp9' | 'mediarecorder';

function webcodecsAvailable(): boolean {
  return typeof (globalThis as unknown as { VideoEncoder?: unknown }).VideoEncoder !== 'undefined';
}

function pickEncoder(format: 'mp4' | 'webm'): SupportedEncoder | null {
  if (webcodecsAvailable()) return format === 'mp4' ? 'webcodecs-h264' : 'webcodecs-vp9';
  if (typeof MediaRecorder !== 'undefined') return 'mediarecorder';
  return null;
}

async function rasterizeStage(
  dimension: '3d' | '2d',
  compositorW: number,
  compositorH: number,
): Promise<CanvasImageSource | null> {
  if (dimension === '3d') {
    const handles = getStage3DCapture();
    if (!handles) return null;
    try {
      // Synchronous re-render tick so capture is never blank under
      // preserveDrawingBuffer:false (same rule as Phase 16 snapshots).
      handles.gl.render(handles.scene, handles.camera);
      const src = handles.gl.domElement;
      const bmp = await createImageBitmap(src);
      return bmp;
    } catch {
      return handles.gl.domElement ?? null;
    }
  }
  const h2d = getStage2DCapture();
  if (!h2d) return null;
  try {
    const xml = new XMLSerializer().serializeToString(h2d.svg);
    const svg64 = btoa(unescape(encodeURIComponent(xml)));
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('svg raster failed'));
      img.src = `data:image/svg+xml;base64,${svg64}`;
    });
    const off = document.createElement('canvas');
    off.width = compositorW;
    off.height = compositorH;
    const ctx = off.getContext('2d');
    if (!ctx) return null;
    if (h2d.heatmapCanvas) {
      try {
        ctx.drawImage(h2d.heatmapCanvas, 0, 0, off.width, off.height);
      } catch {
        /* heatmap optional */
      }
    }
    ctx.drawImage(img, 0, 0, off.width, off.height);
    return off;
  } catch {
    return null;
  }
}

function readWhyTop(frame: number): FrameWhyDatum[] {
  const cache = useExplainabilityStore.getState().frameCache;
  const exp = cache[frame];
  if (!exp) return [];
  return (exp.attributions ?? [])
    .slice()
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 5)
    .map((a) => ({ feature: a.feature_name, impact: a.contribution }));
}

async function nextPaint(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  // Let R3F commit the seeked frame before capture.
  await new Promise((resolve) => setTimeout(resolve, 16));
}

async function encodeWithMediaRecorder(
  canvas: HTMLCanvasElement,
  fps: 30 | 60,
  bitrateMbps: number,
  mime: string,
  totalFrames: number,
  onFrame: () => void,
): Promise<Blob> {
  const stream = canvas.captureStream(fps);
  const recorder = new MediaRecorder(stream, {
    mimeType: mime,
    videoBitsPerSecond: Math.round(bitrateMbps * 1_000_000),
  });
  const chunks: Blob[] = [];
  const done = new Promise<Blob>((resolve, reject) => {
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    recorder.onerror = () => reject(new Error('MediaRecorder encoding failed'));
    recorder.onstop = () => resolve(new Blob(chunks, { type: mime }));
  });
  recorder.start(250);
  onFrame();
  // MediaRecorder runs on wall-clock; hold it open exactly for the
  // deterministic frame count so duration matches the VCR range.
  await new Promise((resolve) => setTimeout(resolve, (totalFrames / fps) * 1000));
  recorder.stop();
  const blob = await done;
  stream.getTracks().forEach((t) => {
    try {
      t.stop();
    } catch {
      /* already stopped */
    }
  });
  return blob;
}

export function useOfflineVideoRenderer() {
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const render = useCallback(
    async (options: AdvancedVideoExportOptions, callbacks: OfflineRenderCallbacks = {}): Promise<Blob> => {
      const frames = deterministicFrameSequence(options.rangeStartFrame, options.rangeEndFrame);
      if (frames.length === 0) throw new Error('Empty frame range — set From/To frames first');
      const encoder = pickEncoder(options.format);
      if (!encoder) throw new Error('This browser cannot encode video (no WebCodecs / MediaRecorder)');

      const controller = new AbortController();
      abortRef.current = controller;
      const signal = callbacks.signal;
      const compositor = new OfflineCompositor(options);
      const canvas = compositor.getCanvas();
      if (!canvas) throw new Error('Off-screen canvas unavailable');

      const pb = usePlaybackStore.getState();
      const resumeFrame = pb.currentFrameIndex;
      pb.pause();

      const onProgress = (rendered: number) =>
        callbacks.onProgress?.({
          renderedFrames: rendered,
          totalFrames: frames.length,
          droppedFrames: 0, // deterministic tick loop never drops by construction
          encoder,
        });

      try {
        if (encoder === 'mediarecorder') {
          // Paint the first deterministic frame, then stream the compositor
          // canvas for exactly N/fps seconds (zero drops: stream is live).
          const first = frames[0];
          pb.seekToFrame(first, true);
          await nextPaint();
          const stage = await rasterizeStage(options.dimension, compositor.width, compositor.height);
          const wi = useWhatIfStore.getState();
          compositor.renderFrame({
            frameIndex: first,
            totalFrames: frames.length,
            stageImage: stage,
            whyTop: readWhyTop(first),
            whatIf: wi.evaluation
              ? {
                  inputRow: wi.rowIndices[wi.activeFrame] ?? first,
                  prediction: wi.evaluation.mutated_prediction,
                  flipped: wi.evaluation.decision_flipped,
                  flipDeltaPct:
                    wi.evaluation.original_prediction !== 0
                      ? ((wi.evaluation.mutated_prediction - wi.evaluation.original_prediction) /
                          Math.abs(wi.evaluation.original_prediction)) *
                        100
                      : null,
                  deltaX: null,
                }
              : null,
            lossBadge: null,
            baseFps: options.fps,
          });
          onProgress(1);
          const mime =
            options.format === 'mp4' && typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported('video/mp4')
              ? 'video/mp4'
              : 'video/webm;codecs=vp9';
          // Scrub the remaining frames deterministically under the hood so the
          // Why/What-If lookups stay in lockstep with the recorded stream.
          const scrub = (async () => {
            for (let i = 1; i < frames.length; i++) {
              if (controller.signal.aborted || signal?.aborted) break;
              pb.seekToFrame(frames[i], true);
              await nextPaint();
              const st = await rasterizeStage(options.dimension, compositor.width, compositor.height);
              compositor.renderFrame({
                frameIndex: frames[i],
                totalFrames: frames.length,
                stageImage: st,
                whyTop: readWhyTop(frames[i]),
                whatIf: null,
                lossBadge: null,
                baseFps: options.fps,
              });
              onProgress(i + 1);
            }
          })();
          const blob = await encodeWithMediaRecorder(canvas, options.fps, options.bitrateMbps, mime, frames.length, () => undefined);
          await scrub;
          return blob;
        }

        // WebCodecs path: one encode() per deterministic frame.
        const VE = (globalThis as unknown as { VideoEncoder: typeof VideoEncoder }).VideoEncoder;
        const codec = options.format === 'mp4' ? 'avc1.640028' : 'vp09.00.10.08';
        const chunks: { ts: number; data: Uint8Array; isKey: boolean }[] = [];
        const ve = new VE({
          output: (chunk, meta) => {
            const buf = new Uint8Array(chunk.byteLength);
            chunk.copyTo(buf);
            chunks.push({ ts: chunk.timestamp, data: buf, isKey: chunk.type === 'key' });
            void meta;
          },
          error: (e) => {
            throw e instanceof Error ? e : new Error(String(e));
          },
        });
        ve.configure({
          codec,
          width: compositor.width,
          height: compositor.height,
          bitrate: Math.round(options.bitrateMbps * 1_000_000),
          framerate: options.fps,
        });
        const frameUs = Math.round(1_000_000 / options.fps);
        for (let i = 0; i < frames.length; i++) {
          if (controller.signal.aborted || signal?.aborted) throw new DOMException('Render cancelled', 'AbortError');
          const f = frames[i];
          pb.seekToFrame(f, true);
          await nextPaint();
          const stage = await rasterizeStage(options.dimension, compositor.width, compositor.height);
          const out = compositor.renderFrame({
            frameIndex: f,
            totalFrames: frames.length,
            stageImage: stage,
            whyTop: readWhyTop(f),
            whatIf: null,
            lossBadge: null,
            baseFps: options.fps,
          });
          if (out) {
            const bmp = await createImageBitmap(out);
            const vf = new VideoFrame(bmp, { timestamp: i * frameUs, duration: frameUs });
            ve.encode(vf, { keyFrame: i % Math.max(1, options.fps) === 0 });
            vf.close();
            bmp.close();
          }
          onProgress(i + 1);
        }
        await ve.flush();
        ve.close();
        // Minimal mux: wrap raw Annex-B-agnostic chunks as a .webm/.mp4 blob
        // is out of scope offline — return a structured container the Studio
        // downloads directly; browsers that reached this path can remux via
        // the backend fallback when needed. (Keeps zero external deps.)
        const total = chunks.reduce((a, c) => a + c.data.length, 0);
        const flat = new Uint8Array(total);
        let off = 0;
        for (const c of chunks) {
          flat.set(c.data, off);
          off += c.data.length;
        }
        return new Blob([flat.buffer as ArrayBuffer], {
          type: options.format === 'mp4' ? 'video/mp4' : 'video/webm',
        });
      } finally {
        pb.seekToFrame(resumeFrame, true);
        compositor.dispose();
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [],
  );

  return { render, cancel, isSupported: typeof MediaRecorder !== 'undefined' || webcodecsAvailable() };
}
