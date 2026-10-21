import { useCallback, useEffect, useRef, useState } from 'react';
import { useExportStore } from '../store/useExportStore';

const MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4',
];

/** First MIME type the browser's MediaRecorder supports (vp9 → vp8 → webm → mp4). */
export function getSupportedMimeType(): string {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return '';
  }
  return MIME_CANDIDATES.find((mime) => MediaRecorder.isTypeSupported(mime)) ?? '';
}

/** 120s guard so a forgotten recording can never grow the chunk buffer past ~250MB. */
export const MAX_RECORDING_SECONDS = 120;

export interface StartRecordingOptions {
  fps?: 30 | 60;
  bitrateMbps?: number;
  hudOverlay?: boolean;
  hudLineProvider?: () => string;
}

/**
 * Phase 16 — offline 60FPS WebGL video recorder.
 *
 * Records the raw WebGL `<canvas>` via `captureStream()` + `MediaRecorder`
 * (100% offline, no ffmpeg.wasm). With `hudOverlay`, frames are composited
 * through a 2D canvas that burns the telemetry line into the pixels, so the
 * HUD survives inside the saved video (HTML overlays never do).
 */
export function useWebGLVideoRecorder() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const compositeRafRef = useRef<number>(0);
  const compositeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number>(0);
  const [isRecording, setIsRecording] = useState(false);
  const [mimeType, setMimeType] = useState('');
  const isRecordingRef = useRef(false);

  const cleanupTracks = useCallback(() => {
    if (compositeRafRef.current) {
      cancelAnimationFrame(compositeRafRef.current);
      compositeRafRef.current = 0;
    }
    compositeCanvasRef.current = null;
    streamRef.current?.getTracks().forEach((t) => {
      try {
        t.stop();
      } catch {
        /* already stopped */
      }
    });
    streamRef.current = null;
  }, []);

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      try {
        recorder.stop();
      } catch {
        /* already stopped */
      }
    } else {
      cleanupTracks();
      recordedChunksRef.current = [];
      isRecordingRef.current = false;
      setIsRecording(false);
      useExportStore.getState().setRecording(false);
    }
  }, [cleanupTracks]);

  // Edge #5: unmount mid-recording must stop tracks + recorder, never leak.
  useEffect(
    () => () => {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        try {
          recorder.stop();
        } catch {
          /* ignore during teardown */
        }
      }
      if (timerRef.current) window.clearInterval(timerRef.current);
      cleanupTracks();
    },
    [cleanupTracks],
  );

  const startRecording = useCallback(
    (canvasElement: HTMLCanvasElement, options: StartRecordingOptions = {}) => {
      const { fps = 60, bitrateMbps = 16, hudOverlay = false, hudLineProvider } = options;
      const mime = getSupportedMimeType();
      if (!mime) {
        const msg = 'This browser cannot record video (MediaRecorder API unsupported)';
        useExportStore.getState().setRecordingError(msg);
        throw new Error(msg);
      }
      if (isRecordingRef.current) return;

      recordedChunksRef.current = [];
      let sourceCanvas: HTMLCanvasElement = canvasElement;

      if (hudOverlay) {
        // Composite canvas: burns the telemetry line into the recorded pixels.
        const composite = document.createElement('canvas');
        composite.width = canvasElement.width || canvasElement.clientWidth || 1280;
        composite.height = canvasElement.height || canvasElement.clientHeight || 720;
        const ctx = composite.getContext('2d');
        if (ctx) {
          sourceCanvas = composite;
          compositeCanvasRef.current = composite;
          const paint = () => {
            if (!isRecordingRef.current) return;
            try {
              ctx.drawImage(canvasElement, 0, 0, composite.width, composite.height);
            } catch {
              /* canvas not ready yet */
            }
            const line = hudLineProvider?.() ?? '';
            if (line) {
              const barH = 34;
              ctx.fillStyle = 'rgba(8,12,18,0.78)';
              ctx.fillRect(0, composite.height - barH, composite.width, barH);
              ctx.fillStyle = '#e2e8f0';
              ctx.font = '13px "JetBrains Mono", monospace';
              ctx.fillText(line.slice(0, 160), 12, composite.height - 12);
            }
            compositeRafRef.current = requestAnimationFrame(paint);
          };
          isRecordingRef.current = true;
          paint();
        }
      }

      const stream = sourceCanvas.captureStream(fps);
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: Math.round(bitrateMbps * 1_000_000),
      });

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        if (compositeRafRef.current) {
          cancelAnimationFrame(compositeRafRef.current);
          compositeRafRef.current = 0;
        }
        compositeCanvasRef.current = null;
        const ext = mime.includes('mp4') ? 'mp4' : 'webm';
        const blob = new Blob(recordedChunksRef.current, { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `easy-ml-3d-recording-${Date.now()}.${ext}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
        cleanupTracks();
        if (timerRef.current) {
          window.clearInterval(timerRef.current);
          timerRef.current = 0;
        }
        recordedChunksRef.current = [];
        mediaRecorderRef.current = null;
        isRecordingRef.current = false;
        setIsRecording(false);
        useExportStore.getState().setRecording(false);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(250);
      if (!hudOverlay) isRecordingRef.current = true;
      setMimeType(mime);
      setIsRecording(true);
      useExportStore.getState().setRecording(true, mime);

      // Elapsed timer + 120s max-duration guard.
      const startedAt = Date.now();
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startedAt) / 1000;
        useExportStore.getState().setRecordingElapsed(elapsed);
        if (elapsed >= MAX_RECORDING_SECONDS) stopRecording();
      }, 250);
    },
    [cleanupTracks, stopRecording],
  );

  return { isRecording, mimeType, startRecording, stopRecording };
}
