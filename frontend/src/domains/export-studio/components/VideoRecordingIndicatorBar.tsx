import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useExportStore } from '../store/useExportStore';
import { requestStopRecording } from '../hooks/videoRecorderControl';

/**
 * Phase 16 — floating video-recording HUD bar (spec §2.2).
 * Blinking red REC pill with elapsed time + playhead frame, plus the
 * one-click Stop & Save action. Rendered only while recording.
 */
export function VideoRecordingIndicatorBar() {
  const isRecording = useExportStore((s) => s.isRecording);
  const elapsed = useExportStore((s) => s.recordingElapsedSec);
  const mime = useExportStore((s) => s.recordingMime);
  const frame = usePlaybackStore((s) => s.currentFrameIndex);
  const total = usePlaybackStore((s) => s.totalFrames);

  if (!isRecording) return null;

  const ext = mime.includes('mp4') ? 'mp4' : 'webm';
  const mm = Math.floor(elapsed / 60);
  const ss = (elapsed % 60).toFixed(1).padStart(4, '0');

  return (
    <div
      className="pointer-events-auto absolute left-1/2 top-3 z-40 flex -translate-x-1/2 items-center gap-3 rounded-full px-4 py-1.5 text-xs font-medium"
      style={{ background: 'rgba(10,14,20,0.88)', border: '1px solid rgba(239,68,68,0.7)', color: '#fff' }}
      role="status"
      aria-label="Video recording in progress"
    >
      <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full" style={{ background: '#ef4444' }} aria-hidden />
      <span className="font-mono-tabular">
        🔴 REC {mm}:{ss} (Frame {String(frame).padStart(4, '0')}/{String(Math.max(total - 1, 0)).padStart(4, '0')}) · {ext}
      </span>
      <button
        onClick={() => requestStopRecording()}
        className="rounded-full px-3 py-1 text-[11px] font-semibold"
        style={{ background: '#dc2626', color: '#fff' }}
      >
        ⏹ Stop &amp; Save Video
      </button>
    </div>
  );
}
