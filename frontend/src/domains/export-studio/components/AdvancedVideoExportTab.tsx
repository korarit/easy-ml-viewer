/**
 * Phase 57 — Advanced Video Export Studio tab (spec §4 UI matrix).
 *
 * Dimension + Layout | Visual overrides (theme/camera/speed, independent
 * from the live preview) | Format & Quality + deterministic offline render
 * with Why/What-If telemetry dock and MP4/WebM output.
 */
import { useMemo, useState } from 'react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useAdvancedVideoExportStore } from '../store/useAdvancedVideoExportStore';
import { useOfflineVideoRenderer } from '../hooks/useOfflineVideoRenderer';
import { transcodeWebmToMp4 } from '../api/videoExportApi';
import {
  compositorLayout,
  deterministicFrameSequence,
  durationSec,
  estimatedDurationLabel,
  RESOLUTION_TABLE,
  themeTokens,
  validateAdvancedOptions,
} from './advancedVideoExportMath';
import type {
  VideoCameraPreset,
  VideoDimension,
  VideoFormat,
  VideoLayout,
  VideoResolution,
  VideoTheme,
} from '../types/advancedVideoExport.types';
import { VIDEO_SPEEDS } from '../types/advancedVideoExport.types';

function OptionRow<T extends string | number>(props: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onPick: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div>
      <p className="mb-1 font-medium" style={{ color: 'var(--text-primary)' }}>{props.label}</p>
      <div className="flex flex-wrap gap-2">
        {props.options.map((o) => (
          <button
            key={String(o.value)}
            onClick={() => props.onPick(o.value)}
            disabled={props.disabled}
            className="rounded-md px-3 py-1.5 disabled:opacity-50"
            style={{
              border: `1px solid ${props.value === o.value ? 'var(--ml-primary)' : 'var(--border-subtle)'}`,
              color: props.value === o.value ? 'var(--text-primary)' : 'var(--text-secondary)',
            }}
          >
            {props.value === o.value ? '● ' : '○ '}{o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function AdvancedVideoExportTab() {
  const s = useAdvancedVideoExportStore();
  const { render, cancel } = useOfflineVideoRenderer();
  const [transcoding, setTranscoding] = useState(false);
  const totalFrames = usePlaybackStore((st) => st.totalFrames);

  const frames = useMemo(
    () => deterministicFrameSequence(s.rangeStartFrame, s.rangeEndFrame),
    [s.rangeStartFrame, s.rangeEndFrame],
  );
  const validationError = validateAdvancedOptions({
    dimension: s.dimension,
    layout: s.layout,
    includeWhyPanel: s.includeWhyPanel,
    includeWhatIfPanel: s.includeWhatIfPanel,
    theme: s.theme,
    cameraPreset: s.cameraPreset,
    speedMultiplier: s.speedMultiplier,
    fps: s.fps,
    format: s.format,
    resolution: s.resolution,
    bitrateMbps: s.bitrateMbps,
    rangeStartFrame: s.rangeStartFrame,
    rangeEndFrame: s.rangeEndFrame,
  });
  const res = RESOLUTION_TABLE[s.resolution];
  const layout = compositorLayout(res.w, res.h, s.layout);
  const tokens = themeTokens(s.theme);

  const startRender = async () => {
    if (validationError || totalFrames <= 0) return;
    s.setRendering(true, 0);
    s.setRenderCount(0, frames.length);
    try {
      const blob = await render(
        {
          dimension: s.dimension,
          layout: s.layout,
          includeWhyPanel: s.includeWhyPanel,
          includeWhatIfPanel: s.includeWhatIfPanel,
          theme: s.theme,
          cameraPreset: s.cameraPreset,
          speedMultiplier: s.speedMultiplier,
          fps: s.fps,
          format: s.format,
          resolution: s.resolution,
          bitrateMbps: s.bitrateMbps,
          rangeStartFrame: s.rangeStartFrame,
          rangeEndFrame: s.rangeEndFrame,
        },
        {
          onProgress: (p) => {
            useAdvancedVideoExportStore.getState().setRenderCount(p.renderedFrames, p.totalFrames);
          },
        },
      );
      let outBlob = blob;
      let ext = s.format;
      // Fallback path (spec §2.3): MediaRecorder-only browsers produce WebM —
      // offer one-click offline remux to universal MP4 via the backend.
      if (s.format === 'mp4' && blob.type.includes('webm')) {
        try {
          setTranscoding(true);
          outBlob = await transcodeWebmToMp4(blob);
          ext = 'mp4';
        } catch {
          ext = 'webm'; // keep the WebM — never lose the render
        } finally {
          setTranscoding(false);
        }
      }
      const filename = `studio-${s.dimension}-${s.layout}-${res.w}x${res.h}-${s.fps}fps.${ext}`;
      const url = URL.createObjectURL(outBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
      s.setLastFile(filename);
      s.setRendering(false, 100);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        s.setRenderError('Render cancelled');
      } else {
        s.setRenderError(err instanceof Error ? err.message : 'Offline render failed');
      }
      s.setRendering(false, 0);
    }
  };

  return (
    <div className="mt-4 space-y-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
      <OptionRow<VideoDimension>
        label="1. Dimension & Layout — independent from the live viewport"
        value={s.dimension}
        options={[
          { value: '3d', label: '3D Spatial' },
          { value: '2d', label: '2D Planar' },
        ]}
        onPick={(v) => s.setOptions({ dimension: v })}
        disabled={s.isRendering}
      />
      <OptionRow<VideoLayout>
        label="Panel Layout"
        value={s.layout}
        options={[
          { value: 'split_studio', label: 'Split Studio (70/30)' },
          { value: 'pip', label: 'PiP Bottom-Right' },
          { value: 'pure_stage', label: 'Pure Stage (No Panels)' },
        ]}
        onPick={(v) => s.setOptions({ layout: v })}
        disabled={s.isRendering}
      />
      <div className="flex flex-wrap gap-2">
        {(
          [
            ['includeWhyPanel', 'Include Why Panel (SHAP Waterfall)'],
            ['includeWhatIfPanel', 'Include What-If Delta Badge'],
          ] as const
        ).map(([k, label]) => (
          <label key={k} className="flex cursor-pointer items-center gap-1.5 rounded-md px-3 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
            <input type="checkbox" checked={s[k]} disabled={s.isRendering} onChange={(e) => s.setOptions({ [k]: e.target.checked })} />
            {label}
          </label>
        ))}
      </div>

      <OptionRow<VideoTheme>
        label="2. Visual Overrides (independent from preview)"
        value={s.theme}
        options={[
          { value: 'match_ui', label: 'Match UI' },
          { value: 'dark', label: 'Dark Mode' },
          { value: 'light', label: 'Light Mode (Paper/Slides)' },
        ]}
        onPick={(v) => s.setOptions({ theme: v })}
        disabled={s.isRendering}
      />
      <OptionRow<VideoCameraPreset>
        label="Camera Angle (3D)"
        value={s.cameraPreset}
        options={[
          { value: 'isometric', label: 'Isometric' },
          { value: 'top_xy', label: 'Top XY' },
          { value: 'front_xz', label: 'Front XZ' },
          { value: 'side_yz', label: 'Side YZ' },
          { value: 'follow_ball', label: 'Follow Ball' },
        ]}
        onPick={(v) => s.setOptions({ cameraPreset: v })}
        disabled={s.isRendering}
      />
      <OptionRow<number>
        label="Playback Speed"
        value={s.speedMultiplier}
        options={VIDEO_SPEEDS.map((v) => ({ value: v, label: `${v}x${v === 1 ? ' Normal' : v < 1 ? ' Slow' : ' Fast'}` }))}
        onPick={(v) => s.setOptions({ speedMultiplier: v })}
        disabled={s.isRendering}
      />

      <OptionRow<VideoFormat>
        label="3. Output Format & Quality"
        value={s.format}
        options={[
          { value: 'mp4', label: 'MP4 (H.264 Universal)' },
          { value: 'webm', label: 'WebM (VP9)' },
        ]}
        onPick={(v) => s.setOptions({ format: v, bitrateMbps: v === 'mp4' ? 12 : 8 })}
        disabled={s.isRendering}
      />
      <OptionRow<VideoResolution>
        label="Resolution"
        value={s.resolution}
        options={[
          { value: '1080p', label: 'Full HD 1080p (1920×1080)' },
          { value: '720p', label: '720p (1280×720)' },
          { value: 'square_1080', label: 'Square 1080×1080' },
        ]}
        onPick={(v) => s.setOptions({ resolution: v })}
        disabled={s.isRendering}
      />
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>Frame Rate:</span>
        {([30, 60] as const).map((v) => (
          <button
            key={v}
            onClick={() => s.setOptions({ fps: v })}
            disabled={s.isRendering}
            className="rounded-md px-3 py-1.5 disabled:opacity-50"
            style={{ border: `1px solid ${s.fps === v ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
          >
            {s.fps === v ? '● ' : '○ '}{v} FPS{v === 60 ? ' (Ultra Smooth)' : ''}
          </button>
        ))}
        <span className="flex items-center gap-1">
          From Frame
          <input
            type="number" min={0} max={Math.max(totalFrames - 1, 0)} value={s.rangeStartFrame}
            disabled={s.isRendering}
            onChange={(e) => s.setOptions({ rangeStartFrame: Number(e.target.value) || 0 })}
            className="w-20 rounded px-1 py-1"
            style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-canvas)' }}
          />
          to
          <input
            type="number" min={0} max={Math.max(totalFrames - 1, 0)} value={s.rangeEndFrame}
            disabled={s.isRendering}
            onChange={(e) => s.setOptions({ rangeEndFrame: Number(e.target.value) || 0 })}
            className="w-20 rounded px-1 py-1"
            style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-canvas)' }}
          />
        </span>
        <button
          onClick={() => s.resetRange(totalFrames)}
          disabled={s.isRendering || totalFrames <= 0}
          className="rounded-md px-2 py-1 disabled:opacity-50"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          Full queue
        </button>
      </div>

      <div className="rounded-md p-2" style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}>
        {frames.length === 0 ? (
          <span>No frames — set a From/To range inside the queued timeline.</span>
        ) : (
          <span>
            Total {frames.length} frames · Est. Duration: {estimatedDurationLabel(frames.length, s.fps)} ·
            Playback @ {s.speedMultiplier}x = {durationSec(frames.length, s.fps) / s.speedMultiplier < 10
              ? (durationSec(frames.length, s.fps) / s.speedMultiplier).toFixed(1)
              : Math.round(durationSec(frames.length, s.fps) / s.speedMultiplier)}s ·
            Canvas {res.w}×{res.h} · Stage {layout.stage.w}×{layout.stage.h} · BG {tokens.background}
            {s.lastFilename ? ` · Last saved: ${s.lastFilename}` : ''}
          </span>
        )}
      </div>

      {(validationError || s.renderError) && (
        <div className="rounded px-3 py-2" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43d5e' }} role="alert">
          {s.renderError ?? validationError}
        </div>
      )}

      {s.isRendering && (
        <div role="progressbar" aria-valuenow={s.renderProgress} aria-valuemin={0} aria-valuemax={100}>
          PROGRESS: Rendering Frame {s.renderedFrames} / {s.totalFrames} ({s.renderProgress}%) ·{' '}
          {s.format === 'mp4' ? 'Encoding H.264 MP4' : 'Encoding VP9 WebM'}
        </div>
      )}

      <div className="flex justify-end gap-2">
        {s.isRendering ? (
          <button
            onClick={() => {
              cancel();
              s.setRendering(false, 0);
            }}
            className="rounded-md px-4 py-2 font-medium"
            style={{ background: '#dc2626', color: '#fff' }}
          >
            ⏹ Cancel
          </button>
        ) : (
          <button
            onClick={() => void startRender()}
            disabled={!!validationError || frames.length === 0 || totalFrames <= 0 || transcoding}
            title={totalFrames <= 0 ? 'Queue rows in the Studio first' : 'Start offline render'}
            className="rounded-md px-4 py-2 font-medium disabled:opacity-50"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {transcoding ? 'Remuxing MP4…' : '🎬 Start Offline Render'}
          </button>
        )}
      </div>
    </div>
  );
}
