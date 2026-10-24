/**
 * Phase 57 — offline compositor math (pure, no DOM).
 * Mirrors `backend/app/domains/export_reporting/video_export_service.py`
 * so the frontend plan preview never needs a network round-trip.
 */
import type {
  AdvancedVideoExportOptions,
  CompositorLayout,
  VideoResolution,
  VideoTheme,
  VideoThemeTokens,
} from '../types/advancedVideoExport.types';

export const RESOLUTION_TABLE: Record<VideoResolution, { w: number; h: number }> = {
  '1080p': { w: 1920, h: 1080 },
  '720p': { w: 1280, h: 720 },
  square_1080: { w: 1080, h: 1080 },
};

const THEME_TABLE: Record<VideoTheme, VideoThemeTokens> = {
  dark: {
    background: '#0b1220',
    grid: '#334155',
    hudText: '#e2e8f0',
    hudPanel: 'rgba(8,12,18,0.78)',
    accentLoss: '#f43f5e',
    ambientBoost: 1.0,
    directionalBoost: 1.0,
  },
  light: {
    background: '#f8fafc',
    grid: '#cbd5e1',
    hudText: '#0f172a',
    hudPanel: 'rgba(248,250,252,0.85)',
    accentLoss: '#dc2626',
    ambientBoost: 1.4,
    directionalBoost: 1.2,
  },
  match_ui: {
    background: '#0b1220',
    grid: '#334155',
    hudText: '#e2e8f0',
    hudPanel: 'rgba(8,12,18,0.78)',
    accentLoss: '#f43f5e',
    ambientBoost: 1.0,
    directionalBoost: 1.0,
  },
};

export function themeTokens(theme: VideoTheme): VideoThemeTokens {
  return { ...THEME_TABLE[theme] };
}

export function compositorLayout(
  width: number,
  height: number,
  layout: AdvancedVideoExportOptions['layout'],
): CompositorLayout {
  if (layout === 'split_studio') {
    const stageW = Math.round(width * 0.7);
    return {
      stage: { x: 0, y: 0, w: stageW, h: height },
      telemetry: { x: stageW, y: 0, w: width - stageW, h: height },
      hudBar: { x: 0, y: height - 36, w: stageW, h: 36 },
    };
  }
  if (layout === 'pip') {
    const pipW = Math.round(width * 0.27);
    const pipH = Math.round(height * 0.33);
    const margin = Math.round(width * 0.02);
    return {
      stage: { x: 0, y: 0, w: width, h: height },
      telemetry: { x: width - pipW - margin, y: height - pipH - margin - 36, w: pipW, h: pipH },
      hudBar: { x: 0, y: height - 36, w: width, h: 36 },
    };
  }
  return {
    stage: { x: 0, y: 0, w: width, h: height },
    telemetry: { x: 0, y: 0, w: 0, h: 0 },
    hudBar: { x: 0, y: height - 36, w: width, h: 36 },
  };
}

/** Total frames in [start..end]; clamps negatives to 0. */
export function frameCount(start: number, end: number): number {
  const s = Math.max(0, Math.floor(start));
  const e = Math.max(0, Math.floor(end));
  if (e < s) return 0;
  return e - s + 1;
}

export function durationSec(nFrames: number, fps: 30 | 60): number {
  if (nFrames <= 0) return 0;
  return nFrames / fps;
}

export function estimatedDurationLabel(nFrames: number, fps: 30 | 60): string {
  return `${durationSec(nFrames, fps).toFixed(1)}s (${nFrames} frames @ ${fps}fps)`;
}

/** DoD #5 helper: deterministic tick sequence — zero drops by construction. */
export function deterministicFrameSequence(start: number, end: number): number[] {
  const n = frameCount(start, end);
  const s = Math.max(0, Math.floor(start));
  return Array.from({ length: n }, (_, i) => s + i);
}

export function validateAdvancedOptions(o: AdvancedVideoExportOptions): string | null {
  if (o.rangeStartFrame < 0 || o.rangeEndFrame < 0) return 'Frame range must be >= 0';
  if (o.rangeEndFrame < o.rangeStartFrame) return 'End frame must be >= start frame';
  const n = frameCount(o.rangeStartFrame, o.rangeEndFrame);
  if (n > 10_000) return 'At most 10,000 frames per export';
  if (!(o.bitrateMbps >= 1 && o.bitrateMbps <= 40)) return 'Bitrate must be within 1–40 Mbps';
  if (![0.25, 0.5, 1.0, 2.0, 4.0].includes(o.speedMultiplier)) return 'Unsupported speed multiplier';
  return null;
}
