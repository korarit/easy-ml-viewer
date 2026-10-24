/** Phase 57 — Advanced Video Export Studio option contract (spec §3). */

export type VideoDimension = '3d' | '2d';
export type VideoLayout = 'split_studio' | 'pip' | 'pure_stage';
export type VideoTheme = 'match_ui' | 'dark' | 'light';
export type VideoCameraPreset = 'isometric' | 'top_xy' | 'front_xz' | 'side_yz' | 'follow_ball';
export type VideoFormat = 'mp4' | 'webm';
export type VideoResolution = '1080p' | '720p' | 'square_1080';

export interface AdvancedVideoExportOptions {
  dimension: VideoDimension;
  layout: VideoLayout;
  includeWhyPanel: boolean;
  includeWhatIfPanel: boolean;
  theme: VideoTheme;
  cameraPreset: VideoCameraPreset;
  speedMultiplier: number; // 0.25, 0.5, 1.0, 2.0, 4.0
  fps: 30 | 60;
  format: VideoFormat;
  resolution: VideoResolution;
  bitrateMbps: number;
  rangeStartFrame: number;
  rangeEndFrame: number;
}

export interface VideoThemeTokens {
  background: string;
  grid: string;
  hudText: string;
  hudPanel: string;
  accentLoss: string;
  ambientBoost: number;
  directionalBoost: number;
}

export interface LayoutRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface CompositorLayout {
  stage: LayoutRect;
  telemetry: LayoutRect;
  hudBar: LayoutRect;
}

export interface FrameWhyDatum {
  feature: string;
  impact: number;
}

export interface FrameWhatIfDatum {
  inputRow: number;
  prediction: number;
  flipped: boolean;
  flipDeltaPct: number | null;
  deltaX: number[] | null;
}

export interface OfflineRenderProgress {
  renderedFrames: number;
  totalFrames: number;
  droppedFrames: number;
  encoder: 'webcodecs-h264' | 'webcodecs-vp9' | 'mediarecorder' | 'frame-png-sequence';
}

export const DEFAULT_ADVANCED_VIDEO_OPTIONS: AdvancedVideoExportOptions = {
  dimension: '3d',
  layout: 'split_studio',
  includeWhyPanel: true,
  includeWhatIfPanel: true,
  theme: 'dark',
  cameraPreset: 'isometric',
  speedMultiplier: 1.0,
  fps: 60,
  format: 'mp4',
  resolution: '1080p',
  bitrateMbps: 12,
  rangeStartFrame: 0,
  rangeEndFrame: 0,
};

export const VIDEO_SPEEDS: readonly number[] = [0.25, 0.5, 1.0, 2.0, 4.0] as const;
