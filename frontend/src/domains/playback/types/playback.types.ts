/** Phase 10 — shared playback types (VCR studio). */

export type PlaybackStatus = 'idle' | 'playing' | 'paused' | 'scrubbing';
export type LoopMode = 'none' | 'all' | 'ab_range';

export interface FrameTrajectoryPoint {
  frame_index: number;
  dataset_row_index: number;
  position_world: [number, number, number];
  prediction: number;
  confidence: number;
  actual_target: number | null;
}

export interface PlaybackSequenceManifest {
  session_id: string;
  total_frames: number;
  default_fps: number;
  frames_summary: FrameTrajectoryPoint[];
  discontinuity_frames: number[];
}

export interface SpeedPreset {
  multiplier: number;
  label: string;
  fpsLabel: string;
}

export const SPEED_PRESETS: SpeedPreset[] = [
  { multiplier: 0.25, label: '0.25x', fpsLabel: '1 FPS' },
  { multiplier: 0.5, label: '0.5x', fpsLabel: '2 FPS' },
  { multiplier: 1.0, label: '1.0x', fpsLabel: '5 FPS' },
  { multiplier: 2.0, label: '2.0x', fpsLabel: '10 FPS' },
  { multiplier: 5.0, label: '5.0x', fpsLabel: '25 FPS' },
  { multiplier: 10.0, label: '10.0x', fpsLabel: '60 FPS' },
];

export const TRAIL_LENGTH_OPTIONS = [0, 10, 25, 50] as const;
