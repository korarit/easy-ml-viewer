import { describe, it, expect } from 'vitest';
import {
  RESOLUTION_TABLE,
  compositorLayout,
  deterministicFrameSequence,
  durationSec,
  frameCount,
  themeTokens,
  validateAdvancedOptions,
} from '../components/advancedVideoExportMath';
import { DEFAULT_ADVANCED_VIDEO_OPTIONS } from '../types/advancedVideoExport.types';

describe('Phase 57 advanced video export math', () => {
  it('split-studio layout matches the spec 1344/576 geometry at 1080p', () => {
    const layout = compositorLayout(1920, 1080, 'split_studio');
    expect(layout.stage).toEqual({ x: 0, y: 0, w: 1344, h: 1080 });
    expect(layout.telemetry).toEqual({ x: 1344, y: 0, w: 576, h: 1080 });
  });

  it('pip telemetry floats inside the fullscreen stage', () => {
    const layout = compositorLayout(1920, 1080, 'pip');
    expect(layout.stage).toEqual({ x: 0, y: 0, w: 1920, h: 1080 });
    expect(layout.telemetry.w).toBeGreaterThan(0);
    expect(layout.telemetry.x + layout.telemetry.w).toBeLessThanOrEqual(1920);
    expect(layout.telemetry.y + layout.telemetry.h).toBeLessThanOrEqual(1080);
  });

  it('pure stage carries no telemetry dock', () => {
    expect(compositorLayout(1920, 1080, 'pure_stage').telemetry.w).toBe(0);
  });

  it('light theme uses the paper palette with boosted studio lights', () => {
    const tokens = themeTokens('light');
    expect(tokens.background).toBe('#f8fafc');
    expect(tokens.grid).toBe('#cbd5e1');
    expect(tokens.ambientBoost).toBe(1.4);
    expect(tokens.directionalBoost).toBe(1.2);
    expect(tokens.hudText).toBe('#0f172a');
  });

  it('200 frames at 60fps last ~3.33s with zero drops by construction', () => {
    const seq = deterministicFrameSequence(0, 199);
    expect(seq).toHaveLength(200);
    expect(seq[0]).toBe(0);
    expect(seq[199]).toBe(199);
    expect(new Set(seq).size).toBe(200); // no duplicates = no drops/duplicates
    expect(durationSec(seq.length, 60)).toBeCloseTo(200 / 60, 3);
  });

  it('resolution table covers 1080p, 720p and square', () => {
    expect(RESOLUTION_TABLE['1080p']).toEqual({ w: 1920, h: 1080 });
    expect(RESOLUTION_TABLE['720p']).toEqual({ w: 1280, h: 720 });
    expect(RESOLUTION_TABLE.square_1080).toEqual({ w: 1080, h: 1080 });
  });

  it('validates ranges, bitrate and speed multipliers', () => {
    expect(validateAdvancedOptions({ ...DEFAULT_ADVANCED_VIDEO_OPTIONS, rangeStartFrame: 0, rangeEndFrame: 199 })).toBeNull();
    expect(frameCount(10, 5)).toBe(0);
    expect(validateAdvancedOptions({ ...DEFAULT_ADVANCED_VIDEO_OPTIONS, rangeStartFrame: 10, rangeEndFrame: 5 })).not.toBeNull();
    expect(validateAdvancedOptions({ ...DEFAULT_ADVANCED_VIDEO_OPTIONS, bitrateMbps: 99 })).not.toBeNull();
    expect(validateAdvancedOptions({ ...DEFAULT_ADVANCED_VIDEO_OPTIONS, speedMultiplier: 3 })).not.toBeNull();
  });
});
