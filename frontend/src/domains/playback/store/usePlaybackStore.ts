import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { LoopMode, PlaybackStatus } from '../types/playback.types';

export type { LoopMode, PlaybackStatus };

/**
 * Phase 10 — high-frequency VCR playback store.
 *
 * 60FPS rule: 3D components MUST NOT subscribe to `exactFrameFloat`
 * (that would re-render React 60x/sec). Read it imperatively via
 * `usePlaybackStore.getState()` inside R3F `useFrame()` and mutate
 * three.js objects directly. Panels subscribe only to
 * `currentFrameIndex` (integer), so sub-frame interpolation
 * (5.01 … 5.99) causes zero React re-renders.
 */
export interface PlaybackStoreState {
  status: PlaybackStatus;
  totalFrames: number;
  currentFrameIndex: number;
  exactFrameFloat: number;
  baseFps: number;
  speedMultiplier: number;
  loopMode: LoopMode;
  loopRange: [number, number] | null;
  trailLength: number;
  smoothInterpolation: boolean;
  wasPlayingBeforeScrub: boolean;
  trailResetCounter: number;

  initializeQueue: (totalFrames: number) => void;
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  stepFrame: (delta: number) => void;
  seekToFrame: (frameIndex: number, isJump?: boolean) => void;
  advanceByDeltaMs: (deltaMs: number) => void;
  startScrubbing: () => void;
  endScrubbing: () => void;
  setLoopRangePoint: (point: 'A' | 'B', frameIndex: number) => void;
  setSpeedMultiplier: (multiplier: number) => void;
  setLoopMode: (mode: LoopMode) => void;
  cycleLoopMode: () => void;
  setLoopRange: (range: [number, number] | null) => void;
  clearLoopRange: () => void;
  setTrailLength: (length: number) => void;
  setSmoothInterpolation: (enabled: boolean) => void;
  setBaseFps: (fps: number) => void;
}

export const LOOP_MODE_CYCLE: LoopMode[] = ['none', 'all', 'ab_range'];

export const usePlaybackStore = create<PlaybackStoreState>()(
  subscribeWithSelector((set, get) => ({
    status: 'idle',
    totalFrames: 0,
    currentFrameIndex: 0,
    exactFrameFloat: 0,
    baseFps: 5,
    speedMultiplier: 1.0,
    loopMode: 'all',
    loopRange: null,
    trailLength: 25,
    smoothInterpolation: true,
    wasPlayingBeforeScrub: false,
    trailResetCounter: 0,

    initializeQueue: (totalFrames) =>
      set({
        totalFrames: Math.max(0, Math.floor(totalFrames)),
        currentFrameIndex: 0,
        exactFrameFloat: 0,
        status: totalFrames > 0 ? 'paused' : 'idle',
        loopRange: totalFrames > 1 ? [0, totalFrames - 1] : null,
      }),

    play: () => {
      const { totalFrames, currentFrameIndex } = get();
      if (totalFrames <= 1) return;
      if (currentFrameIndex >= totalFrames - 1) {
        set({ currentFrameIndex: 0, exactFrameFloat: 0, status: 'playing', trailResetCounter: get().trailResetCounter + 1 });
      } else {
        set({ status: 'playing' });
      }
    },

    pause: () => {
      if (get().status === 'playing') set({ status: 'paused' });
    },

    togglePlayPause: () => {
      const { status, play, pause } = get();
      if (status === 'playing') pause();
      else play();
    },

    stepFrame: (delta) => {
      const { totalFrames, currentFrameIndex, trailResetCounter } = get();
      if (totalFrames <= 1) return;
      const next = Math.max(0, Math.min(totalFrames - 1, currentFrameIndex + delta));
      const isLargeJump = Math.abs(next - currentFrameIndex) > 5;
      set({
        status: 'paused',
        currentFrameIndex: next,
        exactFrameFloat: next,
        trailResetCounter: isLargeJump ? trailResetCounter + 1 : trailResetCounter,
      });
    },

    seekToFrame: (frameIndex, isJump = true) => {
      const { totalFrames, currentFrameIndex, trailResetCounter } = get();
      if (totalFrames <= 0) return;
      const clamped = Math.max(0, Math.min(totalFrames - 1, Math.round(frameIndex)));
      const isFar = isJump && Math.abs(clamped - currentFrameIndex) > 5;
      set({
        currentFrameIndex: clamped,
        exactFrameFloat: clamped,
        trailResetCounter: isFar ? trailResetCounter + 1 : trailResetCounter,
      });
    },

    advanceByDeltaMs: (deltaMs) => {
      const { status, totalFrames, exactFrameFloat, baseFps, speedMultiplier, loopMode, loopRange, trailResetCounter } = get();
      if (status !== 'playing' || totalFrames <= 1) return;

      // Page Visibility guard: clamp to 100ms so tab-switch gaps never jump.
      const safeDeltaSec = Math.min(Math.max(deltaMs, 0), 100) / 1000.0;
      const frameAdvance = safeDeltaSec * baseFps * speedMultiplier;
      let nextFloat = exactFrameFloat + frameAdvance;

      const minBound = loopMode === 'ab_range' && loopRange ? loopRange[0] : 0;
      const maxBound = loopMode === 'ab_range' && loopRange ? loopRange[1] : totalFrames - 1;

      if (nextFloat >= maxBound) {
        if (loopMode === 'none') {
          // Single set() call keeps integer subscribers to one notification.
          set((s) =>
            s.currentFrameIndex === maxBound && s.exactFrameFloat === maxBound && s.status === 'paused'
              ? s
              : { status: 'paused', currentFrameIndex: maxBound, exactFrameFloat: maxBound },
          );
          return;
        }
        nextFloat = minBound;
        set({
          exactFrameFloat: nextFloat,
          currentFrameIndex: Math.floor(nextFloat),
          trailResetCounter: trailResetCounter + 1,
        });
        return;
      }

      const nextInt = Math.floor(nextFloat);
      // Zero-sub-frame-re-render: when the integer frame is unchanged,
      // keep the same `currentFrameIndex` reference so selector subscribers
      // (s => s.currentFrameIndex) never re-fire mid-frame.
      set((s) =>
        s.exactFrameFloat === nextFloat
          ? s
          : { exactFrameFloat: nextFloat, currentFrameIndex: nextInt },
      );
    },

    startScrubbing: () => {
      const { status } = get();
      if (status === 'scrubbing') return;
      set({ status: 'scrubbing', wasPlayingBeforeScrub: status === 'playing' });
    },

    endScrubbing: () => {
      const { wasPlayingBeforeScrub } = get();
      set({ status: wasPlayingBeforeScrub ? 'playing' : 'paused', wasPlayingBeforeScrub: false });
    },

    setLoopRangePoint: (point, frameIndex) => {
      const { totalFrames, loopRange } = get();
      if (totalFrames <= 1) return;
      const clampedIdx = Math.max(0, Math.min(totalFrames - 1, Math.round(frameIndex)));
      const currentA = loopRange ? loopRange[0] : 0;
      const currentB = loopRange ? loopRange[1] : Math.max(0, totalFrames - 1);
      if (point === 'A') {
        set({ loopRange: [Math.min(clampedIdx, currentB - 1), currentB], loopMode: 'ab_range' });
      } else {
        set({ loopRange: [currentA, Math.max(clampedIdx, currentA + 1)], loopMode: 'ab_range' });
      }
    },

    setSpeedMultiplier: (multiplier) => {
      if (!Number.isFinite(multiplier) || multiplier <= 0) return;
      set({ speedMultiplier: multiplier });
    },

    setLoopMode: (mode) => set({ loopMode: mode }),

    cycleLoopMode: () => {
      const { loopMode } = get();
      const next = LOOP_MODE_CYCLE[(LOOP_MODE_CYCLE.indexOf(loopMode) + 1) % LOOP_MODE_CYCLE.length];
      set({ loopMode: next });
    },

    setLoopRange: (range) => {
      const { totalFrames } = get();
      if (range === null) {
        set({ loopRange: null });
        return;
      }
      if (totalFrames <= 1) return;
      const a = Math.max(0, Math.min(totalFrames - 1, Math.round(range[0])));
      const b = Math.max(0, Math.min(totalFrames - 1, Math.round(range[1])));
      set({ loopRange: [Math.min(a, b), Math.max(a, b)] });
    },

    clearLoopRange: () => set({ loopRange: null, loopMode: 'all' }),

    setTrailLength: (length) => {
      if (!Number.isFinite(length) || length < 0) return;
      set({ trailLength: Math.floor(length) });
    },

    setSmoothInterpolation: (enabled) => set({ smoothInterpolation: enabled }),

    setBaseFps: (fps) => {
      if (!Number.isFinite(fps) || fps <= 0) return;
      set({ baseFps: fps });
    },
  })),
);
