import { useEffect } from 'react';
import { usePlaybackStore } from '../store/usePlaybackStore';

/**
 * Phase 10 — high-precision `requestAnimationFrame` drive loop.
 *
 * Reads state imperatively via `getState()` and advances
 * `exactFrameFloat` by wall-clock delta — zero React re-renders per
 * tick. The store itself clamps `deltaMs` to 100ms (tab-switch guard),
 * and this hook additionally skips ticks while `document.hidden`.
 */
export function usePlaybackRafLoop() {
  useEffect(() => {
    let rafId = 0;
    let lastTs = 0;
    let running = false;

    const tick = (ts: number) => {
      if (!running) return;
      const st = usePlaybackStore.getState();
      if (st.status !== 'playing') {
        running = false;
        return;
      }
      if (document.hidden) {
        // Keep the loop alive but freeze the clock — no catch-up jump.
        lastTs = ts;
        rafId = requestAnimationFrame(tick);
        return;
      }
      const deltaMs = lastTs === 0 ? 16.7 : ts - lastTs;
      lastTs = ts;
      st.advanceByDeltaMs(deltaMs);
      rafId = requestAnimationFrame(tick);
    };

    const start = () => {
      if (running) return;
      running = true;
      lastTs = 0;
      rafId = requestAnimationFrame(tick);
    };

    const unsubscribe = usePlaybackStore.subscribe(
      (s) => s.status,
      (status) => {
        if (status === 'playing') start();
        else running = false;
      },
    );
    if (usePlaybackStore.getState().status === 'playing') start();

    return () => {
      running = false;
      cancelAnimationFrame(rafId);
      unsubscribe();
    };
  }, []);
}
