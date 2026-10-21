import { useEffect } from 'react';
import { usePlaybackStore } from '../store/usePlaybackStore';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Phase 10 — global VCR keyboard shortcuts.
 *
 * Edge case #3: typing Space inside `<input>` / `<textarea>` / `<select>`
 * (or any contenteditable) must never toggle playback.
 */
export function usePlaybackKeyboardShortcuts(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (INPUT_TAGS.has(target.tagName) || target.isContentEditable)
      ) {
        return;
      }
      const st = usePlaybackStore.getState();
      if (st.totalFrames <= 1 && e.key !== 'Tab') {
        // Single/empty queue: no transport shortcuts apply.
        return;
      }
      switch (e.key) {
        case ' ':
          e.preventDefault();
          st.togglePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          st.stepFrame(e.shiftKey ? -10 : -1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          st.stepFrame(e.shiftKey ? 10 : 1);
          break;
        case 'Home':
          e.preventDefault();
          st.seekToFrame(0);
          break;
        case 'End':
          e.preventDefault();
          st.seekToFrame(st.totalFrames - 1);
          break;
        case '[':
          st.setLoopRangePoint('A', st.currentFrameIndex);
          break;
        case ']':
          st.setLoopRangePoint('B', st.currentFrameIndex);
          break;
        case 'l':
        case 'L':
          st.cycleLoopMode();
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}
