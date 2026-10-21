import { useEffect } from 'react';
import { useTimelineStore } from '../store/useTimelineStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Phase 11 — timeline keyboard shortcuts (mounted alongside the VCR panel).
 *
 * - `B`: bookmark editor at the current frame (edits the note if one exists)
 * - `N` / `.`: next event · `P` / `,`: previous event
 * - `Ctrl/⌘ + =` / `-`: zoom in/out · `Ctrl/⌘ + 0`: reset zoom
 */
export function useTimelineKeyboardShortcuts(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (INPUT_TAGS.has(target.tagName) || target.isContentEditable)) {
        return;
      }
      const tl = useTimelineStore.getState();
      if (!tl.data || tl.data.totalFrames <= 1) return;
      const rows = tl.rowIndices;
      switch (e.key) {
        case 'b':
        case 'B':
          if (tl.editorOpen) break;
          e.preventDefault();
          {
            const frame = usePlaybackStore.getState().currentFrameIndex;
            const existing = tl.data.bookmarks.find(
              (bm) =>
                bm.dataset_row_index === rows[frame] ||
                (bm.frame_index === frame && rows.indexOf(bm.dataset_row_index) === -1),
            );
            tl.openEditor(frame, existing);
          }
          break;
        case 'n':
        case 'N':
        case '.':
          e.preventDefault();
          tl.jumpToNext(rows);
          break;
        case 'p':
        case 'P':
        case ',':
          e.preventDefault();
          tl.jumpToPrev(rows);
          break;
        case '=':
        case '+':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            tl.zoomBy(1 / 1.5);
          }
          break;
        case '-':
        case '_':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            tl.zoomBy(1.5);
          }
          break;
        case '0':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            tl.resetZoom();
          }
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}
