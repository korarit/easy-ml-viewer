import { useEffect } from 'react';
import { requestQuickRecord, requestStopRecording } from './videoRecorderControl';
import { useExportStore } from '../store/useExportStore';
import { useWebGLSnapshot } from './useWebGLSnapshot';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Phase 16 — global export keyboard shortcuts.
 * - `Ctrl/Cmd+E`: open the Export Studio modal
 * - `Ctrl/Cmd+Shift+S`: quick 3D snapshot (PNG, current scale/background)
 * - `Ctrl/Cmd+Shift+R`: start / stop the 3D video recording
 *
 * No-ops outside the 3D studio (modal never mounted there, so the control
 * registry stays empty and shortcuts safely do nothing).
 */
export function useExportKeyboardShortcuts(enabled = true) {
  const { captureSnapshot } = useWebGLSnapshot();

  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const key = e.key.toLowerCase();
      const target = e.target as HTMLElement | null;
      const typing =
        !!target && (INPUT_TAGS.has(target.tagName) || target.isContentEditable);
      // Ctrl+E opens Export Studio (allowed even inside inputs).
      if (key === 'e' && !e.shiftKey) {
        e.preventDefault();
        useExportStore.getState().setModal(true);
        return;
      }
      if (!e.shiftKey) return;
      if (typing) return;
      if (key === 's') {
        e.preventDefault();
        void captureSnapshot().catch(() => {
          /* error is surfaced in the export store */
        });
      } else if (key === 'r') {
        e.preventDefault();
        if (useExportStore.getState().isRecording) requestStopRecording();
        else requestQuickRecord();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled, captureSnapshot]);
}
