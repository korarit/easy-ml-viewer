import { useEffect } from 'react';
import { useWhatIfStore } from '../store/useWhatIfStore';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Phase 13 — what-if sandbox keyboard shortcuts.
 *
 * - `W`: toggle the What-If Sandbox panel
 * - `R` (panel open): reset every feature to the current frame's base row
 * - `F`: run Find Nearest Counterfactual
 */
export function useWhatIfKeyboardShortcuts(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (INPUT_TAGS.has(target.tagName) || target.isContentEditable)) return;
      const store = useWhatIfStore.getState();
      if (e.key === 'w' || e.key === 'W') {
        e.preventDefault();
        store.togglePanel();
      } else if ((e.key === 'r' || e.key === 'R') && store.panelOpen) {
        e.preventDefault();
        store.resetAll();
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        void store.findCounterfactual();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}
