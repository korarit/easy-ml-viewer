import { useEffect } from 'react';
import { useExplainabilityStore } from '../store/useExplainabilityStore';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Phase 12 — explainability keyboard shortcuts (mounted with the studio).
 *
 * - `H`: hide/show the 3D floating HUD callout
 * - `E`: open/close the explainability waterfall panel
 */
export function useExplainabilityKeyboardShortcuts(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (INPUT_TAGS.has(target.tagName) || target.isContentEditable)) {
        return;
      }
      const store = useExplainabilityStore.getState();
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        store.toggleHud();
      } else if (e.key === 'e' || e.key === 'E') {
        e.preventDefault();
        store.togglePanel();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}
