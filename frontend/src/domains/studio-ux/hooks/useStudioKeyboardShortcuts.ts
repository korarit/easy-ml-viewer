import { useEffect } from 'react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useStudioLayoutStore } from '../store/useStudioLayoutStore';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Phase 14 — global studio keyboard shortcuts.
 *
 * - `Ctrl/Cmd+K`: command palette (works even inside inputs)
 * - `Tab` (outside inputs/modals): cinema fullscreen toggle
 * - `1/2/3/0`: camera presets (1=Iso, 2=Top, 3=Front, 0=Follow ball)
 * - `Tab` inside inputs keeps its native focus behavior (spec edge #4).
 */
export function useStudioKeyboardShortcuts(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      // Command palette always wins (even inside text fields).
      if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        useStudioLayoutStore.getState().toggleCommandPalette();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      const target = e.target as HTMLElement | null;
      const typing =
        !!target && (INPUT_TAGS.has(target.tagName) || target.isContentEditable);
      const layout = useStudioLayoutStore.getState();

      if (e.key === 'Tab') {
        // Modal open or typing -> preserve native focus traversal.
        if (typing || layout.commandPaletteOpen || layout.aliasModalOpen) return;
        e.preventDefault();
        layout.toggleCinema();
        return;
      }
      if (e.key === 'Escape') {
        if (layout.commandPaletteOpen) layout.setCommandPalette(false);
        else if (layout.aliasModalOpen) layout.setAliasModal(false);
        else if (layout.cinemaMode) layout.setCinema(false);
        return;
      }
      if (typing) return;

      const stage = useStage3DStore.getState();
      if (e.key === '1') stage.setCameraPreset('isometric');
      else if (e.key === '2') stage.setCameraPreset('top_xy');
      else if (e.key === '3') stage.setCameraPreset('front_xz');
      else if (e.key === '0') stage.setCameraPreset('follow_ball');
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}

/** Sync VCR frame -> Stage3D active ball (bi-directional scrub follow). */
export function usePlaybackToStageSync(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const unsub = usePlaybackStore.subscribe(
      (s) => s.currentFrameIndex,
      (frame) => useStage3DStore.getState().setActiveFrameIndex(frame),
    );
    return unsub;
  }, [enabled]);
}

