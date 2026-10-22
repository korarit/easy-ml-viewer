import { create } from 'zustand';
import type { DualComparisonResult } from '../api/dualCompareApi';

interface DualCompareState {
  isCompareOpen: boolean;
  isComparing: boolean;
  compareError: string | null;
  result: DualComparisonResult | null;
  /** Shared frame-locked VCR cursor (both stages read this — spec §2.2). */
  sharedFrameFloat: number;
  /** Currently focused disagreement frame (`null` = none selected). */
  focusedConflict: number | null;
  isCameraSynced: boolean;
  isVcrSynced: boolean;
  setCompareOpen: (open: boolean) => void;
  setComparing: (busy: boolean) => void;
  setCompareError: (msg: string | null) => void;
  setResult: (result: DualComparisonResult | null) => void;
  setSharedFrame: (frame: number) => void;
  setFocusedConflict: (frame: number | null) => void;
  setCameraSynced: (synced: boolean) => void;
  setVcrSynced: (synced: boolean) => void;
  reset: () => void;
}

export const useDualCompareStore = create<DualCompareState>()((set) => ({
  isCompareOpen: false,
  isComparing: false,
  compareError: null,
  result: null,
  sharedFrameFloat: 0,
  focusedConflict: null,
  isCameraSynced: true,
  isVcrSynced: true,

  setCompareOpen: (open) => set({ isCompareOpen: open, ...(open ? { compareError: null } : {}) }),
  setComparing: (busy) => set({ isComparing: busy }),
  setCompareError: (msg) => set({ compareError: msg }),
  setResult: (result) =>
    set({
      result,
      sharedFrameFloat: 0,
      focusedConflict: result && result.disagreement_indices.length > 0 ? result.disagreement_indices[0] : null,
    }),
  setSharedFrame: (frame) => set({ sharedFrameFloat: frame }),
  setFocusedConflict: (frame) => set({ focusedConflict: frame }),
  setCameraSynced: (synced) => set({ isCameraSynced: synced }),
  setVcrSynced: (synced) => set({ isVcrSynced: synced }),
  reset: () =>
    set({
      result: null,
      compareError: null,
      sharedFrameFloat: 0,
      focusedConflict: null,
    }),
}));
