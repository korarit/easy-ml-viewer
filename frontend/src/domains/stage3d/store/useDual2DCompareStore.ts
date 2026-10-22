import { create } from 'zustand';
import type { Dual2DCompareResult } from '../../recording/api/dual2DCompareApi';

export type Dual2DLayout = 'side-by-side' | 'superimposed';

interface Dual2DCompareState {
  isModalOpen: boolean;
  isComparing: boolean;
  compareError: string | null;
  result: Dual2DCompareResult | null;
  layout: Dual2DLayout;
  showConflictHatch: boolean;
  focusedConflict: number | null;
  setModalOpen: (open: boolean) => void;
  setComparing: (busy: boolean) => void;
  setCompareError: (msg: string | null) => void;
  setResult: (result: Dual2DCompareResult | null) => void;
  setLayout: (layout: Dual2DLayout) => void;
  setShowConflictHatch: (show: boolean) => void;
  setFocusedConflict: (frame: number | null) => void;
  reset: () => void;
}

export const useDual2DCompareStore = create<Dual2DCompareState>()((set) => ({
  isModalOpen: false,
  isComparing: false,
  compareError: null,
  result: null,
  layout: 'superimposed',
  showConflictHatch: true,
  focusedConflict: null,

  setModalOpen: (open) => set({ isModalOpen: open, ...(open ? { compareError: null } : {}) }),
  setComparing: (busy) => set({ isComparing: busy }),
  setCompareError: (msg) => set({ compareError: msg }),
  setResult: (result) =>
    set({
      result,
      focusedConflict:
        result && result.conflictFrameIndices.length > 0 ? result.conflictFrameIndices[0] : null,
    }),
  setLayout: (layout) => set({ layout }),
  setShowConflictHatch: (show) => set({ showConflictHatch: show }),
  setFocusedConflict: (frame) => set({ focusedConflict: frame }),
  reset: () => ({
    result: null,
    compareError: null,
    focusedConflict: null,
    isComparing: false,
  }),
}));
