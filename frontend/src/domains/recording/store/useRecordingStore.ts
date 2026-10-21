import { create } from 'zustand';
import type { MlvBookmark, PrebakedMlvSession } from '../types/recording.types';

interface RecordingState {
  isBakeModalOpen: boolean;
  isBaking: boolean;
  bakeProgress: number;
  bakeError: string | null;
  lastBakedFilename: string | null;
  replaySession: PrebakedMlvSession | null;
  isReplayMode: boolean;
  loadError: string | null;
  isLoadingMlv: boolean;
  setBakeModal: (open: boolean) => void;
  setBaking: (baking: boolean, progress?: number) => void;
  setBakeError: (msg: string | null) => void;
  setLastBaked: (filename: string | null) => void;
  enterReplay: (session: PrebakedMlvSession) => void;
  exitReplay: () => void;
  setLoadError: (msg: string | null) => void;
  setLoadingMlv: (loading: boolean) => void;
  addReplayBookmark: (bookmark: MlvBookmark) => void;
}

export const useRecordingStore = create<RecordingState>()((set) => ({
  isBakeModalOpen: false,
  isBaking: false,
  bakeProgress: 0,
  bakeError: null,
  lastBakedFilename: null,
  replaySession: null,
  isReplayMode: false,
  loadError: null,
  isLoadingMlv: false,

  setBakeModal: (open) => set({ isBakeModalOpen: open, ...(open ? { bakeError: null } : {}) }),
  setBaking: (baking, progress) =>
    set((s) => ({ isBaking: baking, bakeProgress: progress ?? (baking ? s.bakeProgress : 0) })),
  setBakeError: (msg) => set({ bakeError: msg }),
  setLastBaked: (filename) => set({ lastBakedFilename: filename }),
  enterReplay: (session) =>
    set({ replaySession: session, isReplayMode: true, loadError: null }),
  exitReplay: () => set({ replaySession: null, isReplayMode: false }),
  setLoadError: (msg) => set({ loadError: msg }),
  setLoadingMlv: (loading) => set({ isLoadingMlv: loading }),
  addReplayBookmark: (bookmark) =>
    set((s) =>
      s.replaySession
        ? {
            replaySession: {
              ...s.replaySession,
              header: {
                ...s.replaySession.header,
                bookmarks: [...s.replaySession.header.bookmarks, bookmark],
              },
            },
          }
        : {},
    ),
}));
