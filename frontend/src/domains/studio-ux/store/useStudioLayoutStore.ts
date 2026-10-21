import { create } from 'zustand';

interface StudioLayoutState {
  cinemaMode: boolean;
  commandPaletteOpen: boolean;
  aliasModalOpen: boolean;
  toggleCinema: () => void;
  setCinema: (on: boolean) => void;
  setCommandPalette: (open: boolean) => void;
  toggleCommandPalette: () => void;
  setAliasModal: (open: boolean) => void;
}

/**
 * Phase 14 — studio layout state (cinema fullscreen + modal visibility).
 * Kept outside React context so `Tab` / `Ctrl+K` shortcuts can toggle it
 * from anywhere without re-rendering the 3D canvas.
 */
export const useStudioLayoutStore = create<StudioLayoutState>()((set) => ({
  cinemaMode: false,
  commandPaletteOpen: false,
  aliasModalOpen: false,
  toggleCinema: () => set((s) => ({ cinemaMode: !s.cinemaMode })),
  setCinema: (on) => set({ cinemaMode: on }),
  setCommandPalette: (open) => set({ commandPaletteOpen: open }),
  toggleCommandPalette: () => set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),
  setAliasModal: (open) => set({ aliasModalOpen: open }),
}));

