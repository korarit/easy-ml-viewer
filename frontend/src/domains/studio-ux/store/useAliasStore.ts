import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AliasConfig, ClassAlias, FeatureAlias } from '../types/studioUx.types';
import { fetchAutosuggestAliases, fetchAliasConfig, upsertAliasConfig } from '../api/studioUxApi';

interface AliasState {
  sessionId: string;
  classAliases: Record<string, ClassAlias>;
  featureAliases: Record<string, FeatureAlias>;
  isLoading: boolean;
  isSaving: boolean;
  error: string | null;
  lastSavedAt: string | null;

  setSessionId: (id: string) => void;
  loadFromServer: (sessionId: string) => Promise<void>;
  saveToServer: () => Promise<void>;
  autosuggest: (featureNames: string[], classValues: Array<string | number>) => Promise<void>;
  setClassAlias: (key: string, patch: Partial<ClassAlias>) => void;
  setFeatureAlias: (key: string, patch: Partial<FeatureAlias>) => void;
  importJson: (raw: unknown) => void;
  exportJson: () => AliasConfig;
  clearError: () => void;
}

function asRecord<T>(value: unknown): Record<string, T> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, T>;
  }
  return {};
}

/**
 * Phase 14 — class/feature alias store.
 * Persisted to localStorage so HUD + waterfall labels survive reloads (spec §6).
 */
export const useAliasStore = create<AliasState>()(
  persist(
    (set, get) => ({
      sessionId: 'default',
      classAliases: {},
      featureAliases: {},
      isLoading: false,
      isSaving: false,
      error: null,
      lastSavedAt: null,

      setSessionId: (id) => set({ sessionId: id || 'default' }),

      loadFromServer: async (sessionId) => {
        set({ isLoading: true, error: null, sessionId });
        try {
          const cfg = await fetchAliasConfig(sessionId);
          set({
            classAliases: cfg.class_aliases ?? {},
            featureAliases: cfg.feature_aliases ?? {},
          });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        } finally {
          set({ isLoading: false });
        }
      },

      saveToServer: async () => {
        const s = get();
        set({ isSaving: true, error: null });
        try {
          const cfg = await upsertAliasConfig({
            session_id: s.sessionId,
            class_aliases: s.classAliases,
            feature_aliases: s.featureAliases,
          });
          set({
            classAliases: cfg.class_aliases ?? {},
            featureAliases: cfg.feature_aliases ?? {},
            lastSavedAt: new Date().toISOString(),
          });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        } finally {
          set({ isSaving: false });
        }
      },

      autosuggest: async (featureNames, classValues) => {
        set({ isLoading: true, error: null });
        try {
          const cfg = await fetchAutosuggestAliases(featureNames, classValues);
          set({
            classAliases: cfg.class_aliases ?? {},
            featureAliases: cfg.feature_aliases ?? {},
          });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        } finally {
          set({ isLoading: false });
        }
      },

      setClassAlias: (key, patch) =>
        set((s) => ({
          classAliases: { ...s.classAliases, [key]: { ...s.classAliases[key], ...patch } as ClassAlias },
        })),

      setFeatureAlias: (key, patch) =>
        set((s) => ({
          featureAliases: {
            ...s.featureAliases,
            [key]: { ...s.featureAliases[key], ...patch } as FeatureAlias,
          },
        })),

      importJson: (raw) => {
        try {
          const blob = (raw as { alias_json?: unknown })?.alias_json ?? raw;
          const obj = blob as Partial<AliasConfig>;
          set({
            classAliases: asRecord<ClassAlias>(obj.class_aliases),
            featureAliases: asRecord<FeatureAlias>(obj.feature_aliases),
            error: null,
          });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        }
      },

      exportJson: () => {
        const s = get();
        return {
          session_id: s.sessionId,
          class_aliases: s.classAliases,
          feature_aliases: s.featureAliases,
        };
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'easy-ml-viewer:studio-aliases:v1',
      partialize: (s) => ({
        sessionId: s.sessionId,
        classAliases: s.classAliases,
        featureAliases: s.featureAliases,
      }),
    },
  ),
);

/** Human-readable class label with raw-value fallback. */
export function resolveClassLabel(raw: string | number, fallbackPrefix = 'Class'): string {
  const aliases = useAliasStore.getState().classAliases;
  const hit = aliases[String(raw)];
  if (hit?.display_label) return hit.display_label;
  return `${fallbackPrefix} ${String(raw)}`;
}

/** Human-readable feature label with unit suffix. */
export function resolveFeatureLabel(raw: string): string {
  const aliases = useAliasStore.getState().featureAliases;
  const hit = aliases[raw];
  if (!hit?.display_name) return raw;
  return hit.unit_suffix ? `${hit.display_name} (${hit.unit_suffix})` : hit.display_name;
}

