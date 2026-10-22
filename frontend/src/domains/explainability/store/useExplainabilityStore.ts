import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  fetchBuildExplainabilitySession,
  fetchFrameExplanation,
  fetchFrameExplanationBatch,
  type BuildExplainabilitySessionParams,
} from '../api/explainabilityApi';
import type {
  FrameExplanation,
  FrameExplanationPayload,
  TopKOption,
} from '../types/explainability.types';

function payloadToExplanation(payload: FrameExplanationPayload): FrameExplanation {
  return {
    frame_index: payload.frame_index,
    dataset_row_index: payload.dataset_row_index,
    base_value: payload.base_value,
    output_value: payload.output_value,
    other_contributions_sum: payload.other_contributions_sum,
    attributions: payload.attributions,
  };
}

/** Resolve a TopKOption against the model's feature count ('all' → D). */
export function resolveTopK(topK: TopKOption, featureCount: number): number {
  if (topK === 'all') return Math.max(1, featureCount);
  return Math.max(1, Math.min(topK, Math.max(1, featureCount)));
}

interface ExplainabilityState {
  sessionId: string | null;
  buildKey: string | null;
  rowIndices: number[];
  featureCount: number;
  isClassification: boolean;
  baseValue: number | null;
  explanation: FrameExplanation | null;
  explanationFrame: number | null;
  isLoading: boolean;
  explainError: string | null;
  topK: TopKOption;
  hudVisible: boolean;
  hudCompact: boolean;
  panelCollapsed: boolean;
  /** Prefetched explanations by frame — playback reads here, not the API. */
  frameCache: Record<number, FrameExplanation>;
  isPrefetching: boolean;
  loadSession: (params: BuildExplainabilitySessionParams & { buildKey: string }) => Promise<void>;
  fetchFrame: (frameIndex: number) => Promise<void>;
  /** Batch-fill the cache; skips frames already cached. */
  prefetchFrames: (frameIndices: number[]) => Promise<void>;
  setTopK: (topK: TopKOption) => void;
  toggleHud: () => void;
  toggleCompactHud: () => void;
  togglePanel: () => void;
  clearError: () => void;
}

let inFlightAbort: AbortController | null = null;
let requestSeq = 0;

export const useExplainabilityStore = create<ExplainabilityState>()(
  subscribeWithSelector((set, get) => ({
    sessionId: null,
    buildKey: null,
    rowIndices: [],
    featureCount: 0,
    isClassification: false,
    baseValue: null,
    explanation: null,
    explanationFrame: null,
    isLoading: false,
    explainError: null,
    topK: 10,
    hudVisible: true,
    hudCompact: false,
    panelCollapsed: false,
    frameCache: {},
    isPrefetching: false,

    loadSession: async (params) => {
      const s = get();
      if (s.buildKey === params.buildKey && s.sessionId) return;
      if (inFlightAbort) inFlightAbort.abort();
      const controller = new AbortController();
      inFlightAbort = controller;
      const seq = (requestSeq += 1);
      const prevCtx = s.buildKey?.split('|queue:')[0];
      const nextCtx = params.buildKey.split('|queue:')[0];
      const isNewContext = prevCtx !== undefined && prevCtx !== nextCtx;
      set({
        isLoading: true,
        explainError: null,
        ...(isNewContext
          ? {
              sessionId: null,
              explanation: null,
              explanationFrame: null,
              rowIndices: [],
              featureCount: 0,
              baseValue: null,
            }
          : {}),
      });
      try {
        const payload = await fetchBuildExplainabilitySession(
          {
            datasetId: params.datasetId,
            modelId: params.modelId,
            mappingId: params.mappingId,
            selectedRowIndices: params.selectedRowIndices,
            ...(get().sessionId && !isNewContext ? { sessionId: get().sessionId as string } : {}),
          },
          controller.signal,
        );
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({
          sessionId: payload.session_id,
          buildKey: params.buildKey,
          rowIndices: params.selectedRowIndices,
          featureCount: payload.feature_names.length,
          isClassification: payload.is_classification,
          baseValue: payload.base_value,
          explanation: null,
          explanationFrame: null,
          // Stale frames belong to the old session — drop them so playback
          // can never show a previous queue's reasons.
          frameCache: isNewContext || !get().sessionId ? {} : get().frameCache,
        });
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ explainError: err instanceof Error ? err.message : String(err) });
      } finally {
        if (inFlightAbort === controller) inFlightAbort = null;
        if (seq === requestSeq) set({ isLoading: false });
      }
    },

    fetchFrame: async (frameIndex) => {
      const { sessionId, featureCount, topK, explanationFrame, frameCache } = get();
      if (!sessionId) return;
      const cached = frameCache[frameIndex];
      if (cached && explanationFrame === frameIndex) return;
      if (cached) {
        set({ explanation: cached, explanationFrame: frameIndex });
        return;
      }
      if (inFlightAbort) inFlightAbort.abort();
      const controller = new AbortController();
      inFlightAbort = controller;
      const seq = (requestSeq += 1);
      set({ isLoading: true, explainError: null });
      try {
        const payload = await fetchFrameExplanation(
          sessionId,
          frameIndex,
          resolveTopK(topK, featureCount),
          controller.signal,
        );
        if (controller.signal.aborted || seq !== requestSeq) return;
        const entity = payloadToExplanation(payload);
        set((s) => ({
          explanation: entity,
          explanationFrame: frameIndex,
          frameCache: { ...s.frameCache, [frameIndex]: entity },
        }));
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ explainError: err instanceof Error ? err.message : String(err) });
      } finally {
        if (inFlightAbort === controller) inFlightAbort = null;
        if (seq === requestSeq) set({ isLoading: false });
      }
    },

    prefetchFrames: async (frameIndices) => {
      const { sessionId, featureCount, topK, frameCache } = get();
      if (!sessionId || get().isPrefetching) return;
      const missing = [...new Set(frameIndices)].filter(
        (f) => Number.isInteger(f) && f >= 0 && !(f in frameCache),
      );
      if (missing.length === 0) return;
      set({ isPrefetching: true });
      try {
        const payloads = await fetchFrameExplanationBatch(
          sessionId,
          missing.slice(0, 2000),
          resolveTopK(topK, featureCount),
        );
        if (!payloads.length) return;
        set((s) => {
          // Ignore late responses from a previous session/topK.
          if (s.sessionId !== sessionId || s.topK !== topK) return s;
          const next = { ...s.frameCache };
          for (const p of payloads) {
            next[p.frame_index] = payloadToExplanation(p);
          }
          return { frameCache: next };
        });
      } catch {
        /* Prefetch is best-effort — single fetchFrame covers misses. */
      } finally {
        if (get().sessionId === sessionId) set({ isPrefetching: false });
      }
    },

    setTopK: (topK) => {
      set({ topK, explanation: null, explanationFrame: null, frameCache: {} });
    },

    toggleHud: () => set((s) => ({ hudVisible: !s.hudVisible })),

    toggleCompactHud: () => set((s) => ({ hudCompact: !s.hudCompact })),

    togglePanel: () => set((s) => ({ panelCollapsed: !s.panelCollapsed })),

    clearError: () => set({ explainError: null }),
  })),
);
