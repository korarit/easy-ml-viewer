import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  fetchBuildExplainabilitySession,
  fetchFrameExplanation,
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
  loadSession: (params: BuildExplainabilitySessionParams & { buildKey: string }) => Promise<void>;
  fetchFrame: (frameIndex: number) => Promise<void>;
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
      const { sessionId, featureCount, topK, explanationFrame } = get();
      if (!sessionId) return;
      if (explanationFrame === frameIndex && get().explanation) return;
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
        set({ explanation: payloadToExplanation(payload), explanationFrame: frameIndex });
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ explainError: err instanceof Error ? err.message : String(err) });
      } finally {
        if (inFlightAbort === controller) inFlightAbort = null;
        if (seq === requestSeq) set({ isLoading: false });
      }
    },

    setTopK: (topK) => {
      set({ topK, explanation: null, explanationFrame: null });
    },

    toggleHud: () => set((s) => ({ hudVisible: !s.hudVisible })),

    toggleCompactHud: () => set((s) => ({ hudCompact: !s.hudCompact })),

    togglePanel: () => set((s) => ({ panelCollapsed: !s.panelCollapsed })),

    clearError: () => set({ explainError: null }),
  })),
);
