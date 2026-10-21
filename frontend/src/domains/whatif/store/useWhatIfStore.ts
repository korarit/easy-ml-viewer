import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  fetchBuildWhatIfSession,
  fetchCounterfactual,
  fetchEvaluateWhatIf,
  type BuildWhatIfSessionParams,
} from '../api/whatifApi';
import type {
  CounterfactualResult,
  WhatIfEvaluation,
  WhatIfSessionPayload,
} from '../types/whatif.types';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';

/** 30ms trailing throttle for slider drags (spec §5.3, 33Hz max). */
const EVALUATE_THROTTLE_MS = 30;

function toMutatedPayload(
  mutated: Record<number, number>,
  featureNames: string[],
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(mutated)) {
    const name = featureNames[Number(k)];
    if (name !== undefined && Number.isFinite(v)) out[name] = v;
  }
  return out;
}

interface WhatIfState {
  sessionId: string | null;
  buildKey: string | null;
  rowIndices: number[];
  featureNames: string[];
  featureRanges: WhatIfSessionPayload['feature_ranges'];
  isClassification: boolean;
  axisXFeatureIndex: number;
  axisYFeatureIndex: number;
  activeFrame: number;
  baseRowValues: number[];
  mutatedValues: Record<number, number>;
  lockedIndices: number[];
  evaluation: WhatIfEvaluation | null;
  counterfactual: CounterfactualResult | null;
  targetClass: number | null;
  targetThreshold: number | null;
  panelOpen: boolean;
  isSessionLoading: boolean;
  sessionError: string | null;
  isEvaluating: boolean;
  evalError: string | null;
  lastEvalMs: number | null;
  isFinding: boolean;
  findError: string | null;

  loadSession: (params: BuildWhatIfSessionParams & { buildKey: string }) => Promise<void>;
  setFrame: (frameIndex: number) => Promise<void>;
  setMutatedValue: (featureIndex: number, value: number) => void;
  toggleLock: (featureIndex: number) => void;
  resetAll: () => void;
  applyRecipe: () => void;
  setTargetClass: (v: number | null) => void;
  setTargetThreshold: (v: number | null) => void;
  findCounterfactual: () => Promise<void>;
  togglePanel: () => void;
  clearErrors: () => void;
}

let sessionAbort: AbortController | null = null;
let evalAbort: AbortController | null = null;
let findAbort: AbortController | null = null;
let requestSeq = 0;
let evalTimer: ReturnType<typeof setTimeout> | null = null;

async function runEvaluate(
  sessionId: string,
  frame: number,
  mutated: Record<number, number>,
  featureNames: string[],
  set: (p: Partial<WhatIfState>) => void,
  get: () => WhatIfState,
): Promise<void> {
  if (evalAbort) evalAbort.abort();
  const controller = new AbortController();
  evalAbort = controller;
  const seq = (requestSeq += 1);
  // Keep the previous evaluation on screen (ghost stays put) while the
  // throttled request flies — only the spinner reflects the pending state.
  set({ isEvaluating: true, evalError: null });
  try {
    const payload = await fetchEvaluateWhatIf(
      sessionId,
      frame,
      toMutatedPayload(mutated, featureNames),
      controller.signal,
    );
    if (controller.signal.aborted || seq !== requestSeq) return;
    const latency = (payload as unknown as { _latencyMs?: number })._latencyMs ?? null;
    set({
      evaluation: payload,
      baseRowValues: payload.base_row_values,
      lastEvalMs: latency !== null ? Math.round(latency * 10) / 10 : null,
    });
    void get;
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return;
    if (controller.signal.aborted || seq !== requestSeq) return;
    set({ evalError: err instanceof Error ? err.message : String(err) });
  } finally {
    if (evalAbort === controller) evalAbort = null;
    if (seq === requestSeq) set({ isEvaluating: false });
  }
}

function scheduleEvaluate(
  set: (p: Partial<WhatIfState>) => void,
  get: () => WhatIfState,
): void {
  if (evalTimer) clearTimeout(evalTimer);
  evalTimer = setTimeout(() => {
    evalTimer = null;
    const s = get();
    if (!s.sessionId) return;
    void runEvaluate(s.sessionId, s.activeFrame, s.mutatedValues, s.featureNames, set, get);
  }, EVALUATE_THROTTLE_MS);
}

export const useWhatIfStore = create<WhatIfState>()(
  subscribeWithSelector((set, get) => ({
    sessionId: null,
    buildKey: null,
    rowIndices: [],
    featureNames: [],
    featureRanges: [],
    isClassification: false,
    axisXFeatureIndex: 0,
    axisYFeatureIndex: 1,
    activeFrame: 0,
    baseRowValues: [],
    mutatedValues: {},
    lockedIndices: [],
    evaluation: null,
    counterfactual: null,
    targetClass: null,
    targetThreshold: null,
    panelOpen: true,
    isSessionLoading: false,
    sessionError: null,
    isEvaluating: false,
    evalError: null,
    lastEvalMs: null,
    isFinding: false,
    findError: null,

    loadSession: async (params) => {
      const s = get();
      if (s.buildKey === params.buildKey && s.sessionId) return;
      if (sessionAbort) sessionAbort.abort();
      const controller = new AbortController();
      sessionAbort = controller;
      const seq = (requestSeq += 1);
      const prevCtx = s.buildKey?.split('|queue:')[0];
      const nextCtx = params.buildKey.split('|queue:')[0];
      const isNewContext = prevCtx !== undefined && prevCtx !== nextCtx;
      set({
        isSessionLoading: true,
        sessionError: null,
        ...(isNewContext
          ? {
              sessionId: null,
              evaluation: null,
              counterfactual: null,
              baseRowValues: [],
              mutatedValues: {},
              lockedIndices: [],
              targetClass: null,
              targetThreshold: null,
            }
          : {}),
      });
      try {
        const payload = await fetchBuildWhatIfSession(params, controller.signal);
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({
          sessionId: payload.session_id,
          buildKey: params.buildKey,
          rowIndices: params.selectedRowIndices,
          featureNames: payload.feature_names,
          featureRanges: payload.feature_ranges,
          isClassification: payload.is_classification,
          axisXFeatureIndex: payload.axis_x_feature_index,
          axisYFeatureIndex: payload.axis_y_feature_index,
          // Any session (re)build starts from a clean sandbox — the queue
          // rows may differ even when the dataset/model context is the same.
          evaluation: null,
          counterfactual: null,
          baseRowValues: [],
          mutatedValues: {},
          findError: null,
          evalError: null,
        });
        // Prime the base row for the current frame (spec: sandbox starts clean).
        await get().setFrame(usePlaybackStore.getState().currentFrameIndex);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ sessionError: err instanceof Error ? err.message : String(err) });
      } finally {
        if (sessionAbort === controller) sessionAbort = null;
        if (seq === requestSeq) set({ isSessionLoading: false });
      }
    },

    setFrame: async (frameIndex) => {
      const s = get();
      if (!s.sessionId) return;
      const clamped = Math.max(0, Math.min(frameIndex, Math.max(s.rowIndices.length - 1, 0)));
      if (clamped === s.activeFrame && s.baseRowValues.length > 0 && s.evaluation) return;
      // Edge #1: changing frames auto-resets the sandbox to the new base row.
      // The stale evaluation is cleared so the ghost never points at the
      // previous frame while the base re-evaluates.
      set({
        activeFrame: clamped,
        mutatedValues: {},
        evaluation: null,
        counterfactual: null,
        findError: null,
      });
      await runEvaluate(s.sessionId, clamped, {}, s.featureNames, set, get);
    },

    setMutatedValue: (featureIndex, value) => {
      const s = get();
      if (!s.sessionId) return;
      // Auto-pause on edit: entering the sandbox freezes the VCR (spec §2.4).
      usePlaybackStore.getState().pause();
      if (s.lockedIndices.includes(featureIndex)) return;
      if (!Number.isFinite(value)) return;
      set({ mutatedValues: { ...s.mutatedValues, [featureIndex]: value } });
      scheduleEvaluate(set, get);
    },

    toggleLock: (featureIndex) => {
      const s = get();
      const locked = s.lockedIndices.includes(featureIndex)
        ? s.lockedIndices.filter((i) => i !== featureIndex)
        : [...s.lockedIndices, featureIndex];
      const mutated = { ...s.mutatedValues };
      delete mutated[featureIndex];
      set({ lockedIndices: locked, mutatedValues: mutated });
      if (s.sessionId) scheduleEvaluate(set, get);
    },

    resetAll: () => {
      const s = get();
      if (evalTimer) clearTimeout(evalTimer);
      set({ mutatedValues: {}, counterfactual: null, findError: null, evalError: null });
      if (s.sessionId) void runEvaluate(s.sessionId, s.activeFrame, {}, s.featureNames, set, get);
    },

    applyRecipe: () => {
      const s = get();
      const ev = s.counterfactual?.evaluation;
      if (!ev) return;
      const next: Record<number, number> = {};
      for (const c of ev.changed_features) {
        if (!s.lockedIndices.includes(c.feature_index)) next[c.feature_index] = c.mutated_value;
      }
      usePlaybackStore.getState().pause();
      set({ mutatedValues: next });
      if (s.sessionId) scheduleEvaluate(set, get);
    },

    setTargetClass: (v) => set({ targetClass: v, counterfactual: null, findError: null }),

    setTargetThreshold: (v) => set({ targetThreshold: v, counterfactual: null, findError: null }),

    findCounterfactual: async () => {
      const s = get();
      if (!s.sessionId || s.isFinding) return;
      if (!s.isClassification && s.targetThreshold === null) {
        set({ findError: 'Set a target prediction first (e.g. Target >= 500.0)' });
        return;
      }
      if (s.lockedIndices.length >= s.featureNames.length && s.featureNames.length > 0) {
        set({ findError: 'กรุณาปลดล็อกอย่างน้อย 1 Feature เพื่อค้นหา Counterfactual' });
        return;
      }
      if (findAbort) findAbort.abort();
      const controller = new AbortController();
      findAbort = controller;
      const seq = (requestSeq += 1);
      set({ isFinding: true, findError: null });
      try {
        const payload = await fetchCounterfactual(
          s.sessionId,
          s.activeFrame,
          {
            ...(s.isClassification && s.targetClass !== null
              ? { targetClassOrThreshold: s.targetClass }
              : !s.isClassification && s.targetThreshold !== null
                ? { targetClassOrThreshold: s.targetThreshold }
                : {}),
            lockedFeatureIndices: s.lockedIndices,
          },
          controller.signal,
        );
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ counterfactual: payload });
        if (!payload.found) {
          set({ findError: payload.status_message });
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ findError: err instanceof Error ? err.message : String(err) });
      } finally {
        if (findAbort === controller) findAbort = null;
        if (seq === requestSeq) set({ isFinding: false });
      }
    },

    togglePanel: () => set((st) => ({ panelOpen: !st.panelOpen })),

    clearErrors: () => set({ sessionError: null, evalError: null, findError: null }),
  })),
);
