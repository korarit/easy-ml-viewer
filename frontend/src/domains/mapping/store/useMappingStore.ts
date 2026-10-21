import { create } from 'zustand';
import {
  fetchAutoSuggest,
  fetchCommitMapping,
  fetchMapByOrder,
  fetchValidatePreview,
  type DryRunReport,
  type FeatureMappingItem,
  type ImputationStrategy,
} from '../api/mappingApi';

interface MappingStoreState {
  mappingId: string | null;
  items: FeatureMappingItem[];
  targetColumnName: string | null;
  targetLabelMap: Record<string, number> | null;
  isValid: boolean;
  duplicateColumns: string[];
  leakageWarning: string | null;
  featureErrors: Record<number, string>;
  previewMatrixHead: Array<Array<number | null>>;
  testedRows: number;
  outputShape: [number, number] | null;
  supportsNan: boolean;
  isLoading: boolean;
  isCommitting: boolean;
  mappingError: string | null;
  lastDatasetId: string | null;
  lastModelId: string | null;
  requestAutoSuggest: (datasetId: string, modelId: string) => Promise<void>;
  requestMapByOrder: (datasetId: string, modelId: string) => Promise<void>;
  updateFeatureMapping: (featureIndex: number, patch: Partial<FeatureMappingItem>) => Promise<void>;
  setGlobalNaNStrategy: (strategy: ImputationStrategy) => Promise<void>;
  setTargetColumn: (name: string | null) => Promise<void>;
  setTargetLabelMap: (map: Record<string, number> | null) => void;
  commitMapping: () => Promise<boolean>;
  clearError: () => void;
}

let validateTimer: ReturnType<typeof setTimeout> | null = null;

function applyReport(
  set: (p: Partial<MappingStoreState>) => void,
  report: DryRunReport,
  items?: FeatureMappingItem[],
) {
  set({
    isValid: report.is_valid,
    duplicateColumns: report.duplicate_columns_used,
    leakageWarning: report.target_leakage_warning,
    featureErrors: report.feature_errors,
    previewMatrixHead: report.preview_matrix_head,
    testedRows: report.tested_row_count,
    outputShape: report.output_shape,
    ...(items ? { items } : {}),
  });
}

export const useMappingStore = create<MappingStoreState>((set, get) => ({
  mappingId: null,
  items: [],
  targetColumnName: null,
  targetLabelMap: null,
  isValid: false,
  duplicateColumns: [],
  leakageWarning: null,
  featureErrors: {},
  previewMatrixHead: [],
  testedRows: 0,
  outputShape: null,
  supportsNan: false,
  isLoading: false,
  isCommitting: false,
  mappingError: null,
  lastDatasetId: null,
  lastModelId: null,

  requestAutoSuggest: async (datasetId, modelId) => {
    set({ isLoading: true, mappingError: null, lastDatasetId: datasetId, lastModelId: modelId });
    try {
      const res = await fetchAutoSuggest(datasetId, modelId);
      set({
        mappingId: res.mapping.mapping_id,
        items: res.mapping.items,
        targetColumnName: res.mapping.target_column_name,
        targetLabelMap: res.mapping.target_label_map,
        supportsNan: res.supports_nan_input,
      });
      applyReport(set, res.report);
    } catch (err) {
      set({ mappingError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isLoading: false });
    }
  },

  requestMapByOrder: async (datasetId, modelId) => {
    set({ isLoading: true, mappingError: null, lastDatasetId: datasetId, lastModelId: modelId });
    try {
      const res = await fetchMapByOrder(datasetId, modelId);
      set({
        mappingId: res.mapping.mapping_id,
        items: res.mapping.items,
        targetColumnName: res.mapping.target_column_name,
        supportsNan: res.supports_nan_input,
      });
      applyReport(set, res.report);
    } catch (err) {
      set({ mappingError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isLoading: false });
    }
  },

  updateFeatureMapping: async (featureIndex, patch) => {
    const s = get();
    const next = s.items.map((it) =>
      it.feature_index === featureIndex
        ? { ...it, ...patch, match_tier: patch.dataset_column_name !== undefined ? 'manual' as const : it.match_tier }
        : it,
    );
    set({ items: next });
    // Debounced dry-run (§5.2): avoid hammering the API while switching dropdowns.
    if (!s.lastDatasetId || !s.lastModelId) return;
    if (validateTimer) clearTimeout(validateTimer);
    validateTimer = setTimeout(() => {
      const cur = get();
      if (!cur.lastDatasetId || !cur.lastModelId) return;
      void fetchValidatePreview({
        dataset_id: cur.lastDatasetId,
        model_id: cur.lastModelId,
        items: cur.items,
        target_column_name: cur.targetColumnName,
      })
        .then((report) => {
          set({ supportsNan: report.supports_nan_input });
          applyReport(set, report, report.items);
        })
        .catch((err) => set({ mappingError: err instanceof Error ? err.message : String(err) }));
    }, 250);
  },

  setGlobalNaNStrategy: async (strategy) => {
    const s = get();
    const next = s.items.map((it) => ({ ...it, imputation_strategy: strategy }));
    set({ items: next });
    if (!s.lastDatasetId || !s.lastModelId) return;
    try {
      const report = await fetchValidatePreview({
        dataset_id: s.lastDatasetId,
        model_id: s.lastModelId,
        items: next,
        target_column_name: s.targetColumnName,
      });
      applyReport(set, report, report.items);
    } catch (err) {
      set({ mappingError: err instanceof Error ? err.message : String(err) });
    }
  },

  setTargetColumn: async (name) => {
    const s = get();
    set({ targetColumnName: name, targetLabelMap: null });
    if (!s.lastDatasetId || !s.lastModelId) return;
    try {
      const report = await fetchValidatePreview({
        dataset_id: s.lastDatasetId,
        model_id: s.lastModelId,
        items: s.items,
        target_column_name: name,
      });
      applyReport(set, report, report.items);
      if (report.target_leakage_warning) {
        // Keep the warning visible; commit still allowed but flagged.
      }
    } catch (err) {
      set({ mappingError: err instanceof Error ? err.message : String(err) });
    }
  },

  setTargetLabelMap: (map) => set({ targetLabelMap: map }),

  commitMapping: async () => {
    const s = get();
    if (!s.lastDatasetId || !s.lastModelId) {
      set({ mappingError: 'Load a dataset and a model first.' });
      return false;
    }
    set({ isCommitting: true, mappingError: null });
    try {
      const res = await fetchCommitMapping({
        dataset_id: s.lastDatasetId,
        model_id: s.lastModelId,
        items: s.items,
        target_column_name: s.targetColumnName,
        target_label_map: s.targetLabelMap,
      });
      set({
        mappingId: res.mapping.mapping_id,
        targetLabelMap: res.mapping.target_label_map,
        supportsNan: res.supports_nan_input,
      });
      applyReport(set, res.report);
      return res.report.is_valid;
    } catch (err) {
      set({ mappingError: err instanceof Error ? err.message : String(err) });
      return false;
    } finally {
      set({ isCommitting: false });
    }
  },

  clearError: () => set({ mappingError: null }),
}));
