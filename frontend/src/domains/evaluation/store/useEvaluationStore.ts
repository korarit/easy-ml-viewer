import { create } from 'zustand';
import { fetchComputeEvaluation, type GlobalEvaluationSummary } from '../api/evaluationApi';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';

export type ErrorSliceType =
  | 'worst_abs'
  | 'over_pred'
  | 'under_pred'
  | 'accurate'
  | 'ood'
  | 'misclassified'
  | 'uncertain';

export const SLICE_LABELS: Record<ErrorSliceType, string> = {
  worst_abs: 'Top 50 Worst Errors',
  over_pred: 'Top 50 Over-predicted',
  under_pred: 'Top 50 Under-predicted',
  accurate: 'Top 50 Most Accurate',
  ood: 'OOD Feature Outliers',
  misclassified: 'All Misclassified',
  uncertain: 'Most Uncertain Rows',
};

interface EvaluationStoreState {
  summary: GlobalEvaluationSummary | null;
  isComputing: boolean;
  computeError: string | null;
  activeSlicerTag: string | null;
  lastComputeKey: string | null;
  lastTargetColumn: string | null;
  computeGlobalEvaluation: (
    datasetId: string,
    modelId: string,
    mappingId: string,
    targetLabelMap?: Record<string, number | string> | null,
  ) => Promise<void>;
  applyErrorSliceToQueue: (sliceType: ErrorSliceType) => void;
  applyConfusionCellToQueue: (actualIdx: number, predictedIdx: number) => void;
  clearEvaluation: () => void;
}

function pushToQueue(indices: number[], tag: string) {
  const unique = [...new Set(indices)];
  useTrajectoryQueueStore.setState({
    selectedRowIndices: unique,
    selectedLookupSet: new Set<number>(unique),
    lastClickedRowIndex: unique.length > 0 ? unique[unique.length - 1] : null,
    truncatedNotice:
      unique.length > 0
        ? `Error slicer active: ${tag} (${unique.length} frames queued).`
        : null,
  });
}

export const useEvaluationStore = create<EvaluationStoreState>((set, get) => ({
  summary: null,
  isComputing: false,
  computeError: null,
  activeSlicerTag: null,
  lastComputeKey: null,
  lastTargetColumn: null,

  computeGlobalEvaluation: async (datasetId, modelId, mappingId, targetLabelMap) => {
    const key = `${datasetId}|${modelId}|${mappingId}|${JSON.stringify(targetLabelMap ?? null)}`;
    if (get().isComputing) return;
    set({ isComputing: true, computeError: null });
    try {
      const summary = await fetchComputeEvaluation({
        dataset_id: datasetId,
        model_id: modelId,
        mapping_id: mappingId,
        top_k: 50,
        target_label_map: targetLabelMap ?? null,
      });
      set({ summary, lastComputeKey: key, activeSlicerTag: null });
    } catch (err) {
      set({ computeError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isComputing: false });
    }
  },

  applyErrorSliceToQueue: (sliceType) => {
    const s = get().summary;
    if (!s) return;
    let indices: number[] = [];
    if (sliceType === 'ood') {
      indices = s.ood_outlier_indices;
    } else if (sliceType === 'misclassified' || sliceType === 'uncertain') {
      indices =
        sliceType === 'misclassified'
          ? (s.classification?.all_misclassified_indices ?? [])
          : (s.classification?.most_uncertain_indices ?? []);
    } else {
      const r = s.regression;
      if (!r) return;
      indices =
        sliceType === 'worst_abs'
          ? r.top_worst_abs_error_indices
          : sliceType === 'over_pred'
            ? r.top_over_predicted_indices
            : sliceType === 'under_pred'
              ? r.top_under_predicted_indices
              : r.top_accurate_indices;
    }
    if (indices.length === 0) return;
    const tag = SLICE_LABELS[sliceType];
    set({ activeSlicerTag: `${tag} — ${indices.length} rows queued` });
    pushToQueue(indices, tag);
  },

  applyConfusionCellToQueue: (actualIdx, predictedIdx) => {
    const s = get().summary;
    const cell = s?.classification?.cells.find(
      (c) => c.actual_class_idx === actualIdx && c.predicted_class_idx === predictedIdx,
    );
    if (!cell || cell.row_count === 0) return;
    const diagonal = actualIdx === predictedIdx;
    const tag = diagonal
      ? `Correct: ${cell.actual_label} → ${cell.predicted_label} (${cell.row_count})`
      : `Error cell: ${cell.actual_label} → ${cell.predicted_label} (${cell.row_count})`;
    set({ activeSlicerTag: `${tag} queued` });
    pushToQueue(cell.sample_row_indices, tag);
  },

  clearEvaluation: () =>
    set({ summary: null, activeSlicerTag: null, computeError: null, lastComputeKey: null }),
}));
