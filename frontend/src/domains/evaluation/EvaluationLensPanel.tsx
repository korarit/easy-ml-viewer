import { useEffect, useState } from 'react';
import { useDatasetStore } from '../dataset/store/useDatasetStore';
import { useMappingStore } from '../mapping/store/useMappingStore';
import { useModelStore } from '../model_engine/store/useModelStore';
import { GlobalMetricsRibbon } from './components/GlobalMetricsRibbon';
import { ResidualSlicerPanel } from './components/ResidualSlicerPanel';
import { InteractiveConfusionMatrix } from './components/InteractiveConfusionMatrix';
import { OutlierSlicerCard } from './components/OutlierSlicerCard';
import { TargetLabelMapperModal } from './components/TargetLabelMapperModal';
import { useEvaluationStore, type ErrorSliceType } from './store/useEvaluationStore';

function siblingTargetLabelMap(): Record<string, number | string> | null {
  try {
    const raw = useMappingStore.getState().targetLabelMap;
    if (!raw) return null;
    const out: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(raw)) out[k] = v as number | string;
    return out;
  } catch {
    return null;
  }
}

/** Global Evaluation Lens + error auto-queue panel (Phase 07, Step 4 section). */
export function EvaluationLensPanel() {
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId, targetColumnName } = useMappingStore();
  const eval_ = useEvaluationStore();
  const [mapperOpen, setMapperOpen] = useState(false);

  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const ready = !!datasetId && !!modelId && !!mappingId;

  const runEvaluation = (labelMap?: Record<string, number | string> | null) => {
    if (!datasetId || !modelId || !mappingId) return;
    void eval_.computeGlobalEvaluation(datasetId, modelId, mappingId, labelMap ?? siblingTargetLabelMap());
  };

  // Auto-open the label mapper when the backend reports a mismatch.
  useEffect(() => {
    if (eval_.summary?.needs_label_mapping) setMapperOpen(true);
  }, [eval_.summary?.needs_label_mapping, eval_.summary?.label_mapping_hint]);

  // Alt+E: queue Top-50 worst errors (§2.5 keyboard shortcut).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA');
      if (e.altKey && (e.key === 'e' || e.key === 'E' || e.key === 'E'.toLowerCase()) && !typing) {
        e.preventDefault();
        useEvaluationStore.getState().applyErrorSliceToQueue('worst_abs');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const summary = eval_.summary;

  return (
    <section
      className="flex flex-col gap-3 rounded-md p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="Global evaluation lens"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold tracking-wide">
          GLOBAL EVALUATION LENS &amp; ERROR SLICER
          {targetColumnName && (
            <span className="font-mono-tabular ml-2 text-[11px font-normal]" style={{ color: 'var(--text-muted)' }}>
              (Target: {targetColumnName})
            </span>
          )}
        </h2>
        <div className="flex items-center gap-2">
          {summary?.needs_label_mapping && (
            <button
              onClick={() => setMapperOpen(true)}
              className="rounded-md px-3 py-1.5 text-xs font-medium"
              style={{ border: '1px solid var(--ml-anomaly)', color: 'var(--ml-anomaly)' }}
            >
              Re-Map Target Labels
            </button>
          )}
          <button
            onClick={() => runEvaluation()}
            disabled={!ready || eval_.isComputing}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
            title={ready ? 'Run global evaluation (chunked, <150ms typical)' : 'Load a dataset + model and commit a mapping first'}
          >
            {eval_.isComputing ? 'Evaluating…' : summary ? 'Re-run Evaluation' : 'Run Global Evaluation'}
          </button>
        </div>
      </div>

      {!ready && (
        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Commit a column mapping (Step 3) to enable global evaluation. Unlabeled datasets still get prediction
          stats + OOD outliers.
        </p>
      )}

      {eval_.computeError && (
        <p className="rounded-md px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }} role="alert">
          {eval_.computeError}
        </p>
      )}

      {summary && (
        <>
          <GlobalMetricsRibbon summary={summary} />
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            {summary.regression && (
              <ResidualSlicerPanel
                report={summary.regression}
                onSlice={(t: ErrorSliceType) => eval_.applyErrorSliceToQueue(t)}
                disabled={eval_.isComputing}
              />
            )}
            {summary.classification && (
              <InteractiveConfusionMatrix
                report={summary.classification}
                onCellClick={(a, p) => eval_.applyConfusionCellToQueue(a, p)}
                onSliceUncertain={() => eval_.applyErrorSliceToQueue('uncertain')}
                onSliceMisclassified={() => eval_.applyErrorSliceToQueue('misclassified')}
                disabled={eval_.isComputing}
              />
            )}
            <OutlierSlicerCard
              outlierCount={summary.ood_outlier_count}
              queuedCount={summary.ood_outlier_indices.length}
              onSliceOod={() => eval_.applyErrorSliceToQueue('ood')}
              disabled={eval_.isComputing}
            />
          </div>
          {eval_.activeSlicerTag && (
            <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--ml-primary)' }} role="status">
              ACTIVE SLICER: {eval_.activeSlicerTag}
            </p>
          )}
        </>
      )}

      <TargetLabelMapperModal
        open={mapperOpen && !!summary?.needs_label_mapping}
        rawLabels={summary?.raw_target_labels ?? []}
        expectedClasses={summary?.expected_classes ?? []}
        hint={summary?.label_mapping_hint ?? null}
        isComputing={eval_.isComputing}
        onConfirm={(labelMap) => {
          setMapperOpen(false);
          useMappingStore.getState().setTargetLabelMap(
            Object.fromEntries(Object.entries(labelMap).map(([k, v]) => [k, Number(v)])),
          );
          runEvaluation(labelMap);
        }}
        onClose={() => setMapperOpen(false)}
      />
    </section>
  );
}
