import { useEffect } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { useDatasetStore } from '../dataset/store/useDatasetStore';
import { useModelStore } from '../model_engine/store/useModelStore';
import { useSystemStore } from '../../core/store/useSystemStore';
import { useMappingStore } from './store/useMappingStore';
import { MappingHeaderBar } from './components/MappingHeaderBar';
import { FeatureMappingRowCard } from './components/FeatureMappingRowCard';
import { TargetColumnSelectorCard } from './components/TargetColumnSelectorCard';
import { DryRunValidationDrawer } from './components/DryRunValidationDrawer';

/** Step 3 studio: interactive column mapper + dry-run validator (Phase 05). */
export function MappingStepView() {
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { setActiveStep, unlockStep } = useSystemStore();
  const {
    items,
    mappingError,
    isLoading,
    isCommitting,
    isValid,
    leakageWarning,
    duplicateColumns,
    featureErrors,
    requestAutoSuggest,
    requestMapByOrder,
    commitMapping,
    clearError,
  } = useMappingStore();

  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const ready = !!datasetId && !!modelId;

  // Zero-click initial auto-match when entering Step 3 with both artifacts.
  useEffect(() => {
    if (ready) void requestAutoSuggest(datasetId, modelId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, modelId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === 'm' || e.key === 'M')) {
        e.preventDefault();
        if (ready) void requestAutoSuggest(datasetId, modelId);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        void handleCommit();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, datasetId, modelId]);

  const handleCommit = async () => {
    const ok = await commitMapping();
    if (ok) unlockStep(4);
  };

  const weightByIndex = new Map(
    (currentModel?.features ?? []).map((f) => [f.index, f.learned_weight_or_importance]),
  );

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-3 overflow-auto p-4">
      <MappingHeaderBar
        onAutoMatch={() => ready && void requestAutoSuggest(datasetId, modelId)}
        onMapByOrder={() => ready && void requestMapByOrder(datasetId, modelId)}
      />

      {!ready && (
        <div
          className="rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="alert"
        >
          Load a dataset (Step 1) and a model (Step 2) first — mapping needs both.
        </div>
      )}

      {mappingError && (
        <div
          className="flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="alert"
        >
          <span>{mappingError}</span>
          <button onClick={clearError} className="underline">Dismiss</button>
        </div>
      )}

      {leakageWarning && (
        <div
          className="rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="alert"
        >
          {leakageWarning}
        </div>
      )}

      {duplicateColumns.length > 0 && (
        <div
          className="rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="alert"
        >
          Duplicate Column Warning: {duplicateColumns.join(', ')} is mapped to more than one feature.
        </div>
      )}

      {isLoading && (
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--ml-primary)' }}>
          Running 3-tier fuzzy auto-match + dry-run on first 100 rows…
        </p>
      )}

      {items.map((item) => (
        <FeatureMappingRowCard
          key={item.feature_index}
          item={item}
          columns={currentDataset?.columns ?? []}
          weight={weightByIndex.get(item.feature_index) ?? null}
          error={featureErrors[item.feature_index]}
        />
      ))}

      {items.length > 0 && <TargetColumnSelectorCard columns={currentDataset?.columns ?? []} />}
      <DryRunValidationDrawer />

      <div className="flex justify-between">
        <button
          onClick={() => setActiveStep(2)}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          <ArrowLeft size={14} /> Back to Step 2
        </button>
        <button
          onClick={() => void handleCommit()}
          disabled={!ready || !isValid || isCommitting}
          title={!isValid ? 'Resolve mapping errors first' : 'Confirm mapping and go to Step 4 (Ctrl+Enter)'}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium disabled:opacity-40"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          {isCommitting ? 'Validating…' : 'Confirm Mapping & Go to Step 4 Queue'} <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
