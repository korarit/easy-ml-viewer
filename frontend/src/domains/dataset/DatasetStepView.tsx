import { useEffect } from 'react';
import { ArrowRight } from 'lucide-react';
import { DatasetDropzone } from './components/DatasetDropzone';
import { MlvInstantDropzone } from '../recording/components/MlvInstantDropzone';
import { SheetSelectorBar } from './components/SheetSelectorBar';
import { MemorySavingsBanner } from './components/MemorySavingsBanner';
import { ColumnProfileTable } from './components/ColumnProfileTable';
import { DatasetSamplePreview } from './components/DatasetSamplePreview';
import { useDatasetStore } from './store/useDatasetStore';
import { useSystemStore } from '../../core/store/useSystemStore';

/** Step 1 studio: ingestion → telemetry → schema inspector → sample preview. */
export function DatasetStepView() {
  const { currentDataset, isUploading, uploadError, clearError } = useDatasetStore();
  const { unlockStep } = useSystemStore();
  const numericCount =
    currentDataset?.columns.filter((c) => c.logical_type === 'numeric').length ?? 0;
  const canProceed = currentDataset !== null && numericCount >= 1 && !isUploading;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA');
      if ((e.ctrlKey || e.metaKey) && (e.key === 'o' || e.key === 'O')) {
        e.preventDefault();
        document.querySelector<HTMLInputElement>('#root input[type="file"]')?.click();
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'f' || e.key === 'F') && currentDataset) {
        e.preventDefault();
        document.querySelector<HTMLInputElement>('#column-search')?.focus();
      }
      if (e.key === 'Enter' && !typing && canProceed) unlockStep(2);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [currentDataset, canProceed, unlockStep]);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-3 overflow-auto p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-sm font-semibold tracking-wide">STEP 1: DATASET INGESTION & PROFILING</h1>
        {isUploading && (
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--ml-primary)' }}>
            1. Uploading → 2. Encoding → 3. float32 → 4. Profiles…
          </span>
        )}
      </div>

      {uploadError && (
        <div
          className="flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="alert"
        >
          <span>{uploadError}</span>
          <button onClick={clearError} className="underline">
            Dismiss
          </button>
        </div>
      )}

      <DatasetDropzone />
      <MlvInstantDropzone />
      <SheetSelectorBar />
      <MemorySavingsBanner />
      <ColumnProfileTable />
      <DatasetSamplePreview />

      <div className="flex justify-end">
        <button
          disabled={!canProceed}
          onClick={() => unlockStep(2)}
          title={canProceed ? 'Go to Step 2 (Enter)' : 'Load a dataset with at least 1 numeric column first'}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium disabled:opacity-40"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          Proceed to Step 2: Load ML Model <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
