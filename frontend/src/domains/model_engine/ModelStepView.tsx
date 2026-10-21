import { useEffect } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { ModelUploadDropzone } from './components/ModelUploadDropzone';
import { ModelArchitectureCard } from './components/ModelArchitectureCard';
import { ModelFeatureSchemaTable } from './components/ModelFeatureSchemaTable';
import { HyperparameterAccordion } from './components/HyperparameterAccordion';
import { useModelStore } from './store/useModelStore';
import { useSystemStore } from '../../core/store/useSystemStore';

/** Step 2 studio: artifact upload → architecture card → feature schema. */
export function ModelStepView() {
  const { currentModel, isLoadingModel, modelError, clearModelError } = useModelStore();
  const { setActiveStep, unlockStep } = useSystemStore();
  const canProceed = currentModel !== null && !isLoadingModel;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === 'm' || e.key === 'M')) {
        e.preventDefault();
        document.querySelector<HTMLInputElement>('#root input[type="file"]')?.click();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-3 overflow-auto p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-sm font-semibold tracking-wide">STEP 2: MODEL ENGINE & SCHEMA INTROSPECTION</h1>
        {isLoadingModel && (
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--ml-primary)' }}>
            Deserializing → fitted check → schema…
          </span>
        )}
      </div>

      {modelError && (
        <div
          className="flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="alert"
        >
          <span>{modelError}</span>
          <button onClick={clearModelError} className="underline">
            Dismiss
          </button>
        </div>
      )}

      <ModelUploadDropzone />
      <ModelArchitectureCard />
      <ModelFeatureSchemaTable />
      <HyperparameterAccordion />

      <div className="flex justify-between">
        <button
          onClick={() => setActiveStep(1)}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          <ArrowLeft size={14} /> Back to Step 1
        </button>
        <button
          disabled={!canProceed}
          onClick={() => unlockStep(3)}
          title={canProceed ? 'Go to Step 3' : 'Load a fitted model first'}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium disabled:opacity-40"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          Proceed to Step 3: Column Mapping <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
