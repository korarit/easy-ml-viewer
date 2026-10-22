import { useEffect } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { ModelUploadDropzone } from './components/ModelUploadDropzone';
import { ModelArchitectureCard } from './components/ModelArchitectureCard';
import { ModelFeatureSchemaTable } from './components/ModelFeatureSchemaTable';
import { HyperparameterAccordion } from './components/HyperparameterAccordion';
import { LinearSliceInspector } from './components/LinearSliceInspector';
import { LogisticInspectorPanel } from './components/LogisticInspectorPanel';
import { RegularizedInspectorPanel } from './components/RegularizedInspectorPanel';
import { ElasticNetInspectorPanel } from './components/ElasticNetInspectorPanel';
import { SgdMarginInspectorPanel } from './components/SgdMarginInspectorPanel';
import { KnnRadarInspectorPanel } from './components/KnnRadarInspectorPanel';
import { SvmKernelInspectorPanel } from './components/SvmKernelInspectorPanel';
import { GnbBellInspectorPanel } from './components/GnbBellInspectorPanel';
import { DiscreteNbInspectorPanel } from './components/DiscreteNbInspectorPanel';
import { DiscriminantInspectorPanel } from './components/DiscriminantInspectorPanel';
import { TreeInspectorPanel } from './components/TreeInspectorPanel';
import { ForestInspectorPanel } from './components/ForestInspectorPanel';
import { ExtraTreesInspectorPanel } from './components/ExtraTreesInspectorPanel';
import { AdaBoostInspectorPanel } from './components/AdaBoostInspectorPanel';
import { SklearnGbmInspectorPanel } from './components/SklearnGbmInspectorPanel';
import { HistGbmInspectorPanel } from './components/HistGbmInspectorPanel';
import { XgboostInspectorPanel } from './components/XgboostInspectorPanel';
import { LightgbmInspectorPanel } from './components/LightgbmInspectorPanel';
import { CatBoostInspectorPanel } from './components/CatBoostInspectorPanel';
import { MlpInspectorPanel } from './components/MlpInspectorPanel';
import { PipelineInspectorPanel } from './components/PipelineInspectorPanel';
import { OnnxClassicalInspectorPanel } from './components/OnnxClassicalInspectorPanel';
import { OnnxDeepInspectorPanel } from './components/OnnxDeepInspectorPanel';
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
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="text-[11px tracking-widest uppercase]" style={{ color: 'var(--text-muted)' }}>
            Step 2 of 5
          </p>
          <h1 className="text-sm font-semibold">Choose your model</h1>
          <p className="mt-0.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
            Drop a trained model file — we read its inputs and check it can predict.
          </p>
        </div>
        {isLoadingModel && (
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--ml-primary)' }} role="status">
            Reading model… checking it can predict.
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
      <LinearSliceInspector />
      <LogisticInspectorPanel />
      <RegularizedInspectorPanel />
      <ElasticNetInspectorPanel />
      <SgdMarginInspectorPanel />
      <KnnRadarInspectorPanel />
      <SvmKernelInspectorPanel />
      <GnbBellInspectorPanel />
      <DiscreteNbInspectorPanel />
      <DiscriminantInspectorPanel />
      <TreeInspectorPanel />
      <ForestInspectorPanel />
      <ExtraTreesInspectorPanel />
      <AdaBoostInspectorPanel />
      <SklearnGbmInspectorPanel />
      <HistGbmInspectorPanel />
      <XgboostInspectorPanel />
      <LightgbmInspectorPanel />
      <CatBoostInspectorPanel />
      <MlpInspectorPanel />
      <PipelineInspectorPanel />
      <OnnxClassicalInspectorPanel />
      <OnnxDeepInspectorPanel />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          onClick={() => setActiveStep(1)}
          className="flex min-h-[40px] items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          <ArrowLeft size={14} /> Back
        </button>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs" style={{ color: currentModel ? 'var(--ml-class-0)' : 'var(--text-muted)' }}>
            {currentModel ? `✓ Model ready: ${currentModel.algorithm_key}` : '○ Drop a trained model file above'}
          </span>
          <button
            disabled={!canProceed}
            onClick={() => unlockStep(3)}
            title={canProceed ? 'Go to Step 3' : 'Load a trained model first'}
            className="flex min-h-[40px] items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium disabled:opacity-40"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            Continue to mapping <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
