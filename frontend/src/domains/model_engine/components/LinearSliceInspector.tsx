import { FlaskConical } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useLinearSliceStore } from '../store/useLinearSliceStore';
import { LinearFormulaBanner } from './LinearFormulaBanner';
import { LinearContributionCard } from './LinearContributionCard';
import { LinearSurfaceSandbox } from './LinearSurfaceSandbox';

/**
 * Phase 04 §1 — Sandbox Inspector: bootstraps the sample model, runs the
 * single-row equation trace and the 3D hyperplane surface preview.
 */
export function LinearSliceInspector() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const {
    modelId,
    rowInput,
    actualInput,
    traces,
    surface,
    isTracing,
    isSurfacing,
    sliceError,
    bindModel,
    setRowInput,
    setActualInput,
    runTrace,
    runSurface,
    clearError,
  } = useLinearSliceStore();

  const isLinear = currentModel?.algorithm_key === 'linear_regression';
  if (!isLinear || !currentModel) return null;

  const bound = modelId === currentModel.model_id;

  const bootstrap = async () => {
    try {
      // Prefer the already-loaded model; fall back to the bundled sample.
      const id = currentModel.model_id;
      bindModel(
        id,
        currentModel.features.map((f) => f.name),
        currentModel.features.map((f) => f.learned_weight_or_importance),
        currentModel.intercept_or_base,
      );
    } catch {
      const key = 'sample_house_price_linear';
      await loadSampleModel(key);
    }
  };

  const testSlice = async () => {
    if (!bound) {
      bindModel(
        currentModel.model_id,
        currentModel.features.map((f) => f.name),
        currentModel.features.map((f) => f.learned_weight_or_importance),
        currentModel.intercept_or_base,
      );
    }
    await runTrace();
    await useLinearSliceStore.getState().runSurface();
  };

  const loadSample = async () => {
    try {
      await loadSampleModel('sample_house_price_linear');
    } catch {
      /* error surfaces via model store */
    }
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Linear Regression Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <FlaskConical size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Equation Trace & 3D Hyperplane
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSample()}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            Load Sample Data
          </button>
          <button
            onClick={() => void (bound ? testSlice() : bootstrap().then(() => void testSlice()))}
            disabled={isTracing || isSurfacing}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {isTracing || isSurfacing ? 'Tracing…' : 'Test Slice with Sample Data'}
          </button>
        </div>
      </div>

      <div className="grid gap-2 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Input row (comma-separated, {currentModel.features.length} values)
          <input
            value={rowInput}
            onChange={(e) => setRowInput(e.target.value)}
            placeholder={currentModel.features.map((f) => f.name).join(', ')}
            className="font-mono-tabular rounded px-2 py-1.5"
            style={{
              border: '1px solid var(--border-strong)',
              background: 'var(--surface-elevated)',
              color: 'var(--text-primary)',
            }}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Actual target y (optional — enables residual)
          <input
            value={actualInput}
            onChange={(e) => setActualInput(e.target.value)}
            placeholder="3900000"
            className="font-mono-tabular rounded px-2 py-1.5"
            style={{
              border: '1px solid var(--border-strong)',
              background: 'var(--surface-elevated)',
              color: 'var(--text-primary)',
            }}
          />
        </label>
      </div>

      {sliceError && (
        <div
          className="flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="alert"
        >
          <span>{sliceError}</span>
          <button onClick={clearError} className="underline">
            Dismiss
          </button>
        </div>
      )}

      {(traces.length > 0 || bound) && <LinearFormulaBanner />}
      <LinearContributionCard />
      {(traces.length > 0 || surface) && <LinearSurfaceSandbox />}

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => void runTrace()}
          disabled={!bound || isTracing}
          className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          Re-run Trace
        </button>
        <button
          onClick={() => void runSurface()}
          disabled={!bound || isSurfacing}
          className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          Rebuild Surface
        </button>
      </div>
    </section>
  );
}
