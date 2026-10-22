import { useMemo } from 'react';
import { Workflow } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { usePipelineSliceStore } from '../store/usePipelineSliceStore';
import { PipelineStepBreadcrumb } from './PipelineStepBreadcrumb';
import { PipelineTransformationCard } from './PipelineTransformationCard';
import { PipelineSurfaceSandbox } from './PipelineSurfaceSandbox';

export const PIPELINE_ALGORITHM_KEYS = ['sklearn_pipeline'] as const;

/**
 * Phase 37 — Sandbox inspector for Pipeline tunnels: breadcrumb, tracer,
 * grouped/expanded waterfall and the delegated 3D stage.
 */
export function PipelineInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = usePipelineSliceStore();

  const isPipeline =
    currentModel !== null &&
    ((PIPELINE_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '') ||
      currentModel.is_pipeline === true);

  const bindIfNeeded = () => {
    if (!isPipeline || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
    });
  };
  bindIfNeeded();

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;

  const contribs = useMemo(() => {
    if (!trace) return [];
    if (s.waterfallMode === 'expanded') {
      return Object.entries(trace.extra_mechanics.transformed_row_top16 ?? {})
        .map(([k, v]) => [k, v] as [string, number])
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    }
    return Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [trace, s.waterfallMode]);
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await usePipelineSliceStore.getState().runSurface();
  };

  if (!isPipeline || !currentModel) return null;

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Pipeline Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Workflow size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Pipeline Tunnel + Delegated Stage
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_pipe_numeric')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SimpleImputer + StandardScaler + LogisticRegression"
          >
            Load Numeric Pipe
          </button>
          <button
            onClick={() => void loadSampleModel('sample_pipe_mixed')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="ColumnTransformer (Scaler + OHE) + RandomForest"
          >
            Load Mixed CT Pipe
          </button>
          <button
            onClick={() => void testSlice()}
            disabled={s.isTracing || s.isSurfacing}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {s.isTracing || s.isSurfacing ? 'Tracing…' : 'Test Slice with Sample Data'}
          </button>
        </div>
      </div>

      {mech && (
        <PipelineStepBreadcrumb steps={mech.pipeline_steps} terminalEstimator={mech.delegated_estimator} />
      )}

      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        Model: {mech ? `${mech.delegated_estimator} · raw {${mech.raw_feature_names.join(', ')}} → ${mech.n_transformed_features} cols` : '—'}
      </p>

      <div className="grid gap-2 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Input row (comma-separated, use NaN for missing, {s.featureNames.length} values)
          <input
            value={s.rowInput}
            onChange={(e) => s.setRowInput(e.target.value)}
            placeholder={s.featureNames.join(', ')}
            className="font-mono-tabular rounded px-2 py-1.5"
            style={{
              border: '1px solid var(--border-strong)',
              background: 'var(--surface-elevated)',
              color: 'var(--text-primary)',
            }}
          />
        </label>
        <div className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Waterfall mode
          <div className="flex gap-2">
            <label className="flex items-center gap-1">
              <input
                type="radio"
                name="wf-mode"
                checked={s.waterfallMode === 'grouped'}
                onChange={() => s.setWaterfallMode('grouped')}
              />
              Grouped by Original Column
            </label>
            <label className="flex items-center gap-1">
              <input
                type="radio"
                name="wf-mode"
                checked={s.waterfallMode === 'expanded'}
                onChange={() => s.setWaterfallMode('expanded')}
              />
              Expanded Encoded Features
            </label>
          </div>
        </div>
      </div>

      {s.sliceError && (
        <div
          className="flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="alert"
        >
          <span>{s.sliceError}</span>
          <button onClick={s.clearError} className="underline">
            Dismiss
          </button>
        </div>
      )}

      {trace && mech && (
        <>
          <PipelineTransformationCard mech={mech} waterfallMode={s.waterfallMode} />
          <div
            className="rounded-lg px-4 py-3 text-xs"
            style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
          >
            <p className="font-mono-tabular mb-2" style={{ color: 'var(--text-secondary)' }}>
              Output: {(mech.probabilities ?? []).map((p) => p.toFixed(3)).join(' / ')} · base{' '}
              {trace.base_value.toFixed(4)} + Σcontrib {contribs.reduce((a, [, v]) => a + v, 0).toFixed(4)} ·{' '}
              {s.waterfallMode === 'grouped' ? 'grouped raw attribution' : 'expanded encoded attribution'}
            </p>
            <div className="flex flex-col gap-1">
              {contribs.map(([name, v]) => (
                <div key={name} className="flex items-center gap-2">
                  <span className="font-mono-tabular w-40 truncate" style={{ color: 'var(--text-secondary)' }} title={name}>
                    {name}
                  </span>
                  <div className="relative h-3 flex-1 rounded" style={{ background: 'var(--surface-elevated)' }}>
                    <div
                      className="absolute top-0 h-3 rounded"
                      style={{
                        left: v < 0 ? `${50 + (v / maxAbs) * 50}%` : '50%',
                        width: `${(Math.abs(v) / maxAbs) * 50}%`,
                        background: v >= 0 ? 'var(--ml-positive)' : 'var(--ml-negative)',
                      }}
                    />
                    <div className="absolute top-0 left-1/2 h-3 w-px" style={{ background: 'var(--border-strong)' }} />
                  </div>
                  <span className="font-mono-tabular w-20 text-right" style={{ color: 'var(--text-primary)' }}>
                    {v >= 0 ? '+' : ''}{v.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <PipelineSurfaceSandbox />
    </section>
  );
}
