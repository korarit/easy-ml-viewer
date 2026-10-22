import { useMemo, useState } from 'react';
import { Brain } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useMlpSliceStore } from '../store/useMlpSliceStore';
import { MlpSurfaceSandbox } from './MlpSurfaceSandbox';

export const MLP_ALGORITHM_KEYS = ['mlp', 'mlp_regressor'] as const;

/**
 * Phase 36 — Sandbox inspector for MLP: layer activation health, dead-ReLU
 * banner, Input x Gradient waterfall and the floating synapse-brain stage.
 */
export function MlpInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useMlpSliceStore();
  const [hovered, setHovered] = useState<{ layer: number; index: number } | null>(null);

  const isMlp =
    currentModel !== null &&
    (MLP_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  const bindIfNeeded = () => {
    if (!isMlp || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      isRegressor: currentModel.task_type === 'regression',
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
  };
  bindIfNeeded();

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isRegressor = s.isRegressor || mech?.is_regressor === true;

  const contribs = useMemo(() => {
    if (!trace) return [];
    return Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [trace]);
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const telemetry = (mech?.layer_telemetry ?? []) as Array<{
    layer_index: number;
    total_neurons: number;
    rendered_neurons: number;
    active_neurons: number;
    dead_neurons: number;
    max_activation: number;
  }>;
  const deadBanner = useMemo(() => {
    if (!mech) return [];
    return telemetry
      .filter((t) => t.dead_neurons > 0)
      .map((t) => {
        const pct = ((t.dead_neurons / Math.max(t.total_neurons, 1)) * 100).toFixed(1);
        return `Layer ${t.layer_index}: ${t.dead_neurons} / ${t.total_neurons} dead ReLU (${pct}%)`;
      });
  }, [mech, telemetry]);

  const hoverEquation = useMemo(() => {
    if (!hovered || !trace) return null;
    const acts = (mech?.layer_activations_sample ?? []) as number[][];
    const layerActs = acts[hovered.layer] ?? [];
    const rendered = (mech?.rendered_node_indices ?? []) as number[][];
    const slot = (rendered[hovered.layer] ?? []).indexOf(hovered.index);
    const a = slot >= 0 && slot < layerActs.length ? layerActs[slot] : null;
    return { a, slot };
  }, [hovered, trace, mech]);

  const testSlice = async () => {
    await s.runTrace();
    await useMlpSliceStore.getState().runSurface();
  };

  if (!isMlp || !currentModel) return null;

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="MLP Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Brain size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Floating Synapse Brain + Layer Pulses
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_binary_mlp')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="MLPClassifier binary demo (64-32 ReLU)"
          >
            Load Binary Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multi_mlp')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="MLPClassifier 3-class demo"
          >
            Load Multi Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_reg_mlp')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="MLPRegressor demo"
          >
            Load Regressor
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

      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        Model: {mech ? `${mech.algorithm} · ${(mech.layer_dimensions as number[]).join(' → ')} · ${mech.activation} → ${mech.out_activation}` : '—'}
        {mech?.has_missing_values ? ' · ◆ NaN in row' : ''}
      </p>

      {deadBanner.length > 0 && (
        <div
          className="rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-class-2)', border: '1px solid rgba(245,158,11,0.4)' }}
          role="alert"
        >
          ◆ Dead ReLU: {deadBanner.join(' · ')}
        </div>
      )}

      <div className="grid gap-2 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Input row (comma-separated, {s.featureNames.length} numeric values)
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
          Layer activation health
          {telemetry.length > 0 ? (
            <ol className="font-mono-tabular flex flex-col gap-0.5 text-[11px]">
              {telemetry.map((t) => (
                <li key={t.layer_index}>
                  L{t.layer_index} [{t.total_neurons}{t.rendered_neurons < t.total_neurons ? `→top${t.rendered_neurons}` : ''}]:{' '}
                  active {t.active_neurons}/{t.total_neurons}
                  {t.dead_neurons > 0 ? ` · ☠ dead ${t.dead_neurons}` : ''} · max {t.max_activation.toFixed(2)}
                </li>
              ))}
            </ol>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect layer health.</span>
          )}
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
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="font-mono-tabular mb-2" style={{ color: 'var(--text-secondary)' }}>
            Output: {isRegressor ? Number(mech.linked_prediction).toFixed(4) : `${(mech.probabilities as number[]).map((p) => p.toFixed(3)).join(' / ')}`} ·{' '}
            base {trace.base_value.toFixed(4)} + Σcontrib {contribs.reduce((a, [, v]) => a + v, 0).toFixed(4)} ·{' '}
            {mech.attribution_method}
          </p>
          <div className="flex flex-col gap-1">
            {contribs.map(([name, v]) => (
              <div key={name} className="flex items-center gap-2">
                <span className="font-mono-tabular w-32 truncate" style={{ color: 'var(--text-secondary)' }} title={name}>
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
          {hoverEquation && hovered && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
              σ(Σ a<sub>i</sub>w<sub>ij</sub> + b<sub>j</sub>) @ L{hovered.layer} neuron #{hovered.index}
              {hoverEquation.a != null ? ` → a = ${hoverEquation.a.toFixed(4)}` : ' → a = —'}
            </p>
          )}
        </div>
      )}

      <MlpSurfaceSandbox hovered={hovered} onHoverNode={setHovered} />
    </section>
  );
}
