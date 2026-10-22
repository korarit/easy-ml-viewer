import { useMemo } from 'react';
import { BrainCircuit } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useOnnxDeepSliceStore } from '../store/useOnnxDeepSliceStore';
import { OnnxDeepStageRouter } from './OnnxDeepStageRouter';
import {
  batchModeLabel,
  formatLatentCoords,
  latentPulseHeight,
  providerBadgeLabel,
} from '../../stage3d/components/surfaces/onnxDeepSurfaceMath';

export const ONNX_DEEP_ALGORITHM_KEYS = ['onnx_deep', 'onnx_deep_regressor'] as const;

/**
 * Phase 39 — Sandbox inspector for ONNX deep nets: provider/batch telemetry
 * pill, operator mini-graph, deep layer pulse bars, finite-difference
 * waterfall and the dual 3D stage (input surface ↔ latent manifold).
 */
export function OnnxDeepInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useOnnxDeepSliceStore();

  const isDeep =
    currentModel !== null &&
    (ONNX_DEEP_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  const bindIfNeeded = () => {
    if (!isDeep || !currentModel) return;
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
  const isRegressor = mech?.is_regressor === true;

  const contribs = useMemo(() => {
    if (!trace) return [];
    return Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [trace]);
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const pulseSigma = useMemo(() => {
    if (!mech || mech.layer_telemetry.length === 0) return 1;
    const mean = mech.layer_telemetry.reduce((a, t) => a + Math.abs(t.mean_activation), 0) / mech.layer_telemetry.length;
    return mean > 1e-9 ? mean : 1;
  }, [mech]);

  const testSlice = async () => {
    await s.runTrace();
    await useOnnxDeepSliceStore.getState().runSurface();
  };

  if (!isDeep || !currentModel) return null;

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="ONNX Deep Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <BrainCircuit size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — {mech?.domain_badge ?? 'ONNX Deep Net Probe Injector'}
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_deep_mlp_onnx')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Tabular DNN classifier (dynamic batch)"
          >
            Load Deep MLP
          </button>
          <button
            onClick={() => void loadSampleModel('sample_deep_static_onnx')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Same DNN with static batch_size=1 export"
          >
            Load Static-Batch
          </button>
          <button
            onClick={() => void loadSampleModel('sample_deep_reg_onnx')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Tabular DNN regressor"
          >
            Load Deep Regressor
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
        <div className="flex flex-wrap gap-2 text-[11px]">
          <span
            className="font-mono-tabular rounded px-2 py-1"
            style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
            title="Execution provider auto-selector"
          >
            Provider: {providerBadgeLabel(mech.provider)}
          </span>
          <span
            className="font-mono-tabular rounded px-2 py-1"
            style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
            title="Static batch_size=1 fallback state"
          >
            Batch: {batchModeLabel(mech.batch_mode, mech.static_batch_size)}
          </span>
          <span
            className="font-mono-tabular rounded px-2 py-1"
            style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
            title="Injected intermediate probes"
          >
            Probes: {mech.n_probed_layers} layers{mech.injection_active ? '' : ' (passthrough)'} · penultimate{' '}
            {mech.penultimate_layer ?? '—'}
          </span>
        </div>
      )}

      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        Model:{' '}
        {mech
          ? `${mech.detected_operator} · ${mech.attribution_method} · output ${mech.output_transform}`
          : '—'}
        {mech?.has_missing_values ? ' · ◆ NaN in row' : ''}
      </p>

      <div className="grid gap-2 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Input row (comma-separated, {s.featureNames.length} values)
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
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Target class index
          <input
            type="number"
            min={0}
            value={s.classIdx}
            onChange={(e) => s.setClassIdx(Number(e.target.value))}
            className="font-mono-tabular rounded px-2 py-1.5"
            style={{
              border: '1px solid var(--border-strong)',
              background: 'var(--surface-elevated)',
              color: 'var(--text-primary)',
            }}
          />
        </label>
      </div>

      {mech ? (
        <div
          className="font-mono-tabular flex flex-wrap items-center gap-1.5 rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
          title="ONNX graph operator chain"
        >
          {mech.node_chain.map((op, i) => (
            <span key={`${op}-${i}`} className="flex items-center gap-1.5">
              {i > 0 && <span style={{ color: 'var(--text-muted)' }}>➔</span>}
              <span
                className="rounded px-1.5 py-0.5"
                style={{
                  border: '1px solid var(--border-strong)',
                  background: op.includes('Gemm') || op.includes('MatMul') ? 'rgba(129,140,248,0.12)' : 'transparent',
                }}
              >
                {op}
              </span>
            </span>
          ))}
        </div>
      ) : (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Run a trace to inspect the deep graph operators and probe taps.
        </p>
      )}

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
            Output:{' '}
            {isRegressor
              ? Number(mech.predicted_value ?? trace.final_prediction).toFixed(4)
              : `${(mech.probabilities as number[]).map((p) => p.toFixed(3)).join(' / ')}`} · base{' '}
            {trace.base_value.toFixed(4)} + Σcontrib {contribs.reduce((a, [, v]) => a + v, 0).toFixed(4)} ·{' '}
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
                  <div className="absolute left-1/2 top-0 h-3 w-px" style={{ background: 'var(--border-strong)' }} />
                </div>
                <span className="font-mono-tabular w-20 text-right" style={{ color: 'var(--text-primary)' }}>
                  {v >= 0 ? '+' : ''}{v.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
          <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
            (*) Active row in latent space: {formatLatentCoords(mech.latent_coords_3d)}
          </p>
        </div>
      )}

      {mech && mech.layer_telemetry.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 text-[11px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
            Deep layer pulse & activation histogram ({mech.n_probed_layers} injected probes)
          </p>
          <div className="flex flex-col gap-1.5">
            {mech.layer_telemetry.map((t) => (
              <div key={t.layer_name} className="flex items-center gap-2">
                <span
                  className="font-mono-tabular w-40 truncate"
                  style={{ color: 'var(--text-secondary)' }}
                  title={`${t.layer_name} (D=${t.width})`}
                >
                  {t.layer_name} (D={t.width})
                </span>
                <div className="relative h-3 flex-1 rounded" style={{ background: 'var(--surface-elevated)' }}>
                  <div
                    className="absolute left-0 top-0 h-3 rounded"
                    style={{
                      width: `${latentPulseHeight(t.mean_activation, pulseSigma) * 100}%`,
                      background: 'var(--ml-primary)',
                    }}
                  />
                </div>
                <span className="font-mono-tabular w-44 text-right" style={{ color: 'var(--text-secondary)' }}>
                  mean {t.mean_activation.toFixed(3)} · sparse {(t.sparsity * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <OnnxDeepStageRouter />
    </section>
  );
}
