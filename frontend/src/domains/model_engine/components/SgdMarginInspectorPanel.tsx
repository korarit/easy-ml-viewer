import { useEffect } from 'react';
import { Gauge } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useSgdSliceStore, type SgdMetricMode } from '../store/useSgdSliceStore';
import { SgdMarginSurfaceSandbox } from './SgdMarginSurfaceSandbox';
import {
  isInsideCorridor,
  marginStatusFor,
  pseudoConfidence,
} from '../../stage3d/components/surfaces/sgdSurfaceMath';

export const SGD_ALGORITHM_KEYS = ['sgd_classifier', 'sgd_regressor'] as const;

/**
 * Phase 21 — Sandbox inspector for SGDClassifier / SGDRegressor (plus
 * Perceptron & PassiveAggressive): loss & penalty badge, training-step
 * counter, margin-corridor telemetry (raw score, geometric distance,
 * safety gauge), metric-mode switcher, per-row waterfall and the 3D
 * margin-corridor sandbox.
 */
export function SgdMarginInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useSgdSliceStore();

  const isSgd =
    currentModel !== null &&
    (SGD_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isSgd || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const hp = (currentModel.hyperparameters ?? {}) as Record<string, unknown>;
    const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      weights: currentModel.features.map((f) => f.learned_weight_or_importance),
      intercept: currentModel.intercept_or_base,
      loss: str(hp['loss']),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSgd, currentModel?.model_id]);

  if (!isSgd || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isClassifier = s.isClassifier;
  const loss = mech?.loss ?? s.loss ?? '—';
  const regime = mech?.loss_regime ?? '';
  const penalty = mech?.penalty ?? '—';
  const alpha = mech && Number.isFinite(mech.alpha) ? mech.alpha : null;
  const epsilon = mech && Number.isFinite(mech.epsilon) ? mech.epsilon : null;
  const lr = mech?.learning_rate ?? '—';
  const nIter = mech?.n_iter ?? null;
  const tSteps = mech?.total_weight_updates_t ?? null;
  const native = mech?.has_native_proba ?? s.hasNativeProba;
  const rawScore = mech && Number.isFinite(mech.raw_score) ? (mech.raw_score as number) : null;
  const geomDist = mech && Number.isFinite(mech.geometric_distance) ? (mech.geometric_distance as number) : null;
  const wNorm = mech && Number.isFinite(mech.w_norm) ? (mech.w_norm as number) : null;
  const marginWidth = mech && Number.isFinite(mech.margin_width) ? (mech.margin_width as number) : null;
  const inside = mech?.inside_margin ?? (rawScore !== null && marginWidth !== null ? isInsideCorridor(rawScore, marginWidth) : null);
  const status = mech?.margin_status ?? (rawScore !== null && marginWidth !== null ? marginStatusFor(rawScore, marginWidth, !isClassifier) : '—');
  const probaRow = mech?.probabilities ?? null;
  const pseudo = native === false;
  const displayProba =
    probaRow && probaRow.length > 0 ? Math.max(...probaRow) : rawScore !== null ? pseudoConfidence(rawScore) : null;

  const metricValue = ((): string => {
    if (s.metricMode === 'geometric') return geomDist !== null ? `${geomDist >= 0 ? '+' : ''}${geomDist.toFixed(3)} units` : '—';
    if (s.metricMode === 'proba')
      return displayProba !== null ? `${(displayProba * 100).toFixed(1)}%${pseudo ? ' (pseudo)' : ''}` : '—';
    return rawScore !== null ? `${rawScore >= 0 ? '+' : ''}${rawScore.toFixed(3)}` : '—';
  })();

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  // Margin safety gauge position in [-marginWidth, +marginWidth] (classifier).
  const gaugePos =
    rawScore !== null && marginWidth !== null && marginWidth > 0
      ? Math.min(1, Math.max(-1, rawScore / marginWidth))
      : null;

  const testSlice = async () => {
    await s.runTrace();
    await useSgdSliceStore.getState().runSurface();
    const t = useSgdSliceStore.getState().traces[0]?.extra_mechanics;
    if (t && Array.isArray(t.classes) && t.classes.length > 2) {
      useSgdSliceStore.setState({ classCount: t.classes.length });
    }
  };

  const metricModes: Array<{ key: SgdMetricMode; label: string }> = isClassifier
    ? [
        { key: 'raw', label: 'Raw Margin f(x)' },
        { key: 'geometric', label: 'Geometric Dist' },
        { key: 'proba', label: native === false ? 'Pseudo-P' : 'Probability' },
      ]
    : [{ key: 'raw', label: 'Prediction ŷ' }];

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="SGD Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Gauge size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — SGD Margin Corridor {regime ? `(${regime})` : ''}
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_credit_sgd_hinge')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SGDClassifier hinge demo (no native proba)"
          >
            Load Hinge Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_credit_sgd_log')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SGDClassifier log-loss demo (native proba)"
          >
            Load Log-Loss Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_housing_sgd')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SGDRegressor huber demo (epsilon tube)"
          >
            Load Regressor Sample
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
        Loss: {loss}{regime ? ` (${regime})` : ''} | Penalty: {penalty}
        {alpha !== null ? ` (α=${alpha})` : ''}
        {!isClassifier && epsilon !== null ? ` | ε=${epsilon}` : ''} | LR: {lr}
        {nIter !== null ? ` | n_iter: ${nIter}` : ''}
        {tSteps !== null ? ` | Steps (t_): ${tSteps.toLocaleString()}` : ''}
      </p>

      {isClassifier && (
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Metric mode">
          <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
            Metric:
          </span>
          {metricModes.map((m) => (
            <button
              key={m.key}
              onClick={() => s.setMetricMode(m.key)}
              className="font-mono-tabular rounded px-2 py-1 text-[11px]"
              style={{
                border: '1px solid var(--border-strong)',
                background: s.metricMode === m.key ? 'var(--ml-primary)' : 'var(--surface-elevated)',
                color: s.metricMode === m.key ? '#fff' : 'var(--text-primary)',
              }}
            >
              {m.label}
            </button>
          ))}
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-primary)' }}>
            {metricValue}
          </span>
        </div>
      )}

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
        <div className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Frame margin telemetry
          {trace ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              {isClassifier ? (
                <>
                  <div>
                    Raw Score f(x): {rawScore !== null ? `${rawScore >= 0 ? '+' : ''}${rawScore.toFixed(3)}` : '—'}
                    {'  '}(Status: {inside ? '⚠️ ' : ''}{status})
                  </div>
                  <div>
                    Geometric Dist f/‖w‖: {geomDist !== null ? `${geomDist >= 0 ? '+' : ''}${geomDist.toFixed(3)}` : '—'} units from plane
                  </div>
                  {gaugePos !== null && marginWidth !== null && (
                    <div aria-label="Margin safety gauge">
                      Safety: [{(-marginWidth).toFixed(1)} |{'='.repeat(Math.round(((gaugePos + 1) / 2) * 12)).padEnd(12, '·')}({rawScore !== null ? rawScore.toFixed(2) : '—'})| +{marginWidth.toFixed(1)}]
                    </div>
                  )}
                  <div>
                    Native predict_proba: {native === true ? 'yes' : native === false ? 'N/A (pseudo)' : '—'}
                    {displayProba !== null ? ` | Conf: ${(displayProba * 100).toFixed(1)}%` : ''}
                  </div>
                </>
              ) : (
                <>
                  <div>
                    Predicted (ŷ): {typeof trace.final_prediction === 'number' ? trace.final_prediction.toFixed(3) : String(trace.final_prediction)}
                  </div>
                  <div>Intercept (b): {trace.base_value >= 0 ? '+' : ''}{trace.base_value.toFixed(3)}</div>
                  {marginWidth !== null ? <div>ε-tube half-width: ±{marginWidth}</div> : null}
                </>
              )}
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect margin telemetry.</span>
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

      {pseudo && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          Hinge/perceptron losses expose no native predict_proba — confidence is a calibrated
          σ(f(x)) pseudo-score for coloring only; the margin f(x) is the source of truth.
        </p>
      )}

      {contribs.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Feature margin force breakdown (wᵢ·xᵢ + b)
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                Intercept (b)
              </span>
              <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                {trace.base_value >= 0 ? '+' : ''}{trace.base_value.toFixed(3)}
              </span>
            </li>
            {contribs.map(([name, v]) => (
              <li key={name} className="flex items-center gap-2">
                <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                  {name}
                </span>
                <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                  {v >= 0 ? '+' : ''}{v.toFixed(3)}
                </span>
                <span
                  aria-hidden
                  className="h-2 rounded"
                  style={{
                    width: `${Math.max(4, (Math.abs(v) / maxAbs) * 160)}px`,
                    background: v >= 0 ? 'var(--ml-primary)' : 'var(--ml-class-1)',
                    opacity: 0.8,
                  }}
                />
              </li>
            ))}
          </ul>
          {s.phaseShift !== null && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-muted)' }}>
              GPU phase-shift offset (b + off-axis Σwx): {s.phaseShift.toFixed(3)}
            </p>
          )}
        </div>
      )}

      <SgdMarginSurfaceSandbox />

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => void s.runTrace()}
          disabled={!s.modelId || s.isTracing}
          className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          Re-run Trace
        </button>
        <button
          onClick={() => void s.runSurface()}
          disabled={!s.modelId || s.isSurfacing}
          className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          Rebuild Surface
        </button>
      </div>
    </section>
  );
}
