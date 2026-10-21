import { useEffect } from 'react';
import { FlaskConical } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useLogisticSliceStore } from '../store/useLogisticSliceStore';
import { sigmoid } from '../../stage3d/components/surfaces/sigmoidSurfaceMath';
import { LogisticSurfaceSandbox } from './LogisticSurfaceSandbox';

/**
 * Phase 18 — Sandbox inspector for Logistic Regression:
 * solver/penalty/C header, probability-vs-logodds toggle, target-class
 * selector, decision-threshold slider, per-row log-odds trace and
 * waterfall of w_j * x_j contributions.
 */
export function LogisticInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useLogisticSliceStore();

  const isLogistic = currentModel?.algorithm_key === 'logistic_regression';
  useEffect(() => {
    if (!isLogistic || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const hp = (currentModel.hyperparameters ?? {}) as Record<string, unknown>;
    const num = (v: unknown): number | null =>
      typeof v === 'number' && Number.isFinite(v) ? v : null;
    s.bindModel({
      modelId: currentModel.model_id,
      featureNames: currentModel.features.map((f) => f.name),
      weights: currentModel.features.map((f) => f.learned_weight_or_importance),
      intercept: currentModel.intercept_or_base,
      classes: (currentModel.classes ?? []).map((c) => c as string | number),
      solver: typeof hp['solver'] === 'string' ? (hp['solver'] as string) : null,
      penalty: typeof hp['penalty'] === 'string' ? (hp['penalty'] as string) : null,
      cValue: num(hp['C']),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLogistic, currentModel?.model_id]);

  if (!isLogistic || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const logit = mech?.logit ?? null;
  const proba =
    mech && mech.probabilities.length > 0
      ? (mech.probabilities[Math.min(s.targetClassIdx, mech.probabilities.length - 1)] ?? null)
      : null;
  const verdict =
    trace !== null && proba !== null
      ? { crossed: proba >= s.threshold, proba }
      : null;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await useLogisticSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Logistic Regression Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <FlaskConical size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Log-Odds Trace & 3D Sigmoid S-Surface
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_credit_default_logistic')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            Load Sample Data
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
        Solver: {s.solver ?? 'lbfgs'} | Penalty: {s.penalty ?? 'l2'} | C:{' '}
        {s.cValue !== null ? s.cValue : '—'}
        {trace && mech?.intercept === 0 ? ' | Intercept: 0.0 (Disabled)' : ''}
      </p>

      <div className="flex flex-wrap items-center gap-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
        <span className="uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          Surface view
        </span>
        <label className="flex items-center gap-1">
          <input
            type="radio"
            name="logistic-zmode"
            checked={s.zMode === 'probability'}
            onChange={() => s.setZMode('probability')}
          />
          Probability S-Curve
        </label>
        <label className="flex items-center gap-1">
          <input
            type="radio"
            name="logistic-zmode"
            checked={s.zMode === 'logodds'}
            onChange={() => s.setZMode('logodds')}
          />
          Linear Log-Odds (z)
        </label>
        {s.classes.length > 0 && (
          <label className="flex items-center gap-1.5">
            Target class
            <select
              aria-label="Target class"
              value={s.targetClassIdx}
              onChange={(e) => void s.setTargetClassIdx(Number(e.target.value))}
              className="rounded px-1.5 py-0.5"
              style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
            >
              {s.classes.map((c, i) => (
                <option key={`${c}-${i}`} value={i}>
                  {String(c)}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex items-center gap-2">
          Threshold τ
          <input
            aria-label="Decision threshold"
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={s.threshold}
            onChange={(e) => s.setThreshold(Number(e.target.value))}
          />
          <span className="font-mono-tabular">{s.threshold.toFixed(2)}</span>
        </label>
      </div>

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
          Frame internal flow
          {trace && logit !== null && proba !== null ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div>
                Linear Log-Odds (z): {logit >= 0 ? '+' : ''}
                {logit.toFixed(3)}
              </div>
              <div>
                Sigmoid σ(z): {(sigmoid(logit) * 100).toFixed(1)}%{' '}
                {sigmoid(logit) >= s.threshold ? '(≥ τ)' : '(< τ)'}
              </div>
              <div>
                Verdict: CLASS {String(trace.final_prediction)}
                {verdict ? (verdict.crossed ? ' (above threshold)' : ' (below threshold)') : ''}
              </div>
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect log-odds flow.</span>
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

      {contribs.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Top log-odds contributors (wᵢ·xᵢ + b)
          </p>
          <ul className="flex flex-col gap-1">
            {contribs.map(([name, v]) => (
              <li key={name} className="flex items-center gap-2">
                <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                  {name}
                </span>
                <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                  {v >= 0 ? '+' : ''}
                  {v.toFixed(3)}
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

      <LogisticSurfaceSandbox />

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
