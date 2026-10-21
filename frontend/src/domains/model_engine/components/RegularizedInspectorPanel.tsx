import { useEffect } from 'react';
import { Scale } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useRegularizedSliceStore } from '../store/useRegularizedSliceStore';
import { RegularizedSurfaceSandbox } from './RegularizedSurfaceSandbox';

export const REGULARIZED_ALGORITHM_KEYS = ['regularized_linear', 'ridge_classifier'] as const;

/**
 * Phase 19 — Sandbox inspector for Ridge / Lasso regularized linear models:
 * penalty badge + alpha, sparsity gauge, norm telemetry, active/zeroed
 * feature pills, per-row waterfall and the 3D regularized-plane sandbox.
 */
export function RegularizedInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useRegularizedSliceStore();

  const isRegularized = currentModel !== null && REGULARIZED_ALGORITHM_KEYS.includes(
    (currentModel.algorithm_key ?? '') as 'regularized_linear' | 'ridge_classifier',
  );

  useEffect(() => {
    if (!isRegularized || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const hp = (currentModel.hyperparameters ?? {}) as Record<string, unknown>;
    const num = (v: unknown): number | null =>
      typeof v === 'number' && Number.isFinite(v) ? v : null;
    const alpha =
      num(hp['alpha']) ?? (typeof currentModel.intercept_or_base === 'number' ? null : null);
    const penalty =
      currentModel.algorithm_key === 'ridge_classifier'
        ? 'L2 (Ridge)'
        : /lasso/i.test(currentModel.display_name) ? 'L1 (Lasso)' : null;
    // Multi-output ridge reports (n_targets, D) weights via mean; target count
    // is unknown from metadata alone — refresh from the first trace instead.
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      weights: currentModel.features.map((f) => f.learned_weight_or_importance),
      intercept: currentModel.intercept_or_base,
      penaltyType: penalty,
      alpha,
      targetCount: 1,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRegularized, currentModel?.model_id]);

  if (!isRegularized || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const penalty = mech?.penalty_type ?? s.penaltyType ?? '—';
  const alpha = mech && Number.isFinite(mech.alpha) ? mech.alpha : (s.alpha ?? null);
  const activeCount = mech?.active_weight_count ?? null;
  const sparsity = mech && Number.isFinite(mech.sparsity_ratio) ? mech.sparsity_ratio : null;
  const l1 = mech && Number.isFinite(mech.l1_norm) ? mech.l1_norm : null;
  const l2 = mech && Number.isFinite(mech.l2_norm) ? mech.l2_norm : null;
  const zeroedSet = new Set(mech?.zeroed_indices ?? []);
  const isClassifier = (mech?.is_classifier ?? currentModel.algorithm_key === 'ridge_classifier') === true;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;
  const activeTotal = s.featureNames.length;
  const prunedTotal = mech?.zeroed_indices?.length ?? 0;

  const testSlice = async () => {
    await s.runTrace();
    await useRegularizedSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Regularized Linear Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Scale size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Regularized Plane & L1/L2 Weight Vectors
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_sparse_lasso')}
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
        Penalty: {penalty} | α: {alpha !== null ? alpha : '—'}
        {sparsity !== null && activeCount !== null
          ? ` | Sparsity: ${activeCount} Active / ${prunedTotal} Zeroed (${(sparsity * 100).toFixed(1)}% pruned)`
          : ''}
        {l1 !== null && l2 !== null ? ` | ‖w‖₁ ${l1.toFixed(3)} ‖w‖₂ ${l2.toFixed(3)}` : ''}
        {isClassifier ? ' | Pseudo-proba from Ridge score' : ''}
      </p>

      {sparsity !== null && activeCount !== null && (
        <div
          className="font-mono-tabular text-[11px]"
          style={{ color: 'var(--text-secondary)' }}
          aria-label="Sparsity gauge"
        >
          Sparsity Gauge: [{'█'.repeat(Math.round((1 - sparsity) * 12)).padEnd(12, '░')}] {activeCount} Active /{' '}
          {prunedTotal} Zeroed ({(sparsity * 100).toFixed(1)}% sparse)
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        {s.featureNames.map((n, i) => {
          const pruned = zeroedSet.has(i);
          const w = s.weights[i];
          return (
            <button
              key={`${n}-${i}`}
              title={pruned ? `${n} pruned (w=0)` : `${n} w=${typeof w === 'number' ? w.toFixed(4) : '—'}`}
              onClick={() => {
                // Quick-assign: first click sets X, second sets Y.
                const st = useRegularizedSliceStore.getState();
                if (st.axisX !== i && st.axisY !== i) st.setAxisX(i);
                else if (st.axisX === i) st.setAxisY(i);
                else st.setAxisX(i);
              }}
              className="font-mono-tabular rounded px-2 py-1"
              style={{
                border: pruned ? '1px dashed var(--border-strong)' : '1px solid var(--border-strong)',
                background: pruned ? 'transparent' : 'var(--surface-elevated)',
                color: pruned ? 'var(--text-muted)' : 'var(--text-primary)',
                textDecoration: pruned ? 'line-through' : 'none',
              }}
            >
              {pruned ? '[Pruned] ' : '[X/Y] '}{n}{typeof w === 'number' ? ` (${w >= 0 ? '+' : ''}${w.toFixed(2)})` : ''}
            </button>
          );
        })}
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
          {trace ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div>
                Predicted (ŷ): {typeof trace.final_prediction === 'number' ? trace.final_prediction.toFixed(3) : String(trace.final_prediction)}
              </div>
              <div>Intercept (b): {trace.base_value >= 0 ? '+' : ''}{trace.base_value.toFixed(3)}</div>
              {isClassifier && mech?.probabilities && (
                <div>
                  Pseudo-proba: [{mech.probabilities.map((p) => `${(p * 100).toFixed(1)}%`).join(' / ')}]
                </div>
              )}
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect w·x contributions.</span>
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
            Top contributors (wᵢ·xᵢ + b)
          </p>
          <ul className="flex flex-col gap-1">
            {contribs.map(([name, v]) => {
              const idx = s.featureNames.indexOf(name);
              const pruned = idx >= 0 && zeroedSet.has(idx);
              return (
                <li key={name} className="flex items-center gap-2">
                  <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                    {name}{pruned ? ' (L1 PRUNED)' : ''}
                  </span>
                  <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                    {v >= 0 ? '+' : ''}{v.toFixed(3)}
                  </span>
                  <span
                    aria-hidden
                    className="h-2 rounded"
                    style={{
                      width: `${Math.max(4, (Math.abs(v) / maxAbs) * 160)}px`,
                      background: pruned ? 'var(--text-muted)' : v >= 0 ? 'var(--ml-primary)' : 'var(--ml-class-1)',
                      opacity: pruned ? 0.4 : 0.8,
                    }}
                  />
                </li>
              );
            })}
          </ul>
          {s.phaseShift !== null && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-muted)' }}>
              GPU phase-shift offset (b + off-axis Σwx): {s.phaseShift.toFixed(3)}
            </p>
          )}
        </div>
      )}

      <RegularizedSurfaceSandbox />

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

      {activeTotal === 0 || (mech && mech.active_weight_count === 0) ? (
        <p className="text-[11px]" style={{ color: 'var(--ml-anomaly)' }} role="note">
          All features pruned by high L1 α — plane is flat at Z = b.
        </p>
      ) : null}
    </section>
  );
}
