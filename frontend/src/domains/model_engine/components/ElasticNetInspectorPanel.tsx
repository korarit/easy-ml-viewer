import { useEffect } from 'react';
import { Blend } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useElasticNetSliceStore } from '../store/useElasticNetSliceStore';
import { ElasticNetSurfaceSandbox } from './ElasticNetSurfaceSandbox';

export const ELASTICNET_ALGORITHM_KEYS = ['elastic_net'] as const;

/**
 * Phase 20 — Sandbox inspector for ElasticNet / MultiTaskElasticNet:
 * L1/L2 tug-of-war gauge, λ₁/λ₂, penalty burdens, sparsity + convergence
 * badges, multi-task switcher, per-row waterfall with penalty shares,
 * and the 3D plane + constraint-ring sandbox.
 */
export function ElasticNetInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useElasticNetSliceStore();

  const isElasticNet =
    currentModel !== null &&
    (ELASTICNET_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isElasticNet || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const hp = (currentModel.hyperparameters ?? {}) as Record<string, unknown>;
    const num = (v: unknown): number | null =>
      typeof v === 'number' && Number.isFinite(v) ? v : null;
    // alpha_/l1_ratio_ winners when present (CV), else constructor values.
    const alpha = num(hp['alpha_']) ?? num(hp['alpha']);
    let l1 = num(hp['l1_ratio_']) ?? num(hp['l1_ratio']);
    if (Array.isArray(hp['l1_ratio_'])) l1 = num((hp['l1_ratio_'] as unknown[])[0]);
    if (Array.isArray(hp['l1_ratio'])) l1 = l1 ?? num((hp['l1_ratio'] as unknown[])[0]);
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      weights: currentModel.features.map((f) => f.learned_weight_or_importance),
      intercept: currentModel.intercept_or_base,
      alpha,
      l1Ratio: l1,
      taskCount: 1,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isElasticNet, currentModel?.model_id]);

  if (!isElasticNet || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const rho = mech && Number.isFinite(mech.l1_ratio) ? mech.l1_ratio : (s.l1Ratio ?? null);
  const alpha = mech && Number.isFinite(mech.alpha) ? mech.alpha : (s.alpha ?? null);
  const lambda1 = mech && Number.isFinite(mech.lambda_1) ? mech.lambda_1 : null;
  const lambda2 = mech && Number.isFinite(mech.lambda_2) ? mech.lambda_2 : null;
  const l1Burden = mech && Number.isFinite(mech.l1_penalty_burden) ? mech.l1_penalty_burden : null;
  const l2Burden = mech && Number.isFinite(mech.l2_penalty_burden) ? mech.l2_penalty_burden : null;
  const totalReg = mech && Number.isFinite(mech.total_regularization) ? mech.total_regularization : null;
  const activeCount = mech?.active_weight_count ?? null;
  const sparsity = mech && Number.isFinite(mech.sparsity_ratio) ? mech.sparsity_ratio : null;
  const nIter = mech?.n_iter ?? null;
  const maxIter = mech?.max_iter ?? null;
  const converged = mech ? !mech.convergence_warning : null;
  const positiveOnly = mech?.positive_only === true;
  const nTasks = mech?.n_tasks ?? s.taskCount ?? 1;
  const zeroedSet = new Set(mech?.zeroed_indices ?? []);
  const penaltyShares = mech?.penalty_shares ?? null;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;
  const prunedTotal = mech?.zeroed_indices?.length ?? 0;
  const rhoPct = rho !== null ? Math.round(rho * 100) : null;

  const testSlice = async () => {
    await s.runTrace();
    await useElasticNetSliceStore.getState().runSurface();
    // Refresh the task count from the trace (multitask reports n_tasks).
    const t = useElasticNetSliceStore.getState().traces[0]?.extra_mechanics;
    if (t && typeof t.n_tasks === 'number' && t.n_tasks > 1) {
      useElasticNetSliceStore.setState({ taskCount: t.n_tasks });
    }
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="ElasticNet Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Blend size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — ElasticNet Plane & L1/L2 Tug-of-War
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_housing_elasticnet')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Single-task ElasticNet demo"
          >
            Load Sample Data
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multitask_elasticnet')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="MultiTaskElasticNet demo (price + rent yield)"
          >
            Load MultiTask Sample
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

      {/* L1 / L2 Balance Gauge (Tug-of-War Bar) */}
      {rhoPct !== null ? (
        <div aria-label="L1 L2 balance gauge">
          <div
            className="font-mono-tabular flex items-center gap-1 text-[11px]"
            style={{ color: 'var(--text-secondary)' }}
          >
            <span style={{ color: '#38bdf8' }}>◀ L2 Ridge ({100 - rhoPct}%)</span>
            <span className="mx-1">|</span>
            <span style={{ color: '#e879f9' }}>({rhoPct}%) L1 Lasso ▶</span>
            <span style={{ color: 'var(--text-muted)' }}>
              α={alpha !== null ? alpha.toFixed(4) : '—'} ρ={rho !== null ? rho.toFixed(2) : '—'}
            </span>
          </div>
          <div
            className="mt-1 flex h-3 w-full overflow-hidden rounded"
            style={{ border: '1px solid var(--border-strong)' }}
            role="img"
            aria-label={`L1 ratio ${rhoPct} percent`}
          >
            <div style={{ width: `${100 - rhoPct}%`, background: '#38bdf8', opacity: 0.85 }} />
            <div style={{ width: `${rhoPct}%`, background: '#e879f9', opacity: 0.85 }} />
          </div>
          <p className="font-mono-tabular mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            Effective λ₁ (L1): {lambda1 !== null ? lambda1.toFixed(4) : '—'}
            {'  '}λ₂ (L2): {lambda2 !== null ? lambda2.toFixed(4) : '—'}
            {'  '}Burden L1={l1Burden !== null ? l1Burden.toFixed(3) : '—'} L2=
            {l2Burden !== null ? l2Burden.toFixed(3) : '—'} Total=
            {totalReg !== null ? totalReg.toFixed(3) : '—'}
          </p>
        </div>
      ) : (
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          Penalty: ElasticNet (L1+L2) | α: {alpha !== null ? alpha : '—'} — run a trace to resolve ρ.
        </p>
      )}

      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {sparsity !== null && activeCount !== null
          ? `Sparsity: ${activeCount} Active / ${prunedTotal} Pruned (${(sparsity * 100).toFixed(1)}% pruned)`
          : 'Sparsity: —'}
        {nIter !== null && maxIter !== null
          ? ` | n_iter: ${nIter} / ${maxIter} ${converged === false ? '⚠ Convergence Warning: hit max_iter' : '✓ converged'}`
          : ''}
        {positiveOnly ? ' | Non-negative weights (w ≥ 0)' : ''}
        {nTasks > 1 ? ` | MultiTask: ${nTasks} outputs` : ''}
      </p>

      {sparsity !== null && activeCount !== null && (
        <div
          className="font-mono-tabular text-[11px]"
          style={{ color: 'var(--text-secondary)' }}
          aria-label="Sparsity gauge"
        >
          Sparsity Gauge: [{'█'.repeat(Math.round((1 - sparsity) * 12)).padEnd(12, '░')}] {activeCount} Active /{' '}
          {prunedTotal} Pruned ({(sparsity * 100).toFixed(1)}% sparse)
        </div>
      )}

      {/* Task selector for MultiTaskElasticNet */}
      {nTasks > 1 && (
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Task selector">
          <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
            Task:
          </span>
          {Array.from({ length: nTasks }, (_, i) => (
            <button
              key={i}
              onClick={() => void s.setTaskIdx(i)}
              className="font-mono-tabular rounded px-2 py-1 text-[11px]"
              style={{
                border: '1px solid var(--border-strong)',
                background: s.taskIdx === i ? 'var(--ml-primary)' : 'var(--surface-elevated)',
                color: s.taskIdx === i ? '#fff' : 'var(--text-primary)',
              }}
            >
              task {i}
            </button>
          ))}
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
                const st = useElasticNetSliceStore.getState();
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
              {nTasks > 1 && <div>Task: {mech?.active_task_idx} / {nTasks}</div>}
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

      {converged === false && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          ⚠ Coordinate Descent hit max_iter={maxIter} without converging — try a larger α or max_iter.
        </p>
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
              const share = idx >= 0 && penaltyShares && idx < penaltyShares.length ? penaltyShares[idx] : null;
              return (
                <li key={name} className="flex items-center gap-2">
                  <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                    {name}{pruned ? ' (JOINT PRUNED)' : ''}
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
                  <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
                    {share !== null ? `(Reg: ${share.toFixed(3)})` : '(Reg: N/A)'}
                  </span>
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

      <ElasticNetSurfaceSandbox />

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

      {s.featureNames.length === 0 || (mech && mech.active_weight_count === 0) ? (
        <p className="text-[11px]" style={{ color: 'var(--ml-anomaly)' }} role="note">
          All features pruned by high α — plane is flat at Z = b.
        </p>
      ) : null}
    </section>
  );
}
