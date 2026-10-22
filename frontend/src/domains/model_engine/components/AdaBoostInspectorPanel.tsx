import { useEffect, useMemo } from 'react';
import { TrendingUp } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useAdaBoostSliceStore } from '../store/useAdaBoostSliceStore';
import { AdaBoostSurfaceSandbox } from './AdaBoostSurfaceSandbox';

export const ADABOOST_ALGORITHM_KEYS = ['adaboost', 'adaboost_regressor'] as const;

/**
 * Phase 30 — Sandbox inspector for adaboost cascades: staged boosting
 * curve with decision-threshold crossing, weak-learner weight/error
 * table, ensemble waterfall and the 3D cascade ladder.
 */
export function AdaBoostInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useAdaBoostSliceStore();

  const isAdaBoost =
    currentModel !== null &&
    (ADABOOST_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isAdaBoost || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      isRegressor: currentModel.task_type === 'regression',
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAdaBoost, currentModel?.model_id]);

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isRegressor = s.isRegressor || mech?.is_regressor === true;
  const learners = (mech?.top_influential_learners ?? []) as Array<{
    stage: number;
    weight: number;
    error: number;
    feature_idx: number;
    feature_name: string | null;
    threshold: number;
    delta: number;
    cumulative: number;
  }>;

  const curve = useMemo(() => {
    if (!mech) return null;
    const xs = mech.sampled_stage_indices as number[];
    const ys = mech.staged_trajectory as number[];
    if (xs.length === 0) return null;
    const W = 320;
    const H = 120;
    const pad = 8;
    const lo = Math.min(...ys, isRegressor ? Math.min(...ys) : 0);
    const hi = Math.max(...ys, isRegressor ? Math.max(...ys) : 1);
    const span = hi - lo || 1;
    const px = (i: number) => pad + (i / Math.max(1, xs.length - 1)) * (W - pad * 2);
    const py = (v: number) => H - pad - ((v - lo) / span) * (H - pad * 2);
    const line = ys.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
    const crossed = (mech.crossed_threshold_stage ?? null) as number | null;
    return { W, H, line, py, crossed, lo, hi, threshY: !isRegressor ? py(0.5) : null };
  }, [mech, isRegressor]);

  if (!isAdaBoost || !currentModel) return null;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;
  const total = (mech?.n_estimators_actual ?? 0) as number;

  const testSlice = async () => {
    await s.runTrace();
    await useAdaBoostSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="AdaBoost Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <TrendingUp size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Sequential Weak-Learner Cascade
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_binary_adaboost')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="AdaBoostClassifier binary demo"
          >
            Load Binary Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multi_adaboost')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="AdaBoostClassifier 3-class demo"
          >
            Load Multi Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_reg_adaboost')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="AdaBoostRegressor demo"
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
        Model: {mech ? `${mech.algorithm} · ${total} learners` : '—'}
        {mech?.early_terminated ? ' · ⚠ early-terminated' : ''}
        {mech?.learning_rate != null ? ` · lr ${(mech.learning_rate as number).toFixed(2)}` : ''}
        {mech && (mech.crossed_threshold_stage as number | null) != null && !isRegressor
          ? ` · crossed 0.5 @ stage ${mech.crossed_threshold_stage}`
          : ''}
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
        <div className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Staged boosting curve
          {trace && mech && curve ? (
            <svg
              viewBox={`0 0 ${curve.W} ${curve.H}`}
              className="rounded"
              style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)' }}
              role="img"
              aria-label="Staged cumulative prediction curve"
            >
              {curve.threshY != null && (
                <line x1={0} x2={curve.W} y1={curve.threshY} y2={curve.threshY} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1} />
              )}
              <path d={curve.line} fill="none" stroke="var(--ml-primary)" strokeWidth={2} />
              <text x={curve.W - 4} y={curve.threshY != null ? curve.threshY - 4 : 12} textAnchor="end" fontSize={9} fill="#94a3b8">
                {trace.final_prediction != null ? `S_M=${(mech.staged_trajectory as number[]).slice(-1)[0].toFixed(3)}` : ''}
              </text>
            </svg>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to draw the staged curve.</span>
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

      {trace && mech && learners.length > 0 && (
        <div
          className="overflow-x-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Top influential weak learners (|ΔS| leaders)
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">Stage</th>
                <th className="px-2 py-1 text-right">α weight</th>
                <th className="px-2 py-1 text-right">train err</th>
                <th className="px-2 py-1 text-left">Split rule</th>
                <th className="px-2 py-1 text-right">ΔS</th>
                <th className="px-2 py-1 text-right">Cumulative</th>
              </tr>
            </thead>
            <tbody>
              {learners.map((l) => (
                <tr key={l.stage} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                  <td className="px-2 py-1">#{l.stage + 1}</td>
                  <td className="px-2 py-1 text-right">{l.weight.toFixed(3)}</td>
                  <td className="px-2 py-1 text-right">{l.error.toFixed(3)}</td>
                  <td className="px-2 py-1">
                    {l.feature_name != null ? `${l.feature_name} ≤ ${l.threshold}` : `non-tree base (Δ distributed)`}
                  </td>
                  <td className="px-2 py-1 text-right" style={{ color: l.delta >= 0 ? 'var(--ml-primary)' : 'var(--ml-class-1)' }}>
                    {l.delta >= 0 ? '+' : ''}{l.delta.toFixed(4)}
                  </td>
                  <td className="px-2 py-1 text-right">{l.cumulative.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {trace && mech && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Ensemble waterfall: base S₀ + ΣΔS = staged final
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                S₀ (base)
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
        </div>
      )}

      <AdaBoostSurfaceSandbox />

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
