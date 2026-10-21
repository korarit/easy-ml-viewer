import { useEffect } from 'react';
import { Mountain } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useGnbSliceStore } from '../store/useGnbSliceStore';
import { GnbBellSurfaceSandbox } from './GnbBellSurfaceSandbox';

export const GNB_ALGORITHM_KEYS = ['gaussian_nb'] as const;

/**
 * Phase 24 — Sandbox inspector for GaussianNB: class priors, Bayesian
 * posterior table, z-score & log-likelihood breakdown and the 3D
 * bivariate bell-hills sandbox.
 */
export function GnbBellInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useGnbSliceStore();

  const isGnb =
    currentModel !== null &&
    (GNB_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isGnb || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGnb, currentModel?.model_id]);

  if (!isGnb || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isBinary = mech?.is_binary !== false;
  const priors = mech?.class_priors ?? null;
  const counts = mech?.class_counts ?? null;
  const outlier = mech?.is_low_density_outlier === true;
  const minZ = mech && Number.isFinite(mech.min_z_score) ? (mech.min_z_score as number) : null;
  const winning = mech && Number.isFinite(mech.winning_class_idx) ? (mech.winning_class_idx as number) : null;

  const joint = mech?.joint_log_probs ?? [];
  const proba = mech?.probabilities ?? [];
  const order = joint.map((_, i) => i).sort((a, b) => (joint[b] ?? 0) - (joint[a] ?? 0));

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => {
        if (isBinary) return Math.abs(b[1]) - Math.abs(a[1]);
        return a[1] - b[1];
      })
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const zScores = mech?.winning_class_z_scores ?? [];
  const zNames = s.featureNames;

  const testSlice = async () => {
    await s.runTrace();
    await useGnbSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="GaussianNB Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Mountain size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Gaussian Bell Hills
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_iris_gnb')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="GaussianNB 3-class iris demo"
          >
            Load Iris Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_credit_gnb')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="GaussianNB binary demo (log-odds waterfall)"
          >
            Load Binary Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_credit_gnb_priors')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="GaussianNB with custom priors demo"
          >
            Load Priors Sample
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
        Classes: {s.classCount}
        {priors ? ` | Priors: ${priors.map((p) => p.toFixed(3)).join(' / ')}` : ''}
        {counts && counts.every(Number.isFinite) ? ` | Counts: ${counts.map((c) => Math.round(c)).join(' / ')}` : ''}
        {mech ? ` | var_smoothing: ${mech.var_smoothing}` : ''}
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
          Frame posterior telemetry
          {trace && mech ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div>
                Winner: Class {String(trace.final_prediction)}
                {winning !== null && proba[winning] !== undefined
                  ? ` (${(proba[winning] * 100).toFixed(1)}% posterior)`
                  : ''}
              </div>
              <div>
                Min z-distance: {minZ !== null ? `${minZ.toFixed(2)}σ` : '—'}
                {outlier ? ' ⚠ LOW DENSITY OUTLIER' : ''}
              </div>
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect posterior telemetry.</span>
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

      {outlier && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          LOW DENSITY OUTLIER: this row sits &gt; 4σ from every class center — the 100.0% posterior is
          a flat-plain artifact, not a confident hilltop.
        </p>
      )}

      {trace && mech && (
        <div
          className="overflow-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Bayesian posterior & bell-hill heights
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">Class</th>
                <th className="px-2 py-1 text-right">Posterior</th>
                <th className="px-2 py-1 text-right">Joint Log-Prob</th>
                <th className="px-2 py-1 text-right">Prior π</th>
              </tr>
            </thead>
            <tbody>
              {order.map((k) => (
                <tr
                  key={k}
                  style={{
                    borderTop: '1px solid var(--border-subtle)',
                    background: k === winning ? 'rgba(34,211,238,0.08)' : undefined,
                  }}
                >
                  <td className="px-2 py-1">Class {mech.classes?.[k] ?? k}{k === winning ? ' ★' : ''}</td>
                  <td className="px-2 py-1 text-right">{((proba[k] ?? 0) * 100).toFixed(1)}%</td>
                  <td className="px-2 py-1 text-right">{(joint[k] ?? 0).toFixed(3)}</td>
                  <td className="px-2 py-1 text-right">{(priors?.[k] ?? 0).toFixed(3)}</td>
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
            {isBinary
              ? 'Log-odds waterfall (class 1 vs class 0): base = log(π₁/π₀)'
              : `Log-likelihood waterfall (target class ${mech.resolved_class_idx}): base = log(πₖ)`}
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                {isBinary ? 'Prior log-ratio' : 'Prior log πₖ'}
              </span>
              <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                {trace.base_value >= 0 ? '+' : ''}{trace.base_value.toFixed(3)}
              </span>
            </li>
            {contribs.map(([name, v], i) => (
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
                {!isBinary && (() => {
                  const zi = s.featureNames.indexOf(name);
                  const z = zi >= 0 ? zScores[zi] : undefined;
                  return z !== undefined && Number.isFinite(z) ? (
                    <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
                      z={z.toFixed(2)}σ
                    </span>
                  ) : null;
                })()}
              </li>
            ))}
          </ul>
          {isBinary && zScores.length > 0 && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-muted)' }}>
              Z-distances (winner): {zNames.map((n, i) => `${n}=${(zScores[i] ?? 0).toFixed(2)}σ`).join(' · ')}
            </p>
          )}
        </div>
      )}

      <GnbBellSurfaceSandbox />

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
