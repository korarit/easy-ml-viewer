import { useEffect } from 'react';
import { Hexagon } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useDiscriminantSliceStore } from '../store/useDiscriminantSliceStore';
import { DiscriminantSurfaceSandbox } from './DiscriminantSurfaceSandbox';

export const DISCRIMINANT_ALGORITHM_KEYS = ['discriminant'] as const;

/**
 * Phase 26 — Sandbox inspector for LDA/QDA: covariance badge, paradox
 * alert, Mahalanobis-vs-Euclidean table, linear/quadratic waterfall and
 * the 3D tilted-ellipsoid sandbox.
 */
export function DiscriminantInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useDiscriminantSliceStore();

  const isDiscriminant =
    currentModel !== null &&
    (DISCRIMINANT_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isDiscriminant || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDiscriminant, currentModel?.model_id]);

  if (!isDiscriminant || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isBinary = mech?.is_binary !== false;
  const isQda = mech?.discriminant === 'qda';
  const priors = mech?.priors ?? null;
  const paradox = mech?.paradox_detected === true;
  const winning = mech && Number.isFinite(mech.winning_class_idx) ? (mech.winning_class_idx as number) : null;

  const joint = (mech?.decision_scores as number[] | undefined) ?? [];
  const proba = mech?.probabilities ?? [];
  const mahal = mech?.mahalanobis_distances ?? [];
  const euclid = mech?.euclidean_distances ?? [];
  const order = joint.map((_, i) => i).sort((a, b) => (joint[b] ?? 0) - (joint[a] ?? 0));

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => {
        if (isBinary) return Math.abs(b[1]) - Math.abs(a[1]);
        return a[1] - b[1];
      })
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await useDiscriminantSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Discriminant Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Hexagon size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Covariance Ellipsoids
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_binary_lda')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LDA binary demo (paradox)"
          >
            Load LDA Binary
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multi_qda')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="QDA 3-class demo (tilted ellipsoids)"
          >
            Load QDA Multi
          </button>
          <button
            onClick={() => void loadSampleModel('sample_lsqr_lda')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LDA lsqr + store_covariance demo"
          >
            Load LDA lsqr
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
        Model: {mech ? `${mech.algorithm} · solver=${mech.solver}` : '—'}
        {mech ? ` · reg_param=${mech.reg_param}` : ''}
        {mech?.is_covariance_approximated ? ' · ⚠ covariance approximated' : ''}
        {mech && !mech.is_covariance_approximated ? ' · Exact covariance' : ''}
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
          Frame distance telemetry
          {trace && mech ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div>
                Winner: Class {String(trace.final_prediction)}
                {winning !== null && proba[winning] !== undefined
                  ? ` (${(proba[winning] * 100).toFixed(1)}% posterior)`
                  : ''}
              </div>
              <div>
                Mahal-nearest: Class {mech.nearest_mahalanobis_class} · Euclid-nearest: Class{' '}
                {mech.nearest_euclidean_class}
              </div>
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect distance telemetry.</span>
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

      {paradox && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          ⚡ PARADOX ACTIVE: the Euclidean-nearest class is NOT the Mahalanobis-nearest class — the
          ellipse stretch overrules the straight-line distance.
        </p>
      )}

      {trace && mech && (
        <div
          className="overflow-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Mahalanobis vs Euclidean distance comparison
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">Class</th>
                <th className="px-2 py-1 text-right">Prior</th>
                <th className="px-2 py-1 text-right">Euclidean</th>
                <th className="px-2 py-1 text-right">Mahalanobis</th>
                <th className="px-2 py-1 text-right">Posterior</th>
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
                  <td className="px-2 py-1 text-right">{(priors?.[k] ?? 0).toFixed(3)}</td>
                  <td className="px-2 py-1 text-right">{(euclid[k] ?? 0).toFixed(3)}</td>
                  <td className="px-2 py-1 text-right">{(mahal[k] ?? 0).toFixed(3)}</td>
                  <td className="px-2 py-1 text-right">{((proba[k] ?? 0) * 100).toFixed(1)}%</td>
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
            {isQda
              ? `Quadratic precision waterfall (target class ${mech.resolved_class_idx})`
              : isBinary
                ? 'Linear waterfall (class 1 vs 0): base = intercept'
                : `Linear waterfall (target class ${mech.resolved_class_idx}): base = interceptₖ`}
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                {isQda ? '−0.5 log|Σ| + log π' : 'Intercept'}
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

      <DiscriminantSurfaceSandbox />

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
