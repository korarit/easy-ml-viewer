import { useEffect, useMemo } from 'react';
import { Boxes } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useDiscreteNbSliceStore } from '../store/useDiscreteNbSliceStore';
import { DiscreteNbSurfaceSandbox } from './DiscreteNbSurfaceSandbox';
import {
  filterContributions,
  presenceLabel,
  topDiscriminative,
} from '../../stage3d/components/surfaces/discreteNbSurfaceMath';

export const DISCRETE_NB_ALGORITHM_KEYS = ['discrete_nb'] as const;

/**
 * Phase 25 — Sandbox inspector for Multinomial / Bernoulli / Complement /
 * Categorical NB: variant + smoothing badge, posterior tower table,
 * presence/absence waterfall and the 3D non-negative surface + towers.
 */
export function DiscreteNbInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useDiscreteNbSliceStore();

  const isDiscreteNb =
    currentModel !== null &&
    (DISCRETE_NB_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isDiscreteNb || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDiscreteNb, currentModel?.model_id]);

  const liveValues = useMemo(() => {
    const vals = s.rowInput
      .split(/[\s,;]+/)
      .map((v) => v.trim())
      .filter(Boolean)
      .map(Number);
    return vals.length === s.featureNames.length && vals.every(Number.isFinite) ? vals : [];
  }, [s.rowInput, s.featureNames.length]);

  if (!isDiscreteNb || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isBinary = mech?.is_binary !== false;
  const variant = mech?.nb_variant ?? null;
  const isBernoulli = variant === 'bernoulli';
  const joint = mech?.joint_log_probs ?? [];
  const proba = mech?.probabilities ?? [];
  const order = joint.map((_, i) => i).sort((a, b) => (joint[b] ?? 0) - (joint[a] ?? 0));
  const winning = mech && Number.isFinite(mech.winning_class_idx) ? (mech.winning_class_idx as number) : null;

  const visibleContribs = trace
    ? filterContributions(
        trace.feature_contributions,
        (mech?.presence_flags as Record<string, string>) ?? {},
        isBernoulli ? s.presenceFilter : 'all',
        liveValues,
        s.featureNames,
        (mech?.binarize as number | null | undefined) ?? null,
      )
    : {};
  const contribEntries = Object.entries(visibleContribs).sort((a, b) => {
    if (isBinary) return Math.abs(b[1]) - Math.abs(a[1]);
    return a[1] - b[1];
  });
  const maxAbs = contribEntries.length > 0 ? Math.max(...contribEntries.map(([, v]) => Math.abs(v)), 1e-9) : 1;
  const topRanked = trace ? topDiscriminative(trace.feature_contributions, 5) : [];

  const testSlice = async () => {
    await s.runTrace();
    await useDiscreteNbSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="DiscreteNB Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Boxes size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Log-Likelihood Towers
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_text_multinomial')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="MultinomialNB word-count demo"
          >
            Load Multinomial
          </button>
          <button
            onClick={() => void loadSampleModel('sample_binary_bernoulli')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="BernoulliNB binary-presence demo"
          >
            Load Bernoulli
          </button>
          <button
            onClick={() => void loadSampleModel('sample_imbalanced_complement')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="ComplementNB imbalanced-counts demo"
          >
            Load Complement
          </button>
          <button
            onClick={() => void loadSampleModel('sample_categorical_catnb')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="CategoricalNB integer-category demo"
          >
            Load Categorical
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
        Variant: {mech ? `${mech.algorithm} · α=${mech.alpha}` : '—'}
        {isBernoulli && mech ? ` · binarize=${mech.binarize === null ? 'None (pre-binarized)' : mech.binarize}` : ''}
        {mech ? ` · Classes: ${s.classCount}` : ''}
        {mech?.requires_non_negative_input ? ' · count domain X≥0' : ''}
      </p>

      <div className="grid gap-2 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Input row (comma-separated, {s.featureNames.length} values, ≥ 0)
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
          Frame tower telemetry
          {trace && mech ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div>
                Winner: Class {String(trace.final_prediction)}
                {winning !== null && proba[winning] !== undefined
                  ? ` (${(proba[winning] * 100).toFixed(1)}% posterior)`
                  : ''}
              </div>
              <div>
                Tops: {topRanked.map((t) => `${t.name}=${t.value >= 0 ? '+' : ''}${t.value.toFixed(2)}`).join(' · ') || '—'}
              </div>
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect tower telemetry.</span>
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

      {mech?.unseen_category_clamped === true && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          UNSEEN CATEGORY CLAMPED: a grid/test value exceeded the trained category range and was
          clipped to the nearest known category.
        </p>
      )}

      {trace && mech && (
        <div
          className="overflow-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            3D probability towers & log-likelihood trace
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">Class</th>
                <th className="px-2 py-1 text-left">Tower</th>
                <th className="px-2 py-1 text-right">Posterior</th>
                <th className="px-2 py-1 text-right">Joint Log-P</th>
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
                  <td className="px-2 py-1">Class {mech.classes?.[k] ?? k}{k === winning ? ' ♛' : ''}</td>
                  <td className="px-2 py-1">
                    <span
                      aria-hidden
                      className="inline-block h-2 rounded"
                      style={{ width: `${Math.max(6, (proba[k] ?? 0) * 160)}px`, background: 'var(--ml-primary)', opacity: 0.85 }}
                    />
                  </td>
                  <td className="px-2 py-1 text-right">{((proba[k] ?? 0) * 100).toFixed(1)}%</td>
                  <td className="px-2 py-1 text-right">{(joint[k] ?? 0).toFixed(3)}</td>
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
              ? `Log-odds contributions (class 1 vs 0): base = ${variant === 'complement' ? '0.0' : 'log(π₁/π₀)'}`
              : `Log-likelihood contributions (target class ${mech.resolved_class_idx}): base = ${variant === 'complement' ? '0.0' : 'log πₖ'}`}
            {isBernoulli ? ` · showing ${s.presenceFilter === 'all' ? 'presence (1) + absence (0)' : 'presence (1) only'}` : ''}
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                {isBinary ? (variant === 'complement' ? 'Zero base' : 'Prior log-ratio') : 'Prior log πₖ'}
              </span>
              <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                {trace.base_value >= 0 ? '+' : ''}{trace.base_value.toFixed(3)}
              </span>
            </li>
            {contribEntries.map(([name, v]) => {
              const flag = isBernoulli
                ? ((mech.presence_flags as Record<string, string> | undefined)?.[name] ??
                  presenceLabel(liveValues[s.featureNames.indexOf(name)] ?? 0, (mech.binarize as number | null | undefined) ?? null))
                : null;
              return (
                <li key={name} className="flex items-center gap-2">
                  <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }} title={flag ? `${name} (${flag})` : name}>
                    {name}{flag ? (flag === 'present' ? ' =1' : ' =0') : ''}
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
              );
            })}
          </ul>
          {mech.contributions_pruned === true && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-muted)' }}>
              High-dimensional vocab: showing top-30 +/− of {s.featureNames.length} features · hidden tail{' '}
              {mech.pruned_remainder_sum >= 0 ? '+' : ''}{(mech.pruned_remainder_sum as number).toFixed(3)} (
              {mech.pruned_feature_count} features folded in, waterfall identity preserved)
            </p>
          )}
        </div>
      )}

      <DiscreteNbSurfaceSandbox />

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
