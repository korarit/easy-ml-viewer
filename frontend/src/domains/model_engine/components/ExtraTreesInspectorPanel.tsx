import { useEffect } from 'react';
import { Shuffle } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useExtraTreesSliceStore } from '../store/useExtraTreesSliceStore';
import { ExtraTreesSurfaceSandbox } from './ExtraTreesSurfaceSandbox';
import {
  etVoteShares,
  extraTreesStepLabel,
  haloLabel,
} from '../../stage3d/components/surfaces/extraTreesSurfaceMath';

export const EXTRA_TREES_ALGORITHM_KEYS = ['extra_trees', 'extra_trees_regressor'] as const;

/**
 * Phase 29 — Sandbox inspector for extra-randomized forests: uncertainty
 * halo badge, vote-distribution bar, contribution error bars (±1 SD),
 * tree spotlight and the 3D halo arena.
 */
export function ExtraTreesInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useExtraTreesSliceStore();

  const isExtraTrees =
    currentModel !== null &&
    (EXTRA_TREES_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isExtraTrees || !currentModel) return;
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
  }, [isExtraTrees, currentModel?.model_id]);

  if (!isExtraTrees || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isRegressor = s.isRegressor || mech?.is_regressor === true;
  const isBinary = mech?.is_binary !== false && !isRegressor;
  const treePreds = (mech?.tree_predictions_sample ?? []) as number[];
  const [posShare, negShare] = etVoteShares(treePreds);
  const total = (mech?.n_estimators_total ?? 0) as number;
  const exported = (mech?.n_estimators_exported ?? 0) as number;
  const cstd = (mech?.contribution_std ?? {}) as Record<string, number>;
  const halo = (mech?.uncertainty_radius ?? 0) as number;
  const selectedPred = treePreds[Math.min(s.selectedTree, Math.max(0, treePreds.length - 1))] ?? null;
  const firstPath = (mech?.first_tree_active_path ?? []) as Array<{
    node_id: number;
    next_node_id: number;
    feature_idx: number;
    feature_name: string;
    threshold: number;
    went_left: boolean;
    delta: number;
  }>;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => {
        if (isBinary || isRegressor) return Math.abs(b[1]) - Math.abs(a[1]);
        return a[1] - b[1];
      })
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;
  const maxStd = Math.max(0, ...Object.values(cstd).map((v) => Math.abs(v)), 1e-9);

  const testSlice = async () => {
    await s.runTrace();
    await useExtraTreesSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Extra Trees Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Shuffle size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Randomized Terraces & Uncertainty Halo
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_binary_extratrees')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="ExtraTreesClassifier binary demo"
          >
            Load Binary Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multi_extratrees')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="ExtraTreesClassifier 3-class demo"
          >
            Load Multi Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_reg_extratrees')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="ExtraTreesRegressor demo"
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
        Model: {mech ? `${mech.algorithm} · ${total} trees (${exported} exported)` : '—'}
        {mech ? ` · bootstrap ${String((mech as unknown as { bootstrap?: boolean }).bootstrap ?? false)}` : ''}
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
          Uncertainty halo
          {trace && mech ? (
            <div className="rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div className="font-mono-tabular">
                Halo r: {(halo as number).toFixed(3)} ({haloLabel(halo as number)}) · std{' '}
                {(mech.inter_tree_std as number).toFixed(3)} · mean{' '}
                {(mech.inter_tree_mean as number).toFixed(3)}
              </div>
              <div className="font-mono-tabular">
                90% tree range: [{(mech.prediction_p05 as number).toFixed(3)} ..{' '}
                {(mech.prediction_p95 as number).toFixed(3)}] · pred {String(trace.final_prediction)}
              </div>
              <div className="mt-1 flex h-2 w-full overflow-hidden rounded" aria-hidden>
                <span style={{ width: `${posShare * 100}%`, background: 'var(--ml-primary)' }} />
                <span style={{ width: `${negShare * 100}%`, background: '#f59e0b' }} />
              </div>
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect the halo.</span>
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
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Tree spotlight — #{Math.min(s.selectedTree, Math.max(0, treePreds.length - 1))} of {total}
            {selectedPred != null ? ` · T = ${selectedPred.toFixed(3)}` : ''} · leaf #
            {((mech.leaf_indices_sample ?? []) as number[])[Math.min(s.selectedTree, Math.max(0, treePreds.length - 1))] ?? '—'}
          </p>
          <label className="flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
            Spotlight tree
            <input
              aria-label="Spotlight tree"
              type="range"
              min={0}
              max={Math.max(0, treePreds.length - 1)}
              value={Math.min(s.selectedTree, Math.max(0, treePreds.length - 1))}
              onChange={(e) => s.setSelectedTree(Number(e.target.value))}
              className="w-48"
            />
          </label>
          {firstPath.length > 0 && (
            <ol className="font-mono-tabular mt-2 flex flex-col gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {firstPath.slice(0, 8).map((st) => (
                <li key={`${st.node_id}-${st.next_node_id}`} className="flex items-center gap-2">
                  <span style={{ color: 'var(--ml-primary)' }}>#{st.node_id} → #{st.next_node_id}</span>
                  <span>{extraTreesStepLabel(st.feature_name, st.threshold, st.went_left, st.delta)}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}

      {trace && mech && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Ensemble waterfall + variance: base E(root) + ΣΦ = ExtraTrees mean (±1 SD whiskers)
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                E(root)
              </span>
              <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                {trace.base_value >= 0 ? '+' : ''}{trace.base_value.toFixed(3)}
              </span>
            </li>
            {contribs.map(([name, v]) => {
              const sd = Math.abs(cstd[name] ?? 0);
              return (
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
                  <span className="font-mono-tabular" style={{ color: 'var(--text-muted)', minWidth: 110 }}>
                    ±{sd.toFixed(3)}
                    <span
                      aria-hidden
                      className="ml-1 inline-block h-1 rounded"
                      style={{ width: `${Math.max(2, (sd / maxStd) * 48)}px`, background: 'var(--text-muted)', opacity: 0.7 }}
                    />
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <ExtraTreesSurfaceSandbox />

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
