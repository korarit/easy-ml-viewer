import { useEffect } from 'react';
import { GitBranch } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useTreeSliceStore } from '../store/useTreeSliceStore';
import { TreeSurfaceSandbox } from './TreeSurfaceSandbox';
import { leafBadge, stepLabel } from '../../stage3d/components/surfaces/treeSurfaceMath';

export const TREE_ALGORITHM_KEYS = ['decision_tree', 'decision_tree_regressor'] as const;

/**
 * Phase 27 — Sandbox inspector for single decision trees: depth badge,
 * active-path breadcrumb, leaf badge, exact node-delta waterfall and the
 * 3D stepped-voxel terrace sandbox.
 */
export function TreeInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useTreeSliceStore();

  const isTree =
    currentModel !== null &&
    (TREE_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isTree || !currentModel) return;
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
  }, [isTree, currentModel?.model_id]);

  if (!isTree || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isRegressor = s.isRegressor || mech?.is_regressor === true;
  const isBinary = mech?.is_binary !== false && !isRegressor;
  const steps = (mech?.active_path as Array<{
    node_id: number;
    next_node_id: number;
    feature_idx: number;
    feature_name: string;
    threshold: number;
    went_left: boolean;
    delta: number;
    impurity: number;
  }>) ?? [];

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => {
        if (isBinary || isRegressor) return Math.abs(b[1]) - Math.abs(a[1]);
        return a[1] - b[1];
      })
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await useTreeSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="Decision Tree Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <GitBranch size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Stepped Voxel Terraces
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_binary_tree')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="DecisionTreeClassifier binary demo"
          >
            Load Binary Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multi_tree')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="DecisionTreeClassifier 3-class demo"
          >
            Load Multi Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_reg_tree')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="DecisionTreeRegressor demo"
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
        Model: {mech ? `${mech.algorithm} · ${mech.criterion}` : '—'}
        {mech ? ` · depth ${mech.max_depth} · ${mech.node_count} nodes / ${mech.n_leaves} leaves` : ''}
        {mech?.calibrated_wrapper ? ' · ⚠ calibrated wrapper (trace shows inner tree)' : ''}
        {mech?.is_multi_output ? ' · multi-output: tracing output 0' : ''}
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
          Leaf badge
          {trace && mech ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              <div>
                {leafBadge(
                  mech.leaf_id as number,
                  mech.leaf_impurity as number,
                  mech.leaf_n_samples as number,
                  mech.leaf_expectation as number,
                  isRegressor,
                )}
              </div>
              <div>
                Prediction: {String(trace.final_prediction)}
                {!isRegressor && (mech.probabilities as number[]).length > 0
                  ? ` (${(((mech.probabilities as number[])[mech.resolved_class_idx as number] ?? 0) * 100).toFixed(1)}%)`
                  : ''}
              </div>
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect the active leaf.</span>
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

      {trace && mech && steps.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Active decision path (root → leaf #{mech.leaf_id as number})
          </p>
          <ol className="font-mono-tabular flex flex-col gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {steps.map((st) => (
              <li key={`${st.node_id}-${st.next_node_id}`} className="flex items-center gap-2">
                <span style={{ color: 'var(--ml-primary)' }}>#{st.node_id} → #{st.next_node_id}</span>
                <span>{stepLabel(st.feature_name, st.threshold, st.went_left, st.delta)}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {trace && mech && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Exact node-delta waterfall: base E(root) + ΣΔ = E(leaf)
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

      <TreeSurfaceSandbox />

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
