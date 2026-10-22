import { useMemo } from 'react';
import { Boxes } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useOnnxClassicalSliceStore } from '../store/useOnnxClassicalSliceStore';
import { OnnxClassicalStageRouter } from './OnnxClassicalStageRouter';
import { OnnxGraphInspectorCard } from './OnnxGraphInspectorCard';

export const ONNX_CLASSICAL_ALGORITHM_KEYS = ['onnx_classical', 'onnx_classical_regressor'] as const;

/**
 * Phase 38 — Sandbox inspector for ONNX classical ML: domain badge, operator
 * mini-graph, input-signature table, Input x Gradient / path-share waterfall
 * and the auto-morphing 3D stage.
 */
export function OnnxClassicalInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useOnnxClassicalSliceStore();

  const isOnnx =
    currentModel !== null &&
    (ONNX_CLASSICAL_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  const bindIfNeeded = () => {
    if (!isOnnx || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
    });
  };
  bindIfNeeded();

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isRegressor = mech?.is_regressor === true;

  const contribs = useMemo(() => {
    if (!trace) return [];
    return Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [trace]);
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await useOnnxClassicalSliceStore.getState().runSurface();
  };

  if (!isOnnx || !currentModel) return null;

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="ONNX Classical Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Boxes size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — {mech?.domain_badge ?? 'ONNX ai.onnx.ml Classical'}
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_rf_onnx')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="RandomForest → TreeEnsembleClassifier (fraud)"
          >
            Load Tree RF
          </button>
          <button
            onClick={() => void loadSampleModel('sample_logreg_onnx')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LogisticRegression → LinearClassifier (default)"
          >
            Load Linear
          </button>
          <button
            onClick={() => void loadSampleModel('sample_svm_onnx')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SVC RBF → SVMClassifier (rings)"
          >
            Load SVM
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
        Model:{' '}
        {mech
          ? `${mech.detected_operator} · family ${mech.operator_family} · ${mech.attribution_method}`
          : '—'}
        {mech?.has_missing_values ? ' · ◆ NaN in row' : ''}
        {mech?.non_numeric_columns && mech.non_numeric_columns.length > 0
          ? ` · ◆ string tensors [${mech.non_numeric_columns.join(', ')}] get zero-gradient attribution`
          : ''}
      </p>

      <div className="grid gap-2 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Input row (comma-separated, {s.featureNames.length} values; strings pass through)
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
        <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Target class index
          <input
            type="number"
            min={0}
            value={s.classIdx}
            onChange={(e) => s.setClassIdx(Number(e.target.value))}
            className="font-mono-tabular rounded px-2 py-1.5"
            style={{
              border: '1px solid var(--border-strong)',
              background: 'var(--surface-elevated)',
              color: 'var(--text-primary)',
            }}
          />
        </label>
      </div>

      <OnnxGraphInspectorCard mech={mech} />

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
          <p className="font-mono-tabular mb-2" style={{ color: 'var(--text-secondary)' }}>
            Output:{' '}
            {isRegressor
              ? Number(mech.predicted_value ?? trace.final_prediction).toFixed(4)
              : `${(mech.probabilities as number[]).map((p) => p.toFixed(3)).join(' / ')}`} · base{' '}
            {trace.base_value.toFixed(4)} + Σcontrib {contribs.reduce((a, [, v]) => a + v, 0).toFixed(4)} ·{' '}
            {mech.attribution_method}
          </p>
          <div className="flex flex-col gap-1">
            {contribs.map(([name, v]) => (
              <div key={name} className="flex items-center gap-2">
                <span className="font-mono-tabular w-32 truncate" style={{ color: 'var(--text-secondary)' }} title={name}>
                  {name}
                </span>
                <div className="relative h-3 flex-1 rounded" style={{ background: 'var(--surface-elevated)' }}>
                  <div
                    className="absolute top-0 h-3 rounded"
                    style={{
                      left: v < 0 ? `${50 + (v / maxAbs) * 50}%` : '50%',
                      width: `${(Math.abs(v) / maxAbs) * 50}%`,
                      background: v >= 0 ? 'var(--ml-positive)' : 'var(--ml-negative)',
                    }}
                  />
                  <div className="absolute left-1/2 top-0 h-3 w-px" style={{ background: 'var(--border-strong)' }} />
                </div>
                <span className="font-mono-tabular w-20 text-right" style={{ color: 'var(--text-primary)' }}>
                  {v >= 0 ? '+' : ''}{v.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
          {(mech.decision_path?.length ?? 0) > 0 && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
              decision_path: {mech.decision_path!.slice(0, 4).map((st) => `T${st.tree_id}#${st.node_id}(f${st.feature_idx}${st.went_true ? '≤' : '>'}${st.threshold.toFixed(2)})`).join(' → ')}
              {mech.decision_path_truncated ? ' …' : ''} · leaves {mech.active_leaf_count}/{mech.n_trees}
            </p>
          )}
          {mech.kernel_type && (
            <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
              kernel {mech.kernel_type} params [{(mech.kernel_params ?? []).map((p) => p.toFixed(3)).join(', ')}] ·{' '}
              {mech.n_support_vectors} support vectors
            </p>
          )}
        </div>
      )}

      <OnnxClassicalStageRouter />
    </section>
  );
}
