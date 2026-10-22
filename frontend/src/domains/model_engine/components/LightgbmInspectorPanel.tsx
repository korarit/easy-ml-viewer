import { useEffect, useMemo } from 'react';
import { Sprout } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useLightGbmSliceStore } from '../store/useLightGbmSliceStore';
import { LightgbmSurfaceSandbox } from './LightgbmSurfaceSandbox';

export const LIGHTGBM_ALGORITHM_KEYS = ['lightgbm', 'lightgbm_regressor'] as const;

/**
 * Phase 34 — Sandbox inspector for LightGBM: staged linked curve, Exact
 * C++ TreeSHAP waterfall, leaf-depth gauge, top-SHAP table, split-line
 * context, metric switcher and the asymmetric leaf-wise 3D stage.
 */
export function LightgbmInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useLightGbmSliceStore();

  const isLgbm =
    currentModel !== null &&
    (LIGHTGBM_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isLgbm || !currentModel) return;
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
  }, [isLgbm, currentModel?.model_id]);

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isRegressor = s.isRegressor || mech?.is_regressor === true;
  const topShap = (mech?.top_shap_features ?? []) as Array<{
    rank: number;
    feature_idx: number;
    feature_name: string;
    shap_value: number;
  }>;
  const importances = (mech?.feature_importances ?? {}) as Record<string, number[]>;
  const metricVals = (importances[s.importanceMetric] ?? []) as number[];
  const maxMetric = metricVals.length > 0 ? Math.max(...metricVals, 1e-9) : 1;

  const curve = useMemo(() => {
    if (!mech) return null;
    const xs = mech.sampled_iteration_indices as number[];
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
    return { W, H, line, threshY: !isRegressor ? py(0.5) : null };
  }, [mech, isRegressor]);

  if (!isLgbm || !currentModel) return null;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;
  const total = (mech?.n_iterations ?? 0) as number;
  const depthStat = (mech?.tree_depth_stats?.[0] ?? null) as {
    min_leaf_depth: number;
    max_leaf_depth: number;
    num_leaves: number;
  } | null;

  const testSlice = async () => {
    await s.runTrace();
    await useLightGbmSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="LightGBM Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Sprout size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — Leaf-Wise Exact TreeSHAP
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_binary_lightgbm')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LGBMClassifier binary demo"
          >
            Load Binary Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_multi_lightgbm')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LGBMClassifier 3-class demo"
          >
            Load Multi Clf
          </button>
          <button
            onClick={() => void loadSampleModel('sample_reg_lightgbm')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LGBMRegressor demo"
          >
            Load Regressor
          </button>
          <button
            onClick={() => void loadSampleModel('sample_binary_lightgbm_native')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Native .txt booster demo"
          >
            Load Native .txt
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
        Model: {mech ? `${mech.algorithm} · ${total} iters · ${mech.objective} · ${mech.num_leaves} leaves` : '—'}
        {mech?.negated_target_zero_view ? ' · class-0 (negated raw-score) view' : ''}
        {mech?.has_missing_values ? ' · ◆ NaN in row' : ''}
      </p>

      {mech && depthStat && mech.active_leaf_depth != null && (
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          Leaf-wise asymmetry gauge: active leaf depth {mech.active_leaf_depth} (min {depthStat.min_leaf_depth} ·
          max {depthStat.max_leaf_depth} · {depthStat.num_leaves} leaves in tree #0)
        </p>
      )}

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
          Staged linked curve
          {trace && mech && curve ? (
            <svg
              viewBox={`0 0 ${curve.W} ${curve.H}`}
              className="rounded"
              style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)' }}
              role="img"
              aria-label="Staged linked prediction curve"
            >
              {curve.threshY != null && (
                <line x1={0} x2={curve.W} y1={curve.threshY} y2={curve.threshY} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1} />
              )}
              <path d={curve.line} fill="none" stroke="var(--ml-primary)" strokeWidth={2} />
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

      {trace && mech && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Importance ({s.importanceMetric}) — from booster.feature_importance
          </p>
          <ul className="flex flex-col gap-1">
            {metricVals.map((v, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                  {s.featureNames[i] ?? `feature_${i}`}
                </span>
                <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                  {v.toFixed(2)}
                </span>
                <span
                  aria-hidden
                  className="h-2 rounded"
                  style={{
                    width: `${Math.max(4, (v / maxMetric) * 160)}px`,
                    background: 'var(--ml-primary)',
                    opacity: 0.8,
                  }}
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      {trace && mech && topShap.length > 0 && (
        <div
          className="overflow-x-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Top TreeSHAP features (|φ| leaders)
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">Rank</th>
                <th className="px-2 py-1 text-left">Feature</th>
                <th className="px-2 py-1 text-right">Exact SHAP φ</th>
              </tr>
            </thead>
            <tbody>
              {topShap.map((f) => (
                <tr key={f.feature_idx} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                  <td className="px-2 py-1">#{f.rank}</td>
                  <td className="px-2 py-1">{f.feature_name}</td>
                  <td className="px-2 py-1 text-right" style={{ color: f.shap_value >= 0 ? 'var(--ml-primary)' : 'var(--ml-class-1)' }}>
                    {f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                  </td>
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
            Exact TreeSHAP waterfall: bias + Σφ = raw score (margin)
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                Bias (base score)
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

      <LightgbmSurfaceSandbox />

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
