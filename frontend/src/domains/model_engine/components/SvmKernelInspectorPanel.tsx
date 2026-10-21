import { useEffect, useMemo } from 'react';
import { Zap } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useSvmSliceStore } from '../store/useSvmSliceStore';
import { SvmKernelSurfaceSandbox } from './SvmKernelSurfaceSandbox';
import { kernelBadge, svmMarginLabel } from '../../stage3d/components/surfaces/svmSurfaceMath';

export const SVM_ALGORITHM_KEYS = ['svm_classifier', 'svm_regressor', 'svm_oneclass'] as const;

/**
 * Phase 23 — Sandbox inspector for SVC/NuSVC/LinearSVC, SVR/NuSVR/
 * LinearSVR and OneClassSVM: kernel & hyperparameter banner, support
 * vector counts, Platt-vs-margin telemetry, top-5 kernel force table
 * and the 3D kernel-saddle sandbox.
 */
export function SvmKernelInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useSvmSliceStore();

  const isSvm =
    currentModel !== null &&
    (SVM_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isSvm || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSvm, currentModel?.model_id]);

  const liveFeatures = useMemo(() => {
    const vals = s.rowInput
      .split(/[\s,;]+/)
      .map((x) => x.trim())
      .filter(Boolean)
      .map(Number);
    if (vals.length !== s.featureNames.length || vals.some((v) => !Number.isFinite(v))) return null;
    return vals;
  }, [s.rowInput, s.featureNames]);

  if (!isSvm || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isClassifier = s.isClassifier;
  const isOneClass = s.isOneClass;
  const kernel = typeof mech?.kernel === 'string' ? mech.kernel : '—';
  const gamma = mech && mech.gamma !== undefined ? mech.gamma : '—';
  const badge = typeof kernel === 'string' ? kernelBadge(kernel, typeof gamma === 'number' ? gamma : String(gamma)) : kernel;
  const cVal = mech && typeof mech.C === 'number' && Number.isFinite(mech.C) ? mech.C : null;
  const nuVal = mech && typeof mech.nu === 'number' && Number.isFinite(mech.nu) ? mech.nu : null;
  const degree = mech && Number.isFinite(mech.degree) ? mech.degree : null;
  const coef0 = mech && Number.isFinite(mech.coef0) ? mech.coef0 : null;
  const nSupport = mech?.n_support_per_class ?? null;
  const totalSV = mech?.total_support_vectors ?? null;
  const native = mech?.has_native_proba ?? null;
  const rawScore = mech && Number.isFinite(mech.raw_score) ? (mech.raw_score as number) : null;
  const status = mech?.margin_status ?? (rawScore !== null ? svmMarginLabel(rawScore, isOneClass) : '—');
  const inside = mech?.inside_margin ?? null;
  const probaRow = mech?.probabilities ?? null;
  const pseudo = mech?.is_pseudo_proba === true;
  const displayProba = probaRow && probaRow.length > 0 ? Math.max(...probaRow) : null;
  const isOvo = mech?.is_ovo === true;
  const plattConflict = mech?.platt_conflict === true;
  const isLinear = mech?.is_linear_model === true;

  const forces = useMemo(() => {
    if (!mech) return [];
    const rows = (mech.kernel_forces_top ?? []) as Array<{
      sv_position: number;
      support_index: number;
      kernel_value: number;
      force: number;
    }>;
    const maxAbs = Math.max(...rows.map((r) => Math.abs(r.force ?? 0)), 1e-12);
    return rows.map((r) => ({ ...r, maxAbs }));
  }, [mech]);

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await useSvmSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="SVM Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Zap size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — SVM Kernel Saddle ({badge})
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_moons_svm_rbf')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SVC rbf on two moons (kernel saddle demo)"
          >
            Load Moons Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_credit_svm_linear')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="LinearSVC demo (margin-corridor plane)"
          >
            Load Linear Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_housing_svr')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="SVR rbf demo (epsilon-tube)"
          >
            Load SVR Sample
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
        Kernel: {kernel}
        {typeof gamma === 'number' ? ` (γ=${gamma.toFixed(4)})` : ` (γ=${String(gamma)})`}
        {cVal !== null ? ` | C=${cVal}` : ''}
        {nuVal !== null ? ` | ν=${nuVal}` : ''}
        {degree !== null && kernel === 'poly' ? ` | degree=${degree}` : ''}
        {coef0 !== null && (kernel === 'poly' || kernel === 'sigmoid') ? ` | coef0=${coef0}` : ''}
        {nSupport ? ` | SVs: ${nSupport.join(' / ')} (total ${totalSV})` : totalSV !== null ? ` | SVs: ${totalSV}` : ''}
      </p>
      {isLinear && (
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Linear model — no stored support vectors; renders as a margin-corridor plane (Phase-21 style).
        </p>
      )}
      {isOvo && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          OvO multiclass: decision scores are pairwise (C·(C−1)/2 columns). The surface shows a
          predict == target indicator for the selected class.
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
          Frame kernel telemetry
          {trace && mech ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              {isOneClass ? (
                <>
                  <div>
                    Envelope: {mech.predicted_label ?? '—'} (f(x) = {rawScore !== null ? `${rawScore >= 0 ? '+' : ''}${rawScore.toFixed(3)}` : '—'})
                  </div>
                  <div>Status: {inside === true ? 'INLIER (f(x) ≥ 0)' : inside === false ? 'OUTLIER (f(x) < 0)' : status}</div>
                </>
              ) : isClassifier ? (
                <>
                  <div>
                    Decision f(x): {rawScore !== null ? `${rawScore >= 0 ? '+' : ''}${rawScore.toFixed(3)}` : '—'}
                    {'  '}(Status: {inside ? '⚠️ ' : ''}{status})
                  </div>
                  <div>
                    Platt proba: {native === true ? 'native' : native === false ? 'pseudo σ/softmax' : '—'}
                    {displayProba !== null ? ` | Conf: ${(displayProba * 100).toFixed(1)}%` : ''}
                  </div>
                  <div>
                    Intercept (b): {typeof trace.base_value === 'number' ? `${trace.base_value >= 0 ? '+' : ''}${trace.base_value.toFixed(3)}` : '—'}
                  </div>
                </>
              ) : (
                <>
                  <div>
                    Predicted (ŷ): {typeof trace.final_prediction === 'number' ? trace.final_prediction.toFixed(3) : String(trace.final_prediction)}
                  </div>
                  <div>Intercept (b): {typeof trace.base_value === 'number' ? `${trace.base_value >= 0 ? '+' : ''}${trace.base_value.toFixed(3)}` : '—'}</div>
                  {typeof mech.margin_width === 'number' ? <div>ε-tube half-width: ±{mech.margin_width}</div> : null}
                </>
              )}
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect kernel telemetry.</span>
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

      {pseudo && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          probability=False (sklearn default) — confidence is a calibrated σ(f(x))/softmax pseudo-score
          for coloring only; the margin f(x) is the source of truth.
        </p>
      )}
      {plattConflict && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          Platt Calibration Offset: predict() and argmax(predict_proba()) disagree near f(x) ≈ 0 —
          predict() is the verdict.
        </p>
      )}

      {forces.length > 0 && (
        <div
          className="overflow-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Top-{forces.length} support-vector forces (dual_s × K(x_s, x_t))
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">SV</th>
                <th className="px-2 py-1 text-right">K(x_s, x_t)</th>
                <th className="px-2 py-1 text-right">Force</th>
                <th className="px-2 py-1 text-left">Pull</th>
              </tr>
            </thead>
            <tbody>
              {forces.map((f, i) => (
                <tr key={`${f.sv_position}-${i}`} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                  <td className="px-2 py-1">SV #{f.support_index} (pos {f.sv_position})</td>
                  <td className="px-2 py-1 text-right">{f.kernel_value.toFixed(3)}</td>
                  <td className="px-2 py-1 text-right">{f.force >= 0 ? '+' : ''}{f.force.toFixed(3)}</td>
                  <td className="px-2 py-1">
                    <span className="flex items-center gap-1.5">
                      <span
                        aria-hidden
                        className="inline-block h-2 rounded"
                        style={{
                          width: `${Math.max(4, (Math.abs(f.force) / f.maxAbs) * 72)}px`,
                          background: f.force >= 0 ? 'var(--ml-primary)' : 'var(--ml-class-1)',
                          opacity: 0.85,
                        }}
                      />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {contribs.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            {isLinear ? 'Feature margin force (wᵢ·xᵢ + b)' : 'Kernel directional sensitivity (−|x − weighted SV|)'}
          </p>
          <ul className="flex flex-col gap-1">
            <li className="flex items-center gap-2">
              <span className="w-36 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                Intercept (b)
              </span>
              <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 90 }}>
                {typeof trace?.base_value === 'number' ? `${trace.base_value >= 0 ? '+' : ''}${trace.base_value.toFixed(3)}` : '—'}
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

      {liveFeatures === null && s.featureNames.length > 0 && (
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Tip: enter {s.featureNames.length} comma-separated numbers to anchor the query ball on the saddle.
        </p>
      )}

      <SvmKernelSurfaceSandbox />

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
