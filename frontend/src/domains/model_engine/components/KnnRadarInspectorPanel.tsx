import { useEffect, useMemo } from 'react';
import { Radar } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';
import { useKnnSliceStore } from '../store/useKnnSliceStore';
import { KnnRadarSurfaceSandbox } from './KnnRadarSurfaceSandbox';
import { radarShapeLabel, tallyFractions } from '../../stage3d/components/surfaces/knnSurfaceMath';

export const KNN_ALGORITHM_KEYS = ['knn_classifier', 'knn_regressor'] as const;

/**
 * Phase 22 — Sandbox inspector for KNNClassifier / KNNRegressor (plus
 * RadiusNeighbors): neighbor vote tally, radar shell & metric badge,
 * spider-line neighbor table (3D projected vs full D-dim distance) and
 * the 3D radar-sphere sandbox.
 */
export function KnnRadarInspectorPanel() {
  const { currentModel, loadSampleModel, isLoadingModel } = useModelStore();
  const s = useKnnSliceStore();

  const isKnn =
    currentModel !== null &&
    (KNN_ALGORITHM_KEYS as readonly string[]).includes(currentModel.algorithm_key ?? '');

  useEffect(() => {
    if (!isKnn || !currentModel) return;
    if (s.modelId === currentModel.model_id) return;
    const classLabels = currentModel.classes ?? [];
    s.bindModel({
      modelId: currentModel.model_id,
      algorithmKey: currentModel.algorithm_key,
      featureNames: currentModel.features.map((f) => f.name),
      classCount: classLabels.length > 0 ? classLabels.length : 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isKnn, currentModel?.model_id]);

  const liveFeatures = useMemo(() => {
    const vals = s.rowInput
      .split(/[\s,;]+/)
      .map((x) => x.trim())
      .filter(Boolean)
      .map(Number);
    if (vals.length !== s.featureNames.length || vals.some((v) => !Number.isFinite(v))) return null;
    return vals;
  }, [s.rowInput, s.featureNames]);

  if (!isKnn || !currentModel) return null;

  const trace = s.traces[0] ?? null;
  const mech = trace?.extra_mechanics ?? null;
  const isClassifier = s.isClassifier;
  const shape = (mech?.radar_geometry_shape ?? 'sphere') as 'sphere' | 'octahedron' | 'cube';
  const k = mech?.n_neighbors ?? null;
  const kEff = mech?.n_neighbors_effective ?? null;
  const weightsMode = mech?.weights ?? '—';
  const metric = mech?.effective_metric ?? '—';
  const algorithm = mech?.algorithm ?? '—';
  const radarRadius = mech && Number.isFinite(mech.radar_radius) ? (mech.radar_radius as number) : null;
  const isOutlier = mech?.is_outlier === true;
  const tally = tallyFractions(mech?.vote_tally ?? [], kEff ?? 0);
  const winning = tally.length > 0 ? tally[0] : null;

  const neighbors = useMemo(() => {
    if (!mech) return [];
    const dists = mech.neighbor_distances ?? [];
    const origs = mech.neighbor_original_indices ?? [];
    const weights = mech.neighbor_weights ?? [];
    const labels = mech.neighbor_labels ?? mech.neighbor_targets ?? [];
    const cloud = mech.reference_cloud;
    const order = dists.map((_, i) => i).sort((a, b) => (dists[a] ?? 0) - (dists[b] ?? 0));
    return order.map((src, rank) => {
      const coord =
        cloud && origs[src] !== undefined && cloud.index_map[String(origs[src])] !== undefined
          ? (cloud.coordinates[cloud.index_map[String(origs[src])]] ?? [])
          : [];
      let proj: number | null = null;
      if (liveFeatures && coord.length > 0) {
        const qx = s.axisX < liveFeatures.length ? liveFeatures[s.axisX] : 0;
        const qy = s.axisY < liveFeatures.length ? liveFeatures[s.axisY] : 0;
        const nx = s.axisX < coord.length ? coord[s.axisX] : 0;
        const ny = s.axisY < coord.length ? coord[s.axisY] : 0;
        proj = Math.sqrt((qx - nx) ** 2 + (qy - ny) ** 2);
      }
      return {
        rank: rank + 1,
        orig: origs[src],
        label: labels[src],
        fullDist: dists[src] ?? 0,
        projDist: proj,
        weight: weights[src] ?? 0,
      };
    });
  }, [mech, liveFeatures, s.axisX, s.axisY]);

  const shownNeighbors = neighbors.slice(0, 15);
  const hiddenCount = Math.max(0, neighbors.length - shownNeighbors.length);
  const maxWeight = shownNeighbors.length > 0 ? Math.max(...shownNeighbors.map((n) => n.weight), 1e-12) : 1;

  const contribs = trace
    ? Object.entries(trace.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];
  const maxAbs = contribs.length > 0 ? Math.max(...contribs.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  const testSlice = async () => {
    await s.runTrace();
    await useKnnSliceStore.getState().runSurface();
  };

  return (
    <section
      className="flex flex-col gap-3 rounded-lg p-4"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--ml-primary)' }}
      aria-label="KNN Sandbox Inspector"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          <Radar size={15} style={{ color: 'var(--ml-primary)' }} />
          Sandbox Inspector — KNN Radar Sphere ({radarShapeLabel(shape)})
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => void loadSampleModel('sample_iris_knn')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="KNeighborsClassifier n_neighbors=5 demo"
          >
            Load Iris Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_credit_knn_manhattan')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="KNeighborsClassifier manhattan demo (octahedron radar)"
          >
            Load Manhattan Sample
          </button>
          <button
            onClick={() => void loadSampleModel('sample_housing_knn')}
            disabled={isLoadingModel}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="KNeighborsRegressor demo"
          >
            Load Regressor Sample
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
        K = {k ?? '—'} (effective {kEff ?? '—'}) | Weights: {weightsMode} | Metric: {metric} | Index:{' '}
        {algorithm}
        {mech?.is_radius ? ` | Radius: ${mech.radius}` : ''}
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
          Frame radar telemetry
          {trace && mech ? (
            <div className="font-mono-tabular rounded px-2 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
              {isClassifier ? (
                <>
                  <div>
                    Verdict: {isOutlier ? 'OUTLIER (0 neighbors)' : `Class ${String(trace.final_prediction)}`}
                    {winning ? ` (${(winning.weightShare * 100).toFixed(1)}% weight | ${winning.count}/${kEff} votes)` : ''}
                  </div>
                  <div>
                    Radar radius R_t: {radarRadius !== null ? radarRadius.toFixed(3) : '—'} units
                  </div>
                  <div>
                    Native predict_proba: {mech.has_native_proba ? 'yes' : 'N/A'}
                    {mech.probabilities && mech.probabilities.length > 0
                      ? ` | Conf: ${(Math.max(...mech.probabilities) * 100).toFixed(1)}%`
                      : ''}
                  </div>
                </>
              ) : (
                <>
                  <div>
                    Predicted (ŷ):{' '}
                    {typeof trace.final_prediction === 'number'
                      ? trace.final_prediction.toFixed(3)
                      : String(trace.final_prediction)}
                  </div>
                  <div>
                    Weighted neighbor mean:{' '}
                    {typeof mech.weighted_neighbor_mean === 'number'
                      ? mech.weighted_neighbor_mean.toFixed(3)
                      : '—'}
                  </div>
                  <div>Radar radius R_t: {radarRadius !== null ? radarRadius.toFixed(3) : '—'} units</div>
                </>
              )}
            </div>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Run a trace to inspect radar telemetry.</span>
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

      {isOutlier && (
        <p
          className="rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="note"
        >
          OUTLIER: 0 neighbors found within radius R — the ball sits outside every radar shell.
        </p>
      )}

      {isClassifier && tally.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Neighbor vote tally (Radar R_t = {radarRadius !== null ? radarRadius.toFixed(3) : '—'} units)
          </p>
          <ul className="flex flex-col gap-1">
            {tally.map((t) => (
              <li key={t.label} className="flex items-center gap-2">
                <span className="w-24 truncate font-mono-tabular" style={{ color: 'var(--text-primary)' }}>
                  Class {t.label}
                </span>
                <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)', minWidth: 120 }}>
                  {t.count} votes · {(t.weightShare * 100).toFixed(1)}%
                </span>
                <span
                  aria-hidden
                  className="h-2 rounded"
                  style={{
                    width: `${Math.max(4, t.weightShare * 200)}px`,
                    background: 'var(--ml-primary)',
                    opacity: 0.8,
                  }}
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      {shownNeighbors.length > 0 && (
        <div
          className="overflow-auto rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Active neighbors connected by spider lines (top-{shownNeighbors.length}
            {hiddenCount > 0 ? ` of ${neighbors.length}` : ''})
          </p>
          <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                <th className="px-2 py-1 text-left">Rank</th>
                <th className="px-2 py-1 text-left">Ref ID</th>
                <th className="px-2 py-1 text-left">{isClassifier ? 'Class' : 'Target'}</th>
                <th className="px-2 py-1 text-right">3D Dist</th>
                <th className="px-2 py-1 text-right">Full D-Dist</th>
                <th className="px-2 py-1 text-left">Vote Weight</th>
              </tr>
            </thead>
            <tbody>
              {shownNeighbors.map((n) => (
                <tr key={`${n.rank}-${n.orig}`} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                  <td className="px-2 py-1">#{n.rank}</td>
                  <td className="px-2 py-1">Row #{n.orig}</td>
                  <td className="px-2 py-1">{String(n.label)}</td>
                  <td className="px-2 py-1 text-right">{n.projDist !== null ? n.projDist.toFixed(3) : '—'}</td>
                  <td className="px-2 py-1 text-right">{n.fullDist.toFixed(3)}</td>
                  <td className="px-2 py-1">
                    <span className="flex items-center gap-1.5">
                      {(n.weight * 100).toFixed(1)}%
                      <span
                        aria-hidden
                        className="inline-block h-2 rounded"
                        style={{
                          width: `${Math.max(4, (n.weight / maxWeight) * 72)}px`,
                          background: 'var(--ml-primary)',
                          opacity: 0.85,
                        }}
                      />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {hiddenCount > 0 && (
            <p className="font-mono-tabular mt-1" style={{ color: 'var(--text-muted)' }}>
              … +{hiddenCount} more neighbors in the vote tally (spider lines capped at top-15).
            </p>
          )}
        </div>
      )}

      {contribs.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="mb-2 uppercase tracking-widest" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Distance waterfall −|x − centroid| (closer to winning centroid ≈ 0)
          </p>
          <ul className="flex flex-col gap-1">
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
                    background: 'var(--ml-primary)',
                    opacity: 0.8,
                  }}
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      <KnnRadarSurfaceSandbox />

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
