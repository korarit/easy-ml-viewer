import { useEffect, useMemo } from 'react';
import { useDatasetStore } from '../dataset/store/useDatasetStore';
import { useModelStore } from '../model_engine/store/useModelStore';
import { useMappingStore } from '../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../trajectory_queue/store/useTrajectoryQueueStore';
import { useSystemStore } from '../../core/store/useSystemStore';
import { useStage3DStore } from './store/useStage3DStore';
import { useProjectionStore } from '../projection3d/store/useProjectionStore';
import type { AvailableFeature } from '../projection3d/types/projection.types';
import { ProjectionControlToolbar } from '../projection3d/components/ProjectionControlToolbar';
import { PcaVarianceBadge } from '../projection3d/components/PcaVarianceBadge';
import { PlaybackStudioPanel } from '../playback/components/PlaybackStudioPanel';
import { usePlaybackStore } from '../playback/store/usePlaybackStore';
import { Stage3DCanvas } from './components/Stage3DCanvas';

/** Step 5 studio: 3D projection stage — axis explorer + PCA manifold (Phases 08–09). */
export function Stage3DStepView() {
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId, items: mappingItems } = useMappingStore();
  const queueIndices = useTrajectoryQueueStore((s) => s.selectedRowIndices);
  const { setActiveStep } = useSystemStore();

  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const points = useStage3DStore((s) => s.points);
  const playbackFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const projError = useProjectionStore((s) => s.projError);
  const projLoading = useProjectionStore((s) => s.isLoading);
  const projMode = useProjectionStore((s) => s.mode);
  const projX = useProjectionStore((s) => s.xFeature);
  const projY = useProjectionStore((s) => s.yFeature);
  const projZ = useProjectionStore((s) => s.zMode);
  const projPinning = useProjectionStore((s) => s.pinningMode);
  const projLod = useProjectionStore((s) => s.lodLevel);
  const projRobust = useProjectionStore((s) => s.useRobustRange);

  const ready = !!(datasetId && modelId && mappingId && mappingItems.length >= 1);

  const featureNames = useMemo(
    () =>
      [...mappingItems]
        .sort((a, b) => a.feature_index - b.feature_index)
        .map((m) => m.dataset_column_name || m.model_feature_name),
    [mappingItems],
  );

  const availableFeatures = useMemo<AvailableFeature[]>(() => {
    const importanceByIndex = new Map<number, number | null>(
      (currentModel?.features ?? []).map((f) => [f.index, f.learned_weight_or_importance]),
    );
    return [...mappingItems]
      .sort((a, b) => a.feature_index - b.feature_index)
      .filter((m) => m.dataset_column_name)
      .map((m) => ({
        name: m.dataset_column_name as string,
        featureIndex: m.feature_index,
        importance: importanceByIndex.get(m.feature_index) ?? null,
      }));
  }, [mappingItems, currentModel]);

  // Auto-pick the top-2 important features as X/Y (spec §3 step 1).
  useEffect(() => {
    if (availableFeatures.length > 0) useProjectionStore.getState().ensureDefaults(availableFeatures);
  }, [availableFeatures]);

  // Bake the LOD surface (debounced 150ms so scrubbing current-row slices
  // downgrades gracefully instead of spamming the backend).
  const queueKey = queueIndices.join(',');
  const frameForPinning = projPinning === 'current_row' ? playbackFrameIndex : -1;
  useEffect(() => {
    if (!ready || !datasetId || !modelId || !mappingId) return;
    const timer = window.setTimeout(() => {
      void useProjectionStore.getState().fetchSurface({
        datasetId,
        modelId,
        mappingId,
        queueRowIndices: queueIndices,
        currentFrameIndex: frameForPinning >= 0 ? frameForPinning : null,
        featureNames,
      });
    }, 150);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, datasetId, modelId, mappingId, queueKey, projMode, projX, projY, projZ, projPinning, projLod, projRobust, frameForPinning, featureNames.join('|')]);

  if (!ready) {
    return (
      <section
        className="m-6 w-full max-w-xl self-center rounded-lg p-6"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      >
        <p className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Step 5 · 3D Stage
        </p>
        <h1 className="mt-1 text-lg font-semibold">3D Stage needs a mapped queue first</h1>
        <p className="mt-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
          Finish Steps 1–4 (dataset → model → column mapping → trajectory queue), then open this studio to
          inspect the queue in 3D.
        </p>
        <button
          onClick={() => setActiveStep(4)}
          className="mt-4 rounded-md px-3 py-2 text-xs font-medium"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          Back to Trajectory Queue
        </button>
      </section>
    );
  }

  return (
    <section className="flex min-h-0 w-full flex-1 flex-col gap-2 p-3">
      <ProjectionControlToolbar availableFeatures={availableFeatures} />
      {projError && (
        <div className="rounded px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43f5e' }} role="alert">
          Surface failed: {projError}
        </div>
      )}
      {projLoading && points.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-sm" style={{ color: 'var(--text-secondary)' }}>
          Baking LOD decision surface…
        </div>
      ) : points.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-sm" style={{ color: 'var(--text-secondary)' }}>
          Queue is empty — select rows in Step 4 to populate the 3D stage.
        </div>
      ) : (
        <>
          <div className="relative flex min-h-0 flex-1" style={{ minHeight: 420 }}>
            <Stage3DCanvas />
            <PcaVarianceBadge featureNames={featureNames} />
          </div>
          <PlaybackStudioPanel />
        </>
      )}
    </section>
  );
}
