import { useEffect } from 'react';
import { useDatasetStore } from '../../dataset/store/useDatasetStore';
import { useModelStore } from '../../model_engine/store/useModelStore';
import { useMappingStore } from '../../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useWhatIfStore } from '../store/useWhatIfStore';
import { useWhatIfKeyboardShortcuts } from '../hooks/useWhatIfKeyboardShortcuts';
import { WhatIfSandboxPanel } from './WhatIfSandboxPanel';

/**
 * Phase 13 — studio wiring for the What-If sandbox.
 *
 * Owns the session <-> queue sync (debounced rebuild so queue edits never
 * spam the backend) and the playback-frame <-> sandbox-frame sync
 * (frame changes auto-reset the sandbox, spec §7 edge #1).
 */
export function WhatIfStudioSection() {
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId } = useMappingStore();
  const queueIndices = useTrajectoryQueueStore((s) => s.selectedRowIndices);
  const axisX = useStage3DStore((s) => s.axisXFeatureIndex);
  const axisY = useStage3DStore((s) => s.axisYFeatureIndex);
  const playbackFrame = usePlaybackStore((s) => s.currentFrameIndex);

  useWhatIfKeyboardShortcuts(true);

  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const queueKey = queueIndices.join(',');

  // Session sync: same context + queue + axes => same session.
  useEffect(() => {
    if (!datasetId || !modelId || !mappingId || queueIndices.length === 0) return;
    const timer = window.setTimeout(() => {
      void useWhatIfStore.getState().loadSession({
        datasetId,
        modelId,
        mappingId,
        selectedRowIndices: queueIndices,
        axisXFeatureIndex: axisX,
        axisYFeatureIndex: axisY,
        buildKey: `${datasetId}|${modelId}|${mappingId}|queue:${queueKey}|axes:${axisX},${axisY}`,
      });
    }, 300);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, modelId, mappingId, queueKey, axisX, axisY]);

  // Frame sync: the VCR frame drives the sandbox base row (auto-reset).
  useEffect(() => {
    void useWhatIfStore.getState().setFrame(playbackFrame);
  }, [playbackFrame]);

  return <WhatIfSandboxPanel />;
}
