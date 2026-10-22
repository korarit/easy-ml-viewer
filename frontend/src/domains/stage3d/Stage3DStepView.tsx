import { useEffect, useMemo, useState } from 'react';
import { useDatasetStore } from '../dataset/store/useDatasetStore';
import { useModelStore } from '../model_engine/store/useModelStore';
import { useMappingStore } from '../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../trajectory_queue/store/useTrajectoryQueueStore';
import { useSystemStore } from '../../core/store/useSystemStore';
import { useStage3DStore } from './store/useStage3DStore';
import { Stage2DCanvas } from './components/Stage2DCanvas';
import { LayerVisibilityControls2D, ViewModeSwitcherPill } from './components/ViewModeSwitcherPill';
import { useProjectionStore } from '../projection3d/store/useProjectionStore';
import type { AvailableFeature } from '../projection3d/types/projection.types';
import { ProjectionControlToolbar } from '../projection3d/components/ProjectionControlToolbar';
import { PcaVarianceBadge } from '../projection3d/components/PcaVarianceBadge';
import {
  PlaybackSyncHost,
  PlaybackTransportStrip,
  TimelineInspectorPanel,
} from '../playback/components/PlaybackStudioPanel';
import { PlaybackOptionsBar } from '../playback/components/PlaybackOptionsBar';
import { WhatIfStudioSection } from '../whatif/components/WhatIfStudioSection';
import { WhyPanel } from '../explainability/components/WhyPanel';
import { usePlaybackStore } from '../playback/store/usePlaybackStore';
import { Stage3DCanvas } from './components/Stage3DCanvas';
import { CommandPaletteModal } from '../studio-ux/components/CommandPaletteModal';
import { BakeMlvModal } from '../recording/components/BakeMlvModal';
import { DualCompareModal } from '../recording/components/DualCompareModal';
import { ExportStudioModal } from '../export-studio/components/ExportStudioModal';
import { VideoRecordingIndicatorBar } from '../export-studio/components/VideoRecordingIndicatorBar';
import { useExportKeyboardShortcuts } from '../export-studio/hooks/useExportKeyboardShortcuts';
import { MlvReplayModeBanner } from '../recording/components/MlvReplayModeBanner';
import { AliasSettingsModal } from '../studio-ux/components/AliasSettingsModal';
import { CameraPresetToolbar } from '../studio-ux/components/CameraPresetToolbar';
import { CinemaModeOverlay } from '../studio-ux/components/CinemaModeOverlay';
import { useStudioLayoutStore } from '../studio-ux/store/useStudioLayoutStore';
import { usePlaybackToStageSync } from '../studio-ux/hooks/useStudioKeyboardShortcuts';

type DockTab = 'why' | 'whatif' | 'timeline' | 'scene';

/**
 * Step 5 studio — viewport-fit layout, no page scroll.
 *
 * Vertical budget (top → bottom, all visible without scrolling):
 *   1. compact projection strip (one row, ~40px)
 *   2. canvas + right dock tabs (flex-1, internal scroll only)
 *   3. slim transport strip (scrubber + play + speed/loop, ~104px)
 *
 * Camera presets live on the canvas HUD; full tuning lives in the Scene tab.
 */
export function Stage3DStepView() {
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId, items: mappingItems } = useMappingStore();
  const queueIndices = useTrajectoryQueueStore((s) => s.selectedRowIndices);
  const { setActiveStep } = useSystemStore();
  const [dockTab, setDockTab] = useState<DockTab>('why');

  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const points = useStage3DStore((s) => s.points);
  const playbackFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const projError = useProjectionStore((s) => s.projError);
  const contourError = useProjectionStore((s) => s.contourError);
  const viewMode = useStage3DStore((s) => s.viewMode);
  const projLoading = useProjectionStore((s) => s.isLoading);
  const projMode = useProjectionStore((s) => s.mode);
  const projX = useProjectionStore((s) => s.xFeature);
  const projY = useProjectionStore((s) => s.yFeature);
  const projZ = useProjectionStore((s) => s.zMode);
  const projPinning = useProjectionStore((s) => s.pinningMode);
  const projLod = useProjectionStore((s) => s.lodLevel);
  const projRobust = useProjectionStore((s) => s.useRobustRange);

  const ready = !!(datasetId && modelId && mappingId && mappingItems.length >= 1);
  const cinemaMode = useStudioLayoutStore((s) => s.cinemaMode);
  const setCinema = useStudioLayoutStore((s) => s.setCinema);
  usePlaybackToStageSync(ready);
  useExportKeyboardShortcuts(ready);

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
    // Don't bake while axes still point at a previous dataset's columns
    // (e.g. feat_income/feat_age after switching to SVR housing).
    // ensureDefaults above will re-pick valid axes and re-trigger this effect.
    if (projMode === 'axis_explorer' && featureNames.length > 0) {
      if (!featureNames.includes(projX) || !featureNames.includes(projY)) return;
    }
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

  // Phase 41 — bake the dense 2D contour grid when the planar viewport is
  // active (frontend cache makes 2D↔3D toggles free after the first bake).
  useEffect(() => {
    if (!ready || !datasetId || !modelId || !mappingId || viewMode !== '2d') return;
    if (featureNames.length > 0 && (!featureNames.includes(projX) || !featureNames.includes(projY))) return;
    const timer = window.setTimeout(() => {
      void useProjectionStore.getState().fetchContour({
        datasetId,
        modelId,
        mappingId,
        queueRowIndices: queueIndices,
        currentFrameIndex: frameForPinning >= 0 ? frameForPinning : null,
      });
    }, 150);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, datasetId, modelId, mappingId, queueKey, viewMode, projX, projY, projPinning, projRobust, frameForPinning, featureNames.join('|')]);

  // Phase 43 — live KNN radar: refetch the per-frame neighbor state as the
  // VCR scrubs (debounced; the contour cache already gates non-KNN scenes).
  const surfaceVisualType = useStage3DStore((s) => s.surfaceVisualType);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);
  useEffect(() => {
    if (!ready || !datasetId || !modelId || !mappingId || viewMode !== '2d') return;
    if (surfaceVisualType !== 'voronoi_knn_proximity') return;
    if (featureNames.length > 0 && (!featureNames.includes(projX) || !featureNames.includes(projY))) return;
    const rowIndex = points[activeFrameIndex]?.original_row_index;
    if (rowIndex === undefined || rowIndex < 0) return;
    const timer = window.setTimeout(() => {
      void useProjectionStore.getState().fetchKnnFrame({
        datasetId,
        modelId,
        mappingId,
        xFeature: projX,
        yFeature: projY,
        rowIndex,
      });
    }, 120);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, datasetId, modelId, mappingId, viewMode, surfaceVisualType, projX, projY, activeFrameIndex, points.length, featureNames.join('|')]);

  if (!ready) {
    return (
      <section
        className="m-6 w-full max-w-xl self-center rounded-lg p-6"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      >
        <p className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Step 5 of 5 · 3D studio
        </p>
        <h1 className="mt-1 text-lg font-semibold">Pick rows first to see them in 3D</h1>
        <p className="mt-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
          Work through Steps 1–4 (file → model → matching → pick rows), then come back here —
          your queue becomes a 3D flight you can play, scrub and explain.
        </p>
        <button
          onClick={() => setActiveStep(4)}
          className="mt-4 min-h-[40px] rounded-md px-3 py-2 text-xs font-medium"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          Go pick rows
        </button>
      </section>
    );
  }

  const tabs: Array<{ key: DockTab; label: string; title: string }> = [
    { key: 'why', label: 'Why', title: 'Why the model predicted this (per-frame reasons)' },
    { key: 'whatif', label: 'What-if', title: 'Test changes on the current frame (W)' },
    { key: 'timeline', label: 'Timeline', title: 'Events, pins and notes across frames' },
    { key: 'scene', label: 'Scene', title: 'Surface detail, motion and camera' },
  ];

  return (
    <section className="flex h-full min-h-0 w-full flex-1 flex-col gap-2 overflow-hidden p-2">
      <PlaybackSyncHost />
      <MlvReplayModeBanner />
      {!cinemaMode && (
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <ViewModeSwitcherPill />
          <div className="min-w-0 flex-1">
            <ProjectionControlToolbar variant="compact" availableFeatures={availableFeatures} />
          </div>
        </div>
      )}
      {cinemaMode && (
        <div className="flex shrink-0 items-center justify-center gap-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          <span>Cinema mode — panels hidden.</span>
          <button
            onClick={() => setCinema(false)}
            className="min-h-[32px] rounded px-2 py-1 text-[11px] font-medium"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            Exit (Tab / Esc)
          </button>
        </div>
      )}
      {projError && (
        <div className="shrink-0 rounded px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43f5e' }} role="alert">
          Surface failed: {projError}
        </div>
      )}
      {contourError && viewMode === '2d' && (
        <div className="shrink-0 rounded px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43f5e' }} role="alert">
          2D contour failed: {contourError}
        </div>
      )}
      {projLoading && points.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-sm" style={{ color: 'var(--text-secondary)' }} role="status">
          Drawing your 3D scene…
        </div>
      ) : points.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center" style={{ color: 'var(--text-secondary)' }}>
          <p className="text-sm">No frames yet — pick rows in Step 4 to fill this stage.</p>
          <button
            onClick={() => setActiveStep(4)}
            className="min-h-[40px] rounded-md px-3 py-2 text-xs font-medium"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            Go pick rows
          </button>
        </div>
      ) : (
        <>
          {/* Main row: canvas (flex-1) + dock tabs (fixed 320px, internal scroll). */}
          <div className="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row">
            <div className="relative min-h-[220px] flex-1 lg:min-h-0">
              {viewMode === '2d' ? <Stage2DCanvas /> : <Stage3DCanvas />}
              <VideoRecordingIndicatorBar />
              {!cinemaMode && viewMode === '3d' && <PcaVarianceBadge featureNames={featureNames} />}
            </div>
            {!cinemaMode && (
              <aside
                className="flex h-56 shrink-0 flex-col overflow-hidden rounded-lg lg:h-auto lg:w-[320px]"
                style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
                aria-label="Studio dock"
              >
                <div className="flex shrink-0 gap-1 border-b px-2 pt-1.5" style={{ borderColor: 'var(--border-subtle)' }} role="tablist" aria-label="Dock panels">
                  {tabs.map((t) => (
                    <button
                      key={t.key}
                      role="tab"
                      aria-selected={dockTab === t.key}
                      title={t.title}
                      onClick={() => setDockTab(t.key)}
                      className="min-h-[32px] rounded-t px-2.5 py-1 text-xs font-medium"
                      style={{
                        color: dockTab === t.key ? 'var(--text-primary)' : 'var(--text-muted)',
                        background: dockTab === t.key ? 'var(--surface-elevated)' : 'transparent',
                        borderBottom: dockTab === t.key ? '2px solid var(--ml-primary)' : '2px solid transparent',
                      }}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-2">
                  {dockTab === 'why' && <WhyPanel />}
                  {dockTab === 'whatif' && <WhatIfStudioSection />}
                  {dockTab === 'timeline' && <TimelineInspectorPanel />}
                  {dockTab === 'scene' && (
                    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto" aria-label="Scene tuning">
                      <LayerVisibilityControls2D />
                      <ProjectionControlToolbar variant="full" availableFeatures={availableFeatures} />
                      <div
                        className="rounded-lg px-3 py-2"
                        style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
                      >
                        <p className="mb-1.5 text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                          Motion
                        </p>
                        <PlaybackOptionsBar variant="scene" />
                      </div>
                      <div
                        className="rounded-lg px-3 py-2"
                        style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
                      >
                        <CameraPresetToolbar />
                        <p className="mt-1.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                          ⌘K palette · Tab cinema · Click any 3D point to scrub · 1–5 camera · R reset · G drops
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </aside>
            )}
          </div>
          {/* Bottom strip: fixed height, always visible — never overlaps the canvas. */}
          {!cinemaMode && <PlaybackTransportStrip />}
          <CommandPaletteModal />
          <AliasSettingsModal />
          <BakeMlvModal />
          <DualCompareModal />
          <ExportStudioModal />
          <CinemaModeOverlay />
        </>
      )}
    </section>
  );
}
