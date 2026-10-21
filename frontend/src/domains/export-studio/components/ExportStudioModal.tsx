import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDatasetStore } from '../../dataset/store/useDatasetStore';
import { useModelStore } from '../../model_engine/store/useModelStore';
import { useMappingStore } from '../../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';
import { useTimelineStore } from '../../timeline/store/useTimelineStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import {
  downloadExportBlob,
  exportFilename,
  fetchExportEnrichedTable,
  fetchExportPreview,
} from '../api/exportApi';
import type { ExportPreviewResponse, ExportTableParams, SnapshotBackground, SnapshotScale } from '../types/export.types';
import { useExportStore } from '../store/useExportStore';
import { useWebGLSnapshot } from '../hooks/useWebGLSnapshot';
import { useWebGLVideoRecorder } from '../hooks/useWebGLVideoRecorder';
import { registerRecorderControls } from '../hooks/videoRecorderControl';

function findStageCanvas(): HTMLCanvasElement | null {
  return document.querySelector('[data-stage3d-canvas="true"] canvas');
}

function hudLine(): string {
  const pb = usePlaybackStore.getState();
  const total = pb.totalFrames;
  const frame = pb.currentFrameIndex;
  const fps = pb.baseFps * pb.speedMultiplier;
  return `REC Frame ${String(frame).padStart(4, '0')}/${String(Math.max(total - 1, 0)).padStart(4, '0')} · ${fps.toFixed(1)} fps · ${new Date().toLocaleTimeString()}`;
}

const TABS = [
  { id: 'data', label: '1. Enriched Data (.csv/.xlsx)' },
  { id: 'video', label: '2. 3D Video Clip (.webm)' },
  { id: 'snapshot', label: '3. 3D Snapshot (.png)' },
] as const;

/**
 * Phase 16 — unified Export & Reporting Studio modal (spec §2.1).
 * Mounted persistently inside the 3D studio so the single video-recorder
 * instance (and its Ctrl+Shift+R control registration) survives tab switches.
 */
export function ExportStudioModal() {
  const open = useExportStore((s) => s.isModalOpen);
  const setOpen = useExportStore((s) => s.setModal);
  const activeTab = useExportStore((s) => s.activeTab);
  const setTab = useExportStore((s) => s.setTab);
  const format = useExportStore((s) => s.format);
  const setFormat = useExportStore((s) => s.setFormat);
  const scope = useExportStore((s) => s.scope);
  const setScope = useExportStore((s) => s.setScope);
  const rangeStart = useExportStore((s) => s.rangeStart);
  const rangeEnd = useExportStore((s) => s.rangeEnd);
  const setRange = useExportStore((s) => s.setRange);
  const enrichment = useExportStore((s) => s.enrichment);
  const setEnrichment = useExportStore((s) => s.setEnrichment);
  const isExporting = useExportStore((s) => s.isExporting);
  const setExporting = useExportStore((s) => s.setExporting);
  const exportError = useExportStore((s) => s.exportError);
  const setExportError = useExportStore((s) => s.setExportError);
  const setLastExport = useExportStore((s) => s.setLastExport);
  const lastExportFilename = useExportStore((s) => s.lastExportFilename);

  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId } = useMappingStore();
  const queueIndices = useTrajectoryQueueStore((s) => s.selectedRowIndices);
  const timelineSession = useTimelineStore((s) => s.data?.session_id ?? null);

  const [preview, setPreview] = useState<ExportPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [fps, setFps] = useState<30 | 60>(60);
  const [hudOverlay, setHudOverlay] = useState(true);
  const [autoStopAtB, setAutoStopAtB] = useState(true);
  const [videoError, setVideoError] = useState<string | null>(null);
  const snapshotScale = useExportStore((s) => s.snapshotScale);
  const snapshotBackground = useExportStore((s) => s.snapshotBackground);
  const setSnapshotOptions = useExportStore((s) => s.setSnapshotOptions);
  const snapshotError = useExportStore((s) => s.snapshotError);
  const isSnapshotting = useExportStore((s) => s.isSnapshotting);

  const { isRecording, startRecording, stopRecording } = useWebGLVideoRecorder();
  const { captureSnapshot } = useWebGLSnapshot();

  const loopRange = usePlaybackStore((s) => s.loopRange);

  // Register recorder controls for the indicator bar + global shortcuts.
  const quickRecord = useCallback(() => {
    const canvas = findStageCanvas();
    if (!canvas) {
      useExportStore.getState().setRecordingError('3D canvas not found — open the 3D Studio step first');
      return;
    }
    setVideoError(null);
    try {
      startRecording(canvas, {
        fps,
        bitrateMbps: fps === 60 ? 16 : 8,
        hudOverlay,
        hudLineProvider: hudOverlay ? hudLine : undefined,
      });
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : 'Recording failed to start');
    }
  }, [fps, hudOverlay, startRecording]);

  useEffect(() => {
    registerRecorderControls({ startQuickRecord: quickRecord, stopRecord: stopRecording });
    return () => registerRecorderControls(null);
  }, [quickRecord, stopRecording]);

  const buildParams = useCallback((): ExportTableParams | null => {
    if (!currentDataset || !currentModel || !mappingId) return null;
    const end = rangeEnd >= 0 ? rangeEnd : Math.max(queueIndices.length - 1, 0);
    return {
      datasetId: currentDataset.dataset_id,
      modelId: currentModel.model_id,
      mappingId,
      sessionId: timelineSession,
      selectedRowIndices: queueIndices,
      format,
      scope,
      rangeStart,
      rangeEnd: end,
      includeAttributionColumns: enrichment.includeAttributionColumns,
      includeBookmarks: enrichment.includeBookmarks,
    };
  }, [currentDataset, currentModel, mappingId, timelineSession, queueIndices, format, scope, rangeStart, rangeEnd, enrichment]);

  const queuedCount = queueIndices.length;
  const exportTargetCount = useMemo(() => {
    if (scope === 'sub_range') {
      const end = rangeEnd >= 0 ? rangeEnd : Math.max(queuedCount - 1, 0);
      return Math.max(0, Math.min(end, queuedCount - 1) - Math.min(rangeStart, queuedCount) + 1);
    }
    return queuedCount;
  }, [scope, rangeStart, rangeEnd, queuedCount]);

  // Live preview (first 20 rows) whenever the data tab config changes.
  useEffect(() => {
    if (!open || activeTab !== 'data') return;
    const params = buildParams();
    if (!params || queueIndices.length === 0) {
      setPreview(null);
      return;
    }
    let cancelled = false;
    setPreviewLoading(true);
    void fetchExportPreview(params)
      .then((p) => {
        if (!cancelled) setPreview(p);
      })
      .catch(() => {
        if (!cancelled) setPreview(null);
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, activeTab, format, scope, rangeStart, rangeEnd, queueIndices.length, enrichment.includeAttributionColumns, enrichment.includeBookmarks]);

  // A-B auto-stop watcher: stop when the playhead reaches loop B.
  useEffect(() => {
    if (!isRecording || !autoStopAtB || !loopRange) return;
    const id = window.setInterval(() => {
      const pb = usePlaybackStore.getState();
      if (pb.currentFrameIndex >= loopRange[1]) {
        pb.pause();
        stopRecording();
      }
    }, 200);
    return () => window.clearInterval(id);
  }, [isRecording, autoStopAtB, loopRange, stopRecording]);

  if (!open) return null;

  const ready = !!(currentDataset && currentModel && mappingId && queueIndices.length > 0);

  const onExport = async () => {
    const params = buildParams();
    if (!params || queueIndices.length === 0) {
      setExportError('Select dataset → model → mapping → queue rows before exporting.');
      return;
    }
    setExportError(null);
    setExporting(true, 15);
    const tick = window.setInterval(() => {
      const cur = useExportStore.getState().exportProgress;
      if (cur < 85) useExportStore.getState().setExporting(true, cur + 12);
    }, 250);
    try {
      const blob = await fetchExportEnrichedTable(params);
      useExportStore.getState().setExporting(true, 100);
      const filename = exportFilename(params, currentDataset?.dataset_id ?? 'export', exportTargetCount);
      downloadExportBlob(blob, filename);
      setLastExport(filename);
      setOpen(false);
    } catch (err) {
      setExportError(`Export failed: ${err instanceof Error ? err.message : 'unknown error'}`);
    } finally {
      window.clearInterval(tick);
      useExportStore.getState().setExporting(false, 0);
    }
  };

  const onRecordAB = () => {
    if (!loopRange) {
      setVideoError('Set an A-B loop range on the VCR timeline first (or use Manual Start/Stop).');
      return;
    }
    const canvas = findStageCanvas();
    if (!canvas) {
      setVideoError('3D canvas not found — open the 3D Studio step first');
      return;
    }
    setVideoError(null);
    usePlaybackStore.getState().seekToFrame(loopRange[0], true);
    try {
      startRecording(canvas, {
        fps,
        bitrateMbps: fps === 60 ? 16 : 8,
        hudOverlay,
        hudLineProvider: hudOverlay ? hudLine : undefined,
      });
      usePlaybackStore.getState().play();
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : 'Recording failed to start');
    }
  };

  const onSnapshot = (scale: SnapshotScale, background: SnapshotBackground) => {
    setSnapshotOptions(scale, background);
    void captureSnapshot({ scale, background }).catch(() => {
      /* error is surfaced in the export store */
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(4,8,14,0.6)' }}
      role="dialog"
      aria-modal="true"
      aria-label="Export and reporting studio"
      onClick={() => !isRecording && setOpen(false)}
    >
      <div
        className="w-full max-w-3xl rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">📤 Export &amp; Reporting Studio</h2>
          <button
            onClick={() => !isRecording && setOpen(false)}
            className="rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
            aria-label="Close export studio"
          >
            ✕
          </button>
        </div>

        <div className="mt-3 flex gap-2" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={activeTab === t.id}
              onClick={() => setTab(t.id)}
              className="rounded-md px-3 py-1.5 text-xs font-medium"
              style={{
                border: `1px solid ${activeTab === t.id ? 'var(--ml-primary)' : 'var(--border-subtle)'}`,
                color: activeTab === t.id ? 'var(--text-primary)' : 'var(--text-secondary)',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {activeTab === 'data' && (
          <div className="mt-4 space-y-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
            <div>
              <p className="mb-1 font-medium" style={{ color: 'var(--text-primary)' }}>Select Output Format</p>
              <div className="flex flex-wrap gap-2">
                {(
                  [
                    ['csv', 'CSV (UTF-8 BOM — Thai-safe in Excel)'],
                    ['xlsx', 'Excel Workbook (.xlsx — styled header)'],
                    ['parquet', 'Apache Parquet (.parquet — fast)'],
                  ] as const
                ).map(([v, label]) => (
                  <button
                    key={v}
                    onClick={() => setFormat(v)}
                    className="rounded-md px-3 py-1.5"
                    style={{
                      border: `1px solid ${format === v ? 'var(--ml-primary)' : 'var(--border-subtle)'}`,
                      color: format === v ? 'var(--text-primary)' : 'var(--text-secondary)',
                    }}
                  >
                    {format === v ? '● ' : '○ '}{label}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <p className="mb-1 font-medium" style={{ color: 'var(--text-primary)' }}>Row Filter Scope</p>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setScope('all_queued')}
                  className="rounded-md px-3 py-1.5"
                  style={{ border: `1px solid ${scope === 'all_queued' ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
                >
                  {scope === 'all_queued' ? '● ' : '○ '}All Queued Frames ({queuedCount} rows)
                </button>
                <button
                  onClick={() => setScope('anomaly_bookmarked')}
                  className="rounded-md px-3 py-1.5"
                  style={{ border: `1px solid ${scope === 'anomaly_bookmarked' ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
                  title="Requires a timeline session with bookmarks / auto-event pins"
                >
                  {scope === 'anomaly_bookmarked' ? '● ' : '○ '}Only Anomaly &amp; Bookmarked Rows
                </button>
                <button
                  onClick={() => setScope('sub_range')}
                  className="rounded-md px-3 py-1.5"
                  style={{ border: `1px solid ${scope === 'sub_range' ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
                >
                  {scope === 'sub_range' ? '● ' : '○ '}Sub-range Loop [A .. B]
                </button>
                {scope === 'sub_range' && (
                  <span className="flex items-center gap-1">
                    A
                    <input
                      type="number"
                      min={0}
                      max={Math.max(queuedCount - 1, 0)}
                      value={rangeStart}
                      onChange={(e) => setRange(Number(e.target.value) || 0, rangeEnd)}
                      className="w-20 rounded px-1 py-1"
                      style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-canvas)' }}
                    />
                    B
                    <input
                      type="number"
                      min={0}
                      max={Math.max(queuedCount - 1, 0)}
                      value={rangeEnd >= 0 ? rangeEnd : Math.max(queuedCount - 1, 0)}
                      onChange={(e) => setRange(rangeStart, Number(e.target.value) || 0)}
                      className="w-20 rounded px-1 py-1"
                      style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-canvas)' }}
                    />
                  </span>
                )}
              </div>
            </div>
            <div>
              <p className="mb-1 font-medium" style={{ color: 'var(--text-primary)' }}>Appended Model Intelligence Columns</p>
              <div className="flex flex-wrap gap-2">
                {(
                  [
                    ['includeAttributionColumns', 'Top Positive/Negative SHAP Drivers'],
                    ['includeBookmarks', 'Bookmark Notes + Anomaly Flags'],
                  ] as const
                ).map(([k, label]) => (
                  <label key={k} className="flex cursor-pointer items-center gap-1.5 rounded-md px-3 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
                    <input
                      type="checkbox"
                      checked={enrichment[k]}
                      onChange={(e) => setEnrichment({ [k]: e.target.checked })}
                    />
                    {label}
                  </label>
                ))}
              </div>
              <p className="mt-1" style={{ color: 'var(--text-muted)' }}>
                Always included: model_prediction, label alias, confidence, residual_error / is_misclassified.
              </p>
            </div>
            <div className="rounded-md p-2" style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}>
              {previewLoading ? (
                <span>Loading 20-row preview…</span>
              ) : preview ? (
                <span>
                  Preview: {preview.rows.length} of {preview.total_frames} rows · {preview.header.length} columns
                  {lastExportFilename ? ` · Last saved: ${lastExportFilename}` : ''}
                </span>
              ) : (
                <span>No preview — select queue rows in Step 4 first.</span>
              )}
            </div>
            {exportError && (
              <div className="rounded px-3 py-2" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43d5e' }} role="alert">
                {exportError}
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setOpen(false)}
                className="rounded-md px-3 py-2"
                style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
              >
                Cancel
              </button>
              <button
                onClick={() => void onExport()}
                disabled={isExporting || !ready}
                title={ready ? 'Export enriched table' : 'Select dataset → model → mapping → queue rows first'}
                className="rounded-md px-4 py-2 font-medium disabled:opacity-50"
                style={{ background: 'var(--ml-primary)', color: '#fff' }}
              >
                {isExporting ? 'Exporting…' : `⬇ Export Enriched Table (${exportTargetCount} rows)`}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'video' && (
          <div className="mt-4 space-y-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>Frame Rate &amp; Quality:</span>
              {([30, 60] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setFps(v)}
                  disabled={isRecording}
                  className="rounded-md px-3 py-1.5 disabled:opacity-50"
                  style={{ border: `1px solid ${fps === v ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
                >
                  {fps === v ? '● ' : '○ '}{v} FPS ({v === 60 ? 'Smooth Studio — 16 Mbps' : 'Standard — 8 Mbps'})
                </button>
              ))}
            </div>
            <label className="flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={hudOverlay} onChange={(e) => setHudOverlay(e.target.checked)} disabled={isRecording} />
              HUD Watermark Overlay — burn frame number &amp; FPS into the video pixels
            </label>
            <label className="flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={autoStopAtB} onChange={(e) => setAutoStopAtB(e.target.checked)} />
              Auto-start &amp; auto-stop at the A-B loop range
              {loopRange ? ` (A=${loopRange[0]}, B=${loopRange[1]})` : ' (no loop set)'}
            </label>
            {videoError && (
              <div className="rounded px-3 py-2" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43d5e' }} role="alert">
                {videoError}
              </div>
            )}
            <div className="flex justify-end gap-2">
              {!isRecording ? (
                <>
                  <button
                    onClick={onRecordAB}
                    className="rounded-md px-4 py-2 font-medium"
                    style={{ background: 'var(--ml-primary)', color: '#fff' }}
                  >
                    🎬 Record A-B Sequence
                  </button>
                  <button
                    onClick={quickRecord}
                    className="rounded-md px-4 py-2"
                    style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
                  >
                    ● Manual Start Recording
                  </button>
                </>
              ) : (
                <button
                  onClick={stopRecording}
                  className="rounded-md px-4 py-2 font-medium"
                  style={{ background: '#dc2626', color: '#fff' }}
                >
                  ⏹ Stop &amp; Save Video
                </button>
              )}
            </div>
          </div>
        )}

        {activeTab === 'snapshot' && (
          <div className="mt-4 space-y-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>Resolution Scale:</span>
              {([1, 2] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setSnapshotOptions(v, snapshotBackground)}
                  className="rounded-md px-3 py-1.5"
                  style={{ border: `1px solid ${snapshotScale === v ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
                >
                  {snapshotScale === v ? '● ' : '○ '}{v === 1 ? '1x Native Viewport' : '2x Crisp Presentation (Retina)'}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>Background:</span>
              {(
                [
                  ['studio', 'Studio Dark Background'],
                  ['transparent', 'Transparent Background (Alpha PNG)'],
                ] as [SnapshotBackground, string][]
              ).map(([v, label]) => (
                <button
                  key={v}
                  onClick={() => setSnapshotOptions(snapshotScale, v)}
                  className="rounded-md px-3 py-1.5"
                  style={{ border: `1px solid ${snapshotBackground === v ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
                >
                  {snapshotBackground === v ? '● ' : '○ '}{label}
                </button>
              ))}
            </div>
            {snapshotError && (
              <div className="rounded px-3 py-2" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43d5e' }} role="alert">
                {snapshotError}
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => onSnapshot(snapshotScale, snapshotBackground)}
                disabled={isSnapshotting}
                className="rounded-md px-4 py-2 font-medium disabled:opacity-50"
                style={{ background: 'var(--ml-primary)', color: '#fff' }}
              >
                {isSnapshotting ? 'Capturing…' : '📸 Capture 3D Snapshot (.png)'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
