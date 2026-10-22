import { useEffect, useState } from 'react';
import { useDatasetStore } from '../../dataset/store/useDatasetStore';
import { useModelStore } from '../../model_engine/store/useModelStore';
import { useMappingStore } from '../../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';
import { useTimelineStore } from '../../timeline/store/useTimelineStore';
import { downloadMlvBlob, fetchBakeMlv, fetchEstimateMlvSize } from '../api/recordingApi';
import { formatEstimate, lodToResolution, type BakeLod } from '../types/recording.types';
import { useRecordingStore } from '../store/useRecordingStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';

/**
 * Phase 15 — Bake `.mlv` recording modal (spec §2.1).
 * Packs the 3D world + trajectory queue into EMLV v1.0 (MsgPack + Zstd L3).
 */
export function BakeMlvModal() {
  const open = useRecordingStore((s) => s.isBakeModalOpen);
  const setOpen = useRecordingStore((s) => s.setBakeModal);
  const isBaking = useRecordingStore((s) => s.isBaking);
  const setBaking = useRecordingStore((s) => s.setBaking);
  const progress = useRecordingStore((s) => s.bakeProgress);
  const bakeError = useRecordingStore((s) => s.bakeError);
  const setBakeError = useRecordingStore((s) => s.setBakeError);
  const setLastBaked = useRecordingStore((s) => s.setLastBaked);

  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId } = useMappingStore();
  const queueIndices = useTrajectoryQueueStore((s) => s.selectedRowIndices);
  const timelineSession = useTimelineStore((s) => s.data?.session_id ?? null);

  const [title, setTitle] = useState('Untitled Recording');
  const [notes, setNotes] = useState('');
  const [lod, setLod] = useState<BakeLod>('standard');
  const [includeXai, setIncludeXai] = useState(true);
  const [includeBookmarks, setIncludeBookmarks] = useState(true);
  const [includeContour2D, setIncludeContour2D] = useState(true);
  const [estimate, setEstimate] = useState<number | null>(null);
  const liveContour = useProjectionStore((s) => s.contourData);

  const resolution = lodToResolution(lod);
  const ready = !!(currentDataset && currentModel && mappingId && queueIndices.length > 0);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setEstimate(null);
    void fetchEstimateMlvSize({
      totalFrames: queueIndices.length,
      surfaceResolution: resolution,
      includeExplainability: includeXai,
    })
      .then((bytes) => {
        if (!cancelled) setEstimate(bytes);
      })
      .catch(() => {
        if (!cancelled) setEstimate(null);
      });
    return () => {
      cancelled = true;
    };
  }, [open, queueIndices.length, resolution, includeXai]);

  if (!open) return null;

  const onBake = async () => {
    if (!currentDataset || !currentModel || !mappingId || queueIndices.length === 0) {
      setBakeError('Select dataset → model → mapping → queue rows before baking.');
      return;
    }
    setBakeError(null);
    setBaking(true, 12);
    const tick = window.setInterval(() => {
      const cur = useRecordingStore.getState().bakeProgress;
      if (cur < 85) useRecordingStore.getState().setBaking(true, cur + 9);
    }, 220);
    try {
      // Phase 48 — slim live-contour snapshot for the .mlv 2D cache block
      // (heatmap + decision isolines only; overlays stay live-computed).
      let contour2D: Record<string, unknown> | null = null;
      if (includeContour2D && liveContour) {
        contour2D = {
          x_feature: liveContour.xFeature,
          y_feature: liveContour.yFeature,
          grid_resolution: liveContour.gridResolution,
          prediction_task: liveContour.predictionTask,
          heatmap_min: liveContour.heatmapMin,
          heatmap_max: liveContour.heatmapMax,
          heatmap_quantized: Array.from(liveContour.heatmapQuantized),
          contours: liveContour.contours.map((c) => ({
            threshold: c.threshold,
            class_index: c.classIndex,
            kind: c.kind,
            line_segments: c.lineSegments.map((s) => [s[0], s[1], s[2], s[3]]),
          })),
        };
      }
      const blob = await fetchBakeMlv({
        sessionId: timelineSession ?? `sess_${Date.now().toString(36)}`,
        datasetId: currentDataset.dataset_id,
        modelId: currentModel.model_id,
        mappingId,
        selectedRowIndices: queueIndices,
        title: title.trim() || 'Untitled Recording',
        notes,
        surfaceResolution: resolution,
        includeExplainability: includeXai,
        includeBookmarks,
        contour2D,
      });
      useRecordingStore.getState().setBaking(true, 100);
      const safe = (title.trim() || 'recording').replace(/[^\w\-]+/g, '_').slice(0, 60);
      downloadMlvBlob(blob, `${safe}.mlv`);
      setLastBaked(`${safe}.mlv`);
      setOpen(false);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Bake failed';
      const code = (err as { code?: string }).code;
      setBakeError(
        code === 'MLV_TOO_MANY_FRAMES'
          ? 'คิวใหญ่เกิน 10,000 เฟรม — กรุณาลดจำนวนแถวที่เลือกก่อน Bake'
          : `Bake failed: ${msg}`,
      );
    } finally {
      window.clearInterval(tick);
      setBaking(false, 0);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.55)' }}
      role="dialog"
      aria-modal="true"
      aria-label="Bake .mlv recording"
      onClick={() => {
        if (!isBaking) setOpen(false);
      }}
    >
      <div
        className="w-full max-w-lg rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">📦 BAKE .MLV 3D RECORDING PACKAGE</h2>
          <button
            onClick={() => setOpen(false)}
            disabled={isBaking}
            className="rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)' }}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <label className="mt-4 block text-xs" style={{ color: 'var(--text-secondary)' }}>
          Recording Title
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            maxLength={160}
            className="mt-1 w-full rounded-md px-3 py-2 text-sm"
            style={{
              background: 'var(--surface-elevated)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
            }}
            placeholder="Credit Scoring XGBoost - Q3 Audit"
          />
        </label>
        <label className="mt-3 block text-xs" style={{ color: 'var(--text-secondary)' }}>
          Analyst Notes
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            maxLength={2000}
            rows={2}
            className="mt-1 w-full rounded-md px-3 py-2 text-sm"
            style={{
              background: 'var(--surface-elevated)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
            }}
            placeholder="ตรวจสอบเคสลูกค้ากลุ่มรายได้แปรผัน…"
          />
        </label>

        <fieldset className="mt-4 rounded-md p-3 text-xs" style={{ border: '1px solid var(--border-subtle)' }}>
          <legend className="px-1" style={{ color: 'var(--text-muted)' }}>
            Included Tracks & Assets
          </legend>
          <label className="flex items-center gap-2 py-1">
            <span style={{ color: 'var(--text-secondary)' }}>Surface Mesh Resolution</span>
            <select
              value={lod}
              onChange={(e) => setLod(e.target.value as BakeLod)}
              className="ml-auto rounded px-2 py-1"
              style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
            >
              <option value="standard">Standard 60×60 LOD</option>
              <option value="high">High 100×100 LOD</option>
            </select>
          </label>
          <label className="flex items-center gap-2 py-1">
            <input type="checkbox" checked={includeXai} onChange={(e) => setIncludeXai(e.target.checked)} />
            <span>Local Feature Contributions Waterfall (Top 10 / frame)</span>
          </label>
          <label className="flex items-center gap-2 py-1">
            <input
              type="checkbox"
              checked={includeBookmarks}
              onChange={(e) => setIncludeBookmarks(e.target.checked)}
            />
            <span>Auto-Anomaly Pins & User Bookmarks</span>
          </label>
          <label className="flex items-center gap-2 py-1" title="Instant 2D on reload — ~45 KB, switch 2D↔3D in 0.05 s without recompute">
            <input
              type="checkbox"
              checked={includeContour2D}
              onChange={(e) => setIncludeContour2D(e.target.checked)}
              disabled={!liveContour}
            />
            <span>
              2D Contour Cache (instant 2D on reload, +~45 KB)
              {!liveContour && ' — bake the 2D contour first (open 2D Planar view)'}
            </span>
          </label>
        </fieldset>

        <p className="mt-3 font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Format: EMLV v1.0 (MsgPack + Zstd L3 + Float32 Columnar) ·{' '}
          {estimate !== null ? formatEstimate(estimate, queueIndices.length) : 'estimating…'}
        </p>

        {isBaking && (
          <div className="mt-3" role="status" aria-label="Baking progress">
            <div
              className="h-2 overflow-hidden rounded"
              style={{ background: 'var(--surface-elevated)' }}
            >
              <div
                className="h-full transition-all"
                style={{ width: `${progress}%`, background: 'var(--ml-primary)' }}
              />
            </div>
            <p className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              Baking Frames & Compressing (Zstd L3)… {progress}%
            </p>
          </div>
        )}
        {bakeError && (
          <p className="mt-3 text-xs" style={{ color: 'var(--ml-class-1)' }} role="alert">
            {bakeError}
          </p>
        )}
        {!ready && (
          <p className="mt-3 text-xs" style={{ color: 'var(--ml-anomaly)' }}>
            Queue is empty — select rows in Step 4 before baking.
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            onClick={() => setOpen(false)}
            disabled={isBaking}
            className="rounded-md px-3 py-2 text-xs"
            style={{ border: '1px solid var(--border-subtle)' }}
          >
            Cancel
          </button>
          <button
            onClick={() => void onBake()}
            disabled={isBaking || !ready}
            className="rounded-md px-4 py-2 text-xs font-semibold disabled:opacity-50"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {isBaking ? 'Baking…' : '⬇ Bake & Download'}
          </button>
        </div>
      </div>
    </div>
  );
}
