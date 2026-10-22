import { useRef, useState } from 'react';
import { useDatasetStore } from '../../dataset/store/useDatasetStore';
import { useModelStore } from '../../model_engine/store/useModelStore';
import { useMappingStore } from '../../mapping/store/useMappingStore';
import { useTrajectoryQueueStore } from '../../trajectory_queue/store/useTrajectoryQueueStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { API_BASE_URL } from '../../../core/api/client';
import { fetchDual2DCompare } from '../../recording/api/dual2DCompareApi';
import { useDual2DCompareStore } from '../store/useDual2DCompareStore';
import { DualCompare2DCanvas } from './DualCompare2DCanvas';

/**
 * Phase 48 — Dual 2D Compare Studio modal (spec §3.1).
 * Compares the current Model A against an uploaded Model B on the shared
 * (X, Y) plane: side-by-side sync view + superimposed overlay with the
 * striped-magenta disagreement area and Alt+D conflict navigation.
 */
export function DualCompare2DModal() {
  const open = useDual2DCompareStore((s) => s.isModalOpen);
  const setOpen = useDual2DCompareStore((s) => s.setModalOpen);
  const busy = useDual2DCompareStore((s) => s.isComparing);
  const setBusy = useDual2DCompareStore((s) => s.setComparing);
  const error = useDual2DCompareStore((s) => s.compareError);
  const setError = useDual2DCompareStore((s) => s.setCompareError);
  const result = useDual2DCompareStore((s) => s.result);
  const setResult = useDual2DCompareStore((s) => s.setResult);
  const layout = useDual2DCompareStore((s) => s.layout);
  const setLayout = useDual2DCompareStore((s) => s.setLayout);
  const showConflictHatch = useDual2DCompareStore((s) => s.showConflictHatch);
  const setShowConflictHatch = useDual2DCompareStore((s) => s.setShowConflictHatch);

  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId } = useMappingStore();
  const queueIndices = useTrajectoryQueueStore((s) => s.selectedRowIndices);
  const xFeature = useProjectionStore((s) => s.xFeature);
  const yFeature = useProjectionStore((s) => s.yFeature);

  const fileRef = useRef<HTMLInputElement>(null);
  const [modelBId, setModelBId] = useState<string | null>(null);
  const [modelBName, setModelBName] = useState<string | null>(null);
  const [uploadingB, setUploadingB] = useState(false);

  if (!open) return null;
  const ready = !!(currentDataset && currentModel && mappingId && queueIndices.length > 0);

  const uploadModelB = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      setError('Select a Model B file first (.pkl/.joblib/.onnx …).');
      return;
    }
    setUploadingB(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file, file.name);
      const res = await fetch(`${API_BASE_URL}/api/v1/models/upload`, { method: 'POST', body: form });
      if (!res.ok) {
        const data = (await res.json().catch(() => null)) as { error?: { message?: string } } | null;
        throw new Error(data?.error?.message ?? `Model B upload failed (${res.status})`);
      }
      const data = (await res.json()) as { model_id: string; display_name?: string; algorithm_key?: string };
      setModelBId(data.model_id);
      setModelBName(data.display_name ?? data.algorithm_key ?? file.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploadingB(false);
    }
  };

  const runCompare = async () => {
    if (!currentDataset || !currentModel || !mappingId) {
      setError('Complete Steps 1–4 first (dataset → model → mapping → queue).');
      return;
    }
    if (!modelBId) {
      setError('Upload Model B first.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const body = await fetchDual2DCompare({
        datasetId: currentDataset.dataset_id,
        modelAId: currentModel.model_id,
        modelBId,
        mappingId,
        xFeature,
        yFeature,
        gridResolution: 120,
        selectedRowIndices: queueIndices,
      });
      setResult(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(2,6,16,0.72)' }}
      role="dialog"
      aria-label="Dual 2D compare studio"
      onClick={() => setOpen(false)}
    >
      <div
        className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Dual 2D Compare — Synchronized Decision Planes</h2>
          <button
            onClick={() => setOpen(false)}
            className="rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          >
            Close
          </button>
        </div>
        <p className="mb-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Model A (current): {currentModel?.display_name ?? '—'} · X {xFeature} · Y {yFeature} · 100% offline
        </p>

        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          <label className="flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
            Model B file
            <input ref={fileRef} type="file" accept=".pkl,.joblib,.onnx,.json,.ubj,.model,.txt,.cbm" className="text-[11px]" />
          </label>
          <button
            onClick={() => void uploadModelB()}
            disabled={uploadingB}
            className="rounded px-3 py-1.5 font-medium disabled:opacity-50"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
          >
            {uploadingB ? 'Uploading…' : 'Upload Model B'}
          </button>
          {modelBName && (
            <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-primary)' }}>
              B: {modelBName}
            </span>
          )}
          <button
            onClick={() => void runCompare()}
            disabled={busy || !ready || !modelBId}
            title={ready ? 'Compare on the shared 2D plane' : 'Complete Steps 1–4 first'}
            className="rounded px-3 py-1.5 font-medium disabled:opacity-50"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {busy ? 'Comparing…' : 'Compare on 2D Plane'}
          </button>
          <div className="flex overflow-hidden rounded" style={{ border: '1px solid var(--border-subtle)' }} role="tablist" aria-label="Dual 2D layout">
            {(['superimposed', 'side-by-side'] as const).map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={layout === m}
                onClick={() => setLayout(m)}
                className="px-2.5 py-1 text-[11px]"
                style={layout === m ? { background: 'var(--ml-primary)', color: '#fff' } : { color: 'var(--text-secondary)' }}
              >
                {m === 'superimposed' ? 'Superimposed Overlay' : 'Side-by-Side Sync'}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={showConflictHatch} onChange={(e) => setShowConflictHatch(e.target.checked)} />
            Highlight Conflict Zones (Striped Magenta)
          </label>
        </div>

        {error && (
          <div className="mb-3 rounded px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }} role="alert">
            {error}
          </div>
        )}

        <div className="min-h-[320px] flex-1">
          {result ? (
            <DualCompare2DCanvas
              modelAName={result.modelAName}
              modelBName={result.modelBName}
              boundariesA={result.contourA.contours}
              boundariesB={result.contourB.contours}
              disagreementAreaPct={result.disagreementPct}
              consensusNotice={result.consensusNotice || undefined}
              conflictFrameIndices={result.conflictFrameIndices}
              layout={layout}
              showConflictHatch={showConflictHatch}
              onNextConflict={(f) => usePlaybackStore.getState().seekToFrame(f, true)}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>
              Upload Model B, then Compare — both decision boundaries render on one (X, Y) plane with the conflict
              area highlighted.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
