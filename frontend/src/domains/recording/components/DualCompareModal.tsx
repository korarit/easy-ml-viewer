import { useEffect, useRef, useState } from 'react';
import {
  fetchCompareMlvs,
  nextDisagreementIndex,
  prevDisagreementIndex,
} from '../api/dualCompareApi';
import { useDualCompareStore } from '../store/useDualCompareStore';

/**
 * Phase 40 — Synchronized Dual `.mlv` Split-Screen Comparison Studio.
 * Uploads two recordings, frame-locks them via `sharedFrameFloat`, and
 * exposes the disagreement track with Alt+D style prev/next navigation.
 */
export function DualCompareModal() {
  const open = useDualCompareStore((s) => s.isCompareOpen);
  const setOpen = useDualCompareStore((s) => s.setCompareOpen);
  const busy = useDualCompareStore((s) => s.isComparing);
  const setBusy = useDualCompareStore((s) => s.setComparing);
  const error = useDualCompareStore((s) => s.compareError);
  const setError = useDualCompareStore((s) => s.setCompareError);
  const result = useDualCompareStore((s) => s.result);
  const setResult = useDualCompareStore((s) => s.setResult);
  const sharedFrame = useDualCompareStore((s) => s.sharedFrameFloat);
  const setSharedFrame = useDualCompareStore((s) => s.setSharedFrame);
  const focused = useDualCompareStore((s) => s.focusedConflict);
  const setFocused = useDualCompareStore((s) => s.setFocusedConflict);
  const camSync = useDualCompareStore((s) => s.isCameraSynced);
  const setCamSync = useDualCompareStore((s) => s.setCameraSynced);
  const vcrSync = useDualCompareStore((s) => s.isVcrSynced);
  const setVcrSync = useDualCompareStore((s) => s.setVcrSynced);

  const fileARef = useRef<HTMLInputElement>(null);
  const fileBRef = useRef<HTMLInputElement>(null);
  const [nameA, setNameA] = useState<string | null>(null);
  const [nameB, setNameB] = useState<string | null>(null);

  // Alt+D jumps to the next disagreement frame while the studio is open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (!e.altKey || e.key.toLowerCase() !== 'd') return;
      e.preventDefault();
      const state = useDualCompareStore.getState();
      const list = state.result?.disagreement_indices ?? [];
      if (list.length === 0) return;
      const cursor = state.focusedConflict ?? Math.floor(state.sharedFrameFloat);
      const next = nextDisagreementIndex(list, cursor);
      if (next !== null) {
        state.setFocusedConflict(next);
        if (state.isVcrSynced) state.setSharedFrame(next);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  if (!open) return null;

  const indices = result?.disagreement_indices ?? [];
  const jumpTo = (frame: number | null) => {
    if (frame === null || !result) return;
    setFocused(frame);
    if (vcrSync) setSharedFrame(frame);
  };
  const jumpNext = () => jumpTo(nextDisagreementIndex(indices, focused ?? Math.floor(sharedFrame)));
  const jumpPrev = () => jumpTo(prevDisagreementIndex(indices, focused ?? Math.floor(sharedFrame)));

  const runCompare = async () => {
    const fa = fileARef.current?.files?.[0];
    const fb = fileBRef.current?.files?.[0];
    if (!fa || !fb) {
      setError('Select two .mlv files first (Model A + Model B).');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const body = await fetchCompareMlvs(fa, fb);
      setResult(body);
      setSharedFrame(0);
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
      aria-label="Dual model compare studio"
      onClick={() => setOpen(false)}
    >
      <div
        className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Dual Model Compare Studio</h2>
          <button
            onClick={() => setOpen(false)}
            className="rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          >
            Close
          </button>
        </div>
        <p className="mb-4 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Synchronized split-screen compare — frame-locked VCR + synced 3D camera (single WebGL context, two scissor
          viewports).
        </p>

        <div className="mb-3 grid grid-cols-2 gap-3">
          <label className="block text-xs" style={{ color: 'var(--text-secondary)' }}>
            Model A (.mlv)
            <input
              ref={fileARef}
              type="file"
              accept=".mlv"
              className="mt-1 block w-full text-[11px]"
              onChange={(e) => setNameA(e.target.files?.[0]?.name ?? null)}
            />
            {nameA && <span className="font-mono-tabular text-[10px]">{nameA}</span>}
          </label>
          <label className="block text-xs" style={{ color: 'var(--text-secondary)' }}>
            Model B (.mlv)
            <input
              ref={fileBRef}
              type="file"
              accept=".mlv"
              className="mt-1 block w-full text-[11px]"
              onChange={(e) => setNameB(e.target.files?.[0]?.name ?? null)}
            />
            {nameB && <span className="font-mono-tabular text-[10px]">{nameB}</span>}
          </label>
        </div>

        <div className="mb-4 flex flex-wrap items-center gap-3 text-xs">
          <button
            onClick={() => void runCompare()}
            disabled={busy}
            className="rounded px-3 py-1.5 font-medium disabled:opacity-50"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {busy ? 'Comparing…' : 'Compare 2 Models'}
          </button>
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={camSync} onChange={(e) => setCamSync(e.target.checked)} />
            Sync 3D Camera Orbit
          </label>
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={vcrSync} onChange={(e) => setVcrSync(e.target.checked)} />
            Sync VCR Scrubber
          </label>
        </div>

        {error && (
          <div className="mb-3 rounded px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }} role="alert">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-3 text-xs">
            {result.truncated && (
              <div className="rounded px-3 py-2" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }} role="status">
                {result.truncation_notice}
              </div>
            )}
            <div className="grid grid-cols-4 gap-2 text-center">
              {[
                { label: 'Aligned frames', value: String(result.total_aligned_frames) },
                { label: 'Conflicts', value: String(result.disagreement_count) },
                { label: 'Agreement', value: `${(result.agreement_rate * 100).toFixed(1)}%` },
                { label: result.is_classification ? 'Mean conf. gap' : 'Mean |Δ|', value: result.mean_divergence.toFixed(4) },
              ].map((c) => (
                <div key={c.label} className="rounded p-2" style={{ background: 'var(--surface-elevated)' }}>
                  <div className="text-base font-semibold">{c.value}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{c.label}</div>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={jumpPrev}
                disabled={indices.length === 0}
                className="rounded px-2 py-1 disabled:opacity-40"
                style={{ border: '1px solid var(--border-subtle)' }}
                title="Previous conflict"
              >
                ◀ Prev
              </button>
              <button
                onClick={jumpNext}
                disabled={indices.length === 0}
                className="rounded px-2 py-1 disabled:opacity-40"
                style={{ border: '1px solid var(--border-subtle)' }}
                title="Jump to next disagreement (Alt+D)"
              >
                Next Conflict ▶
              </button>
              <span style={{ color: 'var(--text-secondary)' }}>
                {focused !== null ? (
                  <>Frame {focused} — {result.file_a_name} vs {result.file_b_name} conflict</>
                ) : (
                  'No conflicts — models agree on every aligned frame.'
                )}
              </span>
            </div>

            <label className="block" style={{ color: 'var(--text-secondary)' }}>
              Shared VCR cursor (frame-locked{!vcrSync ? ' — VCR sync off' : ''}): {Math.floor(sharedFrame)}
              <input
                type="range"
                min={0}
                max={Math.max(0, result.total_aligned_frames - 1)}
                value={Math.min(Math.floor(sharedFrame), Math.max(0, result.total_aligned_frames - 1))}
                onChange={(e) => {
                  const f = Number(e.target.value);
                  if (vcrSync) setSharedFrame(f);
                  else setFocused(f);
                }}
                className="mt-1 block w-full"
              />
            </label>

            <div
              className="font-mono-tabular max-h-32 overflow-y-auto rounded p-2 text-[10px] leading-relaxed"
              style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
            >
              conflict frames: {indices.length > 0 ? indices.slice(0, 200).join(', ') : '—'}
              {indices.length > 200 ? ` … (+${indices.length - 200} more)` : ''}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
