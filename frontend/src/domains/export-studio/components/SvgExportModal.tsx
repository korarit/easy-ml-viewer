import { useEffect, useMemo, useState } from 'react';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { useDual2DCompareStore } from '../../stage3d/store/useDual2DCompareStore';
import { buildDecisionSvg, downloadSvgFile } from './svgBuilder';
import { fetchServerSvg, downloadBlob } from '../api/svgExportApi';

type SvgTheme = 'precision-dark' | 'paper-clean';

/**
 * Phase 48 — 2D Vector Export Studio (spec §3.2).
 * Offline-first: builds the SVG locally from the live contour (zero network),
 * with an optional server-rendered fallback. High-res PNG guidance included
 * (canvas scale select lives in ExportStudioModal; 4K note here).
 */
export function SvgExportModal() {
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState<SvgTheme>('precision-dark');
  const [width, setWidth] = useState(1200);
  const [height, setHeight] = useState(900);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFile, setLastFile] = useState<string | null>(null);

  const contourData = useProjectionStore((s) => s.contourData);
  const points = useStage3DStore((s) => s.points);
  const dualResult = useDual2DCompareStore((s) => s.result);

  const segments = useMemo(() => {
    if (!contourData) return [];
    return contourData.contours
      .filter((c) => c.kind === 'decision')
      .flatMap((c) => c.lineSegments)
      .map((s) => [s[0], s[1], s[2], s[3]]);
  }, [contourData]);

  const svgPoints = useMemo(
    () =>
      points.slice(0, 20000).map((p) => ({
        x: p.world_x,
        y: p.world_y,
        actual: p.residual_error !== null && Math.abs(p.residual_error) > 0.5 ? 1 : 0,
      })),
    [points],
  );

  // Re-open via a tiny custom event so the studio header stays decoupled.
  // (ExportStudioModal owns the main Export button; this modal listens for
  // `open-svg-export` dispatched from the 2D toolbar.)
  useEffect(() => {
    const onOpen = () => {
      setError(null);
      setOpen(true);
    };
    window.addEventListener('open-svg-export', onOpen as EventListener);
    return () => window.removeEventListener('open-svg-export', onOpen as EventListener);
  }, []);

  if (!open) return null;
  const ready = !!contourData && segments.length > 0;
  const filename = `decision-2d-${contourData?.xFeature ?? 'x'}-vs-${contourData?.yFeature ?? 'y'}`;

  const buildLocal = () => {
    if (!contourData) return;
    setError(null);
    try {
      const svg = buildDecisionSvg({
        xLabel: contourData.xFeature,
        yLabel: contourData.yFeature,
        boundarySegments: segments,
        dataPoints: svgPoints,
        width,
        height,
        theme,
        title: `2D Decision Boundary — ${contourData.xFeature} vs ${contourData.yFeature}`,
        disagreementPct: dualResult?.disagreementPct ?? null,
      });
      downloadSvgFile(svg, filename);
      setLastFile(`${filename}.svg`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const buildServer = async () => {
    if (!contourData) return;
    setBusy(true);
    setError(null);
    try {
      const blob = await fetchServerSvg({
        xLabel: contourData.xFeature,
        yLabel: contourData.yFeature,
        boundarySegments: segments,
        dataPoints: svgPoints,
        width,
        height,
        theme,
        title: `2D Decision Boundary — ${contourData.xFeature} vs ${contourData.yFeature}`,
        disagreementPct: dualResult?.disagreementPct ?? null,
      });
      downloadBlob(blob, `${filename}.svg`);
      setLastFile(`${filename}.svg (server)`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(4,8,14,0.6)' }}
      role="dialog"
      aria-modal="true"
      aria-label="2D vector export"
      onClick={() => setOpen(false)}
    >
      <div
        className="w-full max-w-lg rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">2D Vector Export (.svg)</h2>
          <button
            onClick={() => setOpen(false)}
            className="rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
            aria-label="Close 2D vector export"
          >
            ✕
          </button>
        </div>
        <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Publication-quality vector — zoom without pixelation, opens in any browser/Illustrator/Inkscape, 100%
          offline. {segments.length} boundary segments · {svgPoints.length} points.
        </p>

        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {(['precision-dark', 'paper-clean'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTheme(t)}
              className="rounded-md px-3 py-1.5"
              style={{ border: `1px solid ${theme === t ? 'var(--ml-primary)' : 'var(--border-subtle)'}` }}
            >
              {theme === t ? '● ' : '○ '}{t === 'precision-dark' ? 'Precision Dark' : 'Paper Clean Light'}
            </button>
          ))}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <label className="flex items-center gap-1">
            W
            <select
              value={width}
              onChange={(e) => setWidth(Number(e.target.value))}
              className="rounded px-1 py-1"
              style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-canvas)' }}
              aria-label="SVG width"
            >
              {[800, 1200, 1600, 2400].map((v) => (
                <option key={v} value={v}>{v}px</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1">
            H
            <select
              value={height}
              onChange={(e) => setHeight(Number(e.target.value))}
              className="rounded px-1 py-1"
              style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-canvas)' }}
              aria-label="SVG height"
            >
              {[600, 900, 1200, 1600].map((v) => (
                <option key={v} value={v}>{v}px</option>
              ))}
            </select>
          </label>
          <span style={{ color: 'var(--text-muted)' }}>
            High-res PNG: use Export → 3D Snapshot at 2x (Retina). 4K canvas capture auto-falls back to 1080p on
            low-memory browsers.
          </span>
        </div>

        {error && (
          <div className="mt-3 rounded px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: '#f43d5e' }} role="alert">
            {error}
          </div>
        )}
        {lastFile && !error && (
          <div className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }} role="status">
            Saved: {lastFile}
          </div>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            onClick={() => setOpen(false)}
            className="rounded-md px-3 py-2 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          >
            Cancel
          </button>
          <button
            onClick={() => void buildServer()}
            disabled={busy || !ready}
            title={ready ? 'Render on the backend' : 'Bake the 2D contour first'}
            className="rounded-md px-3 py-2 text-xs disabled:opacity-50"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
          >
            {busy ? 'Rendering…' : 'Server SVG'}
          </button>
          <button
            onClick={buildLocal}
            disabled={!ready}
            title={ready ? 'Build locally — zero network requests' : 'Bake the 2D contour first'}
            className="rounded-md px-4 py-2 text-xs font-medium disabled:opacity-50"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            ⬇ Download SVG (Offline)
          </button>
        </div>
      </div>
    </div>
  );
}

/** Dispatch from any toolbar to open the offline SVG exporter. */
export function openSvgExportModal(): void {
  window.dispatchEvent(new CustomEvent('open-svg-export'));
}
