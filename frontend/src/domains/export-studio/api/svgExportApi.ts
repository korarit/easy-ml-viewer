/** Phase 48 — server-side SVG fallback (offline-first: used only on demand). */
import { API_BASE_URL } from '../../../core/api/client';

export interface Svg2DExportParams {
  xLabel: string;
  yLabel: string;
  boundarySegments: number[][];
  dataPoints: Array<{ x: number; y: number; actual: number }>;
  width: number;
  height: number;
  theme: 'precision-dark' | 'paper-clean';
  title: string;
  disagreementPct: number | null;
}

export async function fetchServerSvg(params: Svg2DExportParams): Promise<Blob> {
  const res = await fetch(`${API_BASE_URL}/api/v1/exports/svg-2d`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_label: params.xLabel,
      y_label: params.yLabel,
      boundary_segments: params.boundarySegments,
      data_points: params.dataPoints,
      width: params.width,
      height: params.height,
      theme: params.theme,
      title: params.title,
      disagreement_pct: params.disagreementPct,
    }),
  });
  if (!res.ok) throw new Error(`SVG export failed (${res.status})`);
  return res.blob();
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5000);
}
