/**
 * Phase 09 — offline colormap lookup tables (no CDN textures/shaders).
 * Canonical multi-stop gradients sampled in sRGB; sampleColormap(t) lerps.
 */

export type ColormapName = 'viridis' | 'plasma' | 'coolwarm' | 'rdylgn' | 'neon';

type RGB = [number, number, number];

function hex(h: string): RGB {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const STOPS: Record<ColormapName, RGB[]> = {
  viridis: ['#440154', '#482878', '#3b528b', '#287d8e', '#21918c', '#5ec962', '#fde725'].map(hex),
  plasma: ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'].map(hex),
  coolwarm: ['#3b4cc0', '#7a9ff0', '#dddddd', '#f0948a', '#b40426'].map(hex),
  rdylgn: ['#d73027', '#f46d43', '#fee08b', '#a6d96a', '#1a9850'].map(hex),
  neon: ['#0b0f14', '#5b21b6', '#ff2bd6', '#22d3ee'].map(hex),
};

export const COLORMAP_LABELS: Record<ColormapName, string> = {
  viridis: 'Viridis',
  plasma: 'Plasma',
  coolwarm: 'CoolWarm (Blue-Red)',
  rdylgn: 'RdYlGn',
  neon: 'Cyberpunk Neon',
};

/** Sample the LUT at t in [0,1] (clamped). Returns linear-ish sRGB triple. */
export function sampleColormap(name: ColormapName, t: number): RGB {
  const stops = STOPS[name];
  const c = Math.min(1, Math.max(0, t));
  const pos = c * (stops.length - 1);
  const i = Math.min(Math.floor(pos), stops.length - 2);
  const f = pos - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
