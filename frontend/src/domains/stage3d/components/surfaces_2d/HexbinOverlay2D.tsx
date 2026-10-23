import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath, hexbinFill } from './pointIntelligence2dMath';

export type HexbinMode = 'count' | 'error' | 'class';

export interface HexbinOverlay2DProps {
  cells: HexbinCell2D[];
  maxBinCount: number;
  mode?: HexbinMode;
  classColor?: (cls: number | null) => string;
  visible?: boolean;
  /** Cap rendered cells (densest first) so huge queues hold 60 FPS. */
  cellCap?: number;
}

/**
 * Phase 49 (mechanism D) — Vectorized hexbin grid overlay.
 *
 * Cells batch by quantized fill bucket (≤8 paths per mode) with a shared
 * translucent slate stroke, keeping draw calls flat. Hover reveals
 * count / error / dominant class via <title>.
 */
export function HexbinOverlay2D({
  cells,
  maxBinCount,
  mode = 'count',
  classColor = () => '#94a3b8',
  visible = true,
  cellCap = 600,
}: HexbinOverlay2DProps) {
  if (!visible || cells.length === 0) return null;
  const ranked = [...cells].sort((a, b) => b.count - a.count).slice(0, Math.max(1, cellCap));
  const buckets = new Map<string, { d: string; titles: string[] }>();
  const keyOf = (fill: string) => fill;
  for (const c of ranked) {
    const fill = hexbinFill(mode, { ...c, maxBinCount }, classColor);
    const key = keyOf(fill);
    const slot = buckets.get(key) ?? { d: '', titles: [] };
    slot.d += hexPath(c.centerX, c.centerY, c.radius);
    if (slot.titles.length < 4) {
      slot.titles.push(`n=${c.count} err=${(c.errorRate * 100).toFixed(0)}% cls=${c.dominantClass ?? '–'}`);
    }
    buckets.set(key, slot);
    if (buckets.size >= 24) break;
  }
  return (
    <g className="hexbin-overlay-2d pointer-events-none" aria-label="Hexbin density overlay">
      {[...buckets.entries()].map(([fill, slot], i) => (
        <path
          key={`${mode}-${i}`}
          d={slot.d}
          fill={fill}
          fillOpacity={mode === 'count' ? 0.34 : 0.42}
          stroke="#0f172a"
          strokeWidth={0.02}
          strokeOpacity={0.65}
        >
          <title>{`Hexbin ${mode} · ${slot.titles.join(' · ')}${ranked.length < cells.length ? ` · +${cells.length - ranked.length} more` : ''}`}</title>
        </path>
      ))}
    </g>
  );
}
