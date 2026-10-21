import type { AutoEventPin, PinClusterNode } from '../types/timeline.types';

/**
 * Phase 11 — spatial pin clustering (edge case #1).
 *
 * Pins must be pre-sorted by `frame_index` ascending (the backend emits
 * them sorted). Pins whose screen x-positions fall within
 * `minPinSeparationPx` collapse into one `PinClusterNode`; zooming in
 * spreads them back into singletons.
 */
export function computeScreenPinClusters(
  pins: AutoEventPin[],
  visibleStartFrame: number,
  visibleEndFrame: number,
  canvasWidthPx: number,
  minPinSeparationPx = 14,
): PinClusterNode[] {
  const spanFrames = Math.max(1, visibleEndFrame - visibleStartFrame);
  const width = Math.max(1, canvasWidthPx);
  const sorted = [...pins].sort((a, b) => a.frame_index - b.frame_index);
  const visiblePins = sorted.filter(
    (p) => p.frame_index >= visibleStartFrame && p.frame_index <= visibleEndFrame,
  );

  const clusters: PinClusterNode[] = [];
  for (const pin of visiblePins) {
    const xPx = ((pin.frame_index - visibleStartFrame) / spanFrames) * width;
    const lastCluster = clusters[clusters.length - 1];

    if (lastCluster && Math.abs(xPx - lastCluster.centerXPx) < minPinSeparationPx) {
      lastCluster.pins.push(pin);
      lastCluster.endFrame = pin.frame_index;
      lastCluster.centerXPx =
        (lastCluster.centerXPx * (lastCluster.pins.length - 1) + xPx) / lastCluster.pins.length;
    } else {
      clusters.push({
        startFrame: pin.frame_index,
        endFrame: pin.frame_index,
        centerXPx: xPx,
        pins: [pin],
      });
    }
  }
  return clusters;
}
