/**
 * Phase 16 — live WebGL capture registry.
 *
 * `Stage3DCaptureRegistry` — `Stage3DCanvas` registers its R3F
 * `gl / scene / camera` handles here on `onCreated` (and clears them on
 * unmount). The snapshot hook then performs the spec-mandated synchronous
 * `gl.render(scene, camera)` → `toBlob()` tick so captures are never blank
 * under `preserveDrawingBuffer: false`.
 *
 * Phase 56 — 2D planar capture: `Stage2DCanvas` registers its live SVG root
 * (and optional heatmap canvas) so the export suite can snapshot the exact
 * on-screen 2D frame — hexbin grid, dual-rings, comet trail included — and
 * the 60 FPS WebM recorder can composite it frame-by-frame (lossless
 * frame-by-frame VCR replay with moving trails, spec §5).
 */
import type * as THREE from 'three';

export interface Stage3DCaptureHandles {
  gl: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.Camera;
}

let handles: Stage3DCaptureHandles | null = null;

export function registerStage3DCapture(h: Stage3DCaptureHandles | null): void {
  handles = h;
}

export function getStage3DCapture(): Stage3DCaptureHandles | null {
  return handles;
}

export interface Stage2DCaptureHandles {
  /** Live 2D SVG root (vector layers are serialized verbatim on export). */
  svg: SVGSVGElement;
  /** Optional heatmap <canvas> (raster underlay, drawn 1:1 under the SVG). */
  heatmapCanvas?: HTMLCanvasElement | null;
  /** World→SVG view metadata for pixel-exact WebM compositing. */
  worldHalf?: number;
}

let handles2d: Stage2DCaptureHandles | null = null;

export function registerStage2DCapture(h: Stage2DCaptureHandles | null): void {
  handles2d = h;
}

export function getStage2DCapture(): Stage2DCaptureHandles | null {
  return handles2d;
}

/**
 * Serialize the registered live 2D SVG root to a standalone string.
 * Returns `null` when no 2D stage is mounted (3D-only session).
 */
export function snapshotStage2DVector(): string | null {
  if (!handles2d) return null;
  try {
    const clone = handles2d.svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    return new XMLSerializer().serializeToString(clone);
  } catch {
    return null;
  }
}
