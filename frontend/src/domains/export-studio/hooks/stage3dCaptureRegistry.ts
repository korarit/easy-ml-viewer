/**
 * Phase 16 — live WebGL capture registry.
 *
 * `Stage3DCanvas` registers its R3F `gl / scene / camera` handles here on
 * `onCreated` (and clears them on unmount). The snapshot hook then performs
 * the spec-mandated synchronous `gl.render(scene, camera)` → `toBlob()`
 * tick so captures are never blank under `preserveDrawingBuffer: false`.
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
