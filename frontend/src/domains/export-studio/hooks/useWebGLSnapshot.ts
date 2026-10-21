import { useCallback } from 'react';
import { useExportStore } from '../store/useExportStore';
import { getStage3DCapture } from './stage3dCaptureRegistry';
import type { SnapshotBackground, SnapshotScale } from '../types/export.types';

export interface SnapshotResult {
  blob: Blob;
  filename: string;
}

/**
 * Standalone quick-snapshot entry point (no React hooks) for the command
 * palette and global shortcuts. Reads current scale/background from the
 * export store when not provided.
 */
export async function captureStageSnapshot(options?: {
  scale?: SnapshotScale;
  background?: SnapshotBackground;
}): Promise<SnapshotResult> {
  const store = useExportStore.getState();
  const scale = options?.scale ?? store.snapshotScale;
  const background = options?.background ?? store.snapshotBackground;
  store.setSnapshotting(true);
  store.setSnapshotError(null);
  try {
    const handles = getStage3DCapture();
    if (!handles) {
      throw new Error('3D stage is not ready yet — open the 3D Studio step first');
    }
    const { gl, scene, camera } = handles;

    const prevPixelRatio = gl.getPixelRatio();
    const prevBackground = scene.background;
    const prevClearAlpha = gl.getClearAlpha();
    try {
      if (scale === 2) gl.setPixelRatio(Math.min(2, (window.devicePixelRatio || 1) * 2, 4));
      if (background === 'transparent') {
        scene.background = null;
        gl.setClearColor(0x000000, 0);
      }
      // Synchronous pre-render: NO await may sit between these two lines.
      gl.render(scene, camera);
      const blob = await new Promise<Blob>((resolve, reject) => {
        gl.domElement.toBlob((b) => {
          if (b) resolve(b);
          else reject(new Error('Snapshot capture returned an empty image'));
        }, 'image/png');
      });
      const filename = `easy-ml-3d-snapshot-${Date.now()}${scale === 2 ? '@2x' : ''}.png`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
      return { blob, filename };
    } finally {
      gl.setPixelRatio(prevPixelRatio);
      scene.background = prevBackground;
      gl.setClearColor(0x000000, prevClearAlpha);
      gl.render(scene, camera);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Snapshot failed';
    useExportStore.getState().setSnapshotError(msg);
    throw err instanceof Error ? err : new Error(msg);
  } finally {
    useExportStore.getState().setSnapshotting(false);
  }
}

/**
 * Phase 16 — high-resolution 3D studio snapshot (offline, 100% in-browser).
 *
 * Edge #1 (blank PNG): `gl.render(scene, camera)` is invoked synchronously
 * in the same tick immediately before `gl.domElement.toBlob()`, so the
 * drawing buffer is always fresh despite `preserveDrawingBuffer: false`.
 * `scale: 2` temporarily doubles the renderer pixel ratio for a crisp
 * Retina capture, and `transparent` temporarily lifts the scene background.
 */
export function useWebGLSnapshot() {
  const captureSnapshot = useCallback(
    async (options?: { scale?: SnapshotScale; background?: SnapshotBackground }): Promise<SnapshotResult> =>
      captureStageSnapshot(options),
    [],
  );

  return { captureSnapshot };
}
