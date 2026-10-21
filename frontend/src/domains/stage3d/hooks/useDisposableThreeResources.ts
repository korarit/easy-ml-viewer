import { useEffect } from 'react';
import * as THREE from 'three';

/**
 * Phase 08 — Zero VRAM Leak Policy.
 * Disposes every tracked geometry / material when the dependency list
 * changes or the owning component unmounts (Three.js never frees GPU
 * buffers on React unmount by itself).
 */
export function useDisposableThreeResources(
  geometries: Array<THREE.BufferGeometry | null | undefined>,
  materials: Array<THREE.Material | null | undefined>,
): void {
  useEffect(() => {
    return () => {
      for (const geo of geometries) {
        try {
          geo?.dispose();
        } catch {
          /* already disposed — safe to ignore */
        }
      }
      for (const mat of materials) {
        try {
          mat?.dispose();
        } catch {
          /* already disposed — safe to ignore */
        }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...geometries, ...materials]);
}
