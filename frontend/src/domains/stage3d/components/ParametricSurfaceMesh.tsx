import { useMemo } from 'react';
import * as THREE from 'three';
import { useStage3DStore } from '../store/useStage3DStore';
import { useDisposableThreeResources } from '../hooks/useDisposableThreeResources';
import { worldToThree } from './stage3dSpace';

/**
 * Phase 08 — translucent model surface (flat tilted hyperplane for the
 * linear slice; geometry-agnostic for future model families in Phase 09+).
 * Vertices arrive interleaved [wx,wy,wz,...] and are swizzled to Y-up once.
 */
export function ParametricSurfaceMesh() {
  const vertices = useStage3DStore((s) => s.surfaceVerticesFloat32);
  const resolution = useStage3DStore((s) => s.surfaceResolution);
  const visible = useStage3DStore((s) => s.showSurface);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    if (!vertices || vertices.length < resolution * resolution * 3) return geo;
    const count = resolution * resolution;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const wx = vertices[i * 3];
      const wy = vertices[i * 3 + 1];
      const wz = vertices[i * 3 + 2];
      const [tx, ty, tz] = worldToThree(wx, wy, wz);
      pos[i * 3] = tx;
      pos[i * 3 + 1] = ty;
      pos[i * 3 + 2] = tz;
    }
    const index: number[] = [];
    for (let r = 0; r < resolution - 1; r += 1) {
      for (let c = 0; c < resolution - 1; c += 1) {
        const a = r * resolution + c;
        const b = a + 1;
        const d = a + resolution;
        const e = d + 1;
        index.push(a, d, b, b, d, e);
      }
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();
    return geo;
  }, [vertices, resolution]);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#4f8ff7',
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
        roughness: 0.55,
        metalness: 0.15,
        depthWrite: false,
      }),
    [],
  );
  useDisposableThreeResources([geometry], [material]);

  if (!visible || !vertices) return null;
  return <mesh geometry={geometry} material={material} />;
}
