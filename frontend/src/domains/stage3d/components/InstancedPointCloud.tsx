import { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useStage3DStore } from '../store/useStage3DStore';
import { useDisposableThreeResources } from '../hooks/useDisposableThreeResources';
import { worldToThree } from './stage3dSpace';

// Module-scope scratch — reused for every instance matrix (no per-frame alloc).
const _scratchObj = new THREE.Object3D();
const _scratchColor = new THREE.Color();

function colorForResidual(residual: number | null, zSpan: number): string {
  if (residual === null || !Number.isFinite(residual)) return '#8fa3bd';
  const threshold = Math.max(zSpan * 0.05, 1e-9);
  const abs = Math.abs(residual);
  if (abs <= threshold) return '#34d399';
  if (abs <= threshold * 3) return '#fbbf24';
  return '#f43f5e';
}

/**
 * Phase 08 — background trajectory queue as ONE draw call.
 * All queued points except the active frame render through a single
 * THREE.InstancedMesh; matrices + colors update only when data changes.
 */
export function InstancedPointCloud() {
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);
  const visible = useStage3DStore((s) => s.showBackgroundCloud);
  const zScale = useStage3DStore((s) => s.axes.z);
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const background = useMemo(
    () => points.filter((p) => p.frame_index !== activeFrameIndex),
    [points, activeFrameIndex],
  );

  const geometry = useMemo(() => new THREE.SphereGeometry(0.09, 10, 10), []);
  const material = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.1 }), []);
  useDisposableThreeResources([geometry], [material]);

  const zSpan = zScale ? Math.max(zScale.raw_max - zScale.raw_min, 1e-9) : 1;

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let i = 0; i < background.length; i += 1) {
      const p = background[i];
      const [tx, ty, tz] = worldToThree(p.world_x, p.world_y, p.world_z_pred);
      _scratchObj.position.set(tx, ty, tz);
      _scratchObj.updateMatrix();
      mesh.setMatrixAt(i, _scratchObj.matrix);
      mesh.setColorAt(i, _scratchColor.set(colorForResidual(p.residual_error, zSpan)));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [background, zSpan]);

  if (!visible || background.length === 0) return null;
  return (
    <instancedMesh
      ref={meshRef}
      key={background.length}
      args={[geometry, material, background.length]}
      frustumCulled={false}
    />
  );
}
