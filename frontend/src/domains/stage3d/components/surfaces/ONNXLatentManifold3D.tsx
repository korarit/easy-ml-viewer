import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { normalizeLatentCloud } from './onnxDeepSurfaceMath';

export interface ONNXLatentManifold3DProps {
  latentCoords3D: Float32Array; // Flat [x0, y0, z0, x1, y1, z1, ...]
  predictedClasses: Uint16Array;
  activeFrameIndex: number;
  layerTelemetry: Array<{ layerName: string; meanActivation: number; sparsity: number }>;
}

const CLASS_PALETTE = ['#34d399', '#f472b6', '#38bdf8', '#fbbf24', '#a78bfa', '#f87171'];

function classColor(cls: number): THREE.Color {
  const hex = CLASS_PALETTE[cls % CLASS_PALETTE.length] ?? '#94a3b8';
  return new THREE.Color(hex);
}

/**
 * Phase 39 — 3D latent-space PCA manifold cloud (scene contents, 1 draw call).
 *
 * Renders the penultimate-layer `(PC1, PC2, PC3)` star-cluster as an
 * `InstancedMesh` of soft spheres plus one glowing marker for the active
 * VCR frame. Mount inside a `<Canvas>` (see `OnnxDeepStageRouter`).
 */
export function ONNXLatentManifold3D(props: ONNXLatentManifold3DProps) {
  const { latentCoords3D, predictedClasses, activeFrameIndex, layerTelemetry } = props;

  const points = useMemo(() => {
    const rows: number[][] = [];
    const n = Math.floor(latentCoords3D.length / 3);
    for (let i = 0; i < n; i += 1) {
      rows.push([latentCoords3D[i * 3] ?? 0, latentCoords3D[i * 3 + 1] ?? 0, latentCoords3D[i * 3 + 2] ?? 0]);
    }
    return normalizeLatentCloud(rows);
  }, [latentCoords3D]);

  const cloudRef = useRef<THREE.InstancedMesh>(null!);
  useEffect(() => {
    const mesh = cloudRef.current;
    if (!mesh || points.length === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3(0.09, 0.09, 0.09);
    points.forEach((p, i) => {
      pos.set(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, classColor(predictedClasses[i] ?? 0));
    });
    mesh.count = points.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [points, predictedClasses]);

  const active = points[Math.min(Math.max(activeFrameIndex, 0), Math.max(points.length - 1, 0))] ?? [0, 0, 0];
  // Mean sparsity across probed layers softens the cloud when most neurons sleep.
  const meanSparsity = useMemo(() => {
    const vals = layerTelemetry.map((t) => t.sparsity).filter(Number.isFinite);
    if (vals.length === 0) return 0;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [layerTelemetry]);
  const cloudOpacity = 0.95 - Math.min(0.45, Math.max(0, meanSparsity) * 0.45);

  return (
    <group>
      <instancedMesh ref={cloudRef} args={[undefined, undefined, Math.max(points.length, 1)]} frustumCulled={false}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshStandardMaterial roughness={0.35} metalness={0.1} transparent opacity={cloudOpacity} />
      </instancedMesh>
      {points.length > 0 && (
        <mesh position={[active[0] ?? 0, active[1] ?? 0, active[2] ?? 0]}>
          <sphereGeometry args={[0.16, 20, 20]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
      )}
      {points.length > 0 && (
        <mesh position={[active[0] ?? 0, active[1] ?? 0, active[2] ?? 0]}>
          <sphereGeometry args={[0.24, 20, 20]} />
          <meshBasicMaterial color="#f472b6" transparent opacity={0.35} />
        </mesh>
      )}
      <gridHelper args={[6, 12, '#334155', '#1e293b']} position={[0, -2.6, 0]} />
    </group>
  );
}
