import * as THREE from 'three';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useProjectionStore } from '../store/useProjectionStore';
import { useSurfaceMeshGeometry } from '../hooks/useSurfaceMeshGeometry';

/**
 * Phase 09 — LOD decision surface with offline colormap vertex colors,
 * adjustable opacity and an optional wireframe overlay sharing the same
 * geometry (zero extra vertex buffers).
 */
export function DecisionSurfaceMesh() {
  const surfaceData = useProjectionStore((s) => s.surfaceData);
  const colormap = useProjectionStore((s) => s.colormap);
  const opacity = useProjectionStore((s) => s.opacity);
  const wireframe = useProjectionStore((s) => s.wireframe);
  const showSurface = useStage3DStore((s) => s.showSurface);

  const geometry = useSurfaceMeshGeometry({
    xRange: surfaceData?.xRange ?? [0, 1],
    yRange: surfaceData?.yRange ?? [0, 1],
    zHeightsWorld: surfaceData?.zHeightsWorld ?? null,
    rawValues: surfaceData?.rawPredictions ?? null,
    resolution: surfaceData?.gridResolution ?? 0,
    colormap,
  });

  if (!showSurface || !geometry || !surfaceData) return null;
  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          transparent
          opacity={opacity}
          roughness={0.55}
          metalness={0.1}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {wireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial wireframe color="#0b0f14" transparent opacity={0.35} />
        </mesh>
      )}
    </group>
  );
}
