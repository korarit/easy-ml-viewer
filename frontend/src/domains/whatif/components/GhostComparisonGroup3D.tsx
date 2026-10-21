import { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { useWhatIfStore } from '../store/useWhatIfStore';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';
import { worldToThree, formatTickLabel } from '../../stage3d/components/stage3dSpace';

/**
 * Phase 13 — ghost baseline + mutated sphere + dashed delta vector.
 *
 * Reuse contract (spec §5.2): the ghost sphere, mutated sphere, dashed
 * arrow and cone live in the scene exactly once. Slider edits only flip
 * `visible` and rewrite the preallocated 2-point arrow buffer — nothing is
 * mounted/unmounted per 30ms tick, keeping draw calls flat.
 */
export function GhostComparisonGroup3D() {
  const evaluation = useWhatIfStore((s) => s.evaluation);
  const activeFrame = useWhatIfStore((s) => s.activeFrame);
  const isClassification = useWhatIfStore((s) => s.isClassification);

  const active =
    evaluation !== null &&
    evaluation.base_frame_index === activeFrame &&
    (evaluation.world_distance > 1e-9 || evaluation.changed_features.length > 0);

  const ghostPos = useMemo<[number, number, number]>(
    () =>
      evaluation
        ? worldToThree(...evaluation.original_position_world)
        : ([0, 0, 0] as [number, number, number]),
    [evaluation],
  );
  const mutatedPos = useMemo<[number, number, number]>(
    () =>
      evaluation
        ? worldToThree(...evaluation.mutated_position_world)
        : ([0, 0, 0] as [number, number, number]),
    [evaluation],
  );
  const midPos = useMemo<[number, number, number]>(
    () =>
      [
        (ghostPos[0] + mutatedPos[0]) / 2,
        (ghostPos[1] + mutatedPos[1]) / 2 + 0.35,
        (ghostPos[2] + mutatedPos[2]) / 2,
      ] as [number, number, number],
    [ghostPos, mutatedPos],
  );

  const ghostGeo = useMemo(() => new THREE.SphereGeometry(0.22, 24, 24), []);
  const ghostFillMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#94a3b8',
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      }),
    [],
  );
  const ghostWireMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: '#cbd5e1', wireframe: true, transparent: true, opacity: 0.7 }),
    [],
  );
  const mutatedGeo = useMemo(() => new THREE.SphereGeometry(0.24, 24, 24), []);
  const mutatedMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#38bdf8',
        emissive: '#0369a1',
        emissiveIntensity: 1.0,
        roughness: 0.3,
      }),
    [],
  );
  const flippedMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#34d399',
        emissive: '#065f46',
        emissiveIntensity: 1.2,
        roughness: 0.3,
      }),
    [],
  );
  const arrowGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);
  const arrowMat = useMemo(
    () =>
      new THREE.LineDashedMaterial({
        color: '#e2e8f0',
        dashSize: 0.18,
        gapSize: 0.12,
        transparent: true,
        opacity: 0.95,
      }),
    [],
  );
  const coneGeo = useMemo(() => new THREE.ConeGeometry(0.09, 0.26, 12), []);
  const coneMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: '#e2e8f0', transparent: true, opacity: 0.95 }),
    [],
  );
  useDisposableThreeResources(
    [ghostGeo, mutatedGeo, arrowGeo, coneGeo],
    [ghostFillMat, ghostWireMat, mutatedMat, flippedMat, arrowMat, coneMat],
  );

  const arrowRef = useRef<THREE.LineSegments>(null);
  const coneRef = useRef<THREE.Mesh>(null);

  // Rewrite the preallocated arrow buffer + aim the cone head (no re-mounts).
  useLayoutEffect(() => {
    const line = arrowRef.current;
    if (line) {
      const attr = line.geometry.getAttribute('position') as THREE.BufferAttribute;
      const arr = attr.array as Float32Array;
      arr[0] = ghostPos[0];
      arr[1] = ghostPos[1];
      arr[2] = ghostPos[2];
      arr[3] = mutatedPos[0];
      arr[4] = mutatedPos[1];
      arr[5] = mutatedPos[2];
      attr.needsUpdate = true;
      line.computeLineDistances();
    }
    const cone = coneRef.current;
    if (cone) {
      const dir = new THREE.Vector3(
        mutatedPos[0] - ghostPos[0],
        mutatedPos[1] - ghostPos[1],
        mutatedPos[2] - ghostPos[2],
      );
      const len = dir.length();
      if (len > 1e-6) {
        dir.normalize();
        cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        cone.position.set(
          mutatedPos[0] - dir.x * 0.24,
          mutatedPos[1] - dir.y * 0.24,
          mutatedPos[2] - dir.z * 0.24,
        );
      } else {
        cone.position.set(mutatedPos[0], mutatedPos[1], mutatedPos[2]);
        cone.quaternion.identity();
      }
    }
  }, [ghostPos, mutatedPos]);

  if (!active || !evaluation) return null;

  const deltaZ = evaluation.mutated_position_world[2] - evaluation.original_position_world[2];
  const deltaPred = evaluation.mutated_prediction - evaluation.original_prediction;
  const trace = evaluation.internal_model_trace as {
    axis_x_feature_index?: number;
    axis_y_feature_index?: number;
  };
  const nonAxisShift =
    evaluation.changed_features.length > 0 &&
    (trace?.axis_x_feature_index === undefined ||
      !evaluation.changed_features.some(
        (c) => c.feature_index === trace?.axis_x_feature_index || c.feature_index === trace?.axis_y_feature_index,
      ));

  return (
    <group>
      {/* Ghost baseline sphere: translucent fill + wireframe shell. */}
      <mesh geometry={ghostGeo} material={ghostFillMat} position={ghostPos} />
      <mesh geometry={ghostGeo} material={ghostWireMat} position={ghostPos} />
      {/* Mutated active sphere: green when the decision flipped. */}
      <mesh
        geometry={mutatedGeo}
        material={evaluation.decision_flipped ? flippedMat : mutatedMat}
        position={mutatedPos}
      />
      {/* Dashed delta vector + cone head. */}
      <lineSegments ref={arrowRef} geometry={arrowGeo} material={arrowMat} />
      <mesh ref={coneRef} geometry={coneGeo} material={coneMat} />
      <Html position={midPos} center distanceFactor={20} zIndexRange={[15, 0]} occlude={false}>
        <div style={{ pointerEvents: 'none' }}>
          <div
            data-testid="whatif-delta-label"
            className="font-mono-tabular rounded px-1.5 py-0.5 text-[10px] whitespace-nowrap"
            style={{
              background: 'rgba(10,15,22,0.85)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
            }}
          >
            ΔZ {deltaZ >= 0 ? '+' : ''}
            {formatTickLabel(deltaZ)} · dist {formatTickLabel(evaluation.world_distance)}
            {isClassification && (
              <> · ΔP {deltaPred >= 0 ? '+' : ''}{formatTickLabel(deltaPred)}</>
            )}
            {nonAxisShift && <> · ΔZ vertical shift (non-axis feature)</>}
            {evaluation.decision_flipped && <> · 🎉 flipped</>}
          </div>
        </div>
      </Html>
    </group>
  );
}
