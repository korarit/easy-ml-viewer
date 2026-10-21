import { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useStage3DStore } from '../store/useStage3DStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useDisposableThreeResources } from '../hooks/useDisposableThreeResources';
import { GROUND_WORLD_Z, worldToThree } from './stage3dSpace';

function segmentGeometry(a: [number, number, number], b: [number, number, number]): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([...a, ...b]), 3));
  return geo;
}

/**
 * Phase 08 — glowing active-row sphere, orthogonal drop-lines, ground
 * target ring, axis projection readouts, and the residual connector
 * between predicted (ŷ) and ground-truth (y) heights.
 */
export function ActiveDataPointSphere() {
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);
  // Phase 10: when the smooth VCR actor owns the ball, this legacy sphere
  // yields the ball mesh (drop-lines, ring, label and residual stay live).
  const playbackOwnsBall = usePlaybackStore((s) => s.totalFrames > 1 && s.smoothInterpolation);
  const showDropLines = useStage3DStore((s) => s.showDropLines);
  const showResidualBars = useStage3DStore((s) => s.showResidualBars);
  const zScale = useStage3DStore((s) => s.axes.z);

  const point = points.length > 0 ? points[Math.min(activeFrameIndex, points.length - 1)] : null;
  const pulseRef = useRef<THREE.Mesh>(null);
  const dashedRef = useRef<THREE.LineSegments>(null);

  const ballPos = useMemo(
    () => (point ? worldToThree(point.world_x, point.world_y, point.world_z_pred) : ([0, 0, 0] as const)),
    [point],
  );
  const groundPos = useMemo(
    () => (point ? worldToThree(point.world_x, point.world_y, GROUND_WORLD_Z) : ([0, GROUND_WORLD_Z, 0] as const)),
    [point],
  );
  const actualPos = useMemo(
    () =>
      point && point.world_z_actual !== null
        ? worldToThree(point.world_x, point.world_y, point.world_z_actual)
        : null,
    [point],
  );

  const dropGeo = useMemo(() => segmentGeometry(ballPos as [number, number, number], groundPos as [number, number, number]), [ballPos, groundPos]);
  const projXGeo = useMemo(
    () => segmentGeometry(ballPos as [number, number, number], worldToThree(-5, point?.world_y ?? 0, GROUND_WORLD_Z)),
    [ballPos, point],
  );
  const projYGeo = useMemo(
    () => segmentGeometry(ballPos as [number, number, number], worldToThree(point?.world_x ?? 0, 5, GROUND_WORLD_Z)),
    [ballPos, point],
  );
  const residualGeo = useMemo(
    () =>
      actualPos
        ? segmentGeometry(ballPos as [number, number, number], actualPos)
        : new THREE.BufferGeometry(),
    [ballPos, actualPos],
  );
  const ballGeo = useMemo(() => new THREE.SphereGeometry(0.22, 24, 24), []);
  const ballMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#fb923c', emissive: '#c2410c', emissiveIntensity: 0.9, roughness: 0.3 }),
    [],
  );
  const pinGeo = useMemo(() => new THREE.OctahedronGeometry(0.16), []);
  const pinMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#e2e8f0', emissive: '#64748b', emissiveIntensity: 0.5, roughness: 0.35 }),
    [],
  );
  const ringGeo = useMemo(() => new THREE.RingGeometry(0.28, 0.42, 40), []);
  const ringMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: '#fb923c', transparent: true, opacity: 0.85, side: THREE.DoubleSide }),
    [],
  );
  const dropMat = useMemo(
    () => new THREE.LineDashedMaterial({ color: '#fb923c', dashSize: 0.25, gapSize: 0.16, transparent: true, opacity: 0.9 }),
    [],
  );
  const projMat = useMemo(
    () => new THREE.LineDashedMaterial({ color: '#7d8ea3', dashSize: 0.2, gapSize: 0.14, transparent: true, opacity: 0.6 }),
    [],
  );
  useDisposableThreeResources(
    [dropGeo, projXGeo, projYGeo, residualGeo, ballGeo, pinGeo, ringGeo],
    [ballMat, pinMat, ringMat, dropMat, projMat],
  );

  useLayoutEffect(() => {
    dashedRef.current?.computeLineDistances();
  }, [dropGeo, projXGeo, projYGeo]);

  // Gentle pulse — scalar-only, zero allocation per frame.
  useFrame(({ clock }) => {
    const m = pulseRef.current;
    if (m) m.scale.setScalar(1 + 0.07 * Math.sin(clock.elapsedTime * 3));
  });

  if (!point) return null;
  const zSpan = zScale ? Math.max(zScale.raw_max - zScale.raw_min, 1e-9) : 1;
  const absResidual = point.residual_error !== null ? Math.abs(point.residual_error) : null;
  const residualColor =
    absResidual === null ? '#8fa3bd' : absResidual <= zSpan * 0.05 ? '#34d399' : '#f43f5e';

  return (
    <group>
      {!playbackOwnsBall && (
        <mesh ref={pulseRef} geometry={ballGeo} material={ballMat} position={ballPos} />
      )}
      {showDropLines && !playbackOwnsBall && (
        <>
          <lineSegments ref={dashedRef} geometry={dropGeo} material={dropMat} />
          <lineSegments geometry={projXGeo} material={projMat} />
          <lineSegments geometry={projYGeo} material={projMat} />
          <mesh geometry={ringGeo} material={ringMat} position={groundPos} rotation={[-Math.PI / 2, 0, 0]} />
        </>
      )}
      {showResidualBars && actualPos && (
        <>
          <mesh geometry={pinGeo} material={pinMat} position={actualPos} />
          <lineSegments geometry={residualGeo}>
            <lineBasicMaterial color={residualColor} transparent opacity={0.95} />
          </lineSegments>
        </>
      )}
      {/* Phase 12: the floating ball label moved to the single-instance
          ActiveBallHudOverlay (explainability domain) — no per-actor <Html>. */}
    </group>
  );
}
