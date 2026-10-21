import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';
import {
  GROUND_WORLD_Z,
  worldToThree,
} from '../../stage3d/components/stage3dSpace';

function finiteOr(value: number, fallback: number) {
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Phase 10 — glowing active-sample sphere driven at 60FPS.
 *
 * Zero-React-re-render: position is written imperatively in `useFrame`
 * from `getState().exactFrameFloat` with smooth lerp between frame `t`
 * and `t+1`. The floating label subscribes only to the integer
 * `currentFrameIndex`, and the drop-line buffer is preallocated
 * (no `new THREE.Vector3()` per frame).
 */
export function ActiveSampleSphere3D() {
  const points = useStage3DStore((s) => s.points);
  const playbackActive = usePlaybackStore((s) => s.totalFrames > 1 && s.smoothInterpolation);
  void usePlaybackStore((s) => s.trailResetCounter);

  const groupRef = useRef<THREE.Group>(null);
  const dropRef = useRef<THREE.LineSegments>(null);
  const ringRef = useRef<THREE.Mesh>(null);

  const ballGeo = useMemo(() => new THREE.SphereGeometry(0.22, 24, 24), []);
  const ballMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#fb923c',
        emissive: '#c2410c',
        emissiveIntensity: 1.1,
        roughness: 0.3,
      }),
    [],
  );
  const ringGeo = useMemo(() => new THREE.RingGeometry(0.28, 0.42, 40), []);
  const ringMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#fb923c',
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
      }),
    [],
  );
  const dropGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);
  const dropMat = useMemo(
    () =>
      new THREE.LineDashedMaterial({
        color: '#fb923c',
        dashSize: 0.25,
        gapSize: 0.16,
        transparent: true,
        opacity: 0.9,
      }),
    [],
  );
  useDisposableThreeResources(
    [ballGeo, ringGeo, dropGeo],
    [ballMat, ringMat, dropMat],
  );

  useFrame(({ clock }) => {
    const group = groupRef.current;
    if (!group) return;
    const st = usePlaybackStore.getState();
    const pts = useStage3DStore.getState().points;
    const n = pts.length;
    if (n === 0) {
      group.visible = false;
      return;
    }
    group.visible = true;
    const exact = finiteOr(st.exactFrameFloat, 0);
    // Edge #6: clamp both endpoints so 499.8 never reads index 500.
    const t0 = Math.max(0, Math.min(n - 1, Math.floor(exact)));
    const t1 = Math.max(0, Math.min(n - 1, t0 + 1));
    const alpha = st.smoothInterpolation ? Math.max(0, Math.min(1, exact - Math.floor(exact))) : 0;
    const a = pts[t0];
    const b = pts[t1];
    const wx = a.world_x + (b.world_x - a.world_x) * alpha;
    const wy = a.world_y + (b.world_y - a.world_y) * alpha;
    const wz = a.world_z_pred + (b.world_z_pred - a.world_z_pred) * alpha;
    const [tx, ty, tz] = worldToThree(
      finiteOr(wx, 0),
      finiteOr(wy, 0),
      finiteOr(wz, 0),
    );
    const pulse = 1 + 0.07 * Math.sin(clock.elapsedTime * 3);
    group.position.set(tx, ty, tz);
    group.scale.setScalar(pulse);

    // Drop-line: ball -> ground (preallocated 2-point buffer, in-place write).
    const line = dropRef.current;
    if (line) {
      const attr = (line.geometry.getAttribute('position') as THREE.BufferAttribute);
      const arr = attr.array as Float32Array;
      const [, gy, gz] = worldToThree(finiteOr(wx, 0), finiteOr(wy, 0), GROUND_WORLD_Z);
      arr[0] = 0;
      arr[1] = 0;
      arr[2] = 0;
      arr[3] = gy - ty;
      arr[4] = gz - tz;
      arr[5] = 0;
      attr.needsUpdate = true;
      line.computeLineDistances();
    }
    if (ringRef.current) {
      ringRef.current.position.set(tx, GROUND_WORLD_Z, tz);
    }
  });

  // The legacy integer-frame sphere (Phase 08) owns the single-row /
  // non-interpolated cases; this actor only takes over for smooth VCR motion.
  // Phase 12: the floating ball label moved to the single-instance
  // ActiveBallHudOverlay (explainability domain) — no per-actor <Html>.
  if (!playbackActive || points.length === 0) return null;

  return (
    <group>
      <group ref={groupRef}>
        <mesh geometry={ballGeo} material={ballMat} />
        <lineSegments ref={dropRef} geometry={dropGeo} material={dropMat} />
      </group>
      <mesh ref={ringRef} geometry={ringGeo} material={ringMat} rotation={[-Math.PI / 2, 0, 0]} />
    </group>
  );
}
