import { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useStage3DStore } from '../store/useStage3DStore';
import { worldToThree } from './stage3dSpace';

// Module-scope scratch objects — never allocate inside useFrame (GC-stutter guard, §5.2).
const _desiredPos = new THREE.Vector3();
const _desiredTarget = new THREE.Vector3();
const _followOffset = new THREE.Vector3(6, 4.5, 8);

type ControlsRef = React.ElementRef<typeof OrbitControls>;

/**
 * Phase 08 — smooth 1-click camera presets.
 * Every preset lerps `camera.position` + `controls.target` with an
 * exponential damp factor; follow-ball tracks the active trajectory point.
 */
export function CameraRigController() {
  const controlsRef = useRef<ControlsRef>(null);
  const camera = useThree((s) => s.camera);

  useFrame((_, rawDelta) => {
    const controls = controlsRef.current;
    if (!controls) return;
    const { cameraPreset, points, activeFrameIndex } = useStage3DStore.getState();
    const delta = Math.min(rawDelta, 0.05);
    const k = 1 - Math.exp(-4.5 * delta);

    const active = points.length > 0 ? points[Math.min(activeFrameIndex, points.length - 1)] : null;

    switch (cameraPreset) {
      case 'isometric':
        _desiredPos.set(10.5, 8.5, 12.5);
        _desiredTarget.set(0, -0.5, 0);
        break;
      case 'top_xy':
        _desiredPos.set(0, 17, 0.02);
        _desiredTarget.set(0, 0, 0);
        break;
      case 'front_xz':
        _desiredPos.set(0, 0.8, 15.5);
        _desiredTarget.set(0, 0, 0);
        break;
      case 'side_yz':
        _desiredPos.set(15.5, 0.8, 0);
        _desiredTarget.set(0, 0, 0);
        break;
      case 'follow_ball': {
        if (!active) {
          _desiredPos.set(10.5, 8.5, 12.5);
          _desiredTarget.set(0, -0.5, 0);
        } else {
          const [bx, by, bz] = worldToThree(active.world_x, active.world_y, active.world_z_pred);
          _desiredTarget.set(bx, by, bz);
          _desiredPos.set(bx, by, bz).add(_followOffset);
        }
        break;
      }
    }

    camera.position.lerp(_desiredPos, k);
    controls.target.lerp(_desiredTarget, k);
    controls.update();
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.12}
      minDistance={4}
      maxDistance={45}
      maxPolarAngle={Math.PI * 0.495}
      target={[0, -0.5, 0]}
    />
  );
}

export function useResetCameraOnMount() {
  const camera = useThree((s) => s.camera);
  const mounted = useRef(false);
  if (!mounted.current) {
    mounted.current = true;
    camera.position.set(10.5, 8.5, 12.5);
  }
  return null;
}
