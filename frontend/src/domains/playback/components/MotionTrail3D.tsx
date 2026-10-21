import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';
import { worldToThree } from '../../stage3d/components/stage3dSpace';

export const MAX_TRAIL_POINTS = 50;

const HEAD = new THREE.Color('#fdba74');
const TAIL = new THREE.Color('#431407');

/**
 * Phase 10 — 3D glowing motion trail (comet tail over the last K frames).
 *
 * Pre-allocated `Float32Array(MAX_TRAIL_POINTS * 3)` ring buffer, updated
 * in place every frame: zero GC pressure, zero React re-renders. A far
 * frame jump naturally rebuilds the trail from the new neighbourhood
 * (jump => `trailResetCounter` bumps, trail snaps to the new frames).
 */
export function MotionTrail3D() {
  const trailLength = usePlaybackStore((s) => s.trailLength);
  void usePlaybackStore((s) => s.trailResetCounter);

  const prevK = useRef(-1);

  const { lineObj, posAttr, colAttr, geo } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
    const colors = new Float32Array(MAX_TRAIL_POINTS * 3);
    const posAttr = new THREE.BufferAttribute(positions, 3);
    const colAttr = new THREE.BufferAttribute(colors, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', posAttr);
    geo.setAttribute('color', colAttr);
    geo.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const lineObj = new THREE.Line(geo, mat);
    lineObj.frustumCulled = false;
    return { lineObj, posAttr, colAttr, geo, mat };
  }, []);
  useDisposableThreeResources([geo], [lineObj.material as THREE.Material]);

  useFrame(() => {
    const line = lineObj;
    const st = usePlaybackStore.getState();
    const pts = useStage3DStore.getState().points;
    const n = pts.length;
    const K = Math.max(0, Math.min(st.trailLength, MAX_TRAIL_POINTS));
    if (K <= 0 || n === 0) {
      line.visible = false;
      geo.setDrawRange(0, 0);
      return;
    }
    line.visible = true;

    // Alpha fade ramp depends only on K — rebuild colors on K change.
    if (prevK.current !== K) {
      prevK.current = K;
      const colors = colAttr.array as Float32Array;
      const tmp = new THREE.Color();
      for (let i = 0; i < MAX_TRAIL_POINTS; i += 1) {
        const f = K <= 1 ? 1 : i / (K - 1);
        tmp.copy(TAIL).lerp(HEAD, f * f);
        colors[i * 3] = tmp.r;
        colors[i * 3 + 1] = tmp.g;
        colors[i * 3 + 2] = tmp.b;
      }
      colAttr.needsUpdate = true;
    }

    const exact = Number.isFinite(st.exactFrameFloat) ? st.exactFrameFloat : 0;
    const head = Math.max(0, Math.min(n - 1, Math.floor(exact)));
    const count = Math.max(0, Math.min(K, head + 1));
    const positions = posAttr.array as Float32Array;
    for (let i = 0; i < count; i += 1) {
      const frame = head - count + 1 + i;
      const p = pts[frame];
      let wx = p.world_x;
      let wy = p.world_y;
      let wz = p.world_z_pred;
      if (i === count - 1 && st.smoothInterpolation && head + 1 < n) {
        // Interpolate the head segment for butter-smooth 60FPS motion.
        const q = pts[head + 1];
        const alpha = Math.max(0, Math.min(1, exact - Math.floor(exact)));
        wx += (q.world_x - wx) * alpha;
        wy += (q.world_y - wy) * alpha;
        wz += (q.world_z_pred - wz) * alpha;
      }
      const [tx, ty, tz] = worldToThree(
        Number.isFinite(wx) ? wx : 0,
        Number.isFinite(wy) ? wy : 0,
        Number.isFinite(wz) ? wz : 0,
      );
      positions[i * 3] = tx;
      positions[i * 3 + 1] = ty;
      positions[i * 3 + 2] = tz;
    }
    posAttr.needsUpdate = true;
    geo.setDrawRange(0, count);
    geo.computeBoundingSphere();
  });

  if (trailLength <= 0) return null;
  return <primitive object={lineObj} />;
}
