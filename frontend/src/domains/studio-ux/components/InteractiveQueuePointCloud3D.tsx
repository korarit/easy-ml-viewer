import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { ThreeEvent } from '@react-three/fiber';
import { create } from 'zustand';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { worldToThree } from '../../stage3d/components/stage3dSpace';

interface Props {
  positionsBuffer?: Float32Array | null;
  totalFrames?: number | null;
}

const HOVER_THROTTLE_MS = 50;
const DRAG_GUARD_PX = 4;
const RECICK_WINDOW_MS = 800;

const _dummy = new THREE.Object3D();

interface HoverState {
  frame: number | null;
  x: number;
  y: number;
  setHover: (frame: number | null, x?: number, y?: number) => void;
}

export const useInteractiveHoverStore = create<HoverState>()((set) => ({
  frame: null,
  x: 0,
  y: 0,
  setHover: (frame, x = 0, y = 0) => set({ frame, x, y }),
}));

function residualColor(residual: number | null, zSpan: number): string {
  if (residual === null || !Number.isFinite(residual)) return '#8fa3bd';
  const threshold = Math.max(zSpan * 0.05, 1e-9);
  const abs = Math.abs(residual);
  if (abs <= threshold) return '#34d399';
  if (abs <= threshold * 3) return '#fbb24c';
  return '#f43f5e';
}

/**
 * Phase 14 — interactive queue point cloud (single InstancedMesh draw call).
 *
 * - Hover highlights the instance + shows a throttled (50ms) mini-tooltip.
 * - Click scrubs the VCR playhead to the instance frame; OrbitControls drags
 *   (>4px pointer travel) are ignored so camera rotation never scrubs.
 * - Repeated clicks on a stacked position cycle through overlapping frames.
 */
export function InteractiveQueuePointCloud3D({ positionsBuffer, totalFrames }: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const downPos = useRef({ x: 0, y: 0 });
  const lastHoverAt = useRef(0);
  const lastClick = useRef({ instanceId: -1, at: 0, cycle: 0 });
  const storePoints = useStage3DStore((s) => s.points);
  const zScale = useStage3DStore((s) => s.axes.z);

  const derived = useMemo(() => {
    if (positionsBuffer && totalFrames && totalFrames > 0) {
      const arr: Array<{ frame_index: number; tx: number; ty: number; tz: number; residual: number | null }> = [];
      for (let i = 0; i < totalFrames; i += 1) {
        arr.push({
          frame_index: i,
          tx: positionsBuffer[i * 3],
          ty: positionsBuffer[i * 3 + 1],
          tz: positionsBuffer[i * 3 + 2],
          residual: null,
        });
      }
      return arr;
    }
    return storePoints.map((p) => {
      const [tx, ty, tz] = worldToThree(p.world_x, p.world_y, p.world_z_pred);
      return { frame_index: p.frame_index, tx, ty, tz, residual: p.residual_error };
    });
  }, [positionsBuffer, totalFrames, storePoints]);

  const count = derived.length;
  const zSpan = zScale ? Math.max(zScale.raw_max - zScale.raw_min, 1e-9) : 1;
  const scratchColor = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || count === 0) return;
    for (let i = 0; i < count; i += 1) {
      const p = derived[i];
      _dummy.position.set(p.tx, p.ty, p.tz);
      _dummy.updateMatrix();
      mesh.setMatrixAt(i, _dummy.matrix);
      mesh.setColorAt(i, scratchColor.set(residualColor(p.residual, zSpan)));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    // Critical for raycast correctness when points leave the origin cube.
    mesh.computeBoundingSphere();
  }, [derived, count, zSpan, scratchColor]);

  useEffect(() => () => {
    document.body.style.cursor = '';
    useInteractiveHoverStore.getState().setHover(null);
  }, []);

  if (count === 0) return null;

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    downPos.current = { x: e.clientX, y: e.clientY };
  };

  const resolveFrame = (instanceId: number): number => {
    const now = performance.now();
    const prev = lastClick.current;
    if (prev.instanceId === instanceId && now - prev.at < RECICK_WINDOW_MS) {
      const anchor = derived[instanceId];
      if (anchor) {
        const eps = 1e-6;
        const group = derived.filter(
          (p) =>
            Math.abs(p.tx - anchor.tx) < eps &&
            Math.abs(p.ty - anchor.ty) < eps &&
            Math.abs(p.tz - anchor.tz) < eps,
        );
        if (group.length > 1) {
          const next = group[(prev.cycle + 1) % group.length];
          lastClick.current = { instanceId, at: now, cycle: prev.cycle + 1 };
          return next.frame_index;
        }
      }
    }
    lastClick.current = { instanceId, at: now, cycle: 0 };
    const hit = derived[instanceId];
    return hit ? hit.frame_index : instanceId;
  };

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    const dx = e.clientX - downPos.current.x;
    const dy = e.clientY - downPos.current.y;
    if (Math.hypot(dx, dy) > DRAG_GUARD_PX) return;
    if (typeof e.instanceId !== 'number') return;
    e.stopPropagation();
    const frame = resolveFrame(e.instanceId);
    usePlaybackStore.getState().seekToFrame(frame, true);
    useStage3DStore.getState().setActiveFrameIndex(frame);
  };

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    const now = performance.now();
    if (now - lastHoverAt.current < HOVER_THROTTLE_MS) return;
    lastHoverAt.current = now;
    if (typeof e.instanceId === 'number') {
      const hit = derived[e.instanceId];
      document.body.style.cursor = 'pointer';
      useInteractiveHoverStore.getState().setHover(hit ? hit.frame_index : e.instanceId, e.clientX, e.clientY);
      const ring = ringRef.current;
      if (ring && hit) {
        ring.visible = true;
        ring.position.set(hit.tx, hit.ty, hit.tz);
      }
    } else {
      document.body.style.cursor = '';
      useInteractiveHoverStore.getState().setHover(null);
      if (ringRef.current) ringRef.current.visible = false;
    }
  };

  const handlePointerOut = () => {
    document.body.style.cursor = '';
    useInteractiveHoverStore.getState().setHover(null);
    if (ringRef.current) ringRef.current.visible = false;
  };

  return (
    <group>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, count]}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
        frustumCulled={false}
      >
        <sphereGeometry args={[0.09, 12, 12]} />
        <meshStandardMaterial roughness={0.3} metalness={0.2} />
      </instancedMesh>
      {/* Hover highlight ring (billboarded torus facing the camera). */}
      <mesh ref={ringRef} visible={false}>
        <torusGeometry args={[0.16, 0.018, 8, 32]} />
        <meshBasicMaterial color="#e2e8f0" transparent opacity={0.95} depthTest={false} />
      </mesh>
    </group>
  );
}

/** DOM mini-tooltip — mount OUTSIDE the R3F Canvas (fixed position). */
export function InteractiveHoverTooltip() {
  const frame = useInteractiveHoverStore((s) => s.frame);
  const x = useInteractiveHoverStore((s) => s.x);
  const y = useInteractiveHoverStore((s) => s.y);
  const point = useStage3DStore((s) =>
    frame === null ? undefined : s.points.find((p) => p.frame_index === frame),
  );
  if (frame === null) return null;
  const left = Math.min(x + 14, Math.max(window.innerWidth - 250, 8));
  const top = Math.max(y - 58, 8);
  const title = `Frame #${String(frame).padStart(4, '0')}`;
  const subtitle = point
    ? `Row #${point.original_row_index} · Pred ${point.raw_z_pred.toFixed(3)} · Click to Scrub`
    : 'Click to Scrub';
  return (
    <div
      data-testid="interactive-point-tooltip"
      className="pointer-events-none fixed z-[60] max-w-[230px] rounded-lg px-2.5 py-1.5 text-[11px] leading-snug"
      style={{
        left,
        top,
        background: 'rgba(8,12,18,0.92)',
        border: '1px solid rgba(148,163,184,0.35)',
        color: '#e2e8f0',
        boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
      }}
    >
      <div className="font-semibold text-white">{title}</div>
      <div style={{ color: '#94a3b8' }}>{subtitle}</div>
    </div>
  );
}

