import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useRegularizedSliceStore } from '../../../model_engine/store/useRegularizedSliceStore';
import {
  liveRegularizedHeight,
  regularizedHeightColor,
} from './regularizedSurfaceMath';

/**
 * Phase 19 — Regularized tilted-plane mini-preview for the ridge/lasso sandbox.
 *
 * Renders the 60×60 analytical plane (own tiny WebGL canvas so the Model step
 * needs no dataset/stage context) with height-normalized coloring, force
 * vector arrows (X sky / Y violet / resultant gold) and L1-pruned axis
 * dimming. When `liveFeatures` is provided and the backend shipped the
 * resolved `weight_row`, heights recompute on the CPU with zero API recall.
 */

export interface RegularizedPlane3DProps {
  liveFeatures?: number[] | null;
}

function normalizeHeights(zMin: number, zMax: number, z: number): number {
  const span = zMax - zMin;
  if (!Number.isFinite(span) || span <= 1e-9) return 1;
  return ((z - zMin) / span) * 2;
}

export function RegularizedPlane3D(props: RegularizedPlane3DProps) {
  const surface = useRegularizedSliceStore((s) => s.surface);
  const axisX = useRegularizedSliceStore((s) => s.axisX);
  const axisY = useRegularizedSliceStore((s) => s.axisY);

  const { geometry, actualMin, actualMax, liveRecompute, centerHeight, arrowLens } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const empty = {
      geometry: geo,
      actualMin: 0,
      actualMax: 1,
      liveRecompute: false,
      centerHeight: 1,
      arrowLens: { x: 0, y: 0, r: 0 } as { x: number; y: number; r: number },
    };
    if (!surface || surface.x_grid.length < 2 || surface.y_grid.length < 2) return empty;
    const res = surface.resolution;
    if (surface.z_values.length < res * res) return empty;

    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const y0 = ys[0];
    const spanX = xs[xs.length - 1] - x0 || 1;
    const spanY = ys[ys.length - 1] - y0 || 1;

    const overlays = surface.extra_overlays;
    const weightRow = Array.isArray(overlays?.weight_row) ? (overlays.weight_row as number[]) : null;
    const baseIntercept = typeof overlays?.base_intercept === 'number' ? (overlays.base_intercept as number) : null;
    const live = props.liveFeatures;
    const canLive =
      overlays?.gpu_formula_valid === true &&
      weightRow !== null &&
      baseIntercept !== null &&
      live !== null &&
      live !== undefined &&
      live.length > 0 &&
      live.every((v) => Number.isFinite(v));

    const raw = new Float64Array(res * res);
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const z =
          canLive && weightRow !== null && baseIntercept !== null && live
            ? liveRegularizedHeight(weightRow, baseIntercept, live, axisX, axisY, xs[c], ys[r])
            : (surface.z_values[idx] ?? 0);
        const safe = Number.isFinite(z) ? z : 0;
        raw[idx] = safe;
        if (safe < zMin) zMin = safe;
        if (safe > zMax) zMax = safe;
      }
    }
    if (!Number.isFinite(zMin) || !Number.isFinite(zMax)) {
      zMin = 0;
      zMax = 1;
    }

    const pos = new Float32Array(res * res * 3);
    const col = new Float32Array(res * res * 3);
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const h = normalizeHeights(zMin, zMax, raw[idx]);
        pos[idx * 3] = ((xs[c] - x0) / spanX) * 5 - 2.5;
        pos[idx * 3 + 1] = h;
        pos[idx * 3 + 2] = ((ys[r] - y0) / spanY) * 5 - 2.5;
        const t = zMax - zMin <= 1e-9 ? 0.5 : (raw[idx] - zMin) / (zMax - zMin);
        const rgb = regularizedHeightColor(t);
        col[idx * 3] = rgb[0];
        col[idx * 3 + 1] = rgb[1];
        col[idx * 3 + 2] = rgb[2];
      }
    }
    const index: number[] = [];
    for (let r = 0; r < res - 1; r += 1) {
      for (let c = 0; c < res - 1; c += 1) {
        const a = r * res + c;
        const b = a + 1;
        const d = a + res;
        const e = d + 1;
        index.push(a, d, b, b, d, e);
      }
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();

    const wx = typeof overlays?.wx === 'number' ? (overlays.wx as number) : 0;
    const wy = typeof overlays?.wy === 'number' ? (overlays.wy as number) : 0;
    const maxAbs = Math.max(Math.abs(wx), Math.abs(wy), 1e-9);
    const scale = (w: number) => (Math.abs(w) <= 1e-12 ? 0 : 0.35 + 1.15 * (Math.abs(w) / maxAbs));
    const lx = scale(wx);
    const ly = scale(wy);
    const lr = Math.min(1.8, Math.hypot(lx, ly));
    const midIdx = Math.floor((res * res) / 2);
    const centerH = normalizeHeights(zMin, zMax, raw[midIdx] ?? zMin);
    return {
      geometry: geo,
      actualMin: zMin,
      actualMax: zMax,
      liveRecompute: canLive,
      centerHeight: centerH,
      arrowLens: { x: lx, y: ly, r: lr },
    };
  }, [surface, props.liveFeatures, axisX, axisY]);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide,
        roughness: 0.55,
        metalness: 0.1,
        depthWrite: false,
      }),
    [],
  );
  useDisposableThreeResources([geometry], [material]);

  if (!surface) return null;
  const overlays = surface.extra_overlays;
  const isXPruned = overlays?.is_x_pruned === true;
  const isYPruned = overlays?.is_y_pruned === true;
  const flat = overlays?.zero_influence_axes === true;
  const isClassifier = overlays?.is_classifier === true;
  const boundary = typeof overlays?.boundary_threshold === 'number' ? (overlays.boundary_threshold as number) : null;
  const showBoundary = isClassifier && boundary !== null;
  const boundaryH = showBoundary ? normalizeHeights(actualMin, actualMax, boundary as number) : 0;

  const arrow = (dir: [number, number, number], len: number, color: string, key: string) => {
    if (len <= 1e-9) return null;
    const v = new THREE.Vector3(...dir).normalize();
    return (
      <arrowHelper
        key={key}
        args={[v, new THREE.Vector3(0, centerHeight, 0), len, color, 0.22, 0.12]}
      />
    );
  };

  return (
    <div className="flex flex-col gap-1">
      <div
        className="relative w-full overflow-hidden rounded"
        style={{ height: 280, border: '1px solid var(--border-subtle)', background: '#0b0f14' }}
      >
        <Canvas
          dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
          camera={{ position: [4.5, 3.8, 5.5], fov: 42, near: 0.1, far: 100 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
        >
          <ambientLight intensity={0.75} />
          <directionalLight position={[6, 8, 4]} intensity={1.1} />
          <mesh geometry={geometry} material={material} />
          {arrow([1, 0, 0], arrowLens.x, '#38bdf8', 'wx')}
          {arrow([0, 0, 1], arrowLens.y, '#c084fc', 'wy')}
          {arrow([1, 0, 1], arrowLens.r, '#facc15', 'wr')}
          {showBoundary && (
            <gridHelper args={[5.1, 12, '#facc15', '#1e293b']} position={[0, boundaryH, 0]} />
          )}
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          z∈[{actualMin.toFixed(2)}..{actualMax.toFixed(2)}]{liveRecompute ? ' · live phase-shift' : ''} · drag to orbit
        </span>
        {(isXPruned || isYPruned) && (
          <span
            className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: 'rgba(71,85,105,0.85)', color: '#e2e8f0' }}
          >
            L1 PRUNED: Weight = 0.000 (No Effect on Output)
            {isXPruned && isYPruned ? ' [X+Y]' : isXPruned ? ' [X]' : ' [Y]'}
          </span>
        )}
      </div>
      {flat && (
        <p className="text-[11px]" style={{ color: 'var(--ml-anomaly)' }} role="note">
          Flat surface — both plane axes carry zero weight.
        </p>
      )}
    </div>
  );
}
