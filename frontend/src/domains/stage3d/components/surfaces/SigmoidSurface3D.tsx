import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useLogisticSliceStore } from '../../../model_engine/store/useLogisticSliceStore';
import {
  liveSigmoidHeight,
  probabilityDivergingColor,
} from './sigmoidSurfaceMath';

/**
 * Phase 18 — 3D Sigmoid S-Surface mini-preview for the logistic sandbox.
 *
 * Renders the 60×60 probability surface (own tiny WebGL canvas so the
 * Model step needs no dataset/stage context) with a diverging
 * blue→white→orange colormap, an emerald boundary band at P = τ, and a
 * linear-logodds plane mode. When `liveFeatures` is provided and the
 * backend shipped the resolved-class `weight_row`, heights recompute on
 * the CPU from σ(w·x + b) in <1 ms — off-axis What-If edits shift the
 * wave phase with zero API recall.
 */

export interface SigmoidSurface3DProps {
  threshold?: number;
  zMode?: 'probability' | 'logodds';
  /** Live full feature vector for What-If phase-shift recompute (optional). */
  liveFeatures?: number[] | null;
}

const EMERALD: [number, number, number] = [0x10 / 255, 0xb9 / 255, 0x81 / 255];
const BOUNDARY_BAND = 0.025;

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function logit(p: number): number {
  const t = clamp01(p);
  const lo = Math.max(t, 1e-6);
  const hi = Math.max(1 - t, 1e-6);
  return Math.log(lo / hi);
}

export function SigmoidSurface3D(props: SigmoidSurface3DProps) {
  const surface = useLogisticSliceStore((s) => s.surface);
  const storeThreshold = useLogisticSliceStore((s) => s.threshold);
  const storeZMode = useLogisticSliceStore((s) => s.zMode);
  const axisX = useLogisticSliceStore((s) => s.axisX);
  const axisY = useLogisticSliceStore((s) => s.axisY);

  const threshold = props.threshold ?? storeThreshold;
  const zMode = props.zMode ?? storeZMode;

  const { geometry, actualMin, actualMax, liveRecompute } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const empty = { geometry: geo, actualMin: 0, actualMax: 1, liveRecompute: false };
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
    const weightRow = Array.isArray(overlays?.weight_row) ? (overlays?.weight_row as number[]) : null;
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

    const probs = new Float32Array(res * res);
    let pMin = Infinity;
    let pMax = -Infinity;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const p =
          canLive && weightRow !== null && baseIntercept !== null && live
            ? liveSigmoidHeight(weightRow, baseIntercept, live, axisX, axisY, xs[c], ys[r])
            : clamp01(surface.z_values[idx] ?? 0.5);
        probs[idx] = p;
        if (p < pMin) pMin = p;
        if (p > pMax) pMax = p;
      }
    }

    // Local frame: plane spans [-2.5, +2.5], height 0..2.
    const pos = new Float32Array(res * res * 3);
    const col = new Float32Array(res * res * 3);
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const p = probs[idx];
        pos[idx * 3] = ((xs[c] - x0) / spanX) * 5 - 2.5;
        pos[idx * 3 + 1] = zMode === 'probability' ? p * 2 : THREE.MathUtils.clamp(logit(p) / 8 + 1, 0, 2);
        pos[idx * 3 + 2] = ((ys[r] - y0) / spanY) * 5 - 2.5;
        let rgb = probabilityDivergingColor(p);
        if (zMode === 'probability' && Math.abs(p - threshold) <= BOUNDARY_BAND) rgb = EMERALD;
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
    return {
      geometry: geo,
      actualMin: pMin === Infinity ? 0 : pMin,
      actualMax: pMax === -Infinity ? 1 : pMax,
      liveRecompute: canLive,
    };
  }, [surface, threshold, zMode, props.liveFeatures, axisX, axisY]);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        side: THREE.DoubleSide,
        roughness: 0.5,
        metalness: 0.1,
        depthWrite: false,
      }),
    [],
  );
  useDisposableThreeResources([geometry], [material]);

  if (!surface) return null;
  const zeroAxes = surface.extra_overlays?.zero_influence_axes === true;
  const thresholdHeight = threshold * 2;

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
          {zMode === 'probability' && !zeroAxes && (
            <gridHelper args={[5.1, 12, '#10b981', '#1e293b']} position={[0, thresholdHeight, 0]} />
          )}
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          P∈[{actualMin.toFixed(2)}..{actualMax.toFixed(2)}] τ={threshold.toFixed(2)}
          {liveRecompute ? ' · live phase-shift' : ''} · drag to orbit
        </span>
      </div>
      {zeroAxes && (
        <p className="text-[11px]" style={{ color: 'var(--ml-anomaly)' }} role="note">
          Flat surface — both plane axes carry zero weight.
        </p>
      )}
    </div>
  );
}
