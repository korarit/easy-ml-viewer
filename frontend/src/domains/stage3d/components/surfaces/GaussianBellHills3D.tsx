import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useGnbSliceStore } from '../../../model_engine/store/useGnbSliceStore';
import {
  bellHillHeight,
  hillColor,
  normalizeHillHeight,
  posteriorFromHills,
} from './gnbSurfaceMath';

/**
 * Phase 24 — multi-class bivariate Gaussian bell hills + 1σ/2σ
 * footprint rings + vertical density drop-pins.
 *
 * Own tiny WebGL canvas (no dataset/stage context needed): one
 * translucent mesh per visible class (axis-aligned footprints —
 * conditional independence, never diagonally tilted), ellipse rings
 * for the 1σ/2σ footprints, and a vertical pin per class from Z=0 to
 * the hill surface at the live query point with a glowing tip.
 * Posterior mode normalizes the hills per-vertex (H_k / ΣH).
 */

export interface GaussianBellHills3DProps {
  liveFeatures?: number[] | null;
}

interface HillScene {
  meshes: Array<{ geo: THREE.BufferGeometry; mat: THREE.Material; k: number }>;
  rings: Array<{ geo: THREE.BufferGeometry; k: number }>;
  pins: THREE.BufferGeometry;
  pinTips: THREE.BufferGeometry;
  peaks: THREE.BufferGeometry;
  queryPos: THREE.Vector3;
  pinCount: number;
  ringCount: number;
  hMax: number;
  outlier: boolean;
  truncated: boolean;
}

export function GaussianBellHills3D(props: GaussianBellHills3DProps) {
  const surface = useGnbSliceStore((s) => s.surface);
  const trace = useGnbSliceStore((s) => s.traces[0] ?? null);
  const axisX = useGnbSliceStore((s) => s.axisX);
  const axisY = useGnbSliceStore((s) => s.axisY);
  const surfaceMode = useGnbSliceStore((s) => s.surfaceMode);
  const visibleHills = useGnbSliceStore((s) => s.visibleHills);
  const showPins = useGnbSliceStore((s) => s.showPins);
  const showRings = useGnbSliceStore((s) => s.showRings);

  const scene: HillScene = useMemo(() => {
    const empty: HillScene = {
      meshes: [],
      rings: [],
      pins: new THREE.BufferGeometry(),
      pinTips: new THREE.BufferGeometry(),
      peaks: new THREE.BufferGeometry(),
      queryPos: new THREE.Vector3(0, 0.1, 0),
      pinCount: 0,
      ringCount: 0,
      hMax: 0,
      outlier: false,
      truncated: false,
    };
    if (!surface || surface.x_grid.length < 2 || surface.y_grid.length < 2) return empty;
    const res = surface.resolution;
    if (surface.z_values.length < res * res) return empty;
    const overlays = surface.extra_overlays;
    const hills = (overlays?.bell_hills_per_class ?? {}) as Record<string, number[]>;
    const order = (overlays?.hill_class_order ?? []) as number[];
    const peaks2d = (overlays?.peaks_2d ?? []) as number[][];
    const stds2d = (overlays?.stds_2d ?? []) as number[][];
    const priors = (overlays?.class_priors ?? []) as number[];
    const keys = order.length > 0 ? order.map(String) : Object.keys(hills);
    const shown = keys
      .map((k) => Number(k))
      .filter((k) => visibleHills.includes(k) && Array.isArray(hills[String(k)]));

    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const y0 = ys[0];
    const spanX = xs[xs.length - 1] - x0 || 1;
    const spanY = ys[ys.length - 1] - y0 || 1;
    const posterior = surfaceMode === 'posterior';

    // Per-vertex posterior normalization needs the full stack.
    const stacks = shown.map((k) => hills[String(k)] ?? []);
    let hMax = 1e-12;
    const disps = stacks.map((arr) =>
      arr.map((_, idx) => {
        let v = arr[idx] ?? 0;
        if (posterior) {
          let total = 0;
          for (const other of stacks) total += other[idx] ?? 0;
          v = total > 0 ? v / total : 0;
        }
        if (Number.isFinite(v) && v > hMax) hMax = v;
        return Number.isFinite(v) && v > 0 ? v : 0;
      }),
    );

    const meshes: HillScene['meshes'] = [];
    shown.forEach((k, hi) => {
      const disp = disps[hi];
      const posArr = new Float32Array(res * res * 3);
      for (let r = 0; r < res; r += 1) {
        for (let c = 0; c < res; c += 1) {
          const idx = r * res + c;
          posArr[idx * 3] = ((xs[c] - x0) / spanX) * 5 - 2.5;
          posArr[idx * 3 + 1] = normalizeHillHeight(hMax, disp[idx]);
          posArr[idx * 3 + 2] = ((ys[r] - y0) / spanY) * 5 - 2.5;
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
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
      geo.setIndex(index);
      geo.computeVertexNormals();
      const rgb = hillColor(k);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
        transparent: true,
        opacity: 0.55,
        side: THREE.DoubleSide,
        roughness: 0.5,
        metalness: 0.05,
        depthWrite: false,
      });
      meshes.push({ geo, mat, k });
    });

    // 1σ / 2σ footprint rings on the base plane (axis-aligned ellipses).
    const rings: HillScene['rings'] = [];
    if (showRings) {
      shown.forEach((k, hi) => {
        const orderIdx = order.length > 0 ? order.indexOf(k) : hi;
        const peak = peaks2d[orderIdx] ?? peaks2d[hi] ?? [x0, y0];
        const std = stds2d[orderIdx] ?? stds2d[hi] ?? [spanX / 6, spanY / 6];
        const cx = ((peak[0] - x0) / spanX) * 5 - 2.5;
        const cz = ((peak[1] - y0) / spanY) * 5 - 2.5;
        const rx = (std[0] / spanX) * 5;
        const rz = (std[1] / spanY) * 5;
        [1, 2].forEach((mult) => {
          const curve = new THREE.EllipseCurve(cx, cz, rx * mult, rz * mult, 0, Math.PI * 2);
          const pts2 = curve.getPoints(72);
          const pts3 = pts2.map((p) => new THREE.Vector3(p.x, 0.03 + mult * 0.012, p.y));
          const geo = new THREE.BufferGeometry().setFromPoints(pts3);
          rings.push({ geo, k });
        });
      });
    }

    // Drop-pins at the live query point (generative heights for comparison).
    const live = props.liveFeatures;
    let queryPos = new THREE.Vector3(0, 0.1, 0);
    let pins = new THREE.BufferGeometry();
    let pinTips = new THREE.BufferGeometry();
    let pinCount = 0;
    if (live && live.length > 0) {
      const lx = live[axisX];
      const ly = live[axisY];
      if (Number.isFinite(lx) && Number.isFinite(ly)) {
        const qx = ((lx - x0) / spanX) * 5 - 2.5;
        const qz = ((ly - y0) / spanY) * 5 - 2.5;
        queryPos = new THREE.Vector3(qx, 0.1, qz);
        const rawHeights = shown.map((k, hi) => {
          const orderIdx = order.length > 0 ? order.indexOf(k) : hi;
          const peak = peaks2d[orderIdx] ?? peaks2d[hi] ?? [0, 0];
          const std = stds2d[orderIdx] ?? stds2d[hi] ?? [1, 1];
          return bellHillHeight(priors[k] ?? 0, peak[0], peak[1], std[0], std[1], lx, ly);
        });
        const postHeights = posteriorFromHills(rawHeights);
        const pinPos = new Float32Array(shown.length * 6);
        const tipPos = new Float32Array(shown.length * 3);
        shown.forEach((k, hi) => {
          const h = posterior ? postHeights[hi] : rawHeights[hi];
          const y = normalizeHillHeight(posterior ? 1 : hMax, h);
          pinPos[hi * 6] = qx;
          pinPos[hi * 6 + 1] = 0;
          pinPos[hi * 6 + 2] = qz;
          pinPos[hi * 6 + 3] = qx;
          pinPos[hi * 6 + 4] = Math.max(y, 0.06);
          pinPos[hi * 6 + 5] = qz;
          tipPos[hi * 3] = qx;
          tipPos[hi * 3 + 1] = Math.max(y, 0.06);
          tipPos[hi * 3 + 2] = qz;
        });
        pins = new THREE.BufferGeometry();
        pins.setAttribute('position', new THREE.BufferAttribute(pinPos, 3));
        pinTips = new THREE.BufferGeometry();
        pinTips.setAttribute('position', new THREE.BufferAttribute(tipPos, 3));
        pinCount = shown.length;
      }
    }

    // Peak markers.
    let peaksGeo = new THREE.BufferGeometry();
    if (shown.length > 0) {
      const pp = new Float32Array(shown.length * 3);
      shown.forEach((k, hi) => {
        const orderIdx = order.length > 0 ? order.indexOf(k) : hi;
        const peak = peaks2d[orderIdx] ?? peaks2d[hi] ?? [x0, y0];
        pp[hi * 3] = ((peak[0] - x0) / spanX) * 5 - 2.5;
        pp[hi * 3 + 1] = normalizeHillHeight(posterior ? 1 : hMax, posterior ? 1 : priors[k] ?? 0) + 0.05;
        pp[hi * 3 + 2] = ((peak[1] - y0) / spanY) * 5 - 2.5;
      });
      peaksGeo = new THREE.BufferGeometry();
      peaksGeo.setAttribute('position', new THREE.BufferAttribute(pp, 3));
    }

    return {
      meshes,
      rings,
      pins,
      pinTips,
      peaks: peaksGeo,
      queryPos,
      pinCount,
      ringCount: rings.length,
      hMax,
      outlier: trace?.extra_mechanics?.is_low_density_outlier === true,
      truncated: overlays?.hills_truncated_to_top5 === true,
    };
  }, [surface, trace, props.liveFeatures, axisX, axisY, surfaceMode, visibleHills, showRings]);

  const ringMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#f8fafc'),
        transparent: true,
        opacity: 0.5,
      }),
    [],
  );
  const pinMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#e2e8f0'),
        transparent: true,
        opacity: 0.85,
      }),
    [],
  );
  const tipMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.12,
        color: new THREE.Color('#fef08a'),
        transparent: true,
        opacity: 1.0,
        sizeAttenuation: true,
        depthWrite: false,
      }),
    [],
  );
  const peakMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.1,
        color: new THREE.Color('#ffffff'),
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: true,
        depthWrite: false,
      }),
    [],
  );
  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#22d3ee'),
        emissive: new THREE.Color('#0e7490'),
        emissiveIntensity: 0.9,
      }),
    [],
  );
  useDisposableThreeResources(
    [
      ...scene.meshes.map((m) => m.geo),
      ...scene.rings.map((r) => r.geo),
      scene.pins,
      scene.pinTips,
      scene.peaks,
    ],
    [...scene.meshes.map((m) => m.mat), ringMaterial, pinMaterial, tipMaterial, peakMaterial, ballMaterial],
  );

  if (!surface) return null;

  return (
    <div className="flex flex-col gap-1">
      <div
        className="relative w-full overflow-hidden rounded"
        style={{ height: 300, border: '1px solid var(--border-subtle)', background: '#0b0f14' }}
      >
        <Canvas
          dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
          camera={{ position: [4.5, 3.8, 5.5], fov: 42, near: 0.1, far: 100 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
        >
          <ambientLight intensity={0.8} />
          <directionalLight position={[6, 8, 4]} intensity={1.15} />
          {scene.meshes.map((m) => (
            <mesh key={m.k} geometry={m.geo} material={m.mat} />
          ))}
          {showRings &&
            scene.rings.map((r, i) => <lineLoop key={`${r.k}-${i}`} geometry={r.geo} material={ringMaterial} />)}
          {showPins && scene.pinCount > 0 && (
            <>
              <lineSegments geometry={scene.pins} material={pinMaterial} />
              <points geometry={scene.pinTips} material={tipMaterial} />
            </>
          )}
          <points geometry={scene.peaks} material={peakMaterial} />
          <mesh position={scene.queryPos} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#67e8f6', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          {surfaceMode === 'generative' ? '▲ Generative bell hills P(x|C)·π' : '▲ Posterior hills P(C|x)'}
          {scene.truncated ? ' · top-5 priors' : ''}
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          {scene.meshes.length} hills · {scene.pinCount} drop-pins · {scene.ringCount} σ-rings · drag to orbit
        </span>
        {scene.outlier && (
          <span
            className="font-mono-tabular pointer-events-none absolute top-8 left-2 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: 'rgba(245,158,11,0.16)', color: '#fcd34d', border: '1px solid rgba(245,158,11,0.45)' }}
          >
            ⚠ LOW DENSITY OUTLIER: far from all centers (&gt; 4σ)
          </span>
        )}
      </div>
    </div>
  );
}
