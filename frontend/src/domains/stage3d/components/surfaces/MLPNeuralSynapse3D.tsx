import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import {
  mlpActivationSigma,
  mlpFluxColor,
  mlpFluxPolarity,
  mlpLayerX,
  mlpNodeRadius,
  mlpNodeY,
} from './mlpSurfaceMath';

export interface MlpSynapseEdgeProp {
  fromLayer: number;
  u: number;
  v: number;
  flux: number;
}

export interface MLPNeuralSynapse3DProps {
  layerDimensions: number[];
  renderedNodeIndices: number[][];
  /** Per-layer activations for the current frame (arrays or Float32Arrays). */
  nodeActivations: Array<ArrayLike<number>>;
  deadNeuronMasks: boolean[][];
  synapseEdges: MlpSynapseEdgeProp[];
  fluxThreshold?: number;
  showDeadOnly?: boolean;
  hoveredNode?: { layer: number; index: number } | null;
  onHoverNode?: (node: { layer: number; index: number } | null) => void;
}

const MAX_RENDERED_EDGES = 512;

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _pos = new THREE.Vector3();
const _scl = new THREE.Vector3();
const _col = new THREE.Color();

interface NodeSlot {
  key: string;
  layer: number;
  slot: number;
  origIdx: number;
  x: number;
  y: number;
  z: number;
}

/**
 * Phase 36 — 3D floating synapse network: one InstancedMesh draw call for all
 * neurons plus a single LineSegments cloud for synapse flux pulses.
 *
 * Cyan = excitatory (+flux), crimson = inhibitory (−flux), hollow grey =
 * dead ReLU. Frame updates write straight into instance/color buffers
 * (no per-frame allocations) to hold 60 FPS scrubbing.
 */
export function MLPNeuralSynapse3D(props: MLPNeuralSynapse3DProps) {
  const {
    layerDimensions,
    renderedNodeIndices,
    nodeActivations,
    deadNeuronMasks,
    synapseEdges,
    fluxThreshold = 0,
    showDeadOnly = false,
    hoveredNode = null,
    onHoverNode,
  } = props;
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const linesRef = useRef<THREE.LineSegments>(null!);

  const layerCount = layerDimensions.length;

  const slots = useMemo<NodeSlot[]>(() => {
    const out: NodeSlot[] = [];
    for (let l = 0; l < layerCount; l += 1) {
      const rendered = renderedNodeIndices[l] ?? [];
      const count = rendered.length;
      for (let s = 0; s < count; s += 1) {
        const origIdx = rendered[s] ?? s;
        out.push({
          key: `${l}:${origIdx}`,
          layer: l,
          slot: s,
          origIdx,
          x: mlpLayerX(l, layerCount),
          y: mlpNodeY(s, count),
          z: ((origIdx * 37 + l * 11) % 13) / 13 - 0.5,
        });
      }
    }
    return out;
  }, [layerCount, renderedNodeIndices]);

  const slotIndex = useMemo(() => {
    const map = new Map<string, number>();
    slots.forEach((s, i) => map.set(s.key, i));
    return map;
  }, [slots]);

  const edges = useMemo(() => {
    const kept = synapseEdges.filter((e) => Math.abs(e.flux) >= fluxThreshold);
    kept.sort((a, b) => Math.abs(b.flux) - Math.abs(a.flux));
    return kept.slice(0, MAX_RENDERED_EDGES);
  }, [synapseEdges, fluxThreshold]);

  const linePositions = useMemo(() => {
    const arr = new Float32Array(Math.max(edges.length, 1) * 6);
    edges.forEach((e, i) => {
      const a = slotIndex.get(`${e.fromLayer}:${e.u}`);
      const b = slotIndex.get(`${e.fromLayer + 1}:${e.v}`);
      const o = i * 6;
      if (a == null || b == null) {
        arr[o] = arr[o + 1] = arr[o + 2] = arr[o + 3] = arr[o + 4] = arr[o + 5] = 0;
        return;
      }
      arr[o] = slots[a].x;
      arr[o + 1] = slots[a].y;
      arr[o + 2] = slots[a].z;
      arr[o + 3] = slots[b].x;
      arr[o + 4] = slots[b].y;
      arr[o + 5] = slots[b].z;
    });
    return arr;
  }, [edges, slotIndex, slots]);

  const lineColors = useMemo(() => {
    const arr = new Float32Array(Math.max(edges.length, 1) * 6);
    edges.forEach((e, i) => {
      const [r, g, b] = mlpFluxColor(e.flux);
      const dim = 0.35 + 0.65 * Math.min(1, Math.abs(e.flux));
      const o = i * 6;
      arr[o] = r * dim;
      arr[o + 1] = g * dim;
      arr[o + 2] = b * dim;
      arr[o + 3] = r * dim;
      arr[o + 4] = g * dim;
      arr[o + 5] = b * dim;
    });
    return arr;
  }, [edges]);

  const lineGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
    return geo;
  }, [linePositions, lineColors]);
  const lineMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // Per-frame value write: instance matrices + colors in place, zero allocs.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || slots.length === 0) return;
    const sigmaByLayer = new Map<number, number>();
    for (let l = 0; l < layerCount; l += 1) {
      const acts = nodeActivations[l];
      const sample: number[] = [];
      if (acts) {
        for (let i = 0; i < Math.min(acts.length, 64); i += 1) sample.push(Number(acts[i]) || 0);
      }
      sigmaByLayer.set(l, mlpActivationSigma(sample));
    }
    slots.forEach((s, i) => {
      const acts = nodeActivations[s.layer];
      const raw = acts && s.slot < acts.length ? Number(acts[s.slot]) || 0 : 0;
      const dead = deadNeuronMasks[s.layer]?.[s.origIdx] === true;
      const sigma = sigmaByLayer.get(s.layer) ?? 1;
      const r = dead ? 0.05 : mlpNodeRadius(Math.abs(raw), sigma);
      const dimmed = showDeadOnly && !dead;
      _pos.set(s.x, s.y, s.z);
      _scl.setScalar(dimmed ? r * 0.4 : r);
      _m.compose(_pos, _q, _scl);
      mesh.setMatrixAt(i, _m);
      const isOutput = s.layer === layerCount - 1;
      if (dead) {
        _col.setRGB(0.42, 0.45, 0.5);
      } else if (isOutput) {
        const t = Math.min(1, Math.max(0, raw));
        _col.setRGB(0.35 + t * 0.65, 0.75, 0.45);
      } else {
        const glow = Math.min(1, Math.abs(raw) / (sigma + 1e-9));
        _col.setRGB(0.2 + glow * 0.25, 0.65 + glow * 0.3, 0.9);
      }
      if (hoveredNode && hoveredNode.layer === s.layer && hoveredNode.index === s.origIdx) {
        _col.setRGB(1.0, 0.85, 0.3);
        _scl.setScalar(r * 1.5);
        _m.compose(_pos, _q, _scl);
        mesh.setMatrixAt(i, _m);
      }
      if (dimmed) _col.multiplyScalar(0.25);
      mesh.setColorAt(i, _col);
    });
    mesh.count = slots.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    const lines = linesRef.current;
    if (lines) {
      (lines.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
      (lines.geometry.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
    }
  }, [slots, nodeActivations, deadNeuronMasks, layerCount, showDeadOnly, hoveredNode]);

  useDisposableThreeResources([lineGeo], [lineMat]);

  const excitatory = edges.filter((e) => mlpFluxPolarity(e.flux) === 'excitatory').length;
  const inhibitory = edges.filter((e) => mlpFluxPolarity(e.flux) === 'inhibitory').length;

  return (
    <div className="flex flex-col gap-1">
      <div
        className="relative w-full overflow-hidden rounded"
        style={{ height: 320, border: '1px solid var(--border-subtle)', background: '#0b0f14' }}
      >
        <Canvas
          dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
          camera={{ position: [0.4, 1.6, 7.2], fov: 45, near: 0.1, far: 100 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          onPointerMissed={() => onHoverNode?.(null)}
        >
          <ambientLight intensity={0.85} />
          <directionalLight position={[4, 6, 5]} intensity={1.1} />
          <instancedMesh
            ref={meshRef}
            args={[undefined, undefined, Math.max(slots.length, 1)]}
            frustumCulled={false}
            onPointerMove={(e) => {
              const id = e.instanceId;
              if (id == null || !onHoverNode) return;
              const s = slots[id];
              if (s) onHoverNode({ layer: s.layer, index: s.origIdx });
            }}
            onPointerOut={() => onHoverNode?.(null)}
          >
            <sphereGeometry args={[1, 14, 14]} />
            <meshStandardMaterial roughness={0.35} metalness={0.1} emissive="#0e7490" emissiveIntensity={0.55} />
          </instancedMesh>
          {edges.length > 0 && (
            <lineSegments ref={linesRef} geometry={lineGeo} material={lineMat} frustumCulled={false} />
          )}
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#a5f3fc', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          ◆ MLP synapse brain ({layerDimensions.join(' → ')} · {edges.length} flux lines)
        </span>
        <span className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]" style={{ color: '#94a3b8' }}>
          cyan = excitatory (+flux {excitatory}) · crimson = inhibitory (−flux {inhibitory}) · grey = dead ReLU · hover to spotlight · drag to orbit
        </span>
      </div>
    </div>
  );
}
