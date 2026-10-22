import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMlpSliceStore, type MlpViewMode } from '../store/useMlpSliceStore';
import { MLPNeuralSynapse3D } from '../../stage3d/components/surfaces/MLPNeuralSynapse3D';
import { mlpGridToStage, mlpPosteriorHeight, normalizeMlpHeight } from '../../stage3d/components/surfaces/mlpSurfaceMath';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';

interface MlpSurfaceSandboxProps {
  hovered: { layer: number; index: number } | null;
  onHoverNode: (node: { layer: number; index: number } | null) => void;
}

/** Phase 36 — view-mode switcher, axis/class controls, synapse brain + decision surface. */
export function MlpSurfaceSandbox({ hovered, onHoverNode }: MlpSurfaceSandboxProps) {
  const {
    featureNames,
    axisX,
    axisY,
    classIdx,
    classCount,
    surface,
    isSurfacing,
    isRegressor,
    viewMode,
    showDeadOnly,
    fluxThreshold,
    showSurface,
    showWireframe,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setViewMode,
    setShowDeadOnly,
    setFluxThreshold,
    setShowSurface,
    setShowWireframe,
    traces,
  } = useMlpSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useMlpSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const mech = traces[0]?.extra_mechanics ?? null;
  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const zRange = (overlays?.regression_z_range ?? [0, 1]) as [number, number];

  const brainProps = useMemo(() => {
    if (!mech) return null;
    const layerDimensions = (mech.layer_dimensions ?? []) as number[];
    const renderedNodeIndices = (mech.rendered_node_indices ?? []) as number[][];
    const layerActs = (mech.layer_activations_sample ?? []) as number[][];
    const deadNeuronMasks = (mech.dead_neuron_masks ?? []) as boolean[][];
    const synapseEdges = (mech.synapse_edges ?? []) as Array<{ fromLayer: number; u: number; v: number; flux: number }>;
    return { layerDimensions, renderedNodeIndices, nodeActivations: layerActs, deadNeuronMasks, synapseEdges };
  }, [mech]);

  if (featureNames.length === 0) return null;
  const telemetry = surface?.telemetry;
  const showBrain = viewMode === 'brain' || viewMode === 'split';
  const showSurf = (viewMode === 'surface' || viewMode === 'split') && showSurface;

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D synapse network + surface
        </span>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          View
          <select
            aria-label="View mode"
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as MlpViewMode)}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            <option value="split">Split Dual View</option>
            <option value="brain">3D Synapse Brain Only</option>
            <option value="surface">3D Surface Only</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Axis X
          <select
            aria-label="Axis X"
            value={axisX}
            onChange={(e) => setAxisX(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {featureNames.map((n, i) => (
              <option key={n} value={i}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Axis Y
          <select
            aria-label="Axis Y"
            value={axisY}
            onChange={(e) => setAxisY(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {featureNames.map((n, i) => (
              <option key={n} value={i}>
                {n}
              </option>
            ))}
          </select>
        </label>
        {!isRegressor && (
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            Target
            <select
              aria-label="Target class"
              value={classIdx}
              onChange={(e) => setClassIdx(Number(e.target.value))}
              className="rounded px-1.5 py-0.5"
              style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
            >
              {Array.from({ length: classCount }, (_, i) => (
                <option key={i} value={i}>
                  class {i}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Flux ≥ {fluxThreshold.toFixed(2)}
          <input
            aria-label="Flux threshold"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={Math.min(fluxThreshold, 1)}
            onChange={(e) => setFluxThreshold(Number(e.target.value))}
            className="w-24"
          />
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showDeadOnly} onChange={(e) => setShowDeadOnly(e.target.checked)} />
          Highlight dead neurons
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showSurface} onChange={(e) => setShowSurface(e.target.checked)} />
          Surface
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showWireframe} onChange={(e) => setShowWireframe(e.target.checked)} />
          Wireframe
        </label>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} voxels ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      <div className={viewMode === 'split' ? 'mt-2 grid gap-2 md:grid-cols-2' : 'mt-2'}>
        {showBrain &&
          (brainProps ? (
            <MLPNeuralSynapse3D
              layerDimensions={brainProps.layerDimensions}
              renderedNodeIndices={brainProps.renderedNodeIndices}
              nodeActivations={brainProps.nodeActivations}
              deadNeuronMasks={brainProps.deadNeuronMasks}
              synapseEdges={brainProps.synapseEdges}
              fluxThreshold={fluxThreshold}
              showDeadOnly={showDeadOnly}
              hoveredNode={hovered}
              onHoverNode={onHoverNode}
            />
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Run a trace to light up the synapse brain.</p>
          ))}
        {showSurf &&
          (surface && res >= 2 ? (
            <MlpTerrain
              surface={surface}
              res={res}
              zRange={zRange}
              isRegressor={isRegressor}
              showWireframe={showWireframe}
            />
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Run the surface pass to grow the decision manifold.</p>
          ))}
      </div>

      {showTelemetry && surface && (
        <pre
          className="font-mono-tabular mt-2 max-h-48 overflow-auto rounded p-2 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        >
          {JSON.stringify(
            {
              visual_type: surface.visual_type,
              resolution: surface.resolution,
              layer_dimensions: surface.extra_overlays?.layer_dimensions,
              activation: surface.extra_overlays?.activation,
              out_activation: surface.extra_overlays?.out_activation,
              telemetry: surface.telemetry,
              z_sample: surface.z_values.slice(0, 8),
            },
            null,
            2,
          )}
        </pre>
      )}
      <p className="font-mono-tabular mt-1" style={{ color: 'var(--text-muted)' }}>
        Tip: Ctrl+Shift+S toggles the JSON telemetry drawer.
      </p>
    </div>
  );
}

function MlpTerrain(props: {
  surface: { x_grid: number[]; y_grid: number[]; z_values: number[] };
  res: number;
  zRange: [number, number];
  isRegressor: boolean;
  showWireframe: boolean;
}) {
  const { surface, res, zRange, isRegressor, showWireframe } = props;
  const meshRef = useRef<THREE.InstancedMesh>(null!);

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const toH = (raw: number): number =>
      isRegressor ? normalizeMlpHeight(zRange[0], zRange[1], raw) : mlpPosteriorHeight(raw);
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const raw = surface.z_values[r * res + c] ?? 0;
        const h = toH(raw);
        out.push({
          x: -2.5 + (c + 0.5) * cell,
          y: Math.max(h, 0.02) / 2,
          z: -2.5 + (r + 0.5) * cell,
          sx: cell * 0.92,
          sy: Math.max(h, 0.02),
          sz: cell * 0.92,
          shade: isRegressor ? h / 2 : Math.min(1, Math.max(0, raw)),
        });
      }
    }
    return out;
  }, [surface, res, isRegressor, zRange]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || cells.length === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const col = new THREE.Color();
    cells.forEach((cell, i) => {
      pos.set(cell.x, cell.y, cell.z);
      scl.set(cell.sx, cell.sy, cell.sz);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      // Morphing manifold: violet domes rising out of a teal basin.
      col.setRGB(0.35 + cell.shade * 0.45, 0.4 + cell.shade * 0.25, 0.75 + cell.shade * 0.2);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  const x0 = surface.x_grid[0] ?? 0;
  const x1 = surface.x_grid[surface.x_grid.length - 1] ?? 1;
  const y0 = surface.y_grid[0] ?? 0;
  const y1 = surface.y_grid[surface.y_grid.length - 1] ?? 1;

  useDisposableThreeResources([], []);

  return (
    <div className="flex flex-col gap-1">
      <div
        className="relative w-full overflow-hidden rounded"
        style={{ height: 320, border: '1px solid var(--border-subtle)', background: '#0b0f14' }}
      >
        <Canvas
          dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
          camera={{ position: [5.2, 4.4, 6.4], fov: 42, near: 0.1, far: 100 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
        >
          <ambientLight intensity={0.8} />
          <directionalLight position={[6, 8, 4]} intensity={1.15} />
          <instancedMesh ref={meshRef} args={[undefined, undefined, Math.max(cells.length, 1)]} frustumCulled={false}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial roughness={0.6} metalness={0.05} wireframe={showWireframe} />
          </instancedMesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#a5f3fc', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          ◆ MLP morphing manifold ({res}×{res} · X [{x0.toFixed(2)}..{x1.toFixed(2)}] · Y [{y0.toFixed(2)}..{y1.toFixed(2)}])
        </span>
        <span className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]" style={{ color: '#94a3b8' }}>
          X [{mlpGridToStage(x0, x0, x1).toFixed(1)}..{mlpGridToStage(x1, x0, x1).toFixed(1)}] · drag to orbit
        </span>
      </div>
    </div>
  );
}
