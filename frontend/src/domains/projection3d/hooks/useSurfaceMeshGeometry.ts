import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { sampleColormap, type ColormapName } from '../components/colormaps';

export interface SurfaceMeshInput {
  xRange: [number, number];
  yRange: [number, number];
  zHeightsWorld: Float32Array | null;
  rawValues: Float32Array | null;
  resolution: number;
  colormap: ColormapName;
}

function mapLinear(raw: number, lo: number, hi: number): number {
  const span = hi - lo;
  if (!Number.isFinite(raw) || Math.abs(span) < 1e-9) return 0;
  const t = (raw - lo) / span;
  return -5 + Math.min(1.5, Math.max(-0.5, t)) * 10;
}

function paintColors(
  colorAttr: THREE.BufferAttribute,
  rawValues: Float32Array,
  zHeights: Float32Array,
  colormap: ColormapName,
) {
  const src = rawValues.length === zHeights.length ? rawValues : zHeights;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < src.length; i += 1) {
    const v = src[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-12) {
    lo = 0;
    hi = 1;
  }
  for (let i = 0; i < src.length; i += 1) {
    const v = src[i];
    const t = Number.isFinite(v) ? (v - lo) / (hi - lo) : 0.5;
    const [r, g, b] = sampleColormap(colormap, t);
    colorAttr.setXYZ(i, r, g, b);
  }
  colorAttr.needsUpdate = true;
}

/**
 * Phase 09 — LOD surface geometry with in-place BufferAttribute updates.
 *
 * Same topology (resolution + raw ranges) → heights/colors are written into
 * the existing Float32Array and only `needsUpdate` + `computeVertexNormals()`
 * run (§5.2, zero reallocation during dynamic slicing). A resolution or
 * range change disposes the old geometry and rebuilds (VRAM-leak guard).
 * World X/Y are reconstructed from raw ranges; heights arrive in world
 * units; everything is swizzled to Y-up once at build time.
 */
export function useSurfaceMeshGeometry(input: SurfaceMeshInput): THREE.BufferGeometry | null {
  const { xRange, yRange, zHeightsWorld, rawValues, resolution, colormap } = input;
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const geoRef = useRef<THREE.BufferGeometry | null>(null);
  const keyRef = useRef<string>('');

  useEffect(() => {
    const count = resolution * resolution;
    if (!zHeightsWorld || zHeightsWorld.length !== count) {
      if (geoRef.current) {
        try {
          geoRef.current.dispose();
        } catch {
          /* already disposed */
        }
        geoRef.current = null;
        keyRef.current = '';
        setGeometry(null);
      }
      return;
    }
    const key = `${resolution}|${xRange[0]},${xRange[1]}|${yRange[0]},${yRange[1]}`;
    const geo = geoRef.current;
    if (geo && keyRef.current === key) {
      // In-place path: same grid, new heights/colors only.
      const pos = geo.attributes.position as THREE.BufferAttribute;
      const col = geo.attributes.color as THREE.BufferAttribute;
      for (let i = 0; i < count; i += 1) {
        const h = zHeightsWorld[i];
        pos.setY(i, Number.isFinite(h) ? h : 0);
      }
      pos.needsUpdate = true;
      paintColors(col, rawValues ?? zHeightsWorld, zHeightsWorld, colormap);
      geo.computeVertexNormals();
      return;
    }
    // Rebuild path: dispose the old LOD buffer first.
    if (geo) {
      try {
        geo.dispose();
      } catch {
        /* already disposed */
      }
    }
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const [x0, x1] = xRange;
    const [y0, y1] = yRange;
    for (let r = 0; r < resolution; r += 1) {
      const rawY = y0 + ((y1 - y0) * r) / Math.max(resolution - 1, 1);
      for (let c = 0; c < resolution; c += 1) {
        const rawX = x0 + ((x1 - x0) * c) / Math.max(resolution - 1, 1);
        const i = r * resolution + c;
        const h = zHeightsWorld[i];
        positions[i * 3] = mapLinear(rawX, x0, x1);
        positions[i * 3 + 1] = Number.isFinite(h) ? h : 0;
        positions[i * 3 + 2] = mapLinear(rawY, y0, y1);
      }
    }
    const index: number[] = [];
    for (let r = 0; r < resolution - 1; r += 1) {
      for (let c = 0; c < resolution - 1; c += 1) {
        const a = r * resolution + c;
        const b = a + 1;
        const d = a + resolution;
        const e = d + 1;
        index.push(a, d, b, b, d, e);
      }
    }
    const next = new THREE.BufferGeometry();
    next.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    next.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    next.setIndex(index);
    const colorAttr = next.attributes.color as THREE.BufferAttribute;
    paintColors(colorAttr, rawValues ?? zHeightsWorld, zHeightsWorld, colormap);
    next.computeVertexNormals();
    geoRef.current = next;
    keyRef.current = key;
    setGeometry(next);
  }, [xRange, yRange, zHeightsWorld, rawValues, resolution, colormap]);

  useEffect(() => {
    return () => {
      if (geoRef.current) {
        try {
          geoRef.current.dispose();
        } catch {
          /* already disposed */
        }
        geoRef.current = null;
      }
    };
  }, []);

  return geometry;
}
