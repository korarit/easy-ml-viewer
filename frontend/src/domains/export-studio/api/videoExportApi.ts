/** Phase 57 — Advanced Video Export Studio API client (100% offline localhost). */
import type { AdvancedVideoExportOptions } from '../types/advancedVideoExport.types';

const BASE = '/api/v1/exports';

function toBackendPayload(o: AdvancedVideoExportOptions) {
  return {
    dimension: o.dimension,
    layout: o.layout,
    include_why_panel: o.includeWhyPanel,
    include_what_if_panel: o.includeWhatIfPanel,
    theme: o.theme,
    camera_preset: o.cameraPreset,
    speed_multiplier: o.speedMultiplier,
    fps: o.fps,
    format: o.format,
    resolution: o.resolution,
    bitrate_mbps: o.bitrateMbps,
    range_start: o.rangeStartFrame,
    range_end: o.rangeEndFrame,
  };
}

export interface VideoPlanResponse {
  width: number;
  height: number;
  n_frames: number;
  duration_sec: number;
  playback_sec: number;
  estimated_bytes: number;
  layout_rects: Record<string, { x: number; y: number; w: number; h: number }>;
  theme_tokens: Record<string, string | number>;
  video_encoder_config: { codec: string; container: string; fps: number; bitrate_bps: number };
}

export async function fetchVideoPlan(options: AdvancedVideoExportOptions): Promise<VideoPlanResponse> {
  const res = await fetch(`${BASE}/video-plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toBackendPayload(options)),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message ?? `Video plan failed (HTTP ${res.status})`);
  }
  return res.json() as Promise<VideoPlanResponse>;
}

/** Fallback path: upload a .webm clip, get an .mp4 back (or 501 when no PyAV). */
export async function transcodeWebmToMp4(blob: Blob, filename = 'studio-clip.webm'): Promise<Blob> {
  const form = new FormData();
  form.append('file', blob, filename);
  const res = await fetch(`${BASE}/transcode-mp4`, { method: 'POST', body: form });
  if (res.status === 501) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message ?? 'MP4 transcode backend unavailable');
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message ?? `Transcode failed (HTTP ${res.status})`);
  }
  return await res.blob();
}
