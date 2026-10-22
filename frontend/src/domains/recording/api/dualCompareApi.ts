/** Phase 40 — typed client for the synchronized dual `.mlv` compare endpoint. */
import { API_BASE_URL } from '../../../core/api/client';

export interface DualComparisonResult {
  total_frames_a: number;
  total_frames_b: number;
  total_aligned_frames: number;
  truncated: boolean;
  truncation_notice: string;
  is_classification: boolean;
  divergence_kind: string;
  disagreement_count: number;
  disagreement_indices: number[];
  disagreement_indices_truncated: boolean;
  probability_divergence: number[];
  absolute_error: number[];
  agreement_rate: number;
  mean_divergence: number;
  max_divergence: number;
  file_a_name: string;
  file_b_name: string;
}

/** Index of the next disagreement frame strictly after `fromFrame` (wraps around). */
export function nextDisagreementIndex(indices: number[], fromFrame: number): number | null {
  if (indices.length === 0) return null;
  const sorted = [...indices].sort((a, b) => a - b);
  for (const idx of sorted) if (idx > fromFrame) return idx;
  return sorted[0];
}

/** Index of the previous disagreement frame strictly before `fromFrame` (wraps around). */
export function prevDisagreementIndex(indices: number[], fromFrame: number): number | null {
  if (indices.length === 0) return null;
  const sorted = [...indices].sort((a, b) => a - b);
  for (let i = sorted.length - 1; i >= 0; i--) if (sorted[i] < fromFrame) return sorted[i];
  return sorted[sorted.length - 1];
}

export async function fetchCompareMlvs(fileA: File, fileB: File): Promise<DualComparisonResult> {
  const form = new FormData();
  form.append('file_a', fileA, fileA.name);
  form.append('file_b', fileB, fileB.name);
  const res = await fetch(`${API_BASE_URL}/api/v1/recordings/compare`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    let code = `HTTP_${res.status}`;
    let message = `Compare failed (${res.status})`;
    try {
      const data = (await res.clone().json()) as {
        error?: { code?: string; message?: string };
      };
      if (data?.error?.code) code = data.error.code;
      if (data?.error?.message) message = data.error.message;
    } catch {
      /* keep defaults */
    }
    const err = new Error(message) as Error & { code: string; httpStatus: number };
    err.code = code;
    err.httpStatus = res.status;
    throw err;
  }
  return (await res.json()) as DualComparisonResult;
}
