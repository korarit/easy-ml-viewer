/** Phase 12 — local explainability shared types. */

export interface FeatureAttributionItem {
  feature_index: number;
  feature_name: string;
  raw_value: number;
  percentile: number;
  z_score: number;
  contribution: number;
}

export interface FrameExplanation {
  frame_index: number;
  dataset_row_index: number;
  base_value: number;
  output_value: number;
  other_contributions_sum: number;
  attributions: FeatureAttributionItem[];
}

export interface ExplainabilitySessionPayload {
  session_id: string;
  total_frames: number;
  is_classification: boolean;
  base_value: number;
  feature_names: string[];
  row_indices: number[];
}

export interface FrameExplanationPayload {
  frame_index: number;
  dataset_row_index: number;
  base_value: number;
  output_value: number;
  other_contributions_sum: number;
  attributions: FeatureAttributionItem[];
}

/** Top-K selector options (edge #2: wide models collapse the tail). */
export const TOP_K_OPTIONS = [5, 10, 15] as const;
export type TopKOption = (typeof TOP_K_OPTIONS)[number] | 'all';

/** P99/P01 outlier band (edge #6 + test-phase §3.1 badge rule). */
export function isOutlierPercentile(percentile: number): boolean {
  return percentile >= 99 || percentile <= 1;
}

/**
 * Smart raw-value formatter (edge #6): scientific notation below 1e-4 or
 * above 1e6 so long decimals never overflow the inspector table.
 */
export function formatRawValue(raw: number): string {
  if (!Number.isFinite(raw)) return '—';
  const abs = Math.abs(raw);
  if (abs !== 0 && (abs >= 1_000_000 || abs < 0.0001)) return raw.toExponential(2);
  if (Number.isInteger(raw) && abs < 100_000) return String(raw);
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(raw);
}

/** Signed contribution label, e.g. `+0.312` / `-0.145`. */
export function formatContribution(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const rounded = Math.round(value * 1000) / 1000;
  return `${rounded >= 0 ? '+' : ''}${rounded.toFixed(3)}`;
}
