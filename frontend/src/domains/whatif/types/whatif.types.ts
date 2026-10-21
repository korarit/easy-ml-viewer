/** Phase 13 — what-if sandbox + counterfactual shared types. */

export type FeatureKind = 'float' | 'integer' | 'binary';

export interface FeatureRange {
  feature_index: number;
  feature_name: string;
  min_value: number;
  max_value: number;
  mean_value: number;
  step: number;
  kind: FeatureKind;
  locked: boolean;
}

export interface FeatureChangeDelta {
  feature_index: number;
  feature_name: string;
  original_value: number;
  mutated_value: number;
  delta_value: number;
  delta_percent: number;
  is_out_of_distribution: boolean;
  is_locked: boolean;
}

export interface WhatIfEvaluation {
  base_frame_index: number;
  dataset_row_index: number;
  original_position_world: [number, number, number];
  mutated_position_world: [number, number, number];
  original_prediction: number;
  mutated_prediction: number;
  original_confidence: number;
  mutated_confidence: number;
  original_class: number | null;
  mutated_class: number | null;
  decision_flipped: boolean;
  world_distance: number;
  changed_features: FeatureChangeDelta[];
  base_row_values: number[];
  mutated_row_values: number[];
  internal_model_trace: Record<string, unknown>;
}

export interface CounterfactualResult {
  found: boolean;
  status_message: string;
  evaluation: WhatIfEvaluation | null;
  locked_feature_indices: number[];
  target_description: string;
}

export interface WhatIfSessionPayload {
  session_id: string;
  total_frames: number;
  is_classification: boolean;
  axis_x_feature_index: number;
  axis_y_feature_index: number;
  feature_names: string[];
  feature_kinds: FeatureKind[];
  row_indices: number[];
  feature_ranges: FeatureRange[];
}

/** True when the user changed ≥1 slider away from the base row. */
export function isSandboxActive(mutated: Record<number, number>, base: number[]): boolean {
  return Object.entries(mutated).some(([k, v]) => {
    const baseVal = base[Number(k)];
    return baseVal !== undefined && Math.abs(v - baseVal) > 1e-9;
  });
}

/** `+75.0 (+12.4%)` delta chip label (spec §2.1). */
export function formatDelta(delta: number, percent: number): string {
  if (!Number.isFinite(delta)) return '—';
  const sign = delta >= 0 ? '+' : '';
  const pct = Number.isFinite(percent) ? ` (${sign}${percent.toFixed(1)}%)` : '';
  const abs = Math.abs(delta);
  const body = abs !== 0 && (abs >= 1_000_000 || abs < 0.0001) ? delta.toExponential(2) : trim(delta);
  return `Δ ${sign}${body}${pct}`;
}

function trim(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 100_000) return String(v);
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(v);
}

/** Smart value formatter shared by sliders and scorecards. */
export function formatWhatIfValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  return trim(v);
}

/**
 * One-line counterfactual recipe, e.g.
 * "credit_score 610 → 655 (+45), debt_ratio 0.48 → 0.35 (-0.13)".
 */
export function buildRecipeSummary(changes: FeatureChangeDelta[]): string {
  if (changes.length === 0) return 'No changes needed — already at the target.';
  return changes
    .map((c) => {
      const sign = c.delta_value >= 0 ? '+' : '';
      return `${c.feature_name} ${trim(c.original_value)} → ${trim(c.mutated_value)} (${sign}${trim(c.delta_value)})`;
    })
    .join(', ');
}

/** Whether a mutated value left the dataset [min, max] envelope. */
export function isOutOfDistribution(value: number, range: FeatureRange): boolean {
  return value < range.min_value || value > range.max_value;
}
