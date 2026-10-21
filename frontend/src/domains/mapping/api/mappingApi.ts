/** Phase 05 — typed client for mapping endpoints. */
import { apiFetch } from '../../../core/api/client';

export type MatchTier = 'exact' | 'normalized' | 'fuzzy' | 'positional' | 'manual' | 'unmapped';
export type ImputationStrategy = 'zero' | 'mean' | 'median' | 'custom' | 'pass_nan';

export interface FeatureMappingItem {
  feature_index: number;
  model_feature_name: string;
  dataset_column_name: string | null;
  match_tier: MatchTier;
  match_confidence: number;
  imputation_strategy: ImputationStrategy;
  custom_fill_value: number;
  resolved_fill_value: number | null;
  is_type_compatible: boolean;
  cast_warning_or_error: string | null;
}

export interface DryRunReport {
  is_valid: boolean;
  tested_row_count: number;
  output_shape: [number, number];
  duplicate_columns_used: string[];
  target_leakage_warning: string | null;
  feature_errors: Record<number, string>;
  preview_matrix_head: Array<Array<number | null>>;
}

export interface MappingConfig {
  mapping_id: string;
  dataset_id: string;
  model_id: string;
  items: FeatureMappingItem[];
  target_column_name: string | null;
  target_label_map: Record<string, number> | null;
  is_valid: boolean;
  duplicate_columns: string[];
}

export async function fetchAutoSuggest(
  datasetId: string,
  modelId: string,
): Promise<{ mapping: MappingConfig; report: DryRunReport; supports_nan_input: boolean }> {
  return apiFetch(`/api/v1/mappings/auto-suggest`, {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId }),
  });
}

export async function fetchMapByOrder(
  datasetId: string,
  modelId: string,
): Promise<{ mapping: MappingConfig; report: DryRunReport; supports_nan_input: boolean }> {
  return apiFetch(`/api/v1/mappings/map-by-order`, {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId }),
  });
}

export async function fetchValidatePreview(body: {
  dataset_id: string;
  model_id: string;
  items: FeatureMappingItem[];
  target_column_name: string | null;
}): Promise<DryRunReport & { items: FeatureMappingItem[]; supports_nan_input: boolean }> {
  return apiFetch(`/api/v1/mappings/validate-preview`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function fetchCommitMapping(body: {
  dataset_id: string;
  model_id: string;
  items: FeatureMappingItem[];
  target_column_name: string | null;
  target_label_map?: Record<string, number> | null;
}): Promise<{ mapping: MappingConfig; report: DryRunReport; supports_nan_input: boolean }> {
  return apiFetch(`/api/v1/mappings/validate-and-save`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
