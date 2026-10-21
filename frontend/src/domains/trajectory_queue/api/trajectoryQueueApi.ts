/** Phase 06 — typed client for dataset query + resolve-selection endpoints. */
import { apiFetch } from '../../../core/api/client';

export type QueryOperator = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'between';

export interface FilterRule {
  id: string;
  column: string;
  operator: QueryOperator;
  value: number | string | boolean | null;
  value_to?: number | null;
}

export interface QueryRequestBody {
  filters: Array<{ column: string; operator: QueryOperator; value?: number | string | boolean | null; value_to?: number | null }>;
  sort_by?: string | null;
  sort_ascending?: boolean;
  offset?: number;
  limit?: number;
  columns?: string[] | null;
}

export interface QueryResponse {
  total_dataset_rows: number;
  total_filtered_rows: number;
  offset: number;
  limit: number;
  rows: Array<Record<string, unknown> & { _original_row_index: number }>;
  matching_row_indices_preview: number[];
}

export type SelectionMode = 'first_n' | 'random_n' | 'range' | 'filtered_all';

export interface ResolveSelectionResponse {
  row_indices: number[];
  total_matching: number;
  truncated: boolean;
  mode: SelectionMode;
}

function toBackendRule(r: FilterRule) {
  return {
    column: r.column,
    operator: r.operator,
    value: r.value,
    value_to: r.value_to ?? null,
  };
}

export async function fetchDatasetQuery(datasetId: string, body: {
  filters: FilterRule[];
  sortBy: string | null;
  sortAscending: boolean;
  offset: number;
  limit: number;
  columns?: string[] | null;
}, signal?: AbortSignal): Promise<QueryResponse> {
  const payload: QueryRequestBody = {
    filters: body.filters.map(toBackendRule),
    sort_by: body.sortBy,
    sort_ascending: body.sortAscending,
    offset: body.offset,
    limit: body.limit,
    columns: body.columns ?? null,
  };
  return apiFetch<QueryResponse>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/query`, {
    method: 'POST',
    body: JSON.stringify(payload),
    signal,
  });
}

export async function fetchResolveSelection(datasetId: string, body: {
  mode: SelectionMode;
  filters: FilterRule[];
  sortBy: string | null;
  sortAscending: boolean;
  n?: number;
  seed?: number;
  rangeStart?: number;
  rangeEnd?: number | null;
  rangeStep?: number;
}): Promise<ResolveSelectionResponse> {
  return apiFetch<ResolveSelectionResponse>(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/resolve-selection`,
    {
      method: 'POST',
      body: JSON.stringify({
        mode: body.mode,
        filters: body.filters.map(toBackendRule),
        sort_by: body.sortBy,
        sort_ascending: body.sortAscending,
        n: body.n ?? 100,
        seed: body.seed ?? 42,
        range_start: body.rangeStart ?? 0,
        range_end: body.rangeEnd ?? null,
        range_step: body.rangeStep ?? 1,
      }),
    },
  );
}
