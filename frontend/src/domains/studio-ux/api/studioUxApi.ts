/** Phase 14 — typed client for the studio alias endpoints. */
import { apiFetch } from '../../../core/api/client';
import type { AliasConfig } from '../types/studioUx.types';

export async function fetchAliasConfig(sessionId: string, signal?: AbortSignal): Promise<AliasConfig> {
  return apiFetch<AliasConfig>(`/api/v1/alias/${encodeURIComponent(sessionId)}`, {
    method: 'GET',
    signal,
  });
}

export async function upsertAliasConfig(config: AliasConfig, signal?: AbortSignal): Promise<AliasConfig> {
  return apiFetch<AliasConfig>(`/api/v1/alias/${encodeURIComponent(config.session_id)}`, {
    method: 'PUT',
    body: JSON.stringify({
      session_id: config.session_id,
      class_aliases: config.class_aliases,
      feature_aliases: config.feature_aliases,
    }),
    signal,
  });
}

export async function fetchAutosuggestAliases(
  featureNames: string[],
  classValues: Array<string | number>,
  signal?: AbortSignal,
): Promise<AliasConfig> {
  return apiFetch<AliasConfig>('/api/v1/alias/autosuggest/preview', {
    method: 'POST',
    body: JSON.stringify({ feature_names: featureNames, class_values: classValues }),
    signal,
  });
}

