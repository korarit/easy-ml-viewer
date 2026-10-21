import { create } from 'zustand';
import { API_BASE_URL } from '../../../core/api/client';

export type ColumnLogicalType = 'numeric' | 'categorical' | 'boolean' | 'datetime';

export interface ColumnProfile {
  index: number;
  name: string;
  logical_type: ColumnLogicalType;
  storage_dtype: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  min_value: number | null;
  max_value: number | null;
  mean_value: number | null;
  std_value: number | null;
  median_value: number | null;
  histogram_bins: number[];
  histogram_counts: number[];
  sample_values: Array<string | number | boolean | null>;
}

export interface DatasetProfileResponse {
  dataset_id: string;
  filename: string;
  file_format?: string;
  detected_encoding?: string;
  detected_delimiter?: string | null;
  row_count: number;
  column_count: number;
  original_ram_mb: number;
  optimized_ram_mb: number;
  inf_sanitized_count: number;
  available_sheets: string[];
  selected_sheet: string | null;
  columns: ColumnProfile[];
  sample_rows: Array<Record<string, unknown>>;
}

export type TypeFilter = 'all' | ColumnLogicalType;

interface DatasetStoreState {
  currentDataset: DatasetProfileResponse | null;
  lastFile: File | null;
  isUploading: boolean;
  uploadProgress: string | null;
  uploadError: string | null;
  searchQuery: string;
  typeFilter: TypeFilter;
  uploadDataset: (file: File, sheetName?: string) => Promise<void>;
  loadSampleDataset: (sampleName: string) => Promise<void>;
  setSearchQuery: (q: string) => void;
  setTypeFilter: (t: TypeFilter) => void;
  clearError: () => void;
}

async function throwIfError(res: Response): Promise<void> {
  if (res.ok) return;
  let message = `Upload failed (${res.status})`;
  try {
    const data = (await res.json()) as { error?: { code?: string; message?: string } };
    if (data?.error?.message) message = `${data.error.code ?? 'ERROR'}: ${data.error.message}`;
  } catch {
    /* keep default */
  }
  throw new Error(message);
}

export const ACCEPTED_EXTENSIONS = '.csv,.tsv,.xlsx,.xls,.parquet';

export function isAcceptedFile(name: string): boolean {
  const lower = name.toLowerCase();
  return ['.csv', '.tsv', '.xlsx', '.xls', '.parquet'].some((ext) => lower.endsWith(ext));
}

export const useDatasetStore = create<DatasetStoreState>((set) => ({
  currentDataset: null,
  lastFile: null,
  isUploading: false,
  uploadProgress: null,
  uploadError: null,
  searchQuery: '',
  typeFilter: 'all',

  uploadDataset: async (file, sheetName) => {
    set({ isUploading: true, uploadError: null, uploadProgress: 'Uploading stream…', lastFile: file });
    try {
      const form = new FormData();
      form.append('file', file);
      if (sheetName) form.append('sheet_name', sheetName);
      set({ uploadProgress: 'Detecting encoding → downcasting to float32…' });
      const res = await fetch(`${API_BASE_URL}/api/v1/datasets/upload`, { method: 'POST', body: form });
      await throwIfError(res);
      set({ uploadProgress: 'Computing column profiles…' });
      const data = (await res.json()) as DatasetProfileResponse;
      set({ currentDataset: data, searchQuery: '', typeFilter: 'all' });
    } catch (err) {
      set({ uploadError: err instanceof Error ? err.message : String(err) });
      throw err;
    } finally {
      set({ isUploading: false, uploadProgress: null });
    }
  },

  loadSampleDataset: async (sampleName) => {
    set({ isUploading: true, uploadError: null, uploadProgress: `Loading sample “${sampleName}”…` });
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/datasets/load-sample/${encodeURIComponent(sampleName)}`, {
        method: 'POST',
      });
      await throwIfError(res);
      const data = (await res.json()) as DatasetProfileResponse;
      set({ currentDataset: data, searchQuery: '', typeFilter: 'all' });
    } catch (err) {
      set({ uploadError: err instanceof Error ? err.message : String(err) });
      throw err;
    } finally {
      set({ isUploading: false, uploadProgress: null });
    }
  },

  setSearchQuery: (q) => set({ searchQuery: q }),
  setTypeFilter: (t) => set({ typeFilter: t }),
  clearError: () => set({ uploadError: null }),
}));
