import { create } from 'zustand';
import {
  fetchDatasetQuery,
  fetchResolveSelection,
  type FilterRule,
  type QueryOperator,
} from '../api/trajectoryQueueApi';

export type { FilterRule, QueryOperator };
export type QueuePreset = 'first_100' | 'random_100' | 'filtered_all' | 'clear';

export type QueueRow = Record<string, unknown> & { _original_row_index: number };

interface TrajectoryQueueState {
  datasetId: string | null;
  mappedColumns: string[];
  targetColumn: string | null;
  showUnmappedColumns: boolean;
  selectedRowIndices: number[];
  selectedLookupSet: Set<number>;
  lastClickedRowIndex: number | null;
  filters: FilterRule[];
  sortBy: string | null;
  sortAscending: boolean;
  totalDatasetRows: number;
  totalFilteredRows: number;
  loadedRows: QueueRow[];
  loadedOffset: number;
  windowLimit: number;
  isLoading: boolean;
  isResolving: boolean;
  queryError: string | null;
  truncatedNotice: string | null;
  initialize: (datasetId: string, mappedColumns: string[], targetColumn: string | null) => Promise<void>;
  fetchWindow: (offset: number) => Promise<void>;
  loadMore: () => Promise<void>;
  toggleRowSelection: (origRowIndex: number, shiftKey?: boolean, visibleIndices?: number[]) => void;
  removeFromQueue: (origRowIndex: number) => void;
  selectPreset: (preset: QueuePreset) => Promise<void>;
  selectRange: (startRow: number, endRow: number, step: number) => void;
  reorderQueue: (fromIdx: number, toIdx: number) => void;
  setFilters: (filters: FilterRule[]) => void;
  setSort: (sortBy: string | null, sortAscending: boolean) => void;
  setShowUnmappedColumns: (show: boolean) => void;
  clearError: () => void;
}

let filterIdCounter = 0;
export function createFilterRule(column = '', operator: QueryOperator = 'gte', value: number | string = 0): FilterRule {
  filterIdCounter += 1;
  return { id: `rule-${Date.now()}-${filterIdCounter}`, column, operator, value };
}

/** Frontend LRU page cache: offset -> rows (max 10 pages x 200 rows). */
const pageCache = new Map<number, QueueRow[]>();
const MAX_CACHED_PAGES = 10;
let inFlightAbort: AbortController | null = null;

function cacheWindow(offset: number, rows: QueueRow[]) {
  pageCache.set(offset, rows);
  while (pageCache.size > MAX_CACHED_PAGES) {
    const oldest = pageCache.keys().next().value as number | undefined;
    if (oldest === undefined) break;
    pageCache.delete(oldest);
  }
}

function syncSelection(rows: number[]) {
  return new Set<number>(rows);
}

export const useTrajectoryQueueStore = create<TrajectoryQueueState>((set, get) => ({
  datasetId: null,
  mappedColumns: [],
  targetColumn: null,
  showUnmappedColumns: false,
  selectedRowIndices: [],
  selectedLookupSet: new Set<number>(),
  lastClickedRowIndex: null,
  filters: [],
  sortBy: null,
  sortAscending: true,
  totalDatasetRows: 0,
  totalFilteredRows: 0,
  loadedRows: [],
  loadedOffset: 0,
  windowLimit: 200,
  isLoading: false,
  isResolving: false,
  queryError: null,
  truncatedNotice: null,

  initialize: async (datasetId, mappedColumns, targetColumn) => {
    const prev = get();
    if (prev.datasetId === datasetId && prev.mappedColumns.join('|') === mappedColumns.join('|')) return;
    pageCache.clear();
    if (inFlightAbort) inFlightAbort.abort();
    set({
      datasetId,
      mappedColumns,
      targetColumn,
      filters: [],
      sortBy: null,
      sortAscending: true,
      selectedRowIndices: [],
      selectedLookupSet: new Set<number>(),
      lastClickedRowIndex: null,
      loadedRows: [],
      loadedOffset: 0,
      queryError: null,
      truncatedNotice: null,
      totalDatasetRows: 0,
      totalFilteredRows: 0,
    });
    // Default queue: First 100 rows so users can jump into the 3D studio in 1 click.
    await get().selectPreset('first_100');
    await get().fetchWindow(0);
  },

  fetchWindow: async (offset) => {
    const s = get();
    if (!s.datasetId) return;
    const cached = pageCache.get(offset);
    if (cached && offset !== 0) {
      set({ loadedRows: cached, loadedOffset: offset, queryError: null });
      return;
    }
    if (inFlightAbort) inFlightAbort.abort();
    const controller = new AbortController();
    inFlightAbort = controller;
    set({ isLoading: true, queryError: null });
    try {
      const res = await fetchDatasetQuery(
        s.datasetId,
        {
          filters: s.filters,
          sortBy: s.sortBy,
          sortAscending: s.sortAscending,
          offset,
          limit: s.windowLimit,
          columns: null, // full window: table needs every mapped column anyway
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      cacheWindow(offset, res.rows as QueueRow[]);
      set({
        loadedRows: res.rows as QueueRow[],
        loadedOffset: res.offset,
        totalDatasetRows: res.total_dataset_rows,
        totalFilteredRows: res.total_filtered_rows,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      set({ queryError: err instanceof Error ? err.message : String(err) });
    } finally {
      if (inFlightAbort === controller) inFlightAbort = null;
      set({ isLoading: false });
    }
  },

  loadMore: async () => {
    const s = get();
    const next = s.loadedOffset + s.windowLimit;
    if (next >= s.totalFilteredRows) return;
    if (inFlightAbort) inFlightAbort.abort();
    const controller = new AbortController();
    inFlightAbort = controller;
    set({ isLoading: true, queryError: null });
    try {
      const res = await fetchDatasetQuery(
        s.datasetId as string,
        {
          filters: s.filters,
          sortBy: s.sortBy,
          sortAscending: s.sortAscending,
          offset: next,
          limit: s.windowLimit,
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      const merged = [...s.loadedRows, ...(res.rows as QueueRow[])];
      cacheWindow(next, res.rows as QueueRow[]);
      set({
        loadedRows: merged,
        loadedOffset: s.loadedOffset, // keep anchor; merged list grows downward
        totalDatasetRows: res.total_dataset_rows,
        totalFilteredRows: res.total_filtered_rows,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      set({ queryError: err instanceof Error ? err.message : String(err) });
    } finally {
      if (inFlightAbort === controller) inFlightAbort = null;
      set({ isLoading: false });
    }
  },

  toggleRowSelection: (origRowIndex, shiftKey = false, visibleIndices) => {
    const s = get();
    if (shiftKey && s.lastClickedRowIndex !== null) {
      if (visibleIndices && visibleIndices.length > 0) {
        // Edge case 4: only pick filtered-visible rows between anchor and target.
        const anchorPos = visibleIndices.indexOf(s.lastClickedRowIndex);
        const targetPos = visibleIndices.indexOf(origRowIndex);
        if (anchorPos !== -1 && targetPos !== -1) {
          const [lo, hi] = anchorPos < targetPos ? [anchorPos, targetPos] : [targetPos, anchorPos];
          const span = visibleIndices.slice(lo, hi + 1);
          const merged = [...s.selectedRowIndices];
          const lookup = new Set(s.selectedLookupSet);
          for (const idx of span) {
            if (!lookup.has(idx)) {
              lookup.add(idx);
              merged.push(idx);
            }
          }
          set({ selectedRowIndices: merged, selectedLookupSet: lookup, lastClickedRowIndex: origRowIndex });
          return;
        }
      }
      // Fallback: plain numeric range (no active filter context).
      const anchor = s.lastClickedRowIndex;
      const [lo, hi] = anchor < origRowIndex ? [anchor, origRowIndex] : [origRowIndex, anchor];
      const merged = [...s.selectedRowIndices];
      const lookup = new Set(s.selectedLookupSet);
      for (let i = lo; i <= hi; i += 1) {
        if (!lookup.has(i)) {
          lookup.add(i);
          merged.push(i);
        }
      }
      set({ selectedRowIndices: merged, selectedLookupSet: lookup, lastClickedRowIndex: origRowIndex });
      return;
    }
    const lookup = new Set(s.selectedLookupSet);
    let next: number[];
    if (lookup.has(origRowIndex)) {
      lookup.delete(origRowIndex);
      next = s.selectedRowIndices.filter((i) => i !== origRowIndex);
    } else {
      lookup.add(origRowIndex);
      next = [...s.selectedRowIndices, origRowIndex];
    }
    set({ selectedRowIndices: next, selectedLookupSet: lookup, lastClickedRowIndex: origRowIndex });
  },

  removeFromQueue: (origRowIndex) => {
    const s = get();
    if (!s.selectedLookupSet.has(origRowIndex)) return;
    const lookup = new Set(s.selectedLookupSet);
    lookup.delete(origRowIndex);
    set({
      selectedRowIndices: s.selectedRowIndices.filter((i) => i !== origRowIndex),
      selectedLookupSet: lookup,
    });
  },

  selectPreset: async (preset) => {
    const s = get();
    if (preset === 'clear') {
      set({ selectedRowIndices: [], selectedLookupSet: new Set<number>(), lastClickedRowIndex: null, truncatedNotice: null });
      return;
    }
    if (!s.datasetId) return;
    set({ isResolving: true, queryError: null, truncatedNotice: null });
    try {
      const mode = preset === 'first_100' ? 'first_n' : preset === 'random_100' ? 'random_n' : 'filtered_all';
      const res = await fetchResolveSelection(s.datasetId, {
        mode,
        filters: s.filters,
        sortBy: s.sortBy,
        sortAscending: s.sortAscending,
        n: preset === 'filtered_all' ? 10000 : 100,
        seed: 42,
      });
      set({
        selectedRowIndices: res.row_indices,
        selectedLookupSet: syncSelection(res.row_indices),
        lastClickedRowIndex: res.row_indices.length > 0 ? res.row_indices[res.row_indices.length - 1] : null,
        truncatedNotice: res.truncated
          ? `Capped at ${res.row_indices.length.toLocaleString()} frames (${res.total_matching.toLocaleString()} matching) — stride-subsampled.`
          : null,
      });
    } catch (err) {
      set({ queryError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isResolving: false });
    }
  },

  selectRange: (startRow, endRow, step) => {
    const stepSafe = Math.max(1, Math.floor(step));
    const out: number[] = [];
    if (startRow <= endRow) {
      for (let i = startRow; i <= endRow; i += stepSafe) out.push(i);
    } else {
      for (let i = startRow; i >= endRow; i -= stepSafe) out.push(i);
    }
    set({
      selectedRowIndices: out,
      selectedLookupSet: syncSelection(out),
      lastClickedRowIndex: out.length > 0 ? out[out.length - 1] : null,
      truncatedNotice: out.length >= 10000 ? 'Capped at 10,000 animated frames.' : null,
    });
  },

  reorderQueue: (fromIdx, toIdx) => {
    const s = get();
    if (fromIdx < 0 || toIdx < 0 || fromIdx >= s.selectedRowIndices.length || toIdx >= s.selectedRowIndices.length) return;
    const next = [...s.selectedRowIndices];
    const [moved] = next.splice(fromIdx, 1);
    next.splice(toIdx, 0, moved);
    set({ selectedRowIndices: next, selectedLookupSet: syncSelection(next) });
  },

  setFilters: (filters) => {
    pageCache.clear();
    set({ filters, loadedRows: [], loadedOffset: 0 });
    void get().fetchWindow(0);
  },

  setSort: (sortBy, sortAscending) => {
    pageCache.clear();
    set({ sortBy, sortAscending, loadedRows: [], loadedOffset: 0 });
    void get().fetchWindow(0);
  },

  setShowUnmappedColumns: (show) => set({ showUnmappedColumns: show }),

  clearError: () => set({ queryError: null }),
}));
