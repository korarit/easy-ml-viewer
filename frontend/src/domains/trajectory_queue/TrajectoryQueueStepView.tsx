import { useEffect, useMemo } from 'react';
import { useDatasetStore } from '../dataset/store/useDatasetStore';
import { useMappingStore } from '../mapping/store/useMappingStore';
import { useSystemStore } from '../../core/store/useSystemStore';
import { DataSelectorToolbar } from './components/DataSelectorToolbar';
import { FilterConditionBar } from './components/FilterConditionBar';
import { TrajectorySortSelector } from './components/TrajectorySortSelector';
import { VirtualizedDatasetTable } from './components/VirtualizedDatasetTable';
import { TrajectoryQueueDock } from './components/TrajectoryQueueDock';
import { EvaluationLensPanel } from '../evaluation/EvaluationLensPanel';
import { useTrajectoryQueueStore } from './store/useTrajectoryQueueStore';

/** Step 4 studio: virtualized selector + multi-select + trajectory queue (Phase 06). */
export function TrajectoryQueueStepView() {
  const { currentDataset } = useDatasetStore();
  const { items: mappingItems, targetColumnName } = useMappingStore();
  const { setActiveStep, unlockStep } = useSystemStore();
  const q = useTrajectoryQueueStore();

  const mappedColumns = useMemo(
    () => mappingItems.map((m) => m.dataset_column_name).filter((c): c is string => !!c),
    [mappingItems],
  );
  const allColumns = useMemo(
    () => currentDataset?.columns.map((c) => c.name) ?? [],
    [currentDataset],
  );
  const visibleColumns = useMemo(() => {
    if (q.showUnmappedColumns || mappedColumns.length === 0) return allColumns;
    const target = targetColumnName ? [targetColumnName] : [];
    return [...mappedColumns, ...target.filter((t) => !mappedColumns.includes(t))];
  }, [q.showUnmappedColumns, mappedColumns, targetColumnName, allColumns]);

  const columnBadges = useMemo(() => {
    const badges: Record<string, string> = {};
    mappingItems.forEach((m) => {
      if (m.dataset_column_name) badges[m.dataset_column_name] = `#${m.feature_index}`;
    });
    if (targetColumnName) badges[targetColumnName] = 'TGT';
    return badges;
  }, [mappingItems, targetColumnName]);

  const frameIndexByRow = useMemo(() => {
    const map = new Map<number, number>();
    q.selectedRowIndices.forEach((rowIdx, frameIdx) => {
      if (!map.has(rowIdx)) map.set(rowIdx, frameIdx);
    });
    return map;
  }, [q.selectedRowIndices]);

  // Initial default queue + first window when entering Step 4.
  const datasetId = currentDataset?.dataset_id ?? null;
  useEffect(() => {
    if (datasetId) void q.initialize(datasetId, mappedColumns, targetColumnName);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  // Keyboard shortcuts (§2.4): Esc clear, Ctrl+A select filtered, Del removes anchor.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA');
      if (e.key === 'Escape' && !typing) {
        void q.selectPreset('clear');
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'a' || e.key === 'A') && !typing) {
        e.preventDefault();
        void q.selectPreset('filtered_all');
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && !typing) {
        const anchor = useTrajectoryQueueStore.getState().lastClickedRowIndex;
        if (anchor !== null) useTrajectoryQueueStore.getState().removeFromQueue(anchor);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!currentDataset) {
    return (
      <div className="mx-auto w-full max-w-5xl p-4">
        <div
          className="rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="alert"
        >
          Load a dataset (Step 1) first — the queue needs rows to select from.
        </div>
      </div>
    );
  }

  const visibleIndices = q.loadedRows.map((r) => r._original_row_index);
  const canLoadMore = q.loadedRows.length < q.totalFilteredRows;

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-3 overflow-auto p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-sm font-semibold tracking-wide">STEP 4: DATA SELECTOR & TRAJECTORY QUEUE</h1>
        <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Total rows: {q.totalDatasetRows.toLocaleString()} · Filtered:{' '}
          {q.totalFilteredRows.toLocaleString()}
        </span>
      </div>

      <DataSelectorToolbar />

      <div className="flex flex-col gap-3 lg:flex-row">
        <div className="flex-1">
          <FilterConditionBar availableColumns={allColumns} />
        </div>
        <TrajectorySortSelector availableColumns={allColumns} />
      </div>

      {q.queryError && (
        <div
          className="flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs"
          style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
          role="alert"
        >
          <span>{q.queryError}</span>
          <button onClick={q.clearError} className="underline">Dismiss</button>
        </div>
      )}

      {q.totalFilteredRows === 0 && !q.isLoading ? (
        <div
          className="rounded-md px-3 py-6 text-center text-xs"
          style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}
        >
          <p>No rows match the current filters.</p>
          <button onClick={() => q.setFilters([])} className="mt-2 underline" style={{ color: 'var(--ml-primary)' }}>
            Reset filters
          </button>
        </div>
      ) : (
        <>
          <VirtualizedDatasetTable
            rows={q.loadedRows}
            columns={visibleColumns}
            columnBadges={columnBadges}
            selectedLookupSet={q.selectedLookupSet}
            frameIndexByRow={frameIndexByRow}
            onToggleRow={(orig, shift) => q.toggleRowSelection(orig, shift, visibleIndices)}
            isLoading={q.isLoading}
          />
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
              <input
                type="checkbox"
                checked={q.showUnmappedColumns}
                onChange={(e) => q.setShowUnmappedColumns(e.target.checked)}
              />
              Show unmapped columns
            </label>
            {canLoadMore && (
              <button
                onClick={() => void q.loadMore()}
                disabled={q.isLoading}
                className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
                style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
              >
                {q.isLoading ? 'Loading…' : `Load next 200 (${q.loadedRows.length}/${q.totalFilteredRows})`}
              </button>
            )}
          </div>
        </>
      )}

      <TrajectoryQueueDock
        selectedRowIndices={q.selectedRowIndices}
        truncatedNotice={q.truncatedNotice}
        isLaunchDisabled={q.selectedRowIndices.length === 0}
        onRemove={q.removeFromQueue}
        onMove={q.reorderQueue}
        onClear={() => void q.selectPreset('clear')}
        onBack={() => setActiveStep(3)}
        onLaunch={() => unlockStep(5)}
      />

      <EvaluationLensPanel />
    </div>
  );
}
