import { useState } from 'react';
import { useAliasStore } from '../store/useAliasStore';
import { useStudioLayoutStore } from '../store/useStudioLayoutStore';
import { isValidHexColor, truncateLabel } from '../types/studioUx.types';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';

/**
 * Phase 14 — label & feature aliasing studio modal.
 * Edits apply instantly to local state (HUD/waterfall read the store);
 * `Save to Server` persists the session config via the alias API.
 */
export function AliasSettingsModal() {
  const open = useStudioLayoutStore((s) => s.aliasModalOpen);
  const setOpen = useStudioLayoutStore((s) => s.setAliasModal);
  const sessionId = useAliasStore((s) => s.sessionId);
  const setSessionId = useAliasStore((s) => s.setSessionId);
  const classAliases = useAliasStore((s) => s.classAliases);
  const featureAliases = useAliasStore((s) => s.featureAliases);
  const setClassAlias = useAliasStore((s) => s.setClassAlias);
  const setFeatureAlias = useAliasStore((s) => s.setFeatureAlias);
  const autosuggest = useAliasStore((s) => s.autosuggest);
  const saveToServer = useAliasStore((s) => s.saveToServer);
  const importJson = useAliasStore((s) => s.importJson);
  const exportJson = useAliasStore((s) => s.exportJson);
  const isLoading = useAliasStore((s) => s.isLoading);
  const isSaving = useAliasStore((s) => s.isSaving);
  const error = useAliasStore((s) => s.error);
  const featureNames = useStage3DStore((s) => s.featureNames);
  const [importText, setImportText] = useState('');
  const [colorError, setColorError] = useState<string | null>(null);

  if (!open) return null;

  const classEntries = Object.entries(classAliases);
  const featureEntries = Object.entries(featureAliases);

  const doAutosuggest = () => {
    const names = featureNames.length > 0 ? featureNames : Object.keys(featureAliases);
    const values = classEntries.length > 0 ? classEntries.map(([k]) => k) : ['0', '1'];
    void autosuggest(names, values);
  };

  const doExport = () => {
    const blob = JSON.stringify(exportJson(), null, 2);
    void navigator.clipboard.writeText(blob).catch(() => setImportText(blob));
    setImportText(blob);
  };

  const doImport = () => {
    try {
      importJson(JSON.parse(importText));
    } catch {
      useAliasStore.setState({ error: 'Import JSON is not valid — paste an exported alias JSON blob.' });
    }
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-start justify-center overflow-y-auto pt-[8vh] pb-8"
      style={{ background: 'rgba(4,8,14,0.55)' }}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
      role="dialog"
      aria-label="Alias settings"
    >
      <div
        className="w-[min(720px,94vw)] rounded-xl p-4"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Class & Feature Aliasing Studio
          </h2>
          <button
            onClick={() => setOpen(false)}
            className="rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
            aria-label="Close alias settings"
          >
            ✕
          </button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <label className="flex items-center gap-1.5">
            Session
            <input
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              maxLength={64}
              className="font-mono-tabular rounded px-2 py-1"
              style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
            />
          </label>
          <button
            onClick={doAutosuggest}
            disabled={isLoading}
            className="rounded px-2.5 py-1.5 font-medium"
            style={{ background: 'var(--ml-primary)', color: '#fff', opacity: isLoading ? 0.6 : 1 }}
          >
            {isLoading ? 'Suggesting…' : '✨ Auto-Suggest Aliases'}
          </button>
          <button
            onClick={() => void saveToServer()}
            disabled={isSaving}
            className="rounded px-2.5 py-1.5"
            style={{ border: '1px solid var(--border-strong)', opacity: isSaving ? 0.6 : 1 }}
          >
            {isSaving ? 'Saving…' : 'Save to Server'}
          </button>
          <button onClick={doExport} className="rounded px-2.5 py-1.5" style={{ border: '1px solid var(--border-subtle)' }}>
            Export JSON
          </button>
        </div>
        {error && (
          <p className="mt-2 text-xs" style={{ color: '#f43f5e' }} role="alert">{error}</p>
        )}
        {colorError && (
          <p className="mt-2 text-xs" style={{ color: '#f59e0b' }} role="alert">{colorError}</p>
        )}

        <h3 className="mt-4 text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-muted)' }}>
          Class labels ({classEntries.length})
        </h3>
        {classEntries.length === 0 && (
          <p className="mt-1 text-xs" style={{ color: 'var(--text-muted)' }}>
            No class aliases yet — run Auto-Suggest or add rows from the palette.
          </p>
        )}
        <div className="mt-1.5 space-y-1.5">
          {classEntries.map(([key, alias]) => (
            <div key={key} className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono-tabular w-14" style={{ color: 'var(--text-muted)' }} title={`Raw value: ${key}`}>
                {truncateLabel(key, 12)}
              </span>
              <input
                value={alias.display_label}
                maxLength={64}
                onChange={(e) => setClassAlias(key, { display_label: e.target.value })}
                className="max-w-[180px] flex-1 truncate rounded px-2 py-1"
                style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
                title={alias.display_label}
                aria-label={`Display label for class ${key}`}
              />
              <input
                type="color"
                value={isValidHexColor(alias.hex_color) ? alias.hex_color : '#3B82F6'}
                onChange={(e) => {
                  const v = e.target.value;
                  if (!isValidHexColor(v)) {
                    setColorError(`Invalid color ${v} — kept previous ${alias.hex_color}`);
                    return;
                  }
                  setColorError(null);
                  setClassAlias(key, { hex_color: v.toUpperCase() });
                }}
                aria-label={`Color for class ${key}`}
                className="h-7 w-9 cursor-pointer rounded"
              />
              <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>{alias.hex_color}</span>
            </div>
          ))}
        </div>

        <h3 className="mt-4 text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-muted)' }}>
          Feature display names ({featureEntries.length})
        </h3>
        <div className="mt-1.5 max-h-[30vh] space-y-1.5 overflow-y-auto">
          {featureEntries.map(([key, alias]) => (
            <div key={key} className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono-tabular w-36 truncate" style={{ color: 'var(--text-muted)' }} title={key}>
                {key}
              </span>
              <input
                value={alias.display_name}
                maxLength={64}
                onChange={(e) => setFeatureAlias(key, { display_name: e.target.value })}
                className="max-w-[180px] flex-1 truncate rounded px-2 py-1"
                style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
                title={alias.display_name}
                aria-label={`Display name for ${key}`}
              />
              <input
                value={alias.unit_suffix}
                maxLength={16}
                onChange={(e) => setFeatureAlias(key, { unit_suffix: e.target.value })}
                placeholder="unit"
                className="font-mono-tabular w-16 rounded px-2 py-1"
                style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
                aria-label={`Unit for ${key}`}
              />
            </div>
          ))}
          {featureEntries.length === 0 && (
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              No feature aliases yet — run Auto-Suggest to prettify column names.
            </p>
          )}
        </div>

        <h3 className="mt-4 text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-muted)' }}>
          Import alias JSON
        </h3>
        <textarea
          value={importText}
          onChange={(e) => setImportText(e.target.value)}
          rows={3}
          placeholder='Paste {"session_id":…, "class_aliases":…, "feature_aliases":…}'
          className="font-mono-tabular mt-1.5 w-full rounded p-2 text-[11px]"
          style={{ border: '1px solid var(--border-subtle)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
        />
        <div className="mt-2 flex justify-end gap-2">
          <button onClick={doImport} className="rounded px-3 py-1.5 text-xs" style={{ border: '1px solid var(--border-strong)' }}>
            Import
          </button>
          <button onClick={() => setOpen(false)} className="rounded px-3 py-1.5 text-xs font-medium" style={{ background: 'var(--ml-primary)', color: '#fff' }}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

