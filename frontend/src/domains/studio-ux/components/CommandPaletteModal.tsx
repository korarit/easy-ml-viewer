import { useEffect, useMemo, useRef, useState } from 'react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useStudioLayoutStore } from '../store/useStudioLayoutStore';
import { useCommandPaletteRegistry } from '../hooks/useCommandPaletteRegistry';
import { matchesCommand, type CommandCategory } from '../types/studioUx.types';

const TABS: Array<'All' | CommandCategory> = ['All', 'Navigation', '3D Axes & Camera', 'Playback', 'Export & Bake'];

/**
 * Phase 14 — universal command palette (`Ctrl+K`).
 * Fuzzy subsequence search; numeric input jumps straight to that frame.
 */
export function CommandPaletteModal() {
  const open = useStudioLayoutStore((s) => s.commandPaletteOpen);
  const setOpen = useStudioLayoutStore((s) => s.setCommandPalette);
  const { commands } = useCommandPaletteRegistry();
  const [query, setQuery] = useState('');
  const [tab, setTab] = useState<(typeof TABS)[number]>('All');
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery('');
      setTab('All');
      setActiveIdx(0);
      const t = window.setTimeout(() => inputRef.current?.focus(), 30);
      return () => window.clearTimeout(t);
    }
    return undefined;
  }, [open ]);

  const numericJump = useMemo(() => {
    const n = Number(query.trim());
    if (!Number.isInteger(n) || n < 0) return null;
    const total = usePlaybackStore.getState().totalFrames;
    if (n >= total) return null;
    return n;
  }, [query]);

  const results = useMemo(() => {
    const scoped = tab === 'All' ? commands : commands.filter((c) => c.category === tab);
    const q = query.trim();
    if (!q) return scoped.slice(0, 30);
    return scoped
      .map((cmd) => ({ cmd, score: matchesCommand(q, cmd) }))
      .filter((r) => r.score >= 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 30)
      .map((r) => r.cmd);
  }, [commands, query, tab]);

  useEffect(() => setActiveIdx(0), [query, tab]);

  if (!open) return null;

  const runIndex = (idx: number) => {
    const cmd = results[idx];
    if (!cmd) return;
    setOpen(false);
    void cmd.run();
  };

  const jumpToNumeric = () => {
    if (numericJump === null) return;
    setOpen(false);
    usePlaybackStore.getState().seekToFrame(numericJump, true);
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-start justify-center pt-[12vh]"
      style={{ background: 'rgba(4,8,14,0.55)' }}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
      role="dialog"
      aria-label="Command palette"
    >
      <div
        className="w-[min(640px,92vw)] overflow-hidden rounded-xl"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
      >
        <div className="flex items-center gap-2 border-b px-3 py-2" style={{ borderColor: 'var(--border-subtle)' }}>
          <span aria-hidden>🔍</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault();
                setActiveIdx((i) => Math.min(i + 1, results.length - 1));
              } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setActiveIdx((i) => Math.max(i - 1, 0));
              } else if (e.key === 'Enter') {
                e.preventDefault();
                if (numericJump !== null && activeIdx === 0) jumpToNumeric();
                else runIndex(activeIdx);
              }
            }}
            placeholder="Search commands, jump to row, change axes… (e.g. 245, top, axis x)"
            className="w-full bg-transparent text-sm outline-none"
            style={{ color: 'var(--text-primary)' }}
            aria-label="Command search"
          />
          <kbd className="font-mono-tabular rounded px-1.5 py-0.5 text-[10px]" style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
            esc
          </kbd>
        </div>
        <div className="flex flex-wrap gap-1 px-3 pt-2">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="font-mono-tabular rounded px-2 py-1 text-[11px]"
              style={{
                background: tab === t ? 'var(--ml-primary)' : 'transparent',
                color: tab === t ? '#fff' : 'var(--text-secondary)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="max-h-[46vh] overflow-y-auto p-1.5" role="listbox" aria-label="Commands">
          {numericJump !== null && (
            <button
              onClick={jumpToNumeric}
              className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm"
              style={{ background: activeIdx === 0 ? 'var(--surface-elevated)' : 'transparent', color: 'var(--text-primary)' }}
              role="option"
              aria-selected={activeIdx === 0}
            >
              <span>🎯 Go to Frame #{numericJump}</span>
              <kbd className="font-mono-tabular text-[10px]" style={{ color: 'var(--text-muted)' }}>Enter</kbd>
            </button>
          )}
          {results.map((cmd, i) => {
            const offset = numericJump !== null ? 1 : 0;
            const active = i + offset === activeIdx;
            return (
              <button
                key={cmd.id}
                onClick={() => runIndex(i)}
                onMouseEnter={() => setActiveIdx(i + offset)}
                className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left"
                style={{ background: active ? 'var(--surface-elevated)' : 'transparent' }}
                role="option"
                aria-selected={active}
              >
                <span>
                  <span className="block text-sm" style={{ color: 'var(--text-primary)' }}>{cmd.title}</span>
                  <span className="block text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    {cmd.category} · {cmd.hint}
                  </span>
                </span>
                <kbd className="font-mono-tabular text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {cmd.category === 'Navigation' ? 'jump' : 'Enter'}
                </kbd>
              </button>
            );
          })}
          {results.length === 0 && numericJump === null && (
            <p className="px-3 py-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
              No matching commands — try a frame number like “245”.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

