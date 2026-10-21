/** Phase 14 — modern studio UX shared types (command palette + aliasing + cinema). */

export interface ClassAlias {
  raw_class_value: string | number;
  display_label: string;
  hex_color: string;
}

export interface FeatureAlias {
  raw_feature_name: string;
  display_name: string;
  unit_suffix: string;
}

export interface AliasConfig {
  session_id: string;
  class_aliases: Record<string, ClassAlias>;
  feature_aliases: Record<string, FeatureAlias>;
}

export type CommandCategory =
  | 'Navigation'
  | '3D Axes & Camera'
  | 'Playback'
  | 'Export & Bake';

export interface StudioCommand {
  id: string;
  title: string;
  hint: string;
  keywords: string;
  category: CommandCategory;
  run: () => void | Promise<void>;
}

/** Case-insensitive subsequence fuzzy match with a simple contiguity score. */
export function fuzzyScore(query: string, target: string): number {
  const q = query.trim().toLowerCase();
  const t = target.toLowerCase();
  if (!q) return 1;
  if (t.includes(q)) return 100 + q.length;
  let qi = 0;
  let score = 0;
  let lastHit = -2;
  for (let i = 0; i < t.length && qi < q.length; i += 1) {
    if (t[i] === q[qi]) {
      score += i === lastHit + 1 ? 3 : 1;
      lastHit = i;
      qi += 1;
    }
  }
  return qi === q.length ? score : -1;
}

export function matchesCommand(query: string, cmd: StudioCommand): number {
  const haystacks = [cmd.title, cmd.hint, cmd.keywords, cmd.category];
  let best = -1;
  for (const h of haystacks) {
    const s = fuzzyScore(query, h);
    if (s > best) best = s;
    if (query.trim() && h.toLowerCase().includes(query.trim().toLowerCase())) {
      best = Math.max(best, 100);
    }
  }
  // Numeric queries also match "go to frame" by construction in the registry.
  return best;
}

/** Truncate long alias labels for HUD chips (spec edge #5). */
export function truncateLabel(label: string, max = 24): string {
  if (label.length <= max) return label;
  return `${label.slice(0, max - 1)}…`;
}

export function isValidHexColor(value: string): boolean {
  return /^#[0-9A-Fa-f]{6}$/.test(value.trim());
}

