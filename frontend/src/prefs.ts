// Per-browser UI preferences. Storage can be unavailable (private mode, blocked site data), so every access is
// guarded and the app works with the defaults.

export type Theme = "light" | "dark" | "system";
export type RenderQuality = "high" | "balanced" | "low";

export interface Prefs {
  theme: Theme;
  renderQuality: RenderQuality;
  showLayerLabels: boolean;
}

const KEY = "model-surgery:prefs";
export const DEFAULT_PREFS: Prefs = { theme: "light", renderQuality: "balanced", showLayerLabels: true };

export function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_PREFS };
    const p = JSON.parse(raw) as Partial<Prefs>;
    return {
      theme: p.theme === "dark" || p.theme === "system" ? p.theme : "light",
      renderQuality: p.renderQuality === "high" || p.renderQuality === "low" ? p.renderQuality : "balanced",
      showLayerLabels: p.showLayerLabels !== false,
    };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

export function savePrefs(p: Prefs): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* not persisted; the choice still applies to this session */
  }
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

/** Resolved light/dark, for code that cannot use CSS variables (WebGL). */
export function isDark(): boolean {
  const t = document.documentElement.dataset.theme;
  return t === "dark" || (t === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
}

/** Reads a CSS custom property as a hex/rgb string (for three.js colours). */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
