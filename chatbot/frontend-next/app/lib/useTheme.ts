"use client";

import { useCallback, useEffect, useState } from "react";
import { THEME_PREF_KEY, type Theme } from "./theme";

export type { Theme };

/**
 * Light / dark preference, applied as `data-theme` on <html> so every design
 * token swaps at once. The layout's inline bootstrap applies a stored "dark"
 * before paint; this hook only reads that back and handles later changes.
 */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>("light");

  useEffect(() => {
    setThemeState(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  }, []);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    if (next === "dark") document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    try {
      window.localStorage.setItem(THEME_PREF_KEY, next);
    } catch {
      // Blocked storage: the choice still applies for this session.
    }
  }, []);

  return { theme, setTheme };
}
