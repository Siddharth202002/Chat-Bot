"use client";

import { useEffect, useState } from "react";

/**
 * SSR-safe media query hook. Returns `false` on the server and during the first
 * client render, then settles to the real value in an effect — so markup always
 * matches between server and client and never triggers a hydration mismatch.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);

    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/** Matches the Tailwind `md` breakpoint. */
export const useIsDesktop = () => useMediaQuery("(min-width: 768px)");
