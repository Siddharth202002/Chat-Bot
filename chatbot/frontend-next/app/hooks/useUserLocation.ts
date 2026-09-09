"use client";

import {
  deleteStoredLocation,
  fetchStoredLocation,
  postCoordinates,
  postLocationFailure,
  postManualCity,
  requestBrowserPosition,
  type GeolocationFailure,
  type LocationStatus,
  type UserLocation,
} from "@/app/lib/location";
import { useCallback, useEffect, useRef, useState } from "react";

/** User-facing copy for each way the browser can refuse to locate us. */
const FAILURE_COPY: Record<GeolocationFailure, string> = {
  denied: "Location permission is required for this request.",
  timeout: "Your browser took too long to find you.",
  unavailable: "Your device could not work out where it is.",
  unsupported: "This browser does not support location.",
  // Deliberately NOT phrased as a permission problem. There is no browser
  // setting that fixes this, so sending the user to look for one wastes their
  // time -- the site itself has to be served over HTTPS.
  insecure:
    "Location needs a secure (HTTPS) connection, and this page is served over HTTP.",
};

const FAILURE_STATUS: Record<GeolocationFailure, LocationStatus> = {
  denied: "denied",
  timeout: "error",
  unavailable: "error",
  unsupported: "unsupported",
  insecure: "insecure",
};

/** Narrow a rejection message back to a `GeolocationFailure`. */
function asFailure(message: string): GeolocationFailure {
  switch (message) {
    case "denied":
    case "timeout":
    case "unsupported":
    case "unavailable":
    case "insecure":
      return message;
    default:
      return "unavailable";
  }
}

interface UseUserLocationOptions {
  /** Gate rehydration on there being a session to rehydrate for. */
  enabled: boolean;
  /** Called when the API reports 401 so the page can tear down the session. */
  onUnauthorized: () => void;
  onToast: (variant: "success" | "error" | "info", message: string) => void;
}

/**
 * Owns the user's location for the session: rehydrates what the backend already
 * stored, prompts for GPS at most once, and exposes a manual city fallback.
 *
 * The one rule that shapes everything here: `navigator.geolocation` must be
 * touched as rarely as possible. Every browser shows its own permission UI, so
 * a second prompt reads as a bug. Hence the `ready` short circuit, the refusal
 * to re-prompt after a denial, and the in-flight promise guard.
 */
export function useUserLocation({
  enabled,
  onUnauthorized,
  onToast,
}: UseUserLocationOptions) {
  const [status, setStatus] = useState<LocationStatus>("idle");
  const [location, setLocation] = useState<UserLocation | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);

  /** False after unmount so a late-resolving fetch never calls setState. */
  const mountedRef = useRef(true);
  /**
   * Mirrors of the two pieces of state `ensureLocation` reads. Keeping them in
   * refs means `ensureLocation` has a stable identity across status changes —
   * which matters because `sendMessage` in page.tsx depends on it.
   */
  const statusRef = useRef<LocationStatus>("idle");
  const locationRef = useRef<UserLocation | null>(null);
  /** The in-flight ensureLocation(), so two rapid sends share ONE prompt. */
  const inFlightRef = useRef<Promise<UserLocation | null> | null>(null);
  /**
   * True once we have raised the browser's permission UI in this session and it
   * did not give us coordinates. Budgets us to exactly one automatic attempt.
   */
  const attemptedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const setPhase = useCallback((next: LocationStatus, nextMessage: string | null = null) => {
    statusRef.current = next;
    if (!mountedRef.current) return;
    setStatus(next);
    setMessage(nextMessage);
  }, []);

  const storeLocation = useCallback((next: UserLocation | null) => {
    locationRef.current = next;
    if (!mountedRef.current) return;
    setLocation(next);
  }, []);

  /* ── Rehydrate ────────────────────────────────────────────────── */

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    (async () => {
      try {
        const stored = await fetchStoredLocation();
        if (cancelled || !mountedRef.current) return;

        if (stored.status === "ready" && stored.location) {
          storeLocation(stored.location);
          attemptedRef.current = false;
          setPhase("ready");
          return;
        }
        // Nothing usable is stored, so drop anything we were still holding —
        // otherwise a second sign-in would inherit the previous user's city.
        storeLocation(null);

        if (stored.status === "denied") {
          setPhase("denied", FAILURE_COPY.denied);
          return;
        }
        if (stored.status === "unsupported") {
          setPhase("unsupported", FAILURE_COPY.unsupported);
          return;
        }
        // "none", plus the retryable "timeout"/"unavailable": stay idle and let
        // the next location-dependent request prompt normally.
        // This never touches navigator.geolocation — mount must not prompt.
        setPhase("idle");
      } catch (err) {
        if (cancelled || !mountedRef.current) return;
        if ((err as Error).message === "unauthorized") onUnauthorized();
        // Any other rehydrate failure is silent: idle is the correct fallback
        // and a toast on page load would be noise.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [enabled, onUnauthorized, setPhase, storeLocation]);

  /* ── Ensure ───────────────────────────────────────────────────── */

  /**
   * Resolve a location for a request that needs one. Returns `null` rather
   * than throwing — the caller sends the message either way.
   */
  const ensureLocation = useCallback(async (): Promise<UserLocation | null> => {
    // Already resolved this session — hand it straight back. This short circuit
    // is what stops the browser prompting again on every weather question.
    if (statusRef.current === "ready" && locationRef.current) return locationRef.current;

    // The user said no, or cannot say yes. Re-prompting is futile and rude;
    // LocationStatus offers the manual city input instead.
    if (
      statusRef.current === "denied" ||
      statusRef.current === "unsupported" ||
      // No amount of asking makes an http:// origin secure.
      statusRef.current === "insecure"
    ) {
      return null;
    }

    // A timeout is NOT proof the user will eventually say yes. Chrome reports an
    // *ignored* permission bubble as TIMEOUT (code 3), never PERMISSION_DENIED —
    // so "error" is the state a user who simply walked away from the prompt
    // lands in. Without this guard the next weather-shaped message would call
    // getCurrentPosition() again and re-raise the bubble, which is precisely the
    // re-prompt loop this hook exists to avoid. One automatic attempt per
    // session; after that the manual city input is the way forward.
    if (statusRef.current === "error" && attemptedRef.current) return null;

    // Two sends in quick succession must produce one prompt, not two.
    if (inFlightRef.current) return inFlightRef.current;

    const run = (async (): Promise<UserLocation | null> => {
      setPhase("prompting");

      let coords: { latitude: number; longitude: number };
      // Spend the session's single automatic attempt. Set before the call, so a
      // rejection can never leave us thinking we still have one in hand.
      attemptedRef.current = true;
      try {
        coords = await requestBrowserPosition();
      } catch (err) {
        const reason = asFailure((err as Error).message);
        try {
          // Best effort: tell the backend why we have nothing so the LLM can
          // ask for a city rather than silently guessing.
          await postLocationFailure(reason);
        } catch {
          // Recording the failure is not worth surfacing its own failure.
        }
        // A fresh problem deserves to be seen even if an older banner was dismissed.
        if (mountedRef.current) setDismissed(false);
        setPhase(FAILURE_STATUS[reason], FAILURE_COPY[reason]);
        return null;
      }

      // Coordinates in hand means permission was granted, so no further call can
      // raise a bubble — give the attempt back. Otherwise a transient failure in
      // postCoordinates below would wedge us for the rest of the session.
      attemptedRef.current = false;

      setPhase("resolving");
      try {
        const resolved = await postCoordinates(coords.latitude, coords.longitude);
        storeLocation(resolved);
        if (mountedRef.current) setDismissed(false);
        setPhase("ready");
        return resolved;
      } catch (err) {
        const apiMessage = (err as Error).message;
        if (apiMessage === "unauthorized") {
          setPhase("error", "Your session expired. Please sign in again.");
          onUnauthorized();
          return null;
        }
        if (mountedRef.current) setDismissed(false);
        setPhase("error", apiMessage);
        return null;
      }
    })();

    inFlightRef.current = run;
    try {
      return await run;
    } finally {
      inFlightRef.current = null;
    }
  }, [onUnauthorized, setPhase, storeLocation]);

  /* ── Manual city ──────────────────────────────────────────────── */

  const submitManualCity = useCallback(
    async (city: string) => {
      const trimmed = city.trim();
      if (!trimmed) return;

      setPhase("resolving");
      try {
        const resolved = await postManualCity(trimmed);
        storeLocation(resolved);
        attemptedRef.current = false;
        if (mountedRef.current) setDismissed(false);
        setPhase("ready");
        onToast("success", `Using ${resolved.label}`);
      } catch (err) {
        const apiMessage = (err as Error).message;
        if (apiMessage === "unauthorized") {
          setPhase("error", "Your session expired. Please sign in again.");
          onToast("error", "Please sign in again.");
          onUnauthorized();
          return;
        }
        setPhase("error", apiMessage);
        onToast("error", apiMessage);
      }
    },
    [onToast, onUnauthorized, setPhase, storeLocation]
  );

  const clearLocation = useCallback(async () => {
    try {
      await deleteStoredLocation();
    } catch (err) {
      const apiMessage = (err as Error).message;
      if (apiMessage === "unauthorized") {
        onUnauthorized();
        return;
      }
      onToast("error", apiMessage);
      return;
    }
    storeLocation(null);
    // "Stop sharing" is an explicit reset, so hand back the session's automatic
    // attempt — the next location-dependent message may prompt again.
    attemptedRef.current = false;
    if (mountedRef.current) setDismissed(false);
    setPhase("idle");
    onToast("info", "Location sharing stopped");
  }, [onToast, onUnauthorized, setPhase, storeLocation]);

  /** Hide the denial banner for the rest of the session. */
  const dismiss = useCallback(() => setDismissed(true), []);

  return {
    status,
    location,
    message,
    ensureLocation,
    submitManualCity,
    clearLocation,
    dismissed,
    dismiss,
  };
}
