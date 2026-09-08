const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** A location the backend has resolved and stored for the signed-in user. */
export interface UserLocation {
  latitude: number;
  longitude: number;
  city: string | null;
  state: string | null;
  country: string | null;
  country_code: string | null;
  timezone: string | null;
  /** Always a non-empty display string — safe to render directly. */
  label: string;
  /** `"browser_gps"` or `"manual"`. */
  source: string;
  updated_at: string;
}

export type LocationStatus =
  | "idle"
  | "prompting"
  | "resolving"
  | "ready"
  | "denied"
  | "unsupported"
  | "error";

/** The four ways `navigator.geolocation` can leave us without coordinates. */
export type GeolocationFailure = "denied" | "timeout" | "unavailable" | "unsupported";

export interface StoredLocationResponse {
  status: string;
  location: UserLocation | null;
}

/**
 * Every rejection from this module is a `LocationApiError`, so callers can read
 * `.status` for HTTP-specific handling and `.message` for something showable.
 * A 401 always carries the exact message `"unauthorized"` — the caller uses it
 * as the signal to tear down the session, not as user-facing copy.
 */
export class LocationApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "LocationApiError";
    this.status = status;
  }
}

/* ── Intent detection ───────────────────────────────────────────── */

/**
 * Phrases that point at the *user* rather than at a place — "me", "my", "I",
 * "here", "nearby", "outside". These are unconditional: an own-location
 * deictic beats a named place, because the place is doing some other job in
 * the sentence ("the weather near me in the evening", "anywhere nearby in
 * Koregaon Park"). Checked first, before EXPLICIT_PLACE.
 */
const OWN_LOCATION_DEICTIC: RegExp[] = [
  /\bwhere am i\b/,
  /\bmy (?:current )?location\b/,
  /\bmy coordinates\b/,
  /\bam i\b[^.?!]*\blocated\b/,
  /\bnear me\b/,
  /\baround me\b/,
  /\bnearby\b/,
  /\bmy city\b/,
  /\bmy area\b/,
  /\bclose to me\b/,
  /\bweather here\b/,
  /\bweather outside\b/,
];

/**
 * Words that follow a preposition without naming a place. Mostly time: "in the
 * evening", "at night", "in January". Without this stop-list every temporal
 * phrase would read as a location and suppress the lookup.
 */
const NON_PLACE_AFTER_PREPOSITION = [
  "the\\s+moment",
  "me",
  "my",
  "here",
  "us",
  "now",
  "today",
  "tomorrow",
  "tonight",
  "this",
  "next",
  "last",
  "morning",
  "afternoon",
  "evening",
  "night",
  "weekend",
  "the",
  "a",
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
  "january",
  "february",
  "march",
  "april",
  "may",
  "june",
  "july",
  "august",
  "september",
  "october",
  "november",
  "december",
].join("|");

/**
 * A preposition followed by something that looks like a place name — "in
 * London", "near pune". Case-INSENSITIVE on purpose: lowercase place names are
 * completely normal in chat, and requiring a capital sent "weather in london?"
 * down the GPS path for a question that was never about the user.
 *
 * `for` and `of` are deliberately absent from the preposition set: in this
 * domain they are overwhelmingly temporal or grammatical ("forecast for
 * tomorrow", "chance of rain") rather than locative.
 */
const EXPLICIT_PLACE = new RegExp(
  `\\b(?:in|at|near|around)\\s+(?!(?:${NON_PLACE_AFTER_PREPOSITION})\\b)[\\p{L}][\\p{L}'’.-]*`,
  "iu"
);

/**
 * Weather-shaped phrasings that imply "here" only when no place is named — so
 * they are checked *after* EXPLICIT_PLACE ("is it raining?" needs a location,
 * "is it raining in Paris?" does not). The hot/cold variants are the ones that
 * earn their keep; the rest overlap BARE_WEATHER_TOPIC below.
 */
const WEATHER_INTENT: RegExp[] = [
  /\bweather today\b/,
  /\btoday['’]?s weather\b/,
  /\bcurrent weather\b/,
  /\bweather now\b/,
  /\bis it raining\b/,
  /\bis it hot\b/,
  /\bis it cold\b/,
  /\bhow hot is it\b/,
  /\bhow cold is it\b/,
  /\btemperature (?:right )?now\b/,
  /\bwhat['’]?s the temperature\b/,
  /\bforecast (?:for )?today\b/,
];

/**
 * A bare weather word with no place attached ("What's the weather?") implies
 * "here". `\b` anchoring is what keeps "Whether I should use TypeScript" out.
 */
const BARE_WEATHER_TOPIC =
  /\b(?:weather|temperature|forecast|humidity|rain(?:ing)?|snow(?:ing)?)\b/;

/**
 * A bare mention of location, the counterpart to BARE_WEATHER_TOPIC.
 *
 * Without this, "location" only ever matched as part of "my location", so
 * "what is the current location" — and any phrasing where the possessive is
 * missing or mangled — silently skipped the geolocation request and the
 * assistant answered that it could not determine where the user was, despite
 * the browser being perfectly able to say. A false positive here costs one
 * permission prompt the user can decline; a false negative makes the whole
 * feature look broken, so this leans toward asking.
 */
const BARE_LOCATION_TOPIC = /\b(?:location|coordinates|geolocation|whereabouts)\b/;

/**
 * Re-separate a contraction glued to the following word.
 *
 * Typing "what'smy current location" is common enough, and it defeats every
 * `\b`-anchored pattern below: in "what'smy" there is no word boundary before
 * "my", so `\bmy location\b` cannot match. Splitting after the "'s" restores
 * the boundary. Whitespace is collapsed afterwards because the patterns spell
 * their gaps as single literal spaces.
 */
function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/(['’]s)(?=[a-z])/g, "$1 ")
    .replace(/\s+/g, " ");
}

/**
 * Heuristic: does this message need the user's own location to be answerable?
 *
 * Five stages, and the order is the whole design:
 *   1. an own-location deictic ("near me") always wins;
 *   2. otherwise a named place ("in London") rules the location out;
 *   3. otherwise a weather-shaped phrasing implies "here";
 *   4. otherwise a bare weather topic implies "here";
 *   5. otherwise a bare mention of location is about the user.
 * Everything runs on a normalized copy — capitalisation carries no signal.
 */
export function needsLocation(text: string): boolean {
  if (!text) return false;

  const lowered = normalize(text);
  if (OWN_LOCATION_DEICTIC.some((pattern) => pattern.test(lowered))) return true;
  if (EXPLICIT_PLACE.test(lowered)) return false;
  if (WEATHER_INTENT.some((pattern) => pattern.test(lowered))) return true;
  return BARE_WEATHER_TOPIC.test(lowered) || BARE_LOCATION_TOPIC.test(lowered);
}

/* ── Browser geolocation ────────────────────────────────────────── */

/**
 * Deliberately below the browser's own patience. `sendMessage` sets its loading
 * flag before awaiting a location, and the composer plus stop button are gated
 * on that flag — so this timeout is the floor of a dead-UI window that also
 * includes the backend's ~1.1s Nominatim rate limiter and the geocode round
 * trip. 10s put the worst case past 11s, which reads as a hung app; 8s keeps it
 * under. Long enough for a coarse network fix, short enough not to feel broken.
 */
const DEFAULT_TIMEOUT_MS = 8_000;

/**
 * Promise wrapper around `getCurrentPosition`. Rejects with an `Error` whose
 * message is exactly one of the `GeolocationFailure` values, so callers can
 * switch on it without touching the browser's numeric error codes.
 *
 * `enableHighAccuracy: false` is deliberate: weather and "what's near me" only
 * need city-level precision, and the coarse network fix comes back far faster
 * and without waking the GPS radio — much cheaper on battery.
 */
export function requestBrowserPosition(
  timeoutMs?: number
): Promise<{ latitude: number; longitude: number }> {
  return new Promise((resolve, reject) => {
    if (typeof navigator === "undefined" || !("geolocation" in navigator)) {
      reject(new Error("unsupported"));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) =>
        resolve({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        }),
      (error) => {
        if (error.code === error.PERMISSION_DENIED) reject(new Error("denied"));
        else if (error.code === error.TIMEOUT) reject(new Error("timeout"));
        else reject(new Error("unavailable"));
      },
      {
        enableHighAccuracy: false,
        timeout: timeoutMs ?? DEFAULT_TIMEOUT_MS,
        maximumAge: 300_000,
      }
    );
  });
}

/* ── API client ─────────────────────────────────────────────────── */

/**
 * FastAPI's `detail` is a string for `HTTPException` but an array of
 * `{ loc, msg, type }` objects for validation errors — and absent for a plain
 * 502 from a proxy. Flatten all three into one showable string; never let an
 * object escape into JSX.
 */
function extractDetail(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;

  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();

  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return "";
      })
      .filter((part) => part.trim().length > 0);
    if (parts.length > 0) return parts.join("; ");
  }

  return fallback;
}

async function callLocationApi<T>(init: RequestInit, fallback: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/location`, { credentials: "include", ...init });
  } catch {
    throw new LocationApiError("Could not reach the server. Check API connectivity.", 0);
  }

  let payload: unknown = null;
  try {
    payload = await res.json();
  } catch {
    // 204s and upstream HTML error pages both land here — `payload` stays null
    // and `extractDetail` falls back to the caller's copy.
  }

  // Surfaced as a sentinel rather than prose so callers can trigger re-auth.
  if (res.status === 401) throw new LocationApiError("unauthorized", 401);
  if (!res.ok) throw new LocationApiError(extractDetail(payload, fallback), res.status);

  return payload as T;
}

function requireLocation(payload: StoredLocationResponse | null): UserLocation {
  if (!payload?.location) {
    throw new LocationApiError("The server did not return a usable location.", 502);
  }
  return payload.location;
}

/** Send browser GPS coordinates; resolves to the reverse-geocoded location. */
export async function postCoordinates(
  latitude: number,
  longitude: number
): Promise<UserLocation> {
  const payload = await callLocationApi<StoredLocationResponse>(
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latitude, longitude }),
    },
    "Could not save your location."
  );
  return requireLocation(payload);
}

/** Send a place the user typed; resolves to the forward-geocoded location. */
export async function postManualCity(city: string): Promise<UserLocation> {
  const payload = await callLocationApi<StoredLocationResponse>(
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city: city.trim() }),
    },
    "Could not find that place."
  );
  return requireLocation(payload);
}

/**
 * Record *why* we have no coordinates. Best-effort telemetry for the backend
 * so the LLM can ask for a city instead of guessing — callers may swallow it.
 */
export async function postLocationFailure(reason: GeolocationFailure): Promise<void> {
  await callLocationApi<StoredLocationResponse>(
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: reason }),
    },
    "Could not record the location status."
  );
}

/** Rehydrate whatever the backend already knows. `status: "none"` means nothing. */
export async function fetchStoredLocation(): Promise<StoredLocationResponse> {
  const payload = await callLocationApi<StoredLocationResponse>(
    { method: "GET" },
    "Could not load your saved location."
  );
  return {
    status: typeof payload?.status === "string" ? payload.status : "none",
    location: payload?.location ?? null,
  };
}

/** Forget the stored location server-side. */
export async function deleteStoredLocation(): Promise<void> {
  await callLocationApi<{ status: string }>(
    { method: "DELETE" },
    "Could not stop location sharing."
  );
}
