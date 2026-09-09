"use client";

import { type LocationStatus as LocationPhase, type UserLocation } from "@/app/lib/location";
import { cn } from "@/app/lib/utils";
import { Loader2, MapPin, MapPinOff, X } from "lucide-react";
import { useState, type FormEvent } from "react";
import Button, { IconButton } from "./ui/Button";

interface LocationStatusProps {
  status: LocationPhase;
  location: UserLocation | null;
  message: string | null;
  onSubmitCity: (city: string) => void;
  onRequest: () => void;
  onClear: () => void;
  onDismiss: () => void;
}

/** The extra line of guidance shown under a failure headline. */
const HINTS: Partial<Record<LocationPhase, string>> = {
  denied: "Enable location for this site in your browser settings, or type a city below.",
  unsupported: "Type a city below and it will be used instead.",
  // Names the real cause. Browsers allow geolocation only on HTTPS (localhost
  // excepted), which is why this works in development and not on a plain-http
  // deployment -- and why no browser setting can fix it.
  insecure:
    "Browsers only allow location on HTTPS, so the browser blocked it before we could ask. Type a city below instead.",
  error: "Try again, or type a city below.",
};

/**
 * A one-line strip that sits directly above the composer and explains what is
 * happening with the user's location. Deliberately quiet and never modal — the
 * composer stays usable in every state, including a hard denial.
 */
export default function LocationStatus({
  status,
  location,
  message,
  onSubmitCity,
  onRequest,
  onClear,
  onDismiss,
}: LocationStatusProps) {
  const [city, setCity] = useState("");
  /** Only for the `ready` state, where the city form is opt-in via "Change". */
  const [changing, setChanging] = useState(false);

  if (status === "ready" && !location) return null;

  // Idle used to render nothing, which meant the ONLY way to share a location
  // was for needsLocation() to correctly parse the user's sentence. A typo or
  // an unusual phrasing left them with no control at all and an assistant
  // insisting it could not find them. This is the manual escape hatch.
  if (status === "idle") {
    return (
      <div className="w-full shrink-0 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto flex w-full max-w-3xl items-center gap-2 pt-2">
          <button
            type="button"
            onClick={onRequest}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5",
              "text-micro text-fg-subtle",
              "transition-colors duration-150 ease-standard",
              "hover:bg-hover hover:text-fg"
            )}
          >
            <MapPin className="h-3 w-3 shrink-0" strokeWidth={1.75} aria-hidden />
            Use my location
          </button>
        </div>
      </div>
    );
  }

  const isFailure =
    status === "denied" ||
    status === "unsupported" ||
    status === "insecure" ||
    status === "error";
  const showForm = isFailure || (status === "ready" && changing);
  const canSubmit = city.trim().length > 0;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = city.trim();
    if (!trimmed) return;
    onSubmitCity(trimmed);
    setCity("");
    setChanging(false);
  }

  const headline =
    status === "prompting"
      ? "Getting your location…"
      : status === "resolving"
        ? "Resolving your location…"
        : status === "ready"
          ? (location?.label ?? "")
          : (message ??
            (status === "denied"
              ? "Location permission is required for this request."
              : status === "unsupported"
                ? "This browser does not support location."
                : status === "insecure"
                  ? "Location needs a secure (HTTPS) connection."
                  : "Could not work out where you are."));

  return (
    <div className="w-full shrink-0 px-4 sm:px-6 lg:px-8">
      <div
        className={cn(
          "mx-auto flex w-full max-w-3xl flex-col gap-1.5 pt-2",
          isFailure && "rounded-md border border-line bg-raised p-2.5"
        )}
      >
        <div className="flex items-center gap-2">
          {status === "prompting" || status === "resolving" ? (
            <Loader2
              className="h-3.5 w-3.5 shrink-0 animate-spin text-fg-subtle"
              strokeWidth={2}
              aria-hidden
            />
          ) : status === "ready" ? (
            <MapPin className="h-3.5 w-3.5 shrink-0 text-fg-subtle" strokeWidth={1.75} aria-hidden />
          ) : (
            <MapPinOff className="h-3.5 w-3.5 shrink-0 text-fg-subtle" strokeWidth={1.75} aria-hidden />
          )}

          <p
            aria-live="polite"
            className={cn(
              "min-w-0 flex-1 truncate text-small",
              isFailure ? "text-fg-muted" : "text-fg-subtle"
            )}
          >
            {headline}
          </p>

          {status === "ready" && (
            <>
              <button
                type="button"
                onClick={() => setChanging((prev) => !prev)}
                aria-expanded={changing}
                className={cn(
                  "shrink-0 rounded-sm px-1.5 py-0.5 text-micro text-fg-subtle",
                  "transition-colors duration-150 ease-standard hover:bg-hover hover:text-fg"
                )}
              >
                Change
              </button>
              <IconButton
                label="Stop sharing location"
                size="sm"
                onClick={onClear}
                className="h-6 w-6 shrink-0 rounded-sm"
              >
                <X className="h-3.5 w-3.5" strokeWidth={2} />
              </IconButton>
            </>
          )}

          {isFailure && (
            <IconButton
              label="Dismiss location notice"
              size="sm"
              onClick={onDismiss}
              className="h-6 w-6 shrink-0 rounded-sm"
            >
              <X className="h-3.5 w-3.5" strokeWidth={2} />
            </IconButton>
          )}
        </div>

        {isFailure && HINTS[status] && (
          <p className="pl-5.5 text-micro text-fg-subtle">{HINTS[status]}</p>
        )}

        {showForm && (
          <form onSubmit={handleSubmit} className="flex items-center gap-2 pl-5.5">
            <input
              type="text"
              value={city}
              onChange={(event) => setCity(event.target.value)}
              placeholder="City, e.g. Pune"
              aria-label="City to use for your location"
              autoComplete="address-level2"
              className={cn(
                "h-8 min-w-0 flex-1 rounded-md border border-line bg-canvas px-2.5",
                "text-small text-fg placeholder:text-fg-subtle",
                "outline-none focus:border-focus sm:max-w-64"
              )}
            />
            <Button type="submit" size="sm" variant="secondary" disabled={!canSubmit}>
              Use city
            </Button>
          </form>
        )}
      </div>
    </div>
  );
}
