"""
Coordinate validation, reverse geocoding and forward geocoding.

Upstream is the public OpenStreetMap Nominatim service (no API key). Its usage
policy is the constraint that shapes this module:

  * an identifying User-Agent is mandatory   -> location_config.nominatim_user_agent
  * at most one request per second           -> a shared AsyncRateLimiter
  * do not repeat identical queries          -> a TTL cache keyed on rounded coords

Every public function either returns a normalized ``Location`` dict or raises
``LocationError``, which carries a machine-readable ``code``. Nothing here
raises a bare requests exception at its caller, and nothing returns a partially
populated result: an answer that says "somewhere" is worse than a structured
error the agent can explain.

This module imports no application modules beyond its own config, so it is
testable with nothing but a stubbed transport.
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any, TypedDict

import requests

import location_config
import service_cache

logger = logging.getLogger("chatbot.location")

# Error codes returned to the agent and (mapped to HTTP status) to the client.
INVALID_COORDINATES = "INVALID_COORDINATES"
LOCATION_NOT_FOUND = "LOCATION_NOT_FOUND"
GEOCODING_TIMEOUT = "GEOCODING_TIMEOUT"
GEOCODING_RATE_LIMITED = "GEOCODING_RATE_LIMITED"
GEOCODING_UNAVAILABLE = "GEOCODING_UNAVAILABLE"
GEOCODING_MALFORMED_RESPONSE = "GEOCODING_MALFORMED_RESPONSE"

# Nominatim spreads the "populated place" name across a dozen possible keys and
# which one is present depends entirely on how that area is tagged in OSM. A
# village has no `city`; a metro address may have `city` but no `town`. Probe in
# descending specificity so the most human-recognisable name wins.
_CITY_KEYS = (
    "city",
    "town",
    "village",
    "municipality",
    "borough",
    "city_district",
    "suburb",
    "hamlet",
    "locality",
    "county",
)
_STATE_KEYS = ("state", "province", "region", "state_district", "county")

# A failed timezone lookup is remembered only briefly -- long enough to stop an
# outage from being re-paid on every request, short enough that recovery is
# picked up quickly. Successes are cached for the full geocode TTL.
_TIMEZONE_FAILURE_TTL_SECONDS = 300.0


class Location(TypedDict):
    """The single normalized shape every location in this app is expressed in."""

    latitude: float
    longitude: float
    city: str | None
    state: str | None
    country: str | None
    country_code: str | None
    timezone: str | None
    label: str


class LocationError(Exception):
    """A location lookup failed in a way the caller is expected to handle."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_coordinates(latitude: Any, longitude: Any) -> tuple[float, float]:
    """
    Coerce and range-check a coordinate pair.

    Raises LocationError(INVALID_COORDINATES) for anything non-numeric, NaN,
    infinite, or outside [-90, 90] / [-180, 180]. Booleans are rejected too:
    ``True`` is an int in Python and would silently geocode as latitude 1.
    """
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise LocationError(INVALID_COORDINATES, "Coordinates must be numbers.")
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise LocationError(INVALID_COORDINATES, "Coordinates must be numbers.") from exc
    # NaN fails every comparison, so the range checks below would pass it.
    if lat != lat or lon != lon:
        raise LocationError(INVALID_COORDINATES, "Coordinates must be finite numbers.")
    if lat in (float("inf"), float("-inf")) or lon in (float("inf"), float("-inf")):
        raise LocationError(INVALID_COORDINATES, "Coordinates must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise LocationError(
            INVALID_COORDINATES, "Latitude must be between -90 and 90."
        )
    if not -180.0 <= lon <= 180.0:
        raise LocationError(
            INVALID_COORDINATES, "Longitude must be between -180 and 180."
        )
    return lat, lon


def build_label(
    city: str | None,
    state: str | None,
    country: str | None,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> str:
    """
    A human display string, never empty.

    Falls back to coordinates when OSM knows nothing nameable about the point
    (mid-ocean, Antarctica), which is honest rather than pretending to a city.
    """
    parts = [part for part in (city, state, country) if part]
    # "Bengaluru, Bengaluru, India" reads like a bug; collapse repeats.
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    if deduped:
        return ", ".join(deduped)
    if latitude is not None and longitude is not None:
        return f"{latitude:.3f}, {longitude:.3f}"
    return "Unknown location"


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

def _geocode_cache() -> service_cache.CacheBackend:
    return service_cache.get_cache(
        "geocode", max_entries=location_config.geocode_cache_max_entries()
    )


def _timezone_cache() -> service_cache.CacheBackend:
    return service_cache.get_cache(
        "timezone", max_entries=location_config.geocode_cache_max_entries()
    )


def _nominatim_limiter() -> service_cache.AsyncRateLimiter:
    return service_cache.get_rate_limiter(
        "nominatim", location_config.nominatim_min_interval_seconds()
    )


async def _nominatim_get(path: str, params: dict[str, Any]) -> Any:
    """
    One rate-limited, timed Nominatim GET returning parsed JSON.

    Deliberately does not retry. A 429 from Nominatim means the application is
    already over its budget, and retrying is how an IP gets blocked; a timeout
    on a 10 s budget means the service is struggling and a second attempt would
    just double the user's wait. The agent surfaces the structured error and
    offers the user a manual city instead.
    """
    try:
        await _nominatim_limiter().acquire(
            max_wait=location_config.nominatim_max_queue_wait_seconds()
        )
    except service_cache.RateLimitTimeout as exc:
        # Our own queue, not Nominatim's: too many callers are already waiting
        # for the shared 1 req/s budget. Same user-facing shape as an upstream
        # 429, and it keeps a flood from stalling every other request.
        logger.warning("Nominatim %s dropped: local rate-limit queue is full", path)
        raise LocationError(
            GEOCODING_RATE_LIMITED,
            "Too many location lookups are queued right now. Try again shortly.",
        ) from exc

    url = f"{location_config.nominatim_base_url()}{path}"
    headers = {
        # Mandatory under Nominatim's usage policy. A request without it is
        # rejected, and one with a generic UA can get the whole IP blocked.
        "User-Agent": location_config.nominatim_user_agent(),
        "Accept": "application/json",
    }
    timeout = location_config.nominatim_timeout_seconds()
    started = monotonic()
    try:
        response = await asyncio.to_thread(
            requests.get, url, params=params, headers=headers, timeout=timeout
        )
    except requests.Timeout as exc:
        logger.warning(
            "Nominatim %s timed out after %.1fs", path, monotonic() - started
        )
        raise LocationError(
            GEOCODING_TIMEOUT, "The location lookup service did not respond in time."
        ) from exc
    except requests.RequestException as exc:
        # Exception CLASS only. A requests connection error embeds the full
        # request URL, whose query string holds the user's coordinates at
        # higher precision than this application is willing to store.
        logger.warning("Nominatim %s failed: %s", path, type(exc).__name__)
        raise LocationError(
            GEOCODING_UNAVAILABLE, "The location lookup service is unreachable."
        ) from exc

    elapsed_ms = (monotonic() - started) * 1000
    status = getattr(response, "status_code", 0)
    # Coordinates are deliberately absent from this line: the query string
    # carries the user's position and this log is not the place for it.
    logger.info(
        "Nominatim %s -> %s in %.0fms", path, status, elapsed_ms
    )

    if status == 429:
        raise LocationError(
            GEOCODING_RATE_LIMITED,
            "The location lookup service is rate limiting requests. Try again shortly.",
        )
    if status in (401, 403):
        raise LocationError(
            GEOCODING_UNAVAILABLE, "The location lookup service rejected the request."
        )
    if status == 404:
        raise LocationError(LOCATION_NOT_FOUND, "That location could not be found.")
    if status >= 400:
        raise LocationError(
            GEOCODING_UNAVAILABLE,
            f"The location lookup service returned an error (HTTP {status}).",
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise LocationError(
            GEOCODING_MALFORMED_RESPONSE,
            "The location lookup service returned an unreadable response.",
        ) from exc
    return payload


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def _first_present(address: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = address.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_place(
    payload: Any,
    *,
    fallback_lat: float | None = None,
    fallback_lon: float | None = None,
) -> Location:
    """
    Turn one Nominatim place object into a Location.

    Raises GEOCODING_MALFORMED_RESPONSE rather than guessing when the payload
    is not a place object at all -- the raw response never reaches the LLM, so a
    shape change upstream must fail loudly here instead of degrading quietly.

    The fallback coordinates are only meaningful for a reverse geocode, where
    the caller's own position is the right answer if the echo is unusable. A
    forward geocode passes none: substituting (0, 0) there would hand back the
    Gulf of Guinea under the name of whatever city was asked for, and the model
    would faithfully report open-ocean weather as London's.
    """
    if not isinstance(payload, dict):
        raise LocationError(
            GEOCODING_MALFORMED_RESPONSE,
            "The location lookup service returned an unexpected response.",
        )
    if payload.get("error"):
        raise LocationError(LOCATION_NOT_FOUND, "That location could not be found.")

    address = payload.get("address")
    if not isinstance(address, dict):
        address = {}

    # Nominatim echoes the snapped position as strings. Prefer them (they are
    # the centroid of the matched place, which is what a forward geocode is
    # for), but fall back to the query coordinates if they are unusable.
    try:
        lat, lon = validate_coordinates(payload.get("lat"), payload.get("lon"))
    except LocationError as exc:
        if fallback_lat is None or fallback_lon is None:
            raise LocationError(
                GEOCODING_MALFORMED_RESPONSE,
                "The location lookup service returned no usable coordinates.",
            ) from exc
        lat, lon = fallback_lat, fallback_lon

    city = _first_present(address, _CITY_KEYS)
    if city is None:
        # A forward geocode of a city returns its name at the top level with
        # addresstype "city"; the address block may only carry the state.
        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            city = name.strip()

    state = _first_present(address, _STATE_KEYS)
    if state == city:
        state = None
    country = _first_present(address, ("country",))
    raw_code = address.get("country_code")
    country_code = raw_code.strip().upper() if isinstance(raw_code, str) and raw_code.strip() else None

    return Location(
        latitude=lat,
        longitude=lon,
        city=city,
        state=state,
        country=country,
        country_code=country_code,
        timezone=None,
        label=build_label(city, state, country, latitude=lat, longitude=lon),
    )


# --------------------------------------------------------------------------
# Timezone
# --------------------------------------------------------------------------

async def resolve_timezone(latitude: float, longitude: float) -> str | None:
    """
    Best-effort IANA timezone name for a coordinate pair.

    Open-Meteo returns it with ``timezone=auto`` and needs no key. This is the
    only reason the call exists: OpenWeather's free 2.5 endpoints return a UTC
    offset in seconds, not a zone name, and a name is what a user recognises.

    Never raises. A missing timezone makes the answer slightly less complete;
    it must not make the answer fail. Cached with the same TTL as geocoding
    because a coordinate's timezone is as static as its city.
    """
    if not location_config.timezone_lookup_enabled():
        return None

    precision = location_config.geocode_coord_precision()
    key = f"tz:{round(latitude, precision)}:{round(longitude, precision)}"
    cache = _timezone_cache()
    cached = await cache.get(key)
    if isinstance(cached, str):
        return cached or None

    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "timezone": "auto",
        # The API requires at least one data selection; this is the cheapest.
        "current": "temperature_2m",
    }

    async def remember_failure() -> None:
        """
        Cache the miss briefly.

        Without this, an Open-Meteo outage costs a full timeout on every
        geocode cache miss -- stacked on the Nominatim limiter wait, that pushes
        POST /api/location past ten seconds and into the frontend's own error
        path, all for a field that is optional.
        """
        await cache.set(key, "", _TIMEZONE_FAILURE_TTL_SECONDS)

    started = monotonic()
    try:
        response = await asyncio.to_thread(
            requests.get,
            location_config.timezone_lookup_url(),
            params=params,
            timeout=location_config.weather_timeout_seconds(),
        )
        status = getattr(response, "status_code", 0)
        if status >= 400:
            logger.info("Timezone lookup returned HTTP %s", status)
            await remember_failure()
            return None
        payload = response.json()
    except Exception as exc:
        # Type only: a connection error's message carries the query string,
        # and that query string is the user's position.
        logger.info(
            "Timezone lookup failed after %.0fms: %s",
            (monotonic() - started) * 1000,
            type(exc).__name__,
        )
        await remember_failure()
        return None

    timezone_name = payload.get("timezone") if isinstance(payload, dict) else None
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        await remember_failure()
        return None
    timezone_name = timezone_name.strip()
    await cache.set(key, timezone_name, location_config.geocode_cache_ttl_seconds())
    return timezone_name


def offset_to_utc_label(offset_seconds: Any) -> str | None:
    """
    Render a UTC offset in seconds as ``UTC+05:30``.

    Used when a provider gives an offset but no zone name, so the normalized
    ``timezone`` field is still something truthful rather than null.
    """
    try:
        total = int(offset_seconds)
    except (TypeError, ValueError):
        return None
    if abs(total) > 18 * 3600:
        return None
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"UTC{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

async def reverse_geocode(
    latitude: Any,
    longitude: Any,
    *,
    with_timezone: bool = True,
) -> Location:
    """
    Coordinates -> normalized location, via Nominatim /reverse.

    The cache key is the coordinate pair rounded to
    GEOCODE_COORD_PRECISION decimals, so consecutive GPS readings from one
    neighbourhood are a single upstream request. A cache hit does zero network
    I/O, which is what keeps a single user query from producing multiple
    Nominatim calls.
    """
    lat, lon = validate_coordinates(latitude, longitude)

    precision = location_config.geocode_coord_precision()
    cache_key = f"rev:{round(lat, precision)}:{round(lon, precision)}"
    cache = _geocode_cache()
    cached = await cache.get(cache_key)
    if isinstance(cached, dict):
        location = Location(**cached)  # type: ignore[typeddict-item]
    else:
        # Only the cache-key precision is sent upstream. Nothing downstream
        # uses more, and there is no reason to hand a third party a position
        # more precise than this application will even store.
        payload = await _nominatim_get(
            "/reverse",
            {
                "lat": f"{round(lat, precision):.{precision}f}",
                "lon": f"{round(lon, precision):.{precision}f}",
                "format": "jsonv2",
                "addressdetails": 1,
                "zoom": 12,
            },
        )
        location = _normalize_place(payload, fallback_lat=lat, fallback_lon=lon)

    if with_timezone and not location.get("timezone"):
        location["timezone"] = await resolve_timezone(lat, lon)
        # Cache the finished record, timezone included. Storing it before the
        # lookup meant every cache hit re-resolved a timezone the entry could
        # simply have carried.
        if not isinstance(cached, dict):
            await cache.set(
                cache_key, dict(location), location_config.geocode_cache_ttl_seconds()
            )
    elif not isinstance(cached, dict):
        await cache.set(
            cache_key, dict(location), location_config.geocode_cache_ttl_seconds()
        )

    # Report the coordinates the caller asked about, not the centroid of
    # whatever administrative area matched -- the weather lookup that follows
    # should use where the user actually is. This runs on the cache-hit path
    # too: otherwise a second caller inherits the first one's exact position.
    # The label is rebuilt with them so it can never describe a different point.
    location["latitude"] = lat
    location["longitude"] = lon
    location["label"] = build_label(
        location.get("city"),
        location.get("state"),
        location.get("country"),
        latitude=lat,
        longitude=lon,
    )
    return location


async def forward_geocode(place: str, *, with_timezone: bool = True) -> Location:
    """
    A place name -> normalized location, via Nominatim /search.

    Used when the user names somewhere explicitly ("weather in London"), and
    when they type a city instead of granting location permission.
    """
    query = (place or "").strip()
    if not query:
        raise LocationError(LOCATION_NOT_FOUND, "No place name was provided.")
    if len(query) > 200:
        # Nominatim rejects very long queries anyway; failing here keeps a
        # pathological input from consuming the rate-limit budget.
        raise LocationError(LOCATION_NOT_FOUND, "That place name is too long.")

    cache_key = f"fwd:{query.casefold()}"
    cache = _geocode_cache()
    cached = await cache.get(cache_key)
    if isinstance(cached, dict):
        location = Location(**cached)  # type: ignore[typeddict-item]
    else:
        payload = await _nominatim_get(
            "/search",
            {
                "q": query,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 1,
            },
        )
        if isinstance(payload, dict):
            # /search normally answers with a list; a dict here is either an
            # error object or a shape change. Both are handled downstream.
            results: list[Any] = [payload]
        elif isinstance(payload, list):
            results = payload
        else:
            raise LocationError(
                GEOCODING_MALFORMED_RESPONSE,
                "The location lookup service returned an unexpected response.",
            )
        if not results:
            raise LocationError(
                LOCATION_NOT_FOUND, f"No place matching '{query}' was found."
            )
        location = _normalize_place(results[0])
        await cache.set(
            cache_key, dict(location), location_config.geocode_cache_ttl_seconds()
        )

    if with_timezone and not location.get("timezone"):
        location["timezone"] = await resolve_timezone(
            location["latitude"], location["longitude"]
        )
    return location
