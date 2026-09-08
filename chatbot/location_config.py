"""
Configuration for the location and weather subsystem.

Every value is read from the environment on each call rather than frozen at
import time, so tests (and a restart-free settings change) can alter behaviour
without re-importing the module. This mirrors memory_config, which is the
project's established configuration convention.

Required for weather to work at all:
    OPENWEATHER_API_KEY      - OpenWeatherMap key. NEVER hardcode it, and never
                               send it to the frontend.

Optional (defaults in brackets):
    WEATHER_PROVIDER              [openweather]  openweather | open-meteo
    OPENWEATHER_BASE_URL          [https://api.openweathermap.org/data/2.5]
    OPENWEATHER_UNITS             [metric]       metric | imperial
    OPEN_METEO_BASE_URL           [https://api.open-meteo.com/v1/forecast]
    WEATHER_TIMEOUT               [10]   seconds per upstream request
    WEATHER_CACHE_TTL             [600]  seconds, hard-capped at 1800

    NOMINATIM_BASE_URL            [https://nominatim.openstreetmap.org]
    NOMINATIM_USER_AGENT          [derived from ASSISTANT_NAME + NOMINATIM_CONTACT]
    NOMINATIM_CONTACT             []     contact URL or email for the User-Agent
    NOMINATIM_TIMEOUT             [10]
    NOMINATIM_MIN_INTERVAL        [1.1]  seconds between requests (policy: <= 1/s)
    NOMINATIM_MAX_QUEUE_WAIT      [15]   give up rather than queue behind a flood
    GEOCODE_CACHE_TTL             [86400]
    GEOCODE_CACHE_MAX_ENTRIES     [512]
    GEOCODE_COORD_PRECISION       [3]    decimals in the cache key (~110 m)

    LOCATION_STORE_PRECISION      [2]    decimals persisted per user (~1.1 km)
    LOCATION_TTL_SECONDS          [3600] how long a stored location stays usable
    LOCATION_DENIAL_TTL_SECONDS   [86400] how long a refusal is remembered
    LOCATION_TIMEZONE_LOOKUP      [true] resolve an IANA timezone for a location
    TIMEZONE_LOOKUP_URL           [https://api.open-meteo.com/v1/forecast]
"""

from __future__ import annotations

import os

DEFAULT_WEATHER_PROVIDER = "openweather"
SUPPORTED_WEATHER_PROVIDERS: frozenset[str] = frozenset({"openweather", "open-meteo"})

# OpenWeather's One Call 3.0 endpoint needs a separate subscription and answers
# 401 on a plain free-tier key, so this integration targets the 2.5 endpoints
# (/weather for current conditions, /forecast for today's high/low and
# precipitation probability). Both are on the free tier.
DEFAULT_OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
DEFAULT_OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
DEFAULT_TIMEZONE_LOOKUP_URL = "https://api.open-meteo.com/v1/forecast"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError:
            value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _get_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, float(raw.strip()))
    except ValueError:
        return default


def _get_url(name: str, default: str) -> str:
    return ((os.getenv(name) or "").strip() or default).rstrip("/")


# --------------------------------------------------------------------------
# Weather provider
# --------------------------------------------------------------------------

def weather_provider() -> str:
    """
    The active provider name.

    Falls back to the keyless open-meteo provider when openweather is selected
    but no key is configured: an unconfigured key should degrade the answer's
    provenance, not remove the capability. weather_service logs that fallback
    once so it is visible to an operator.
    """
    requested = ((os.getenv("WEATHER_PROVIDER") or "").strip().lower()
                 or DEFAULT_WEATHER_PROVIDER)
    if requested not in SUPPORTED_WEATHER_PROVIDERS:
        return DEFAULT_WEATHER_PROVIDER
    if requested == "openweather" and not openweather_api_key():
        return "open-meteo"
    return requested


def configured_weather_provider() -> str:
    """What the environment asked for, before the missing-key fallback."""
    requested = ((os.getenv("WEATHER_PROVIDER") or "").strip().lower()
                 or DEFAULT_WEATHER_PROVIDER)
    return requested if requested in SUPPORTED_WEATHER_PROVIDERS else DEFAULT_WEATHER_PROVIDER


def openweather_api_key() -> str:
    """The OpenWeather key, from the environment only. Empty string when unset."""
    return (os.getenv("OPENWEATHER_API_KEY") or "").strip()


def openweather_base_url() -> str:
    return _get_url("OPENWEATHER_BASE_URL", DEFAULT_OPENWEATHER_BASE_URL)


def open_meteo_base_url() -> str:
    return _get_url("OPEN_METEO_BASE_URL", DEFAULT_OPEN_METEO_BASE_URL)


def weather_units() -> str:
    units = (os.getenv("OPENWEATHER_UNITS") or "").strip().lower()
    return units if units in {"metric", "imperial"} else "metric"


def weather_timeout_seconds() -> float:
    return max(1.0, _get_float("WEATHER_TIMEOUT", 10.0))


def weather_cache_ttl_seconds() -> float:
    """
    How long a provider reading may be reused, capped at 30 minutes.

    The cap is deliberate. A cached reading is served as the answer to "what is
    the weather right now", so an operator who set this to a day to save quota
    would turn the feature into a source of confidently stale numbers. The
    payload also carries retrieved_at so the age is never invisible.
    """
    return min(1800.0, _get_float("WEATHER_CACHE_TTL", 600.0))


# --------------------------------------------------------------------------
# Nominatim / geocoding
# --------------------------------------------------------------------------

def nominatim_base_url() -> str:
    return _get_url("NOMINATIM_BASE_URL", DEFAULT_NOMINATIM_BASE_URL)


def nominatim_contact() -> str:
    return (os.getenv("NOMINATIM_CONTACT") or "").strip()


def nominatim_user_agent() -> str:
    """
    The identifying User-Agent Nominatim's usage policy requires.

    Set NOMINATIM_USER_AGENT to control it outright. Otherwise it is derived
    from the assistant name plus the optional NOMINATIM_CONTACT, so a deployment
    identifies itself without anyone hardcoding a personal address in source.
    """
    explicit = (os.getenv("NOMINATIM_USER_AGENT") or "").strip()
    if explicit:
        return explicit
    app = (os.getenv("ASSISTANT_NAME") or "Zeno AI").strip().replace(" ", "") or "ZenoAI"
    contact = nominatim_contact()
    return f"{app}-Chatbot/1.0 (+{contact})" if contact else f"{app}-Chatbot/1.0"


def nominatim_timeout_seconds() -> float:
    return max(1.0, _get_float("NOMINATIM_TIMEOUT", 10.0))


def nominatim_min_interval_seconds() -> float:
    """
    Minimum gap between Nominatim requests.

    The public service's policy is an absolute maximum of one request per
    second; the default leaves a little headroom rather than sitting exactly on
    the limit, because being throttled costs far more than 100 ms.
    """
    return _get_float("NOMINATIM_MIN_INTERVAL", 1.1)


def nominatim_max_queue_wait_seconds() -> float:
    """
    How long a request may wait for its Nominatim slot before giving up.

    The 1 req/s budget is global, so without a ceiling one client issuing many
    distinct lookups would park everyone else behind it. 15 s allows a genuine
    burst of a dozen or so queued callers and refuses the pathological case.
    """
    return _get_float("NOMINATIM_MAX_QUEUE_WAIT", 15.0, minimum=1.0)


def geocode_cache_ttl_seconds() -> float:
    """
    A day. Reverse geocoding a coordinate is effectively a constant function --
    city boundaries do not move -- so this only has to be short enough that an
    OSM correction eventually lands.
    """
    return _get_float("GEOCODE_CACHE_TTL", 86400.0)


def geocode_cache_max_entries() -> int:
    return _get_int("GEOCODE_CACHE_MAX_ENTRIES", 512, minimum=1)


def geocode_coord_precision() -> int:
    """
    Decimals kept in a reverse-geocode cache key.

    3 decimals is about 110 m, which is far below the resolution of the answer
    (a city name), so rounding turns "every GPS reading from one neighbourhood"
    into a single cache entry -- exactly the de-duplication Nominatim's usage
    policy asks for.
    """
    return _get_int("GEOCODE_COORD_PRECISION", 3, minimum=0, maximum=6)


# --------------------------------------------------------------------------
# Stored user location
# --------------------------------------------------------------------------

def location_store_precision() -> int:
    """
    Decimals persisted for a user's coordinates.

    2 decimals is roughly 1.1 km: enough for a weather lookup and a city name,
    not enough to identify a building. The precise browser reading is used for
    the lookup and then discarded rather than written to disk.
    """
    return _get_int("LOCATION_STORE_PRECISION", 2, minimum=0, maximum=6)


def location_ttl_seconds() -> int:
    """
    How long a stored location stays usable.

    Long enough that a conversation never re-prompts for GPS, short enough that
    a stale location does not follow the user to another city. The frontend
    only asks the browser again once this has lapsed.
    """
    return _get_int("LOCATION_TTL_SECONDS", 3600, minimum=60)


def location_denial_ttl_seconds() -> int:
    """
    How long a recorded geolocation refusal is remembered.

    Separate from (and much longer than) LOCATION_TTL_SECONDS on purpose. A
    coordinate goes stale in an hour; a user's "no" does not, and re-prompting
    for permission because the refusal lapsed is the behaviour users read as a
    bug. A day is long enough to be respectful and short enough that someone
    who changes their mind is not stuck.
    """
    return _get_int("LOCATION_DENIAL_TTL_SECONDS", 86400, minimum=60)


def timezone_lookup_enabled() -> bool:
    return _get_bool("LOCATION_TIMEZONE_LOOKUP", True)


def timezone_lookup_url() -> str:
    """
    Open-Meteo resolves coordinates to an IANA timezone name ("Asia/Kolkata")
    with no key. OpenWeather's 2.5 endpoints only return a UTC offset in
    seconds, so this one extra (cached, best-effort) call is what makes the
    normalized timezone field a real zone name.
    """
    return _get_url("TIMEZONE_LOOKUP_URL", DEFAULT_TIMEZONE_LOOKUP_URL)
