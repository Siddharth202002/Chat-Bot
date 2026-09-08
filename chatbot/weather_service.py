"""
Live weather for a coordinate pair.

The one rule this module exists to enforce: every number a user sees comes from
a weather provider. There is no default, no interpolation and no "typical for
this time of year" path. If the provider cannot be reached, the caller gets a
``WeatherError`` with a code and the agent says so.

Two providers are supported behind one normalized shape:

  openweather  (default)  api.openweathermap.org/data/2.5
      /weather  -> current conditions, and a UTC offset in seconds
      /forecast -> future 3-hourly slots, aggregated into the high/low and
                   worst precipitation probability for the REST of the local
                   day (the endpoint cannot see the hours already elapsed)
      One Call 3.0 is intentionally not used: it needs a separate subscription
      and answers 401 on a plain free-tier key.

  open-meteo   (keyless fallback)  api.open-meteo.com/v1/forecast
      one call for current + daily, and it returns an IANA timezone name

The API key is read from the environment inside this module and never leaves
the server: it is not in any response body, log line or error message.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone as dt_timezone
from time import monotonic, time
from typing import Any

import requests

import location_config
import location_service
import service_cache

logger = logging.getLogger("chatbot.weather")

WEATHER_NOT_CONFIGURED = "WEATHER_NOT_CONFIGURED"
WEATHER_TIMEOUT = "WEATHER_TIMEOUT"
WEATHER_RATE_LIMITED = "WEATHER_RATE_LIMITED"
WEATHER_UNAUTHORIZED = "WEATHER_UNAUTHORIZED"
WEATHER_NOT_FOUND = "WEATHER_NOT_FOUND"
WEATHER_UNAVAILABLE = "WEATHER_UNAVAILABLE"
WEATHER_MALFORMED_RESPONSE = "WEATHER_MALFORMED_RESPONSE"

# What the `outlook` block's high/low/precipitation figures actually span. The
# two providers genuinely differ -- OpenWeather's free /forecast returns only
# future 3-hourly slots, so by afternoon the morning minimum is unrecoverable,
# while Open-Meteo publishes a true calendar-day aggregate. Rather than paper
# over that with one ambiguous "today", the coverage travels with the numbers
# and the prompt makes the model phrase it correctly.
COVERS_REST_OF_TODAY = "rest_of_today"
COVERS_FULL_DAY = "full_day"

# Logged at most once per process so an operator sees the degraded provenance
# without every request repeating it.
_logged_key_fallback = False


class WeatherError(Exception):
    """A weather lookup failed in a way the caller is expected to handle."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def _weather_cache() -> service_cache.CacheBackend:
    return service_cache.get_cache("weather", max_entries=256)


def _units_labels(units: str) -> dict[str, str]:
    if units == "imperial":
        return {"temperature": "°F", "wind_speed": "mph"}
    return {"temperature": "°C", "wind_speed": "m/s"}


def _as_float(value: Any) -> float | None:
    """Numeric coercion that refuses bools, NaN and non-numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 2)


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _raise_for_status(status: int, provider: str) -> None:
    """Map an upstream HTTP status onto a structured, key-free error."""
    if status == 401:
        raise WeatherError(
            WEATHER_UNAUTHORIZED,
            "The weather provider rejected the configured credentials.",
        )
    if status == 403:
        raise WeatherError(
            WEATHER_UNAUTHORIZED,
            "The weather provider denied access to this endpoint.",
        )
    if status == 404:
        raise WeatherError(
            WEATHER_NOT_FOUND, "The weather provider has no data for that location."
        )
    if status == 429:
        raise WeatherError(
            WEATHER_RATE_LIMITED,
            "The weather provider is rate limiting requests. Try again shortly.",
        )
    if status >= 400:
        raise WeatherError(
            WEATHER_UNAVAILABLE,
            f"The weather provider returned an error (HTTP {status}).",
        )
    _ = provider


async def _get_json(
    url: str, params: dict[str, Any], provider: str, label: str
) -> Any:
    """
    One timed weather GET returning parsed JSON, with every failure mode mapped.

    ``params`` may contain the API key, so it is never logged -- only the
    endpoint label, the status and the latency are.
    """
    started = monotonic()
    try:
        response = await asyncio.to_thread(
            requests.get,
            url,
            params=params,
            timeout=location_config.weather_timeout_seconds(),
        )
    except requests.Timeout as exc:
        logger.warning(
            "Weather %s/%s timed out after %.1fs",
            provider,
            label,
            monotonic() - started,
        )
        raise WeatherError(
            WEATHER_TIMEOUT, "The weather provider did not respond in time."
        ) from exc
    except requests.RequestException as exc:
        # Only the exception CLASS is logged, never its message: a requests
        # connection/SSL error embeds the full request URL, and that query
        # string carries both `appid` (the API key) and the user's coordinates.
        logger.warning(
            "Weather %s/%s request failed: %s", provider, label, type(exc).__name__
        )
        raise WeatherError(
            WEATHER_UNAVAILABLE, "The weather provider is unreachable."
        ) from exc

    status = int(getattr(response, "status_code", 0) or 0)
    logger.info(
        "Weather %s/%s -> %s in %.0fms",
        provider,
        label,
        status,
        (monotonic() - started) * 1000,
    )
    _raise_for_status(status, provider)

    try:
        return response.json()
    except Exception as exc:
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned an unreadable response.",
        ) from exc


# --------------------------------------------------------------------------
# OpenWeatherMap 2.5
# --------------------------------------------------------------------------

def _openweather_current(payload: Any, units: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned an unexpected response.",
        )
    main = payload.get("main")
    if not isinstance(main, dict):
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned no current conditions.",
        )
    temperature = _as_float(main.get("temp"))
    if temperature is None:
        # Without a temperature there is nothing worth reporting, and the one
        # thing this module must never do is supply a plausible substitute.
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned no temperature reading.",
        )

    condition = None
    weather_list = payload.get("weather")
    if isinstance(weather_list, list) and weather_list:
        first = weather_list[0]
        if isinstance(first, dict):
            description = first.get("description") or first.get("main")
            if isinstance(description, str) and description.strip():
                condition = description.strip().capitalize()

    wind = payload.get("wind") if isinstance(payload.get("wind"), dict) else {}

    current: dict[str, Any] = {
        "temperature": temperature,
        "feels_like": _as_float(main.get("feels_like")),
        "humidity": _as_int(main.get("humidity")),
        "wind_speed": _as_float(wind.get("speed")),
        "condition": condition,
        "units": _units_labels(units),
    }
    # Only fields the provider actually returned.
    return {key: value for key, value in current.items() if value is not None}


def _openweather_rest_of_today(
    payload: Any, timezone_offset: int | None
) -> dict[str, Any]:
    """
    High/low and precipitation probability for the REMAINDER of the local day.

    Not the calendar day's high and low, and the naming says so. OpenWeather's
    free /forecast endpoint only returns future 3-hourly slots, so the morning
    minimum is already gone by lunchtime: aggregating what is left and calling
    it "today's low" produced a confidently wrong number (27 C reported for a
    Bengaluru day whose actual low was ~22 C). The honest fix is to report what
    the data actually covers and let the prompt phrase it as "for the rest of
    today".

    "Today" is the local day at the queried point, taken from the wall clock
    rather than from the first slot -- deriving it from slots[0] meant that late
    in the evening, when the next slot already belongs to tomorrow, the whole
    block silently became tomorrow's forecast under today's name. Late at night
    there is legitimately nothing left of today, and this returns {} for it.
    """
    if not isinstance(payload, dict):
        return {}
    slots = payload.get("list")
    if not isinstance(slots, list) or not slots:
        return {}

    # /forecast reports the offset on `city`; prefer it over the one /weather
    # gave us, since it belongs to the same payload the slots came from.
    offset = timezone_offset or 0
    city = payload.get("city")
    if isinstance(city, dict):
        city_offset = _as_int(city.get("timezone"))
        if city_offset is not None:
            offset = city_offset

    local_day = int((time() + offset) // 86400)

    highs: list[float] = []
    lows: list[float] = []
    pops: list[float] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        timestamp = _as_int(slot.get("dt"))
        if timestamp is None or (timestamp + offset) // 86400 != local_day:
            continue
        slot_main = slot.get("main") if isinstance(slot.get("main"), dict) else {}
        high = _as_float(slot_main.get("temp_max"))
        low = _as_float(slot_main.get("temp_min"))
        if high is not None:
            highs.append(high)
        if low is not None:
            lows.append(low)
        pop = _as_float(slot.get("pop"))
        if pop is not None:
            pops.append(pop)

    rest: dict[str, Any] = {}
    if highs:
        rest["high"] = round(max(highs), 2)
    if lows:
        rest["low"] = round(min(lows), 2)
    if pops:
        # OpenWeather reports pop as a 0-1 fraction per slot; the figure for the
        # window is the worst slot, which is what "will I need an umbrella"
        # actually means.
        rest["precipitation_probability"] = int(round(max(pops) * 100))
    return rest


async def _fetch_openweather(latitude: float, longitude: float) -> dict[str, Any]:
    api_key = location_config.openweather_api_key()
    if not api_key:
        raise WeatherError(
            WEATHER_NOT_CONFIGURED, "No weather provider is configured on the server."
        )

    units = location_config.weather_units()
    base = location_config.openweather_base_url()
    common = {
        "lat": f"{latitude:.4f}",
        "lon": f"{longitude:.4f}",
        "appid": api_key,
        "units": units,
    }

    # Both calls go out together: the forecast is only needed for the "today"
    # block, so serialising them would double the user's wait for no reason.
    current_task = asyncio.create_task(
        _get_json(f"{base}/weather", dict(common), "openweather", "weather")
    )
    forecast_task = asyncio.create_task(
        _get_json(
            f"{base}/forecast",
            {**common, "cnt": 16},
            "openweather",
            "forecast",
        )
    )
    results = await asyncio.gather(current_task, forecast_task, return_exceptions=True)
    current_payload, forecast_payload = results

    # Current conditions are the answer; the forecast is an enrichment. A failed
    # forecast therefore degrades the reply (no high/low) instead of failing it.
    if isinstance(current_payload, BaseException):
        raise current_payload
    if isinstance(forecast_payload, BaseException):
        logger.info(
            "Weather openweather/forecast unavailable, answering with current only: %s",
            forecast_payload,
        )
        forecast_payload = None

    current = _openweather_current(current_payload, units)
    offset = _as_int(current_payload.get("timezone")) if isinstance(current_payload, dict) else None
    outlook = _openweather_rest_of_today(forecast_payload, offset)
    if outlook:
        outlook["covers"] = COVERS_REST_OF_TODAY

    provider_place = None
    if isinstance(current_payload, dict):
        name = current_payload.get("name")
        if isinstance(name, str) and name.strip():
            provider_place = name.strip()

    return {
        "provider": "openweathermap",
        "current": current,
        "outlook": outlook,
        "timezone_offset_seconds": offset,
        "timezone": location_service.offset_to_utc_label(offset),
        "provider_place": provider_place,
    }


# --------------------------------------------------------------------------
# Open-Meteo (keyless)
# --------------------------------------------------------------------------

_OPEN_METEO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _fetch_open_meteo(latitude: float, longitude: float) -> dict[str, Any]:
    units = location_config.weather_units()
    params: dict[str, Any] = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
        "weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 1,
        # Returns an IANA zone name for the queried point, which is the reason
        # this provider needs no separate timezone lookup.
        "timezone": "auto",
    }
    if units == "imperial":
        params.update(
            {
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
            }
        )
    else:
        # Open-Meteo's default wind unit is km/h, but _units_labels calls metric
        # wind "m/s" (which is what OpenWeather returns). Without this the
        # normalized payload labels an 18 km/h breeze as 18 m/s -- a 3.6x
        # overstatement that reads as a near-gale.
        params["wind_speed_unit"] = "ms"

    payload = await _get_json(
        location_config.open_meteo_base_url(), params, "open-meteo", "forecast"
    )
    if not isinstance(payload, dict):
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned an unexpected response.",
        )

    current_block = payload.get("current")
    if not isinstance(current_block, dict):
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned no current conditions.",
        )
    temperature = _as_float(current_block.get("temperature_2m"))
    if temperature is None:
        raise WeatherError(
            WEATHER_MALFORMED_RESPONSE,
            "The weather provider returned no temperature reading.",
        )

    code = _as_int(current_block.get("weather_code"))
    current: dict[str, Any] = {
        "temperature": temperature,
        "feels_like": _as_float(current_block.get("apparent_temperature")),
        "humidity": _as_int(current_block.get("relative_humidity_2m")),
        "wind_speed": _as_float(current_block.get("wind_speed_10m")),
        "condition": _OPEN_METEO_CODES.get(code) if code is not None else None,
        "units": _units_labels(units),
    }
    current = {key: value for key, value in current.items() if value is not None}

    # Unlike OpenWeather's 3-hourly series, Open-Meteo's `daily` block is the
    # true calendar-day aggregate, so this outlook covers the whole local day.
    outlook: dict[str, Any] = {}
    daily = payload.get("daily")
    if isinstance(daily, dict):
        def _first(key: str) -> Any:
            series = daily.get(key)
            return series[0] if isinstance(series, list) and series else None

        high = _as_float(_first("temperature_2m_max"))
        low = _as_float(_first("temperature_2m_min"))
        pop = _as_int(_first("precipitation_probability_max"))
        if high is not None:
            outlook["high"] = high
        if low is not None:
            outlook["low"] = low
        if pop is not None:
            outlook["precipitation_probability"] = pop
    if outlook:
        outlook["covers"] = COVERS_FULL_DAY

    zone = payload.get("timezone")
    return {
        "provider": "open-meteo",
        "current": current,
        "outlook": outlook,
        "timezone_offset_seconds": _as_int(payload.get("utc_offset_seconds")),
        "timezone": zone.strip() if isinstance(zone, str) and zone.strip() else None,
        "provider_place": None,
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

async def get_weather(
    latitude: Any,
    longitude: Any,
    *,
    label: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """
    Live weather for a coordinate pair, in the app's normalized shape.

    ``label`` and ``timezone`` are display context the caller already resolved
    (usually from location_service) and are passed through so the agent does not
    have to re-derive them. Everything numeric comes from the provider.

    Raises WeatherError on any upstream failure; raises LocationError for
    invalid coordinates, reusing the same validator the location path uses so
    the two agree on what a valid coordinate is.
    """
    lat, lon = location_service.validate_coordinates(latitude, longitude)
    provider = location_config.weather_provider()

    global _logged_key_fallback
    if provider != location_config.configured_weather_provider() and not _logged_key_fallback:
        _logged_key_fallback = True
        logger.warning(
            "OPENWEATHER_API_KEY is not set; falling back to the keyless "
            "open-meteo provider for weather."
        )

    units = location_config.weather_units()
    precision = location_config.geocode_coord_precision()
    cache_key = f"wx:{provider}:{units}:{round(lat, precision)}:{round(lon, precision)}"
    cache = _weather_cache()

    cached = await cache.get(cache_key)
    if isinstance(cached, dict):
        payload = dict(cached)
    else:
        if provider == "open-meteo":
            payload = await _fetch_open_meteo(lat, lon)
        else:
            payload = await _fetch_openweather(lat, lon)
        payload["retrieved_at"] = (
            datetime.now(dt_timezone.utc).replace(microsecond=0).isoformat()
        )
        await cache.set(
            cache_key, dict(payload), location_config.weather_cache_ttl_seconds()
        )

    # An IANA zone name beats the UTC offset label the OpenWeather 2.5
    # endpoints imply: "Asia/Kolkata" is recognisable, "UTC+05:30" is not. The
    # lookup is cached alongside the geocode, so this is almost always free.
    resolved_timezone = timezone or payload.get("timezone")
    if not resolved_timezone or str(resolved_timezone).startswith("UTC"):
        resolved_timezone = (
            await location_service.resolve_timezone(lat, lon) or resolved_timezone
        )

    display = label or payload.get("provider_place") or f"{lat:.3f}, {lon:.3f}"

    return {
        "location": display,
        "latitude": lat,
        "longitude": lon,
        "timezone": resolved_timezone,
        "provider": payload.get("provider"),
        # When the provider was actually called. Without this a cache hit up to
        # WEATHER_CACHE_TTL old is indistinguishable from a live reading, and
        # the model would present it as "currently".
        "retrieved_at": payload.get("retrieved_at"),
        "current": dict(payload.get("current") or {}),
        "outlook": dict(payload.get("outlook") or {}),
    }
