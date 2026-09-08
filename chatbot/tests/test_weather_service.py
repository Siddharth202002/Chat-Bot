"""
Live weather, entirely from stubbed providers.

The rule this module exists to enforce is "every number a user sees came from a
provider", so the tests that matter most here are the ones that prove the
service refuses to answer rather than filling a gap: a missing or non-numeric
temperature is an error, and a field the provider omitted stays omitted.

The second theme is failure isolation. /weather is the answer and /forecast is
an enrichment, so a broken forecast has to degrade the reply (no high/low)
instead of failing it.
"""

from __future__ import annotations

import json

import pytest
import requests

import location_config
import location_service
import service_cache
import weather_service
from conftest import FakeGet, FakeResponse, apply_location_env, unreadable_response

LAT = 12.9716
LON = 77.5946

OPENWEATHER = "api.openweathermap.org"
OPEN_METEO = "api.open-meteo.com"

# 2024-05-30 12:00:00Z. With Bengaluru's +19800 s offset that is 17:30 local,
# so three 3-hourly slots stay inside the local day and the fourth crosses into
# the next one -- which is exactly the boundary _openweather_today has to get
# right for a 23:00 IST request not to summarise the UTC day.
DT0 = 1717070400
OFFSET = 19800


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """
    Pin "now" to DT0 so the local-day boundary is deterministic.

    _openweather_rest_of_today derives the local day from the wall clock rather
    than from the first forecast slot: taking it from slots[0] meant that late
    in the evening, when the next slot already belongs to tomorrow, the whole
    block silently became tomorrow's forecast presented as today's.
    """
    monkeypatch.setattr(weather_service, "time", lambda: float(DT0))

CURRENT_PAYLOAD = {
    "coord": {"lon": LON, "lat": LAT},
    "weather": [
        {"id": 802, "main": "Clouds", "description": "scattered clouds", "icon": "03d"}
    ],
    "base": "stations",
    "main": {
        "temp": 27.31,
        "feels_like": 29.14,
        "temp_min": 26.05,
        "temp_max": 28.02,
        "pressure": 1012,
        "humidity": 74,
    },
    "visibility": 6000,
    "wind": {"speed": 3.6, "deg": 250},
    "clouds": {"all": 40},
    "dt": DT0,
    "sys": {"country": "IN", "sunrise": 1717027200, "sunset": 1717073000},
    "timezone": OFFSET,
    "id": 1277333,
    "name": "Bengaluru",
    "cod": 200,
}


def forecast_slot(dt: int, temp_max: float, temp_min: float, pop: float) -> dict:
    return {
        "dt": dt,
        "main": {
            "temp": (temp_max + temp_min) / 2,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "humidity": 70,
        },
        "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
        "wind": {"speed": 2.5},
        "pop": pop,
        "dt_txt": "2024-05-30 12:00:00",
    }


FORECAST_PAYLOAD = {
    "cod": "200",
    "message": 0,
    "cnt": 4,
    "list": [
        forecast_slot(DT0, 28.0, 25.0, 0.1),
        forecast_slot(DT0 + 10800, 31.5, 24.5, 0.4),
        forecast_slot(DT0 + 21600, 27.0, 22.0, 0.2),
        # Local 02:30 the next day. Its extremes must not leak into "today".
        forecast_slot(DT0 + 32400, 40.0, 10.0, 0.95),
    ],
    "city": {"id": 1277333, "name": "Bengaluru", "timezone": OFFSET},
}

OPEN_METEO_PAYLOAD = {
    "latitude": 12.97,
    "longitude": 77.59,
    "generationtime_ms": 0.12,
    "utc_offset_seconds": OFFSET,
    "timezone": "Asia/Kolkata",
    "timezone_abbreviation": "IST",
    "elevation": 914.0,
    "current": {
        "time": "2024-05-30T17:30",
        "interval": 900,
        "temperature_2m": 26.4,
        "relative_humidity_2m": 71,
        "apparent_temperature": 28.2,
        "weather_code": 2,
        "wind_speed_10m": 9.7,
    },
    "daily": {
        "time": ["2024-05-30"],
        "temperature_2m_max": [30.1],
        "temperature_2m_min": [21.3],
        "precipitation_probability_max": [35],
    },
}


@pytest.fixture(autouse=True)
def clean_service_caches():
    """
    The weather cache is a process global keyed on rounded coordinates.

    Without this reset a report cached by an earlier test answers the next one
    with zero HTTP calls, and every "the provider was asked for X" assertion
    passes vacuously.
    """
    service_cache.reset_for_tests()
    yield
    service_cache.reset_for_tests()


@pytest.fixture(autouse=True)
def location_env(monkeypatch):
    apply_location_env(monkeypatch)


@pytest.fixture
def fake_get(monkeypatch) -> FakeGet:
    fake = FakeGet()
    monkeypatch.setattr(weather_service.requests, "get", fake)
    return fake


@pytest.fixture
def openweather(fake_get: FakeGet) -> FakeGet:
    """Both endpoints answering happily -- the baseline for the success tests."""
    fake_get.route("/weather", FakeResponse(CURRENT_PAYLOAD))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))
    return fake_get


# --------------------------------------------------------------------------
# OpenWeather success
# --------------------------------------------------------------------------

async def test_openweather_report_has_the_normalized_shape(openweather):
    report = await weather_service.get_weather(LAT, LON, label="Bengaluru, Karnataka")

    assert report["location"] == "Bengaluru, Karnataka"
    assert report["latitude"] == LAT
    assert report["longitude"] == LON
    assert report["timezone"] == "UTC+05:30"
    assert report["provider"] == "openweathermap"
    assert report["current"] == {
        "temperature": 27.31,
        "feels_like": 29.14,
        "humidity": 74,
        "wind_speed": 3.6,
        "condition": "Scattered clouds",
        "units": {"temperature": "°C", "wind_speed": "m/s"},
    }
    assert set(report["outlook"]) == {
        "high",
        "low",
        "precipitation_probability",
        "covers",
    }
    # OpenWeather's free /forecast returns only FUTURE 3-hourly slots, so these
    # figures cannot include the hours already elapsed. The block says so, and
    # the prompt turns that into "for the rest of today" rather than a claim
    # about the day's real low.
    assert report["outlook"]["covers"] == weather_service.COVERS_REST_OF_TODAY
    # Stamped when the provider was actually called, so a cached reading is
    # never indistinguishable from a live one.
    assert report["retrieved_at"]


async def test_openweather_falls_back_to_the_provider_place_name(openweather):
    report = await weather_service.get_weather(LAT, LON)
    assert report["location"] == "Bengaluru"


async def test_a_caller_supplied_timezone_wins(openweather):
    """location_service already resolved an IANA name; prefer it over an offset."""
    report = await weather_service.get_weather(LAT, LON, timezone="Asia/Kolkata")
    assert report["timezone"] == "Asia/Kolkata"


async def test_both_openweather_endpoints_are_called(openweather):
    await weather_service.get_weather(LAT, LON)

    assert len(openweather.calls_to("/weather")) == 1
    assert len(openweather.calls_to("/forecast")) == 1


async def test_todays_precipitation_probability_is_the_worst_slot(openweather):
    """
    OpenWeather reports pop as a 0-1 fraction per slot; the day's figure is the
    maximum across today's slots, times 100, as an int -- which is what "will I
    need an umbrella today" means. Slots given 0.1/0.4/0.2 must yield 40.
    """
    report = await weather_service.get_weather(LAT, LON)
    assert report["outlook"]["precipitation_probability"] == 40


async def test_tomorrows_slots_are_excluded_from_today(openweather):
    """
    The next-day slot carries 40.0/10.0/0.95. If "today" were computed on the
    UTC day (or on every slot in the payload) those extremes would show up
    here, and the user would be told to expect 40 degrees.
    """
    report = await weather_service.get_weather(LAT, LON)

    assert report["outlook"]["high"] == 31.5
    assert report["outlook"]["low"] == 22.0
    assert report["outlook"]["precipitation_probability"] == 40


def test_openweather_outlook_uses_the_citys_offset_over_the_current_offset(
    frozen_clock,
):
    """The forecast payload's own offset is authoritative for its slots."""
    outlook = weather_service._openweather_rest_of_today(FORECAST_PAYLOAD, 0)
    assert outlook == {"high": 31.5, "low": 22.0, "precipitation_probability": 40}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not a dict",
        {},
        {"list": []},
        {"list": "not a list"},
        {"list": [{"no_dt": 1}]},
        {"list": [{"dt": "not a number"}]},
    ],
)
def test_openweather_outlook_degrades_to_empty_on_a_useless_forecast(payload):
    assert weather_service._openweather_rest_of_today(payload, OFFSET) == {}


# --------------------------------------------------------------------------
# The API key never leaves the server
# --------------------------------------------------------------------------

async def test_the_api_key_is_sent_as_appid(openweather):
    await weather_service.get_weather(LAT, LON)

    for call in openweather.calls:
        assert call["params"]["appid"] == "test-key"


async def test_the_api_key_never_appears_in_the_result(openweather):
    report = await weather_service.get_weather(LAT, LON)
    assert "test-key" not in json.dumps(report)


async def test_the_api_key_never_appears_in_an_error_message(fake_get):
    fake_get.route("/weather", FakeResponse({"cod": 401}, status_code=401))
    fake_get.route("/forecast", FakeResponse({"cod": 401}, status_code=401))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert "test-key" not in exc_info.value.message
    assert OPENWEATHER not in exc_info.value.message


# --------------------------------------------------------------------------
# Provider failure isolation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "forecast_target",
    [
        FakeResponse({"cod": "500"}, status_code=500),
        FakeResponse({}, status_code=503),
        requests.Timeout("read timed out"),
        requests.ConnectionError("unreachable"),
        unreadable_response(),
    ],
)
async def test_a_broken_forecast_degrades_the_reply_instead_of_failing_it(
    fake_get, forecast_target
):
    """
    Current conditions are the answer; the forecast only adds the high/low.
    A failed forecast must therefore cost the "outlook" block and nothing else
    -- this is a deliberate degradation and must not regress into a hard
    failure. An empty outlook carries no "covers" key either: there is no
    coverage to describe.
    """
    fake_get.route("/weather", FakeResponse(CURRENT_PAYLOAD))
    fake_get.route("/forecast", forecast_target)

    report = await weather_service.get_weather(LAT, LON)

    assert report["current"]["temperature"] == 27.31
    assert report["outlook"] == {}


async def test_a_broken_current_call_fails_even_when_the_forecast_works(fake_get):
    fake_get.route("/weather", FakeResponse({}, status_code=500))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == weather_service.WEATHER_UNAVAILABLE


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,expected_code",
    [
        (400, weather_service.WEATHER_UNAVAILABLE),
        (401, weather_service.WEATHER_UNAUTHORIZED),
        (403, weather_service.WEATHER_UNAUTHORIZED),
        (404, weather_service.WEATHER_NOT_FOUND),
        (429, weather_service.WEATHER_RATE_LIMITED),
        (500, weather_service.WEATHER_UNAVAILABLE),
        (502, weather_service.WEATHER_UNAVAILABLE),
    ],
)
async def test_http_status_maps_to_a_weather_error_code(fake_get, status, expected_code):
    fake_get.route("/weather", FakeResponse({"cod": status}, status_code=status))
    fake_get.route("/forecast", FakeResponse({"cod": status}, status_code=status))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == expected_code


async def test_a_timeout_maps_to_weather_timeout(fake_get):
    fake_get.route("/weather", requests.Timeout("read timed out"))
    fake_get.route("/forecast", requests.Timeout("read timed out"))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == weather_service.WEATHER_TIMEOUT


async def test_a_connection_error_maps_to_weather_unavailable(fake_get):
    fake_get.route("/weather", requests.ConnectionError("unreachable"))
    fake_get.route("/forecast", requests.ConnectionError("unreachable"))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == weather_service.WEATHER_UNAVAILABLE


# --------------------------------------------------------------------------
# Malformed payloads
# --------------------------------------------------------------------------

async def test_an_unreadable_body_maps_to_malformed(fake_get):
    fake_get.route("/weather", unreadable_response())
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == weather_service.WEATHER_MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "payload",
    [
        ["a", "list"],
        "a string",
        7,
        None,
        {},                                    # no "main" block
        {"main": "not a dict"},
        {"main": {}},                          # no temperature
        {"main": {"temp": None}},
        {"main": {"temp": "warm"}},
        {"main": {"temp": ""}},
        {"main": {"temp": True}},              # bool is not a reading
        {"main": {"temp": float("nan")}},
        {"main": {"temp": float("inf")}},
    ],
)
async def test_a_missing_or_unusable_temperature_is_an_error_not_a_guess(fake_get, payload):
    """
    The one thing this module must never do is supply a plausible substitute
    for a reading the provider did not give. No temperature means no answer.
    """
    fake_get.route("/weather", FakeResponse(payload))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == weather_service.WEATHER_MALFORMED_RESPONSE


async def test_fields_the_provider_omitted_are_omitted_not_defaulted(fake_get):
    """A humidity of 0 and "we were not told the humidity" are different facts."""
    fake_get.route("/weather", FakeResponse({"main": {"temp": 21.5}, "name": "Nowhere"}))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    assert report["current"]["temperature"] == 21.5
    assert "humidity" not in report["current"]
    assert "wind_speed" not in report["current"]
    assert "feels_like" not in report["current"]
    assert "condition" not in report["current"]
    assert report["current"]["units"] == {"temperature": "°C", "wind_speed": "m/s"}


async def test_a_condition_falls_back_to_the_weather_group(fake_get):
    payload = {
        "main": {"temp": 21.5},
        "weather": [{"id": 800, "main": "Clear"}],
        "name": "Nowhere",
    }
    fake_get.route("/weather", FakeResponse(payload))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    assert report["current"]["condition"] == "Clear"


# --------------------------------------------------------------------------
# Coordinate validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "latitude,longitude",
    [(91, 0), (0, 181), ("north", 0), (None, 0), (True, 0), (float("nan"), 0)],
)
async def test_invalid_coordinates_are_rejected_before_any_http_call(
    fake_get, latitude, longitude
):
    with pytest.raises(location_service.LocationError) as exc_info:
        await weather_service.get_weather(latitude, longitude)
    assert exc_info.value.code == location_service.INVALID_COORDINATES
    # The same validator the location path uses, and no rate-limit budget or
    # provider quota is spent on input that can never work.
    assert fake_get.count == 0


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------

async def test_a_missing_key_falls_back_to_the_keyless_provider(monkeypatch, fake_get):
    """An unconfigured key should degrade provenance, not remove the capability."""
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    fake_get.route(OPEN_METEO, FakeResponse(OPEN_METEO_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    assert report["provider"] == "open-meteo"
    assert fake_get.count == 1
    assert OPEN_METEO in fake_get.urls[0]
    assert OPENWEATHER not in fake_get.urls[0]


async def test_an_unknown_provider_name_falls_back_to_the_default(monkeypatch, openweather):
    monkeypatch.setenv("WEATHER_PROVIDER", "banana")

    report = await weather_service.get_weather(LAT, LON)

    assert report["provider"] == "openweathermap"


async def test_open_meteo_report_has_the_normalized_shape(monkeypatch, fake_get):
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    fake_get.route(OPEN_METEO, FakeResponse(OPEN_METEO_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON, label="Bengaluru")

    assert report["provider"] == "open-meteo"
    assert report["location"] == "Bengaluru"
    # Open-Meteo is the reason the normalized timezone is a real zone name and
    # not just an offset, so it must pass straight through.
    assert report["timezone"] == "Asia/Kolkata"
    assert report["current"] == {
        "temperature": 26.4,
        "feels_like": 28.2,
        "humidity": 71,
        "wind_speed": 9.7,
        "condition": "Partly cloudy",
        "units": {"temperature": "°C", "wind_speed": "m/s"},
    }
    # Unlike OpenWeather's future-only 3-hourly series, Open-Meteo's `daily`
    # block IS the true calendar-day aggregate, so this outlook covers the
    # whole day and says so. The two providers genuinely differ, and the
    # coverage travels with the numbers rather than being papered over.
    assert report["outlook"] == {
        "high": 30.1,
        "low": 21.3,
        "precipitation_probability": 35,
        "covers": weather_service.COVERS_FULL_DAY,
    }


async def test_open_meteo_metric_wind_is_requested_in_metres_per_second(
    monkeypatch, fake_get
):
    """
    Open-Meteo's default wind unit is km/h, but the metric label says "m/s".

    Without an explicit wind_speed_unit an 18 km/h breeze was reported as
    18 m/s -- a 3.6x overstatement that reads as a near-gale. This is a
    silent-wrong-number guard, so it is asserted on the request itself.
    """
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    fake_get.route(OPEN_METEO, FakeResponse(OPEN_METEO_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    assert fake_get.calls[0]["params"]["wind_speed_unit"] == "ms"
    assert report["current"]["units"]["wind_speed"] == "m/s"


async def test_open_meteo_imperial_still_requests_mph(monkeypatch, fake_get):
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    monkeypatch.setenv("OPENWEATHER_UNITS", "imperial")
    fake_get.route(OPEN_METEO, FakeResponse(OPEN_METEO_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    assert fake_get.calls[0]["params"]["wind_speed_unit"] == "mph"
    assert fake_get.calls[0]["params"]["temperature_unit"] == "fahrenheit"
    assert report["current"]["units"]["wind_speed"] == "mph"


async def test_an_outlook_of_only_tomorrow_is_reported_as_nothing(
    monkeypatch, fake_get
):
    """
    Late at night every remaining slot belongs to tomorrow.

    The honest answer is then an empty outlook, not tomorrow's high and low
    presented as today's -- which is what deriving the day from slots[0] used
    to produce for a request made at ~23:5x local.
    """
    monkeypatch.setattr(weather_service, "time", lambda: float(DT0 - 86400))
    fake_get.route("/weather", FakeResponse(CURRENT_PAYLOAD))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    assert report["outlook"] == {}
    # The current reading is untouched: only the forward-looking block is gone.
    assert report["current"]["temperature"] == 27.31


async def test_a_cache_hit_keeps_the_original_retrieval_timestamp(openweather):
    """
    The whole point of retrieved_at: a reading served from cache must still
    date to when the provider was actually called, so a stale value can never
    masquerade as live.
    """
    first = await weather_service.get_weather(LAT, LON)
    second = await weather_service.get_weather(LAT, LON)

    assert second["retrieved_at"] == first["retrieved_at"]
    assert len(openweather.calls_to("/weather")) == 1


def test_the_weather_cache_ttl_is_capped(monkeypatch):
    """
    An operator saving quota must not be able to turn "right now" into "some
    time yesterday".
    """
    monkeypatch.setenv("WEATHER_CACHE_TTL", "86400")
    assert location_config.weather_cache_ttl_seconds() == 1800


async def test_open_meteo_condition_is_mapped_from_the_weather_code(monkeypatch, fake_get):
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    payload = json.loads(json.dumps(OPEN_METEO_PAYLOAD))
    payload["current"]["weather_code"] = 95
    fake_get.route(OPEN_METEO, FakeResponse(payload))

    report = await weather_service.get_weather(LAT, LON)

    assert report["current"]["condition"] == "Thunderstorm"


async def test_an_unknown_open_meteo_code_omits_the_condition(monkeypatch, fake_get):
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    payload = json.loads(json.dumps(OPEN_METEO_PAYLOAD))
    payload["current"]["weather_code"] = 4242
    fake_get.route(OPEN_METEO, FakeResponse(payload))

    report = await weather_service.get_weather(LAT, LON)

    assert "condition" not in report["current"]


@pytest.mark.parametrize(
    "payload",
    [
        ["a", "list"],
        {},
        {"current": "not a dict"},
        {"current": {}},
        {"current": {"temperature_2m": None}},
        {"current": {"temperature_2m": "warm"}},
    ],
)
async def test_open_meteo_without_a_temperature_is_an_error(monkeypatch, fake_get, payload):
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    fake_get.route(OPEN_METEO, FakeResponse(payload))

    with pytest.raises(weather_service.WeatherError) as exc_info:
        await weather_service.get_weather(LAT, LON)
    assert exc_info.value.code == weather_service.WEATHER_MALFORMED_RESPONSE


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

async def test_a_repeated_lookup_is_served_from_the_cache(openweather):
    first = await weather_service.get_weather(LAT, LON)
    second = await weather_service.get_weather(LAT, LON)

    assert first == second
    # One /weather plus one /forecast, not two of each.
    assert openweather.count == 2


async def test_coordinates_inside_the_key_precision_share_a_cache_entry(openweather):
    await weather_service.get_weather(12.97161, 77.59462)
    await weather_service.get_weather(12.97162, 77.59461)

    assert openweather.count == 2


async def test_a_coarser_coordinate_difference_is_a_new_lookup(openweather):
    await weather_service.get_weather(12.971, 77.594)
    await weather_service.get_weather(12.981, 77.594)

    assert openweather.count == 4


async def test_switching_units_is_a_different_cache_entry(monkeypatch, openweather):
    await weather_service.get_weather(LAT, LON)
    monkeypatch.setenv("OPENWEATHER_UNITS", "imperial")
    await weather_service.get_weather(LAT, LON)

    assert openweather.count == 4


async def test_a_failed_lookup_is_not_cached(fake_get):
    responses = iter([FakeResponse({}, status_code=500), FakeResponse(CURRENT_PAYLOAD)])
    fake_get.route("/weather", lambda url, params: next(responses))
    fake_get.route("/forecast", FakeResponse(FORECAST_PAYLOAD))

    with pytest.raises(weather_service.WeatherError):
        await weather_service.get_weather(LAT, LON)
    report = await weather_service.get_weather(LAT, LON)

    assert report["current"]["temperature"] == 27.31


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

async def test_imperial_units_are_requested_and_labelled(monkeypatch, openweather):
    monkeypatch.setenv("OPENWEATHER_UNITS", "imperial")

    report = await weather_service.get_weather(LAT, LON)

    for call in openweather.calls:
        assert call["params"]["units"] == "imperial"
    assert report["current"]["units"] == {"temperature": "°F", "wind_speed": "mph"}


async def test_metric_is_the_default_and_an_unknown_unit_falls_back_to_it(
    monkeypatch, openweather
):
    monkeypatch.setenv("OPENWEATHER_UNITS", "kelvin")

    report = await weather_service.get_weather(LAT, LON)

    for call in openweather.calls:
        assert call["params"]["units"] == "metric"
    assert report["current"]["units"]["temperature"] == "°C"


async def test_open_meteo_requests_imperial_units_by_name(monkeypatch, fake_get):
    monkeypatch.setenv("WEATHER_PROVIDER", "open-meteo")
    monkeypatch.setenv("OPENWEATHER_UNITS", "imperial")
    fake_get.route(OPEN_METEO, FakeResponse(OPEN_METEO_PAYLOAD))

    report = await weather_service.get_weather(LAT, LON)

    params = fake_get.calls[0]["params"]
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"
    assert report["current"]["units"]["temperature"] == "°F"


# --------------------------------------------------------------------------
# _as_float
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (1, 1.0),
        (1.005, 1.0),
        (-3.14159, -3.14),
        ("2.5", 2.5),
        (0, 0.0),
    ],
)
def test_as_float_coerces_and_rounds(value, expected):
    assert weather_service._as_float(value) == expected


@pytest.mark.parametrize(
    "value",
    [None, True, False, "warm", "", [], {}, float("nan"), float("inf"), float("-inf")],
)
def test_as_float_refuses_anything_that_is_not_a_reading(value):
    assert weather_service._as_float(value) is None
