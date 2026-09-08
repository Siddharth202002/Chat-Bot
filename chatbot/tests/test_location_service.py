"""
Coordinate validation, geocoding, caching and rate limiting.

Every test here runs against a stubbed ``requests.get``: Nominatim is never
contacted, so the suite passes with the network unplugged. The three things
worth watching in this file are the ones Nominatim's usage policy turns into
correctness requirements -- an identifying User-Agent, a cache that collapses
nearby coordinates into one request, and a limiter that spaces requests out.
"""

from __future__ import annotations

import asyncio
import re
import time

import pytest
import requests

import location_config
import location_service
import service_cache
from conftest import FakeGet, FakeResponse, apply_location_env, unreadable_response

# Bengaluru, to six decimals -- the sort of reading a browser hands over.
LAT = 12.971598
LON = 77.594622

NOMINATIM = "nominatim.openstreetmap.org"
OPEN_METEO = "api.open-meteo.com"


def reverse_payload(address: dict, *, lat: str = "12.9761", lon: str = "77.5993", **extra) -> dict:
    """A Nominatim /reverse jsonv2 place object with a caller-chosen address."""
    payload = {
        "place_id": 297738873,
        "licence": "Data (C) OpenStreetMap contributors",
        "osm_type": "way",
        "osm_id": 123456,
        "lat": lat,
        "lon": lon,
        "category": "highway",
        "type": "residential",
        "addresstype": "road",
        "display_name": "Some Road, Somewhere",
        "address": address,
        "boundingbox": ["12.97", "12.98", "77.59", "77.60"],
    }
    payload.update(extra)
    return payload


BENGALURU_ADDRESS = {
    "road": "Kasturba Road",
    "suburb": "Sampangi Rama Nagar",
    "city": "Bengaluru",
    "state_district": "Bangalore Urban",
    "state": "Karnataka",
    "postcode": "560001",
    "country": "India",
    "country_code": "in",
}

LONDON_SEARCH_RESULT = {
    "place_id": 297691559,
    "osm_type": "relation",
    "osm_id": 65606,
    "lat": "51.5074456",
    "lon": "-0.1277653",
    "category": "place",
    "type": "city",
    "addresstype": "city",
    "name": "London",
    "display_name": "London, Greater London, England, United Kingdom",
    "address": {
        "city": "London",
        "state_district": "Greater London",
        "state": "England",
        "country": "United Kingdom",
        "country_code": "gb",
    },
}


@pytest.fixture(autouse=True)
def clean_service_caches():
    """
    The geocode cache and the Nominatim limiter are process globals.

    Without this reset a cached lookup from an earlier test silently satisfies
    the next one, and a test that asserts "exactly one HTTP call" passes for
    the wrong reason.
    """
    service_cache.reset_for_tests()
    yield
    service_cache.reset_for_tests()


@pytest.fixture(autouse=True)
def location_env(monkeypatch):
    apply_location_env(monkeypatch)


@pytest.fixture
def fake_get(monkeypatch) -> FakeGet:
    """Replace the transport location_service actually uses."""
    fake = FakeGet()
    monkeypatch.setattr(location_service.requests, "get", fake)
    return fake


# --------------------------------------------------------------------------
# validate_coordinates
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "latitude,longitude",
    [
        (0, 0),
        (0.0, 0.0),
        (LAT, LON),
        (90, 180),
        (-90, -180),
        (90.0, -180.0),
        ("12.5", "77.5"),  # the HTTP layer can hand over strings
        (-89.999999, 179.999999),
    ],
)
def test_valid_coordinates_are_accepted(latitude, longitude):
    lat, lon = location_service.validate_coordinates(latitude, longitude)
    assert (lat, lon) == (float(latitude), float(longitude))


@pytest.mark.parametrize(
    "latitude,longitude",
    [
        (91, 0),
        (-91, 0),
        (90.0001, 0),
        (0, 181),
        (0, -181),
        (0, 180.0001),
        ("north", 0),
        (0, "east"),
        ("", ""),
        (None, 0),
        (0, None),
        (float("nan"), 0),
        (0, float("nan")),
        (float("inf"), 0),
        (float("-inf"), 0),
        (0, float("inf")),
        (0, float("-inf")),
        ([12.0], 77.0),
        ({}, 0),
    ],
)
def test_invalid_coordinates_are_rejected(latitude, longitude):
    with pytest.raises(location_service.LocationError) as exc_info:
        location_service.validate_coordinates(latitude, longitude)
    assert exc_info.value.code == location_service.INVALID_COORDINATES


@pytest.mark.parametrize("latitude,longitude", [(True, 0), (0, True), (False, False)])
def test_booleans_are_rejected_as_coordinates(latitude, longitude):
    """bool is an int in Python, so True would otherwise geocode as latitude 1."""
    with pytest.raises(location_service.LocationError) as exc_info:
        location_service.validate_coordinates(latitude, longitude)
    assert exc_info.value.code == location_service.INVALID_COORDINATES


# --------------------------------------------------------------------------
# build_label
# --------------------------------------------------------------------------

def test_build_label_joins_the_parts_it_has():
    assert location_service.build_label("Bengaluru", "Karnataka", "India") == (
        "Bengaluru, Karnataka, India"
    )


def test_build_label_deduplicates_repeated_parts():
    """"Singapore, Singapore, Singapore" reads like a bug, not a place."""
    assert location_service.build_label("Singapore", "Singapore", "Singapore") == "Singapore"
    assert location_service.build_label("Bengaluru", "Bengaluru", "India") == (
        "Bengaluru, India"
    )


def test_build_label_skips_blank_parts():
    assert location_service.build_label(None, "", "India") == "India"


def test_build_label_falls_back_to_coordinates():
    label = location_service.build_label(None, None, None, latitude=LAT, longitude=LON)
    assert label == "12.972, 77.595"


def test_build_label_is_never_empty():
    assert location_service.build_label(None, None, None) == "Unknown location"


# --------------------------------------------------------------------------
# reverse_geocode: success and normalization
# --------------------------------------------------------------------------

async def test_reverse_geocode_normalizes_a_nominatim_place(fake_get):
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] == "Bengaluru"
    assert location["state"] == "Karnataka"
    assert location["country"] == "India"
    assert location["country_code"] == "IN"  # uppercased from the "in" upstream
    assert location["label"] == "Bengaluru, Karnataka, India"
    # The queried position wins over the payload's snapped centroid: the weather
    # lookup that follows must use where the user actually is.
    assert location["latitude"] == LAT
    assert location["longitude"] == LON


async def test_reverse_geocode_sends_the_parameters_nominatim_needs(fake_get):
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    await location_service.reverse_geocode(LAT, LON)

    precision = location_config.geocode_coord_precision()
    call = fake_get.calls_to("/reverse")[0]
    assert NOMINATIM in call["url"]
    assert call["params"]["format"] == "jsonv2"
    assert call["params"]["addressdetails"] == 1
    assert float(call["params"]["lat"]) == pytest.approx(LAT, abs=10**-precision)
    assert float(call["params"]["lon"]) == pytest.approx(LON, abs=10**-precision)
    assert call["timeout"] is not None
    # Only the cache-key precision goes upstream. Nothing downstream uses more,
    # and there is no reason to hand a third party a position finer than this
    # application will even store.
    for axis in ("lat", "lon"):
        decimals = call["params"][axis].split(".")[1]
        assert len(decimals) <= precision


async def test_reverse_geocode_sends_a_non_empty_user_agent(fake_get):
    """
    Nominatim's usage policy makes an identifying User-Agent mandatory; a
    request without one is rejected and a generic one can get the IP blocked.
    This assertion is what stops someone quietly dropping the header.
    """
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    await location_service.reverse_geocode(LAT, LON)

    user_agent = fake_get.calls_to("/reverse")[0]["headers"].get("User-Agent")
    assert isinstance(user_agent, str)
    assert user_agent.strip()


@pytest.mark.parametrize(
    "key,expected",
    [
        ("town", "Nandi Hills"),
        ("village", "Nandi Hills"),
        ("municipality", "Nandi Hills"),
        ("county", "Nandi Hills"),
    ],
)
async def test_reverse_geocode_falls_back_through_the_city_keys(fake_get, key, expected):
    """Which key holds the place name depends entirely on the OSM tagging."""
    fake_get.route(
        "/reverse",
        FakeResponse(reverse_payload({key: "Nandi Hills", "country": "India", "country_code": "in"})),
    )

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] == expected
    assert location["label"] == "Nandi Hills, India"


async def test_reverse_geocode_with_nothing_nameable_falls_back_to_coordinates(fake_get):
    """Mid-ocean and Antarctica: coordinates are honest, "None" is a bug."""
    fake_get.route("/reverse", FakeResponse(reverse_payload({"postcode": "00000"})))

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] is None
    assert location["state"] is None
    assert location["country"] is None
    # A coordinate pair, not "" and not "None, None". The digits come from the
    # payload's snapped position rather than the query (see build_label's
    # caller), so this asserts the shape rather than the exact numbers.
    assert re.fullmatch(r"-?\d+\.\d{3}, -?\d+\.\d{3}", location["label"])
    assert "None" not in location["label"]


async def test_the_coordinate_label_matches_the_reported_coordinates(fake_get):
    """
    For an unnameable point the label IS the coordinates, so it must be built
    from the ones actually reported. reverse_geocode pins the record to the
    queried pair, so the label is rebuilt with them rather than keeping the
    payload's snapped centroid -- otherwise the label described a different
    place from the record's own latitude/longitude.
    """
    fake_get.route(
        "/reverse",
        FakeResponse(reverse_payload({"postcode": "00000"}, lat="12.9761", lon="77.5993")),
    )

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["label"] == (
        f"{location['latitude']:.3f}, {location['longitude']:.3f}"
    )


async def test_reverse_geocode_collapses_a_state_equal_to_the_city(fake_get):
    """City states repeat the name across both keys; only one belongs in the label."""
    fake_get.route(
        "/reverse",
        FakeResponse(
            reverse_payload(
                {
                    "city": "Singapore",
                    "state": "Singapore",
                    "country": "Singapore",
                    "country_code": "sg",
                }
            )
        ),
    )

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] == "Singapore"
    assert location["state"] is None
    assert location["label"] == "Singapore"


async def test_reverse_geocode_uses_the_top_level_name_when_the_address_has_none(fake_get):
    fake_get.route(
        "/reverse",
        FakeResponse(
            reverse_payload(
                {"state": "Karnataka", "country": "India", "country_code": "in"},
                name="Bengaluru",
            )
        ),
    )

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] == "Bengaluru"
    assert location["label"] == "Bengaluru, Karnataka, India"


async def test_reverse_geocode_rejects_invalid_coordinates_without_an_http_call(fake_get):
    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(91, 0)
    assert exc_info.value.code == location_service.INVALID_COORDINATES
    assert fake_get.count == 0


# --------------------------------------------------------------------------
# reverse_geocode: failure mapping
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,expected_code",
    [
        (400, location_service.GEOCODING_UNAVAILABLE),
        (401, location_service.GEOCODING_UNAVAILABLE),
        (403, location_service.GEOCODING_UNAVAILABLE),
        (404, location_service.LOCATION_NOT_FOUND),
        (429, location_service.GEOCODING_RATE_LIMITED),
        (500, location_service.GEOCODING_UNAVAILABLE),
        (503, location_service.GEOCODING_UNAVAILABLE),
    ],
)
async def test_reverse_geocode_maps_http_status_to_an_error_code(fake_get, status, expected_code):
    fake_get.route("/reverse", FakeResponse({"error": "nope"}, status_code=status))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    assert exc_info.value.code == expected_code


async def test_reverse_geocode_maps_a_timeout(fake_get):
    fake_get.route("/reverse", requests.Timeout("read timed out"))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    assert exc_info.value.code == location_service.GEOCODING_TIMEOUT


async def test_reverse_geocode_maps_a_connection_error(fake_get):
    fake_get.route("/reverse", requests.ConnectionError("name resolution failed"))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    assert exc_info.value.code == location_service.GEOCODING_UNAVAILABLE


async def test_reverse_geocode_maps_an_unreadable_body(fake_get):
    fake_get.route("/reverse", unreadable_response())

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    assert exc_info.value.code == location_service.GEOCODING_MALFORMED_RESPONSE


@pytest.mark.parametrize("payload", ["just a string", 42, None, [1, 2, 3]])
async def test_reverse_geocode_rejects_a_payload_that_is_not_a_place(fake_get, payload):
    """The raw response never reaches the LLM, so a shape change must fail loudly."""
    fake_get.route("/reverse", FakeResponse(payload))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    assert exc_info.value.code == location_service.GEOCODING_MALFORMED_RESPONSE


async def test_reverse_geocode_maps_nominatims_error_object(fake_get):
    fake_get.route("/reverse", FakeResponse({"error": "Unable to geocode"}))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    assert exc_info.value.code == location_service.LOCATION_NOT_FOUND


async def test_a_geocoding_error_carries_a_human_message_without_coordinates(fake_get):
    fake_get.route("/reverse", FakeResponse({}, status_code=500))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(LAT, LON)
    message = exc_info.value.message
    assert message and message[0].isupper()
    assert "12.97" not in message
    assert NOMINATIM not in message


# --------------------------------------------------------------------------
# reverse_geocode: caching
# --------------------------------------------------------------------------

async def test_reverse_geocode_caches_an_identical_lookup(fake_get):
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    first = await location_service.reverse_geocode(LAT, LON)
    second = await location_service.reverse_geocode(LAT, LON)

    assert first == second
    assert fake_get.count == 1


async def test_reverse_geocode_collapses_coordinates_inside_the_key_precision(fake_get):
    """
    At GEOCODE_COORD_PRECISION=3 (~110 m) two GPS readings from the same
    doorstep are one cache entry -- which is the de-duplication Nominatim's
    usage policy asks for, not a nicety.
    """
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    await location_service.reverse_geocode(12.97161, 77.594622)
    await location_service.reverse_geocode(12.97162, 77.594622)

    assert fake_get.count == 1


async def test_reverse_geocode_treats_a_coarser_difference_as_a_new_lookup(fake_get):
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    await location_service.reverse_geocode(12.971, 77.594)
    await location_service.reverse_geocode(12.981, 77.594)

    assert fake_get.count == 2


async def test_a_failed_lookup_is_not_cached(fake_get):
    """A transient 500 must not poison the cache for the whole TTL."""
    responses = iter(
        [
            FakeResponse({}, status_code=500),
            FakeResponse(reverse_payload(BENGALURU_ADDRESS)),
        ]
    )
    fake_get.route("/reverse", lambda url, params: next(responses))

    with pytest.raises(location_service.LocationError):
        await location_service.reverse_geocode(LAT, LON)
    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] == "Bengaluru"
    assert fake_get.count == 2


# --------------------------------------------------------------------------
# reverse_geocode: rate limiting
# --------------------------------------------------------------------------

async def test_concurrent_lookups_are_spaced_by_the_rate_limiter(monkeypatch, fake_get):
    """
    Nominatim's budget is global, not per user. Two users asking at the same
    moment must queue behind each other; if they fire together the application
    is over its budget and an IP block is the eventual cost.
    """
    interval = 0.20
    monkeypatch.setenv("NOMINATIM_MIN_INTERVAL", str(interval))
    sent_at: list[float] = []

    def record(url, params):
        sent_at.append(time.monotonic())
        return FakeResponse(reverse_payload(BENGALURU_ADDRESS))

    fake_get.route("/reverse", record)

    started = time.monotonic()
    await asyncio.gather(
        location_service.reverse_geocode(12.100, 77.100),
        location_service.reverse_geocode(50.200, 8.200),
    )
    elapsed = time.monotonic() - started

    assert fake_get.count == 2
    # A fraction of the interval is allowed for the platform's timer
    # resolution: the point is that the second request waited, not that it
    # waited to the microsecond.
    assert sent_at[1] - sent_at[0] >= interval * 0.8
    assert elapsed >= interval * 0.8


# --------------------------------------------------------------------------
# forward_geocode
# --------------------------------------------------------------------------

async def test_forward_geocode_normalizes_the_first_search_result(fake_get):
    fake_get.route("/search", FakeResponse([LONDON_SEARCH_RESULT]))

    location = await location_service.forward_geocode("London")

    assert location["city"] == "London"
    assert location["state"] == "England"
    assert location["country"] == "United Kingdom"
    assert location["country_code"] == "GB"
    assert location["latitude"] == pytest.approx(51.5074456)
    assert location["longitude"] == pytest.approx(-0.1277653)
    assert location["label"] == "London, England, United Kingdom"


async def test_forward_geocode_asks_for_a_single_result(fake_get):
    fake_get.route("/search", FakeResponse([LONDON_SEARCH_RESULT]))

    await location_service.forward_geocode("London")

    params = fake_get.calls_to("/search")[0]["params"]
    assert params["q"] == "London"
    assert params["limit"] == 1
    assert params["format"] == "jsonv2"


async def test_forward_geocode_accepts_a_bare_place_object(fake_get):
    """/search normally answers with a list; a lone object is handled too."""
    fake_get.route("/search", FakeResponse(LONDON_SEARCH_RESULT))

    location = await location_service.forward_geocode("London")

    assert location["city"] == "London"


async def test_forward_geocode_maps_an_empty_result_list(fake_get):
    fake_get.route("/search", FakeResponse([]))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode("Nowhereville")
    assert exc_info.value.code == location_service.LOCATION_NOT_FOUND


@pytest.mark.parametrize("place", ["", "   ", "\t\n", None])
async def test_forward_geocode_rejects_a_blank_name_without_an_http_call(fake_get, place):
    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode(place)
    assert exc_info.value.code == location_service.LOCATION_NOT_FOUND
    assert fake_get.count == 0


async def test_forward_geocode_rejects_an_overlong_name_without_an_http_call(fake_get):
    """A pathological input must not consume the shared rate-limit budget."""
    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode("x" * 201)
    assert exc_info.value.code == location_service.LOCATION_NOT_FOUND
    assert fake_get.count == 0


async def test_forward_geocode_cache_ignores_case_and_whitespace(fake_get):
    fake_get.route("/search", FakeResponse([LONDON_SEARCH_RESULT]))

    await location_service.forward_geocode("London")
    await location_service.forward_geocode("london")
    await location_service.forward_geocode("  LONDON  ")

    assert fake_get.count == 1


@pytest.mark.parametrize(
    "status,expected_code",
    [
        (404, location_service.LOCATION_NOT_FOUND),
        (429, location_service.GEOCODING_RATE_LIMITED),
        (500, location_service.GEOCODING_UNAVAILABLE),
    ],
)
async def test_forward_geocode_maps_http_status_to_an_error_code(fake_get, status, expected_code):
    fake_get.route("/search", FakeResponse([], status_code=status))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode("London")
    assert exc_info.value.code == expected_code


async def test_forward_geocode_maps_a_timeout(fake_get):
    fake_get.route("/search", requests.Timeout("read timed out"))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode("London")
    assert exc_info.value.code == location_service.GEOCODING_TIMEOUT


@pytest.mark.parametrize("payload", ["a string", 7, None])
async def test_forward_geocode_rejects_a_payload_that_is_neither_list_nor_object(fake_get, payload):
    fake_get.route("/search", FakeResponse(payload))

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode("London")
    assert exc_info.value.code == location_service.GEOCODING_MALFORMED_RESPONSE


# --------------------------------------------------------------------------
# resolve_timezone
# --------------------------------------------------------------------------

async def test_resolve_timezone_returns_none_when_the_lookup_is_disabled(fake_get):
    assert await location_service.resolve_timezone(LAT, LON) is None
    assert fake_get.count == 0


async def test_resolve_timezone_returns_the_iana_zone_name(monkeypatch, fake_get):
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route(OPEN_METEO, FakeResponse({"timezone": "Asia/Kolkata"}))

    assert await location_service.resolve_timezone(LAT, LON) == "Asia/Kolkata"


async def test_resolve_timezone_is_cached(monkeypatch, fake_get):
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route(OPEN_METEO, FakeResponse({"timezone": "Asia/Kolkata"}))

    await location_service.resolve_timezone(LAT, LON)
    await location_service.resolve_timezone(LAT, LON)

    assert fake_get.count == 1


@pytest.mark.parametrize(
    "target",
    [
        FakeResponse({"timezone": "Asia/Kolkata"}, status_code=500),
        FakeResponse({"timezone": "Asia/Kolkata"}, status_code=429),
        FakeResponse({}),
        FakeResponse({"timezone": ""}),
        FakeResponse({"timezone": "   "}),
        FakeResponse({"timezone": 12345}),
        FakeResponse(["not", "a", "dict"]),
        unreadable_response(),
        requests.Timeout("read timed out"),
        requests.ConnectionError("unreachable"),
        RuntimeError("something unexpected"),
    ],
)
async def test_a_failed_timezone_lookup_never_raises(monkeypatch, fake_get, target):
    """
    A missing timezone makes the answer slightly less complete. It must never
    make a location lookup fail, which is why every branch here returns None.
    """
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route(OPEN_METEO, target)

    assert await location_service.resolve_timezone(LAT, LON) is None


async def test_reverse_geocode_attaches_a_resolved_timezone(monkeypatch, fake_get):
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))
    fake_get.route(OPEN_METEO, FakeResponse({"timezone": "Asia/Kolkata"}))

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["timezone"] == "Asia/Kolkata"


async def test_reverse_geocode_still_succeeds_when_the_timezone_lookup_fails(
    monkeypatch, fake_get
):
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))
    fake_get.route(OPEN_METEO, requests.ConnectionError("unreachable"))

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["city"] == "Bengaluru"
    assert location["timezone"] is None


async def test_reverse_geocode_can_skip_the_timezone_lookup(monkeypatch, fake_get):
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    location = await location_service.reverse_geocode(LAT, LON, with_timezone=False)

    assert location["timezone"] is None
    # Only /reverse went out: an unrouted timezone call would have asserted.
    assert fake_get.count == 1


# --------------------------------------------------------------------------
# offset_to_utc_label
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "offset,expected",
    [
        (19800, "UTC+05:30"),
        (0, "UTC+00:00"),
        (-18000, "UTC-05:00"),
        (3600, "UTC+01:00"),
        (-34200, "UTC-09:30"),
        (50400, "UTC+14:00"),
    ],
)
def test_offset_to_utc_label_renders_an_offset(offset, expected):
    assert location_service.offset_to_utc_label(offset) == expected


@pytest.mark.parametrize("offset", [None, "abc", "", [], {}, 999999, -999999, 64801])
def test_offset_to_utc_label_rejects_what_it_cannot_render(offset):
    """No real zone is more than 14 h off UTC, so 999999 is corrupt input."""
    assert location_service.offset_to_utc_label(offset) is None


# --------------------------------------------------------------------------
# Regressions from the code review
# --------------------------------------------------------------------------

async def test_forward_geocode_refuses_to_invent_coordinates(fake_get):
    """
    A place name must never come back attached to (0, 0).

    _normalize_place used to substitute a (0.0, 0.0) fallback when the payload
    carried no usable lat/lon, while still building city/state/country from the
    address block. geocode_location("London") could therefore return the Gulf
    of Guinea labelled "London", and the model would faithfully report
    open-ocean weather as London's. There is no fallback on this path now.
    """
    fake_get.route(
        "/search",
        FakeResponse(
            {
                "address": {
                    "city": "London",
                    "state": "England",
                    "country": "United Kingdom",
                    "country_code": "gb",
                }
            }
        ),
    )

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.forward_geocode("London")

    assert exc_info.value.code == location_service.GEOCODING_MALFORMED_RESPONSE


async def test_reverse_geocode_still_falls_back_to_the_queried_coordinates(fake_get):
    """
    The fallback is correct HERE: the caller's own position is the right answer
    when the echo is unusable. Only forward_geocode has nothing to fall back to.
    """
    fake_get.route(
        "/reverse",
        FakeResponse(reverse_payload(BENGALURU_ADDRESS, lat="not-a-number", lon="")),
    )

    location = await location_service.reverse_geocode(LAT, LON)

    assert location["latitude"] == LAT
    assert location["longitude"] == LON
    assert location["city"] == "Bengaluru"


async def test_a_cache_hit_reports_the_second_callers_own_coordinates(fake_get):
    """
    Two users in one neighbourhood share a cache entry. The second must get
    back their own position, not the first caller's -- otherwise one user's
    exact coordinates travel into another user's database row and LLM context.
    """
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    first = await location_service.reverse_geocode(LAT, LON)
    # Same 3-decimal cache key, different actual position.
    nearby_lat, nearby_lon = LAT + 0.0002, LON + 0.0002
    second = await location_service.reverse_geocode(nearby_lat, nearby_lon)

    assert len(fake_get.calls_to("/reverse")) == 1, "expected a cache hit"
    assert first["latitude"] == LAT
    assert second["latitude"] == nearby_lat
    assert second["longitude"] == nearby_lon
    assert second["city"] == "Bengaluru"


async def test_a_failed_timezone_lookup_is_cached(monkeypatch, fake_get):
    """
    An Open-Meteo outage must be paid once, not on every geocode cache miss.

    Stacked on the Nominatim limiter wait, an uncached failure pushed
    POST /api/location past ten seconds -- for a field that is optional.
    """
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route("open-meteo.com", FakeResponse({}, status_code=500))

    assert await location_service.resolve_timezone(LAT, LON) is None
    assert await location_service.resolve_timezone(LAT, LON) is None

    assert len(fake_get.calls_to("open-meteo.com")) == 1


async def test_a_timezone_cache_hit_avoids_a_second_lookup(monkeypatch, fake_get):
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route("open-meteo.com", FakeResponse({"timezone": "Asia/Kolkata"}))

    assert await location_service.resolve_timezone(LAT, LON) == "Asia/Kolkata"
    assert await location_service.resolve_timezone(LAT, LON) == "Asia/Kolkata"

    assert len(fake_get.calls_to("open-meteo.com")) == 1


async def test_a_reverse_geocode_cache_hit_does_not_re_resolve_the_timezone(
    monkeypatch, fake_get
):
    """The cache entry is written after the timezone, so it carries one."""
    monkeypatch.setenv("LOCATION_TIMEZONE_LOOKUP", "true")
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))
    fake_get.route("open-meteo.com", FakeResponse({"timezone": "Asia/Kolkata"}))

    first = await location_service.reverse_geocode(LAT, LON)
    second = await location_service.reverse_geocode(LAT, LON)

    assert first["timezone"] == "Asia/Kolkata"
    assert second["timezone"] == "Asia/Kolkata"
    assert len(fake_get.calls_to("open-meteo.com")) == 1


async def test_a_full_rate_limit_queue_is_refused_rather_than_stalled(
    monkeypatch, fake_get
):
    """
    The 1 req/s budget is global, so without a ceiling one client issuing many
    distinct lookups would park every other user behind it for minutes. Past
    the bound a caller gets a fast, honest "rate limited" instead.
    """
    monkeypatch.setenv("NOMINATIM_MIN_INTERVAL", "5")
    monkeypatch.setenv("NOMINATIM_MAX_QUEUE_WAIT", "1")
    fake_get.route("/reverse", FakeResponse(reverse_payload(BENGALURU_ADDRESS)))

    # First call consumes the slot; the second must not wait 5 s for the next.
    await location_service.reverse_geocode(LAT, LON)

    with pytest.raises(location_service.LocationError) as exc_info:
        await location_service.reverse_geocode(50.0, 10.0)

    assert exc_info.value.code == location_service.GEOCODING_RATE_LIMITED


async def test_the_rate_limiter_raises_its_own_timeout(monkeypatch):
    """The limiter layer, independent of the service that translates it."""
    limiter = service_cache.AsyncRateLimiter(5.0)
    await limiter.acquire()

    with pytest.raises(service_cache.RateLimitTimeout):
        await limiter.acquire(max_wait=0.05)
