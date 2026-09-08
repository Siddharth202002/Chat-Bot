"""
HTTP surface: POST/GET/DELETE /api/location.

The app is driven in-process over ASGI with the authentication dependency
overridden, in the same shape as test_memory_api. The chatbot_backend facade
functions are stubbed *as imported into api_server* -- api_server does
``from chatbot_backend import (...)``, so the names live in api_server's own
namespace and that is the only place patching them has any effect.

What these tests are really protecting:
  * an out-of-range coordinate is a 422 from pydantic, before it can reach the
    geocoder and spend the shared Nominatim budget;
  * a geocoding failure maps to an HTTP status the frontend can act on, with a
    human message that leaks no key, URL or coordinates;
  * every endpoint sits behind the existing session auth.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import api_server
import location_service
import location_state
import service_cache
from conftest import apply_location_env

ALICE = {"id": "user-alice", "email": "alice@example.com"}

BENGALURU_RECORD = {
    "latitude": 12.97,
    "longitude": 77.59,
    "city": "Bengaluru",
    "state": "Karnataka",
    "country": "India",
    "country_code": "IN",
    "timezone": "Asia/Kolkata",
    "label": "Bengaluru, Karnataka, India",
    "source": location_state.SOURCE_BROWSER_GPS,
    "updated_at": "2024-05-30T12:00:00Z",
}

PUNE_RECORD = {
    **BENGALURU_RECORD,
    "latitude": 18.52,
    "longitude": 73.86,
    "city": "Pune",
    "state": "Maharashtra",
    "label": "Pune, Maharashtra, India",
    "source": location_state.SOURCE_MANUAL,
}


@pytest.fixture(autouse=True)
def clean_service_caches():
    service_cache.reset_for_tests()
    yield
    service_cache.reset_for_tests()


@pytest.fixture(autouse=True)
def location_env(monkeypatch):
    apply_location_env(monkeypatch)


@pytest.fixture
def as_user():
    """Authenticate the app as a given user for the duration of a test."""

    def _set(user: dict[str, str]) -> None:
        api_server.app.dependency_overrides[api_server.current_user] = lambda: user

    yield _set
    api_server.app.dependency_overrides.clear()


def client() -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=api_server.app), base_url="http://testserver"
    )


class Facade:
    """
    Recording stand-ins for the five chatbot_backend functions api_server calls.

    Every one of them is stubbed for every test, so no request in this file can
    reach a geocoder or the application database.
    """

    def __init__(self) -> None:
        self.coordinate_calls: list[tuple] = []
        self.city_calls: list[tuple] = []
        self.failure_calls: list[tuple] = []
        self.state_calls: list[str] = []
        self.clear_calls: list[str] = []
        self.coordinate_result: object = BENGALURU_RECORD
        self.city_result: object = PUNE_RECORD
        self.state_result: tuple = (location_state.STATUS_NONE, None)
        self.coordinate_error: Exception | None = None
        self.city_error: Exception | None = None
        self.cleared = True

    @property
    def geocoder_calls(self) -> int:
        return len(self.coordinate_calls) + len(self.city_calls)


@pytest.fixture
def facade(monkeypatch) -> Facade:
    state = Facade()

    async def resolve_user_coordinates(user_id, latitude, longitude):
        state.coordinate_calls.append((user_id, latitude, longitude))
        if state.coordinate_error is not None:
            raise state.coordinate_error
        return state.coordinate_result

    async def resolve_user_city(user_id, city):
        state.city_calls.append((user_id, city))
        if state.city_error is not None:
            raise state.city_error
        return state.city_result

    async def record_location_failure(user_id, status):
        state.failure_calls.append((user_id, status))
        return status

    async def get_user_location_state(user_id):
        state.state_calls.append(user_id)
        return state.state_result

    async def clear_user_location(user_id):
        state.clear_calls.append(user_id)
        return state.cleared

    monkeypatch.setattr(api_server, "resolve_user_coordinates", resolve_user_coordinates)
    monkeypatch.setattr(api_server, "resolve_user_city", resolve_user_city)
    monkeypatch.setattr(api_server, "record_location_failure", record_location_failure)
    monkeypatch.setattr(api_server, "get_user_location_state", get_user_location_state)
    monkeypatch.setattr(api_server, "clear_user_location", clear_user_location)
    return state


# --------------------------------------------------------------------------
# POST /api/location -- coordinates
# --------------------------------------------------------------------------

async def test_posting_coordinates_stores_and_echoes_the_location(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.post(
            "/api/location", json={"latitude": 12.971598, "longitude": 77.594622}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["location"]["city"] == "Bengaluru"
    assert body["location"]["label"] == "Bengaluru, Karnataka, India"
    # The authenticated id is used, never anything from the request body.
    assert facade.coordinate_calls == [(ALICE["id"], 12.971598, 77.594622)]


@pytest.mark.parametrize(
    "payload",
    [
        {"latitude": 91, "longitude": 0},
        {"latitude": -91, "longitude": 0},
        {"latitude": 0, "longitude": 181},
        {"latitude": 0, "longitude": -181},
        {"latitude": "north", "longitude": 0},
        {"latitude": None, "longitude": 0, "city": "x" * 201},
    ],
)
async def test_an_out_of_range_coordinate_is_rejected_before_the_geocoder(
    as_user, facade, payload
):
    """
    422 from pydantic, not a 502 from upstream: an impossible coordinate must
    never consume the shared Nominatim rate-limit budget.
    """
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json=payload)

    assert response.status_code == 422
    assert facade.geocoder_calls == 0


@pytest.mark.parametrize(
    "payload", [{"latitude": 90, "longitude": 180}, {"latitude": -90, "longitude": -180}]
)
async def test_the_coordinate_boundaries_are_accepted(as_user, facade, payload):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json=payload)

    assert response.status_code == 200
    assert facade.geocoder_calls == 1


async def test_a_latitude_without_a_longitude_is_a_400(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={"latitude": 12.97})

    assert response.status_code == 400
    assert facade.geocoder_calls == 0


# --------------------------------------------------------------------------
# POST /api/location -- manual city
# --------------------------------------------------------------------------

async def test_posting_a_city_uses_the_manual_path(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={"city": "Pune"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["location"]["city"] == "Pune"
    # Recorded as manual so the agent says "the city you gave me" rather than
    # implying it read the device's GPS.
    assert body["location"]["source"] == location_state.SOURCE_MANUAL
    assert facade.city_calls == [(ALICE["id"], "Pune")]
    assert facade.coordinate_calls == []


async def test_a_city_is_trimmed_before_it_reaches_the_geocoder(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        await http.post("/api/location", json={"city": "  Pune  "})

    assert facade.city_calls == [(ALICE["id"], "Pune")]


async def test_a_city_wins_over_coordinates(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.post(
            "/api/location", json={"city": "Pune", "latitude": 12.97, "longitude": 77.59}
        )

    assert response.status_code == 200
    assert facade.city_calls == [(ALICE["id"], "Pune")]
    assert facade.coordinate_calls == []


async def test_a_blank_city_with_no_coordinates_is_a_400(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={"city": "   "})

    assert response.status_code == 400
    assert facade.geocoder_calls == 0


# --------------------------------------------------------------------------
# POST /api/location -- empty and status payloads
# --------------------------------------------------------------------------

async def test_an_empty_payload_explains_what_is_needed(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Provide latitude and longitude, a city, or a status."
    )
    assert facade.geocoder_calls == 0


@pytest.mark.parametrize("status", sorted(location_state.FAILURE_STATUSES))
async def test_posting_a_failure_status_records_it(as_user, facade, status):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={"status": status})

    assert response.status_code == 200
    assert response.json() == {"status": status, "location": None}
    assert facade.failure_calls == [(ALICE["id"], status)]


@pytest.mark.parametrize("status", ["nonsense", "ready", "none", "", "DENIED!"])
async def test_an_unknown_status_is_rejected(as_user, facade, status):
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={"status": status})

    assert response.status_code == 422
    assert facade.failure_calls == []


async def test_a_status_wins_over_coordinates(as_user, facade):
    """
    Precedence matters: the browser posts a status *instead of* a position, and
    a stale pair in the same body must not be geocoded behind the refusal.
    """
    as_user(ALICE)
    async with client() as http:
        response = await http.post(
            "/api/location", json={"status": "denied", "latitude": 1, "longitude": 1}
        )

    assert response.status_code == 200
    assert response.json() == {"status": "denied", "location": None}
    assert facade.geocoder_calls == 0


# --------------------------------------------------------------------------
# POST /api/location -- LocationError mapping
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code,expected_status",
    [
        (location_service.LOCATION_NOT_FOUND, 404),
        (location_service.GEOCODING_TIMEOUT, 504),
        (location_service.GEOCODING_RATE_LIMITED, 429),
        (location_service.GEOCODING_UNAVAILABLE, 502),
        (location_service.GEOCODING_MALFORMED_RESPONSE, 502),
        (location_service.INVALID_COORDINATES, 400),
    ],
)
async def test_a_geocoding_error_maps_to_an_http_status(
    as_user, facade, code, expected_status
):
    as_user(ALICE)
    facade.coordinate_error = location_service.LocationError(
        code, "The location lookup service is unreachable."
    )
    async with client() as http:
        response = await http.post(
            "/api/location", json={"latitude": 12.97, "longitude": 77.59}
        )

    assert response.status_code == expected_status


async def test_an_unmapped_error_code_falls_through_to_502(as_user, facade):
    as_user(ALICE)
    facade.coordinate_error = location_service.LocationError(
        "SOMETHING_NEW", "The location lookup service is unreachable."
    )
    async with client() as http:
        response = await http.post(
            "/api/location", json={"latitude": 12.97, "longitude": 77.59}
        )

    assert response.status_code == 502


async def test_an_error_detail_leaks_no_key_url_or_coordinates(as_user, facade):
    as_user(ALICE)
    facade.coordinate_error = location_service.LocationError(
        location_service.GEOCODING_UNAVAILABLE,
        "The location lookup service is unreachable.",
    )
    async with client() as http:
        response = await http.post(
            "/api/location", json={"latitude": 12.971598, "longitude": 77.594622}
        )

    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert detail == "The location lookup service is unreachable."
    assert "test-key" not in detail
    assert "nominatim" not in detail.lower()
    assert "openweathermap" not in detail.lower()
    assert "http" not in detail.lower()
    assert "12.97" not in detail
    assert "77.59" not in detail


async def test_a_city_geocoding_error_maps_the_same_way(as_user, facade):
    as_user(ALICE)
    facade.city_error = location_service.LocationError(
        location_service.LOCATION_NOT_FOUND, "No place matching that name was found."
    )
    async with client() as http:
        response = await http.post("/api/location", json={"city": "Atlantis"})

    assert response.status_code == 404
    assert response.json()["detail"] == "No place matching that name was found."


# --------------------------------------------------------------------------
# GET /api/location
# --------------------------------------------------------------------------

async def test_getting_a_stored_location(as_user, facade):
    as_user(ALICE)
    facade.state_result = (location_state.STATUS_READY, BENGALURU_RECORD)
    async with client() as http:
        response = await http.get("/api/location")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["location"]["city"] == "Bengaluru"
    assert body["location"]["timezone"] == "Asia/Kolkata"
    assert facade.state_calls == [ALICE["id"]]


async def test_getting_an_absent_location(as_user, facade):
    as_user(ALICE)
    facade.state_result = (location_state.STATUS_NONE, None)
    async with client() as http:
        response = await http.get("/api/location")

    assert response.status_code == 200
    assert response.json() == {"status": "none", "location": None}


@pytest.mark.parametrize("status", sorted(location_state.FAILURE_STATUSES))
async def test_getting_a_recorded_failure(as_user, facade, status):
    """The UI reads this to know not to re-prompt for browser permission."""
    as_user(ALICE)
    facade.state_result = (status, None)
    async with client() as http:
        response = await http.get("/api/location")

    assert response.status_code == 200
    assert response.json() == {"status": status, "location": None}


# --------------------------------------------------------------------------
# DELETE /api/location
# --------------------------------------------------------------------------

async def test_deleting_the_stored_location(as_user, facade):
    as_user(ALICE)
    async with client() as http:
        response = await http.delete("/api/location")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert facade.clear_calls == [ALICE["id"]]


async def test_deleting_when_nothing_was_stored_is_still_ok(as_user, facade):
    """"Stop sharing" is idempotent; the UI should not have to check first."""
    as_user(ALICE)
    facade.cleared = False
    async with client() as http:
        response = await http.delete("/api/location")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

async def test_unauthenticated_location_requests_are_rejected(facade):
    """All three endpoints follow the existing session-cookie auth."""
    api_server.app.dependency_overrides.clear()
    async with client() as http:
        assert (
            await http.post("/api/location", json={"latitude": 12.97, "longitude": 77.59})
        ).status_code == 401
        assert (await http.get("/api/location")).status_code == 401
        assert (await http.delete("/api/location")).status_code == 401

    assert facade.geocoder_calls == 0
    assert facade.state_calls == []
    assert facade.clear_calls == []


async def test_an_invalid_session_cookie_is_rejected(facade):
    api_server.app.dependency_overrides.clear()
    async with client() as http:
        response = await http.get(
            "/api/location", cookies={"zeno_session": "not.a.real.token"}
        )

    assert response.status_code == 401
    assert facade.state_calls == []


# --------------------------------------------------------------------------
# Regressions from the code review
# --------------------------------------------------------------------------

async def test_one_coordinate_without_the_other_is_rejected(as_user, facade):
    """Half a coordinate pair is a client bug, not a geocodable location."""
    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/location", json={"latitude": 12.97})

    assert response.status_code == 400
    assert "both latitude and longitude" in response.json()["detail"].lower()
    assert facade.geocoder_calls == 0


async def test_unavailable_storage_is_a_503_not_a_500(as_user, facade, monkeypatch):
    """
    A misconfigured or unopened database should say so, not leak a traceback.

    Previously LocationStateUnavailable was uncaught on all three endpoints and
    surfaced as an unhandled 500.
    """
    async def unavailable(*args, **kwargs):
        raise location_state.LocationStateUnavailable("not configured")

    monkeypatch.setattr(api_server, "resolve_user_coordinates", unavailable)
    monkeypatch.setattr(api_server, "get_user_location_state", unavailable)
    monkeypatch.setattr(api_server, "clear_user_location", unavailable)
    as_user(ALICE)

    async with client() as http:
        post = await http.post(
            "/api/location", json={"latitude": 12.97, "longitude": 77.59}
        )
        get = await http.get("/api/location")
        delete = await http.delete("/api/location")

    for response in (post, get, delete):
        assert response.status_code == 503
        assert response.json()["detail"] == "Location storage is unavailable."


async def test_a_reported_failure_wins_over_coordinates(as_user, facade):
    """
    The client is telling us it could not locate the user; believe it.

    Anything else would store a position the browser has already disowned.
    """
    as_user(ALICE)
    async with client() as http:
        response = await http.post(
            "/api/location",
            json={"status": "denied", "latitude": 12.97, "longitude": 77.59},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "denied", "location": None}
    assert facade.geocoder_calls == 0
    assert facade.failure_calls == [(ALICE["id"], "denied")]
