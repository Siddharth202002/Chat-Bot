"""
Per-user location persistence.

Two product rules drive every assertion here:

  * precise GPS coordinates are never written to disk -- the row holds a
    coarsened pair, and the precise reading is used for the lookup and dropped;
  * a stored location expires, so a stale position never follows a user to
    another city and a recorded permission denial eventually lapses.

The store is pointed at a temp SQLite file, in the same shape as the shared
``db`` fixture in conftest, so nothing touches the application database.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

import location_state
from conftest import apply_location_env

ALICE = "user-alice"
BOB = "user-bob"

# A browser-grade reading: six decimals, roughly a doorstep.
PRECISE_LAT = 12.971598
PRECISE_LON = 77.594622

BENGALURU = {
    "latitude": PRECISE_LAT,
    "longitude": PRECISE_LON,
    "city": "Bengaluru",
    "state": "Karnataka",
    "country": "India",
    "country_code": "IN",
    "timezone": "Asia/Kolkata",
    "label": "Bengaluru, Karnataka, India",
}

LONDON = {
    "latitude": 51.507445,
    "longitude": -0.127765,
    "city": "London",
    "state": "England",
    "country": "United Kingdom",
    "country_code": "GB",
    "timezone": "Europe/London",
    "label": "London, England, United Kingdom",
}


@pytest.fixture(autouse=True)
def location_env(monkeypatch):
    apply_location_env(monkeypatch)


@pytest.fixture
async def location_db(tmp_path: Path):
    """
    A temp database with the location store wired to it.

    The original provider (chatbot_backend installs one at import time, pointing
    at the real chat_memory.db) is put back in teardown, so a test in this file
    can never leave the store aimed at a closed temp connection.
    """
    original_provider = location_state._connection_provider
    conn = await aiosqlite.connect(str(tmp_path / "test_location.db"))

    async def connection_provider():
        return conn

    location_state.configure(connection_provider=connection_provider)
    try:
        yield conn
    finally:
        location_state.reset_for_tests()
        if original_provider is not None:
            location_state.configure(connection_provider=original_provider)
        await conn.close()


async def stored_row(conn, user_id: str = ALICE):
    """Read the raw row, to assert on what is actually on disk."""
    async with conn.execute(
        """
        SELECT status, error_code, latitude, longitude, city, state, country,
               country_code, timezone, label, source, updated_at, expires_at
        FROM user_locations WHERE user_id = ?
        """,
        (user_id,),
    ) as cursor:
        return await cursor.fetchone()


async def row_count(conn) -> int:
    async with conn.execute("SELECT COUNT(*) FROM user_locations") as cursor:
        return (await cursor.fetchone())[0]


async def write_raw_row(conn, user_id: str, **columns) -> None:
    """Hand-write a row, for states the public API will not produce."""
    await location_state.ensure_location_tables()
    values = {
        "status": location_state.STATUS_READY,
        "error_code": None,
        "latitude": 12.97,
        "longitude": 77.59,
        "city": "Bengaluru",
        "state": "Karnataka",
        "country": "India",
        "country_code": "IN",
        "timezone": "Asia/Kolkata",
        "label": "Bengaluru, Karnataka, India",
        "source": location_state.SOURCE_BROWSER_GPS,
        "updated_at": iso(minutes_from_now(0)),
        "expires_at": iso(minutes_from_now(60)),
    }
    values.update(columns)
    await conn.execute(
        """
        INSERT OR REPLACE INTO user_locations (
            user_id, status, error_code, latitude, longitude, city, state,
            country, country_code, timezone, label, source, updated_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, *values.values()),
    )
    await conn.commit()


def minutes_from_now(minutes: float) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=minutes)


def iso(value: datetime) -> str:
    return value.isoformat() + "Z"


# --------------------------------------------------------------------------
# save_location / get_state round trip
# --------------------------------------------------------------------------

async def test_a_saved_location_reads_back_as_ready(location_db):
    await location_state.save_location(ALICE, BENGALURU)

    status, record = await location_state.get_state(ALICE)

    assert status == location_state.STATUS_READY
    assert record is not None
    assert record["city"] == "Bengaluru"
    assert record["state"] == "Karnataka"
    assert record["country"] == "India"
    assert record["country_code"] == "IN"
    assert record["timezone"] == "Asia/Kolkata"
    assert record["label"] == "Bengaluru, Karnataka, India"
    assert record["source"] == location_state.SOURCE_BROWSER_GPS
    assert record["updated_at"]


async def test_a_manual_city_records_its_source(location_db):
    """The agent says "the city you gave me" rather than implying it read GPS."""
    await location_state.save_location(ALICE, LONDON, source=location_state.SOURCE_MANUAL)

    _, record = await location_state.get_state(ALICE)

    assert record["source"] == location_state.SOURCE_MANUAL


async def test_save_location_returns_what_was_stored(location_db):
    record = await location_state.save_location(ALICE, BENGALURU)
    _, read_back = await location_state.get_state(ALICE)

    # The API response and the tool must never disagree with the row.
    assert record["latitude"] == read_back["latitude"]
    assert record["longitude"] == read_back["longitude"]


async def test_a_location_without_a_label_gets_a_placeholder(location_db):
    await location_state.save_location(
        ALICE, {**BENGALURU, "label": None, "city": None, "state": None, "country": None}
    )

    _, record = await location_state.get_state(ALICE)

    assert record["label"] == "Unknown location"


async def test_save_location_requires_a_user_id(location_db):
    with pytest.raises(ValueError):
        await location_state.save_location("", BENGALURU)


# --------------------------------------------------------------------------
# Coordinate coarsening
# --------------------------------------------------------------------------

async def test_stored_coordinates_are_coarsened(location_db):
    """
    The privacy requirement: at LOCATION_STORE_PRECISION=2 the row holds ~1.1 km
    of resolution -- enough for a weather lookup and a city name, not enough to
    identify a building. The precise browser reading must not survive the write.
    """
    record = await location_state.save_location(ALICE, BENGALURU)

    assert record["latitude"] == 12.97
    assert record["longitude"] == 77.59

    row = await stored_row(location_db)
    assert row[2] == 12.97
    assert row[3] == 77.59
    assert row[2] != PRECISE_LAT
    assert row[3] != PRECISE_LON


async def test_the_precise_reading_is_not_anywhere_in_the_row(location_db):
    await location_state.save_location(ALICE, BENGALURU)

    row = await stored_row(location_db)

    rendered = " ".join("" if value is None else str(value) for value in row)
    assert "12.971598" not in rendered
    assert "77.594622" not in rendered


async def test_coarsening_follows_the_configured_precision(monkeypatch, location_db):
    monkeypatch.setenv("LOCATION_STORE_PRECISION", "1")

    record = await location_state.save_location(ALICE, BENGALURU)

    assert record["latitude"] == 13.0
    assert record["longitude"] == 77.6


# --------------------------------------------------------------------------
# Upsert
# --------------------------------------------------------------------------

async def test_saving_twice_upserts_a_single_row(location_db):
    await location_state.save_location(ALICE, BENGALURU)
    await location_state.save_location(ALICE, LONDON)

    assert await row_count(location_db) == 1

    _, record = await location_state.get_state(ALICE)
    assert record["city"] == "London"
    assert record["latitude"] == 51.51
    assert record["timezone"] == "Europe/London"


async def test_a_ready_location_clears_an_earlier_failure(location_db):
    await location_state.save_failure(ALICE, "denied")
    await location_state.save_location(ALICE, BENGALURU)

    status, record = await location_state.get_state(ALICE)

    assert status == location_state.STATUS_READY
    assert record["city"] == "Bengaluru"
    assert (await stored_row(location_db))[1] is None  # error_code


# --------------------------------------------------------------------------
# Failure statuses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", sorted(location_state.FAILURE_STATUSES))
async def test_a_recorded_failure_reads_back_as_that_status(location_db, status):
    recorded = await location_state.save_failure(ALICE, status)

    assert recorded == status
    assert await location_state.get_state(ALICE) == (status, None)


@pytest.mark.parametrize("status", ["DENIED", " denied ", "Timeout"])
async def test_a_failure_status_is_normalised(location_db, status):
    recorded = await location_state.save_failure(ALICE, status)
    assert recorded == status.strip().lower()


@pytest.mark.parametrize("status", ["nonsense", "ready", "none", "", None, "  "])
async def test_an_unknown_failure_status_is_rejected(location_db, status):
    with pytest.raises(ValueError):
        await location_state.save_failure(ALICE, status)


async def test_save_failure_requires_a_user_id(location_db):
    with pytest.raises(ValueError):
        await location_state.save_failure("", "denied")


async def test_a_failure_after_a_ready_location_clears_the_coordinates(location_db):
    """
    A denial must not leave the previous position readable. The row keeps the
    status so the agent can explain itself, and nothing else.
    """
    await location_state.save_location(ALICE, BENGALURU)
    await location_state.save_failure(ALICE, "denied")

    row = await stored_row(location_db)

    assert row[0] == "denied"
    assert row[2] is None  # latitude
    assert row[3] is None  # longitude
    assert row[4] is None  # city
    assert row[9] is None  # label
    assert await location_state.get_state(ALICE) == ("denied", None)


# --------------------------------------------------------------------------
# Expiry
# --------------------------------------------------------------------------

async def test_an_expired_ready_row_reads_as_none(monkeypatch, location_db):
    monkeypatch.setenv("LOCATION_TTL_SECONDS", "60")
    await write_raw_row(location_db, ALICE, expires_at=iso(minutes_from_now(-1)))

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_an_expired_failure_row_reads_as_none(monkeypatch, location_db):
    """A denial does eventually lapse, so the frontend may ask again one day."""
    monkeypatch.setenv("LOCATION_TTL_SECONDS", "60")
    await write_raw_row(
        location_db,
        ALICE,
        status="denied",
        error_code="denied",
        latitude=None,
        longitude=None,
        city=None,
        state=None,
        country=None,
        country_code=None,
        timezone=None,
        label=None,
        source=None,
        expires_at=iso(minutes_from_now(-1)),
    )

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_a_row_with_an_unparseable_expiry_reads_as_none(location_db):
    await write_raw_row(location_db, ALICE, expires_at="not a timestamp")

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_a_freshly_written_row_is_still_within_its_ttl(location_db):
    await location_state.save_location(ALICE, BENGALURU)

    status, _ = await location_state.get_state(ALICE)

    assert status == location_state.STATUS_READY


# --------------------------------------------------------------------------
# Corrupt rows
# --------------------------------------------------------------------------

async def test_a_ready_row_without_coordinates_reads_as_none(location_db):
    """Better absent than handing the weather service a None latitude."""
    await write_raw_row(location_db, ALICE, latitude=None, longitude=None)

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_a_ready_row_missing_only_the_longitude_reads_as_none(location_db):
    await write_raw_row(location_db, ALICE, longitude=None)

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_an_unrecognised_status_reads_as_none(location_db):
    await write_raw_row(location_db, ALICE, status="something-new")

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_a_ready_row_without_a_label_gets_a_placeholder(location_db):
    await write_raw_row(location_db, ALICE, label=None, source=None)

    status, record = await location_state.get_state(ALICE)

    assert status == location_state.STATUS_READY
    assert record["label"] == "Unknown location"
    assert record["source"] == location_state.SOURCE_BROWSER_GPS


# --------------------------------------------------------------------------
# clear_location
# --------------------------------------------------------------------------

async def test_clear_location_removes_the_row(location_db):
    await location_state.save_location(ALICE, BENGALURU)

    assert await location_state.clear_location(ALICE) is True
    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)
    assert await row_count(location_db) == 0


async def test_clearing_a_missing_row_reports_nothing_removed(location_db):
    assert await location_state.clear_location(ALICE) is False


async def test_clearing_a_recorded_denial_also_works(location_db):
    await location_state.save_failure(ALICE, "denied")

    assert await location_state.clear_location(ALICE) is True
    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_clear_location_without_a_user_id_is_a_no_op(location_db):
    await location_state.save_location(ALICE, BENGALURU)

    assert await location_state.clear_location("") is False
    assert await row_count(location_db) == 1


# --------------------------------------------------------------------------
# Isolation and empty state
# --------------------------------------------------------------------------

async def test_two_users_locations_never_leak_into_each_other(location_db):
    await location_state.save_location(ALICE, BENGALURU)
    await location_state.save_location(BOB, LONDON)

    _, alice = await location_state.get_state(ALICE)
    _, bob = await location_state.get_state(BOB)

    assert alice["city"] == "Bengaluru"
    assert bob["city"] == "London"


async def test_clearing_one_user_leaves_the_other_alone(location_db):
    await location_state.save_location(ALICE, BENGALURU)
    await location_state.save_location(BOB, LONDON)

    await location_state.clear_location(ALICE)

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)
    assert (await location_state.get_state(BOB))[0] == location_state.STATUS_READY


async def test_one_users_denial_does_not_affect_another(location_db):
    await location_state.save_location(BOB, LONDON)
    await location_state.save_failure(ALICE, "denied")

    assert await location_state.get_state(ALICE) == ("denied", None)
    assert (await location_state.get_state(BOB))[0] == location_state.STATUS_READY


async def test_an_empty_user_id_reads_as_none(location_db):
    assert await location_state.get_state("") == (location_state.STATUS_NONE, None)


async def test_a_fresh_database_reads_as_none(location_db):
    """get_state creates the table on the way in, so this must not raise."""
    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)
    assert await row_count(location_db) == 0


async def test_an_unconfigured_store_raises_rather_than_guessing(location_db):
    location_state.reset_for_tests()

    with pytest.raises(location_state.LocationStateUnavailable):
        await location_state.get_state(ALICE)


async def test_a_none_connection_raises_rather_than_guessing(location_db):
    async def no_connection():
        return None

    location_state.configure(connection_provider=no_connection)

    with pytest.raises(location_state.LocationStateUnavailable):
        await location_state.get_state(ALICE)


# --------------------------------------------------------------------------
# Regressions from the code review
# --------------------------------------------------------------------------

async def test_a_denial_outlives_a_short_coordinate_ttl(location_db, monkeypatch):
    """
    "This position is stale" and "this user said no" are different clocks.

    They used to share LOCATION_TTL_SECONDS, so a refusal was forgotten every
    hour and the browser prompted for permission again -- the behaviour users
    read as a bug. The denial now expires on its own, much longer clock.
    """
    monkeypatch.setenv("LOCATION_TTL_SECONDS", "60")
    monkeypatch.setenv("LOCATION_DENIAL_TTL_SECONDS", "86400")

    await location_state.save_failure(ALICE, "denied")

    # Move well past the coordinate TTL but nowhere near the denial's.
    async with location_db.execute(
        "SELECT expires_at FROM user_locations WHERE user_id = ?", (ALICE,)
    ) as cursor:
        row = await cursor.fetchone()
    expires_at = datetime.fromisoformat(str(row[0]).rstrip("Z"))
    assert expires_at - datetime.utcnow() > timedelta(hours=12)

    status, record = await location_state.get_state(ALICE)
    assert status == "denied"
    assert record is None


async def test_a_denial_still_eventually_expires(location_db, monkeypatch):
    """Respectful, but not permanent: someone who changes their mind is not stuck."""
    monkeypatch.setenv("LOCATION_DENIAL_TTL_SECONDS", "60")
    await location_state.save_failure(ALICE, "denied")

    past = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z"
    await location_db.execute(
        "UPDATE user_locations SET expires_at = ? WHERE user_id = ?", (past, ALICE)
    )
    await location_db.commit()

    assert await location_state.get_state(ALICE) == (location_state.STATUS_NONE, None)


async def test_concurrent_schema_checks_create_the_table_once(location_db):
    """
    ensure_location_tables is called on every request, so its double-checked
    lock has to hold under concurrency rather than racing on the ready flag.
    """
    location_state._tables_ready = False

    await asyncio.gather(*(location_state.ensure_location_tables() for _ in range(8)))

    async with location_db.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type = 'table' AND name = 'user_locations'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1
