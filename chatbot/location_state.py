"""
Per-user location state.

The user's resolved location lives in the project's existing SQLite database
(chat_memory.db), in one row per user alongside the users, chat_threads and
long-term memory tables. No second state system is introduced. The connection
is injected at startup -- this module never imports chatbot_backend, which
keeps the import graph acyclic and lets tests point it at a temp database.

It is deliberately wired to the *request-path* connection (the one the users
and chat_threads tables use), not to the background memory connection: a
SQLite commit() commits everything pending on its connection, so committing a
location upsert on the memory connection could cut one of memory_store's
multi-row batches in half.

Privacy shape of the row, driven by the product rule "do not permanently store
precise GPS coordinates":

  * coordinates are rounded to LOCATION_STORE_PRECISION decimals (default 2,
    about 1.1 km) before they are written. The precise browser reading is used
    for the lookup and then dropped.
  * every row carries an expiry (LOCATION_TTL_SECONDS). An expired row is
    treated as "no location", so a stale position never follows a user to
    another city, and a signed-out user leaves nothing usable behind.
  * a permission failure is recorded as a *status*, not as coordinates, so the
    agent can explain the denial without the frontend re-prompting. It expires
    on its own, much longer clock (LOCATION_DENIAL_TTL_SECONDS): "this position
    is stale" and "this user said no" have nothing to do with each other, and
    sharing one knob meant a refusal was forgotten every hour and the browser
    prompted again.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypedDict

import location_config

logger = logging.getLogger("chatbot.location.state")

ConnectionProvider = Callable[[], Awaitable[Any]]

# Statuses a row can hold. "ready" is the only one with a location attached.
STATUS_READY = "ready"
STATUS_NONE = "none"
FAILURE_STATUSES: frozenset[str] = frozenset(
    {"denied", "timeout", "unavailable", "unsupported"}
)

# Where a location came from. Surfaced to the agent so it can say "the city you
# gave me" rather than implying it read the device's GPS.
SOURCE_BROWSER_GPS = "browser_gps"
SOURCE_MANUAL = "manual"

_connection_provider: ConnectionProvider | None = None
_tables_ready = False

# Serializes the schema check, matching memory_store.ensure_memory_tables. The
# lock is created lazily per event loop: an asyncio.Lock binds permanently to
# the first loop that awaits it, so a module-level singleton raises "bound to a
# different event loop" as soon as a second loop touches it -- which is exactly
# what a pytest run does.
_loop_locks: dict[str, tuple[Any, asyncio.Lock]] = {}


def _lock(name: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    bound = _loop_locks.get(name)
    if bound is None or bound[0] is not loop:
        lock = asyncio.Lock()
        _loop_locks[name] = (loop, lock)
        return lock
    return bound[1]


class StoredLocation(TypedDict):
    latitude: float
    longitude: float
    city: str | None
    state: str | None
    country: str | None
    country_code: str | None
    timezone: str | None
    label: str
    source: str
    updated_at: str


class LocationStateUnavailable(RuntimeError):
    """Raised when the store was never wired up to a database connection."""


def configure(*, connection_provider: ConnectionProvider) -> None:
    """Inject the shared SQLite connection provider."""
    global _connection_provider, _tables_ready
    _connection_provider = connection_provider
    _tables_ready = False


def reset_for_tests() -> None:
    global _connection_provider, _tables_ready
    _connection_provider = None
    _tables_ready = False
    _loop_locks.clear()


async def _connection() -> Any:
    if _connection_provider is None:
        raise LocationStateUnavailable("Location state store is not configured.")
    conn = await _connection_provider()
    if conn is None:
        raise LocationStateUnavailable("Database connection is unavailable.")
    return conn


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(value: datetime) -> str:
    return value.isoformat() + "Z"


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().rstrip("Z"))
    except ValueError:
        return None


async def ensure_location_tables() -> None:
    """Create the location table once. Safe to call on every request."""
    global _tables_ready
    if _tables_ready:
        return
    async with _lock("schema"):
        # Double-checked: the first caller through creates the table while any
        # concurrent callers wait here, then see the flag and return.
        if _tables_ready:
            return
        await _create_location_table()
        _tables_ready = True


async def _create_location_table() -> None:
    conn = await _connection()
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_locations (
            user_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            error_code TEXT,
            latitude REAL,
            longitude REAL,
            city TEXT,
            state TEXT,
            country TEXT,
            country_code TEXT,
            timezone TEXT,
            label TEXT,
            source TEXT,
            updated_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )
    await conn.commit()


async def save_location(
    user_id: str,
    location: dict[str, Any],
    *,
    source: str = SOURCE_BROWSER_GPS,
) -> StoredLocation:
    """
    Upsert a resolved location for a user and return what was stored.

    The returned record is the coarsened one that went into the database, not
    the caller's input, so the API response and the tool never disagree with
    the row about where the user is.
    """
    if not user_id:
        raise ValueError("user_id is required.")

    precision = location_config.location_store_precision()
    latitude = round(float(location["latitude"]), precision)
    longitude = round(float(location["longitude"]), precision)
    now = _utc_now()
    expires_at = now + timedelta(seconds=location_config.location_ttl_seconds())

    record = StoredLocation(
        latitude=latitude,
        longitude=longitude,
        city=location.get("city"),
        state=location.get("state"),
        country=location.get("country"),
        country_code=location.get("country_code"),
        timezone=location.get("timezone"),
        label=location.get("label") or "Unknown location",
        source=source,
        updated_at=_iso(now),
    )

    await ensure_location_tables()
    conn = await _connection()
    await conn.execute(
        """
        INSERT INTO user_locations (
            user_id, status, error_code, latitude, longitude, city, state,
            country, country_code, timezone, label, source, updated_at, expires_at
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            status = excluded.status,
            error_code = NULL,
            latitude = excluded.latitude,
            longitude = excluded.longitude,
            city = excluded.city,
            state = excluded.state,
            country = excluded.country,
            country_code = excluded.country_code,
            timezone = excluded.timezone,
            label = excluded.label,
            source = excluded.source,
            updated_at = excluded.updated_at,
            expires_at = excluded.expires_at
        """,
        (
            user_id,
            STATUS_READY,
            record["latitude"],
            record["longitude"],
            record["city"],
            record["state"],
            record["country"],
            record["country_code"],
            record["timezone"],
            record["label"],
            record["source"],
            record["updated_at"],
            _iso(expires_at),
        ),
    )
    await conn.commit()
    # The label is the coarsest thing worth logging here. Coordinates are
    # deliberately left out: this line would otherwise put a user's position in
    # every operator's log aggregator.
    logger.info(
        "Stored location for user %s (source=%s, resolved=%s)",
        user_id,
        source,
        bool(record["city"]),
    )
    return record


async def save_failure(user_id: str, status: str) -> str:
    """
    Record that the browser could not supply a location.

    Stored as a status rather than an absence so the agent can distinguish
    "the user refused" (explain, offer a city) from "we never asked yet"
    (the frontend will ask on the next location-shaped question). Persisting the
    refusal is also what stops a permission re-prompt loop.
    """
    if not user_id:
        raise ValueError("user_id is required.")
    normalized = (status or "").strip().lower()
    if normalized not in FAILURE_STATUSES:
        raise ValueError(f"Unsupported location status: {status!r}")

    now = _utc_now()
    expires_at = now + timedelta(
        seconds=location_config.location_denial_ttl_seconds()
    )
    await ensure_location_tables()
    conn = await _connection()
    await conn.execute(
        """
        INSERT INTO user_locations (
            user_id, status, error_code, latitude, longitude, city, state,
            country, country_code, timezone, label, source, updated_at, expires_at
        ) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            status = excluded.status,
            error_code = excluded.error_code,
            latitude = NULL,
            longitude = NULL,
            city = NULL,
            state = NULL,
            country = NULL,
            country_code = NULL,
            timezone = NULL,
            label = NULL,
            source = NULL,
            updated_at = excluded.updated_at,
            expires_at = excluded.expires_at
        """,
        (user_id, normalized, normalized, _iso(now), _iso(expires_at)),
    )
    await conn.commit()
    logger.info("Recorded location status '%s' for user %s", normalized, user_id)
    return normalized


async def get_state(user_id: str) -> tuple[str, StoredLocation | None]:
    """
    The user's current location state as ``(status, location)``.

    Status is "ready" with a location, one of FAILURE_STATUSES with None, or
    "none" when there is nothing stored or the row has expired. Expiry is
    evaluated here rather than by a cleanup job so a stale row can never be
    read as current.
    """
    if not user_id:
        return STATUS_NONE, None
    await ensure_location_tables()
    conn = await _connection()
    async with conn.execute(
        """
        SELECT status, latitude, longitude, city, state, country, country_code,
               timezone, label, source, updated_at, expires_at
        FROM user_locations WHERE user_id = ?
        """,
        (user_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        return STATUS_NONE, None

    expires_at = _parse_iso(row[11])
    if expires_at is None or expires_at <= _utc_now():
        return STATUS_NONE, None

    status = str(row[0] or STATUS_NONE)
    if status != STATUS_READY:
        return (status if status in FAILURE_STATUSES else STATUS_NONE), None

    if row[1] is None or row[2] is None:
        # A "ready" row with no coordinates is corrupt; treat it as absent
        # rather than handing the weather service a None.
        logger.warning("Discarding a ready location row with no coordinates.")
        return STATUS_NONE, None

    return STATUS_READY, StoredLocation(
        latitude=float(row[1]),
        longitude=float(row[2]),
        city=row[3],
        state=row[4],
        country=row[5],
        country_code=row[6],
        timezone=row[7],
        label=str(row[8] or "Unknown location"),
        source=str(row[9] or SOURCE_BROWSER_GPS),
        updated_at=str(row[10] or ""),
    )


async def clear_location(user_id: str) -> bool:
    """Forget everything stored for a user. Returns whether a row was removed."""
    if not user_id:
        return False
    await ensure_location_tables()
    conn = await _connection()
    cursor = await conn.execute(
        "DELETE FROM user_locations WHERE user_id = ?", (user_id,)
    )
    await conn.commit()
    return bool(getattr(cursor, "rowcount", 0))
