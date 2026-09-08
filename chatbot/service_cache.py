"""
Cache and rate-limit primitives for outbound third-party API calls.

Why this module exists
----------------------
Nominatim's public service is free but has a hard usage policy: at most one
request per second, and an identifying User-Agent. Blowing through either gets
the server's IP blocked, and a blocked IP is not something a retry loop can fix.
So every Nominatim call goes through a named rate limiter and a TTL cache here.

The abstraction is deliberately thin. The project has no Redis today, so the
default backend is an in-process bounded TTL dict. When Redis does arrive, the
business logic in location_service/weather_service does not change: call
``service_cache.set_backend_factory`` once at startup and the same
``get_cache("geocode")`` handles start hitting Redis instead.

Nothing here imports the application modules, so it stays usable from tests
without a database, an event loop already running, or any configuration.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from time import monotonic
from typing import Any, Callable, Protocol

logger = logging.getLogger("chatbot.services.cache")


class CacheBackend(Protocol):
    """The whole contract a cache has to satisfy to be swappable."""

    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def clear(self) -> None: ...


class InMemoryTTLCache:
    """
    Bounded, per-entry-TTL, async-safe cache.

    Entries are held in an OrderedDict so eviction is least-recently-used: a
    hot coordinate stays cached while a one-off lookup ages out. ``max_entries``
    is a memory ceiling, not a tuning knob -- reverse-geocode results are a few
    hundred bytes each, so the default holds well under a megabyte.

    Values are stored as-is (no copy). Callers must treat what they get back as
    read-only; both services here return freshly built dicts from the cached
    payload rather than handing the cached object to the caller.

    The methods are async to satisfy CacheBackend (a Redis implementation needs
    to await), but hold no lock: none of these bodies contains an await point,
    so on a single event loop each already runs to completion without
    interleaving. A lock here could never be contended.
    """

    def __init__(self, *, max_entries: int = 512) -> None:
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    async def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= monotonic():
            # Lazy expiry: an entry nobody asks for again costs one dict slot
            # until eviction, which is cheaper than a sweeper task.
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        self._entries[key] = (monotonic() + ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    async def clear(self) -> None:
        self._entries.clear()


BackendFactory = Callable[[str], CacheBackend]

_caches: dict[str, CacheBackend] = {}
_backend_factory: BackendFactory | None = None


def set_backend_factory(factory: BackendFactory | None) -> None:
    """
    Install the cache implementation for every named cache.

    Called at most once at startup (e.g. to return a Redis-backed cache). Any
    caches already handed out are dropped so the new backend takes effect.
    """
    global _backend_factory
    _backend_factory = factory
    _caches.clear()


def get_cache(name: str, *, max_entries: int = 512) -> CacheBackend:
    """The named cache, created on first use."""
    cache = _caches.get(name)
    if cache is None:
        if _backend_factory is not None:
            cache = _backend_factory(name)
        else:
            cache = InMemoryTTLCache(max_entries=max_entries)
        _caches[name] = cache
    return cache


async def clear_all_caches() -> None:
    """Drop every cached entry. Used by tests and by an operator-facing reset."""
    for cache in list(_caches.values()):
        try:
            await cache.clear()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to clear a service cache: %s", exc)


_rate_limiters: dict[str, "AsyncRateLimiter"] = {}


def reset_for_tests() -> None:
    """Forget the factory, every cache instance and every rate limiter."""
    global _backend_factory
    _backend_factory = None
    _caches.clear()
    _rate_limiters.clear()


class RateLimitTimeout(Exception):
    """Raised when a caller waited longer than it is willing to for its turn."""


class AsyncRateLimiter:
    """
    Serializes calls and enforces a minimum interval between them.

    Nominatim allows one request per second from a single application. That is
    a *global* budget, not per-user, so this limiter is shared by every request
    handler: two users asking "where am I?" at the same moment queue behind each
    other rather than both firing immediately.

    The lock is created lazily per event loop. An asyncio.Lock binds
    permanently to the first loop that awaits it, so a module-level singleton
    raises "bound to a different event loop" the moment a second loop touches
    it -- which is exactly what a pytest run does. Same reasoning as
    memory_store._lock.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = max(0.0, min_interval_seconds)
        self._loop_state: tuple[Any, asyncio.Lock, float] | None = None

    def _state(self) -> tuple[asyncio.Lock, float]:
        loop = asyncio.get_running_loop()
        state = self._loop_state
        if state is None or state[0] is not loop:
            self._loop_state = (loop, asyncio.Lock(), 0.0)
            state = self._loop_state
        return state[1], state[2]

    def _remember_call(self, at: float) -> None:
        if self._loop_state is not None:
            loop, lock, _ = self._loop_state
            self._loop_state = (loop, lock, at)

    def set_min_interval(self, min_interval_seconds: float) -> None:
        """Re-read the configured interval (config is env-driven per call)."""
        self._min_interval = max(0.0, min_interval_seconds)

    async def acquire(self, max_wait: float | None = None) -> None:
        """
        Wait until it is this caller's turn.

        Holds the lock for the whole wait, so N concurrent callers are spaced
        ``min_interval`` apart instead of all sleeping the same amount and then
        firing together.

        ``max_wait`` bounds the queue. The interval is a global budget, so
        without a bound one client issuing a few hundred distinct lookups would
        park every other user's request behind it for minutes, each holding an
        open connection. Past the bound the caller gets RateLimitTimeout, which
        the services translate into a "rate limited, try again" error -- a fast
        honest failure instead of an indefinite stall.
        """
        started = monotonic()
        lock, _ = self._state()
        try:
            if max_wait is None:
                await lock.acquire()
            else:
                await asyncio.wait_for(lock.acquire(), timeout=max_wait)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise RateLimitTimeout(
                "Timed out waiting for a rate-limited slot."
            ) from exc
        try:
            # Re-read inside the lock: the caller ahead of us in the queue
            # updated last_call while we were waiting for it.
            _, last_call = self._state()
            wait = self._min_interval - (monotonic() - last_call)
            if wait > 0:
                if max_wait is not None and (monotonic() - started) + wait > max_wait:
                    raise RateLimitTimeout(
                        "Timed out waiting for a rate-limited slot."
                    )
                await asyncio.sleep(wait)
            self._remember_call(monotonic())
        finally:
            lock.release()


def get_rate_limiter(name: str, min_interval_seconds: float) -> AsyncRateLimiter:
    """
    The named limiter, created on first use and re-tuned on every call.

    Re-tuning matters because the interval comes from the environment on each
    read (the memory_config convention), so a changed setting must not need a
    process restart to take effect.
    """
    limiter = _rate_limiters.get(name)
    if limiter is None:
        limiter = AsyncRateLimiter(min_interval_seconds)
        _rate_limiters[name] = limiter
    else:
        limiter.set_min_interval(min_interval_seconds)
    return limiter

