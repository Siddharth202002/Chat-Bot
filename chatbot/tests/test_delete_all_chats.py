"""
Bulk chat deletion.

The dangerous part of this feature is not the deletion, it is the scoping: the
checkpoint tables are keyed only by thread_id and have no user column, so a
DELETE written carelessly would take other users' conversations with it. Most
of what follows is there to hold that line.

Everything runs against a temp SQLite file; nothing here touches the real
chat_memory.db or any network.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

import chatbot_backend


ALICE = "user-alice"
BOB = "user-bob"


@pytest.fixture
async def app_db(tmp_path: Path, monkeypatch):
    """A temp database wired in as the backend's request-path connection."""
    conn = await aiosqlite.connect(str(tmp_path / "app.db"))
    await conn.execute(
        """
        CREATE TABLE chat_threads (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Mirrors the checkpointer's shape: keyed by thread_id alone.
    for table in ("checkpoints", "checkpoints_writes", "checkpoints_blobs"):
        await conn.execute(f"CREATE TABLE {table} (thread_id TEXT NOT NULL)")
    await conn.commit()

    async def no_compile():
        return None

    monkeypatch.setattr(chatbot_backend, "_async_conn", conn)
    monkeypatch.setattr(chatbot_backend, "_get_compiled_app", no_compile)
    monkeypatch.setattr(chatbot_backend, "_ensure_app_tables", no_compile)

    async def noop_clear(user_id, thread_id):
        return None

    monkeypatch.setattr(chatbot_backend.memory_store, "clear_thread_state", noop_clear)
    try:
        yield conn
    finally:
        await conn.close()


async def _add_thread(conn, thread_id: str, user_id: str) -> None:
    await conn.execute(
        "INSERT INTO chat_threads (thread_id, user_id, title, created_at, updated_at)"
        " VALUES (?, ?, ?, '', '')",
        (thread_id, user_id, f"chat {thread_id}"),
    )
    for table in ("checkpoints", "checkpoints_writes", "checkpoints_blobs"):
        await conn.execute(f"INSERT INTO {table} (thread_id) VALUES (?)", (thread_id,))
    await conn.commit()


async def _thread_ids(conn, user_id: str) -> set[str]:
    async with conn.execute(
        "SELECT thread_id FROM chat_threads WHERE user_id = ?", (user_id,)
    ) as cursor:
        return {str(row[0]) for row in await cursor.fetchall()}


async def _rows(conn, table: str) -> set[str]:
    async with conn.execute(f"SELECT thread_id FROM {table}") as cursor:
        return {str(row[0]) for row in await cursor.fetchall()}


async def test_deletes_every_thread_for_the_user(app_db):
    for thread_id in ("a1", "a2", "a3"):
        await _add_thread(app_db, thread_id, ALICE)

    deleted = await chatbot_backend.delete_all_chats(ALICE)

    assert deleted == 3
    assert await _thread_ids(app_db, ALICE) == set()


async def test_another_users_chats_are_untouched(app_db):
    """
    The one that matters. The checkpoint tables have no user column, so a
    DELETE that forgot to scope by this user's thread ids would silently wipe
    Bob's conversations too.
    """
    for thread_id in ("a1", "a2"):
        await _add_thread(app_db, thread_id, ALICE)
    for thread_id in ("b1", "b2", "b3"):
        await _add_thread(app_db, thread_id, BOB)

    deleted = await chatbot_backend.delete_all_chats(ALICE)

    assert deleted == 2
    assert await _thread_ids(app_db, BOB) == {"b1", "b2", "b3"}
    for table in ("checkpoints", "checkpoints_writes", "checkpoints_blobs"):
        assert await _rows(app_db, table) == {"b1", "b2", "b3"}


async def test_checkpoint_rows_go_with_the_threads(app_db):
    """Orphaned checkpoint rows would leak the conversation content."""
    await _add_thread(app_db, "a1", ALICE)

    await chatbot_backend.delete_all_chats(ALICE)

    for table in ("checkpoints", "checkpoints_writes", "checkpoints_blobs"):
        assert await _rows(app_db, table) == set()


async def test_deleting_nothing_is_not_an_error(app_db):
    """A user with no chats can still press the button."""
    assert await chatbot_backend.delete_all_chats(ALICE) == 0


async def test_a_missing_optional_table_does_not_abort_the_delete(app_db):
    """Older checkpointer schemas lack some tables; the delete must still run."""
    await _add_thread(app_db, "a1", ALICE)
    await app_db.execute("DROP TABLE checkpoints_blobs")
    await app_db.commit()

    assert await chatbot_backend.delete_all_chats(ALICE) == 1
    assert await _thread_ids(app_db, ALICE) == set()


async def test_an_empty_user_id_is_rejected(app_db):
    """
    Never let a blank user id through: with no WHERE to scope it, that would
    be a request to delete everybody's chats.
    """
    await _add_thread(app_db, "a1", ALICE)

    with pytest.raises(ValueError):
        await chatbot_backend.delete_all_chats("")

    assert await _thread_ids(app_db, ALICE) == {"a1"}


async def test_cached_rag_state_is_dropped(app_db):
    """A deleted thread must not keep answering from its indexed PDF."""
    await _add_thread(app_db, "a1", ALICE)
    chatbot_backend._rag_states[(ALICE, "a1")] = {"status": "ready"}
    chatbot_backend._rag_states[(BOB, "b1")] = {"status": "ready"}

    try:
        await chatbot_backend.delete_all_chats(ALICE)

        assert (ALICE, "a1") not in chatbot_backend._rag_states
        assert (BOB, "b1") in chatbot_backend._rag_states
    finally:
        chatbot_backend._rag_states.pop((BOB, "b1"), None)


async def test_long_term_memories_survive(app_db, monkeypatch):
    """
    Clearing conversations is not the same as being forgotten. Only the
    per-thread extraction state goes.
    """
    cleared: list[tuple[str, str]] = []

    async def record(user_id, thread_id):
        cleared.append((user_id, thread_id))

    monkeypatch.setattr(chatbot_backend.memory_store, "clear_thread_state", record)
    monkeypatch.setattr(
        chatbot_backend.memory_store,
        "delete_memory",
        lambda *a, **k: pytest.fail("user memories must not be deleted"),
    )

    await _add_thread(app_db, "a1", ALICE)
    await _add_thread(app_db, "a2", ALICE)
    await chatbot_backend.delete_all_chats(ALICE)

    assert sorted(cleared) == [(ALICE, "a1"), (ALICE, "a2")]


async def test_a_failing_memory_cleanup_does_not_abort_the_delete(app_db, monkeypatch):
    """The chats are already gone from the database; don't raise over bookkeeping."""

    async def boom(user_id, thread_id):
        raise RuntimeError("memory store unavailable")

    monkeypatch.setattr(chatbot_backend.memory_store, "clear_thread_state", boom)
    await _add_thread(app_db, "a1", ALICE)

    assert await chatbot_backend.delete_all_chats(ALICE) == 1
    assert await _thread_ids(app_db, ALICE) == set()
