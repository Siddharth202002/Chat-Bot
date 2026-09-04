"""
HTTP surface: /api/memories CRUD and the background hook on the chat routes.

The app is driven in-process over ASGI with the authentication dependency
overridden, so these tests cover routing, status codes and -- most importantly
-- that the authenticated user id is the only one that ever reaches the store.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

import api_server
import memory_store

ALICE = {"id": "user-alice", "email": "alice@example.com"}
BOB = {"id": "user-bob", "email": "bob@example.com"}


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


async def test_unauthenticated_requests_are_rejected(db):
    api_server.app.dependency_overrides.clear()
    async with client() as http:
        assert (await http.get("/api/memories")).status_code == 401
        assert (await http.delete("/api/memories/x")).status_code == 401
        assert (await http.patch("/api/memories/x", json={"content": "y"})).status_code == 401


async def test_list_memories_returns_only_the_callers_own(db, as_user):
    await memory_store.store_memories(
        ALICE["id"], [{"content": "User prefers SQLite.", "memory_type": "preference"}]
    )
    await memory_store.store_memories(
        BOB["id"], [{"content": "User is learning Rust.", "memory_type": "skill"}]
    )

    as_user(ALICE)
    async with client() as http:
        response = await http.get("/api/memories")
    assert response.status_code == 200
    memories = response.json()["memories"]
    assert len(memories) == 1
    assert memories[0]["content"] == "User prefers SQLite."

    as_user(BOB)
    async with client() as http:
        memories = (await http.get("/api/memories")).json()["memories"]
    assert [m["content"] for m in memories] == ["User is learning Rust."]


async def test_delete_own_memory(db, as_user):
    await memory_store.store_memories(ALICE["id"], [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(ALICE["id"]))[0]["id"]

    as_user(ALICE)
    async with client() as http:
        assert (await http.delete(f"/api/memories/{memory_id}")).status_code == 200
        assert (await http.delete(f"/api/memories/{memory_id}")).status_code == 404
    assert await memory_store.list_memories(ALICE["id"]) == []


async def test_deleting_another_users_memory_is_404_and_a_no_op(db, as_user):
    await memory_store.store_memories(ALICE["id"], [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(ALICE["id"]))[0]["id"]

    as_user(BOB)
    async with client() as http:
        assert (await http.delete(f"/api/memories/{memory_id}")).status_code == 404
    assert len(await memory_store.list_memories(ALICE["id"])) == 1


async def test_patch_own_memory(db, as_user):
    await memory_store.store_memories(ALICE["id"], [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(ALICE["id"]))[0]["id"]

    as_user(ALICE)
    async with client() as http:
        response = await http.patch(
            f"/api/memories/{memory_id}",
            json={"content": "User prefers Postgres.", "memory_type": "preference"},
        )
    assert response.status_code == 200
    assert response.json()["memory"]["content"] == "User prefers Postgres."


async def test_patching_another_users_memory_is_404(db, as_user):
    await memory_store.store_memories(ALICE["id"], [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(ALICE["id"]))[0]["id"]

    as_user(BOB)
    async with client() as http:
        response = await http.patch(
            f"/api/memories/{memory_id}", json={"content": "hijacked value"}
        )
    assert response.status_code == 404
    assert (await memory_store.get_memory(ALICE["id"], memory_id))[
        "content"
    ] == "User prefers SQLite."


async def test_patch_with_no_fields_is_400(db, as_user):
    as_user(ALICE)
    async with client() as http:
        assert (await http.patch("/api/memories/any", json={})).status_code == 400


async def test_patch_with_an_invalid_type_is_400(db, as_user):
    await memory_store.store_memories(ALICE["id"], [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(ALICE["id"]))[0]["id"]

    as_user(ALICE)
    async with client() as http:
        response = await http.patch(
            f"/api/memories/{memory_id}", json={"memory_type": "banana"}
        )
    assert response.status_code == 400


async def test_chat_schedules_background_extraction(db, as_user, monkeypatch):
    calls: list[tuple] = []

    async def fake_response(message, thread_id="1", user_id=""):
        return "hello back"

    async def fake_memory_turn(user_id, thread_id):
        calls.append((user_id, thread_id))

    monkeypatch.setattr(api_server, "get_response", fake_response)
    monkeypatch.setattr(api_server, "process_memory_turn", fake_memory_turn)

    as_user(ALICE)
    async with client() as http:
        response = await http.post(
            "/api/chat", json={"message": "Remember that I use FastAPI.", "thread_id": "t1"}
        )

    assert response.status_code == 200
    assert response.json()["response"] == "hello back"
    # The authenticated id is used, never anything from the request body.
    assert calls == [(ALICE["id"], "t1")]


async def test_stream_schedules_background_extraction(db, as_user, monkeypatch):
    calls: list[tuple] = []

    async def fake_stream(message, thread_id="1", user_id=""):
        yield "hel"
        yield "lo"

    async def fake_memory_turn(user_id, thread_id):
        calls.append((user_id, thread_id))

    monkeypatch.setattr(api_server, "get_response_stream", fake_stream)
    monkeypatch.setattr(api_server, "process_memory_turn", fake_memory_turn)

    as_user(BOB)
    async with client() as http:
        response = await http.post(
            "/api/chat/stream", json={"message": "I prefer SQLite.", "thread_id": "t9"}
        )

    assert response.status_code == 200
    assert "hel" in response.text
    assert calls == [(BOB["id"], "t9")]


async def test_a_failed_chat_turn_does_not_schedule_extraction(db, as_user, monkeypatch):
    calls: list[tuple] = []

    async def broken_response(message, thread_id="1", user_id=""):
        raise RuntimeError("groq is down")

    async def fake_memory_turn(user_id, thread_id):
        calls.append((user_id, thread_id))

    monkeypatch.setattr(api_server, "get_response", broken_response)
    monkeypatch.setattr(api_server, "process_memory_turn", fake_memory_turn)

    as_user(ALICE)
    async with client() as http:
        response = await http.post("/api/chat", json={"message": "hi", "thread_id": "t1"})

    assert response.status_code == 200
    assert "Error:" in response.json()["response"]
    assert calls == []


async def test_background_extraction_failure_does_not_break_the_response(
    db, as_user, monkeypatch
):
    async def fake_response(message, thread_id="1", user_id=""):
        return "hello back"

    monkeypatch.setattr(api_server, "get_response", fake_response)

    as_user(ALICE)
    async with client() as http:
        response = await http.post(
            "/api/chat", json={"message": "Remember that I use FastAPI.", "thread_id": "t1"}
        )

    # process_memory_turn runs for real here: no Mistral client is configured, so
    # it must complete quietly and leave the 200 intact.
    assert response.status_code == 200
    assert response.json()["response"] == "hello back"
