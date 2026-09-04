"""Storage: create, read, update, delete, dedup, semantic search."""

from __future__ import annotations

import pytest

import memory_store

USER = "user-a"


async def test_store_and_list_memories(db):
    result = await memory_store.store_memories(
        USER,
        [
            {"content": "User prefers SQLite for small projects.", "memory_type": "preference"},
            {"content": "User is learning GraphQL.", "memory_type": "skill"},
        ],
        source_thread_id="t1",
    )
    assert result == {"created": 2, "updated": 0, "skipped": 0}

    memories = await memory_store.list_memories(USER)
    assert len(memories) == 2
    assert {m["memory_type"] for m in memories} == {"preference", "skill"}
    assert all(m["source_thread_id"] == "t1" for m in memories)
    # Embeddings never leak through the public shape.
    assert "embedding" not in memories[0]


async def test_identical_memory_is_skipped(db):
    payload = [{"content": "User prefers SQLite.", "memory_type": "preference"}]
    await memory_store.store_memories(USER, payload)
    second = await memory_store.store_memories(USER, payload)
    assert second == {"created": 0, "updated": 0, "skipped": 1}
    assert len(await memory_store.list_memories(USER)) == 1


async def test_case_and_punctuation_variants_are_deduplicated(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    result = await memory_store.store_memories(USER, [{"content": "user prefers sqlite"}])
    assert result["skipped"] == 1
    assert len(await memory_store.list_memories(USER)) == 1


async def test_similar_memory_updates_instead_of_duplicating(db, monkeypatch):
    monkeypatch.setenv("MEMORY_UPDATE_THRESHOLD", "0.70")
    await memory_store.store_memories(
        USER, [{"content": "User prefers FastAPI.", "memory_type": "preference"}]
    )
    result = await memory_store.store_memories(
        USER, [{"content": "User prefers FastAPI over Django.", "memory_type": "preference"}]
    )
    assert result == {"created": 0, "updated": 1, "skipped": 0}

    memories = await memory_store.list_memories(USER)
    assert len(memories) == 1
    assert memories[0]["content"] == "User prefers FastAPI over Django."


async def test_unrelated_memory_creates_a_new_row(db):
    await memory_store.store_memories(USER, [{"content": "User prefers FastAPI."}])
    result = await memory_store.store_memories(
        USER, [{"content": "User is learning GraphQL."}]
    )
    assert result["created"] == 1
    assert len(await memory_store.list_memories(USER)) == 2


async def test_changed_preference_replaces_the_old_one(db, monkeypatch):
    monkeypatch.setenv("MEMORY_UPDATE_THRESHOLD", "0.55")
    await memory_store.store_memories(
        USER, [{"content": "User prefers Django.", "memory_type": "preference"}]
    )
    await memory_store.store_memories(
        USER, [{"content": "User prefers FastAPI now.", "memory_type": "preference"}]
    )
    memories = await memory_store.list_memories(USER)
    assert len(memories) == 1
    assert "FastAPI" in memories[0]["content"]


async def test_semantic_search_ranks_the_relevant_memory_first(db):
    await memory_store.store_memories(
        USER,
        [
            {"content": "User prefers SQLite for small projects.", "memory_type": "preference"},
            {"content": "User is learning GraphQL.", "memory_type": "skill"},
            {"content": "User lives in Bengaluru.", "memory_type": "personal_fact"},
        ],
    )
    results = await memory_store.search_memories(
        USER, "Should I use SQLite for this small project?", top_k=1
    )
    assert len(results) == 1
    assert "SQLite" in results[0]["content"]
    assert results[0]["score"] > 0


async def test_search_respects_top_k(db):
    # Deliberately unrelated to each other so dedup does not merge them, but
    # all sharing one query term.
    await memory_store.store_memories(
        USER,
        [
            {"content": "User plays guitar alpha."},
            {"content": "User bakes sourdough alpha."},
            {"content": "User runs marathons alpha."},
            {"content": "User collects stamps alpha."},
            {"content": "User studies astronomy alpha."},
            {"content": "User restores bicycles alpha."},
        ],
    )
    assert len(await memory_store.list_memories(USER)) == 6
    results = await memory_store.search_memories(USER, "alpha user", top_k=2)
    assert len(results) == 2


async def test_search_returns_nothing_for_a_user_with_no_memories(db):
    assert await memory_store.search_memories("nobody", "anything") == []


async def test_search_with_blank_query_returns_nothing(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    assert await memory_store.search_memories(USER, "   ") == []


async def test_irrelevant_query_is_filtered_by_min_score(db, monkeypatch):
    monkeypatch.setenv("MEMORY_RETRIEVAL_MIN_SCORE", "0.9")
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    assert await memory_store.search_memories(USER, "quantum chromodynamics lecture") == []


async def test_update_memory_changes_content_and_type(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(USER))[0]["id"]

    updated = await memory_store.update_memory(
        USER, memory_id, content="User prefers Postgres.", memory_type="preference"
    )
    assert updated is not None
    assert updated["content"] == "User prefers Postgres."
    assert updated["memory_type"] == "preference"


async def test_update_rejects_an_invalid_memory_type(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(USER))[0]["id"]
    with pytest.raises(ValueError):
        await memory_store.update_memory(USER, memory_id, memory_type="banana")


async def test_update_rejects_empty_content(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(USER))[0]["id"]
    with pytest.raises(ValueError):
        await memory_store.update_memory(USER, memory_id, content="   ")


async def test_update_of_a_missing_memory_returns_none(db):
    assert await memory_store.update_memory(USER, "nope", content="x" * 20) is None


async def test_delete_memory(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    memory_id = (await memory_store.list_memories(USER))[0]["id"]
    assert await memory_store.delete_memory(USER, memory_id) is True
    assert await memory_store.list_memories(USER) == []
    assert await memory_store.delete_memory(USER, memory_id) is False


async def test_blank_candidates_are_skipped(db):
    result = await memory_store.store_memories(
        USER, [{"content": ""}, {"content": "   "}, {"content": None}]
    )
    assert result["created"] == 0
    assert await memory_store.list_memories(USER) == []


async def test_content_is_truncated_at_the_configured_limit(db, monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_CONTENT_CHARS", "50")
    await memory_store.store_memories(USER, [{"content": "User likes " + "z" * 500}])
    assert len((await memory_store.list_memories(USER))[0]["content"]) == 50


async def test_per_user_ceiling_is_enforced(db, monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_PER_USER", "3")
    result = await memory_store.store_memories(
        USER, [{"content": f"User owns unique gadget alpha{i} bravo{i}."} for i in range(5)]
    )
    assert result["created"] == 3
    assert result["skipped"] == 2
    assert len(await memory_store.list_memories(USER)) == 3


async def test_unknown_memory_type_is_coerced_at_the_store_boundary(db):
    await memory_store.store_memories(USER, [{"content": "User uses SQLite.", "memory_type": "nonsense"}])
    assert (await memory_store.list_memories(USER))[0]["memory_type"] == "other"


async def test_message_counter_increments_and_resets(db):
    assert await memory_store.bump_message_counter(USER, "t1") == 1
    assert await memory_store.bump_message_counter(USER, "t1") == 2
    await memory_store.reset_message_counter(USER, "t1")
    assert await memory_store.bump_message_counter(USER, "t1") == 1


async def test_message_counters_are_per_thread_and_per_user(db):
    await memory_store.bump_message_counter(USER, "t1")
    await memory_store.bump_message_counter(USER, "t1")
    assert await memory_store.bump_message_counter(USER, "t2") == 1
    assert await memory_store.bump_message_counter("user-b", "t1") == 1


async def test_clear_thread_state_leaves_memories_intact(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    await memory_store.bump_message_counter(USER, "t1")
    await memory_store.clear_thread_state(USER, "t1")
    assert await memory_store.bump_message_counter(USER, "t1") == 1
    assert len(await memory_store.list_memories(USER)) == 1
