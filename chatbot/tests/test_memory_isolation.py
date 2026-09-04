"""
Cross-user isolation. This is the highest-severity property of the feature:
no operation may ever read, write or expose another user's memories.
"""

from __future__ import annotations

import memory_store

ALICE = "user-alice"
BOB = "user-bob"


async def _seed(db):
    await memory_store.store_memories(
        ALICE,
        [
            {"content": "User prefers SQLite for small projects.", "memory_type": "preference"},
            {"content": "User works at Acme Corp on billing.", "memory_type": "project"},
        ],
        source_thread_id="alice-thread",
    )
    await memory_store.store_memories(
        BOB,
        [{"content": "User is learning Rust.", "memory_type": "skill"}],
        source_thread_id="bob-thread",
    )


async def test_list_only_returns_the_callers_memories(db):
    await _seed(db)
    alice = await memory_store.list_memories(ALICE)
    bob = await memory_store.list_memories(BOB)

    assert len(alice) == 2
    assert len(bob) == 1
    assert "Rust" not in " ".join(m["content"] for m in alice)
    assert "SQLite" not in " ".join(m["content"] for m in bob)


async def test_semantic_search_never_crosses_users(db):
    await _seed(db)
    # Bob asks about exactly what Alice told the assistant.
    results = await memory_store.search_memories(
        BOB, "Should I use SQLite for this small project at Acme Corp?"
    )
    assert all("SQLite" not in r["content"] for r in results)
    assert all("Acme" not in r["content"] for r in results)


async def test_search_for_a_user_with_no_memories_is_empty_not_global(db):
    await _seed(db)
    assert await memory_store.search_memories("user-carol", "SQLite small projects") == []


async def test_get_memory_by_id_is_scoped(db):
    await _seed(db)
    alice_id = (await memory_store.list_memories(ALICE))[0]["id"]
    assert await memory_store.get_memory(ALICE, alice_id) is not None
    assert await memory_store.get_memory(BOB, alice_id) is None


async def test_delete_cannot_touch_another_users_memory(db):
    await _seed(db)
    alice_id = (await memory_store.list_memories(ALICE))[0]["id"]

    assert await memory_store.delete_memory(BOB, alice_id) is False
    assert len(await memory_store.list_memories(ALICE)) == 2


async def test_update_cannot_touch_another_users_memory(db):
    await _seed(db)
    alice_memory = (await memory_store.list_memories(ALICE))[0]

    assert await memory_store.update_memory(BOB, alice_memory["id"], content="hijacked value") is None
    unchanged = await memory_store.get_memory(ALICE, alice_memory["id"])
    assert unchanged["content"] == alice_memory["content"]


async def test_storing_for_one_user_does_not_dedup_against_another(db):
    """Bob stating Alice's fact must create Bob's own row, not reuse hers."""
    await _seed(db)
    result = await memory_store.store_memories(
        BOB, [{"content": "User prefers SQLite for small projects."}]
    )
    assert result["created"] == 1
    assert len(await memory_store.list_memories(ALICE)) == 2
    assert len(await memory_store.list_memories(BOB)) == 2


async def test_delete_all_is_scoped_to_one_user(db):
    await _seed(db)
    assert await memory_store.delete_all_memories(ALICE) == 2
    assert await memory_store.list_memories(ALICE) == []
    assert len(await memory_store.list_memories(BOB)) == 1


async def test_empty_user_id_reads_nothing(db):
    await _seed(db)
    assert await memory_store.list_memories("") == []
    assert await memory_store.search_memories("", "SQLite") == []
    assert await memory_store.get_memory("", "anything") is None
    assert await memory_store.delete_memory("", "anything") is False
