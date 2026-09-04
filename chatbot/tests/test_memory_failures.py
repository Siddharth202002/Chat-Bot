"""
Failure isolation.

The contract is: memory can degrade, chat cannot. Every dependency of the
memory subsystem -- Mistral, the embeddings API, SQLite -- is failed here and
the assertion is always the same: no exception escapes into the chat path.
"""

from __future__ import annotations

import asyncio

import pytest

import memory_extraction
import memory_store
from conftest import REAL_GET_CLIENT, FakeMistralClient, history

USER = "user-a"
THREAD = "thread-1"

CONVERSATION = history(
    ("user", "I have been using FastAPI for two years."),
    ("assistant", "Noted."),
)


def install_history(monkeypatch, messages=CONVERSATION):
    async def provider(thread_id: str, user_id: str):
        return list(messages)

    memory_extraction.configure(history_provider=provider)


# --------------------------------------------------------------------------
# Mistral failures
# --------------------------------------------------------------------------

async def test_missing_api_key_disables_extraction_quietly(db, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    memory_extraction._client_cache.clear()
    install_history(monkeypatch)

    report = await memory_extraction.run_extraction(USER, THREAD)
    assert report["created"] == 0
    assert await memory_store.list_memories(USER) == []


async def test_mistral_api_error_is_swallowed(db, monkeypatch):
    client = FakeMistralClient(error=RuntimeError("500 Internal Server Error"))
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)
    install_history(monkeypatch)

    report = await memory_extraction.run_extraction(USER, THREAD)
    assert report["created"] == 0
    assert await memory_store.list_memories(USER) == []


async def test_mistral_rate_limit_is_swallowed(db, monkeypatch):
    client = FakeMistralClient(error=RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)
    install_history(monkeypatch)

    assert await memory_extraction.extract_memories(CONVERSATION) == []


async def test_mistral_timeout_is_bounded(db, monkeypatch):
    """Each attempt is capped by MISTRAL_MEMORY_TIMEOUT, never the model's pace."""
    monkeypatch.setenv("MISTRAL_MEMORY_TIMEOUT", "1")
    monkeypatch.setenv("MISTRAL_MEMORY_MAX_ATTEMPTS", "1")  # isolate one attempt
    client = FakeMistralClient(response_text='{"memories": []}', delay=5.0)
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)
    install_history(monkeypatch)

    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await memory_extraction.extract_memories(CONVERSATION)
    assert result == []
    assert loop.time() - started < 3.0


async def test_total_extraction_time_is_bounded_across_retries(db, monkeypatch):
    monkeypatch.setenv("MISTRAL_MEMORY_TIMEOUT", "1")
    monkeypatch.setenv("MISTRAL_MEMORY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("MISTRAL_MEMORY_RETRY_BASE_DELAY", "0")
    client = FakeMistralClient(response_text='{"memories": []}', delay=5.0)
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await memory_extraction.extract_memories(CONVERSATION) == []
    # 3 attempts x 1s cap, plus zero backoff.
    assert loop.time() - started < 6.0


async def test_mistral_empty_response_is_handled(db, monkeypatch):
    for empty in (None, "", "   "):
        client = FakeMistralClient(response_text=empty)
        monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda c=client: c)
        assert await memory_extraction.extract_memories(CONVERSATION) == []


async def test_mistral_invalid_json_never_reaches_the_database(db, monkeypatch):
    client = FakeMistralClient(response_text="I think the user likes FastAPI!")
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)
    install_history(monkeypatch)

    report = await memory_extraction.run_extraction(USER, THREAD)
    assert report["created"] == 0
    assert await memory_store.list_memories(USER) == []


async def test_mistral_valid_json_wrong_shape_is_rejected(db, monkeypatch):
    client = FakeMistralClient(response_text='{"memories": {"content": "oops"}}')
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)
    install_history(monkeypatch)

    report = await memory_extraction.run_extraction(USER, THREAD)
    assert report["created"] == 0


async def test_sdk_not_installed_is_handled(monkeypatch):
    """The real client factory returns None rather than raising on ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mistralai" or name.startswith("mistralai."):
            raise ImportError("No module named 'mistralai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    memory_extraction._client_cache.clear()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert REAL_GET_CLIENT() is None
    finally:
        memory_extraction._client_cache.clear()


async def test_no_api_key_means_no_client(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    memory_extraction._client_cache.clear()
    assert REAL_GET_CLIENT() is None


# --------------------------------------------------------------------------
# Embedding failures
# --------------------------------------------------------------------------

async def test_embedding_failure_still_stores_the_memory(db, embeddings, monkeypatch):
    embeddings.fail_documents = True
    result = await memory_store.store_memories(
        USER, [{"content": "User prefers SQLite for small projects."}]
    )
    assert result["created"] == 1
    assert len(await memory_store.list_memories(USER)) == 1


async def test_embedding_failure_still_deduplicates_on_exact_text(db, embeddings):
    embeddings.fail_documents = True
    payload = [{"content": "User prefers SQLite for small projects."}]
    await memory_store.store_memories(USER, payload)
    second = await memory_store.store_memories(USER, payload)
    assert second["skipped"] == 1
    assert len(await memory_store.list_memories(USER)) == 1


async def test_query_embedding_failure_degrades_to_recency(db, embeddings):
    await memory_store.store_memories(
        USER,
        [
            {"content": "User plays guitar alpha."},
            {"content": "User bakes sourdough bravo."},
            {"content": "User runs marathons charlie."},
        ],
    )
    embeddings.fail_queries = True
    results = await memory_store.search_memories(USER, "anything at all", top_k=2)
    assert len(results) == 2
    assert all(r["score"] is None for r in results)


async def test_search_survives_a_totally_broken_embedder(db, embeddings):
    embeddings.fail_documents = True
    embeddings.fail_queries = True
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    results = await memory_store.search_memories(USER, "database preference")
    assert len(results) == 1


async def test_dimension_mismatch_rows_are_ignored_not_fatal(db, embeddings, monkeypatch):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    # Simulate an embedding-model change: an old row has a different width.
    await db.execute(
        "UPDATE user_memories SET embedding = ?",
        (bytes(4 * 7),),
    )
    await db.commit()
    assert await memory_store.search_memories(USER, "database preference") == []


async def test_corrupt_embedding_blob_is_ignored(db):
    await memory_store.store_memories(USER, [{"content": "User prefers SQLite."}])
    await db.execute("UPDATE user_memories SET embedding = ?", (b"\x00\x01\x02",))
    await db.commit()
    assert await memory_store.search_memories(USER, "database preference") == []
    assert len(await memory_store.list_memories(USER)) == 1


# --------------------------------------------------------------------------
# Database / configuration failures
# --------------------------------------------------------------------------

async def test_unconfigured_store_raises_a_typed_error(monkeypatch):
    memory_store.reset_for_tests()
    with pytest.raises(memory_store.MemoryStoreUnavailable):
        await memory_store.list_memories(USER)


async def test_null_connection_raises_a_typed_error(monkeypatch, embeddings):
    async def no_connection():
        return None

    memory_store.configure(
        connection_provider=no_connection, embedding_provider=lambda: embeddings
    )
    with pytest.raises(memory_store.MemoryStoreUnavailable):
        await memory_store.list_memories(USER)
    memory_store.reset_for_tests()


async def test_database_failure_during_extraction_is_isolated(db, monkeypatch):
    install_history(monkeypatch)
    client = FakeMistralClient(
        response_text='{"memories": [{"content": "User uses FastAPI.", "memory_type": "skill"}]}'
    )
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)

    async def boom(*args, **kwargs):
        raise RuntimeError("attempt to write a readonly database")

    monkeypatch.setattr(memory_store, "store_memories", boom)
    # process_turn is the background entry point: it must never raise.
    await memory_extraction.process_turn(USER, THREAD)


async def test_retrieval_failure_does_not_break_the_chat_path(db, monkeypatch):
    import chatbot_backend

    async def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(memory_store, "search_memories", boom)
    assert await chatbot_backend.retrieve_memory_context(USER, "hello") == []
