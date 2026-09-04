"""
End-to-end extraction orchestration: triggers -> window -> Mistral -> store.

The Mistral client is always a fake here, so these tests assert exactly how many
model calls the cadence rules produce -- which is the whole point of the
trigger system.
"""

from __future__ import annotations

import json

import pytest

import memory_extraction
import memory_store
from conftest import FakeMistralClient, history

USER = "user-a"
THREAD = "thread-1"


def install_mistral(monkeypatch, response: str | None, error: Exception | None = None):
    client = FakeMistralClient(response_text=response, error=error)
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: client)
    return client


def install_history(monkeypatch, messages):
    async def provider(thread_id: str, user_id: str):
        provider.calls.append((thread_id, user_id))
        return list(messages)

    provider.calls = []
    memory_extraction.configure(history_provider=provider)
    return provider


def _sent_text(call):
    """The user-turn text of a recorded Mistral chat call."""
    return "".join(
        m["content"] for m in call["messages"] if m["role"] == "user"
    )


PAYLOAD = json.dumps(
    {
        "memories": [
            {"content": "User has experience with FastAPI.", "memory_type": "skill"},
            {"content": "User prefers SQLite for small projects.", "memory_type": "preference"},
        ]
    }
)

CONVERSATION = history(
    ("user", "I have been using FastAPI for two years."),
    ("assistant", "That is solid experience."),
    ("user", "I prefer SQLite for small projects."),
    ("assistant", "Reasonable choice."),
)


async def test_explicit_request_extracts_and_stores_immediately(db, monkeypatch):
    client = install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    report = await memory_extraction.run_extraction(
        USER, THREAD)

    assert report["trigger"] == "periodic"
    assert report["created"] == 2
    assert len(client.calls) == 1
    assert len(await memory_store.list_memories(USER)) == 2


async def test_a_raised_interval_suppresses_calls(db, monkeypatch):
    """The interval is the only cost control now that regexes are gone."""
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "20")
    client = install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    for _ in range(5):
        report = await memory_extraction.run_extraction(USER, THREAD)
        assert report["trigger"] is None

    assert client.calls == []
    assert await memory_store.list_memories(USER) == []


async def test_periodic_extraction_runs_once_per_interval(db, monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "5")
    client = install_mistral(monkeypatch, '{"memories": []}')
    install_history(monkeypatch, CONVERSATION)

    triggers = [
        (await memory_extraction.run_extraction(USER, THREAD))["trigger"]
        for _ in range(10)
    ]

    assert triggers == [None, None, None, None, "periodic", None, None, None, None, "periodic"]
    assert len(client.calls) == 2


async def test_extraction_window_is_capped(db, monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_WINDOW", "4")
    client = install_mistral(monkeypatch, '{"memories": []}')
    long_history = history(*[("user", f"message number {i}") for i in range(50)])
    install_history(monkeypatch, long_history)

    await memory_extraction.run_extraction(USER, THREAD)

    sent = _sent_text(client.calls[0])
    assert "message number 49" in sent
    assert "message number 45" not in sent


async def test_one_window_governs_every_extraction(db, monkeypatch):
    """There is no separate short window any more -- just one setting."""
    monkeypatch.setenv("MEMORY_EXTRACTION_WINDOW", "6")
    client = install_mistral(monkeypatch, '{"memories": []}')
    long_history = history(*[("user", f"message number {i}") for i in range(50)])
    install_history(monkeypatch, long_history)

    await memory_extraction.run_extraction(USER, THREAD)

    sent = _sent_text(client.calls[0])
    assert "message number 49" in sent
    assert "message number 44" in sent
    assert "message number 43" not in sent


async def test_the_whole_conversation_is_never_sent(db, monkeypatch):
    client = install_mistral(monkeypatch, '{"memories": []}')
    long_history = history(*[("user", "x" * 200) for _ in range(200)])
    install_history(monkeypatch, long_history)

    await memory_extraction.run_extraction(USER, THREAD)
    assert len(_sent_text(client.calls[0])) < 5000


async def test_counter_resets_after_an_extraction(db, monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "3")
    install_mistral(monkeypatch, '{"memories": []}')
    install_history(monkeypatch, CONVERSATION)

    # Three messages to reach the interval, then the extraction resets it.
    triggers = [
        (await memory_extraction.run_extraction(USER, THREAD))["trigger"]
        for _ in range(6)
    ]
    # Fires on message 3 and again on 6, never in between.
    assert triggers == [None, None, "periodic", None, None, "periodic"]


async def test_counter_resets_even_when_mistral_finds_nothing(db, monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "2")
    client = install_mistral(monkeypatch, '{"memories": []}')
    install_history(monkeypatch, CONVERSATION)

    await memory_extraction.run_extraction(USER, THREAD)
    await memory_extraction.run_extraction(USER, THREAD)
    assert len(client.calls) == 1
    await memory_extraction.run_extraction(USER, THREAD)
    assert len(client.calls) == 1


async def test_disabled_flag_stops_everything(db, monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "false")
    client = install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    report = await memory_extraction.run_extraction(
        USER, THREAD)
    assert report["trigger"] is None
    assert client.calls == []


async def test_extraction_is_skipped_when_history_is_empty(db, monkeypatch):
    client = install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, [])

    report = await memory_extraction.run_extraction(USER, THREAD)
    assert report["trigger"] == "periodic"
    assert client.calls == []
    assert await memory_store.list_memories(USER) == []


async def test_history_is_read_with_the_authenticated_user_id(db, monkeypatch):
    install_mistral(monkeypatch, '{"memories": []}')
    provider = install_history(monkeypatch, CONVERSATION)

    await memory_extraction.run_extraction(USER, THREAD)
    assert provider.calls == [(THREAD, USER)]


async def test_memories_are_attributed_to_the_extracting_user_only(db, monkeypatch):
    install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    await memory_extraction.run_extraction(USER, THREAD)
    assert len(await memory_store.list_memories(USER)) == 2
    assert await memory_store.list_memories("someone-else") == []


async def test_repeated_extractions_deduplicate(db, monkeypatch):
    install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    await memory_extraction.run_extraction(USER, THREAD)
    second = await memory_extraction.run_extraction(USER, THREAD)

    assert second["created"] == 0
    assert second["skipped"] == 2
    assert len(await memory_store.list_memories(USER)) == 2


async def test_missing_user_or_thread_is_a_no_op(db, monkeypatch):
    client = install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    assert (await memory_extraction.run_extraction("", THREAD))["trigger"] is None
    assert (await memory_extraction.run_extraction(USER, ""))["trigger"] is None
    assert client.calls == []


async def test_process_turn_swallows_failures(db, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(memory_store, "bump_message_counter", boom)
    # Must not raise: it runs in a background task after the reply was sent.
    await memory_extraction.process_turn(USER, THREAD)


async def test_concurrent_extractions_do_not_duplicate(db, monkeypatch):
    import asyncio

    install_mistral(monkeypatch, PAYLOAD)
    install_history(monkeypatch, CONVERSATION)

    await asyncio.gather(
        *[
            memory_extraction.run_extraction(USER, THREAD)
            for _ in range(5)
        ]
    )
    memories = await memory_store.list_memories(USER)
    assert len(memories) == 2
