"""
Shared fixtures for the long-term memory tests.

Everything here runs fully offline:
  * the database is a fresh temp SQLite file per test,
  * embeddings come from a deterministic bag-of-words fake,
  * the Mistral client factory is stubbed out to return None by default, so a
    test can never accidentally spend a real API call. Tests that exercise the
    Mistral path install their own fake client.
"""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory_extraction  # noqa: E402
import memory_store  # noqa: E402

# Captured before the autouse stub replaces it, so a test can exercise the real
# client factory (e.g. the "SDK not installed" path).
REAL_GET_CLIENT = memory_extraction._get_extraction_client

EMBEDDING_DIM = 64


def _vector(text: str) -> list[float]:
    """Stable bag-of-words vector: overlapping wording -> high cosine."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vector[zlib.crc32(token.encode("utf-8")) % EMBEDDING_DIM] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        vector[0] = 1.0
        norm = 1.0
    return (vector / norm).tolist()


class FakeEmbeddings:
    """Stand-in for the Jina client used by RAG."""

    model = "fake-embeddings"

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0
        self.fail_documents = False
        self.fail_queries = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        if self.fail_documents:
            raise RuntimeError("Jina embeddings API request failed: 503")
        return [_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        if self.fail_queries:
            raise RuntimeError("Jina embeddings API request failed: 503")
        return _vector(text)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class FakeMistralResponse:
    """Mirrors mistralai's ChatCompletionResponse shape."""

    def __init__(self, text):
        self.choices = [_FakeChoice(text)] if text is not None else []


class _FakeChat:
    def __init__(self, owner):
        self._owner = owner

    async def complete_async(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        if self._owner.delay:
            import asyncio

            await asyncio.sleep(self._owner.delay)
        return FakeMistralResponse(self._owner.response_text)


class FakeMistralClient:
    """Minimal stand-in for mistralai.Mistral."""

    def __init__(self, response_text='{"memories": []}', error=None, delay=0.0):
        self.response_text = response_text
        self.error = error
        self.delay = delay
        self.calls = []
        self.chat = _FakeChat(self)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
async def db(tmp_path: Path, embeddings: FakeEmbeddings):
    """A temp database with the memory store wired to it."""
    conn = await aiosqlite.connect(str(tmp_path / "test_memory.db"))
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE)"
    )
    await conn.commit()

    async def connection_provider():
        return conn

    memory_store.configure(
        connection_provider=connection_provider,
        embedding_provider=lambda: embeddings,
    )
    try:
        yield conn
    finally:
        memory_store.reset_for_tests()
        await conn.close()


@pytest.fixture(autouse=True)
def no_real_mistral(monkeypatch):
    """Hard stop against any test reaching the real Mistral API."""
    monkeypatch.setattr(memory_extraction, "_get_extraction_client", lambda: None)
    memory_extraction._client_cache.clear()


@pytest.fixture(autouse=True)
def isolated_history_provider():
    """
    Keep the extraction window off the real chat_memory.db.

    memory_extraction.configure() sets a module global, so without this a test
    that installs a fake provider would leak it into later tests -- and the
    default provider is chatbot_backend.get_chat_history, which opens the real
    application database.
    """
    original = memory_extraction._history_provider

    async def empty_history(thread_id: str, user_id: str) -> list[dict[str, str]]:
        return []

    memory_extraction.configure(history_provider=empty_history)
    try:
        yield
    finally:
        memory_extraction.configure(history_provider=original)


@pytest.fixture(autouse=True)
def memory_env(monkeypatch):
    """Deterministic memory settings; individual tests override as needed."""
    monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "1")
    monkeypatch.setenv("MEMORY_EXTRACTION_WINDOW", "10")
    monkeypatch.setenv("MEMORY_RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MEMORY_RETRIEVAL_MIN_SCORE", "0.30")
    monkeypatch.setenv("MEMORY_DEDUP_THRESHOLD", "0.95")
    monkeypatch.setenv("MEMORY_UPDATE_THRESHOLD", "0.70")
    monkeypatch.setenv("MEMORY_MAX_PER_EXTRACTION", "8")
    monkeypatch.setenv("MEMORY_MAX_CONTENT_CHARS", "400")
    monkeypatch.setenv("MEMORY_MAX_PER_USER", "500")


def history(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    """Build a chat history list from (role, content) tuples."""
    return [{"role": role, "content": content} for role, content in pairs]
