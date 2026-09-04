"""
Storage and retrieval for long-term, user-scoped memories.

Design notes
------------
* Memories live in the project's existing SQLite database (chat_memory.db),
  alongside the users and chat_threads tables. No second database is added --
  but a second *connection* is, because memory writes run in a background task
  and must not commit in the middle of the LangGraph checkpointer's own writes.
  See chatbot_backend._memory_connection.
* Embeddings come from the *existing* Jina client used by RAG and are stored as
  a float32 BLOB on the memory row. Similarity search is a brute-force cosine
  over one user's rows. A user has tens of memories, not millions, so this is
  faster than a vector-index round trip and -- more importantly -- user
  isolation is enforced by the WHERE user_id = ? in the SQL itself rather than
  by remembering to pass a metadata filter to a vector store.
* Every public function takes user_id and scopes its query to it. There is no
  code path here that reads a memory without a user filter.

This module never imports chatbot_backend; the database connection and the
embeddings client are injected via configure() at startup. That keeps the
import graph acyclic and lets tests run the whole thing against a temp DB with
a deterministic fake embedder.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Sequence

import numpy as np

import memory_config

logger = logging.getLogger("chatbot.memory.store")

ConnectionProvider = Callable[[], Awaitable[Any]]
EmbeddingProvider = Callable[[], Any]

_connection_provider: ConnectionProvider | None = None
_embedding_provider: EmbeddingProvider | None = None

_tables_ready = False

# Serializes read-modify-write sequences (dedup/update, counter bumps): without
# it two concurrent turns could both read "no similar memory exists" and both
# insert the same fact.
#
# The locks are created lazily per event loop rather than at import time. An
# asyncio.Lock binds permanently to the first loop that awaits it, so a
# module-level singleton raises "bound to a different event loop" as soon as a
# second loop touches it -- which is exactly what happens in a test run, and
# what would happen to any future caller that spins up its own loop.
_loop_locks: dict[str, tuple[Any, asyncio.Lock]] = {}


def _lock(name: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    bound = _loop_locks.get(name)
    if bound is None or bound[0] is not loop:
        lock = asyncio.Lock()
        _loop_locks[name] = (loop, lock)
        return lock
    return bound[1]


class MemoryStoreUnavailable(RuntimeError):
    """Raised when the store was never wired up to a database connection."""


def configure(
    *,
    connection_provider: ConnectionProvider,
    embedding_provider: EmbeddingProvider,
) -> None:
    """Inject the shared SQLite connection and the RAG embeddings client."""
    global _connection_provider, _embedding_provider, _tables_ready
    _connection_provider = connection_provider
    _embedding_provider = embedding_provider
    _tables_ready = False


def reset_for_tests() -> None:
    """Drop injected providers and the schema-ready flag (test helper)."""
    global _connection_provider, _embedding_provider, _tables_ready
    _connection_provider = None
    _embedding_provider = None
    _tables_ready = False
    _loop_locks.clear()


async def _connection() -> Any:
    if _connection_provider is None:
        raise MemoryStoreUnavailable("Memory store is not configured.")
    conn = await _connection_provider()
    if conn is None:
        raise MemoryStoreUnavailable("Database connection is unavailable.")
    return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def _new_id() -> str:
    return secrets.token_urlsafe(12)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

async def ensure_memory_tables() -> None:
    """Create the memory tables once. Safe to call on every request."""
    global _tables_ready
    if _tables_ready:
        return
    async with _lock("tables"):
        if _tables_ready:
            return
        conn = await _connection()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                source_thread_id TEXT,
                embedding BLOB,
                embedding_model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_memories_user_updated "
            "ON user_memories(user_id, updated_at DESC)"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_extraction_state (
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                messages_since_extraction INTEGER NOT NULL DEFAULT 0,
                last_extracted_at TEXT,
                PRIMARY KEY (user_id, thread_id)
            )
            """
        )
        await conn.commit()
        _tables_ready = True


# --------------------------------------------------------------------------
# Embeddings (reuses the RAG Jina client)
# --------------------------------------------------------------------------

def _normalize_vector(values: Sequence[float]) -> np.ndarray | None:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


async def _embed_documents(texts: list[str]) -> list[np.ndarray | None]:
    """Embed memory contents. Returns None per item when embedding is down."""
    if not texts:
        return []
    if _embedding_provider is None:
        return [None] * len(texts)
    try:
        client = _embedding_provider()
        raw = await asyncio.to_thread(client.embed_documents, texts)
    except Exception as exc:
        logger.warning("Memory embedding failed; storing memories without vectors: %s", exc)
        return [None] * len(texts)
    if not isinstance(raw, list) or len(raw) != len(texts):
        logger.warning(
            "Embedding client returned an unexpected number of vectors for %s texts.",
            len(texts),
        )
        return [None] * len(texts)
    return [_normalize_vector(vec) for vec in raw]


async def _embed_query(text: str) -> np.ndarray | None:
    if _embedding_provider is None:
        return None
    try:
        client = _embedding_provider()
        raw = await asyncio.to_thread(client.embed_query, text)
    except Exception as exc:
        logger.warning("Memory query embedding failed, falling back to recency: %s", exc)
        return None
    return _normalize_vector(raw)


def _pack(vector: np.ndarray | None) -> bytes | None:
    return None if vector is None else vector.astype(np.float32).tobytes()


def _unpack(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    try:
        vector = np.frombuffer(blob, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    return vector


# --------------------------------------------------------------------------
# Row helpers
# --------------------------------------------------------------------------

_PUBLIC_COLUMNS = "id, content, memory_type, source_thread_id, created_at, updated_at"


def _public_row(row: Sequence[Any]) -> dict[str, Any]:
    return {
        "id": str(row[0]),
        "content": str(row[1]),
        "memory_type": str(row[2]),
        "source_thread_id": row[3],
        "created_at": str(row[4]),
        "updated_at": str(row[5]),
    }


def _clean_content(value: Any) -> str:
    """
    Canonical form for stored memory text.

    Whitespace is collapsed to single spaces on *every* write path, not just
    the extraction one: a memory is rendered back into the system prompt as
    a single bullet, so raw newlines from the user-editable PATCH endpoint could
    otherwise fake extra bullets or headings inside that block.
    """
    if not isinstance(value, str):
        return ""
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed[: memory_config.max_content_chars()]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower().rstrip(".")


def _embedding_model_name() -> str | None:
    if _embedding_provider is None:
        return None
    try:
        return str(getattr(_embedding_provider(), "model", "") or "") or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Public CRUD -- every one of these is scoped to user_id
# --------------------------------------------------------------------------

async def list_memories(user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    if not user_id:
        return []
    await ensure_memory_tables()
    conn = await _connection()
    sql = (
        f"SELECT {_PUBLIC_COLUMNS} FROM user_memories "
        "WHERE user_id = ? ORDER BY updated_at DESC"
    )
    params: list[Any] = [user_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    async with conn.execute(sql, params) as cursor:
        rows = await cursor.fetchall()
    return [_public_row(row) for row in rows]


async def get_memory(user_id: str, memory_id: str) -> dict[str, Any] | None:
    if not user_id or not memory_id:
        return None
    await ensure_memory_tables()
    conn = await _connection()
    async with conn.execute(
        f"SELECT {_PUBLIC_COLUMNS} FROM user_memories WHERE id = ? AND user_id = ?",
        (memory_id, user_id),
    ) as cursor:
        row = await cursor.fetchone()
    return _public_row(row) if row is not None else None


async def delete_memory(user_id: str, memory_id: str) -> bool:
    """Delete one memory. Returns False when it does not belong to the user."""
    if not user_id or not memory_id:
        return False
    await ensure_memory_tables()
    conn = await _connection()
    async with _lock("write"):
        cursor = await conn.execute(
            "DELETE FROM user_memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        deleted = cursor.rowcount
        await cursor.close()
        await conn.commit()
    return deleted > 0


async def update_memory(
    user_id: str,
    memory_id: str,
    *,
    content: str | None = None,
    memory_type: str | None = None,
) -> dict[str, Any] | None:
    """Edit a memory's content and/or type. Returns None if not the user's."""
    if not user_id or not memory_id:
        return None

    cleaned_content = _clean_content(content)
    if content is not None and not cleaned_content:
        raise ValueError("Memory content cannot be empty.")

    cleaned_type: str | None = None
    if memory_type is not None:
        candidate = memory_type.strip().lower()
        if candidate not in memory_config.ALLOWED_MEMORY_TYPES:
            raise ValueError(
                "memory_type must be one of: "
                + ", ".join(sorted(memory_config.ALLOWED_MEMORY_TYPES))
            )
        cleaned_type = candidate

    if not cleaned_content and cleaned_type is None:
        return await get_memory(user_id, memory_id)

    await ensure_memory_tables()
    conn = await _connection()

    # Ownership is checked before spending an embedding call on the new text.
    async with conn.execute(
        "SELECT 1 FROM user_memories WHERE id = ? AND user_id = ?",
        (memory_id, user_id),
    ) as cursor:
        if await cursor.fetchone() is None:
            return None

    embedding_blob: bytes | None = None
    if cleaned_content:
        vector = (await _embed_documents([cleaned_content]))[0]
        embedding_blob = _pack(vector)

    async with _lock("write"):
        assignments = ["updated_at = ?"]
        params: list[Any] = [_utc_now()]
        if cleaned_content:
            assignments.extend(["content = ?", "embedding = ?", "embedding_model = ?"])
            params.extend(
                [
                    cleaned_content,
                    embedding_blob,
                    _embedding_model_name() if embedding_blob else None,
                ]
            )
        if cleaned_type is not None:
            assignments.append("memory_type = ?")
            params.append(cleaned_type)
        params.extend([memory_id, user_id])

        cursor = await conn.execute(
            f"UPDATE user_memories SET {', '.join(assignments)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        updated = cursor.rowcount
        await cursor.close()
        await conn.commit()

    if updated == 0:
        return None
    return await get_memory(user_id, memory_id)


async def delete_all_memories(user_id: str) -> int:
    """Wipe one user's memories. Used by account cleanup and tests."""
    if not user_id:
        return 0
    await ensure_memory_tables()
    conn = await _connection()
    async with _lock("write"):
        cursor = await conn.execute(
            "DELETE FROM user_memories WHERE user_id = ?", (user_id,)
        )
        deleted = cursor.rowcount
        await cursor.close()
        await conn.commit()
    return deleted


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

async def _load_vectors(user_id: str) -> list[tuple[dict[str, Any], np.ndarray | None]]:
    conn = await _connection()
    async with conn.execute(
        f"SELECT {_PUBLIC_COLUMNS}, embedding FROM user_memories "
        "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
        (user_id, memory_config.max_memories_per_user()),
    ) as cursor:
        rows = await cursor.fetchall()
    return [(_public_row(row), _unpack(row[6])) for row in rows]


async def search_memories(
    user_id: str,
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over one user's memories.

    Always scoped to user_id -- there is no global search. When the embedding
    API is unavailable the search degrades to the most recently updated
    memories (still capped at top_k, still user-scoped) rather than failing the
    chat turn.
    """
    if not user_id or not query.strip():
        return []
    await ensure_memory_tables()

    limit = top_k if top_k is not None else memory_config.retrieval_top_k()
    if limit <= 0:
        return []

    candidates = await _load_vectors(user_id)
    if not candidates:
        return []

    query_vector = await _embed_query(query)
    if query_vector is None:
        return [dict(row, score=None) for row, _ in candidates[:limit]]

    min_score = memory_config.retrieval_min_score()
    scored: list[tuple[float, dict[str, Any]]] = []
    for row, vector in candidates:
        if vector is None or vector.shape != query_vector.shape:
            continue
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        score = float(np.dot(query_vector, vector / norm))
        if score >= min_score:
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [dict(row, score=round(score, 4)) for score, row in scored[:limit]]


# --------------------------------------------------------------------------
# Write path: dedup / update / insert
# --------------------------------------------------------------------------

def _best_match(
    vector: np.ndarray | None,
    content: str,
    existing: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    """Closest existing memory for a candidate, by exact text or cosine."""
    normalized = _normalize_text(content)
    for row in existing:
        if _normalize_text(row["content"]) == normalized:
            return 1.0, row

    if vector is None:
        return 0.0, None

    best_score = 0.0
    best_row: dict[str, Any] | None = None
    for row in existing:
        other = row.get("vector")
        if other is None or other.shape != vector.shape:
            continue
        norm = float(np.linalg.norm(other))
        if norm == 0.0:
            continue
        score = float(np.dot(vector, other / norm))
        if score > best_score:
            best_score = score
            best_row = row
    return best_score, best_row


# At most this many rows are re-embedded per extraction, so healing a large
# backlog is spread over several turns instead of one huge embedding request.
_BACKFILL_BATCH = 25


async def _backfill_embeddings(user_id: str) -> int:
    """Re-embed this user's memories that have no usable vector."""
    conn = await _connection()
    async with conn.execute(
        "SELECT id, content FROM user_memories "
        "WHERE user_id = ? AND (embedding IS NULL OR length(embedding) = 0) "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, _BACKFILL_BATCH),
    ) as cursor:
        rows = await cursor.fetchall()
    if not rows:
        return 0

    vectors = await _embed_documents([str(row[1]) for row in rows])
    model_name = _embedding_model_name()
    healed = 0
    async with _lock("write"):
        for (memory_id, _), vector in zip(rows, vectors):
            if vector is None:
                continue
            await conn.execute(
                "UPDATE user_memories SET embedding = ?, embedding_model = ? "
                "WHERE id = ? AND user_id = ?",
                (_pack(vector), model_name, memory_id, user_id),
            )
            healed += 1
        if healed:
            await conn.commit()
    if healed:
        logger.info("Backfilled %s memory embeddings for user %s.", healed, user_id)
    return healed


async def store_memories(
    user_id: str,
    candidates: Iterable[dict[str, Any]],
    *,
    source_thread_id: str | None = None,
) -> dict[str, int]:
    """
    Persist extracted memories with deduplication and in-place updates.

    For each candidate, the closest existing memory *for this user* decides:
      score >= MEMORY_DEDUP_THRESHOLD   -> skip, it is already known
      score >= MEMORY_UPDATE_THRESHOLD  -> rewrite the existing memory
      otherwise                         -> insert a new memory
    """
    # Content is canonicalised *before* embedding so the stored vector always
    # describes the exact text stored on the row. Embedding the raw candidate
    # and then truncating it would silently attach a mismatched vector.
    items: list[tuple[str, str]] = []
    result = {"created": 0, "updated": 0, "skipped": 0}
    for candidate in candidates:
        content = _clean_content(candidate.get("content"))
        if not content:
            if candidate.get("content"):
                result["skipped"] += 1
            continue
        memory_type = str(candidate.get("memory_type") or memory_config.FALLBACK_MEMORY_TYPE)
        if memory_type not in memory_config.ALLOWED_MEMORY_TYPES:
            memory_type = memory_config.FALLBACK_MEMORY_TYPE
        items.append((content, memory_type))

    if not user_id or not items:
        return result

    await ensure_memory_tables()
    conn = await _connection()

    # Rows written while the embeddings API was down are invisible to semantic
    # search and to similarity dedup. This runs in the background extraction
    # task, so it is the natural place to heal them.
    await _backfill_embeddings(user_id)

    vectors = await _embed_documents([content for content, _ in items])
    embedding_model = _embedding_model_name()
    dedup_threshold = memory_config.dedup_threshold()
    update_threshold = memory_config.update_threshold()
    max_per_user = memory_config.max_memories_per_user()

    async with _lock("write"):
        existing: list[dict[str, Any]] = [
            dict(row, vector=vector) for row, vector in await _load_vectors(user_id)
        ]

        for (content, memory_type), vector in zip(items, vectors):
            score, match = _best_match(vector, content, existing)
            now = _utc_now()

            if match is not None and score >= dedup_threshold:
                result["skipped"] += 1
                continue

            if match is not None and score >= update_threshold:
                await conn.execute(
                    "UPDATE user_memories SET content = ?, memory_type = ?, embedding = ?, "
                    "embedding_model = ?, source_thread_id = ?, updated_at = ? "
                    "WHERE id = ? AND user_id = ?",
                    (
                        content,
                        memory_type,
                        _pack(vector),
                        embedding_model if vector is not None else None,
                        source_thread_id,
                        now,
                        match["id"],
                        user_id,
                    ),
                )
                match["content"] = content
                match["memory_type"] = memory_type
                match["vector"] = vector
                result["updated"] += 1
                continue

            if len(existing) >= max_per_user:
                logger.warning(
                    "User %s reached MEMORY_MAX_PER_USER (%s); new memory dropped.",
                    user_id,
                    max_per_user,
                )
                result["skipped"] += 1
                continue

            memory_id = _new_id()
            await conn.execute(
                "INSERT INTO user_memories (id, user_id, content, memory_type, "
                "source_thread_id, embedding, embedding_model, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    user_id,
                    content,
                    memory_type,
                    source_thread_id,
                    _pack(vector),
                    embedding_model if vector is not None else None,
                    now,
                    now,
                ),
            )
            existing.append(
                {
                    "id": memory_id,
                    "content": content,
                    "memory_type": memory_type,
                    "vector": vector,
                }
            )
            result["created"] += 1

        await conn.commit()

    return result


# --------------------------------------------------------------------------
# Extraction cadence bookkeeping
# --------------------------------------------------------------------------

async def bump_message_counter(user_id: str, thread_id: str) -> int:
    """Count one more user message for this thread and return the new total."""
    await ensure_memory_tables()
    conn = await _connection()
    async with _lock("write"):
        await conn.execute(
            "INSERT INTO memory_extraction_state (user_id, thread_id, messages_since_extraction) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(user_id, thread_id) DO UPDATE SET "
            "messages_since_extraction = messages_since_extraction + 1",
            (user_id, thread_id),
        )
        async with conn.execute(
            "SELECT messages_since_extraction FROM memory_extraction_state "
            "WHERE user_id = ? AND thread_id = ?",
            (user_id, thread_id),
        ) as cursor:
            row = await cursor.fetchone()
        await conn.commit()
    return int(row[0]) if row else 0


async def thread_has_been_extracted(user_id: str, thread_id: str) -> bool:
    """True once at least one extraction has run for this thread."""
    await ensure_memory_tables()
    conn = await _connection()
    async with conn.execute(
        "SELECT last_extracted_at FROM memory_extraction_state "
        "WHERE user_id = ? AND thread_id = ?",
        (user_id, thread_id),
    ) as cursor:
        row = await cursor.fetchone()
    return bool(row and row[0])


async def reset_message_counter(user_id: str, thread_id: str) -> None:
    await ensure_memory_tables()
    conn = await _connection()
    async with _lock("write"):
        await conn.execute(
            "UPDATE memory_extraction_state SET messages_since_extraction = 0, "
            "last_extracted_at = ? WHERE user_id = ? AND thread_id = ?",
            (_utc_now(), user_id, thread_id),
        )
        await conn.commit()


async def clear_thread_state(user_id: str, thread_id: str) -> None:
    """Drop cadence bookkeeping for a deleted thread. Memories are untouched."""
    await ensure_memory_tables()
    conn = await _connection()
    async with _lock("write"):
        await conn.execute(
            "DELETE FROM memory_extraction_state WHERE user_id = ? AND thread_id = ?",
            (user_id, thread_id),
        )
        await conn.commit()
