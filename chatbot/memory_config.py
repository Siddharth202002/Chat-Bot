"""
Configuration for the long-term memory subsystem.

Every value is read from the environment on each call rather than frozen at
import time, so tests (and a systemd restart-free reload) can change a setting
without re-importing the module. The reads are trivially cheap -- they happen
at most a handful of times per chat turn.

Required env vars for extraction to work at all:
    MISTRAL_API_KEY         - Mistral AI key. NEVER hardcode it.

Optional (defaults in brackets):
    MISTRAL_MEMORY_MODEL             [ministral-8b-latest]
    MISTRAL_MEMORY_TIMEOUT           [20]    seconds, hard cap per attempt
    MISTRAL_MEMORY_MAX_ATTEMPTS      [3]     retries on 429/5xx/timeout
    MISTRAL_MEMORY_RETRY_BASE_DELAY  [1.5]   seconds, doubled each retry
    MEMORY_EXTRACTION_ENABLED       [true]
    MEMORY_EXTRACTION_INTERVAL      [1]     user messages between runs (1 = every)
    MEMORY_EXTRACTION_WINDOW        [10]    messages sent to the extractor
    MEMORY_RETRIEVAL_TOP_K          [5]
    MEMORY_RETRIEVAL_MIN_SCORE      [0.10]  cosine floor (calibrated to Jina v3)
    MEMORY_DEDUP_THRESHOLD          [0.95]  >= this -> identical, skip
    MEMORY_UPDATE_THRESHOLD         [0.82]  >= this -> refine existing memory
    MEMORY_MAX_PER_EXTRACTION       [8]     cap on memories from one model call
    MEMORY_MAX_CONTENT_CHARS        [400]
    MEMORY_MAX_PER_USER             [500]   hard ceiling per user
"""

from __future__ import annotations

import os

# The extraction model name lives here and nowhere else in the codebase.
# ministral-8b is the small/low-cost class: it follows the strict JSON schema
# reliably and returns clean third-person memories, which the 3b model does not.
DEFAULT_MEMORY_MODEL = "ministral-8b-latest"

# Memory types the extractor is allowed to emit. Anything else is coerced to
# "other" rather than rejected, so a model quirk never loses a good memory.
ALLOWED_MEMORY_TYPES: frozenset[str] = frozenset(
    {"preference", "skill", "project", "personal_fact", "workflow", "other"}
)
FALLBACK_MEMORY_TYPE = "other"

def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _get_ratio(name: str, default: float) -> float:
    """A cosine threshold, clamped to [0, 1].

    Out-of-range values are silently meaningless for cosine similarity -- a
    negative dedup threshold would treat every memory as a duplicate and throw
    away real facts -- so they are clamped rather than trusted.
    """
    return min(1.0, max(0.0, _get_float(name, default)))


def memory_api_key() -> str:
    """The Mistral key, from the environment only. Empty string when unset."""
    return (os.getenv("MISTRAL_API_KEY") or "").strip()


def memory_model() -> str:
    return (os.getenv("MISTRAL_MEMORY_MODEL") or "").strip() or DEFAULT_MEMORY_MODEL


def memory_timeout_seconds() -> float:
    return max(1.0, _get_float("MISTRAL_MEMORY_TIMEOUT", 20.0))


def memory_max_attempts() -> int:
    """
    Attempts per extraction, including the first.

    Small hosted models return 429 when the per-key rate limit is hit and 5xx
    under load. Extraction already runs off the response path, so a couple of
    backed-off retries cost the user nothing and are the difference between
    capturing a memory and silently losing it.
    """
    return min(5, _get_int("MISTRAL_MEMORY_MAX_ATTEMPTS", 3))


def memory_retry_base_delay() -> float:
    return max(0.0, _get_float("MISTRAL_MEMORY_RETRY_BASE_DELAY", 1.5))


def extraction_enabled() -> bool:
    return _get_bool("MEMORY_EXTRACTION_ENABLED", True)


def extraction_interval() -> int:
    """
    User messages between extractions. 1 means every message.

    This is the only trigger: there is no phrase matching in front of it, so
    this value alone decides the call volume. 1 gives the best recall (a fact
    is remembered the moment it is mentioned) at one small model call per user
    message. Raise it to cut calls proportionally, at the cost of a fact not
    being available until the next extraction lands.
    """
    return _get_int("MEMORY_EXTRACTION_INTERVAL", 1)


def extraction_window() -> int:
    return _get_int("MEMORY_EXTRACTION_WINDOW", 10)


def retrieval_top_k() -> int:
    return _get_int("MEMORY_RETRIEVAL_TOP_K", 5)


def retrieval_min_score() -> float:
    """
    Cosine floor for a memory to be considered relevant.

    Calibrated against the real Jina v3 asymmetric retrieval embeddings, whose
    scores for short third-person memory sentences sit in a much lower band
    than the 0.5-0.8 people expect from symmetric similarity. Measured:

        query "what is my job"     vs "User is a Python developer."  0.227
        query "What do I do for work?" vs same memory                0.131
        query "What do I do for work?" vs "User's name is Sid."      0.053

    A 0.30 floor discarded every one of those correct matches. Ranking is
    reliable, so this floor only has to reject junk -- MEMORY_RETRIEVAL_TOP_K
    does the real limiting.
    """
    return _get_ratio("MEMORY_RETRIEVAL_MIN_SCORE", 0.10)


def update_threshold() -> float:
    return _get_ratio("MEMORY_UPDATE_THRESHOLD", 0.82)


def dedup_threshold() -> float:
    """
    Never below the update threshold.

    dedup < update is nonsensical -- it would classify a pair as "identical,
    skip" before it could ever be classified as "similar, refine" -- and the
    misconfiguration would silently discard new information.
    """
    return max(_get_ratio("MEMORY_DEDUP_THRESHOLD", 0.95), update_threshold())


def max_memories_per_extraction() -> int:
    return _get_int("MEMORY_MAX_PER_EXTRACTION", 8)


def max_content_chars() -> int:
    return _get_int("MEMORY_MAX_CONTENT_CHARS", 400, minimum=32)


def max_memories_per_user() -> int:
    return _get_int("MEMORY_MAX_PER_USER", 500)
