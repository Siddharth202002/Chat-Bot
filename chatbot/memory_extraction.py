"""
Long-term memory extraction: when to run it, and how to run it with Mistral.

Extraction runs every MEMORY_EXTRACTION_INTERVAL user messages -- 1 by default,
so once per user message. There is deliberately no phrase matching in front of
it: an earlier version gated calls behind regexes for "remember that ...",
"I prefer ...", "I'm using ..." and friends, which made recall depend on
enumerating English phrasings and failed silently whenever one was missed
("I am using FastAPI" was dropped, "I'm using FastAPI" was kept). Judging what
is worth remembering is the model's job; this module only decides when to ask.

Only the last MEMORY_EXTRACTION_WINDOW messages are ever sent to Mistral, so
cost per call stays flat no matter how long the conversation grows. Raise the
interval to trade recall latency for fewer calls.

Everything in this module is failure-isolated: extraction runs after the chat
response has already been returned to the user, and every entry point swallows
its exceptions after logging. A Mistral outage degrades memory quality; it can
never degrade or block a chat turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable

import memory_config
import memory_store

logger = logging.getLogger("chatbot.memory.extraction")

HistoryProvider = Callable[[str, str], Awaitable[list[dict[str, str]]]]

_history_provider: HistoryProvider | None = None


def configure(*, history_provider: HistoryProvider) -> None:
    """Inject the chat-history reader (thread_id, user_id) -> messages."""
    global _history_provider
    _history_provider = history_provider


# --------------------------------------------------------------------------
# Trigger: a message counter, and nothing else
# --------------------------------------------------------------------------

def decide_trigger(messages_since_extraction: int) -> str | None:
    """
    Return the trigger name for this turn, or None to skip extraction.

    One rule: run every MEMORY_EXTRACTION_INTERVAL user messages (default 1,
    i.e. every message).

    There used to be regex trigger detection here -- explicit "remember X"
    patterns plus a list of self-descriptive phrasings like "I prefer" and
    "I'm using". It was removed because it was the wrong tool for the job:
    recall depended on enumerating English phrasings, and every miss was
    silent. "I am using FastAPI" was dropped while "I'm using FastAPI" was
    kept, purely because one pattern spelled the contraction and another did
    not. Deciding what is worth remembering is the extraction model's job; the
    trigger's only job is deciding when to ask it.
    """
    if messages_since_extraction >= memory_config.extraction_interval():
        return "periodic"
    return None


# --------------------------------------------------------------------------
# Mistral client
# --------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You extract long-term memories about a user from a short excerpt of their chat with an assistant.

Return ONLY information that would still be useful in a completely different conversation weeks from now.

STORE facts about the user, such as:
- stated preferences ("prefers SQLite for small projects")
- skills, experience and tools they use ("has two years of FastAPI experience")
- what they are building or working on ("is building a travel chatbot")
- what they are learning ("is learning GraphQL")
- stable personal facts they volunteered (name, role, location, team)
- how they want the assistant to behave ("wants concise technical answers")

DO NOT store:
- questions the user asked, or topics they merely asked about
- anything the assistant said or suggested
- greetings, thanks, small talk, or acknowledgements
- one-off or time-bound requests ("summarise this PDF", "what is the weather today")
- speculation, or anything not clearly stated by the user

Rules:
- Write each memory as a short, self-contained third-person sentence starting with "User".
- One fact per memory. Do not merge unrelated facts.
- If nothing qualifies, return an empty list.
- Never include secrets, passwords, API keys, or payment details.

GROUNDING (most important rule):
Every memory must be supported by words the user actually wrote in the excerpt.
- Never add durations, counts, seniority, job titles, or company names that are
  not written in the excerpt. If the user did not say how long, do not say it.
- Prefer a shorter memory that is certainly true over a richer one that guesses.
- If you are unsure whether the user stated something, leave it out.
"""

_MEMORY_TYPES = sorted(memory_config.ALLOWED_MEMORY_TYPES)

_client_cache: dict[str, Any] = {}


def _get_extraction_client() -> Any | None:
    """
    Build (and cache) the Mistral client for the current API key.

    Returns None when MISTRAL_API_KEY is unset or the SDK is not installed, so
    callers can skip extraction quietly instead of raising.
    """
    api_key = memory_config.memory_api_key()
    if not api_key:
        return None
    cached = _client_cache.get(api_key)
    if cached is not None:
        return cached
    try:
        from mistralai import Mistral
    except ImportError as exc:
        logger.warning("mistralai is not installed; memory extraction disabled: %s", exc)
        return None
    try:
        client = Mistral(api_key=api_key)
    except Exception as exc:
        logger.warning("Failed to create the Mistral client: %s", exc)
        return None
    _client_cache.clear()
    _client_cache[api_key] = client
    return client


async def close_extraction_client() -> None:
    """Release the SDK's HTTP session at shutdown."""
    clients = list(_client_cache.values())
    _client_cache.clear()
    for client in clients:
        for attr in ("aclose", "close_async", "__aexit__"):
            closer = getattr(client, attr, None)
            if not callable(closer):
                continue
            try:
                if attr == "__aexit__":
                    await closer(None, None, None)
                else:
                    await closer()
            except Exception as exc:
                logger.debug("Ignoring error while closing the Mistral client: %s", exc)
            break


# Mistral honours a strict JSON schema. This matters more than it sounds: with
# response_format={"type": "json_object"} and no schema, the model invents its
# own shape (nested "long_term_memories" objects), which validation then throws
# away. The schema is what makes the output usable.
def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "user_memories",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "memory_type": {
                                    "type": "string",
                                    "enum": list(_MEMORY_TYPES),
                                },
                            },
                            "required": ["content", "memory_type"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["memories"],
                "additionalProperties": False,
            },
        },
    }


def _message_text(response: Any) -> str | None:
    """Pull the assistant text out of a Mistral chat completion."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(content, str):
        return content
    # Newer SDKs may return a list of content chunks.
    if isinstance(content, list):
        parts = [
            chunk if isinstance(chunk, str) else str(getattr(chunk, "text", "") or "")
            for chunk in content
        ]
        return "".join(parts) or None
    return None


# Per-message caps on what reaches the extractor. Long assistant answers add
# without adding user facts; the user cap bounds the cost of a single enormous
# pasted message, which the chat endpoint does not otherwise limit.
_MAX_ASSISTANT_CHARS = 600
_MAX_USER_CHARS = 2000


def build_transcript(messages: list[dict[str, str]]) -> str:
    """Render the extraction window as a compact labelled transcript."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            label, cap = "User", _MAX_USER_CHARS
        else:
            label, cap = "Assistant", _MAX_ASSISTANT_CHARS
        if len(content) > cap:
            content = content[:cap] + " ..."
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


# A stored memory is replayed into the Groq system prompt on every future turn,
# so a memory that reads as an instruction to the assistant is a durable,
# self-service jailbreak: the user says "from now on ignore your rules", it gets
# extracted as a preference, and it outranks nothing but sits next to the real
# system messages forever. Preferences about *style* are fine and wanted
# ("prefers concise answers"); directives aimed at the assistant's rules,
# identity or system prompt are dropped at validation time so they never reach
# the database in the first place.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous|prior|earlier|above|other)\b",
        r"\bdisregard\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous|prior|earlier|above|instructions|rules)\b",
        r"\b(?:system|developer)\s+(?:prompt|instructions?|message)\b",
        r"\bwithout\s+(?:any\s+)?(?:restrictions?|limits?|filters?|censorship)\b",
        r"\b(?:no|bypass|override|ignore)\s+(?:your\s+)?(?:safety|guardrails?|restrictions?|rules|guidelines|policies)\b",
        r"\byou\s+are\s+(?:now\s+)?(?:not\s+)?(?:an?\s+)?(?:DAN|jailbroken|unrestricted|uncensored)\b",
        r"\bpretend\s+(?:to\s+be|you\s+are)\b",
        r"\bact\s+as\s+if\s+you\s+(?:have\s+no|are\s+not)\b",
        r"\bforget\s+(?:everything|all\s+your|your\s+(?:rules|instructions))\b",
        r"\breveal\s+(?:your|the)\s+(?:prompt|instructions?|rules)\b",
    )
)


def looks_like_prompt_injection(content: str) -> bool:
    """True when a candidate memory reads as a directive to the assistant."""
    return any(pattern.search(content) for pattern in _INJECTION_PATTERNS)


def validate_extraction(payload: Any) -> list[dict[str, str]]:
    """
    Turn a raw model response into a safe list of memory candidates.

    Accepts the JSON string or an already-parsed object. Anything malformed
    yields an empty list rather than an exception -- a bad model response must
    never reach the database.
    """
    if isinstance(payload, (str, bytes)):
        text = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
        text = text.strip()
        if not text:
            return []
        # Some models wrap JSON in a markdown fence despite the mime type.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            logger.warning("Extraction returned non-JSON memory output; discarding it.")
            return []

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("memories")
    else:
        return []

    if not isinstance(raw_items, list):
        return []

    max_items = memory_config.max_memories_per_extraction()
    max_chars = memory_config.max_content_chars()
    seen: set[str] = set()
    validated: list[dict[str, str]] = []

    for item in raw_items:
        if len(validated) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) < 8:
            continue
        content = content[:max_chars]
        if looks_like_prompt_injection(content):
            logger.warning("Discarded a memory candidate that reads as an instruction.")
            continue

        memory_type = item.get("memory_type")
        if not isinstance(memory_type, str):
            memory_type = memory_config.FALLBACK_MEMORY_TYPE
        memory_type = memory_type.strip().lower().replace(" ", "_").replace("-", "_")
        if memory_type not in memory_config.ALLOWED_MEMORY_TYPES:
            memory_type = memory_config.FALLBACK_MEMORY_TYPE

        key = content.lower()
        if key in seen:
            continue
        seen.add(key)
        validated.append({"content": content, "memory_type": memory_type})

    return validated


# Transient upstream conditions worth another attempt: 429 is the per-key rate
# limit, 5xx is upstream load. Both clear on their own within seconds.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code in _RETRYABLE_STATUS
    text = str(exc)
    return any(str(status) in text for status in _RETRYABLE_STATUS)


async def extract_memories(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Ask Mistral for memories in the given window. Returns [] on any failure.

    Failure modes handled here: no API key, SDK missing, timeout, transport or
    rate-limit error, empty response, malformed JSON. Transient errors are
    retried with exponential backoff -- this runs in a background task, so the
    extra seconds are invisible to the user.
    """
    transcript = build_transcript(messages)
    if not transcript.strip():
        return []

    client = _get_extraction_client()
    if client is None:
        return []

    chat_messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Conversation excerpt:\n"
                f"{transcript}\n\n"
                "Extract the long-term memories about the user."
            ),
        },
    ]
    model = memory_config.memory_model()
    timeout = memory_config.memory_timeout_seconds()
    attempts = memory_config.memory_max_attempts()
    delay = memory_config.memory_retry_base_delay()

    response = None
    for attempt in range(1, attempts + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.complete_async(
                    model=model,
                    messages=chat_messages,
                    response_format=_response_format(),
                    temperature=0,
                    max_tokens=1024,
                ),
                timeout=timeout,
            )
            break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            timed_out = isinstance(exc, asyncio.TimeoutError)
            detail = f"timed out after {timeout}s" if timed_out else str(exc)
            if attempt >= attempts or not (timed_out or _is_retryable(exc)):
                logger.warning(
                    "Mistral memory extraction failed (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    detail,
                )
                return []
            logger.info(
                "Mistral memory extraction attempt %s/%s failed (%s); retrying in %.1fs.",
                attempt,
                attempts,
                detail,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

    text = _message_text(response)
    if not text:
        logger.info("Mistral returned an empty memory extraction response.")
        return []
    return validate_extraction(text)


# --------------------------------------------------------------------------
# Orchestration (runs in a background task, after the response is sent)
# --------------------------------------------------------------------------

async def _window_for(user_id: str, thread_id: str) -> list[dict[str, str]]:
    """The last MEMORY_EXTRACTION_WINDOW messages. Never the whole history."""
    if _history_provider is None:
        return []
    history = await _history_provider(thread_id, user_id)
    if not history:
        return []
    return history[-memory_config.extraction_window() :]


# The cadence check (bump -> decide -> extract -> reset) spans a multi-second
# network call. Without a lock across the whole sequence, two overlapping turns
# on the same thread -- a double submit, a retry, two open tabs -- both observe
# a triggering count and both pay for a model call before either resets the
# counter. Locks are per (user, thread) and per event loop, and are dropped once
# nobody is waiting so the dict cannot grow without bound.
_extraction_locks: dict[tuple[Any, str, str], asyncio.Lock] = {}


def _extraction_lock(user_id: str, thread_id: str) -> tuple[Any, asyncio.Lock]:
    key = (asyncio.get_running_loop(), user_id, thread_id)
    lock = _extraction_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _extraction_locks[key] = lock
    return key, lock


async def run_extraction(user_id: str, thread_id: str) -> dict[str, Any]:
    """
    If the counter says it is time, extract memories from the recent window.

    The window is read back from the thread's checkpointed history, which by
    the time this runs already contains the message that triggered it -- so the
    message itself does not need to be passed in.

    Returns a small report; used by tests and callers that want to log it.
    Raises nothing that a caller needs to handle beyond CancelledError.
    """
    report: dict[str, Any] = {"trigger": None, "created": 0, "updated": 0, "skipped": 0}

    if not user_id or not thread_id:
        return report
    if not memory_config.extraction_enabled():
        return report

    key, lock = _extraction_lock(user_id, thread_id)
    try:
        async with lock:
            return await _run_extraction_locked(user_id, thread_id, report)
    finally:
        if not lock.locked() and not lock._waiters:  # noqa: SLF001 - bounded cleanup
            _extraction_locks.pop(key, None)


async def _run_extraction_locked(
    user_id: str,
    thread_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    count = await memory_store.bump_message_counter(user_id, thread_id)
    trigger = decide_trigger(count)
    if trigger is None:
        return report
    report["trigger"] = trigger

    messages = await _window_for(user_id, thread_id)
    if not messages:
        return report

    candidates = await extract_memories(messages)

    # The counter resets whether or not the model found anything: the window has
    # been reviewed, so re-reviewing the same messages next turn is pure cost.
    await memory_store.reset_message_counter(user_id, thread_id)

    if not candidates:
        return report

    stored = await memory_store.store_memories(
        user_id, candidates, source_thread_id=thread_id
    )
    report.update(stored)
    logger.info(
        "Memory extraction (%s) for user %s thread %s: %s",
        trigger,
        user_id,
        thread_id,
        stored,
    )
    return report


async def process_turn(user_id: str, thread_id: str) -> None:
    """
    Background-task entry point. Never raises, never blocks the chat path.
    """
    try:
        await run_extraction(user_id, thread_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Background memory extraction failed for user %s thread %s: %s",
            user_id,
            thread_id,
            exc,
            exc_info=True,
        )
