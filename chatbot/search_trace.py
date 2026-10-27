"""
TEMPORARY diagnostics for the web-search accuracy investigation.

A user asked for an IPO's listing date and got its opening date back. This
module logs every hop between the question and the answer so the failure point
can be read off the logs instead of guessed:

    TURN_TRACE      turn start, context plan, what was persisted
    SEARCH_TRACE    ddgs dispatch plan, one line per backend HTTP response
    SEARCH_RESULTS  per-backend parsed results, ranking, the final result set
    SEARCH_CONTEXT  the exact tool output handed back to the chat model
    LLM_INPUT       what a chat provider was sent (redacted, clipped)
    LLM_OUTPUT      what it returned (tool calls or answer text)
    MISTRAL_TRACE   the background memory-extraction call

Every line carries ``turn=<id>``, a per-turn correlation id held in a
contextvar, so concurrent /api/chat/stream requests never mix. ddgs runs its
engines on a shared thread pool, which does not propagate contextvars by
itself; install_ddgs_trace() swaps in a pool that does.

Nothing here changes what is searched, ranked, prompted or answered. The ddgs
hooks wrap the library's own functions and return their results untouched.
Set SEARCH_TRACE=0 to silence it; delete this module and its call sites once
the root cause is fixed.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import ceil
from time import monotonic
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

logger = logging.getLogger("chatbot.trace")

# The HTTP hook re-implements ddgs.http_client.HttpClient.request (the library
# drops response headers, which is where the content type lives), so it is only
# installed on the version it was written against.
_DDGS_VERSION = "9.11.4"

_turn_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "search_trace_turn_id", default=None
)
# One dict per DDGS._search call, shared by reference with the engine threads.
_search_run: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "search_trace_run", default=None
)
# The HTTP response an engine just received, read back by the engine hook.
_last_http: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "search_trace_last_http", default=None
)

# thread_id -> the last turn id that ran on it, so the memory-extraction
# background task (which runs after the turn's context is gone) can log under
# the same id. Bounded; a miss only costs the correlation, never the turn.
_thread_turns: OrderedDict[str, str] = OrderedDict()
_THREAD_TURNS_MAX = 512

_installed = False


def enabled() -> bool:
    return (os.getenv("SEARCH_TRACE") or "1").strip().lower() not in {"0", "false", "no", "off"}


# --- Correlation ids --------------------------------------------------------


def turn_id() -> str:
    return _turn_id.get() or "-"


@contextlib.contextmanager
def turn(tid: str | None = None) -> Iterator[str]:
    """Bind a correlation id to everything logged inside the block."""
    tid = tid or uuid4().hex[:12]
    token = _turn_id.set(tid)
    try:
        yield tid
    finally:
        _turn_id.reset(token)


def remember_thread_turn(thread_id: str, tid: str) -> None:
    _thread_turns[thread_id] = tid
    _thread_turns.move_to_end(thread_id)
    while len(_thread_turns) > _THREAD_TURNS_MAX:
        _thread_turns.popitem(last=False)


def turn_for_thread(thread_id: str) -> str | None:
    return _thread_turns.get(thread_id)


# --- Formatting helpers -----------------------------------------------------


def emit(prefix: str, event: str, **fields: Any) -> None:
    if not enabled():
        return
    try:
        payload = json.dumps(fields, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - a diagnostic must never break a turn
        payload = repr(fields)
    logger.info("%s turn=%s event=%s %s", prefix, turn_id(), event, payload)


_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [redacted]"),
    (re.compile(r"\b(?:sk|gsk|pk|rk)[-_][A-Za-z0-9_-]{12,}"), "[redacted-key]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"), "[redacted-key]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|cookie)\s*[=:]\s*\S+"), r"\1=[redacted]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)"), "[phone]"),
)


def redact(text: Any) -> str:
    text = "" if text is None else str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clip(text: Any, limit: int) -> str:
    text = " ".join(("" if text is None else str(text)).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def digest(text: Any) -> str:
    """Short content hash: proves two log lines saw byte-identical text."""
    return hashlib.sha256(("" if text is None else str(text)).encode("utf-8")).hexdigest()[:10]


# --- Date evidence ----------------------------------------------------------
#
# Generic on purpose: no expected answer is baked in. Every date mentioned in a
# result is listed with the IPO-calendar word closest before it, so a log
# reader can see which source calls which date the "listing" date.

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DATE_RE = re.compile(
    rf"\b(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",
    re.IGNORECASE,
)
# No leading \b: Yahoo snippets glue highlighted words together
# ("IPObidding started", "tentativelistingdatefixed"), which a word boundary
# would miss.
_CUES = (
    ("listing", re.compile(r"(?i)list(?:ing|s|ed)?|debut")),
    ("open", re.compile(r"(?i)open(?:s|ing|ed)?|subscription|bidding|ipo\s*date\s*is")),
    ("close", re.compile(r"(?i)clos(?:e|es|ing|ed)|ended")),
    ("allotment", re.compile(r"(?i)allot")),
    ("refund", re.compile(r"(?i)refund")),
    ("demat_credit", re.compile(r"(?i)credit")),
    ("published", re.compile(r"(?i)(?:updated|published|posted)\b")),
)


def _normalise_date(raw: str) -> str:
    cleaned = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"[.,]", " ", cleaned).replace("Sept", "Sep")
    cleaned = " ".join(cleaned.split())
    for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            pass
    for fmt in ("%B %d", "%b %d", "%d %B", "%d %b"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("--%m-%d")
        except ValueError:
            pass
    return raw


def date_mentions(text: Any, limit: int = 10) -> list[dict[str, Any]]:
    text = "" if text is None else str(text)
    mentions: list[dict[str, Any]] = []
    for match in _DATE_RE.finditer(text):
        before = text[max(0, match.start() - 70) : match.start()]
        cue, cue_at = None, -1
        for name, pattern in _CUES:
            for hit in pattern.finditer(before):
                if hit.start() > cue_at:
                    cue, cue_at = name, hit.start()
        mentions.append(
            {
                "date": _normalise_date(match.group(0)),
                "raw": match.group(0),
                "cue": cue,
                "ctx": clip(text[max(0, match.start() - 70) : match.end() + 30], 140),
            }
        )
        if len(mentions) >= limit:
            break
    return mentions


def _probes() -> list[str]:
    """Optional literal strings to flag, e.g. SEARCH_TRACE_PROBES='Oct 28|October 21'."""
    raw = os.getenv("SEARCH_TRACE_PROBES") or ""
    return [p.strip() for p in raw.split("|") if p.strip()]


def probe_hits(text: Any) -> list[str]:
    lowered = ("" if text is None else str(text)).lower()
    return [probe for probe in _probes() if probe.lower() in lowered]


def date_summary(texts: Sequence[Any]) -> dict[str, list[str]]:
    """Every distinct date across ``texts`` -> the cues it appeared under."""
    summary: dict[str, set[str]] = {}
    for text in texts:
        for mention in date_mentions(text, limit=40):
            summary.setdefault(mention["date"], set()).add(mention["cue"] or "none")
    return {date: sorted(cues) for date, cues in sorted(summary.items())}


def _as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return dict(getattr(result, "__dict__", {}) or {})


def result_view(result: Any, body_limit: int = 320) -> dict[str, Any]:
    data = _as_dict(result)
    body = data.get("body", data.get("description", ""))
    text = f"{data.get('title', '')} {body}"
    view = {
        "title": clip(data.get("title"), 160),
        "url": data.get("href") or data.get("url"),
        "snippet": clip(body, body_limit),
        # ddgs text results have no date or source field at all; only news does.
        "published": data.get("date") or None,
        "source": data.get("source") or None,
        "dates": date_mentions(text, limit=6),
    }
    hits = probe_hits(text)
    if hits:
        view["probe_hits"] = hits
    return view


def _result_key(result: Any) -> str:
    data = _as_dict(result)
    return str(data.get("href") or data.get("url") or data.get("image") or "")


# --- Message / LLM boundary -------------------------------------------------


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and not item.get("thought"):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def _system_view(text: str) -> dict[str, Any]:
    head = text.split("\n", 1)[0]
    if text.startswith("LONG-TERM USER MEMORY"):
        lines = sum(1 for line in text.splitlines() if line.startswith("- ("))
        return {"kind": "user_memory", "memories": lines, "chars": len(text), "content": "[redacted]"}
    if text.startswith("CONVERSATION SUMMARY"):
        return {
            "kind": "conversation_summary",
            "chars": len(text),
            "sha": digest(text),
            "excerpt": clip(redact(text), 400),
            "dates": date_mentions(text, limit=6),
        }
    return {"kind": clip(head, 70), "chars": len(text), "sha": digest(text)}


def _search_instructions(system_texts: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for text in system_texts:
        if text.startswith(("LONG-TERM USER MEMORY", "CONVERSATION SUMMARY")):
            continue
        for line in re.split(r"\n(?=- )", text):
            if re.search(r"(?i)search|news|up-to-date|current|today|date", line):
                lines.append(clip(line, 260))
    return lines


def _mentions_today(system_texts: Sequence[str]) -> bool:
    now = datetime.now()
    needles = {now.strftime("%Y-%m-%d"), now.strftime("%B %d, %Y"), now.strftime("%d %B %Y")}
    return any(needle in text for text in system_texts for needle in needles)


def log_llm_input(
    provider: str, model: str, round_index: int, messages: Sequence[Any]
) -> None:
    if not enabled():
        return
    try:
        system_texts = [_message_text(m) for m in messages if getattr(m, "type", "") == "system"]
        conversation = [m for m in messages if getattr(m, "type", "") != "system"]
        # The current turn starts at the last human message; everything before
        # it is history the model can be swayed by.
        last_human = max(
            (i for i, m in enumerate(conversation) if getattr(m, "type", "") == "human"),
            default=0,
        )
        history, current = conversation[:last_human], conversation[last_human:]
        current_view = []
        for m in current:
            kind = getattr(m, "type", "")
            text = _message_text(m)
            entry: dict[str, Any] = {"role": kind, "chars": len(text), "sha": digest(text)}
            if kind == "human":
                entry["text"] = clip(redact(text), 400)
            elif kind == "ai":
                entry["tool_calls"] = [
                    {"name": c.get("name"), "args": clip(redact(json.dumps(c.get("args"), default=str)), 300)}
                    for c in (getattr(m, "tool_calls", None) or [])
                ]
                if text:
                    entry["text"] = clip(redact(text), 300)
            elif kind == "tool":
                entry["tool"] = getattr(m, "name", None)
                # Logged in full (clipped) under SEARCH_CONTEXT; the sha ties
                # this line to that one.
                entry["excerpt"] = clip(text, 300)
            current_view.append(entry)
        emit(
            "LLM_INPUT",
            "request",
            provider=provider,
            model=model,
            round=round_index,
            message_count=len(messages),
            total_chars=sum(len(_message_text(m)) for m in messages),
            system=[_system_view(t) for t in system_texts],
            system_mentions_today=_mentions_today(system_texts),
            search_instructions=_search_instructions(system_texts),
            history_count=len(history),
            history_tail=[
                {
                    "role": getattr(m, "type", ""),
                    "chars": len(_message_text(m)),
                    "excerpt": clip(redact(_message_text(m)), 160),
                    "dates": date_mentions(_message_text(m), limit=4),
                }
                for m in history[-6:]
            ],
            current_turn=current_view,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM_INPUT turn=%s trace failed: %s", turn_id(), exc)


def log_llm_output(provider: str | None, model: str, round_index: int, message: Any) -> None:
    if not enabled():
        return
    try:
        text = _message_text(message)
        metadata = getattr(message, "response_metadata", None) or {}
        tool_calls = [
            {"name": c.get("name"), "args": clip(redact(json.dumps(c.get("args"), default=str)), 400)}
            for c in (getattr(message, "tool_calls", None) or [])
        ]
        emit(
            "LLM_OUTPUT",
            "response",
            provider=provider,
            model=model,
            round=round_index,
            kind="tool_calls" if tool_calls else "answer",
            tool_calls=tool_calls,
            finish_reason=metadata.get("finish_reason"),
            usage=getattr(message, "usage_metadata", None),
            chars=len(text),
            text=clip(redact(text), 2000),
            dates=date_mentions(text),
            probe_hits=probe_hits(text),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM_OUTPUT turn=%s trace failed: %s", turn_id(), exc)


def log_tool_result(
    tool_name: str, tool_args: Any, result: Any, *, failed: bool, round_index: int | None = None
) -> None:
    """The tool output exactly as it goes back into the model's context."""
    if not enabled():
        return
    text = "" if result is None else str(result)
    emit(
        "SEARCH_CONTEXT",
        "tool_result",
        tool=tool_name,
        round=round_index,
        args=clip(redact(json.dumps(tool_args, default=str)), 400),
        failed=failed,
        chars=len(text),
        sha=digest(text),
        text=clip(text, 4000),
        dates=date_mentions(text, limit=20),
        date_summary=date_summary([text]),
        probe_hits=probe_hits(text),
    )


# --- ddgs hooks -------------------------------------------------------------


class _ContextThreadPool(ThreadPoolExecutor):
    """ThreadPoolExecutor that runs each task in a copy of the caller's context."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


_CHALLENGE_MARKERS = (
    "captcha",
    "are you a robot",
    "unusual traffic",
    "anomaly-modal",
    "challenge-form",
    "cf-challenge",
    "verify you are human",
    "bots use duckduckgo too",
)


def _classify(status: int, content_type: str, text: str) -> str:
    lowered = text[:30000].lower()
    challenge = any(marker in lowered for marker in _CHALLENGE_MARKERS)
    if status == 200:
        return "possible_challenge_page" if challenge else "html_ok"
    if status == 202 or challenge:
        return "anti_bot_challenge"
    if status == 429:
        return "rate_limited"
    if status >= 400:
        return "error_page"
    return "non_200"


def _install_http_hook() -> None:
    import primp
    from ddgs import http_client
    from ddgs.exceptions import DDGSException, TimeoutException

    def request(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Same behaviour as HttpClient.request in ddgs 9.11.4, plus one log line.
        method = args[0] if args else kwargs.get("method")
        url = str(args[1] if len(args) > 1 else kwargs.get("url") or "")
        parts = urlsplit(url)
        where = {"host": parts.netloc, "path": parts.path, "method": method}
        started = monotonic()
        try:
            resp = self.client.request(*args, **kwargs)
        except primp.TimeoutError as ex:
            emit("SEARCH_TRACE", "http", **where, outcome="timeout", elapsed_ms=int((monotonic() - started) * 1000))
            raise TimeoutException(ex) from ex
        except Exception as ex:
            emit("SEARCH_TRACE", "http", **where, outcome="transport_error", error=clip(repr(ex), 200))
            msg = f"{type(ex).__name__}: {ex!r}"
            raise DDGSException(msg) from ex
        text = resp.text
        try:
            headers = {str(k).lower(): str(v) for k, v in dict(resp.headers).items()}
        except Exception:  # noqa: BLE001
            headers = {}
        content_type = headers.get("content-type", "")
        status = int(resp.status_code)
        classification = _classify(status, content_type, text or "")
        title = re.search(r"(?is)<title[^>]*>(.*?)</title>", text or "")
        info = {
            **where,
            "status": status,
            "content_type": content_type,
            "bytes": len(resp.content or b""),
            "elapsed_ms": int((monotonic() - started) * 1000),
            "classification": classification,
            "page_title": clip(title.group(1), 120) if title else None,
        }
        if status != 200:
            info["body_excerpt"] = clip(re.sub(r"<[^>]+>", " ", text or ""), 200)
        emit("SEARCH_TRACE", "http", **info)
        _last_http.set(info)
        return http_client.Response(status_code=resp.status_code, content=resp.content, text=text)

    http_client.HttpClient.request = request


def install_ddgs_trace() -> None:
    """Wrap the ddgs dispatch, engine, HTTP and ranking steps with logging."""
    global _installed
    if _installed or not enabled():
        return
    try:
        import ddgs
        from ddgs.base import BaseSearchEngine
        # Not `from ddgs import DDGS`: that is a lazy proxy class, and setting
        # attributes on it never reaches the real one.
        from ddgs.ddgs import DDGS
        from ddgs.similarity import SimpleFilterRanker
    except ImportError as exc:
        logger.warning("SEARCH_TRACE ddgs not importable; search tracing disabled: %s", exc)
        return
    _installed = True

    # Same pool size and naming as DDGS.get_executor; only context propagation
    # is added, so engine-thread log lines keep the turn id.
    if DDGS._executor is None:  # noqa: SLF001
        DDGS._executor = _ContextThreadPool(max_workers=DDGS.threads, thread_name_prefix="DDGS")  # noqa: SLF001

    if getattr(ddgs, "__version__", None) == _DDGS_VERSION:
        _install_http_hook()
    else:
        logger.warning(
            "SEARCH_TRACE ddgs %s != %s; per-backend HTTP status/content-type tracing skipped.",
            getattr(ddgs, "__version__", "?"),
            _DDGS_VERSION,
        )

    original_search = DDGS._search  # noqa: SLF001
    original_get_engines = DDGS._get_engines  # noqa: SLF001
    original_engine_search = BaseSearchEngine.search
    original_extract_results = BaseSearchEngine.extract_results
    original_rank = SimpleFilterRanker.rank

    def extract_results(self: Any, html_text: str) -> Any:
        # Items the engine's items_xpath matched, before post_extract_results
        # filters (ads, redirects). Separates "markup changed" from "filtered".
        results = original_extract_results(self, html_text)
        http = _last_http.get()
        if http is not None:
            http["items_matched_xpath"] = len(results)
        return results

    def _search(self: Any, category: str, query: str, keywords: str | None = None, **kwargs: Any) -> Any:
        run: dict[str, Any] = {
            "category": category,
            "max_results": kwargs.get("max_results", 10),
            "engines": [],
            "href_engine": {},
        }
        token = _search_run.set(run)
        emit(
            "SEARCH_TRACE",
            "dispatch",
            category=category,
            query=clip(redact(keywords or query), 300),
            backend=kwargs.get("backend", "auto"),
            env_SEARCH_BACKENDS=os.getenv("SEARCH_BACKENDS"),
            region=kwargs.get("region"),
            timelimit=kwargs.get("timelimit"),
            max_results=run["max_results"],
        )
        try:
            try:
                results = original_search(self, category, query, keywords, **kwargs)
            except Exception as exc:
                emit(
                    "SEARCH_RESULTS",
                    "final",
                    category=category,
                    outcome="no_results",
                    error=clip(repr(exc), 300),
                    engines_called=run["engines"],
                    engines_not_called=_not_called(run),
                )
                raise
            ranked = run.get("ranked") or results
            final_keys = {_result_key(r) for r in results}
            raw_count = sum(e.get("parsed") or 0 for e in run["engines"])
            contributors = sorted({e["engine"] for e in run["engines"] if e.get("parsed")})
            emit(
                "SEARCH_RESULTS",
                "final",
                category=category,
                raw_count=raw_count,
                deduped_count=run.get("deduped_count", len(results)),
                final_count=len(results),
                engines_called=run["engines"],
                engines_not_called=_not_called(run),
                selected_from=contributors,
                selection_reason=(
                    "ddgs merges every engine that returned parsed results, dedups "
                    "by URL, ranks with SimpleFilterRanker and cuts to max_results; "
                    "an engine is skipped once another engine with the same "
                    "provider has returned results"
                ),
                cut_by_max_results=[
                    _result_key(r) for r in ranked if _result_key(r) not in final_keys
                ],
                results=[
                    {**result_view(r), "engine": run["href_engine"].get(_result_key(r))}
                    for r in results
                ],
                date_summary=date_summary(
                    f"{_as_dict(r).get('title', '')} {_as_dict(r).get('body', '')}" for r in results
                ),
            )
            return results
        finally:
            _search_run.reset(token)

    def _not_called(run: dict[str, Any]) -> list[str]:
        called = {e["engine"] for e in run["engines"]}
        return [name for name, _ in run.get("plan", []) if name not in called]

    def _get_engines(self: Any, category: str, backend: str) -> Any:
        engines = original_get_engines(self, category, backend)
        run = _search_run.get()
        plan = [(e.name, e.provider) for e in engines]
        unique = len({provider for _, provider in plan})
        max_results = run.get("max_results") if run else None
        if run is not None:
            run["plan"] = plan
        emit(
            "SEARCH_TRACE",
            "engine_plan",
            category=category,
            requested=backend,
            order=[{"engine": n, "provider": p} for n, p in plan],
            unique_providers=unique,
            # Mirrors the formula in DDGS._search: engines are submitted this
            # many at a time, so the first ones run in parallel and log lines
            # arrive in completion order, not submission order.
            parallel_batch=min(unique, ceil(max_results / 10) + 1) if max_results else unique,
        )
        return engines

    def engine_search(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        _last_http.set(None)
        started = monotonic()
        record: dict[str, Any] = {"engine": self.name, "provider": self.provider}
        try:
            results = original_engine_search(self, query, *args, **kwargs)
        except Exception as exc:
            record.update(outcome="exception", error=clip(repr(exc), 200))
            raise
        else:
            parsed = len(results or [])
            if results is None:
                record["outcome"] = "no_body (non-200 response, ddgs discards it)"
            else:
                record["outcome"] = "parsed" if parsed else "zero_parsed (page had no result items)"
            record["parsed"] = parsed
            http = _last_http.get() or {}
            emit(
                "SEARCH_RESULTS",
                "engine_results",
                engine=self.name,
                provider=self.provider,
                status=http.get("status"),
                classification=http.get("classification"),
                items_matched_xpath=http.get("items_matched_xpath"),
                outcome=record["outcome"],
                query=clip(redact(query), 300),
                timelimit=kwargs.get("timelimit"),
                parsed=parsed,
                top5=[result_view(r) for r in (results or [])[:5]],
            )
            run = _search_run.get()
            if run is not None:
                for r in results or []:
                    run["href_engine"].setdefault(_result_key(r), self.name)
            return results
        finally:
            http = _last_http.get() or {}
            record.update(
                status=http.get("status"),
                content_type=http.get("content_type"),
                classification=http.get("classification"),
                elapsed_ms=int((monotonic() - started) * 1000),
            )
            run = _search_run.get()
            if run is not None:
                run["engines"].append(record)

    def rank(self: Any, docs: list[dict[str, Any]], query: str) -> Any:
        ranked = original_rank(self, docs, query)
        run = _search_run.get()
        if run is not None:
            run["ranked"] = ranked
            run["deduped_count"] = len(docs)
        tokens = self._extract_tokens(query)
        kept = {id(d) for d in ranked}

        def bucket(doc: dict[str, Any]) -> str:
            if "wikipedia.org" in doc.get("href", ""):
                return "wikipedia"
            title = self._has_any_token(doc.get("title", ""), tokens)
            body = self._has_any_token(doc.get("body", doc.get("description", "")), tokens)
            return "both" if title and body else "title" if title else "body" if body else "neither"

        emit(
            "SEARCH_RESULTS",
            "rank",
            deduped_count=len(docs),
            ranked_count=len(ranked),
            query_tokens=sorted(tokens),
            order=[
                {
                    "url": _result_key(d),
                    "bucket": bucket(d),
                    "engine": (run or {}).get("href_engine", {}).get(_result_key(d)),
                }
                for d in ranked
            ],
            dropped=[
                {"url": _result_key(d), "reason": "wikimedia category page"}
                for d in docs
                if id(d) not in kept
            ],
        )
        return ranked

    DDGS._search = _search  # noqa: SLF001
    DDGS._get_engines = _get_engines  # noqa: SLF001
    BaseSearchEngine.search = engine_search
    BaseSearchEngine.extract_results = extract_results
    SimpleFilterRanker.rank = rank
    logger.info("SEARCH_TRACE ddgs %s tracing installed.", getattr(ddgs, "__version__", "?"))
