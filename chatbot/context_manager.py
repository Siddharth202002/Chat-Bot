"""
Conversation-history lifecycle: when to compress, what to compress, and the
single rolling summary that replaces what was compressed.

Deterministic application logic. The model never decides to summarize; this
module does, from a token estimate:

    projected = system + tool schemas + summary + recent history
                + current turn + reply reserve
    projected >= CONTEXT_COMPRESSION_TRIGGER  ->  fold

A fold hands the summarizer the existing summary plus the "newly old"
messages -- everything before the retained window -- and gets back ONE
updated summary. The retained window is never summarized, and the current
turn (the user's message and this turn's tool calls) is never even offered.

Nothing is deleted. The summary carries a cursor, ``through_id``: the id of
the last message it covers. The checkpointed transcript keeps every message,
so the UI, Edit/Retry and memory extraction see the full conversation, while
the model is shown only what lies after the cursor. Summary and cursor are one
value, written in the same checkpoint as the turn that produced them, so they
cannot drift apart and a failed turn leaves both exactly as they were.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

import context_config
import context_summary
from context_builder import format_summary_block
from context_tokens import TokenCounter, count_messages, get_counter

logger = logging.getLogger("chatbot.context")


class ContextSummary(TypedDict, total=False):
    """The rolling summary as stored in the graph state's ``context_summary``."""

    text: str
    # Id of the newest message folded into ``text``. Everything up to and
    # including it is represented only by the summary in the model's context.
    through_id: str
    prompt_version: str
    model: str
    updated_at: str
    # Running total of messages represented by the summary.
    folded_messages: int
    tokens: int


@dataclass(frozen=True)
class TokenUsage:
    system: int
    tools: int
    summary: int
    history: int
    turn: int
    output_reserve: int

    @property
    def input(self) -> int:
        return self.system + self.tools + self.summary + self.history + self.turn

    @property
    def projected(self) -> int:
        return self.input + self.output_reserve


@dataclass(frozen=True)
class ContextPlan:
    # The summary the model should see, and that the turn should persist when
    # ``summary_changed`` is set. None means "no summary".
    summary: ContextSummary | None
    # Prior messages to show the model, oldest first. Excludes the current turn.
    history: list[BaseMessage]
    summary_changed: bool
    usage_before: TokenUsage
    usage_after: TokenUsage
    compressed_messages: int = 0
    trimmed_messages: int = 0
    # below_trigger | compressed | compressed_insufficient | failed |
    # nothing_to_compress | disabled
    outcome: str = "below_trigger"


def summary_text(summary: ContextSummary | None) -> str:
    return (summary or {}).get("text", "") or ""


def active_history(
    messages: Sequence[BaseMessage], summary: ContextSummary | None
) -> tuple[list[BaseMessage], ContextSummary | None]:
    """
    The messages after the summary's cursor, and the summary if it is usable.

    A cursor that names no message on this branch means the summary describes
    some other history. It is dropped rather than trusted -- the full message
    list is always a safe, if larger, context.
    """
    messages = list(messages)
    if not summary or not summary.get("text") or not summary.get("through_id"):
        return messages, None
    through_id = summary["through_id"]
    for index, message in enumerate(messages):
        if message.id == through_id:
            return messages[index + 1 :], summary
    logger.warning(
        "context_summary_stale reason=cursor_not_found messages=%d", len(messages)
    )
    return messages, None


def last_turn_start(messages: Sequence[BaseMessage]) -> int | None:
    """Index of the last user message: where the in-flight turn begins."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return None


def choose_cut(history: Sequence[BaseMessage], keep: int) -> int:
    """
    How many leading messages of ``history`` to fold, keeping at least ``keep``.

    The cut lands on a user message, so the retained window starts a turn. A
    cut anywhere else could separate a tool call from its result, which every
    provider rejects outright. 0 means there is nothing that can be folded.
    """
    for index in range(len(history) - keep, 0, -1):
        if isinstance(history[index], HumanMessage):
            return index
    return 0


def _next_turn(history: Sequence[BaseMessage]) -> int:
    for index in range(1, len(history)):
        if isinstance(history[index], HumanMessage):
            return index
    return len(history)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ContextManager:
    def __init__(
        self,
        summarizer: context_summary.SummaryService | None,
        *,
        settings_provider: Callable[[], context_config.ContextSettings] = context_config.load,
        counter_factory: Callable[[str], TokenCounter] = get_counter,
        cache_size: int = 128,
    ) -> None:
        self._summarizer = summarizer
        self._settings = settings_provider
        self._counter_factory = counter_factory
        # One lock per thread, so two requests on the same conversation cannot
        # both fold it, while different conversations never wait on each
        # other. Weak values: a lock disappears once no request holds it.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        # Folds already computed, by (thread, base summary, target cursor).
        # The second of two concurrent requests reuses the first one's result,
        # and so does a turn retried after the chat providers failed -- the
        # fold is persisted only with a completed turn, so without this the
        # retry would pay for the same summary again.
        self._cache: OrderedDict[tuple[str, str, str, str], ContextSummary] = OrderedDict()
        self._cache_size = cache_size

    # --- measurement ----------------------------------------------------

    def measure(
        self,
        *,
        system_tokens: int,
        tool_tokens: int,
        summary: ContextSummary | None,
        history: Sequence[BaseMessage],
        turn: Sequence[BaseMessage],
        settings: context_config.ContextSettings | None = None,
        counter: TokenCounter | None = None,
    ) -> TokenUsage:
        settings = settings or self._settings()
        counter = counter or self._counter_factory(settings.tokenizer)
        return TokenUsage(
            system=max(0, system_tokens),
            tools=max(0, tool_tokens),
            summary=counter.count_text(format_summary_block(summary_text(summary))),
            history=count_messages(counter, history),
            turn=count_messages(counter, turn),
            output_reserve=settings.output_reserve_tokens,
        )

    # --- the per-turn entry point ---------------------------------------

    async def prepare(
        self,
        *,
        thread_id: str,
        prior_messages: Sequence[BaseMessage],
        summary: ContextSummary | None,
        turn: Sequence[BaseMessage],
        system_tokens: int,
        tool_tokens: int = 0,
    ) -> ContextPlan:
        """
        Decide this turn's history window, folding old messages if needed.

        ``prior_messages`` is the checkpointed conversation before this turn;
        ``turn`` is the new user message. Never raises for a summarizer
        failure: the worst case is the old summary and a trimmed window.
        """
        settings = self._settings()
        counter = self._counter_factory(settings.tokenizer)
        history, current = active_history(prior_messages, summary)

        def usage_of(s: ContextSummary | None, h: Sequence[BaseMessage]) -> TokenUsage:
            return self.measure(
                system_tokens=system_tokens,
                tool_tokens=tool_tokens,
                summary=s,
                history=h,
                turn=turn,
                settings=settings,
                counter=counter,
            )

        before = usage_of(current, history)
        usage = before
        outcome = "below_trigger"
        compressed = 0
        changed = False

        if not settings.enabled:
            outcome = "disabled"
        elif usage.projected >= settings.trigger:
            for attempt in range(settings.max_compression_attempts):
                # Later passes (only with MAX_COMPRESSION_ATTEMPTS > 1) retain
                # half as much again; a pass that could fold nothing ends it.
                keep = max(2, settings.recent_message_count >> attempt)
                cut = choose_cut(history, keep)
                if cut <= 0 or not history[cut - 1].id:
                    if not changed:
                        outcome = "nothing_to_compress"
                    break
                folded = await self._fold(
                    thread_id, current, list(history[:cut]), settings, counter, usage
                )
                if folded is None:
                    outcome = "failed"
                    break
                current = folded
                history = history[cut:]
                compressed += cut
                changed = True
                usage = usage_of(current, history)
                outcome = "compressed"
                if usage.projected < settings.trigger:
                    break
            if changed and usage.projected >= settings.trigger:
                # Not retried: MAX_COMPRESSION_ATTEMPTS bounds summarizer calls
                # per turn, and the hard-limit trim below still keeps the
                # prompt inside the budget.
                outcome = "compressed_insufficient"
                logger.warning(
                    "context_compression_insufficient thread=%s projected=%d trigger=%d",
                    thread_id,
                    usage.projected,
                    settings.trigger,
                )

        history, trimmed = self._fit(
            history,
            summary=current,
            turn=turn,
            system_tokens=system_tokens,
            tool_tokens=tool_tokens,
            settings=settings,
            counter=counter,
            thread_id=thread_id,
        )
        after = usage_of(current, history)

        log = logger.info if outcome != "below_trigger" else logger.debug
        log(
            "context_plan thread=%s outcome=%s projected_before=%d projected_after=%d "
            "trigger=%d budget=%d compressed=%d retained=%d trimmed=%d summary_tokens=%d",
            thread_id,
            outcome,
            before.projected,
            after.projected,
            settings.trigger,
            settings.budget,
            compressed,
            len(history),
            trimmed,
            after.summary,
        )
        return ContextPlan(
            summary=current,
            history=list(history),
            summary_changed=changed,
            usage_before=before,
            usage_after=after,
            compressed_messages=compressed,
            trimmed_messages=trimmed,
            outcome=outcome,
        )

    def window_for_model(
        self,
        messages: Sequence[BaseMessage],
        summary: ContextSummary | None,
        *,
        system_tokens: int,
        tool_tokens: int = 0,
    ) -> tuple[list[BaseMessage], ContextSummary | None]:
        """
        The conversation to show the model, for a caller holding full state.

        Used by the graph's chat node, which runs once per tool round: the
        in-flight turn (last user message onward) is always kept whole, and
        only older history is trimmed to fit.
        """
        settings = self._settings()
        counter = self._counter_factory(settings.tokenizer)
        history, current = active_history(messages, summary)
        start = last_turn_start(history)
        if start is None:
            return history, current
        prior, turn = history[:start], history[start:]
        prior, _ = self._fit(
            prior,
            summary=current,
            turn=turn,
            system_tokens=system_tokens,
            tool_tokens=tool_tokens,
            settings=settings,
            counter=counter,
            thread_id=None,
        )
        return [*prior, *turn], current

    # --- internals ------------------------------------------------------

    def _fit(
        self,
        history: Sequence[BaseMessage],
        *,
        summary: ContextSummary | None,
        turn: Sequence[BaseMessage],
        system_tokens: int,
        tool_tokens: int,
        settings: context_config.ContextSettings,
        counter: TokenCounter,
        thread_id: str | None,
    ) -> tuple[list[BaseMessage], int]:
        """
        Last-resort reduction: drop the oldest whole turns from the prompt.

        Only the prompt shrinks -- the messages stay in the checkpoint, after
        the cursor, so the next successful fold still summarizes them. This is
        what keeps a turn answerable while the summarizer is down.
        """
        history = list(history)
        trimmed = 0

        def projected() -> int:
            return self.measure(
                system_tokens=system_tokens,
                tool_tokens=tool_tokens,
                summary=summary,
                history=history,
                turn=turn,
                settings=settings,
                counter=counter,
            ).projected

        while history and projected() > settings.hard_limit:
            drop = _next_turn(history)
            history = history[drop:]
            trimmed += drop
        if trimmed:
            logger.warning(
                "context_hard_trim thread=%s dropped_messages=%d hard_limit=%d",
                thread_id,
                trimmed,
                settings.hard_limit,
            )
        if projected() > settings.hard_limit:
            # The current turn alone is over budget (a huge paste). It is sent
            # anyway: refusing would lose the user's message, and the provider
            # chain already handles a provider that rejects the request.
            logger.warning(
                "context_over_budget thread=%s projected=%d hard_limit=%d",
                thread_id,
                projected(),
                settings.hard_limit,
            )
        return history, trimmed

    def _lock_for(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock

    async def _fold(
        self,
        thread_id: str,
        previous: ContextSummary | None,
        chunk: list[BaseMessage],
        settings: context_config.ContextSettings,
        counter: TokenCounter,
        usage: TokenUsage,
    ) -> ContextSummary | None:
        previous_text = summary_text(previous)
        key = (
            thread_id,
            (previous or {}).get("through_id", ""),
            _fingerprint(previous_text),
            str(chunk[-1].id),
        )
        lock = self._lock_for(thread_id)
        async with lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                logger.info(
                    "context_summary_reused thread=%s through=%s", thread_id, key[3]
                )
                return cached

            if self._summarizer is None:
                logger.warning("context_summary_failed thread=%s reason=no_summarizer", thread_id)
                return None

            try:
                result = await self._summarizer.summarize(
                    context_summary.SummaryRequest(
                        previous_summary=previous_text,
                        messages=chunk,
                        target_tokens=settings.summary_target_tokens,
                        max_tokens=settings.summary_max_tokens,
                    )
                )
            except Exception as exc:
                # Nothing has been changed yet: the old summary and every
                # message are still in place, so the turn simply goes ahead
                # without folding.
                logger.warning(
                    "context_summary_failed thread=%s model=%s messages=%d error=%s",
                    thread_id,
                    getattr(self._summarizer, "model_name", ""),
                    len(chunk),
                    type(exc).__name__,
                )
                return None

            new_tokens = counter.count_text(result.text)
            folded_tokens = count_messages(counter, chunk)
            previous_tokens = counter.count_text(previous_text)
            folded: ContextSummary = {
                "text": result.text,
                "through_id": str(chunk[-1].id),
                "prompt_version": context_summary.SUMMARY_PROMPT_VERSION,
                "model": result.model,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "folded_messages": int((previous or {}).get("folded_messages", 0)) + len(chunk),
                "tokens": new_tokens,
            }
            logger.info(
                "context_summary_updated thread=%s model=%s provider=%s fallback=%s "
                "latency_ms=%d folded_messages=%d folded_tokens=%d previous_summary_tokens=%d "
                "new_summary_tokens=%d compression_ratio=%.2f omitted=%d projected=%d",
                thread_id,
                result.model,
                result.provider,
                result.used_fallback,
                result.latency_ms,
                len(chunk),
                folded_tokens,
                previous_tokens,
                new_tokens,
                new_tokens / max(1, folded_tokens + previous_tokens),
                result.omitted_messages,
                usage.projected,
            )
            self._cache[key] = folded
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return folded
