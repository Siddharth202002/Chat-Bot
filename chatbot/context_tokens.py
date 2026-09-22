"""
Token estimation for context budgeting.

The chain spans Groq (gpt-oss, qwen) and Gemini, which tokenize differently,
so no single tokenizer is exact for every turn. The default is therefore a
deterministic, dependency-free estimate that leans high; the budget's safety
margin absorbs the rest. CONTEXT_TOKENIZER=tiktoken swaps in o200k (gpt-oss's
encoding) when it can be loaded -- it may need a one-time download, which is
why it is opt-in and falls back to the estimate if loading fails.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Iterable, Protocol

from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger("chatbot.context")

# Role markers and separators each provider wraps around a message.
MESSAGE_OVERHEAD_TOKENS = 4


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


def content_text(content: Any) -> str:
    """Every text part of a message's content; thought parts are skipped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                if item.get("thought") or item.get("thought_signature"):
                    continue
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


class HeuristicTokenCounter:
    """
    ~4 ASCII characters per token, and one token per non-ASCII character.

    The second rule matters: Devanagari or CJK text runs close to a token per
    character, and a flat chars/4 would undercount it several-fold -- exactly
    the error that lets a prompt blow past the provider's limit.
    """

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return math.ceil((len(text) - non_ascii) / 4) + non_ascii


class TiktokenCounter:
    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


_tiktoken_counter: TiktokenCounter | None = None
_tiktoken_failed = False


def get_counter(kind: str = "heuristic") -> TokenCounter:
    global _tiktoken_counter, _tiktoken_failed
    if kind == "tiktoken" and not _tiktoken_failed:
        if _tiktoken_counter is None:
            try:
                import tiktoken

                _tiktoken_counter = TiktokenCounter(tiktoken.get_encoding("o200k_base"))
            except Exception as exc:
                _tiktoken_failed = True
                logger.warning(
                    "tiktoken unavailable (%s); using the heuristic token estimate.",
                    type(exc).__name__,
                )
                return HeuristicTokenCounter()
        return _tiktoken_counter
    return HeuristicTokenCounter()


def count_message(counter: TokenCounter, message: BaseMessage) -> int:
    tokens = MESSAGE_OVERHEAD_TOKENS + counter.count_text(content_text(message.content))
    if isinstance(message, AIMessage) and message.tool_calls:
        # Tool-call arguments are sent back to the model on every later round.
        for call in message.tool_calls:
            tokens += counter.count_text(call.get("name") or "")
            tokens += counter.count_text(
                json.dumps(call.get("args") or {}, ensure_ascii=False, default=str)
            )
    return tokens


def count_messages(counter: TokenCounter, messages: Iterable[BaseMessage]) -> int:
    return sum(count_message(counter, message) for message in messages)
