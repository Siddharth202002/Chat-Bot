"""
Configuration for rolling conversation compression.

Read from the environment on every call, like memory_config, so tests and a
restart-free reload can change a value without re-importing anything. The
reads happen once per chat turn.

Budget (defaults in brackets, all in estimated tokens):
    CONTEXT_COMPRESSION_ENABLED       [true]
    CONTEXT_BUDGET                    [16000] hard ceiling for one model call
    CONTEXT_COMPRESSION_TRIGGER       [12000] projected usage that triggers a fold
    CONTEXT_RECENT_MESSAGE_COUNT      [10]    messages always kept verbatim
    CONTEXT_SUMMARY_TARGET_TOKENS     [2500]  length the summarizer aims for
    CONTEXT_SUMMARY_MAX_TOKENS        [3000]  hard cap on a stored summary
    CONTEXT_OUTPUT_RESERVE_TOKENS     [4000]  room left for the reply
    CONTEXT_SAFETY_MARGIN_TOKENS      [2000]  slack for token-estimate error
    CONTEXT_MAX_COMPRESSION_ATTEMPTS  [1]     summarizer passes per turn

Summarizer (an internal component -- the user's model picker never affects it):
    CONTEXT_SUMMARY_PROVIDER          [gemini]
    CONTEXT_SUMMARY_MODEL             [gemini-flash-lite-latest]
    CONTEXT_SUMMARY_FALLBACK_PROVIDER []      "", "gemini" or "groq"
    CONTEXT_SUMMARY_FALLBACK_MODEL    []
    CONTEXT_SUMMARY_TIMEOUT           [20]    seconds per provider attempt
    CONTEXT_SUMMARY_FAILURE_COOLDOWN  [60]    seconds to skip folding after all
                                              summarizers failed
    CONTEXT_SUMMARY_MAX_INPUT_TOKENS  [60000] cap on what one fold sends
    CONTEXT_TOOL_RESULT_MAX_CHARS     [2000]  per tool result, summarizer input only
    CONTEXT_TOKENIZER                 [heuristic]  or "tiktoken"
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# The summarizer model name lives here and nowhere else.
DEFAULT_SUMMARY_PROVIDER = "gemini"
DEFAULT_SUMMARY_MODEL = "gemini-flash-lite-latest"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        return default


def _get_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, float(raw.strip()))
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


@dataclass(frozen=True)
class ContextSettings:
    enabled: bool
    budget: int
    trigger: int
    recent_message_count: int
    summary_target_tokens: int
    summary_max_tokens: int
    output_reserve_tokens: int
    safety_margin_tokens: int
    max_compression_attempts: int
    summary_provider: str
    summary_model: str
    fallback_provider: str
    fallback_model: str
    summary_timeout: float
    failure_cooldown: float
    summary_max_input_tokens: int
    tool_result_max_chars: int
    tokenizer: str

    @property
    def hard_limit(self) -> int:
        """Projected usage (reply reserve included) a prompt must stay under."""
        return self.budget - self.safety_margin_tokens


def load() -> ContextSettings:
    budget = _get_int("CONTEXT_BUDGET", 16000, minimum=1000)
    # A trigger above the budget could never fire before the hard limit does,
    # so it is clamped rather than trusted.
    trigger = min(budget, _get_int("CONTEXT_COMPRESSION_TRIGGER", 12000, minimum=500))
    summary_max = _get_int("CONTEXT_SUMMARY_MAX_TOKENS", 3000, minimum=200)
    summary_target = min(
        summary_max, _get_int("CONTEXT_SUMMARY_TARGET_TOKENS", 2500, minimum=100)
    )
    return ContextSettings(
        enabled=_get_bool("CONTEXT_COMPRESSION_ENABLED", True),
        budget=budget,
        trigger=trigger,
        # At least one exchange stays verbatim, or the model loses the thread
        # of the sentence it is answering.
        recent_message_count=_get_int("CONTEXT_RECENT_MESSAGE_COUNT", 10, minimum=2),
        summary_target_tokens=summary_target,
        summary_max_tokens=summary_max,
        output_reserve_tokens=_get_int("CONTEXT_OUTPUT_RESERVE_TOKENS", 4000),
        safety_margin_tokens=_get_int("CONTEXT_SAFETY_MARGIN_TOKENS", 2000),
        max_compression_attempts=_get_int(
            "CONTEXT_MAX_COMPRESSION_ATTEMPTS", 1, minimum=1
        ),
        summary_provider=_get_str("CONTEXT_SUMMARY_PROVIDER", DEFAULT_SUMMARY_PROVIDER).lower(),
        summary_model=_get_str("CONTEXT_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL),
        fallback_provider=_get_str("CONTEXT_SUMMARY_FALLBACK_PROVIDER", "").lower(),
        fallback_model=_get_str("CONTEXT_SUMMARY_FALLBACK_MODEL", ""),
        summary_timeout=_get_float("CONTEXT_SUMMARY_TIMEOUT", 20.0, minimum=1.0),
        failure_cooldown=_get_float("CONTEXT_SUMMARY_FAILURE_COOLDOWN", 60.0),
        summary_max_input_tokens=_get_int(
            "CONTEXT_SUMMARY_MAX_INPUT_TOKENS", 60000, minimum=1000
        ),
        tool_result_max_chars=_get_int("CONTEXT_TOOL_RESULT_MAX_CHARS", 2000, minimum=200),
        tokenizer=_get_str("CONTEXT_TOKENIZER", "heuristic").lower(),
    )
