"""
Trigger cadence: the message counter is the only thing gating a model call.

There is no phrase matching any more -- no "remember that", no "I prefer".
What is worth remembering is the extraction model's decision; these tests only
pin down *when* it gets asked.
"""

from __future__ import annotations

import memory_config
import memory_extraction


def test_default_is_every_user_message():
    assert memory_config.extraction_interval() == 1
    assert memory_extraction.decide_trigger(1) == "periodic"


def test_zero_messages_since_extraction_does_not_trigger():
    assert memory_extraction.decide_trigger(0) is None


def test_interval_is_configurable(monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "3")
    assert memory_extraction.decide_trigger(1) is None
    assert memory_extraction.decide_trigger(2) is None
    assert memory_extraction.decide_trigger(3) == "periodic"
    assert memory_extraction.decide_trigger(4) == "periodic"


def test_large_interval_suppresses_until_reached(monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "10")
    for count in range(1, 10):
        assert memory_extraction.decide_trigger(count) is None
    assert memory_extraction.decide_trigger(10) == "periodic"


def test_non_numeric_interval_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", "not-a-number")
    assert memory_config.extraction_interval() == 1
    assert memory_extraction.decide_trigger(1) == "periodic"


def test_zero_and_negative_intervals_are_floored_to_one(monkeypatch):
    """A zero interval would mean "never enough messages"; floor it at 1."""
    for raw in ("0", "-5"):
        monkeypatch.setenv("MEMORY_EXTRACTION_INTERVAL", raw)
        assert memory_config.extraction_interval() == 1
        assert memory_extraction.decide_trigger(1) == "periodic"


def test_no_phrase_matching_helpers_remain():
    """
    Guards against reintroducing the brittle gate. If local pre-filtering ever
    comes back it should be a deliberate decision, not a quiet reappearance of
    the regexes that dropped "I am using FastAPI" while keeping "I'm using".
    """
    for name in (
        "is_explicit_memory_request",
        "has_memory_signal",
        "_SIGNAL_PATTERNS",
        "_EXPLICIT_PATTERNS",
        "_QUESTION_PREFIX",
    ):
        assert not hasattr(memory_extraction, name), f"{name} came back"
