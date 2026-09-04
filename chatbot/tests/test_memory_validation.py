"""Validation of Mistral's structured output, and the transcript we send it."""

from __future__ import annotations

import json

import pytest

import memory_extraction
from conftest import history


def test_valid_payload_is_accepted():
    payload = json.dumps(
        {
            "memories": [
                {"content": "User has experience with FastAPI.", "memory_type": "skill"},
                {
                    "content": "User prefers SQLite for small projects.",
                    "memory_type": "preference",
                },
                {"content": "User is learning GraphQL.", "memory_type": "skill"},
            ]
        }
    )
    result = memory_extraction.validate_extraction(payload)
    assert len(result) == 3
    assert result[0] == {
        "content": "User has experience with FastAPI.",
        "memory_type": "skill",
    }


def test_empty_memory_list_is_valid():
    assert memory_extraction.validate_extraction('{"memories": []}') == []


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "   ",
        "not json at all",
        "{",
        '{"memories": "not a list"}',
        '{"wrong_key": []}',
        "null",
        "123",
        '{"memories": [null, 5, "text"]}',
    ],
)
def test_malformed_output_never_reaches_the_database(payload):
    assert memory_extraction.validate_extraction(payload) == []


def test_markdown_fenced_json_is_recovered():
    payload = '```json\n{"memories": [{"content": "User uses SQLite.", "memory_type": "preference"}]}\n```'
    result = memory_extraction.validate_extraction(payload)
    assert result == [{"content": "User uses SQLite.", "memory_type": "preference"}]


def test_unknown_memory_type_falls_back_to_other():
    payload = '{"memories": [{"content": "User uses SQLite.", "memory_type": "banana"}]}'
    assert memory_extraction.validate_extraction(payload)[0]["memory_type"] == "other"


def test_memory_type_is_normalised():
    payload = '{"memories": [{"content": "User is a backend engineer.", "memory_type": "Personal Fact"}]}'
    assert memory_extraction.validate_extraction(payload)[0]["memory_type"] == "personal_fact"


def test_missing_memory_type_defaults_to_other():
    payload = '{"memories": [{"content": "User deploys on Linux."}]}'
    assert memory_extraction.validate_extraction(payload)[0]["memory_type"] == "other"


def test_too_short_content_is_dropped():
    payload = '{"memories": [{"content": "hi", "memory_type": "other"}]}'
    assert memory_extraction.validate_extraction(payload) == []


def test_content_is_truncated_and_whitespace_collapsed(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_CONTENT_CHARS", "40")
    payload = json.dumps({"memories": [{"content": "User  likes\n\n" + "x" * 200}]})
    result = memory_extraction.validate_extraction(payload)
    assert len(result[0]["content"]) == 40
    assert "\n" not in result[0]["content"]


def test_extraction_count_is_capped(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_PER_EXTRACTION", "2")
    payload = json.dumps(
        {"memories": [{"content": f"User likes tool number {i}."} for i in range(10)]}
    )
    assert len(memory_extraction.validate_extraction(payload)) == 2


def test_duplicate_content_within_one_response_is_collapsed():
    payload = json.dumps(
        {
            "memories": [
                {"content": "User uses SQLite.", "memory_type": "preference"},
                {"content": "user uses sqlite.", "memory_type": "preference"},
            ]
        }
    )
    assert len(memory_extraction.validate_extraction(payload)) == 1


def test_already_parsed_objects_are_accepted():
    result = memory_extraction.validate_extraction(
        {"memories": [{"content": "User uses SQLite.", "memory_type": "preference"}]}
    )
    assert len(result) == 1


def test_transcript_labels_roles_and_skips_blanks():
    transcript = memory_extraction.build_transcript(
        history(("user", "I use FastAPI."), ("assistant", ""), ("assistant", "Noted."))
    )
    assert transcript == "User: I use FastAPI.\nAssistant: Noted."


def test_transcript_truncates_long_assistant_turns():
    transcript = memory_extraction.build_transcript(
        history(("assistant", "y" * 5000), ("user", "I prefer SQLite."))
    )
    assert len(transcript) < 1000
    assert "User: I prefer SQLite." in transcript
