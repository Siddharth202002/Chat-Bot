"""
Per-provider output handling: reasoning leaks, thought parts, collapses, markup.

The fixtures in tests/fixtures/provider_captures.json are real captures, not
hand-written samples -- "before_fix" is what each provider in the chain actually
streamed for three prompt shapes, and "after_fix" is what the Groq models return
once reasoning_format="hidden" is set. Tests written against invented text would
have passed the whole time this bug was live.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk

import chatbot_backend

CAPTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "provider_captures.json").read_text(
        encoding="utf-8"
    )
)
BEFORE = CAPTURES["before_fix"]
AFTER = CAPTURES["after_fix"]

# Phrases that only appear when a model narrates its own planning. The last two
# are quotes of OUTPUT_FORMAT_POLICY -- the model reading our prompt back to the
# user, which is the most obvious tell that reasoning escaped.
REASONING_TELLS = (
    "thinking process",
    "analyze user input",
    "identify requirements",
    "determine answer format",
    "let me re-read",
    "github-flavoured markdown",
    "never expose the plumbing",
)


def leaked_reasoning(text: str) -> list[str]:
    lowered = text.lower()
    return [tell for tell in REASONING_TELLS if tell in lowered]


# --------------------------------------------------------------------------
# Reasoning must not reach the user
# --------------------------------------------------------------------------

@pytest.mark.parametrize("provider", sorted(AFTER))
def test_groq_reasoning_is_suppressed_at_the_source(provider):
    """
    reasoning_format="hidden" is the fix, not a regex over the answer.

    gpt-oss and qwen3 will happily fold a full chain-of-thought into `content`.
    Groq can drop it server-side, so there is nothing to strip and nothing to
    get wrong -- these are captures taken with the setting applied.
    """
    capture = AFTER[provider]
    assert capture["reasoning_format"] == "hidden"
    assert leaked_reasoning(capture["raw"]) == []
    # reasoning_content must not arrive by the side door either.
    assert capture["additional_kwargs_keys"] == []


def test_the_builders_ask_groq_to_hide_reasoning(monkeypatch):
    """The setting is on the model the chain actually builds."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    for builder in (
        chatbot_backend._build_groq,
        chatbot_backend._build_groq_alt,
        chatbot_backend._build_groq_alt2,
    ):
        assert builder().reasoning_format == "hidden"


def test_the_leaky_provider_is_out_of_the_default_chain():
    """
    OpenRouter's free model has no server-side switch for this, and its capture
    shows why that mattered: the reply opens by restating the task and quoting
    OUTPUT_FORMAT_POLICY. Stripping that reliably is guesswork, so the provider
    is not in the default chain.
    """
    assert leaked_reasoning(BEFORE["openrouter"]["steps"]["raw"])
    assert "openrouter" not in chatbot_backend.DEFAULT_PROVIDER_CHAIN
    # Still buildable, so it can be switched back on deliberately.
    assert "openrouter" in chatbot_backend._PROVIDER_BUILDERS


# --------------------------------------------------------------------------
# Gemini content parts
# --------------------------------------------------------------------------

def test_gemini_thought_parts_are_dropped():
    """
    A thought part carries "text" exactly like an answer part does, so matching
    on the presence of that key leaked reasoning. The type is what decides.
    """
    parts = [
        {"type": "thinking", "text": "The user wants the weather. I should..."},
        {"type": "reasoning", "text": "First call get_current_location."},
        {"type": "text", "text": "It is 15 C in London."},
    ]

    assert chatbot_backend._content_pieces(parts) == ["It is 15 C in London."]
    assert chatbot_backend._message_text(parts) == "It is 15 C in London."


def test_a_flagged_thought_part_is_dropped_even_when_typed_as_text():
    parts = [
        {"type": "text", "text": "Let me think about this.", "thought": True},
        {"type": "text", "text": "The answer is 42."},
    ]

    assert chatbot_backend._content_pieces(parts) == ["The answer is 42."]


def test_an_unrecognised_part_shape_is_ignored_not_guessed_at():
    """Anything new from the SDK is dropped rather than rendered blind."""
    parts = [
        {"functionCall": {"name": "get_weather"}},
        {"text": "no type field at all"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]

    assert chatbot_backend._content_pieces(parts) == []


def test_plain_string_content_still_works():
    assert chatbot_backend._content_pieces("hello") == ["hello"]
    assert chatbot_backend._content_pieces("") == []
    assert chatbot_backend._content_pieces(None) == []


# --------------------------------------------------------------------------
# Degenerate output
# --------------------------------------------------------------------------

def _gemini_collapse() -> str:
    """The captured 15k-character whitespace collapse."""
    raw = BEFORE["gemini"]["list5"]["raw"]
    assert len(raw) > 10_000, "fixture is not the collapse sample"
    return raw


def test_the_captured_collapse_is_still_detected():
    filt = chatbot_backend._DegenerateOutputFilter()
    raw = _gemini_collapse()
    retracted = False
    for index in range(0, len(raw), 12):
        _, retract = filt.feed(raw[index : index + 12])
        retracted = retracted or retract
    assert retracted


class _Collapses:
    """A provider that pads its way to the token ceiling and never recovers."""

    def __init__(self):
        self.calls = 0

    def astream(self, messages):
        async def gen():
            self.calls += 1
            yield AIMessageChunk(content="| No. | Laptop | Price |")
            for _ in range(40):
                yield AIMessageChunk(content=" " * 40)

        return gen()


class _Answers:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def astream(self, messages):
        async def gen():
            self.calls += 1
            yield AIMessageChunk(content=self.text)

        return gen()


def on_screen(emitted: list) -> str:
    """What the client is left showing, honouring STREAM_RESET as it does."""
    shown: list[str] = []
    for item in emitted:
        if item is chatbot_backend.STREAM_RESET:
            shown.clear()
        elif isinstance(item, str):
            shown.append(item)
    return "".join(shown)


def _stub_app():
    """Minimal stand-in for the compiled graph the stream helper reads state from."""

    class _App:
        async def aget_state(self, config):
            return None

        async def aupdate_state(self, config, values, as_node=None):
            return None

    return _App()


async def test_a_collapse_hands_the_turn_to_the_next_provider(monkeypatch):
    """
    The old behaviour held the collapse back and waited for a recovery that
    never came, so the turn ended with an empty reply.

    The collapse is only detectable a couple of hundred characters in, so the
    header row has already been sent. It is retracted -- we noticed, the stream
    is healthy, and all of it came from the provider being dropped -- and the
    next provider answers from an empty screen.
    """
    collapsing, healthy = _Collapses(), _Answers("Here are five laptops.")
    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("gemini", collapsing), ("groq", healthy)],
    )

    emitted = [
        item
        async for item in chatbot_backend._get_response_stream_for_config(
            _stub_app(), {"configurable": {"thread_id": "t"}}, "suggest 5 laptops"
        )
    ]

    assert chatbot_backend.STREAM_RESET in emitted
    text = on_screen(emitted)
    assert text == "Here are five laptops."
    assert healthy.calls == 1
    # No half table left behind, and above all the reply is not "".
    assert "| No. |" not in text
    assert text != ""


async def test_every_provider_collapsing_raises_rather_than_going_quiet(monkeypatch):
    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("gemini", _Collapses()), ("gemini-lite", _Collapses())],
    )

    with pytest.raises(chatbot_backend.AllProvidersUnavailable):
        async for _ in chatbot_backend._get_response_stream_for_config(
            _stub_app(), {"configurable": {"thread_id": "t"}}, "suggest 5 laptops"
        ):
            pass


# --------------------------------------------------------------------------
# finalize_text
# --------------------------------------------------------------------------

def unbalanced_markers(text: str) -> dict[str, int]:
    """
    Markers left over once fenced blocks and inline code are removed.

    The authoritative "does this render without literal asterisks" check runs
    against remark-gfm itself, in the frontend's streamingMarkdown.test.mjs.
    There is no Markdown parser in this project's Python dependencies and this
    is not worth adding one for, so the Python side asserts the structural
    property that check depends on: every marker is paired.
    """
    stripped = re.sub(r"(?s)^ {0,3}(`{3,}|~{3,}).*?(\1|\Z)", "", text, flags=re.M)
    stripped = re.sub(r"`[^`\n]*`", "", stripped)
    # Bullets and rules are not emphasis.
    stripped = re.sub(r"(?m)^\s*\*+(\s)", r"\1", stripped)
    return {
        "**": stripped.count("**") % 2,
        "~~": stripped.count("~~") % 2,
        "*": stripped.replace("**", "").count("*") % 2,
        "`": stripped.count("`") % 2,
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A reply cut off mid **bol", "A reply cut off mid **bol**"),
        ("Trailing marker **", "Trailing marker"),
        ("Closer after a space **word ", "Closer after a space **word**"),
        ("An open `span", "An open `span`"),
        ("Balanced **already** fine", "Balanced **already** fine"),
        ("Plain prose.", "Plain prose."),
        ("", ""),
    ],
)
def test_finalize_text_closes_what_the_model_left_open(text, expected):
    assert chatbot_backend.finalize_text(text) == expected


def test_finalize_text_does_not_close_markers_inside_a_code_fence():
    """`**kwargs` in a Python block is not an unclosed bold."""
    code = "Here:\n\n```python\ndef f(**kwargs):\n    return kwargs\n```\n"
    assert chatbot_backend.finalize_text(code) == code


def test_finalize_text_closes_an_unterminated_fence():
    """A reply cut off inside a code block still renders as a code block."""
    cut = "Run this:\n\n```bash\npython -m venv venv"
    assert chatbot_backend.finalize_text(cut).endswith("\n```")


def test_finalize_text_collapses_runaway_spacing():
    """Gemini's table padding, which is also what a truncated reply ends on."""
    padded = "| No. | Laptop |" + " " * 15_000
    result = chatbot_backend.finalize_text(padded)
    assert len(result) < 100
    assert "| No. | Laptop |" in result


def test_finalize_text_leaves_ordinary_spacing_alone():
    text = "Line one.\n\n- **A:** x\n- **B:** y\n\nDone."
    assert chatbot_backend.finalize_text(text) == text


@pytest.mark.parametrize(
    "provider,prompt",
    [
        (provider, prompt)
        for provider, runs in BEFORE.items()
        for prompt in runs
        # The collapse is rejected outright now, so "finalizes cleanly" is not
        # a property it needs to have.
        if not (provider == "gemini" and prompt == "list5")
    ],
)
def test_real_captures_finalize_with_every_marker_paired(provider, prompt):
    finalized = chatbot_backend.finalize_text(BEFORE[provider][prompt]["raw"], provider)
    assert unbalanced_markers(finalized) == {"**": 0, "~~": 0, "*": 0, "`": 0}


@pytest.mark.parametrize(
    "provider,prompt",
    [(p, k) for p, runs in BEFORE.items() for k in runs],
)
def test_finalize_never_truncates_a_healthy_reply(provider, prompt):
    """Repair only ever appends; it must not eat the answer."""
    raw = BEFORE[provider][prompt]["raw"]
    finalized = chatbot_backend.finalize_text(raw, provider)
    # Runaway padding is the one thing it is allowed to remove.
    assert len(finalized) >= len(re.sub(r"[ \t]{40,}", " ", raw)) - 4


def test_a_truncated_reply_is_repaired_before_it_is_stored():
    """
    The whole point of Task 5: what gets saved is what gets reloaded, so a reply
    stopped at the token ceiling must not keep its dangling markers forever.
    """
    truncated = (
        "Here are five laptops:\n\n"
        "| # | Laptop | Price |\n|---|--------|-------|\n"
        "| 1 | **Dell XPS 13** | **$1,399** |\n"
        "| 2 | **Lenovo ThinkPad X1 Carb"
    )
    finalized = chatbot_backend.finalize_text(truncated, "groq")
    assert unbalanced_markers(finalized) == {"**": 0, "~~": 0, "*": 0, "`": 0}
    assert finalized.startswith("Here are five laptops:")


def test_the_provider_hook_table_is_keyed_like_the_builders():
    """
    Empty by design -- every quirk is fixed at its source -- but if something is
    added later it has to be addressable by the name the chain uses.
    """
    assert set(chatbot_backend._PROVIDER_FINALIZERS) <= set(
        chatbot_backend._PROVIDER_BUILDERS
    )
