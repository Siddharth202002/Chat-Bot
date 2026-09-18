"""
Choosing which model answers a turn.

The picker reorders the provider chain rather than replacing it, so the two
things worth holding down are that the choice is honoured *first* and that the
rest of the chain still catches a rate limit behind it -- and that a
substitution is reported rather than passed off as the picked model.

The other invariant here is subtle and would be easy to break: the chain is a
module-level cache shared by every request, so reordering must never mutate it.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

import api_server
import chatbot_backend


ALICE = {"id": "user-alice", "email": "alice@example.com"}


class FakeModel:
    """A provider that streams one scripted reply, or fails."""

    def __init__(self, name: str, reply: str = "ok", error: Exception | None = None):
        self.name = name
        self.reply = reply
        self.error = error
        self.calls = 0
        # What _model_id_of reads through the tool-binding wrapper.
        self.model_name = f"vendor/{name}-model:free"

    def bind_tools(self, _tools: Any) -> "FakeModel":
        return self

    async def astream(self, _messages: list[Any]):
        self.calls += 1
        if self.error is not None:
            raise self.error
        yield AIMessageChunk(content=self.reply)


@pytest.fixture
def chain(monkeypatch):
    """A three-provider chain the tests can rewrite."""
    models = {
        "groq": FakeModel("groq", "from groq"),
        "gemini": FakeModel("gemini", "from gemini"),
        "openrouter": FakeModel("openrouter", "from openrouter"),
    }
    entries = [(name, model) for name, model in models.items()]

    monkeypatch.setattr(chatbot_backend, "_get_llm_chain", lambda: entries)
    monkeypatch.setattr(chatbot_backend, "_llm_chain", entries)
    return models


def names(entries: list[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in entries]


# --------------------------------------------------------------------------
# Reordering
# --------------------------------------------------------------------------


def test_no_choice_keeps_the_configured_order(chain):
    assert names(chatbot_backend._ordered_chain(None)) == [
        "groq",
        "gemini",
        "openrouter",
    ]


def test_the_picked_model_goes_first_and_the_rest_stay_behind_it(chain):
    """Reordered, not replaced: the others are still there as fallback."""
    assert names(chatbot_backend._ordered_chain("gemini")) == [
        "gemini",
        "groq",
        "openrouter",
    ]


def test_reordering_does_not_mutate_the_shared_chain(chain):
    """
    The chain is one cached list shared by every request. Sorting it in place
    would leak one user's pick into everyone else's turns.
    """
    before = names(chatbot_backend._get_llm_chain())
    reordered = chatbot_backend._ordered_chain("openrouter")

    assert names(reordered) == ["openrouter", "groq", "gemini"]
    assert names(chatbot_backend._get_llm_chain()) == before
    assert reordered is not chatbot_backend._get_llm_chain()


def test_a_model_that_vanished_falls_back_to_the_default_order(chain):
    """A key pulled at runtime must not take the turn down with it."""
    assert names(chatbot_backend._ordered_chain("no-such-model")) == [
        "groq",
        "gemini",
        "openrouter",
    ]


# --------------------------------------------------------------------------
# Listing what can be picked
# --------------------------------------------------------------------------


def test_available_models_describes_the_live_chain(chain):
    models = chatbot_backend.available_models()

    assert [m["id"] for m in models] == ["groq", "gemini", "openrouter"]
    # The label comes off the model that is really configured, stripped of
    # vendor prefix and ":free" packaging.
    assert [m["label"] for m in models] == [
        "groq-model",
        "gemini-model",
        "openrouter-model",
    ]
    assert [m["default"] for m in models] == [True, False, False]
    assert models[0]["family"] == "Groq"


def test_no_configured_provider_is_an_empty_list_not_a_crash(monkeypatch):
    def boom():
        raise RuntimeError("No chat provider is configured.")

    monkeypatch.setattr(chatbot_backend, "_get_llm_chain", boom)
    assert chatbot_backend.available_models() == []
    assert chatbot_backend.available_model_ids() == []


# --------------------------------------------------------------------------
# End to end through a turn
# --------------------------------------------------------------------------


async def drain(stream) -> tuple[str, Any]:
    text, committed = "", None
    async for event in stream:
        if event is chatbot_backend.STREAM_RESET:
            text = ""
        elif isinstance(event, chatbot_backend.StreamTurnCommitted):
            committed = event
        else:
            text += event
    return text, committed


class FakeApp:
    """Just enough graph for the streaming helper."""

    def __init__(self) -> None:
        self.written: Any = None

    async def aget_state(self, _config):
        return None

    async def aupdate_state(self, _config, values, as_node=None):
        self.written = values


async def run_turn(preferred: str | None) -> tuple[str, Any]:
    token = chatbot_backend._active_provider.set(preferred)
    try:
        return await drain(
            chatbot_backend._get_response_stream_for_config(
                FakeApp(), {"configurable": {"thread_id": "t"}}, "hello"
            )
        )
    finally:
        chatbot_backend._active_provider.reset(token)


async def test_the_picked_model_answers_the_turn(chain):
    text, committed = await run_turn("gemini")

    assert text == "from gemini"
    assert committed.answered_by == "gemini"
    assert chain["groq"].calls == 0, "the default should not have been tried"


async def test_a_rate_limited_pick_still_falls_through_the_chain(chain):
    """The whole reason for reordering rather than pinning."""
    chain["gemini"].error = RuntimeError("429 rate limit reached")

    text, committed = await run_turn("gemini")

    assert chain["gemini"].calls == 1, "the pick is tried first"
    assert text == "from groq"
    # And the substitution is reported, so the UI can say so rather than let
    # the user believe they tested Gemini.
    assert committed.answered_by == "groq"


async def test_no_pick_uses_the_head_of_the_chain(chain):
    text, committed = await run_turn(None)
    assert text == "from groq"
    assert committed.answered_by == "groq"


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, chain):
    api_server.app.dependency_overrides[api_server.current_user] = lambda: ALICE

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api_server, "process_memory_turn", noop)
    monkeypatch.setattr(api_server, "initialize_backend", noop)
    monkeypatch.setattr(api_server, "close_backend", noop)
    try:
        with TestClient(api_server.app) as test_client:
            yield test_client
    finally:
        api_server.app.dependency_overrides.clear()


def test_the_models_endpoint_lists_the_chain(client):
    body = client.get("/api/models").json()
    assert [m["id"] for m in body["models"]] == ["groq", "gemini", "openrouter"]
    assert body["default"] == "groq"


def test_listing_models_requires_a_session(chain):
    api_server.app.dependency_overrides.clear()
    with TestClient(api_server.app) as anonymous:
        assert anonymous.get("/api/models").status_code == 401


def test_an_unknown_model_is_rejected_rather_than_quietly_ignored(client, monkeypatch):
    """Answering from the default would look like the pick had been honoured."""
    called: list[Any] = []

    async def fake_stream(*args, **kwargs):
        called.append(kwargs)
        yield "should not happen"

    monkeypatch.setattr(api_server, "get_response_stream", fake_stream)
    response = client.post(
        "/api/chat/stream",
        json={"message": "hi", "thread_id": "t1", "model": "gpt-5-turbo"},
    )

    assert response.status_code == 400
    assert "gpt-5-turbo" in response.json()["detail"]
    assert "groq" in response.json()["detail"], "the valid ids belong in the error"
    assert called == []


def test_a_known_model_reaches_the_backend(client, monkeypatch):
    seen: dict[str, Any] = {}

    async def fake_stream(*args, **kwargs):
        seen.update(kwargs)
        yield "hi"

    monkeypatch.setattr(api_server, "get_response_stream", fake_stream)
    response = client.post(
        "/api/chat/stream",
        json={"message": "hi", "thread_id": "t1", "model": "gemini"},
    )

    assert response.status_code == 200
    assert seen["model"] == "gemini"


def test_omitting_the_model_is_still_valid(client, monkeypatch):
    """Existing clients send no model at all; that must keep working."""
    seen: dict[str, Any] = {}

    async def fake_stream(*args, **kwargs):
        seen.update(kwargs)
        yield "hi"

    monkeypatch.setattr(api_server, "get_response_stream", fake_stream)
    response = client.post("/api/chat/stream", json={"message": "hi", "thread_id": "t1"})

    assert response.status_code == 200
    assert seen["model"] is None


def test_retry_can_switch_model(client, monkeypatch):
    """The pairing that makes the picker worth having."""
    seen: dict[str, Any] = {}

    async def fake_prepare(**kwargs):
        seen.update(kwargs)
        return chatbot_backend.RegenerationPlan(
            thread_id="t1",
            user_id=ALICE["id"],
            message_id=kwargs["message_id"],
            user_input="q1",
            checkpoint_id="ckpt-0",
            mode="retry",
            model=kwargs.get("model"),
        )

    async def fake_regenerate(plan):
        seen["plan"] = plan
        yield "answer"

    monkeypatch.setattr(api_server, "prepare_regeneration", fake_prepare)
    monkeypatch.setattr(api_server, "regenerate_stream", fake_regenerate)

    response = client.post(
        "/api/chat/t1/fork",
        json={"message_id": "m1", "mode": "retry", "model": "openrouter"},
    )

    assert response.status_code == 200
    assert seen["model"] == "openrouter"
    assert seen["plan"].model == "openrouter"
