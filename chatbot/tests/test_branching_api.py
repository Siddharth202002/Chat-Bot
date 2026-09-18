"""
The HTTP surface of Edit and Retry.

test_branching covers what forking does to the checkpoint graph; this covers
the transport around it -- request validation, the status code a client gets
for each way a fork can be refused, and the SSE frames an edited turn arrives
in. The backend is stubbed here precisely so those two concerns stay apart.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api_server
import chatbot_backend


ALICE = {"id": "user-alice", "email": "alice@example.com"}


@pytest.fixture
def client(monkeypatch):
    """The real app with the auth dependency and the chat backend stubbed."""
    api_server.app.dependency_overrides[api_server.current_user] = lambda: ALICE

    async def no_memory_turn(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api_server, "process_memory_turn", no_memory_turn)
    # TestClient runs lifespan on __enter__; these two are what it would do.
    monkeypatch.setattr(api_server, "initialize_backend", no_memory_turn)
    monkeypatch.setattr(api_server, "close_backend", no_memory_turn)

    try:
        with TestClient(api_server.app) as test_client:
            yield test_client
    finally:
        api_server.app.dependency_overrides.clear()


def frames(response) -> list[dict[str, Any]]:
    """The decoded `data:` payloads of an SSE response, in order."""
    out: list[dict[str, Any]] = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                out.append(json.loads(payload))
    return out


def stub_plan(**overrides: Any) -> chatbot_backend.RegenerationPlan:
    return chatbot_backend.RegenerationPlan(
        thread_id=overrides.get("thread_id", "t1"),
        user_id=overrides.get("user_id", ALICE["id"]),
        message_id=overrides.get("message_id", "msg-1"),
        user_input=overrides.get("user_input", "q1"),
        checkpoint_id=overrides.get("checkpoint_id", "ckpt-0"),
        mode=overrides.get("mode", "edit"),
        model=overrides.get("model"),
    )


def install_backend(
    monkeypatch,
    *,
    prepare_error: Exception | None = None,
    events: list[Any] | None = None,
    stream_error: Exception | None = None,
):
    """Record what the route asks the backend for, and script the reply."""
    seen: dict[str, Any] = {}

    async def prepare(**kwargs):
        seen["prepare"] = kwargs
        if prepare_error is not None:
            raise prepare_error
        return stub_plan(
            message_id=kwargs["message_id"],
            user_input=kwargs.get("new_text") or "q1",
            mode="edit" if kwargs.get("new_text") else "retry",
            model=kwargs.get("model"),
        )

    scripted = ["Hello"] if events is None else events

    async def regenerate(plan):
        seen["plan"] = plan
        for event in scripted:
            yield event
        if stream_error is not None:
            raise stream_error

    monkeypatch.setattr(api_server, "prepare_regeneration", prepare)
    monkeypatch.setattr(api_server, "regenerate_stream", regenerate)
    return seen


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "edit", "message": "hi"},                    # no message_id
        {"message_id": "m1", "mode": "edit"},                 # edit with no text
        {"message_id": "m1", "mode": "edit", "message": " "},  # blank text
        {"message_id": "m1", "mode": "retry", "message": "x"}, # retry with text
        {"message_id": "m1", "mode": "delete"},               # unknown mode
        {"message_id": "", "mode": "retry"},                  # empty id
    ],
)
def test_malformed_requests_are_rejected(client, monkeypatch, body):
    called = install_backend(monkeypatch)
    response = client.post("/api/chat/t1/fork", json=body)
    assert response.status_code == 422
    assert "prepare" not in called, "validation must run before the backend"


def test_edit_passes_the_new_text_through(client, monkeypatch):
    seen = install_backend(monkeypatch)
    response = client.post(
        "/api/chat/t1/fork",
        json={"message_id": "m1", "mode": "edit", "message": "under 3000"},
    )
    assert response.status_code == 200
    assert seen["prepare"] == {
        "thread_id": "t1",
        "user_id": ALICE["id"],
        "message_id": "m1",
        "new_text": "under 3000",
        "model": None,
    }


def test_retry_sends_no_replacement_text(client, monkeypatch):
    seen = install_backend(monkeypatch)
    response = client.post(
        "/api/chat/t1/fork", json={"message_id": "m1", "mode": "retry"}
    )
    assert response.status_code == 200
    assert seen["prepare"]["new_text"] is None


# --------------------------------------------------------------------------
# How a refused fork is reported
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (chatbot_backend.ForbiddenError("You do not have access to this chat."), 403),
        (chatbot_backend.BranchPointNotFound("That message is gone."), 404),
        (chatbot_backend.ThreadBusy("Still generating."), 409),
        (ValueError("The edited message cannot be empty."), 400),
    ],
)
def test_a_refused_fork_is_a_status_code_not_a_stream(
    client, monkeypatch, error, status
):
    install_backend(monkeypatch, prepare_error=error)
    response = client.post(
        "/api/chat/t1/fork",
        json={"message_id": "m1", "mode": "edit", "message": "hi"},
    )
    assert response.status_code == status
    # The reason has to survive to the client: it is what the UI shows.
    assert response.json()["detail"] == str(error)


def test_forking_another_users_thread_is_forbidden(client, monkeypatch):
    """The ownership check is the backend's; the route must not swallow it."""
    install_backend(
        monkeypatch,
        prepare_error=chatbot_backend.ForbiddenError(
            "You do not have access to this chat."
        ),
    )
    response = client.post(
        "/api/chat/someone-elses-thread/fork",
        json={"message_id": "m1", "mode": "retry"},
    )
    assert response.status_code == 403


def test_forking_requires_a_session():
    api_server.app.dependency_overrides.clear()
    with TestClient(api_server.app) as anonymous:
        response = anonymous.post(
            "/api/chat/t1/fork", json={"message_id": "m1", "mode": "retry"}
        )
    assert response.status_code == 401


# --------------------------------------------------------------------------
# The stream itself
# --------------------------------------------------------------------------


def test_an_edited_turn_streams_in_the_normal_event_shapes(client, monkeypatch):
    install_backend(
        monkeypatch,
        events=[
            "Here are ",
            chatbot_backend.STREAM_RESET,
            "hotels under ",
            "3000.",
            chatbot_backend.StreamTurnCommitted("user-new", "ai-new"),
        ],
    )
    response = client.post(
        "/api/chat/t1/fork",
        json={"message_id": "m1", "mode": "edit", "message": "under 3000"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert frames(response) == [
        {"token": "Here are "},
        {"reset": True},
        {"token": "hotels under "},
        {"token": "3000."},
        {
            "message_ids": {"user": "user-new", "assistant": "ai-new"},
            "answered_by": None,
        },
        {"done": True},
    ]


def test_the_normal_stream_reports_message_ids_too(client, monkeypatch):
    """Without this a just-sent turn could never be edited without a reload."""

    async def stream(*_args, **_kwargs):
        yield "hi"
        yield chatbot_backend.StreamTurnCommitted("u1", "a1")

    monkeypatch.setattr(api_server, "get_response_stream", stream)
    response = client.post(
        "/api/chat/stream", json={"message": "hello", "thread_id": "t1"}
    )
    assert frames(response) == [
        {"token": "hi"},
        {"message_ids": {"user": "u1", "assistant": "a1"}, "answered_by": None},
        {"done": True},
    ]


def test_a_provider_outage_mid_fork_ends_the_stream_cleanly(client, monkeypatch):
    install_backend(
        monkeypatch,
        events=[],
        stream_error=chatbot_backend.AllProvidersUnavailable("all down"),
    )
    response = client.post(
        "/api/chat/t1/fork", json={"message_id": "m1", "mode": "retry"}
    )
    assert response.status_code == 200
    # No message_ids frame: the client can tell nothing was committed and put
    # the old branch back.
    assert frames(response) == [{"error": "all down"}]


def test_an_interrupted_fork_is_flagged_as_interrupted(client, monkeypatch):
    install_backend(
        monkeypatch,
        events=["half an answer"],
        stream_error=chatbot_backend.ResponseInterrupted("cut short"),
    )
    response = client.post(
        "/api/chat/t1/fork", json={"message_id": "m1", "mode": "retry"}
    )
    assert frames(response) == [
        {"token": "half an answer"},
        {"error": "cut short", "interrupted": True},
    ]
