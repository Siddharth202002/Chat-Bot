"""
Rolling conversation compression: ContextManager + ContextBuilder.

Fully offline: the summarizer and the chat providers are fakes, and the
integration tests run the real streaming path against a real AsyncSqliteSaver
on a temp database.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import chatbot_backend
import context_manager
import context_summary
import context_tokens

ALICE = "user-alice"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeSummarizer:
    """Records every fold; can fail or block on a gate."""

    model_name = "fake-flash-lite"

    def __init__(self, error: Exception | None = None, gate: asyncio.Event | None = None):
        self.error = error
        self.gate = gate
        self.requests: list[context_summary.SummaryRequest] = []

    async def summarize(self, request: context_summary.SummaryRequest) -> context_summary.SummaryResult:
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        ids = ",".join(str(m.id) for m in request.messages)
        return context_summary.SummaryResult(
            text=f"SUMMARY#{len(self.requests)} folded=[{ids}]",
            provider="fake",
            model=self.model_name,
            latency_ms=1,
        )


class FakeModel:
    """Streams scripted replies; a ("tool", name, args) reply streams a tool call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.replies: list[Any] = []
        self.prompts: list[list[Any]] = []

    def bind_tools(self, _tools: Any) -> "FakeModel":
        return self

    async def astream(self, messages: list[Any]):
        self.prompts.append(list(messages))
        reply = self.replies.pop(0) if self.replies else f"{self.name} answer " + "y" * 800
        if isinstance(reply, tuple):
            _, tool_name, args = reply
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": tool_name,
                        "args": json.dumps(args),
                        "id": f"call-{len(self.prompts)}",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return
        yield AIMessageChunk(content=reply)


class FakeTool:
    name = "lookup"

    async def ainvoke(self, args: dict[str, Any]) -> str:
        return f"lookup result for {args}"


def settings_env(monkeypatch, **values: int) -> None:
    defaults = {
        "CONTEXT_BUDGET": 1_000_000,
        "CONTEXT_COMPRESSION_TRIGGER": 2000,
        "CONTEXT_RECENT_MESSAGE_COUNT": 10,
        "CONTEXT_OUTPUT_RESERVE_TOKENS": 0,
        "CONTEXT_SAFETY_MARGIN_TOKENS": 0,
        "CONTEXT_MAX_COMPRESSION_ATTEMPTS": 1,
    }
    defaults.update(values)
    for key, value in defaults.items():
        monkeypatch.setenv(key, str(value))


def conversation(turns: int, start: int = 0, size: int = 400) -> list[Any]:
    messages: list[Any] = []
    for i in range(start, start + turns):
        messages.append(HumanMessage(content="q" * size, id=f"h{i}"))
        messages.append(AIMessage(content="a" * size, id=f"a{i}"))
    return messages


# --------------------------------------------------------------------------
# 1. Rolling algorithm: no-op below trigger, first fold, second fold
# --------------------------------------------------------------------------


async def test_rolling_compression_folds_only_newly_old_messages(monkeypatch):
    # Each message ~104 tokens; the trigger sits just under 20 messages.
    settings_env(monkeypatch, CONTEXT_COMPRESSION_TRIGGER=2000)
    summarizer = FakeSummarizer()
    manager = context_manager.ContextManager(summarizer)
    current = [HumanMessage(content="current question", id="now")]

    # Below the trigger: nothing is folded and nothing changes.
    small = conversation(3)
    plan = await manager.prepare(
        thread_id="t", prior_messages=small, summary=None, turn=current, system_tokens=0
    )
    assert plan.outcome == "below_trigger" and not plan.summary_changed
    assert plan.history == small and summarizer.requests == []

    # First fold: M0..M9 are summarized, the 10 most recent are kept verbatim.
    first_prior = conversation(10)
    plan1 = await manager.prepare(
        thread_id="t", prior_messages=first_prior, summary=None, turn=current, system_tokens=0
    )
    assert plan1.outcome == "compressed" and plan1.summary_changed
    assert [m.id for m in summarizer.requests[0].messages] == [m.id for m in first_prior[:10]]
    assert summarizer.requests[0].previous_summary == ""
    assert plan1.history == first_prior[10:]
    assert isinstance(plan1.history[0], HumanMessage)
    assert plan1.summary["through_id"] == "a4"
    # The current message is never offered to the summarizer.
    assert all(m.id != "now" for r in summarizer.requests for m in r.messages)

    # Second fold: previous summary + (M10..M19), never the summary-covered
    # messages again and never the new retained window.
    second_prior = [*first_prior, *conversation(5, start=10)]
    plan2 = await manager.prepare(
        thread_id="t",
        prior_messages=second_prior,
        summary=plan1.summary,
        turn=current,
        system_tokens=0,
    )
    request = summarizer.requests[1]
    assert request.previous_summary == plan1.summary["text"]
    assert [m.id for m in request.messages] == [m.id for m in second_prior[10:20]]
    assert plan2.history == second_prior[20:]
    assert plan2.summary["through_id"] == "a9"
    assert plan2.summary["folded_messages"] == 20

    # The builder puts exactly one summary section ahead of the recent window.
    built = chatbot_backend._messages_for_model([*plan2.history, *current], plan2.summary)
    summaries = [m for m in built if "CONVERSATION SUMMARY" in str(m.content)]
    assert len(summaries) == 1 and "SUMMARY#2" in summaries[0].content
    assert built[-1] is current[0]


# --------------------------------------------------------------------------
# 2. Summarizer failure never corrupts or loses anything
# --------------------------------------------------------------------------


async def test_summarizer_failure_keeps_conversation_intact(monkeypatch):
    settings_env(monkeypatch, CONTEXT_COMPRESSION_TRIGGER=1500)
    manager = context_manager.ContextManager(FakeSummarizer(error=RuntimeError("503")))
    prior = conversation(10)
    old_summary: context_manager.ContextSummary = {"text": "old summary", "through_id": "a0"}

    plan = await manager.prepare(
        thread_id="t",
        prior_messages=prior,
        summary=old_summary,
        turn=[HumanMessage(content="hi", id="now")],
        system_tokens=0,
    )
    assert plan.outcome == "failed"
    assert not plan.summary_changed
    assert plan.summary == old_summary
    assert plan.history == prior[2:], "everything after the cursor is still sent"

    # The provider-level service falls back, then cools down instead of
    # hammering a dead provider every turn.
    calls: list[str] = []

    class Client:
        def __init__(self, name: str, ok: bool):
            self.name, self.ok = name, ok

        async def ainvoke(self, _prompt):
            calls.append(self.name)
            if not self.ok:
                raise RuntimeError("quota")
            return AIMessage(content="## Decisions\n- use SQLite")

    service = context_summary.LLMSummaryService(
        [
            context_summary.SummaryProvider("gemini", "flash-lite", lambda: Client("gemini", False)),
            context_summary.SummaryProvider("groq", "fallback", lambda: Client("groq", True)),
        ],
        timeout=5,
        failure_cooldown=60,
        tool_result_max_chars=2000,
        max_input_tokens=10_000,
    )
    request = context_summary.SummaryRequest("", conversation(1), 2500, 3000)
    result = await service.summarize(request)
    assert result.used_fallback and result.model == "fallback"
    assert calls == ["gemini", "groq"]


# --------------------------------------------------------------------------
# Integration harness: real graph + real SQLite checkpointer
# --------------------------------------------------------------------------


async def _open_app(db_path: Path, monkeypatch):
    conn = await aiosqlite.connect(str(db_path))
    saver = AsyncSqliteSaver(conn=conn)
    await saver.setup()
    compiled = chatbot_backend.graph.compile(checkpointer=saver)
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_threads (thread_id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    await conn.commit()

    async def get_app():
        return compiled

    monkeypatch.setattr(chatbot_backend, "_async_conn", conn)
    monkeypatch.setattr(chatbot_backend, "_get_compiled_app", get_app)
    return compiled, conn


@pytest.fixture
async def backend(tmp_path: Path, monkeypatch):
    async def noop(*_a, **_k):
        return None

    async def no_memories(*_a, **_k):
        return []

    monkeypatch.setattr(chatbot_backend, "_ensure_app_tables", noop)
    monkeypatch.setattr(chatbot_backend, "retrieve_memory_context", no_memories)
    chatbot_backend._active_generations.clear()

    models = {"primary": FakeModel("primary"), "secondary": FakeModel("secondary")}
    monkeypatch.setattr(chatbot_backend, "_get_llm_chain", lambda: list(models.items()))
    monkeypatch.setattr(chatbot_backend, "tools_by_name", {"lookup": FakeTool()})

    summarizer = FakeSummarizer()
    monkeypatch.setattr(
        chatbot_backend, "_context_manager", context_manager.ContextManager(summarizer)
    )

    # Trigger just above the fixed overhead, so a few ~200-token messages fold.
    counter = context_tokens.get_counter()
    fixed = chatbot_backend._system_tokens(counter) + chatbot_backend._tool_schema_tokens(counter)
    settings_env(
        monkeypatch, CONTEXT_COMPRESSION_TRIGGER=fixed + 700, CONTEXT_RECENT_MESSAGE_COUNT=2
    )

    db_path = tmp_path / "compression.db"
    compiled, conn = await _open_app(db_path, monkeypatch)
    handle = {"models": models, "summarizer": summarizer, "db_path": db_path, "conn": conn}
    try:
        yield handle
    finally:
        chatbot_backend._active_generations.clear()
        await handle["conn"].close()


async def send(text: str, model: str | None = None, thread_id: str = "t1") -> str:
    out = ""
    async for event in chatbot_backend.get_response_stream(
        text, thread_id=thread_id, user_id=ALICE, model=model
    ):
        if event is chatbot_backend.STREAM_RESET:
            out = ""
        elif isinstance(event, str):
            out += event
    return out


async def head_state(thread_id: str = "t1") -> dict[str, Any]:
    app = await chatbot_backend._get_compiled_app()
    return (await app.aget_state({"configurable": {"thread_id": thread_id}})).values


# --------------------------------------------------------------------------
# 3. Streaming end to end: persistence, restart, model picker independence
# --------------------------------------------------------------------------


async def test_streaming_compresses_persists_and_survives_restart(backend, monkeypatch):
    summarizer: FakeSummarizer = backend["summarizer"]
    secondary: FakeModel = backend["models"]["secondary"]

    for i in range(4):
        # The user picks a different main model; the summarizer must not care.
        reply = await send(f"question {i} " + "q" * 800, model="secondary")
        assert reply.startswith("secondary answer")
        assert "SUMMARY#" not in reply, "summarizer output never reaches the stream"

    assert summarizer.requests, "compression triggered"
    assert chatbot_backend._get_context_manager()._summarizer.model_name == "fake-flash-lite"

    state = await head_state()
    summary = state["context_summary"]
    assert summary["text"].startswith("SUMMARY#")
    ids = [m.id for m in state["messages"]]
    assert summary["through_id"] in ids
    # Nothing is deleted: the reloaded transcript still has every turn.
    history = await chatbot_backend.get_chat_history("t1", ALICE)
    assert len(history) == 8

    # The model saw the summary, not the folded messages, and the current
    # question last.
    prompt = secondary.prompts[-1]
    assert any("CONVERSATION SUMMARY" in str(m.content) for m in prompt)
    shown = {m.id for m in prompt}
    assert not shown & set(ids[: ids.index(summary["through_id"]) + 1])
    assert "question 3" in prompt[-1].content

    # Restart: a brand new connection and compiled graph read the summary back.
    await backend["conn"].close()
    _, backend["conn"] = await _open_app(backend["db_path"], monkeypatch)
    reloaded = await head_state()
    assert reloaded["context_summary"] == summary
    await send("after restart", model="secondary")
    assert any("SUMMARY#" in str(m.content) for m in secondary.prompts[-1])


# --------------------------------------------------------------------------
# 4. Tool calls: the active turn is protected, cuts never split a tool pair
# --------------------------------------------------------------------------


async def test_tool_context_is_never_split_or_summarized(backend):
    primary: FakeModel = backend["models"]["primary"]
    summarizer: FakeSummarizer = backend["summarizer"]

    for i in range(5):
        # Every turn: a tool call, its result, then the answer.
        primary.replies = [("tool", "lookup", {"q": i}), f"answer {i} " + "y" * 800]
        await send(f"question {i} " + "q" * 800)

        # This turn's tool round was sent to the model with its result...
        second_round = primary.prompts[-1]
        assert isinstance(second_round[-1], ToolMessage)
        # ...and the conversation shown never begins with an orphaned tool
        # result or tool call: it starts on a user message.
        conv = [m for m in second_round if m.type != "system"]
        assert isinstance(conv[0], HumanMessage)

    assert summarizer.requests, "compression triggered"
    for request in summarizer.requests:
        # Each folded chunk ends on a completed answer, never mid tool loop.
        assert isinstance(request.messages[-1], AIMessage)
        assert not request.messages[-1].tool_calls
        pending_calls = {
            c["id"] for m in request.messages if isinstance(m, AIMessage) for c in m.tool_calls
        }
        results = {m.tool_call_id for m in request.messages if isinstance(m, ToolMessage)}
        assert pending_calls == results


# --------------------------------------------------------------------------
# 5. Concurrency: one fold per thread, other threads never wait
# --------------------------------------------------------------------------


async def test_concurrent_requests_fold_a_thread_once(monkeypatch):
    settings_env(monkeypatch, CONTEXT_COMPRESSION_TRIGGER=2000)
    gate = asyncio.Event()
    summarizer = FakeSummarizer(gate=gate)
    manager = context_manager.ContextManager(summarizer)
    prior = conversation(10)

    def prepare(thread_id: str):
        return manager.prepare(
            thread_id=thread_id,
            prior_messages=prior,
            summary=None,
            turn=[HumanMessage(content="hi", id=f"now-{thread_id}")],
            system_tokens=0,
        )

    same_a = asyncio.create_task(prepare("t1"))
    same_b = asyncio.create_task(prepare("t1"))
    other = asyncio.create_task(prepare("t2"))
    await asyncio.sleep(0.05)
    # t1's second request waits on t1's lock; t2 is already summarizing.
    assert len(summarizer.requests) == 2
    gate.set()
    plan_a, plan_b, plan_other = await asyncio.gather(same_a, same_b, other)

    assert len(summarizer.requests) == 2, "the waiting t1 request reused the fold"
    assert plan_a.summary == plan_b.summary
    assert plan_a.summary["through_id"] == plan_other.summary["through_id"] == "a4"
