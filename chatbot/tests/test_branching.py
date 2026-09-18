"""
Edit and Retry: re-running a turn on a new checkpoint branch.

These run the real graph against a real AsyncSqliteSaver on a temp database.
Only the provider chain and the tool registry are faked, because the thing
under test is the checkpoint topology -- which checkpoint a re-run forks from,
which one the thread's head ends up pointing at, and what the model is shown
as context on the new branch. A mocked checkpointer would test none of that.

Nothing here touches the network or the real chat_memory.db.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from langchain_core.messages import AIMessageChunk

import chatbot_backend


ALICE = "user-alice"
BOB = "user-bob"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeModel:
    """
    A provider that streams a scripted reply.

    ``script`` is a list of replies, consumed one per turn. A reply is either
    plain text, or a ``("tool", name, args, text)`` tuple whose first
    invocation streams a tool call and whose second streams ``text`` -- enough
    to prove that editing a message runs the tool again with new arguments.
    """

    def __init__(self, owner: "FakeProvider", name: str) -> None:
        self._owner = owner
        self.name = name

    def bind_tools(self, _tools: Any) -> "FakeModel":
        return self

    async def astream(self, messages: list[Any]):
        self._owner.prompts.append(list(messages))
        if self._owner.error is not None:
            raise self._owner.error

        reply = self._owner.next_reply()
        if isinstance(reply, tuple):
            kind, tool_name, tool_args, text = reply
            assert kind == "tool"
            key = id(reply)
            if key not in self._owner.tool_calls_made:
                self._owner.tool_calls_made.add(key)
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            # Providers stream arguments as a JSON string;
                            # AIMessageChunk parses them into .tool_calls.
                            "name": tool_name,
                            "args": json.dumps(tool_args),
                            "id": f"call-{len(self._owner.prompts)}",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
                return
            reply = text

        for piece in _in_chunks(str(reply)):
            yield AIMessageChunk(content=piece)


def _in_chunks(text: str, size: int = 8) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


class FakeProvider:
    """One entry in the fallback chain."""

    def __init__(self, name: str, replies: list[Any] | None = None, error: Exception | None = None):
        self.name = name
        self.replies = list(replies or [])
        self.error = error
        self.prompts: list[list[Any]] = []
        self.tool_calls_made: set[int] = set()
        self.model = FakeModel(self, name)

    def next_reply(self) -> Any:
        if not self.replies:
            return f"{self.name} has nothing more to say."
        # The last scripted reply repeats, so a test only has to script the
        # turns it actually cares about.
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    @property
    def last_prompt_texts(self) -> list[str]:
        """Human/AI text of the most recent prompt, system messages dropped."""
        if not self.prompts:
            return []
        return [
            chatbot_backend._message_text(m.content)
            for m in self.prompts[-1]
            if m.type in ("human", "ai") and m.content
        ]


class FakeTool:
    """Records the arguments it was called with."""

    def __init__(self, name: str, result: str = "tool result") -> None:
        self.name = name
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(dict(args))
        return f"{self.result} for {args}"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
async def backend(tmp_path: Path, monkeypatch):
    """
    The real compiled graph and checkpointer, on a temp database.

    Yields a small handle with the provider chain so a test can rewrite the
    script between turns.
    """
    db_path = tmp_path / "branching.db"
    conn = await aiosqlite.connect(str(db_path))

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    checkpointer = AsyncSqliteSaver(conn=conn)
    await checkpointer.setup()
    compiled = chatbot_backend.graph.compile(checkpointer=checkpointer)

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    await conn.commit()

    async def get_app():
        return compiled

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chatbot_backend, "_async_conn", conn)
    monkeypatch.setattr(chatbot_backend, "_get_compiled_app", get_app)
    monkeypatch.setattr(chatbot_backend, "_ensure_app_tables", noop)
    # Memory retrieval would open the real store; the branch behaviour does
    # not depend on it and an empty list is what a cold user gets anyway.
    monkeypatch.setattr(chatbot_backend, "retrieve_memory_context", _empty_memories)
    chatbot_backend._active_generations.clear()

    providers = [FakeProvider("primary")]

    def chain():
        return [(p.name, p.model) for p in providers]

    monkeypatch.setattr(chatbot_backend, "_get_llm_chain", chain)

    handle = _Backend(compiled=compiled, conn=conn, providers=providers)
    try:
        yield handle
    finally:
        chatbot_backend._active_generations.clear()
        await conn.close()


async def _empty_memories(_user_id: str, _query: str) -> list[dict[str, Any]]:
    return []


class _Backend:
    def __init__(self, compiled: Any, conn: Any, providers: list[FakeProvider]):
        self.compiled = compiled
        self.conn = conn
        self.providers = providers

    @property
    def primary(self) -> FakeProvider:
        return self.providers[0]

    def script(self, *replies: Any) -> None:
        self.primary.replies = list(replies)

    async def send(self, text: str, thread_id: str = "t1", user_id: str = ALICE) -> "Turn":
        stream = chatbot_backend.get_response_stream(
            text, thread_id=thread_id, user_id=user_id
        )
        return await _drain(stream)

    async def branch(
        self,
        message_id: str,
        new_text: str | None = None,
        thread_id: str = "t1",
        user_id: str = ALICE,
    ) -> "Turn":
        plan = await chatbot_backend.prepare_regeneration(
            thread_id=thread_id,
            user_id=user_id,
            message_id=message_id,
            new_text=new_text,
        )
        return await _drain(chatbot_backend.regenerate_stream(plan))

    async def history(self, thread_id: str = "t1", user_id: str = ALICE):
        return await chatbot_backend.get_chat_history(thread_id, user_id)

    async def checkpoint_rows(self, thread_id: str = "t1") -> list[tuple[str, str | None]]:
        async with self.conn.execute(
            "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? ORDER BY checkpoint_id",
            (thread_id,),
        ) as cursor:
            return [(row[0], row[1]) for row in await cursor.fetchall()]

    async def head_checkpoint(self, thread_id: str = "t1") -> str:
        snapshot = await self.compiled.aget_state(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        return snapshot.config["configurable"]["checkpoint_id"]


class Turn:
    """What one drained stream produced."""

    def __init__(self, text: str, user_id: str | None, assistant_id: str | None):
        self.text = text
        self.user_message_id = user_id
        self.assistant_message_id = assistant_id


async def _drain(stream) -> Turn:
    text = ""
    user_id: str | None = None
    assistant_id: str | None = None
    async for event in stream:
        if event is chatbot_backend.STREAM_RESET:
            text = ""
        elif isinstance(event, chatbot_backend.StreamTurnCommitted):
            user_id = event.user_message_id
            assistant_id = event.assistant_message_id
        else:
            text += event
    return Turn(text, user_id, assistant_id)


def roles(history: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in history]


# --------------------------------------------------------------------------
# Test 1 -- normal chat still works
# --------------------------------------------------------------------------


async def test_normal_chat_is_unchanged(backend: _Backend):
    backend.script("A1", "A2", "A3")
    await backend.send("q1")
    await backend.send("q2")
    await backend.send("q3")

    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1"),
        ("user", "q2"),
        ("assistant", "A2"),
        ("user", "q3"),
        ("assistant", "A3"),
    ]
    # One checkpoint per streamed turn, in a straight line.
    rows = await backend.checkpoint_rows()
    assert len(rows) == 3
    assert rows[0][1] is None
    assert rows[1][1] == rows[0][0]
    assert rows[2][1] == rows[1][0]


async def test_stream_reports_stable_message_ids(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1")

    history = await backend.history()
    assert [m["id"] for m in history] == [
        turn.user_message_id,
        turn.assistant_message_id,
    ]
    assert all(m["id"] for m in history)


# --------------------------------------------------------------------------
# Test 2 -- edit the latest user message
# --------------------------------------------------------------------------


async def test_edit_latest_message_replaces_the_turn(backend: _Backend):
    backend.script("A1", "A2")
    await backend.send("q1")
    turn2 = await backend.send("q2")

    backend.script("A2-new")
    await backend.branch(turn2.user_message_id, "q2-edited")

    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1"),
        ("user", "q2-edited"),
        ("assistant", "A2-new"),
    ]
    # The edited turn saw the first turn and nothing of the old second one.
    assert backend.primary.last_prompt_texts == ["q1", "A1", "q2-edited"]


async def test_edit_the_only_message_in_a_thread(backend: _Backend):
    """No earlier checkpoint exists, so the branch starts from a clear."""
    backend.script("A1")
    turn = await backend.send("q1")

    backend.script("A1-new")
    await backend.branch(turn.user_message_id, "q1-edited")

    assert roles(await backend.history()) == [
        ("user", "q1-edited"),
        ("assistant", "A1-new"),
    ]
    assert backend.primary.last_prompt_texts == ["q1-edited"]


# --------------------------------------------------------------------------
# Test 3 -- edit an older message discards everything downstream
# --------------------------------------------------------------------------


async def test_edit_older_message_drops_downstream_turns(backend: _Backend):
    backend.script("A1", "A2", "A3")
    await backend.send("q1")
    turn2 = await backend.send("q2")
    await backend.send("q3")

    before = await backend.history()
    assert len(before) == 6

    backend.script("A2-new")
    await backend.branch(turn2.user_message_id, "q2-edited")

    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1"),
        ("user", "q2-edited"),
        ("assistant", "A2-new"),
    ]
    # q3 and A3 are gone from context, not merely appended after.
    assert backend.primary.last_prompt_texts == ["q1", "A1", "q2-edited"]


async def test_old_branch_survives_in_the_checkpoint_table(backend: _Backend):
    backend.script("A1", "A2")
    await backend.send("q1")
    turn2 = await backend.send("q2")
    superseded_head = await backend.head_checkpoint()

    backend.script("A2-new")
    await backend.branch(turn2.user_message_id, "q2-edited")

    rows = await backend.checkpoint_rows()
    ids = {checkpoint_id for checkpoint_id, _ in rows}
    parents = {checkpoint_id: parent for checkpoint_id, parent in rows}

    # The abandoned checkpoint is still there and still readable...
    assert superseded_head in ids
    old_state = await backend.compiled.aget_state(
        {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": superseded_head,
            }
        }
    )
    old_texts = [
        chatbot_backend._message_text(m.content)
        for m in old_state.values["messages"]
    ]
    assert old_texts == ["q1", "A1", "q2", "A2"]

    # ...but the head is the new branch, forked from the same parent.
    head = await backend.head_checkpoint()
    assert head != superseded_head
    assert parents[head] == parents[superseded_head]


# --------------------------------------------------------------------------
# Test 4 -- retry
# --------------------------------------------------------------------------


async def test_retry_keeps_the_user_message_and_regenerates(backend: _Backend):
    backend.script("A1", "A2")
    await backend.send("q1")
    turn2 = await backend.send("q2")

    backend.script("A2-different")
    result = await backend.branch(turn2.assistant_message_id)

    assert result.text == "A2-different"
    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1"),
        ("user", "q2"),
        ("assistant", "A2-different"),
    ]
    assert backend.primary.last_prompt_texts == ["q1", "A1", "q2"]


async def test_retry_by_user_message_id_is_the_same_operation(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1")

    backend.script("A1-different")
    await backend.branch(turn.user_message_id)

    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1-different"),
    ]


# --------------------------------------------------------------------------
# Test 5 -- the new branch is what a later message continues from
# --------------------------------------------------------------------------


async def test_next_message_continues_from_the_edited_branch(backend: _Backend):
    backend.script("A1", "A2")
    await backend.send("q1")
    turn2 = await backend.send("q2")

    backend.script("A2-new")
    await backend.branch(turn2.user_message_id, "q2-edited")

    backend.script("A3")
    await backend.send("q3")

    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1"),
        ("user", "q2-edited"),
        ("assistant", "A2-new"),
        ("user", "q3"),
        ("assistant", "A3"),
    ]
    assert backend.primary.last_prompt_texts == [
        "q1",
        "A1",
        "q2-edited",
        "A2-new",
        "q3",
    ]


async def test_editing_twice_from_the_same_point(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1")

    backend.script("A1-b")
    second = await backend.branch(turn.user_message_id, "q1-b")

    backend.script("A1-c")
    await backend.branch(second.user_message_id, "q1-c")

    assert roles(await backend.history()) == [
        ("user", "q1-c"),
        ("assistant", "A1-c"),
    ]


# --------------------------------------------------------------------------
# Test 6 -- tools re-run with the edited arguments
# --------------------------------------------------------------------------


async def test_edit_reruns_the_tool_with_new_arguments(backend: _Backend, monkeypatch):
    search = FakeTool("flight_search")
    monkeypatch.setattr(
        chatbot_backend, "tools_by_name", {"flight_search": search}
    )

    backend.script(
        ("tool", "flight_search", {"origin": "Delhi"}, "Flights from Delhi.")
    )
    turn = await backend.send("flights from Delhi")
    assert search.calls == [{"origin": "Delhi"}]

    backend.script(
        ("tool", "flight_search", {"origin": "Mumbai"}, "Flights from Mumbai.")
    )
    await backend.branch(turn.user_message_id, "flights from Mumbai")

    assert search.calls == [{"origin": "Delhi"}, {"origin": "Mumbai"}]
    assert roles(await backend.history()) == [
        ("user", "flights from Mumbai"),
        ("assistant", "Flights from Mumbai."),
    ]


# --------------------------------------------------------------------------
# Test 7 -- the fallback chain is the same one normal chat uses
# --------------------------------------------------------------------------


async def test_edit_falls_back_to_the_next_provider(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1")

    broken = FakeProvider("broken", error=RuntimeError("429 rate limited"))
    standby = FakeProvider("standby", replies=["A1-from-standby"])
    backend.providers[:] = [broken, standby]

    result = await backend.branch(turn.user_message_id, "q1-edited")

    assert result.text == "A1-from-standby"
    assert broken.prompts, "the primary should have been tried first"
    assert roles(await backend.history()) == [
        ("user", "q1-edited"),
        ("assistant", "A1-from-standby"),
    ]


async def test_a_failed_edit_leaves_the_existing_branch_active(backend: _Backend):
    backend.script("A1", "A2")
    await backend.send("q1")
    turn2 = await backend.send("q2")
    head_before = await backend.head_checkpoint()

    backend.providers[:] = [FakeProvider("broken", error=RuntimeError("boom"))]
    with pytest.raises(chatbot_backend.AllProvidersUnavailable):
        await backend.branch(turn2.user_message_id, "q2-edited")

    assert await backend.head_checkpoint() == head_before
    assert roles(await backend.history()) == [
        ("user", "q1"),
        ("assistant", "A1"),
        ("user", "q2"),
        ("assistant", "A2"),
    ]


# --------------------------------------------------------------------------
# Test 9 -- authorization and stale input
# --------------------------------------------------------------------------


async def test_another_user_cannot_fork_the_thread(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1", user_id=ALICE)

    with pytest.raises(chatbot_backend.ForbiddenError):
        await chatbot_backend.prepare_regeneration(
            thread_id="t1",
            user_id=BOB,
            message_id=turn.user_message_id,
            new_text="q1-edited",
        )

    # Alice's thread is untouched.
    assert roles(await backend.history()) == [("user", "q1"), ("assistant", "A1")]


async def test_unknown_message_id_is_rejected(backend: _Backend):
    backend.script("A1")
    await backend.send("q1")

    with pytest.raises(chatbot_backend.BranchPointNotFound):
        await chatbot_backend.prepare_regeneration(
            thread_id="t1",
            user_id=ALICE,
            message_id="not-a-real-message-id",
            new_text="q1-edited",
        )


async def test_a_superseded_message_id_is_rejected(backend: _Backend):
    """An id from an abandoned branch must not be forkable."""
    backend.script("A1")
    turn = await backend.send("q1")

    backend.script("A1-new")
    await backend.branch(turn.user_message_id, "q1-edited")

    with pytest.raises(chatbot_backend.BranchPointNotFound):
        await chatbot_backend.prepare_regeneration(
            thread_id="t1",
            user_id=ALICE,
            message_id=turn.user_message_id,
            new_text="again",
        )


async def test_empty_edit_text_is_rejected(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1")

    with pytest.raises(ValueError):
        await chatbot_backend.prepare_regeneration(
            thread_id="t1",
            user_id=ALICE,
            message_id=turn.user_message_id,
            new_text="   ",
        )


async def test_a_second_fork_is_refused_while_one_is_running(backend: _Backend):
    backend.script("A1")
    turn = await backend.send("q1")

    plan = await chatbot_backend.prepare_regeneration(
        thread_id="t1",
        user_id=ALICE,
        message_id=turn.user_message_id,
        new_text="q1-edited",
    )
    stream = chatbot_backend.regenerate_stream(plan)
    # Start it, but do not finish it: the slot is held from here.
    await stream.__anext__()
    try:
        with pytest.raises(chatbot_backend.ThreadBusy):
            await chatbot_backend.prepare_regeneration(
                thread_id="t1",
                user_id=ALICE,
                message_id=turn.user_message_id,
                new_text="q1-edited-again",
            )
    finally:
        await stream.aclose()


# --------------------------------------------------------------------------
# Test 10 -- the active branch is a property of the database, not the process
# --------------------------------------------------------------------------


async def test_the_active_branch_survives_a_restart(backend: _Backend, tmp_path: Path):
    backend.script("A1", "A2")
    await backend.send("q1")
    turn2 = await backend.send("q2")

    backend.script("A2-new")
    await backend.branch(turn2.user_message_id, "q2-edited")

    # Drop the compiled graph and the connection, and reopen the file the way
    # a restarted process would.
    await backend.conn.close()

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(tmp_path / "branching.db"))
    try:
        reopened = chatbot_backend.graph.compile(
            checkpointer=AsyncSqliteSaver(conn=conn)
        )
        snapshot = await reopened.aget_state(
            {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        )
        texts = [
            chatbot_backend._message_text(m.content)
            for m in snapshot.values["messages"]
        ]
        assert texts == ["q1", "A1", "q2-edited", "A2-new"]
    finally:
        await conn.close()
        backend.conn = await aiosqlite.connect(str(tmp_path / "branching.db"))
