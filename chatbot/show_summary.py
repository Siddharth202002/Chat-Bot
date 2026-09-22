"""Print the rolling summary stored for the most recent chats (read-only)."""

import asyncio
import sys

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import chatbot_backend


async def main(limit: int) -> None:
    conn = await aiosqlite.connect(str(chatbot_backend._db_path))
    try:
        app = chatbot_backend.graph.compile(checkpointer=AsyncSqliteSaver(conn=conn))
        async with conn.execute(
            "SELECT thread_id FROM chat_threads ORDER BY updated_at DESC LIMIT ?", (limit,)
        ) as cursor:
            thread_ids = [row[0] for row in await cursor.fetchall()]
        for thread_id in thread_ids:
            state = await app.aget_state({"configurable": {"thread_id": thread_id}})
            values = state.values or {}
            messages = values.get("messages", [])
            summary = values.get("context_summary")
            print(f"=== thread {thread_id}: {len(messages)} messages stored")
            if not summary:
                print("    no summary yet\n")
                continue
            ids = [m.id for m in messages]
            cursor_at = ids.index(summary["through_id"]) + 1 if summary["through_id"] in ids else "?"
            print(
                f"    model={summary.get('model')} folded={summary.get('folded_messages')} "
                f"tokens={summary.get('tokens')} covers first {cursor_at} messages"
            )
            print("    ---- summary ----")
            print("    " + summary["text"].replace("\n", "\n    ") + "\n")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
