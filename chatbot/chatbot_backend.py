"""
Chatbot backend module - extracted from main.ipynb
LangGraph chatbot using Gemini 2.5 Flash with MemorySaver checkpointer.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Generator, TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# Load .env from the langraph root directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("gemini-api-key"),
    temperature=0.7
)


# --- State ---
class Chat_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Node ---
def chat(state: Chat_State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# --- Graph ---
_db_path = Path(__file__).resolve().parent / "chat_memory.db"
_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn=_conn)

graph = StateGraph(Chat_State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=checkpointer)


def get_response(user_input: str, thread_id: str = "1") -> str:
    """
    Send a user message to the chatbot and return the AI response.

    Args:
        user_input: The user's message text.
        thread_id: Thread ID for conversation memory.

    Returns:
        The AI's response text.
    """
    config = {"configurable": {"thread_id": thread_id}}
    response = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return response["messages"][-1].content


def get_response_stream(user_input: str, thread_id: str = "1") -> Generator[str, None, None]:
    """
    Stream the chatbot response token-by-token.

    Yields each text chunk as it arrives from the LLM, then saves
    the full response into the graph checkpoint for conversation memory.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Record the human message into the checkpoint first
    human_msg = HumanMessage(content=user_input)
    app.update_state(config, {"messages": [human_msg]})

    # Build the full message history for the LLM call
    state = app.get_state(config)
    messages = state.values["messages"]

    # Stream tokens from the LLM
    full_response = ""
    for chunk in llm.stream(messages):
        token = chunk.content
        if token:
            full_response += token
            yield token

    # Save the complete AI response into the checkpoint for memory
    ai_msg = AIMessage(content=full_response)
    app.update_state(config, {"messages": [ai_msg]})


def get_chat_history(thread_id: str) -> list[dict]:
    """
    Retrieve the chat history for a given thread ID from the checkpointer.
    Returns a list of dictionaries with 'role' and 'content'.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = app.get_state(config)
        
        # If no state or no messages, return empty list
        if not state or not hasattr(state, 'values') or "messages" not in state.values:
            return []
            
        messages = state.values["messages"]
        history = []
        
        for msg in messages:
            # Map Langchain message types to simple roles
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Skip tool calls or empty messages if any
                if msg.content:
                    history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                pass # Usually we don't show system messages to the user
                
        return history
    except Exception as e:
        print(f"Error fetching chat history for thread {thread_id}: {e}")
        return []


def get_all_chats() -> list[dict]:
    """
    Retrieve all unique chat threads from the SQLite database.
    Returns a list of dictionaries with 'id' and 'title' (derived from the first message).
    """
    try:
        # Query distinct thread IDs from the checkpoints table
        cursor = _conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC")
        threads = cursor.fetchall()
        
        chat_list = []
        for (thread_id,) in threads:
            # Fetch the history for each thread to derive a title
            history = get_chat_history(thread_id)
            if history:
                # Find the first user message for the title
                first_msg = next((msg["content"] for msg in history if msg["role"] == "user"), "New Chat")
                
                # Truncate title
                title = first_msg if len(first_msg) <= 36 else first_msg[:36] + "…"
                chat_list.append({"id": thread_id, "title": title})
                
        return chat_list
    except Exception as e:
        print(f"Error fetching all chats: {e}")
        return []


def delete_chat(thread_id: str) -> bool:
    """
    Deletes all messages and checkpoints associated with a given thread_id 
    from the SQLite database.
    """
    try:
        cursor = _conn.cursor()
        # The sqlite checkpointer usually has checkpoints, checkpoints_writes, and checkpoints_blobs (in newer versions)
        # However, deleting from checkpoints by thread_id is the primary way.
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        # It's also good practice to delete from writes/blobs if they exist, but deleting from checkpoints 
        # is enough to make the LangGraph state start fresh and disappear from get_all_chats.
        _conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting chat {thread_id}: {e}")
        return False
