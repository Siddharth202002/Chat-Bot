"""
Chatbot backend module - extracted from main.ipynb
LangGraph chatbot using Gemini 2.5 Flash with MemorySaver checkpointer.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Generator, TypedDict, Annotated
import requests
from langchain_core.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from groq import Groq

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
    message_chunk_to_message,
)
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq


# Load .env from the langraph root directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# --- LLM Setup ---


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=os.getenv("gemini-api-key"),
#     temperature=0.7
# )

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# tool setup 
search = DuckDuckGoSearchRun()

@tool
def Mathematical_calculations(num1: float, num2: float, operation: str) -> str:
    """Use this tool to perform mathematical calculations between two numbers ."""
    operation = operation.lower().strip()
    if operation == "add":
        return str(num1 + num2)
    elif operation == "subtract":
        return str(num1 - num2)
    elif operation == "multiply":
        return str(num1 * num2)
    elif operation == "divide":
        if num2 == 0:
            return "Cannot divide by zero."
        return str(num1 / num2)
    else:
        return "Invalid operation"


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a symbol using Finnhub."""
    if not FINNHUB_API_KEY:
        return "FINNHUB_API_KEY is not configured."

    url = f"https://finnhub.io/api/v1/quote?symbol={symbol.upper()}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Failed to fetch stock price: {exc}"

    current_price = data.get("c")
    if current_price in (None, 0):
        return f"No stock price data found for {symbol.upper()}."

    return str(current_price)

tools=[search,Mathematical_calculations,get_stock_price]
llm_with_tools = llm.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}


# --- State ---
class Chat_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Node ---
def chat(state: Chat_State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: Chat_State):
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call["id"]

        tool_obj = tools_by_name.get(tool_name)
        if tool_obj is None:
            result = f"Tool '{tool_name}' is not available."
        else:
            try:
                result = tool_obj.invoke(tool_args)
            except Exception as exc:
                result = f"Tool '{tool_name}' failed: {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call_id)
        )

    return {"messages": tool_messages}


def route_tools(state: Chat_State):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# --- Graph ---
_db_path = Path(__file__).resolve().parent / "chat_memory.db"
_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn=_conn)

graph = StateGraph(Chat_State)
graph.add_node("chat", chat)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", route_tools)
graph.add_edge("tools", "chat")


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
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    response = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return response["messages"][-1].content


def get_response_stream(user_input: str, thread_id: str = "1") -> Generator[str, None, None]:
    """
    Stream the chatbot response incrementally.

    `app.stream(..., stream_mode="values")` yields the graph state after the
    node finishes, so it buffers the full answer. For live streaming we stream
    directly from the model, run any requested tools, and persist the completed
    turn back into the LangGraph checkpoint when generation ends.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    state = app.get_state(config)
    prior_messages: list[BaseMessage] = []
    if state and hasattr(state, "values"):
        prior_messages = list(state.values.get("messages", []))

    delta_messages: list[BaseMessage] = [HumanMessage(content=user_input)]
    working_messages = [*prior_messages, *delta_messages]

    while True:
        streamed_chunk = None

        for chunk in llm_with_tools.stream(working_messages):
            streamed_chunk = chunk if streamed_chunk is None else streamed_chunk + chunk

            if isinstance(chunk.content, str):
                if chunk.content:
                    yield chunk.content
            elif isinstance(chunk.content, list):
                for item in chunk.content:
                    if isinstance(item, str) and item:
                        yield item
                    elif isinstance(item, dict) and item.get("text"):
                        yield str(item["text"])

        if streamed_chunk is None:
            break

        ai_message = message_chunk_to_message(streamed_chunk)
        delta_messages.append(ai_message)
        working_messages.append(ai_message)

        if not isinstance(ai_message, AIMessage) or not ai_message.tool_calls:
            break

        tool_messages = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call["id"]

            tool_obj = tools_by_name.get(tool_name)
            if tool_obj is None:
                result = f"Tool '{tool_name}' is not available."
            else:
                try:
                    result = tool_obj.invoke(tool_args)
                except Exception as exc:
                    result = f"Tool '{tool_name}' failed: {exc}"

            tool_messages.append(
                ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call_id)
            )

        if not tool_messages:
            break

        delta_messages.extend(tool_messages)
        working_messages.extend(tool_messages)

    if len(delta_messages) > 1:
        app.update_state(config, {"messages": delta_messages})


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

if __name__ == "__main__":
    result = get_stock_price.invoke({"symbol": "AAPL"})
    print("Result:", result)  
