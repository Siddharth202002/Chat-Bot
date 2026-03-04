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
from langgraph.checkpoint.memory import MemorySaver


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
checkpointer = MemorySaver()

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
