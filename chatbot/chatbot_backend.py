"""
Chatbot backend module - extracted from main.ipynb
LangGraph chatbot using Groq with SQLite-backed memory.
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError as exc:
    MultiServerMCPClient = None  # type: ignore[assignment]
    MCP_IMPORT_ERROR: str | None = str(exc)
else:
    MCP_IMPORT_ERROR = None

# Load .env from the langraph root directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# --- LLM Setup ---
_llm: Any | None = None
_llm_with_tools: Any | None = None
_llm_init_error: str | None = None


def _get_llm() -> Any:
    global _llm, _llm_init_error
    if _llm is not None:
        return _llm
    if _llm_init_error is not None:
        raise RuntimeError(_llm_init_error)

    try:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
        )
        return _llm
    except Exception as exc:
        _llm_init_error = f"Failed to initialize Groq chat model. Details: {exc}"
        raise RuntimeError(_llm_init_error) from exc


def _get_llm_with_tools() -> Any:
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(tools)
    return _llm_with_tools

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=os.getenv("gemini-api-key"),
#     temperature=0.7
# )

# embedding model (initialized lazily to avoid startup crashes if dependency is missing)
_embeddings: Any | None = None
_embedding_init_error: str | None = None

_rag_lock = asyncio.Lock()
_rag_vectorstore: Any | None = None
_rag_retriever: Any | None = None
_rag_status: dict[str, Any] = {
    "status": "empty",
    "message": "No PDF indexed. Upload a PDF to enable RAG answers.",
    "file_name": None,
    "chunks": 0,
    "pages": 0,
    "uploaded_at": None,
}
_upload_dir = Path(__file__).resolve().parent / "uploaded_pdfs"
_upload_dir.mkdir(parents=True, exist_ok=True)
_default_pdf_path = Path(__file__).resolve().parent / "data.pdf"


def _get_embedding_class() -> Any:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings


def _get_pdf_loader_class() -> Any:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        from langchain.document_loaders import PyPDFLoader
    return PyPDFLoader


def _get_text_splitter_class() -> Any:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def _get_faiss_class() -> Any:
    from langchain_community.vectorstores import FAISS

    return FAISS


def _get_embeddings() -> Any:
    global _embeddings, _embedding_init_error
    if _embeddings is not None:
        return _embeddings
    if _embedding_init_error is not None:
        raise RuntimeError(_embedding_init_error)
    try:
        HuggingFaceEmbeddings = _get_embedding_class()
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return _embeddings
    except Exception as exc:
        _embedding_init_error = (
            "Failed to initialize embedding model. Install sentence-transformers "
            "and langchain-huggingface. Details: "
            f"{exc}"
        )
        raise RuntimeError(_embedding_init_error) from exc


def _sanitize_filename(filename: str) -> str:
    raw_name = Path(filename or "uploaded.pdf").name
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).strip("._")
    if not safe_name:
        safe_name = "uploaded.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    return safe_name


def _build_pdf_retriever(pdf_path: Path) -> tuple[Any, Any, int, int]:
    PyPDFLoader = _get_pdf_loader_class()
    RecursiveCharacterTextSplitter = _get_text_splitter_class()
    FAISS = _get_faiss_class()

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    if not documents:
        raise ValueError("No readable text was found in the uploaded PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The PDF could not be split into text chunks for retrieval.")

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source"] = pdf_path.name

    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever, len(documents), len(chunks)


def _safe_unlink(path_str: str | None) -> None:
    if not path_str:
        return
    try:
        path = Path(path_str)
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass


def _cleanup_uploaded_pdfs(keep_path: Path | None = None) -> None:
    for pdf_file in _upload_dir.glob("*.pdf"):
        if keep_path is not None and pdf_file == keep_path:
            continue
        try:
            pdf_file.unlink()
        except OSError:
            pass


async def ingest_pdf(pdf_bytes: bytes, original_filename: str) -> dict[str, Any]:
    """
    Save uploaded PDF, build a FAISS index, and keep only one active uploaded PDF.
    """
    global _rag_vectorstore, _rag_retriever, _rag_status
    if not pdf_bytes:
        raise ValueError("Uploaded file is empty.")

    safe_name = _sanitize_filename(original_filename)
    stamped_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_name}"
    pdf_path = _upload_dir / stamped_name
    pdf_path.write_bytes(pdf_bytes)

    try:
        vectorstore, retriever, pages, chunks = await asyncio.to_thread(
            _build_pdf_retriever, pdf_path
        )
    except Exception:
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    async with _rag_lock:
        previous_path = str(_rag_status.get("stored_path") or "")
        _rag_vectorstore = vectorstore
        _rag_retriever = retriever
        _rag_status = {
            "status": "ready",
            "message": "PDF indexed successfully. Previous uploaded PDF was replaced.",
            "file_name": safe_name,
            "chunks": chunks,
            "pages": pages,
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "stored_path": str(pdf_path),
        }

    if previous_path and Path(previous_path) != pdf_path:
        _safe_unlink(previous_path)
    _cleanup_uploaded_pdfs(keep_path=pdf_path)
    return dict(_rag_status)


def get_rag_status() -> dict[str, Any]:
    """Return current RAG indexing status."""
    return dict(_rag_status)


async def initialize_default_rag_pdf() -> None:
    """
    If chatbot/data.pdf exists, index it once at startup.
    """
    if not _default_pdf_path.exists():
        return
    if _rag_status.get("status") == "ready":
        return
    try:
        await ingest_pdf(_default_pdf_path.read_bytes(), _default_pdf_path.name)
    except Exception as exc:
        print(f"Failed to auto-index default PDF '{_default_pdf_path.name}': {exc}")


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
MCP_SERVER_CONFIG: dict[str, dict[str, str | list[str]]] = {
    "spending-analyzer": {
        "transport": "stdio",
        "command": "C:\\Users\\SiddharthMehendiratt\\.local\\bin\\uv.exe",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "D:\\OneDrive - tripgain.com\\Desktop\\finance tracking mcp\\main.py",
        ],
    }
}


# --- Tools ---
_search_init_error: str | None = None
try:
    search: Any = DuckDuckGoSearchRun()
except Exception as exc:
    _search_init_error = str(exc)

    @tool
    def duckduckgo_search_unavailable(query: str) -> str:
        """Fallback when DuckDuckGo search dependency is not installed."""
        return (
            "DuckDuckGo search is currently unavailable. Install 'ddgs' to enable it. "
            f"Details: {_search_init_error}"
        )

    search = duckduckgo_search_unavailable

@tool
def rag_search(query: str) -> str:
    """Search relevant information from the currently indexed PDF document."""
    if not query.strip():
        return "Please provide a question to search in the uploaded PDF."

    retriever = _rag_retriever
    if retriever is None:
        return (
            "No PDF is indexed yet. Upload a PDF first using the frontend upload button "
            "or /api/rag/upload-pdf."
        )

    # Support both old and new retriever APIs.
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant context was found in the indexed PDF."

    context = "\n\n".join(
        f"{doc.page_content}\n(Source: {doc.metadata})"
        for doc in docs
    )
    return f"Relevant context from the indexed PDF:\n{context}"


@tool
def Mathematical_calculations(num1: float, num2: float, operation: str) -> str:
    """Use this tool to perform mathematical calculations between two numbers."""
    operation = operation.lower().strip()
    if operation == "add":
        return str(num1 + num2)
    if operation == "subtract":
        return str(num1 - num2)
    if operation == "multiply":
        return str(num1 * num2)
    if operation == "divide":
        if num2 == 0:
            return "Cannot divide by zero."
        return str(num1 / num2)
    return "Invalid operation"


@tool
async def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a symbol using Finnhub."""
    if not FINNHUB_API_KEY:
        return "FINNHUB_API_KEY is not configured."

    url = f"https://finnhub.io/api/v1/quote?symbol={symbol.upper()}&token={FINNHUB_API_KEY}"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Failed to fetch stock price: {exc}"

    current_price = data.get("c")
    if current_price in (None, 0):
        return f"No stock price data found for {symbol.upper()}."
    return str(current_price)


base_tools: list[Any] = [search, Mathematical_calculations, get_stock_price, rag_search]
tools: list[Any] = list(base_tools)
tools_by_name: dict[str, Any] = {tool_obj.name: tool_obj for tool_obj in tools}

_mcp_client: Any | None = None
_mcp_status_message: str | None = None
_mcp_initialized = False
_mcp_lock = asyncio.Lock()


def _refresh_tool_registry(extra_tools: list[Any] | None = None) -> None:
    """
    Refresh runtime tool bindings.
    MCP tools are appended to base tools when available.
    """
    global tools, _llm_with_tools, tools_by_name
    runtime_mcp_tools = extra_tools or []
    tools = [*base_tools, *runtime_mcp_tools]
    _llm_with_tools = None
    tools_by_name = {tool_obj.name: tool_obj for tool_obj in tools}


def _messages_for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Inject runtime context for spending-analyzer behavior:
    1) Spending amounts are in INR and should never be labeled as dollars.
    2) MCP status context so the assistant can explain when tools are unavailable.
    """
    system_messages: list[SystemMessage] = [
        SystemMessage(
            content=(
                "For spending-analyzer outputs, all expense amounts are in Indian Rupees (INR). "
                "Never label these amounts as dollars ($). Use INR."
            )
        )
    ]
    if _mcp_status_message is not None:
        system_messages.append(
            SystemMessage(
                content=(
                    "The spending-analyzer MCP tools are currently unavailable. "
                    f"Reason: {_mcp_status_message}. "
                    "If the user asks for spending analysis, explain this clearly."
                )
            )
        )
    return [*system_messages, *messages]


async def _initialize_mcp_client() -> None:
    """
    Initialize MCP client exactly once at backend startup.
    On failure, keep chatbot running with base tools only.
    """
    global _mcp_client, _mcp_initialized, _mcp_status_message
    if _mcp_initialized:
        return

    async with _mcp_lock:
        if _mcp_initialized:
            return
        _mcp_initialized = True

        if MultiServerMCPClient is None:
            _mcp_status_message = (
                "Missing dependency 'langchain_mcp_adapters'. "
                f"Install it to enable spending-analyzer MCP tools. Details: {MCP_IMPORT_ERROR}"
            )
            print(_mcp_status_message)
            return

        try:
            # MCP client is created once and reused across all turns.
            client = MultiServerMCPClient(MCP_SERVER_CONFIG)
            mcp_tools = await client.get_tools()
        except Exception as exc:
            _mcp_status_message = (
                "Could not connect to spending-analyzer MCP server during startup. "
                "Start the MCP server and restart the chatbot. "
                f"Details: {exc}"
            )
            print(_mcp_status_message)
            return

        _mcp_client = client
        _mcp_status_message = None
        # This is where MCP tools become available to the chatbot.
        _refresh_tool_registry(list(mcp_tools))


async def _close_mcp_client() -> None:
    """Close MCP resources during backend shutdown."""
    global _mcp_client, _mcp_initialized, _mcp_status_message
    client = _mcp_client
    _mcp_client = None
    _mcp_initialized = False
    _mcp_status_message = None
    _refresh_tool_registry()

    if client is None:
        return

    try:
        close_method = getattr(client, "aclose", None) or getattr(client, "close", None)
        if callable(close_method):
            close_result = close_method()
            if asyncio.iscoroutine(close_result):
                await close_result
    except Exception as exc:
        print(f"Error closing MCP client: {exc}")


def get_mcp_status() -> dict[str, str]:
    """Return MCP runtime status for API health checks."""
    if _mcp_status_message:
        return {"status": "unavailable", "message": _mcp_status_message}
    if _mcp_initialized:
        return {"status": "available", "message": "spending-analyzer MCP tools loaded."}
    return {"status": "initializing", "message": "MCP client not initialized yet."}


# --- State ---
class Chat_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Nodes ---
async def chat(state: Chat_State) -> dict[str, list[BaseMessage]]:
    messages = state["messages"]
    response = await _get_llm_with_tools().ainvoke(_messages_for_model(messages))
    return {"messages": [response]}


async def _invoke_tool(tool_obj: Any, tool_args: dict[str, Any]) -> Any:
    """Invoke tools in async-safe way for both sync and async tool implementations."""
    ainvoke = getattr(tool_obj, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(tool_args)
    invoke = getattr(tool_obj, "invoke", None)
    if callable(invoke):
        return await asyncio.to_thread(invoke, tool_args)
    raise RuntimeError("Tool does not support invoke/ainvoke.")


async def tool_node(state: Chat_State) -> dict[str, list[BaseMessage]]:
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    tool_messages: list[ToolMessage] = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call["id"]

        tool_obj = tools_by_name.get(tool_name)
        if tool_obj is None:
            result = f"Tool '{tool_name}' is not available."
        else:
            try:
                result = await _invoke_tool(tool_obj, tool_args)
            except Exception as exc:
                result = f"Tool '{tool_name}' failed: {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call_id)
        )

    return {"messages": tool_messages}


def route_tools(state: Chat_State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# --- Graph ---
_db_path = Path(__file__).resolve().parent / "chat_memory.db"

graph = StateGraph(Chat_State)
graph.add_node("chat", chat)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", route_tools)
graph.add_edge("tools", "chat")

_compiled_app: Any | None = None
_async_conn: aiosqlite.Connection | None = None
_checkpointer: AsyncSqliteSaver | None = None
_app_lock = asyncio.Lock()


async def _get_compiled_app() -> Any:
    """
    Lazily initialize and return a graph compiled with AsyncSqliteSaver.
    """
    global _compiled_app, _async_conn, _checkpointer
    if not _mcp_initialized:
        # Safety net for direct backend usage outside FastAPI lifespan.
        await _initialize_mcp_client()

    if _compiled_app is not None:
        return _compiled_app

    async with _app_lock:
        if _compiled_app is not None:
            return _compiled_app

        _async_conn = await aiosqlite.connect(str(_db_path))
        _checkpointer = AsyncSqliteSaver(conn=_async_conn)
        await _checkpointer.setup()
        _compiled_app = graph.compile(checkpointer=_checkpointer)
        return _compiled_app


async def initialize_backend() -> None:
    """Initialize async graph + MCP resources on API startup."""
    await initialize_default_rag_pdf()
    await _initialize_mcp_client()
    await _get_compiled_app()


async def close_backend() -> None:
    """Close async graph + MCP resources on API shutdown."""
    global _compiled_app, _async_conn, _checkpointer
    await _close_mcp_client()
    if _async_conn is not None:
        await _async_conn.close()
        _async_conn = None
    _checkpointer = None
    _compiled_app = None


async def get_response(user_input: str, thread_id: str = "1") -> str:
    """Send a user message to the chatbot and return the AI response."""
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    compiled_app = await _get_compiled_app()
    response = await compiled_app.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    return response["messages"][-1].content


async def get_response_stream(
    user_input: str, thread_id: str = "1"
) -> AsyncGenerator[str, None]:
    """
    Stream the chatbot response incrementally.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    compiled_app = await _get_compiled_app()
    state = await compiled_app.aget_state(config)
    prior_messages: list[BaseMessage] = []
    if state and hasattr(state, "values"):
        prior_messages = list(state.values.get("messages", []))

    delta_messages: list[BaseMessage] = [HumanMessage(content=user_input)]
    # Inject MCP status only for model runtime context.
    working_messages = _messages_for_model([*prior_messages, *delta_messages])
    llm_with_tools = _get_llm_with_tools()

    while True:
        streamed_chunk = None

        async for chunk in llm_with_tools.astream(working_messages):
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

        tool_messages: list[ToolMessage] = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call["id"]

            tool_obj = tools_by_name.get(tool_name)
            if tool_obj is None:
                result = f"Tool '{tool_name}' is not available."
            else:
                try:
                    result = await _invoke_tool(tool_obj, tool_args)
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
        # Both "chat" and "tools" write `messages`; specify writer node.
        await compiled_app.aupdate_state(
            config,
            {"messages": delta_messages},
            as_node="chat",
        )


async def get_chat_history(thread_id: str) -> list[dict[str, str]]:
    """
    Retrieve the chat history for a given thread ID from the checkpointer.
    Returns a list of dictionaries with 'role' and 'content'.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        compiled_app = await _get_compiled_app()
        state = await compiled_app.aget_state(config)

        if not state or not hasattr(state, "values") or "messages" not in state.values:
            return []

        messages = state.values["messages"]
        history: list[dict[str, str]] = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                if msg.content:
                    history.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, SystemMessage):
                continue

        return history
    except Exception as exc:
        print(f"Error fetching chat history for thread {thread_id}: {exc}")
        return []


async def _fetch_thread_ids_async() -> list[str]:
    await _get_compiled_app()
    if _async_conn is None:
        return []
    async with _async_conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC"
    ) as cursor:
        rows = await cursor.fetchall()
    return [thread_id for (thread_id,) in rows]


async def get_all_chats() -> list[dict[str, str]]:
    """
    Retrieve all unique chat threads from the SQLite database.
    Returns a list of dictionaries with 'id' and 'title'.
    """
    try:
        threads = await _fetch_thread_ids_async()
        chat_list: list[dict[str, str]] = []

        for thread_id in threads:
            history = await get_chat_history(thread_id)
            if history:
                first_msg = next(
                    (msg["content"] for msg in history if msg["role"] == "user"),
                    "New Chat",
                )
                title = first_msg if len(first_msg) <= 36 else first_msg[:36] + "..."
                chat_list.append({"id": thread_id, "title": title})

        return chat_list
    except Exception as exc:
        print(f"Error fetching all chats: {exc}")
        return []


async def _delete_chat_async(thread_id: str) -> bool:
    try:
        await _get_compiled_app()
        if _async_conn is None:
            return False

        await _async_conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,),
        )
        for table in ("checkpoints_writes", "checkpoints_blobs"):
            try:
                await _async_conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?",
                    (thread_id,),
                )
            except aiosqlite.OperationalError:
                pass

        await _async_conn.commit()
        return True
    except Exception as exc:
        print(f"Error deleting chat {thread_id}: {exc}")
        return False


async def delete_chat(thread_id: str) -> bool:
    """Delete all checkpoints associated with a thread_id."""
    return await _delete_chat_async(thread_id)


if __name__ == "__main__":
    async def _demo() -> None:
        result = await get_stock_price.ainvoke({"symbol": "AAPL"})
        print("Result:", result)

    asyncio.run(_demo())
