"""
FastAPI server wrapping the chatbot backend.
Exposes POST /api/chat, POST /api/chat/stream, and GET /api/health.
"""

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatbot_backend import (
    close_backend,
    delete_chat,
    get_all_chats,
    get_chat_history,
    get_mcp_status,
    get_rag_status,
    get_response,
    get_response_stream,
    ingest_pdf,
    initialize_backend,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await initialize_backend()
    try:
        yield
    finally:
        await close_backend()


app = FastAPI(title="Gemini Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    response: str


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "mcp": get_mcp_status()}


@app.get("/api/rag/status")
async def rag_status() -> dict[str, Any]:
    return get_rag_status()


@app.post("/api/rag/upload-pdf")
async def rag_upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    try:
        result = await ingest_pdf(contents, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    return {"status": "ok", **result}


@app.get("/api/chats")
async def get_chats() -> dict[str, Any]:
    """Fetch a list of all chat threads from the database."""
    try:
        chats = await get_all_chats()
        return {"chats": chats}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        reply = await get_response(req.message, thread_id=req.thread_id)
        return ChatResponse(response=reply)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the chatbot response token-by-token via SSE."""

    async def generate():
        try:
            async for token in get_response_stream(req.message, thread_id=req.thread_id):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Error: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/chat/{thread_id}")
async def get_chat(thread_id: str) -> dict[str, Any]:
    """Fetch chat history for a given thread_id."""
    try:
        history = await get_chat_history(thread_id)
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/chat/{thread_id}")
async def remove_chat(thread_id: str) -> dict[str, Any]:
    """Delete a chat thread and all its messages."""
    try:
        success = await delete_chat(thread_id)
        if success:
            return {"status": "ok"}
        return {"error": "Failed to delete chat"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
