"""
FastAPI server wrapping the chatbot backend.
Exposes POST /api/chat, POST /api/chat/stream, and GET /api/health.
"""

from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
import uvicorn

from chatbot_backend import get_response, get_response_stream, get_chat_history

app = FastAPI(title="Gemini Chatbot API")

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
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        reply = get_response(req.message, thread_id=req.thread_id)
        return ChatResponse(response=reply)
    except Exception as e:
        return ChatResponse(response=f"⚠️ Error: {str(e)}")


import json

@app.post("/api/chat/stream", response_class=EventSourceResponse)
def chat_stream(req: ChatRequest) -> Iterator[ServerSentEvent]:
    """Stream the chatbot response token-by-token via SSE."""
    try:
        for token in get_response_stream(req.message, thread_id=req.thread_id):
            yield ServerSentEvent(data=json.dumps({"token": token}), event="token")
        yield ServerSentEvent(data=json.dumps({"token": ""}), event="done")
    except Exception as e:
        yield ServerSentEvent(data=json.dumps({"error": f"⚠️ Error: {str(e)}"}), event="error")


@app.get("/api/chat/{thread_id}")
def get_chat(thread_id: str):
    """Fetch chat history for a given thread_id."""
    try:
        history = get_chat_history(thread_id)
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
