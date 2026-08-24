"""
FastAPI server wrapping the chatbot backend.
Exposes POST /api/chat, POST /api/chat/stream, and GET /api/health.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatbot_backend import (
    AuthError,
    ForbiddenError,
    JWT_COOKIE_NAME,
    JWT_EXP_SECONDS,
    UserExistsError,
    UserRecord,
    authenticate_user,
    assert_thread_not_foreign,
    close_backend,
    create_access_token,
    create_user,
    delete_chat,
    get_all_chats,
    get_chat_history,
    get_mcp_status,
    get_rag_status,
    get_user_from_token,
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


app = FastAPI(title="Zeno AI Chatbot API", lifespan=lifespan)

# Browsers block cross-origin reads, so the deployed frontend's origin must be
# listed here. Comma-separated, e.g. CORS_ALLOW_ORIGINS="https://chat.example.com".
# Defaults to local development only.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    response: str


class AuthRequest(BaseModel):
    email: str
    password: str


def _set_session_cookie(response: Response, token: str) -> None:
    secure_cookie = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key=JWT_COOKIE_NAME,
        value=token,
        max_age=JWT_EXP_SECONDS,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )


async def current_user(
    zeno_session: str | None = Cookie(default=None, alias=JWT_COOKIE_NAME),
) -> UserRecord:
    try:
        return await get_user_from_token(zeno_session)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "mcp": get_mcp_status()}


@app.post("/api/auth/register")
async def register(req: AuthRequest, response: Response) -> dict[str, Any]:
    try:
        user = await create_user(req.email, req.password)
    except UserExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _set_session_cookie(response, create_access_token(user))
    return {"user": user}


@app.post("/api/auth/login")
async def login(req: AuthRequest, response: Response) -> dict[str, Any]:
    try:
        user = await authenticate_user(req.email, req.password)
    except (AuthError, ValueError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    _set_session_cookie(response, create_access_token(user))
    return {"user": user}


@app.post("/api/auth/logout")
async def logout(response: Response) -> dict[str, str]:
    response.delete_cookie(key=JWT_COOKIE_NAME, path="/")
    return {"status": "ok"}


@app.get("/api/auth/me")
async def me(user: UserRecord = Depends(current_user)) -> dict[str, Any]:
    return {"user": user}


@app.get("/api/rag/status")
async def rag_status(
    thread_id: str,
    user: UserRecord = Depends(current_user),
) -> dict[str, Any]:
    try:
        await assert_thread_not_foreign(user["id"], thread_id)
    except ForbiddenError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return get_rag_status(user["id"], thread_id)


@app.post("/api/rag/upload-pdf")
async def rag_upload_pdf(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    user: UserRecord = Depends(current_user),
) -> dict[str, Any]:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    try:
        result = await ingest_pdf(contents, filename, user["id"], thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ForbiddenError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    return {"status": "ok", **result}


@app.get("/api/chats")
async def get_chats(user: UserRecord = Depends(current_user)) -> dict[str, Any]:
    """Fetch a list of all chat threads from the database."""
    try:
        chats = await get_all_chats(user["id"])
        return {"chats": chats}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: UserRecord = Depends(current_user)) -> ChatResponse:
    try:
        reply = await get_response(req.message, thread_id=req.thread_id, user_id=user["id"])
        return ChatResponse(response=reply)
    except ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest,
    user: UserRecord = Depends(current_user),
) -> StreamingResponse:
    """Stream the chatbot response token-by-token via SSE."""

    async def generate():
        try:
            async for token in get_response_stream(
                req.message,
                thread_id=req.thread_id,
                user_id=user["id"],
            ):
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
async def get_chat(
    thread_id: str,
    user: UserRecord = Depends(current_user),
) -> dict[str, Any]:
    """Fetch chat history for a given thread_id."""
    try:
        history = await get_chat_history(thread_id, user["id"])
        return {"history": history}
    except ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/chat/{thread_id}")
async def remove_chat(
    thread_id: str,
    user: UserRecord = Depends(current_user),
) -> dict[str, Any]:
    """Delete a chat thread and all its messages."""
    try:
        success = await delete_chat(thread_id, user["id"])
        if success:
            return {"status": "ok"}
        return {"error": "Failed to delete chat"}
    except ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
