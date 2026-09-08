"""
FastAPI server wrapping the chatbot backend.
Exposes POST /api/chat, POST /api/chat/stream, and GET /api/health.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import (
    BackgroundTasks,
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

import location_service
import location_state
import memory_config

from chatbot_backend import (
    AllProvidersUnavailable,
    AuthError,
    ForbiddenError,
    JWT_COOKIE_NAME,
    JWT_EXP_SECONDS,
    UserExistsError,
    UserRecord,
    authenticate_user,
    assert_thread_not_foreign,
    clear_user_location,
    close_backend,
    create_access_token,
    create_user,
    delete_chat,
    delete_user_memory,
    get_all_chats,
    get_chat_history,
    get_mcp_status,
    get_rag_status,
    get_user_from_token,
    get_user_location_state,
    get_response,
    get_response_stream,
    ingest_pdf,
    initialize_backend,
    list_user_memories,
    process_memory_turn,
    record_location_failure,
    resolve_user_city,
    resolve_user_coordinates,
    STREAM_RESET,
    update_user_memory,
)


# Memory extraction runs in a background task, so its logs are the only signal
# an operator gets that it is (or is not) working. Without a handler on the
# chatbot loggers, uvicorn shows request lines and nothing else -- a Mistral
# outage would look exactly like a healthy server.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("chatbot").setLevel(os.getenv("MEMORY_LOG_LEVEL", "INFO").upper())


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


class MemoryUpdateRequest(BaseModel):
    content: str | None = Field(default=None, max_length=2000)
    memory_type: str | None = Field(default=None, max_length=64)


class LocationRequest(BaseModel):
    """
    One of three shapes, in priority order:

      {"status": "denied"}                  a browser geolocation failure
      {"city": "Pune"}                      a place the user typed instead
      {"latitude": ..., "longitude": ...}    a browser GPS reading

    Range-checking latitude/longitude here means an out-of-range value is a 422
    from FastAPI before it can reach the geocoder and spend rate-limit budget;
    location_service.validate_coordinates re-checks it for every non-HTTP caller.
    """

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    city: str | None = Field(default=None, max_length=200)
    status: str | None = Field(default=None, max_length=32)

    @field_validator("status")
    @classmethod
    def _known_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in location_state.FAILURE_STATUSES:
            allowed = ", ".join(sorted(location_state.FAILURE_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        return normalized


# Which HTTP status a geocoding failure deserves. A 504/502 tells the frontend
# "upstream problem, worth retrying"; a 400 tells it "this input will never
# work". Anything unmapped falls through to 502.
_LOCATION_ERROR_STATUS: dict[str, int] = {
    location_service.INVALID_COORDINATES: 400,
    location_service.LOCATION_NOT_FOUND: 404,
    location_service.GEOCODING_TIMEOUT: 504,
    location_service.GEOCODING_RATE_LIMITED: 429,
    location_service.GEOCODING_UNAVAILABLE: 502,
    location_service.GEOCODING_MALFORMED_RESPONSE: 502,
}


def _location_response(
    status: str, record: location_state.StoredLocation | None
) -> dict[str, Any]:
    return {"status": status, "location": dict(record) if record else None}


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


@app.post("/api/location")
async def set_location(
    req: LocationRequest,
    user: UserRecord = Depends(current_user),
) -> dict[str, Any]:
    """
    Record where the signed-in user is, so the location and weather tools have
    something real to work with.

    The frontend calls this only when a message actually needs a location, and
    posts a `status` instead of coordinates when the browser refuses -- that
    recorded refusal is what lets the assistant explain itself without the UI
    prompting for permission again.
    """
    if req.latitude is None and req.longitude is None and not (req.city or "").strip()             and req.status is None:
        raise HTTPException(
            status_code=400,
            detail="Provide latitude and longitude, a city, or a status.",
        )

    try:
        # A reported failure wins over any coordinates in the same body: the
        # client is telling us it could not locate the user.
        if req.status is not None:
            recorded = await record_location_failure(user["id"], req.status)
            return {"status": recorded, "location": None}

        if req.city and req.city.strip():
            record = await resolve_user_city(user["id"], req.city.strip())
        elif req.latitude is not None and req.longitude is not None:
            record = await resolve_user_coordinates(
                user["id"], req.latitude, req.longitude
            )
        else:
            # One coordinate without the other.
            raise HTTPException(
                status_code=400,
                detail="Provide both latitude and longitude.",
            )
    except location_service.LocationError as exc:
        # exc.message is written for a person and contains no coordinates,
        # no upstream URL and no credentials, so it is safe to return as-is.
        raise HTTPException(
            status_code=_LOCATION_ERROR_STATUS.get(exc.code, 502),
            detail=exc.message,
        ) from exc
    except location_state.LocationStateUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Location storage is unavailable."
        ) from exc

    return _location_response(location_state.STATUS_READY, record)


@app.get("/api/location")
async def read_location(user: UserRecord = Depends(current_user)) -> dict[str, Any]:
    """The user's stored location state, so the UI can rehydrate without re-prompting."""
    try:
        status, record = await get_user_location_state(user["id"])
    except location_state.LocationStateUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Location storage is unavailable."
        ) from exc
    return _location_response(status, record)


@app.delete("/api/location")
async def remove_location(user: UserRecord = Depends(current_user)) -> dict[str, str]:
    """Forget the user's stored location ("stop sharing")."""
    try:
        await clear_user_location(user["id"])
    except location_state.LocationStateUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Location storage is unavailable."
        ) from exc
    return {"status": "ok"}


@app.get("/api/chats")
async def get_chats(user: UserRecord = Depends(current_user)) -> dict[str, Any]:
    """Fetch a list of all chat threads from the database."""
    try:
        chats = await get_all_chats(user["id"])
        return {"chats": chats}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    user: UserRecord = Depends(current_user),
) -> ChatResponse:
    try:
        reply = await get_response(req.message, thread_id=req.thread_id, user_id=user["id"])
    except ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except AllProvidersUnavailable as e:
        # The provider's own text (quota JSON, "payment required") belongs in
        # the operator's log, not in the user's transcript.
        logging.getLogger("chatbot.api").warning("All chat providers failed: %s", e)
        return ChatResponse(response=str(e))
    except Exception as e:
        logging.getLogger("chatbot.api").exception("Chat request failed.")
        return ChatResponse(response=f"Error: {str(e)}")

    # Long-term memory extraction runs after the response is sent, so a slow or
    # failing extraction call can never delay or break the reply.
    background_tasks.add_task(process_memory_turn, user["id"], req.thread_id)
    return ChatResponse(response=reply)


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
                # The backend retracts text it streamed before discovering the
                # round was a tool call. Forward that as its own event so the
                # client drops it instead of showing abandoned text above the
                # real answer.
                if token is STREAM_RESET:
                    yield f"data: {json.dumps({'reset': True})}\n\n"
                    continue
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except AllProvidersUnavailable as e:
            # Same reasoning as the non-streaming path: the provider's raw
            # quota/billing text stays in the log, the user gets plain English.
            logging.getLogger("chatbot.api").warning("All chat providers failed: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logging.getLogger("chatbot.api").exception("Chat stream failed.")
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
        # Starlette runs this once the stream is fully sent to the client.
        background=BackgroundTask(process_memory_turn, user["id"], req.thread_id),
    )


@app.get("/api/memories")
async def get_memories(user: UserRecord = Depends(current_user)) -> dict[str, Any]:
    """List the authenticated user's long-term memories."""
    memories = await list_user_memories(user["id"])
    return {"memories": memories}


@app.patch("/api/memories/{memory_id}")
async def patch_memory(
    memory_id: str,
    req: MemoryUpdateRequest,
    user: UserRecord = Depends(current_user),
) -> dict[str, Any]:
    """Edit one of the authenticated user's memories."""
    if req.content is None and req.memory_type is None:
        raise HTTPException(status_code=400, detail="Nothing to update.")
    # The store truncates at MEMORY_MAX_CONTENT_CHARS; rejecting here is
    # clearer than silently returning a clipped memory the client did not send.
    max_chars = memory_config.max_content_chars()
    if req.content is not None and len(req.content.strip()) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Memory content must be at most {max_chars} characters.",
        )
    try:
        memory = await update_user_memory(
            user["id"],
            memory_id,
            content=req.content,
            memory_type=req.memory_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"memory": memory}


@app.delete("/api/memories/{memory_id}")
async def remove_memory(
    memory_id: str,
    user: UserRecord = Depends(current_user),
) -> dict[str, str]:
    """Delete one of the authenticated user's memories."""
    if not await delete_user_memory(user["id"], memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "ok"}


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
