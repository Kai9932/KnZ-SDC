"""Chatbot API. Mounted from main.py with app.include_router(chatbot_router)."""
import hashlib
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models import chat_store
from models.career_chatbot import ChatbotError, reply

router = APIRouter(prefix="/api/chatbot")

HISTORY_WINDOW = 12            # messages sent to DeepSeek per request
RATE_LIMIT = 15                # messages ...
RATE_WINDOW = 60               # ... per this many seconds, per user
NO_STORE = {"Cache-Control": "no-store"}

chat_store.init_db()

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _uid(request: Request):
    """Same hashed id as _uid() in main.py, without importing main."""
    u = request.session.get("user")
    if u and u.get("sub"):
        return hashlib.sha256(str(u["sub"]).encode()).hexdigest()
    return None


def _rate_limited(uid: str) -> bool:
    now = time.monotonic()
    with _hits_lock:
        q = _hits[uid]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        return False


def _err(message: str, status: int):
    return JSONResponse({"error": message}, status_code=status, headers=NO_STORE)


class MessageIn(BaseModel):
    conversation_id: str | None = Field(None, max_length=64)
    message: str = Field(..., max_length=1000)


# Plain `def` endpoints: SQLite and the DeepSeek call block, so FastAPI
# runs them in a worker thread.

@router.get("/conversations")
def list_chats(request: Request):
    uid = _uid(request)
    if not uid:
        return _err("Sign in to use the AI Career Counsellor.", 401)
    return JSONResponse({"conversations": chat_store.list_conversations(uid)}, headers=NO_STORE)


@router.get("/conversations/{cid}")
def get_chat(cid: str, request: Request):
    uid = _uid(request)
    if not uid:
        return _err("Sign in to use the AI Career Counsellor.", 401)
    conv = chat_store.get_conversation(uid, cid)
    if not conv:
        return _err("Conversation not found.", 404)
    return JSONResponse(conv, headers=NO_STORE)


@router.delete("/conversations/{cid}")
def delete_chat(cid: str, request: Request):
    uid = _uid(request)
    if not uid:
        return _err("Sign in first.", 401)
    if not chat_store.delete_conversation(uid, cid):
        return _err("Conversation not found.", 404)
    return JSONResponse({"ok": True}, headers=NO_STORE)


@router.delete("/conversations")
def clear_chats(request: Request):
    uid = _uid(request)
    if not uid:
        return _err("Sign in first.", 401)
    return JSONResponse({"ok": True, "deleted": chat_store.delete_all(uid)}, headers=NO_STORE)


@router.post("/message")
def send_message(body: MessageIn, request: Request):
    uid = _uid(request)
    if not uid:
        return _err("Sign in to use the AI Career Counsellor.", 401)

    text = body.message.strip()
    if not text:
        return _err("Type a message first.", 400)
    if _rate_limited(uid):
        return _err("You're sending messages too quickly. Wait a moment and try again.", 429)

    if body.conversation_id:
        conv = chat_store.get_conversation(uid, body.conversation_id)
        if not conv:
            return _err("Conversation not found.", 404)
        cid, title = conv["id"], conv["title"]
    else:
        conv = chat_store.create_conversation(uid, text)
        cid, title = conv["id"], conv["title"]

    if not chat_store.add_message(uid, cid, "user", text):
        return _err("This chat is full. Start a new chat to continue.", 400)

    history = chat_store.recent_messages(uid, cid, HISTORY_WINDOW)
    try:
        answer = reply(history)
    except ChatbotError as e:
        # The user's message stays saved, so they can simply retry.
        return JSONResponse(
            {"error": str(e), "conversation_id": cid, "title": title},
            status_code=502, headers=NO_STORE)

    chat_store.add_message(uid, cid, "assistant", answer)
    return JSONResponse(
        {"conversation_id": cid, "title": title, "reply": answer}, headers=NO_STORE)