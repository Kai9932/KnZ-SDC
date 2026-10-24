"""SQLite storage for the AI Career Counsellor.

Every function takes the hashed user id (the same value as `_uid(request)` in
main.py) and filters on it, so one user can never read or change another
user's conversations.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chatbot.db"

MAX_CONVERSATIONS = 50          # oldest chats are removed beyond this
MAX_MESSAGES_PER_CHAT = 200     # stops a single chat growing forever


@contextmanager
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with _db() as con:
        con.execute("PRAGMA journal_mode = WAL")
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_user
                ON conversations (user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations (id) ON DELETE CASCADE,
                role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content         TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv
                ON messages (conversation_id, id);
            """
        )


def create_conversation(user_id: str, title: str) -> dict:
    cid, now = uuid.uuid4().hex, _now()
    title = (title or "New chat").strip()[:60] or "New chat"
    with _db() as con:
        con.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, user_id, title, now, now),
        )
        # Keep only the newest MAX_CONVERSATIONS chats for this user.
        con.execute(
            "DELETE FROM conversations WHERE user_id = ? AND id NOT IN ("
            "  SELECT id FROM conversations WHERE user_id = ? "
            "  ORDER BY updated_at DESC LIMIT ?)",
            (user_id, user_id, MAX_CONVERSATIONS),
        )
    return {"id": cid, "title": title, "created_at": now, "updated_at": now}


def list_conversations(user_id: str) -> list[dict]:
    with _db() as con:
        rows = con.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(user_id: str, cid: str) -> dict | None:
    """Conversation with all its messages, or None if it isn't this user's."""
    with _db() as con:
        conv = con.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
        if not conv:
            return None
        msgs = con.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (cid,),
        ).fetchall()
    return {**dict(conv), "messages": [dict(m) for m in msgs]}


def add_message(user_id: str, cid: str, role: str, content: str) -> bool:
    """Append a message. Returns False if the chat isn't this user's or is full."""
    now = _now()
    with _db() as con:
        owned = con.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
        if not owned:
            return False
        count = con.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,)
        ).fetchone()[0]
        if count >= MAX_MESSAGES_PER_CHAT:
            return False
        con.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (cid, role, content, now),
        )
        con.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
    return True


def recent_messages(user_id: str, cid: str, limit: int = 12) -> list[dict]:
    """The last `limit` messages (oldest first), formatted for the LLM."""
    with _db() as con:
        rows = con.execute(
            "SELECT m.role, m.content FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE m.conversation_id = ? AND c.user_id = ? "
            "ORDER BY m.id DESC LIMIT ?",
            (cid, user_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def delete_conversation(user_id: str, cid: str) -> bool:
    with _db() as con:
        cur = con.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?", (cid, user_id)
        )
        return cur.rowcount > 0


def delete_all(user_id: str) -> int:
    with _db() as con:
        cur = con.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        return cur.rowcount