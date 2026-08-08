"""
FastAPI Chatbot with Memory, Real Accounts, and Auto-Cleanup
----------------------------------------------------------------
Changes from the previous version:

1. REAL USER ACCOUNTS (replaces shared API keys)
   - POST /auth/register creates a user with a hashed password.
   - POST /auth/login checks the password and returns a JWT access
     token (expires after ACCESS_TOKEN_EXPIRE_MINUTES).
   - Every other endpoint requires that token in the header:
       Authorization: Bearer <token>
   - Session ownership is now tied to a username instead of a
     shared secret key — each person has their own login instead
     of everyone sharing a handful of hardcoded keys.

2. MESSAGE LIMIT
   - Each session keeps at most MAX_MESSAGES_PER_SESSION messages in
     SQLite. Older messages are trimmed automatically after every
     chat turn, so chat_memory.db can't grow forever for one very
     long-lived session. (This is separate from WINDOW_SIZE, which
     controls how much of that history is sent to the LLM per
     request — trimming controls storage, WINDOW_SIZE controls
     what the model sees.)

3. AUTO-EXPIRY
   - Sessions untouched for SESSION_EXPIRY_DAYS are deleted
     automatically — both the session record and its stored
     messages. Cleanup runs once when the server starts, then every
     24 hours in the background for as long as the server keeps
     running.

4. LOGOUT / TOKEN REVOCATION
   - Every token now carries a unique ID (`jti`). POST /auth/logout
     records that ID in a revoked-tokens table, so the token stops
     working immediately — instead of waiting for it to expire on
     its own. Useful if a device is lost or a token leaks.

5. RATE LIMITING
   - /chat is limited to CHAT_RATE_LIMIT per account (default:
     10 requests/minute), so one account can't hammer the endpoint
     (and, if you later swap Ollama for a paid API, run up a bill).

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs
"""

import os
import sqlite3
import secrets
import asyncio
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import bcrypt
from jose import JWTError, ExpiredSignatureError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

# ---- Config ----
MODEL_NAME = "llama3.2"
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers concise."
WINDOW_SIZE = 5                    # exchanges sent to the LLM per request
MAX_MESSAGES_PER_SESSION = 200     # hard cap on stored messages per session
SESSION_EXPIRY_DAYS = 30           # sessions idle longer than this get deleted
CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60  # how often the background cleanup runs
CHAT_RATE_LIMIT = "10/minute"      # per-account limit on the /chat endpoint

DB_FILE = "chat_memory.db"
DB_PATH = f"sqlite:///{DB_FILE}"

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

if not SECRET_KEY:
    raise RuntimeError(
        "No SECRET_KEY configured. Add SECRET_KEY=<a long random string> to your .env file. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

llm = ChatOllama(model=MODEL_NAME, temperature=0.7)
memory_cache: dict[str, ConversationBufferWindowMemory] = {}


# ---------- Database helpers ----------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   username TEXT PRIMARY KEY,
                   password_hash TEXT NOT NULL,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                   session_id TEXT PRIMARY KEY,
                   username TEXT NOT NULL,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                   last_active TEXT DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        # Migration: older versions of this project created `sessions`
        # without `last_active`. CREATE TABLE IF NOT EXISTS won't add
        # missing columns to an existing table, so check for it and
        # add it by hand if it's missing.
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "last_active" not in existing_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN last_active TEXT DEFAULT CURRENT_TIMESTAMP"
            )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS revoked_tokens (
                   jti TEXT PRIMARY KEY,
                   expires_at TEXT NOT NULL
               )"""
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- Auth: password hashing + JWTs ----------

def hash_password(password: str) -> str:
    # bcrypt only uses the first 72 bytes of a password - anything
    # longer is silently ignored, so we reject it up front instead
    # (validated in the /auth/register route).
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire, "jti": secrets.token_hex(16)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def is_token_revoked(jti: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT jti FROM revoked_tokens WHERE jti = ?", (jti,)
        ).fetchone()
    return row is not None


def revoke_token(jti: str, expires_at: datetime) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO revoked_tokens (jti, expires_at) VALUES (?, ?)",
            (jti, expires_at.isoformat()),
        )


def cleanup_expired_revocations() -> int:
    """Revocation entries are only needed until the token itself
    would have expired anyway - after that, prune them so the table
    doesn't grow forever."""
    now = utcnow_iso()
    with get_db() as conn:
        expired = conn.execute(
            "SELECT jti FROM revoked_tokens WHERE expires_at < ?", (now,)
        ).fetchall()
        conn.execute("DELETE FROM revoked_tokens WHERE expires_at < ?", (now,))
    return len(expired)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    unauthorized = HTTPException(
        status_code=401,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        jti = payload.get("jti")
        if username is None or jti is None:
            raise unauthorized
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired, please log in again")
    except JWTError:
        raise unauthorized

    if is_token_revoked(jti):
        raise HTTPException(status_code=401, detail="Token has been logged out, please log in again")

    with get_db() as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        raise unauthorized
    return username


def get_current_user_and_jti(token: str = Depends(oauth2_scheme)) -> tuple[str, str, datetime]:
    """Same checks as get_current_user, but also hands back the jti
    and expiry - only /auth/logout needs these, so it's kept separate
    rather than changing every route's return type."""
    unauthorized = HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        jti = payload.get("jti")
        exp = payload.get("exp")
        if username is None or jti is None or exp is None:
            raise unauthorized
    except JWTError:
        raise unauthorized
    return username, jti, datetime.fromtimestamp(exp, tz=timezone.utc)


# ---------- Session ownership ----------

def create_session(username: str) -> str:
    session_id = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, username, last_active) VALUES (?, ?, ?)",
            (session_id, username, utcnow_iso()),
        )
    return session_id


def session_belongs_to(session_id: str, username: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT username FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row is not None and row[0] == username


def touch_session(session_id: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (utcnow_iso(), session_id),
        )


def delete_session_everywhere(session_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))


def trim_old_messages(session_id: str, max_messages: int) -> None:
    """Keep only the most recent `max_messages` rows for this session
    in the message_store table (created automatically by
    SQLChatMessageHistory the first time a session is used)."""
    with get_db() as conn:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_store'"
        ).fetchone()
        if not table_exists:
            return
        conn.execute(
            """DELETE FROM message_store
               WHERE session_id = ? AND id NOT IN (
                   SELECT id FROM message_store
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?
               )""",
            (session_id, session_id, max_messages),
        )


def cleanup_expired_sessions() -> int:
    """Delete sessions (and their messages) idle longer than
    SESSION_EXPIRY_DAYS. Returns how many were removed."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()
    with get_db() as conn:
        expired = conn.execute(
            "SELECT session_id FROM sessions WHERE last_active < ?", (cutoff,)
        ).fetchall()
        for (session_id,) in expired:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
    return len(expired)


async def periodic_cleanup():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        removed_sessions = cleanup_expired_sessions()
        removed_tokens = cleanup_expired_revocations()
        if removed_sessions or removed_tokens:
            print(f"[cleanup] removed {removed_sessions} expired session(s), "
                  f"{removed_tokens} stale revocation record(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    removed_sessions = cleanup_expired_sessions()
    removed_tokens = cleanup_expired_revocations()
    if removed_sessions or removed_tokens:
        print(f"[startup cleanup] removed {removed_sessions} expired session(s), "
              f"{removed_tokens} stale revocation record(s)")
    task = asyncio.create_task(periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(title="Simple Chatbot with Memory", lifespan=lifespan)


# ---------- Rate limiting ----------

def rate_limit_key(request: Request) -> str:
    """Rate-limit per logged-in account rather than per IP, so it
    can't be dodged by switching networks, and so shared networks
    (e.g. a school Wi-Fi) don't get one account's limit applied to
    everyone on it. Falls back to IP address if there's no valid
    token (e.g. hitting /auth/login too fast)."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        try:
            payload = jwt.decode(
                token, SECRET_KEY, algorithms=[ALGORITHM],
                options={"verify_exp": False},  # just identifying the caller here
            )
            username = payload.get("sub")
            if username:
                return f"user:{username}"
        except JWTError:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------- Chat memory helpers ----------

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in memory_cache:
        history = SQLChatMessageHistory(session_id=session_id, connection=DB_PATH)
        memory_cache[session_id] = ConversationBufferWindowMemory(
            k=WINDOW_SIZE, return_messages=True, chat_memory=history
        )
    return memory_cache[session_id]


# ---------- Schemas ----------

class UserCreate(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


# ---------- Auth routes ----------

@app.post("/auth/register", status_code=201)
def register(user: UserCreate):
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes or fewer")

    with get_db() as conn:
        existing = conn.execute(
            "SELECT username FROM users WHERE username = ?", (user.username,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (user.username, hash_password(user.password)),
        )
    return {"status": "registered", "username": user.username}


@app.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_db() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (form_data.username,)
        ).fetchone()

    if row is None or not verify_password(form_data.password, row[0]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(form_data.username)
    return TokenResponse(access_token=token)


@app.post("/auth/logout")
def logout(auth: tuple[str, str, datetime] = Depends(get_current_user_and_jti)):
    """Invalidates the token that's sent with this request. The
    token was going to expire eventually anyway - this just makes
    it stop working right now, e.g. because a device was lost."""
    _username, jti, expires_at = auth
    revoke_token(jti, expires_at)
    return {"status": "logged out"}


# ---------- Chat routes ----------

@app.post("/session/start", response_model=SessionResponse)
def start_session(username: str = Depends(get_current_user)):
    session_id = create_session(username)
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
def chat(request: Request, req: ChatRequest, username: str = Depends(get_current_user)):
    if not session_belongs_to(req.session_id, username):
        raise HTTPException(status_code=403, detail="Invalid session")

    memory = get_memory(req.session_id)
    memory.chat_memory.add_user_message(req.message)

    windowed_history = memory.load_memory_variables({})["history"]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + windowed_history

    response = llm.invoke(messages)
    memory.chat_memory.add_ai_message(response.content)

    trim_old_messages(req.session_id, MAX_MESSAGES_PER_SESSION)
    touch_session(req.session_id)

    return ChatResponse(reply=response.content)


@app.post("/reset/{session_id}")
def reset(session_id: str, username: str = Depends(get_current_user)):
    if not session_belongs_to(session_id, username):
        raise HTTPException(status_code=403, detail="Invalid session")

    memory_cache.pop(session_id, None)
    delete_session_everywhere(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/")
def health():
    return {"status": "ok"}