"""Sign up / log in / me, with users stored in Turso.

Passwords are hashed with scrypt (standard library). Sessions are signed JWTs (HS256).
Required environment variables: TURSO_DATABASE_URL, TURSO_AUTH_TOKEN, JWT_SECRET (32+ chars).
Optional: OFFICER_CODE (people who know it can sign up as an officer; if unset, nobody can).
"""
import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, field_validator

import db

router = APIRouter(prefix="/auth", tags=["auth"])

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
TOKEN_DAYS, MAX_FAILS, WINDOW_SECONDS = 7, 8, 900
_fails: dict[str, list[float]] = defaultdict(list)
_schema_ready = False

SCHEMA = """CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL DEFAULT 'citizen' CHECK (role IN ('citizen','officer')),
  password_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
)"""


# ---------- passwords ----------
def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${salt.hex()}${h.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, bytes.fromhex(h))
    except Exception:
        return False


_DUMMY = hash_password("not-a-real-password")  # keeps login timing similar for unknown emails


# ---------- tokens ----------
def configured() -> bool:
    return db.configured() and len(os.getenv("JWT_SECRET", "")) >= 32


def _secret() -> str:
    s = os.getenv("JWT_SECRET", "")
    if len(s) < 32:
        raise HTTPException(503, "Accounts are not set up on the server yet.")
    return s


def make_token(user: dict) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": str(user["id"]), "role": user["role"], "iat": now, "exp": now + TOKEN_DAYS * 86400},
        _secret(),
        algorithm="HS256",
    )


def public(user: dict) -> dict:
    return {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}


async def _ready():
    global _schema_ready
    if not configured():
        raise HTTPException(503, "Accounts are not set up on the server yet.")
    if not _schema_ready:
        try:
            await db.execute(SCHEMA)
        except db.TursoError:
            raise HTTPException(503, "The account database is not reachable. Try again in a moment.")
        _schema_ready = True


async def _query(sql: str, args=()):
    try:
        return await db.execute(sql, args)
    except db.TursoError as e:
        if "UNIQUE" in str(e).upper():
            raise
        raise HTTPException(503, "The account database is not reachable. Try again in a moment.")


# ---------- rate limit (in memory, resets when the server restarts) ----------
def _key(request: Request, email: str) -> str:
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else "?")
    return f"{ip}|{email}"


def _check_limit(key: str):
    now = time.time()
    _fails[key] = [t for t in _fails[key] if now - t < WINDOW_SECONDS]
    if len(_fails[key]) >= MAX_FAILS:
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")


# ---------- request bodies ----------
class Signup(BaseModel):
    name: str
    email: str
    password: str
    role: str = "citizen"
    officer_code: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        v = v.strip()
        if not 1 <= len(v) <= 80:
            raise ValueError("Enter your name (up to 80 characters)")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v):
        v = v.strip().lower()
        if len(v) > 254 or not EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def _pw(cls, v):
        if not 8 <= len(v) <= 128:
            raise ValueError("Use a password of 8 to 128 characters")
        return v

    @field_validator("role")
    @classmethod
    def _role(cls, v):
        if v not in ("citizen", "officer"):
            raise ValueError("Role must be citizen or officer")
        return v


class Login(BaseModel):
    email: str
    password: str


# ---------- routes ----------
@router.post("/signup", status_code=201)
async def signup(body: Signup):
    await _ready()
    if body.role == "officer":
        expected = os.getenv("OFFICER_CODE", "")
        if not expected:
            raise HTTPException(403, "Officer sign-ups are turned off.")
        if not hmac.compare_digest((body.officer_code or "").encode(), expected.encode()):
            raise HTTPException(403, "That officer access code is not correct.")
    if await _query("SELECT id FROM users WHERE email = ?", [body.email]):
        raise HTTPException(409, "That email already has an account.")
    try:
        await _query(
            "INSERT INTO users (name, email, role, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            [body.name, body.email, body.role, hash_password(body.password), datetime.now(timezone.utc).isoformat()],
        )
    except db.TursoError:  # two sign-ups with the same email at the same moment
        raise HTTPException(409, "That email already has an account.")
    rows = await _query("SELECT id, name, email, role FROM users WHERE email = ?", [body.email])
    return {"token": make_token(rows[0]), "user": public(rows[0])}


@router.post("/login")
async def login(body: Login, request: Request):
    await _ready()
    email = body.email.strip().lower()
    key = _key(request, email)
    _check_limit(key)
    rows = await _query("SELECT id, name, email, role, password_hash FROM users WHERE email = ?", [email])
    user = rows[0] if rows else None
    ok = verify_password(body.password, user["password_hash"] if user else _DUMMY)
    if not (user and ok):
        _fails[key].append(time.time())
        raise HTTPException(401, "Wrong email or password.")
    _fails.pop(key, None)
    return {"token": make_token(user), "user": public(user)}


@router.get("/me")
async def me(authorization: str | None = Header(None)):
    await _ready()
    token = (authorization or "").removeprefix("Bearer ").strip()
    try:
        claims = jwt.decode(token, _secret(), algorithms=["HS256"])
        uid = int(claims["sub"])
    except Exception:
        raise HTTPException(401, "Please log in again.")
    rows = await _query("SELECT id, name, email, role FROM users WHERE id = ?", [uid])
    if not rows:
        raise HTTPException(401, "Please log in again.")
    return {"user": public(rows[0])}
