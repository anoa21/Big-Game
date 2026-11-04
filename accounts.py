"""Accounts: sign in, and keep your leagues between visits and across devices.

What is stored: a username, a scrambled (scrypt-hashed) password, a hashed recovery code, and the
leagues a person chooses to save (platform, Sleeper username or ESPN link or typed player names).
What is never stored: ESPN cookies, fantasy-site passwords, or Yahoo tokens, and no email address.

Where it lives: any SQL database named by DATABASE_URL (Postgres, such as a free Neon database).
With no DATABASE_URL it uses a file (ACCOUNTS_DB, default /tmp). That survives restarts only if the
file sits on a persistent disk, for example ACCOUNTS_DB=/var/data/accounts.sqlite3 with a Render disk
mounted at /var/data. Without a disk, Render wipes it on every restart and accounts would not survive. /api/auth/me says which one is in use.

Security choices: scrypt password hashing with a random salt; random session tokens kept only as
hashes in the database and sent in an HttpOnly, SameSite=Lax (and Secure on https) cookie; a check
that state-changing requests come from this same site; limits on guessing passwords; the same
answer for "no such user" and "wrong password"; every query parameterized.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# ---- guests and accounts: which features need a free account ------------------------------------------
# GATED_FEATURES (environment variable, comma separated) chooses what guests can't use. Change it on Render and
# restart; no code change needed. Names: plan, league, trade, playbook, coach_ai. Empty means everything is open.
FEATURES = {"plan": "Game plan", "league": "League outlook", "trade": "Trade analyzer", "playbook": "Coach's Playbook", "coach_ai": "AI coach"}
DEFAULT_GATED = "plan,league,coach_ai"


def gated():
    raw = os.getenv("GATED_FEATURES", DEFAULT_GATED)
    return [f for f in (x.strip() for x in raw.split(",")) if f in FEATURES]


def require(request, feature):
    """Raise 401 if `feature` is locked for guests and nobody is signed in. If accounts can't be checked (the database
    is down), the feature stays open: a broken login service must never lock everyone out of the app."""
    if feature not in gated():
        return
    try:
        user = current_user(request)
    except Exception:
        return
    if not user:
        raise HTTPException(401, f"Sign in to use the {FEATURES[feature]}. It's free.")
SESSION_DAYS = 30
MAX_LEAGUES = 8
COOKIE = "bg_session"
COMMON = {"password", "password1", "12345678", "123456789", "1234567890", "qwerty123", "iloveyou", "football",
          "letmein1", "11111111", "abcd1234", "password123", "fantasyfootball", "footballfan"}

# ----------------------------------------------------------------------------- database --
md = sa.MetaData()
users = sa.Table("bg_users", md,
                 sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                 sa.Column("username", sa.String(30), unique=True, nullable=False),
                 sa.Column("display", sa.String(30), nullable=False),
                 sa.Column("pw_hash", sa.Text, nullable=False),
                 sa.Column("recovery_hash", sa.String(64), nullable=False),
                 sa.Column("created_at", sa.Integer, nullable=False))
sessions = sa.Table("bg_sessions", md,
                    sa.Column("token_hash", sa.String(64), primary_key=True),
                    sa.Column("user_id", sa.Integer, nullable=False, index=True),
                    sa.Column("created_at", sa.Integer, nullable=False),
                    sa.Column("expires_at", sa.Integer, nullable=False, index=True))
leagues = sa.Table("bg_leagues", md,
                   sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                   sa.Column("user_id", sa.Integer, nullable=False, index=True),
                   sa.Column("key", sa.String(300), nullable=False),
                   sa.Column("platform", sa.String(10), nullable=False),
                   sa.Column("label", sa.String(60), nullable=False),
                   sa.Column("payload", sa.JSON, nullable=False),
                   sa.Column("created_at", sa.Integer, nullable=False),
                   sa.UniqueConstraint("user_id", "key", name="uq_bg_league_key"))

_engine, _lock = None, threading.Lock()


def db_url():
    u = os.getenv("DATABASE_URL", "").strip()
    if not u:
        return "sqlite:///" + os.getenv("ACCOUNTS_DB", "/tmp/bg_accounts.sqlite3")
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://"):]
    if u.startswith("postgresql://"):
        u = "postgresql+psycopg2://" + u[len("postgresql://"):]
    return u


def persistent():
    """True when accounts live somewhere that survives a restart: a database, or a file on a mounted disk
    (setting ACCOUNTS_DB means the owner pointed it at a persistent disk)."""
    return bool(os.getenv("DATABASE_URL", "").strip()) or bool(os.getenv("ACCOUNTS_DB", "").strip())


def engine():
    """Connect on first use, so a database hiccup never stops the rest of the app from starting."""
    global _engine
    with _lock:
        if _engine is None:
            url = db_url()
            kw = {"pool_pre_ping": True, "pool_recycle": 240,          # idle serverless databases drop connections
                  "connect_args": {"connect_timeout": 8}}              # never let a dead database hang a page
            if url.startswith("sqlite"):
                path = url[len("sqlite:///"):]
                if os.path.dirname(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                kw = {"connect_args": {"check_same_thread": False, "timeout": 15}}
            e = sa.create_engine(url, **kw)
            if url.startswith("sqlite"):                                # safer for several requests at once
                @sa.event.listens_for(e, "connect")
                def _pragmas(dbapi_conn, _):
                    cur = dbapi_conn.cursor(); cur.execute("PRAGMA journal_mode=WAL"); cur.execute("PRAGMA synchronous=NORMAL"); cur.close()
            md.create_all(e)
            _engine = e
        return _engine


def reset_engine_for_tests():
    global _engine
    _engine = None


def need_db():
    try:
        return engine()
    except Exception:
        raise HTTPException(503, "Accounts are temporarily unavailable. The rest of the app still works.")


# ----------------------------------------------------------------------------- passwords --
def _b64(b):
    return base64.b64encode(b).decode()


def hash_password(pw):
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${_b64(salt)}${_b64(h)}"


def verify_password(pw, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(h))
    except Exception:
        return False


DUMMY = hash_password("not-a-real-password")      # so unknown usernames take as long to check as real ones


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def new_recovery_code():
    raw = base64.b32encode(os.urandom(16)).decode().rstrip("=")           # 128 bits
    return "-".join(raw[i:i + 5] for i in range(0, len(raw), 5))


def clean_code(code):
    return re.sub(r"[^A-Z2-7]", "", (code or "").upper())


def check_password_rules(pw, username=""):
    if not isinstance(pw, str) or len(pw) < 8:
        raise HTTPException(400, "Use a password with at least 8 characters.")
    if len(pw) > 128:
        raise HTTPException(400, "That password is too long. Use 128 characters or fewer.")
    if pw.lower() in COMMON or pw.lower() == (username or "").lower():
        raise HTTPException(400, "That password is too easy to guess. Pick something less common.")


def check_username(u):
    u = (u or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{3,30}", u):
        raise HTTPException(400, "Usernames are 3 to 30 characters: letters, numbers, dots, dashes and underscores.")
    return u


# ----------------------------------------------------------------------------- guards --
_fails, _hits = defaultdict(deque), defaultdict(deque)


def client_ip(request):
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def _recent(dq, window):
    now = time.time()
    while dq and now - dq[0] > window:
        dq.popleft()
    return dq


def throttle(request, username=""):
    """At most 30 sign-in attempts per visitor and 8 failures per visitor and name, per 15 minutes."""
    ip = client_ip(request)
    hits = _recent(_hits[ip], 900)
    if len(hits) >= 30 or len(_recent(_fails[(ip, username.lower())], 900)) >= 8:
        raise HTTPException(429, "Too many attempts. Wait about 15 minutes and try again.")
    hits.append(time.time())


def fail(request, username):
    _fails[(client_ip(request), username.lower())].append(time.time())


def same_site(request):
    """Refuse state-changing requests that were triggered from another website."""
    origin = request.headers.get("origin")
    if origin:
        host = urlparse(origin).netloc
        allowed = {request.headers.get("host"), request.headers.get("x-forwarded-host")}
        if host not in allowed:
            raise HTTPException(403, "That request came from another website, so it was blocked.")


def is_https(request):
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


def start_session(conn, response, request, user_id):
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    conn.execute(sessions.delete().where(sessions.c.expires_at < now))               # tidy up old ones
    conn.execute(sessions.insert().values(token_hash=sha(token), user_id=user_id, created_at=now,
                                          expires_at=now + SESSION_DAYS * 86400))
    response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                        secure=is_https(request), path="/")


def current_user(request):
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    with need_db().connect() as conn:
        row = conn.execute(sa.select(users.c.id, users.c.display, users.c.username)
                           .select_from(sessions.join(users, users.c.id == sessions.c.user_id))
                           .where(sessions.c.token_hash == sha(token), sessions.c.expires_at > int(time.time()))).first()
    return None if not row else {"id": row.id, "display": row.display, "username": row.username}


def require_user(request):
    u = current_user(request)
    if not u:
        raise HTTPException(401, "Please sign in first.")
    return u


# ----------------------------------------------------------------------------- saved leagues --
def sanitize_league(platform, label, payload):
    """Keep only the safe fields for each platform. Cookies and tokens are never accepted."""
    payload = payload if isinstance(payload, dict) else {}
    if platform == "sleeper":
        u = str(payload.get("username", "")).strip()[:40]
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,40}", u):
            raise HTTPException(400, "That doesn't look like a Sleeper username.")
        return f"sleeper:{u.lower()}", {"username": u}, label or f"Sleeper: {u}"
    if platform == "espn":
        url = str(payload.get("url", "")).strip()[:300]
        lid, tid = re.search(r"leagueId=(\d+)", url), re.search(r"teamId=(\d+)", url)
        if not lid or not tid:
            raise HTTPException(400, "Save an ESPN team page link that has leagueId= and teamId= in it.")
        return f"espn:{lid.group(1)}:{tid.group(1)}", {"url": url}, label or f"ESPN league {lid.group(1)}"
    if platform == "manual":
        names = [re.sub(r"\s+", " ", str(n)).strip()[:60] for n in (payload.get("players") or [])[:200]]
        names = [n for n in names if len(n) >= 3][:40]               # drop blanks first, then cap
        if not names:
            raise HTTPException(400, "There are no players to save.")
        name = (label or "My typed roster")[:60]
        return f"manual:{name.lower()}", {"players": names}, name
    raise HTTPException(400, "Unknown league type.")


def league_rows(conn, user_id):
    rows = conn.execute(sa.select(leagues).where(leagues.c.user_id == user_id).order_by(leagues.c.created_at, leagues.c.id)).all()
    return [{"id": r.id, "platform": r.platform, "label": r.label, "payload": r.payload} for r in rows]


# ----------------------------------------------------------------------------- endpoints --
class Creds(BaseModel):
    username: str
    password: str


class Recover(BaseModel):
    username: str
    recovery_code: str
    new_password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class Confirm(BaseModel):
    password: str


class LeagueIn(BaseModel):
    platform: str
    label: Optional[str] = None
    payload: dict


@router.get("/api/auth/me")
def me(request: Request):
    try:
        u = current_user(request)
        out = {"user": None, "leagues": [], "persistent": persistent(), "locked": gated()}
        if u:
            with engine().connect() as conn:
                out["user"] = {"username": u["display"]}
                out["leagues"] = league_rows(conn, u["id"])
        return out
    except HTTPException:
        return {"user": None, "leagues": [], "persistent": persistent(), "locked": [], "unavailable": True}


@router.post("/api/auth/signup")
def signup(c: Creds, request: Request, response: Response):
    same_site(request)
    name = check_username(c.username)
    check_password_rules(c.password, name)
    throttle(request, name)
    code = new_recovery_code()
    eng = need_db()
    try:
        with eng.begin() as conn:
            uid = conn.execute(users.insert().values(username=name.lower(), display=name, pw_hash=hash_password(c.password),
                                                     recovery_hash=sha(clean_code(code)), created_at=int(time.time()))).inserted_primary_key[0]
            start_session(conn, response, request, uid)
    except sa.exc.IntegrityError:
        raise HTTPException(400, "That username is taken. Try another one.")
    return {"user": {"username": name}, "recovery_code": code, "leagues": [], "persistent": persistent(), "locked": gated()}


@router.post("/api/auth/login")
def login(c: Creds, request: Request, response: Response):
    same_site(request)
    name = (c.username or "").strip()[:30]
    throttle(request, name)
    with need_db().begin() as conn:
        row = conn.execute(sa.select(users).where(users.c.username == name.lower())).first()
        ok = verify_password(c.password or "", row.pw_hash if row else DUMMY) and row is not None
        if not ok:
            fail(request, name)
            raise HTTPException(401, "That username and password don't match.")
        start_session(conn, response, request, row.id)
        return {"user": {"username": row.display}, "leagues": league_rows(conn, row.id), "persistent": persistent(), "locked": gated()}


@router.post("/api/auth/logout")
def logout(request: Request, response: Response):
    same_site(request)
    token = request.cookies.get(COOKIE)
    if token:
        with need_db().begin() as conn:
            conn.execute(sessions.delete().where(sessions.c.token_hash == sha(token)))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.post("/api/auth/recover")
def recover(r: Recover, request: Request, response: Response):
    """Forgot your password? The recovery code you saved at sign-up sets a new one."""
    same_site(request)
    name = (r.username or "").strip()[:30]
    throttle(request, name)
    check_password_rules(r.new_password, name)
    with need_db().begin() as conn:
        row = conn.execute(sa.select(users).where(users.c.username == name.lower())).first()
        good = row is not None and hmac.compare_digest(row.recovery_hash, sha(clean_code(r.recovery_code)))
        if not good:
            fail(request, name)
            raise HTTPException(401, "That username and recovery code don't match.")
        new_code = new_recovery_code()
        conn.execute(users.update().where(users.c.id == row.id).values(pw_hash=hash_password(r.new_password),
                                                                         recovery_hash=sha(clean_code(new_code))))
        conn.execute(sessions.delete().where(sessions.c.user_id == row.id))           # sign out everywhere else
        start_session(conn, response, request, row.id)
        return {"user": {"username": row.display}, "recovery_code": new_code, "leagues": league_rows(conn, row.id),
                "persistent": persistent(), "locked": gated()}


@router.post("/api/auth/password")
def change_password(p: PasswordChange, request: Request, response: Response):
    same_site(request)
    u = require_user(request)
    throttle(request, u["username"])
    check_password_rules(p.new_password, u["username"])
    with need_db().begin() as conn:
        row = conn.execute(sa.select(users).where(users.c.id == u["id"])).first()
        if not verify_password(p.current_password or "", row.pw_hash):
            fail(request, u["username"])
            raise HTTPException(401, "Your current password isn't right.")
        conn.execute(users.update().where(users.c.id == u["id"]).values(pw_hash=hash_password(p.new_password)))
        conn.execute(sessions.delete().where(sessions.c.user_id == u["id"]))
        start_session(conn, response, request, u["id"])
    return {"ok": True}


@router.post("/api/auth/delete")
def delete_account(c: Confirm, request: Request, response: Response):
    same_site(request)
    u = require_user(request)
    throttle(request, u["username"])
    with need_db().begin() as conn:
        row = conn.execute(sa.select(users).where(users.c.id == u["id"])).first()
        if not verify_password(c.password or "", row.pw_hash):
            fail(request, u["username"])
            raise HTTPException(401, "That password isn't right.")
        conn.execute(leagues.delete().where(leagues.c.user_id == u["id"]))
        conn.execute(sessions.delete().where(sessions.c.user_id == u["id"]))
        conn.execute(users.delete().where(users.c.id == u["id"]))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.post("/api/me/leagues")
def save_league(body: LeagueIn, request: Request):
    same_site(request)
    u = require_user(request)
    key, payload, label = sanitize_league(body.platform, (body.label or "").strip()[:60], body.payload)
    with need_db().begin() as conn:
        have = conn.execute(sa.select(leagues.c.id).where(leagues.c.user_id == u["id"], leagues.c.key == key)).first()
        if have:
            conn.execute(leagues.update().where(leagues.c.id == have.id).values(payload=payload, label=label))
        else:
            n = conn.execute(sa.select(sa.func.count()).select_from(leagues).where(leagues.c.user_id == u["id"])).scalar()
            if n >= MAX_LEAGUES:
                raise HTTPException(400, f"You can save up to {MAX_LEAGUES} leagues. Remove one first.")
            conn.execute(leagues.insert().values(user_id=u["id"], key=key, platform=body.platform, label=label,
                                                 payload=payload, created_at=int(time.time())))
        return {"leagues": league_rows(conn, u["id"])}


@router.delete("/api/me/leagues/{league_id}")
def remove_league(league_id: int, request: Request):
    same_site(request)
    u = require_user(request)
    with need_db().begin() as conn:
        conn.execute(leagues.delete().where(leagues.c.id == league_id, leagues.c.user_id == u["id"]))   # only your own
        return {"leagues": league_rows(conn, u["id"])}
