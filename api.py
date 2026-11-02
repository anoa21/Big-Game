"""Web API + static front end.   Run:  uvicorn api:app --reload

Optional environment variables:
  ANTHROPIC_API_KEY   turns on AI news notes
  YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, BASE_URL   turn on Yahoo import
"""
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import nflreadpy as nfl
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import insight
import model
import providers
import sim
import sleeper

app = FastAPI(title="Big Game")
STATIC = Path(__file__).parent / "static"
BASE_TTL, ENRICH_TTL = 6 * 3600, 30 * 60   # model every 6h, injuries every 30min
_base, _enriched, _lock = {}, {}, threading.Lock()
_sessions = {}                             # Yahoo logins, in memory only


def current_week():
    try:
        return int(nfl.get_current_season()), int(nfl.get_current_week())
    except Exception:
        return 2025, 9


def _retry(fn, tries=3, delay=3):
    """The free nflverse data host occasionally 404s for a moment while it
    updates, so retry before giving up."""
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def predictions(season, week):
    key = (season, week)
    with _lock:
        now = time.time()
        e = _enriched.get(key)
        if e and now - e[0] < ENRICH_TTL:
            return e[1]
        b = _base.get(key)
        if not b or now - b[0] > BASE_TTL:
            try:
                preds, hist = _retry(lambda: model.predict_with_history(
                    [season - 2, season - 1, season], season, week))
                b = _base[key] = (now, preds, hist)
            except Exception:
                if not b:          # nothing older to fall back on
                    raise
        df = sim.enrich(b[1], b[2], season, week)
        _enriched[key] = (now, df)
        return df


COLS = ["player_id", "player_display_name", "position", "team", "opponent_team",
        "final_prob", "floor", "median", "ceiling", "injury_status", "injury",
        "fantasy_points_ppr_L3", "props"]


def to_records(df):
    return json.loads(df[COLS].round(3).to_json(orient="records"))


@app.on_event("startup")
def warm():
    """Pre-build this week's picks, but only after the server has been up for
    a while: small hosts need the port open right away and the warm-up is heavy."""
    s, w = current_week()
    t = threading.Timer(45, lambda: predictions(s, w))
    t.daemon = True
    t.start()


@app.get("/api/defaults")
def defaults():
    s, w = current_week()
    return {"season": s, "week": w, "ai": insight.enabled(),
            "yahoo": bool(os.getenv("YAHOO_CLIENT_ID"))}


@app.get("/api/top")
def top(season: int, week: int, limit: int = 30):
    try:
        df = predictions(season, week)
        return to_records(df[df["injury_status"] != "Out"].head(limit))
    except Exception as e:
        raise HTTPException(502, f"Could not build predictions: {e}")


# ---------------------------------------------------------------- leagues --
class LeagueReq(BaseModel):
    platform: str
    season: int
    week: int
    username: Optional[str] = None
    url: Optional[str] = None
    espn_s2: Optional[str] = None
    swid: Optional[str] = None


def split(preds, league):
    """Your players, plus the best free agents in this league."""
    mine = preds[preds["player_display_name"].map(sleeper.norm).isin(league["mine"])]
    names = preds["player_display_name"].map(sleeper.norm)
    if league.get("available") is not None:
        free = preds[names.isin(league["available"])]
    else:
        free = preds[~names.isin(league["rostered"] | league["mine"])]
    free = free[free["injury_status"] != "Out"].head(8)
    return {"league": league["league"], "players": to_records(mine),
            "free_agents": to_records(free)}


def sleeper_leagues(req):
    try:
        uid = sleeper.get_user_id(req.username)
    except (requests.RequestException, TypeError, KeyError):
        raise HTTPException(404, "Sleeper doesn't recognize that username. Use the "
                                 "username from Sleeper's settings, not your display name.")
    try:
        leagues = sleeper.get_leagues(uid, req.season)
        players = sleeper.get_players()
    except requests.RequestException:
        raise HTTPException(502, "Couldn't reach Sleeper. Try again in a minute.")
    if not leagues:
        raise HTTPException(404, f"Sleeper found your account, but it has no NFL leagues in "
                                 f"the {req.season} season. Check the Season box.")
    out, problems = [], []
    for lg in leagues:
        try:
            r = sleeper.get_my_roster(lg["league_id"], uid)
            mine = {sleeper.norm(n) for _, n, _ in sleeper.roster_names(r, players)}
            rostered = sleeper.all_rostered(lg["league_id"], players)
        except Exception as e:
            problems.append(f"{lg.get('name', 'a league')}: {e}")
            continue
        out.append({"league": lg["name"], "mine": mine, "rostered": rostered})
    if not out:
        raise HTTPException(502, "Found your leagues but couldn't read them ("
                                 + "; ".join(problems[:3]) + ").")
    return out


def yahoo_leagues(request):
    s = _sessions.get(request.cookies.get("bg_sid"))
    if not s:
        raise HTTPException(401, "Connect your Yahoo account first.")
    try:
        if time.time() > s["expires"]:
            t = providers.yahoo_token(os.environ["YAHOO_CLIENT_ID"],
                                      os.environ["YAHOO_CLIENT_SECRET"],
                                      redirect_uri=redirect_uri(),
                                      refresh=s["refresh"])
            s.update(token=t["access_token"], expires=time.time() + t["expires_in"] - 60)
        return [providers.yahoo_league(s["token"], *lg)
                for lg in providers.yahoo_leagues(s["token"])]
    except requests.RequestException:
        raise HTTPException(502, "Yahoo didn't respond. Reconnect and try again.")


@app.post("/api/leagues")
def leagues(req: LeagueReq, request: Request):
    try:
        if req.platform == "sleeper":
            if not req.username:
                raise HTTPException(400, "Enter your Sleeper username.")
            found = sleeper_leagues(req)
        elif req.platform == "espn":
            found = [providers.espn_league(req.url, req.season, req.espn_s2, req.swid)]
        elif req.platform == "yahoo":
            found = yahoo_leagues(request)
        else:
            raise HTTPException(400, "Unknown platform.")
    except (ValueError, PermissionError) as e:
        raise HTTPException(400, str(e))
    except requests.RequestException:
        raise HTTPException(502, "Couldn't reach that platform. Try again shortly.")
    preds = predictions(req.season, req.week)
    return [split(preds, lg) for lg in found]


# ------------------------------------------------------------------ Yahoo --
def redirect_uri():
    return os.getenv("BASE_URL", "http://localhost:8000").rstrip("/") + "/auth/yahoo/callback"


@app.get("/auth/yahoo/login")
def yahoo_login():
    if not os.getenv("YAHOO_CLIENT_ID"):
        raise HTTPException(503, "Yahoo isn't set up on this site yet.")
    return RedirectResponse(providers.yahoo_auth_url(
        os.environ["YAHOO_CLIENT_ID"], redirect_uri(), secrets.token_urlsafe(8)))


@app.get("/auth/yahoo/callback")
def yahoo_callback(code: str = "", error: str = ""):
    if error or not code:
        return RedirectResponse("/?yahoo=failed")
    t = providers.yahoo_token(os.environ["YAHOO_CLIENT_ID"],
                              os.environ["YAHOO_CLIENT_SECRET"],
                              code=code, redirect_uri=redirect_uri())
    sid = secrets.token_urlsafe(24)
    _sessions[sid] = {"token": t["access_token"], "refresh": t["refresh_token"],
                      "expires": time.time() + t["expires_in"] - 60}
    resp = RedirectResponse("/?yahoo=connected")
    resp.set_cookie("bg_sid", sid, httponly=True, samesite="lax",
                    secure=redirect_uri().startswith("https"), max_age=86400)
    return resp


# --------------------------------------------------------------------- AI --
@app.get("/api/insight")
def get_insight(name: str, position: str, team: str, opponent: str,
                season: int, week: int, status: str = ""):
    if not insight.enabled():
        raise HTTPException(503, "AI notes aren't turned on for this site.")
    try:
        return {"text": insight.insight(name, position, team, opponent,
                                        season, week, status or None)}
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    except requests.RequestException:
        raise HTTPException(502, "Couldn't fetch news right now.")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
