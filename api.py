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
import matchup
import model
import providers
import record
import ros
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


_ros, _ros_lock = {}, threading.Lock()


def ros_table(season, week):
    """Rest-of-season numbers for every active QB/RB/WR/TE (cached 30 min)."""
    key = (season, week)
    with _ros_lock:
        e = _ros.get(key)
        if e and time.time() - e[0] < ENRICH_TTL:
            return e[1]
        predictions(season, week)                 # makes sure the stats are loaded
        table = ros.build(_base[key][2], season, week, sim.injury_report(season, week))
        _ros[key] = (time.time(), table)
        return table


COLS = ["player_id", "player_display_name", "position", "team", "opponent_team",
        "final_prob", "floor", "median", "ceiling", "injury_status", "injury",
        "fantasy_points_ppr_L3", "props", "def_rank", "def_allowed"]


def to_records(df):
    return json.loads(df[COLS].round(3).to_json(orient="records"))


@app.on_event("startup")
def warm():
    """Pre-build this week's picks, but only after the server has been up for
    a while: small hosts need the port open right away and the warm-up is heavy."""
    s, w = current_week()
    t = threading.Timer(45, lambda: ros_table(s, w))     # also builds the picks
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


def split(preds, league, table=None):
    """Your players, plus the best free agents in this league."""
    mine = preds[preds["player_display_name"].map(sleeper.norm).isin(league["mine"])]
    names = preds["player_display_name"].map(sleeper.norm)
    if league.get("available") is not None:
        free = preds[names.isin(league["available"])]
    else:
        free = preds[~names.isin(league["rostered"] | league["mine"])]
    free = free[free["injury_status"] != "Out"].head(8)
    out = {"league": league["league"], "players": to_records(mine),
           "free_agents": to_records(free), "roster": [], "slots": league.get("slots"),
           "replacement": None, "league_id": league.get("league_id")}
    if table is not None and len(table):        # what the trade analyzer needs
        tn = table["name"].map(sleeper.norm)
        mine_t = table[tn.isin(league["mine"])].sort_values("ros_pts", ascending=False)
        if league.get("available") is not None:
            fa = tn.isin(league["available"])
        else:
            fa = ~tn.isin(league["rostered"] | league["mine"])
        out["roster"] = [{"id": r.player_id, "name": r.name, "position": r.position, "team": r.team}
                         for r in mine_t.itertuples()]
        out["replacement"] = {k: round(v, 1) for k, v in
                              ros.replacement_ppg(table, set(table.index[fa])).items()}
    return out


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
        out.append({"league": lg["name"], "mine": mine, "rostered": rostered,
                    "slots": sleeper.slots_from(lg.get("roster_positions")),
                    "league_id": lg["league_id"]})
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
    try:
        table = ros_table(req.season, req.week)
    except Exception:
        table = None                     # picks still work without the trade data
    return [split(preds, lg, table) for lg in found]


# ------------------------------------------------------------------ trades --
@app.get("/api/players")
def player_search(season: int, week: int, q: str = "", limit: int = 8):
    key = sleeper.norm(q)
    if len(key) < 2:
        return []
    try:
        table = ros_table(season, week)
    except Exception as e:
        raise HTTPException(502, f"Couldn't load player data: {e}")
    hits = table[table["name"].map(sleeper.norm).str.contains(key, regex=False)]
    hits = hits.sort_values("ros_pts", ascending=False).head(min(limit, 15))
    return [{"id": r.player_id, "name": r.name, "position": r.position, "team": r.team}
            for r in hits.itertuples()]


class TradeReq(BaseModel):
    season: int
    week: int
    give: list[str]
    get: list[str]
    roster: Optional[list[str]] = None
    slots: Optional[dict] = None
    replacement: Optional[dict] = None


@app.post("/api/trade")
def trade(req: TradeReq):
    if not req.give or not req.get:
        raise HTTPException(400, "Add at least one player to each side.")
    if len(req.give) > 5 or len(req.get) > 5:
        raise HTTPException(400, "Trades are limited to 5 players per side.")
    slots = None
    if req.slots:
        slots = {k: max(0, min(int(v), 8)) for k, v in req.slots.items()
                 if k in ({"QB", "RB", "WR", "TE"} | set(ros.FLEX_ELIGIBLE)) and isinstance(v, (int, float))}
    repl = None
    if req.replacement:
        repl = {k: max(0.0, min(float(v), 40.0)) for k, v in req.replacement.items()
                if k in ros.POS and isinstance(v, (int, float))}
    try:
        table = ros_table(req.season, req.week)
        return ros.analyze(table, req.give, req.get, (req.roster or [])[:40], slots, repl)
    except KeyError as e:
        raise HTTPException(400, str(e.args[0]))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Couldn't analyze that trade: {e}")


# ------------------------------------------------------- game plan + record --
class MatchupReq(BaseModel):
    username: str
    league_id: str
    season: int
    week: int


@app.post("/api/matchup")
def matchup_view(req: MatchupReq):
    try:
        uid = sleeper.get_user_id(req.username)
        lg = sleeper.get_league(req.league_id)
        rosters = sleeper.get_rosters(req.league_id)
        matchups = sleeper.get_matchups(req.league_id, req.week)
        users = sleeper.get_league_users(req.league_id)
        sl_players = sleeper.get_players()
    except (requests.RequestException, TypeError, KeyError):
        raise HTTPException(502, "Couldn't reach Sleeper for that league. Try again shortly.")
    me = next((r for r in rosters if r.get("owner_id") == uid or uid in (r.get("co_owners") or [])), None)
    if not me:
        raise HTTPException(404, "Couldn't find your team in that league.")
    mine = next((m for m in matchups or [] if m.get("roster_id") == me["roster_id"]), None)
    if not mine or mine.get("matchup_id") is None:
        raise HTTPException(404, f"No game for your team in week {req.week}. It may be a bye or playoff week in your league.")
    other = next((m for m in matchups if m.get("matchup_id") == mine["matchup_id"]
                  and m["roster_id"] != me["roster_id"]), None)
    if not other:
        raise HTTPException(404, "Couldn't find your opponent for that week.")

    names = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u.get("display_name")
             for u in users or []}
    owner = {r["roster_id"]: names.get(r.get("owner_id")) for r in rosters}
    ss = lg.get("scoring_settings") or {}
    try:
        preds = predictions(req.season, req.week)
        table = ros_table(req.season, req.week)
        hist = _base[(req.season, req.week)][2]
    except Exception as e:
        raise HTTPException(502, f"Couldn't build this week's projections: {e}")
    sampler = matchup.Sampler(hist, preds, table, req.week, rec=float(ss.get("rec", 1.0)),
                              te_bonus=float(ss.get("bonus_rec_te", 0.0)))
    labels = [p for p in (lg.get("roster_positions") or []) if p != "BN"]

    def lineup(entry):
        ids = entry.get("starters") or []
        slots = labels if len(labels) == len(ids) else [None] * len(ids)
        out = []
        for slot, sid in zip(slots, ids):
            p = matchup.resolve(sid, sl_players, table, sampler)
            out.append((slot or p["position"] or "FLEX", p))
        return out

    mine_slots, opp_slots = lineup(mine), lineup(other)
    starters = set(mine.get("starters") or [])
    off = set(me.get("reserve") or []) | set(me.get("taxi") or [])
    bench = [matchup.resolve(pid, sl_players, table, sampler) for pid in mine.get("players") or []
             if pid not in starters and pid not in off]
    result = matchup.analyze(mine_slots, bench, opp_slots, owner.get(me["roster_id"]) or "You",
                             owner.get(other["roster_id"]) or "Your opponent", req.week)
    result["league"] = lg.get("name")
    result["rec"] = float(ss.get("rec", 1.0))
    return result


@app.get("/api/record")
def track_record(season: int):
    try:
        return record.report(season)
    except Exception as e:
        raise HTTPException(502, f"Couldn't grade this season yet: {e}")


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


# Phones and browsers also look for the icon at these standard addresses.
@app.get("/apple-touch-icon.png", include_in_schema=False)
def touch_icon():
    return FileResponse(STATIC / "icon-180.png", media_type="image/png")


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def touch_icon_precomposed():
    return FileResponse(STATIC / "icon-180.png", media_type="image/png")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "icon-rounded-64.png", media_type="image/png")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
