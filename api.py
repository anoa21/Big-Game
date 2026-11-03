"""Web API + static front end.   Run:  uvicorn api:app --reload

Optional environment variables:
  ANTHROPIC_API_KEY   turns on AI news notes
  YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, BASE_URL   turn on Yahoo import
"""
import datetime as dt
import json
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import nflreadpy as nfl
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import accounts
import context
import insight
import leagueview
import matchup
import model
import nextup
import playbook
import providers
import record
import ros
import sim
import sleeper

app = FastAPI(title="Big Game")
app.include_router(accounts.router)
STATIC = Path(__file__).parent / "static"
BASE_TTL, ENRICH_TTL = 6 * 3600, 30 * 60   # model every 6h, injuries every 30min
_base, _enriched, _lock = {}, {}, threading.Lock()
_nu = {}                                   # next-man-up results, per (season, week)
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
        try:
            nu = nextup.compute(b[2], season, week, sim.injury_report(season, week))
        except Exception:
            nu = {"teams": [], "by_player": {}, "evidence": nextup.EVIDENCE}
        _nu[key] = nu
        df["nu_bump"] = df["player_id"].map(lambda i: nu["by_player"].get(i, {}).get("bump"))
        df["nu_why"] = df["player_id"].map(lambda i: nu["by_player"].get(i, {}).get("why"))
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
        "fantasy_points_ppr_L3", "props", "def_rank", "def_allowed", "nu_bump", "nu_why", "ctx_pts", "ctx_line", "ctx_notes"]


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
    players: Optional[list[str]] = None


def split(preds, league, table=None):
    """Your players, plus the best free agents in this league."""
    mine = preds[preds["player_display_name"].map(sleeper.norm).isin(league["mine"])]
    names = preds["player_display_name"].map(sleeper.norm)
    if league.get("manual"):
        free = preds.iloc[0:0]                          # we can't see the rest of the league
    elif league.get("available") is not None:
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
        if league.get("manual"):
            fa = tn.isin(set())                         # nobody counts as a free agent
        elif league.get("available") is not None:
            fa = tn.isin(league["available"])
        else:
            fa = ~tn.isin(league["rostered"] | league["mine"])
        out["fa_ids"] = [i for i in table.index[fa]]
        out["teams"] = [{"name": t["name"], "mine": t["mine"],
                         "roster": [i for i in table.index[tn.isin(t["names"])]]}
                        for t in (league.get("teams") or [])]
        out["roster"] = [{"id": r.player_id, "name": r.name, "position": r.position, "team": r.team}
                         for r in mine_t.itertuples()]
        out["replacement"] = {k: round(v, 1) for k, v in
                              ros.replacement_ppg(table, set(table.index[fa])).items()}
    if league.get("manual"):
        known = set(table["name"].map(sleeper.norm)) if table is not None else set()
        out["manual"] = True
        out["unmatched"] = [t for t in league.get("typed", []) if sleeper.norm(t) not in known]
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
            rosters = sleeper.get_rosters(lg["league_id"])
            try:
                owners = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u.get("display_name")
                          for u in sleeper.get_league_users(lg["league_id"])}
            except Exception:
                owners = {}                  # team names are a nicety; carry on without them
            teams, mine, rostered = [], set(), set()
            for r in rosters:
                names = {sleeper.norm(players.get(pid, {}).get("full_name") or "") for pid in r.get("players") or []}
                names.discard("")
                is_me = r.get("owner_id") == uid or uid in (r.get("co_owners") or [])
                teams.append({"name": owners.get(r.get("owner_id")) or f"Team {r.get('roster_id')}",
                              "mine": is_me, "names": names})
                rostered |= names
                if is_me:
                    mine = names
            if not mine:
                raise ValueError("no team in this league belongs to that username")
        except Exception as e:
            problems.append(f"{lg.get('name', 'a league')}: {e}")
            continue
        out.append({"league": lg["name"], "mine": mine, "rostered": rostered, "teams": teams,
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
        elif req.platform == "manual":
            found = [providers.manual_league((req.players or [])[:60])]
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
    partner: Optional[list[str]] = None
    partner_name: Optional[str] = None


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
        return ros.analyze(table, req.give, req.get, (req.roster or [])[:40], slots, repl,
                           (req.partner or [])[:40], (req.partner_name or "")[:60] or "Their team")
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
    now = dt.datetime.now(dt.timezone.utc)
    cur_season, cur_week = current_week()
    if (req.season, req.week) < (cur_season, cur_week):
        now = None                                 # a past week: nothing is locked, it's just a what-if
    try:
        kicks = {t: c.get("kickoff") for t, c in context.game_context(req.season, req.week, weather=False).items()}
    except Exception:
        kicks = {}
    sampler = matchup.Sampler(hist, preds, table, req.week, rec=float(ss.get("rec", 1.0)),
                              te_bonus=float(ss.get("bonus_rec_te", 0.0)), kickoffs=kicks, now=now)
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
    rostered_ids, rostered_names = set(), set()
    for r in rosters:
        for pid in r.get("players") or []:
            p = sl_players.get(str(pid)) or {}
            if p.get("gsis_id"):
                rostered_ids.add(p["gsis_id"])
            if p.get("full_name"):
                rostered_names.add(sleeper.norm(p["full_name"]))
    fas = matchup.free_agent_pool(sampler, preds, rostered_ids, rostered_names)
    held = [p for p in (me.get("players") or []) if p not in off]
    roster_full = len(held) >= len([x for x in (lg.get("roster_positions") or [])])
    result = matchup.analyze(mine_slots, bench, opp_slots, owner.get(me["roster_id"]) or "You",
                             owner.get(other["roster_id"]) or "Your opponent", req.week,
                             free_agents=fas, roster_full=roster_full)
    result["league"] = lg.get("name")
    result["rec"] = float(ss.get("rec", 1.0))
    return result


_wx = {}                                   # weather view cache, per (season, week)


@app.get("/api/weather")
def weather(season: int, week: int):
    key = (season, week)
    hit = _wx.get(key)
    if hit and time.time() - hit[0] < 30 * 60:
        return hit[1]
    try:
        games = context.week_conditions(season, week)
    except Exception as e:
        raise HTTPException(502, f"Couldn't load the stadium and weather details: {e}")
    _wx[key] = (time.time(), games)
    return games


# ------------------------------------------------------- coach's playbook --
_asks = {}                                 # visitor -> recent AI request times (cost protection)


def _visitor(request):
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def _film(season, week, player_id):
    """The dossier and plain breakdown, always rebuilt on the server from the app's own data."""
    preds = predictions(season, week)
    table = ros_table(season, week)
    key = (season, week)
    try:
        ctx = context.game_context(season, week)
    except Exception:
        ctx = {}
    d = playbook.build_dossier(player_id, season, week, preds, _base[key][2], table, _nu.get(key),
                               ctx, playbook.injury_frame(season, week))
    sections, conf = playbook.breakdown(d)
    return d, sections, conf


@app.get("/api/playbook")
def playbook_view(season: int, week: int, player_id: str):
    try:
        d, sections, conf = _film(season, week, player_id)
    except Exception as e:
        raise HTTPException(502, f"Couldn't pull up the film: {e}")
    if not d.get("name"):
        raise HTTPException(404, "Couldn't find that player.")
    brief = {k: d.get(k) for k in ("player_id", "name", "position", "team", "opponent", "on_slate", "season", "week")}
    if d.get("on_slate"):
        brief["big_game"] = d["odds"]["big_game"]
        brief["floor"], brief["median"], brief["ceiling"] = d["odds"]["floor"], d["odds"]["median"], d["odds"]["ceiling"]
    return {"player": brief, "sections": sections, "confidence": conf, "glance": playbook.glance(d), "ai": playbook.ai_enabled()}


class AskReq(BaseModel):
    season: int
    week: int
    player_id: str
    question: Optional[str] = None


@app.post("/api/playbook/ask")
def playbook_ask(req: AskReq, request: Request):
    if not playbook.ai_enabled():
        raise HTTPException(503, "The AI coach isn't turned on for this site.")
    who, now = _visitor(request), time.time()
    recent = [t for t in _asks.get(who, []) if now - t < 600]
    if len(recent) >= 6:
        raise HTTPException(429, "That's a lot of questions at once. Give the coach a few minutes.")
    _asks[who] = recent + [now]
    try:
        d, sections, _ = _film(req.season, req.week, req.player_id)
    except Exception as e:
        raise HTTPException(502, f"Couldn't pull up the film: {e}")
    if not d.get("on_slate"):
        raise HTTPException(400, d.get("reason") or "He isn't on this week's slate.")
    try:
        return playbook.ask(d, sections, req.question or "")
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    except (ValueError, KeyError):
        raise HTTPException(502, "The coach had trouble putting that into words. Try again.")
    except requests.RequestException:
        raise HTTPException(502, "Couldn't reach the AI coach right now.")


@app.get("/api/nextup")
def next_man_up(season: int, week: int):
    try:
        predictions(season, week)                  # builds and caches the result
        return _nu[(season, week)]
    except Exception as e:
        raise HTTPException(502, f"Couldn't work out who gains this week: {e}")


# ------------------------------------------------- league outlook (strength + playoffs) --
class OutlookReq(BaseModel):
    username: str
    league_id: str
    season: int
    week: int


def _table_ids(table):
    idx = {}
    for pid, name, pos in zip(table.index, table["name"], table["position"]):
        idx[(sleeper.norm(name), pos)] = pid
    return idx


def _roster_ids(roster, sl_players, table, idx):
    out = []
    for sid in roster.get("players") or []:
        p = sl_players.get(str(sid)) or {}
        gsis = p.get("gsis_id")
        pid = gsis if gsis in table.index else idx.get((sleeper.norm(p.get("full_name") or ""), p.get("position")))
        if pid is not None and pid not in out:
            out.append(pid)
    return out


@app.post("/api/league/outlook")
def league_outlook(req: OutlookReq):
    try:
        uid = sleeper.get_user_id(req.username)
        lg = sleeper.get_league(req.league_id)
        rosters = sleeper.get_rosters(req.league_id)
        users = sleeper.get_league_users(req.league_id)
        sl_players = sleeper.get_players()
    except (requests.RequestException, TypeError, KeyError):
        raise HTTPException(502, "Couldn't reach Sleeper for that league. Try again shortly.")
    me = next((r for r in rosters if r.get("owner_id") == uid or uid in (r.get("co_owners") or [])), None)
    if not me:
        raise HTTPException(404, "Couldn't find your team in that league.")
    try:
        table = ros_table(req.season, req.week)
    except Exception as e:
        raise HTTPException(502, f"Couldn't build the projections: {e}")
    st = lg.get("settings") or {}
    playoff_teams = int(st.get("playoff_teams") or 6)
    reg_weeks = int(st.get("playoff_week_start") or 15) - 1
    names = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u.get("display_name") for u in users or []}
    positions = lg.get("roster_positions") or []
    slots = {**(sleeper.slots_from(positions) or ros.DEFAULT_SLOTS), "K": positions.count("K"), "DEF": positions.count("DEF")}
    idx = _table_ids(table)
    ids = {r["roster_id"]: _roster_ids(r, sl_players, table, idx) for r in rosters}
    taken = {i for v in ids.values() for i in v}
    repl = ros.replacement_ppg(table, set(table.index) - taken)
    # standings
    def rec(r):
        s = r.get("settings") or {}
        return {"wins": int(s.get("wins") or 0), "losses": int(s.get("losses") or 0), "ties": int(s.get("ties") or 0),
                "pf": float(s.get("fpts") or 0) + float(s.get("fpts_decimal") or 0) / 100}
    standings = {r["roster_id"]: rec(r) for r in rosters}
    gp = max(s["wins"] + s["losses"] + s["ties"] for s in standings.values())
    remaining = list(range(gp + 1, reg_weeks + 1))
    nums = {rid: leagueview.team_numbers(table, ids[rid], slots, repl, remaining, req.week) for rid in ids}
    # line up the scoring scale with how this league has really been scoring
    scale, aw = 1.0, 0.0
    if gp > 0:
        actual = {rid: standings[rid]["pf"] / max(sum(standings[rid][k] for k in ("wins", "losses", "ties")), 1) for rid in ids}
        proj_avg = sum(n["ppg"] for n in nums.values()) / len(nums)
        scale = min(max((sum(actual.values()) / len(actual)) / proj_avg, 0.7), 1.3) if proj_avg else 1.0
        aw = min(0.5, gp / 20)
    order = list(ids)
    pos_of = {rid: i for i, rid in enumerate(order)}
    teams_sim = []
    for rid in order:
        n_, s_ = nums[rid], standings[rid]
        act = s_["pf"] / max(s_["wins"] + s_["losses"] + s_["ties"], 1) if gp > 0 else None
        mean = {w: ((1 - aw) * n_["by_week"][w] * scale + aw * act) if act is not None else n_["by_week"][w] * scale for w in remaining}
        teams_sim.append({"wins": s_["wins"], "ties": s_["ties"], "pf": s_["pf"], "mean_by_week": mean, "sd": n_["sd"] * scale})
    def pairs(week):
        try:
            ms = sleeper.get_matchups(req.league_id, week) or []
        except Exception:
            return week, []
        groups = {}
        for m in ms:
            if m.get("matchup_id") is not None:
                groups.setdefault(m["matchup_id"], []).append(m["roster_id"])
        return week, [(pos_of[g[0]], pos_of[g[1]]) for g in groups.values() if len(g) == 2 and g[0] in pos_of and g[1] in pos_of]
    with ThreadPoolExecutor(max_workers=6) as ex:
        schedule = {w: p for w, p in ex.map(pairs, remaining) if p}
    my_i = pos_of[me["roster_id"]]
    sim_out = leagueview.simulate(teams_sim, schedule, playoff_teams, my_i)
    # strength ranks
    cards = [{"name": names.get(r.get("owner_id")) or f"Team {r['roster_id']}", "mine": r["roster_id"] == me["roster_id"],
              "numbers": nums[r["roster_id"]]} for r in rosters]
    avg, n_teams = leagueview.strength(cards)
    by_rid = {r["roster_id"]: c for r, c in zip(rosters, cards)}
    out = []
    for i, rid in enumerate(order):
        c, s_ = by_rid[rid], standings[rid]
        out.append({"name": c["name"], "mine": c["mine"], "wins": s_["wins"], "losses": s_["losses"], "ties": s_["ties"],
                    "pf": round(s_["pf"], 1), "playoff": round(float(sim_out["playoff"][i]), 3),
                    "exp_wins": round(float(sim_out["wins"][i]), 1), "seed_median": int(sim_out["seed_median"][i]),
                    "ppg_projected": round(c["numbers"]["ppg"] * scale, 1),
                    "ppg_actual": None if gp == 0 else round(s_["pf"] / max(s_["wins"] + s_["losses"] + s_["ties"], 1), 1),
                    "group": c["numbers"]["group"], "depth": c["numbers"]["depth"], "rank": c["rank"], "top": c["numbers"]["top"],
                    "empty": c["numbers"]["empty"]})
    out.sort(key=lambda t: (-t["playoff"], -t["exp_wins"]))
    swing = [{**s, "opponent": next(c["name"] for c in cards if by_rid[order[s["opponent_index"]]] is c)} for s in sim_out["swing"][:3]]
    for s in swing:
        s.pop("opponent_index")
    mine_card = next(c for c in cards if c["mine"])
    return {"league": lg.get("name"), "season": req.season, "week": req.week, "playoff_teams": playoff_teams,
            "games_played": gp, "regular_weeks": reg_weeks, "remaining": len(remaining), "teams": out,
            "averages": {k: round(v, 1) for k, v in avg.items()}, "verdict": leagueview.verdict(mine_card, avg, n_teams),
            "swing": swing, "scale": round(scale, 2), "slots": slots}


class StrengthTeam(BaseModel):
    name: str
    mine: bool = False
    roster: list[str]


class StrengthReq(BaseModel):
    season: int
    week: int
    teams: list[StrengthTeam]
    slots: Optional[dict] = None
    replacement: Optional[dict] = None


@app.post("/api/league/strength")
def league_strength(req: StrengthReq):
    """Roster strength only, for leagues where we can see every roster but not the standings (ESPN)."""
    if not 2 <= len(req.teams) <= 20:
        raise HTTPException(400, "That doesn't look like a full league.")
    try:
        table = ros_table(req.season, req.week)
    except Exception as e:
        raise HTTPException(502, f"Couldn't build the projections: {e}")
    slots = {**ros.DEFAULT_SLOTS, **{k: max(0, min(int(v), 8)) for k, v in (req.slots or {}).items() if isinstance(v, (int, float))}}
    taken = {i for t in req.teams for i in t.roster[:40]}
    repl = {**ros.replacement_ppg(table, set(table.index) - taken),
            **{k: float(v) for k, v in (req.replacement or {}).items() if k in ros.POS and isinstance(v, (int, float))}}
    cards = [{"name": t.name[:60], "mine": t.mine, "numbers": leagueview.team_numbers(table, t.roster[:40], slots, repl, [], req.week)} for t in req.teams]
    avg, n_teams = leagueview.strength(cards)
    mine = next((c for c in cards if c["mine"]), None)
    out = [{"name": c["name"], "mine": c["mine"], "ppg_projected": c["numbers"]["ppg"], "group": c["numbers"]["group"],
            "depth": c["numbers"]["depth"], "rank": c["rank"], "top": c["numbers"]["top"], "empty": c["numbers"]["empty"]} for c in cards]
    out.sort(key=lambda t: t["rank"]["overall"])
    return {"teams": out, "averages": {k: round(v, 1) for k, v in avg.items()},
            "verdict": leagueview.verdict(mine, avg, n_teams) if mine else []}


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
