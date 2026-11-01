"""Web API + static front end.   Run:  uvicorn api:app --reload"""
import json
import threading
import time
from pathlib import Path

import nflreadpy as nfl
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import model
import sleeper

app = FastAPI(title="Big Game")
STATIC = Path(__file__).parent / "static"
TTL = 6 * 3600                      # recompute predictions every 6 hours
_cache, _lock = {}, threading.Lock()


def current_week():
    try:
        return int(nfl.get_current_season()), int(nfl.get_current_week())
    except Exception:
        return 2025, 9


def predictions(season, week):
    key = (season, week)
    with _lock:                     # one training run at a time
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < TTL:
            return hit[1]
        df = model.predict_week([season - 2, season - 1, season], season, week)
        _cache[key] = (time.time(), df)
        return df


def to_records(df):
    cols = ["player_id", "player_display_name", "position", "team",
            "opponent_team", "big_game_prob", "fantasy_points_ppr_L3",
            "opp_pts_allowed_L6"]
    return json.loads(df[cols].round(3).to_json(orient="records"))


@app.on_event("startup")
def warm():
    s, w = current_week()
    threading.Thread(target=lambda: predictions(s, w), daemon=True).start()


@app.get("/api/defaults")
def defaults():
    s, w = current_week()
    return {"season": s, "week": w}


@app.get("/api/top")
def top(season: int, week: int, limit: int = 30):
    try:
        return to_records(predictions(season, week).head(limit))
    except Exception as e:
        raise HTTPException(502, f"Could not build predictions: {e}")


@app.get("/api/roster")
def roster(username: str, season: int, week: int):
    try:
        uid = sleeper.get_user_id(username)
        leagues = sleeper.get_leagues(uid, season)
        players = sleeper.get_players()
    except (requests.RequestException, TypeError, KeyError):
        raise HTTPException(404, "Sleeper user not found, or Sleeper is unreachable.")
    preds = predictions(season, week)
    out = []
    for lg in leagues:
        try:
            r = sleeper.get_my_roster(lg["league_id"], uid)
        except Exception:
            continue
        mine = sleeper.roster_names(r, players)
        ids = {m[0] for m in mine if m[0]}
        names = {sleeper.norm(m[1]) for m in mine}
        sub = preds[preds["player_id"].isin(ids) |
                    preds["player_display_name"].map(sleeper.norm).isin(names)]
        out.append({"league": lg["name"], "players": to_records(sub)})
    return out


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
