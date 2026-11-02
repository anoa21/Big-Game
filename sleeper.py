"""Minimal Sleeper client (public, read-only, no API key needed)."""
import json, re, time
from pathlib import Path
import requests

BASE = "https://api.sleeper.app/v1"
CACHE = Path(".sleeper_players.json")


def _get(path):
    r = requests.get(f"{BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def get_user_id(username):
    return _get(f"/user/{username}")["user_id"]


def get_leagues(user_id, season):
    return _get(f"/user/{user_id}/leagues/nfl/{season}")


def get_my_roster(league_id, user_id):
    for roster in _get(f"/league/{league_id}/rosters"):
        if roster.get("owner_id") == user_id or user_id in (roster.get("co_owners") or []):
            return roster
    raise ValueError("no team in this league belongs to that username")


def get_players():
    """Sleeper's full player map is ~5MB; Sleeper asks you to fetch it at most
    once a day, so cache it on disk."""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 86400:
        return json.loads(CACHE.read_text())
    data = _get("/players/nfl")
    CACHE.write_text(json.dumps(data))
    return data


_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(name):
    """Normalize a player name so 'Kenneth Walker III' == 'kenneth walker'."""
    words = re.sub(r"[^a-z ]", "", (name or "").lower().replace(".", "")).split()
    return "".join(w for w in words if w not in _SUFFIX)


def roster_names(roster, players):
    """Return [(gsis_id or None, full_name, position)] for a roster."""
    out = []
    for pid in roster.get("players") or []:
        p = players.get(pid, {})
        out.append((p.get("gsis_id"), p.get("full_name") or pid, p.get("position")))
    return out


def all_rostered(league_id, players):
    """Normalized names of every player on any team in the league."""
    names = set()
    for roster in _get(f"/league/{league_id}/rosters"):
        for pid in roster.get("players") or []:
            n = players.get(pid, {}).get("full_name")
            if n:
                names.add(norm(n))
    return names
