"""League import for ESPN and Yahoo. Each function returns a dict:
  {"league": name, "mine": {normalized names}, "rostered": {names} or None,
   "available": {names} or None}
`rostered` = every player on any team (free agents = everyone else).
`available` = players the platform itself lists as free agents.
Neither ESPN nor Yahoo could be tested from the build sandbox, so expect to
adjust field names if a platform changes its responses.
"""
import re
import requests
from sleeper import norm

# ---------------------------------------------------------------- ESPN ----
ESPN = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        "seasons/{season}/segments/0/leagues/{lid}")


def parse_espn_url(url):
    lid = re.search(r"leagueId=(\d+)", url or "")
    tid = re.search(r"teamId=(\d+)", url or "")
    if not lid:
        raise ValueError("Paste the link from your ESPN league page "
                         "(it contains leagueId=...).")
    return lid.group(1), int(tid.group(1)) if tid else None


def espn_league(url, season, espn_s2=None, swid=None):
    lid, team_id = parse_espn_url(url)
    cookies = {}
    if espn_s2 and swid:
        cookies = {"espn_s2": espn_s2.strip(), "SWID": swid.strip()}
    r = requests.get(ESPN.format(season=season, lid=lid),
                     params=[("view", "mRoster"), ("view", "mTeam"),
                             ("view", "mSettings")],
                     cookies=cookies, timeout=30)
    if r.status_code in (401, 403):
        raise PermissionError("This ESPN league is private. Add your espn_s2 "
                              "and SWID cookies to connect it.")
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError:
        raise PermissionError("ESPN did not return league data. If the league "
                              "is private, add your cookies.")
    if team_id is None and swid:
        for t in data.get("teams", []):
            if swid.strip().lower() in [o.lower() for o in t.get("owners", [])]:
                team_id = t["id"]
    if team_id is None:
        raise ValueError("Your link needs teamId=... (open your team page "
                         "on ESPN and copy that link).")
    mine, rostered = set(), set()
    for t in data.get("teams", []):
        for e in (t.get("roster") or {}).get("entries", []):
            name = e["playerPoolEntry"]["player"]["fullName"]
            rostered.add(norm(name))
            if t["id"] == team_id:
                mine.add(norm(name))
    return {"league": (data.get("settings") or {}).get("name", "ESPN league"),
            "mine": mine, "rostered": rostered, "available": None}


# --------------------------------------------------------------- Yahoo ----
Y_AUTH = "https://api.login.yahoo.com/oauth2/request_auth"
Y_TOKEN = "https://api.login.yahoo.com/oauth2/get_token"
Y_API = "https://fantasysports.yahooapis.com/fantasy/v2"


def yahoo_auth_url(client_id, redirect_uri, state):
    q = requests.Request("GET", Y_AUTH, params={
        "client_id": client_id, "redirect_uri": redirect_uri,
        "response_type": "code", "scope": "fspt-r", "state": state}).prepare()
    return q.url


def yahoo_token(client_id, secret, code=None, redirect_uri=None, refresh=None):
    data = ({"grant_type": "refresh_token", "refresh_token": refresh,
             "redirect_uri": redirect_uri} if refresh else
            {"grant_type": "authorization_code", "code": code,
             "redirect_uri": redirect_uri})
    r = requests.post(Y_TOKEN, data=data, auth=(client_id, secret), timeout=30)
    r.raise_for_status()
    return r.json()


def _yget(token, path):
    r = requests.get(Y_API + path, params={"format": "json"},
                     headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _walk(obj):
    """Yield every dict inside Yahoo's deeply nested JSON."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _player_names(data):
    return {norm(d["name"]["full"]) for d in _walk(data)
            if isinstance(d.get("name"), dict) and "full" in d["name"]}


def yahoo_leagues(token):
    """[(league_key, league_name, my_team_key)] for the current NFL season."""
    leagues = {}
    for d in _walk(_yget(token, "/users;use_login=1/games;game_keys=nfl/leagues")):
        if "league_key" in d and "name" in d:
            leagues[d["league_key"]] = d["name"]
    teams = [d["team_key"] for d in
             _walk(_yget(token, "/users;use_login=1/games;game_keys=nfl/teams"))
             if "team_key" in d]
    out = []
    for lk, name in leagues.items():
        tk = next((t for t in teams if t.startswith(lk + ".t.")), None)
        if tk:
            out.append((lk, name, tk))
    return out


def yahoo_league(token, league_key, name, team_key, pages=8):
    mine = _player_names(_yget(token, f"/team/{team_key}/roster/players"))
    available = set()
    for i in range(pages):               # best available first, 25 per page
        data = _yget(token, f"/league/{league_key}/players;status=A;"
                            f"sort=AR;start={i * 25};count=25")
        found = _player_names(data)
        if not found:
            break
        available |= found
    return {"league": name, "mine": mine, "rostered": None,
            "available": available}
