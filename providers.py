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


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36",
           "Accept": "application/json", "Referer": "https://fantasy.espn.com/"}


def parse_espn_url(url):
    lid = re.search(r"leagueId=(\d+)", url or "")
    tid = re.search(r"teamId=(\d+)", url or "")
    sid = re.search(r"seasonId=(\d{4})", url or "")
    if not lid:
        raise ValueError("Paste the link from your ESPN team page. It should contain leagueId= and teamId= "
                         "(open your team on fantasy.espn.com and copy the address bar).")
    return lid.group(1), int(tid.group(1)) if tid else None, int(sid.group(1)) if sid else None


def _cookie(value, braces=False):
    v = (value or "").strip().strip("\"'")
    if braces and v and not v.startswith("{"):
        v = "{" + v.strip("{}") + "}"
    return v


def espn_league(url, season, espn_s2=None, swid=None):
    lid, team_id, link_season = parse_espn_url(url)
    season = link_season or season                    # the season in the link wins over the box
    s2, sw = _cookie(espn_s2), _cookie(swid, braces=True)
    cookies = {"espn_s2": s2, "SWID": sw} if s2 and sw else {}
    try:
        r = requests.get(ESPN.format(season=season, lid=lid), headers=HEADERS, cookies=cookies, timeout=30,
                         params=[("view", "mRoster"), ("view", "mTeam"), ("view", "mSettings")])
    except requests.Timeout:
        raise ValueError("ESPN took too long to answer. Try again in a minute.")
    except requests.RequestException:
        raise ValueError("Couldn't reach ESPN from our server. Try again in a minute, or type your roster by hand.")
    if r.status_code in (401, 403):
        if cookies:
            raise PermissionError("ESPN rejected the cookies. Copy espn_s2 and SWID again while you're logged in to "
                                  "ESPN Fantasy (SWID looks like {XXXXXXXX-XXXX-...} with the curly braces). "
                                  "Or type your roster by hand.")
        raise PermissionError("This ESPN league is private. Add your espn_s2 and SWID cookies, which only a computer "
                              "can copy, or choose \"Type my roster\" to enter your players by hand.")
    if r.status_code == 404:
        raise ValueError(f"ESPN couldn't find league {lid} for the {season} season. Check the leagueId in your link "
                         "and the Season box.")
    if r.status_code >= 400:
        raise ValueError(f"ESPN answered with an error ({r.status_code}). It may be busy or blocking requests. "
                         "Try again in a few minutes, or type your roster by hand.")
    try:
        data = r.json()
    except ValueError:
        raise PermissionError("ESPN sent back a web page instead of league data. That usually means the league is "
                              "private (add your cookies) or ESPN is blocking the request. You can also type your "
                              "roster by hand.")
    if not isinstance(data, dict) or not data.get("teams"):
        raise ValueError("ESPN answered, but without any team rosters. The league may be private, or the season may "
                         "be wrong. You can type your roster by hand instead.")
    if team_id is None and sw:
        for t in data.get("teams", []):
            if sw.lower() in [o.lower() for o in t.get("owners", [])]:
                team_id = t["id"]
    if team_id is None:
        raise ValueError("Your link needs teamId=... Open your own team page on ESPN (the one that shows your "
                         "roster) and copy that link.")
    if team_id not in [t.get("id") for t in data["teams"]]:
        raise ValueError(f"Team {team_id} isn't in league {lid}. Copy the link from your own team page.")
    mine, rostered, teams = set(), set(), []
    for t in data.get("teams", []):
        names = set()
        for e in (t.get("roster") or {}).get("entries", []):
            name = (((e.get("playerPoolEntry") or {}).get("player")) or {}).get("fullName")
            if not name:
                continue
            names.add(norm(name))
            rostered.add(norm(name))
            if t["id"] == team_id:
                mine.add(norm(name))
        label = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip() or f"Team {t['id']}"
        teams.append({"name": label, "mine": t["id"] == team_id, "names": names})
    if not mine:
        raise ValueError("We found your team, but ESPN didn't list any players on it. Try again, or type your roster by hand.")
    return {"league": (data.get("settings") or {}).get("name", "ESPN league"),
            "mine": mine, "rostered": rostered, "available": None, "teams": teams}


def manual_league(lines):
    """A roster typed by hand: one player per line. Works for any fantasy site."""
    typed, seen = [], set()
    for raw in lines or []:
        s = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", str(raw))
        s = re.split(r",|\(|\s+[-\u2013\u2014]\s+|\s{2,}", s)[0].strip()      # keeps hyphenated names whole
        if len(s) >= 3 and norm(s) not in seen:
            seen.add(norm(s)); typed.append(s)
    if not typed:
        raise ValueError("Type at least one player, one per line.")
    return {"league": "My team (typed in)", "mine": {norm(s) for s in typed}, "rostered": set(), "available": None,
            "teams": [], "manual": True, "typed": typed[:40]}


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
