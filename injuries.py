"""Latest injury news: the official NFL report merged with ESPN's running injury feed.

Why two sources. The official report is trustworthy but arrives in steps (the real designations come late in
the week) and carries no timestamps. Right after a game, a hurt player can show no designation at all for days.
ESPN's public injury feed carries a status, a short news note and a date as soon as it is known.

How they merge. For each player we take the more cautious status when the two disagree (a wrongly "healthy"
Out player is a worse mistake than a wrongly cautious Questionable one), and we only show a news note when it
belongs to the status we chose. Everything degrades quietly: if the feed is down, the official report still works,
and the screen says which sources it used.

ESPN's feed is an unofficial public endpoint (no key, no published terms for apps). INJURY_FEED_URL can point at
a different source with the same shape.
"""
import datetime as dt
import os
import re
import threading
import time

import nflreadpy as nfl
import pandas as pd
import requests

import sleeper

FEED = os.getenv("INJURY_FEED_URL", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries")
SEV = {"Out": 3, "Doubtful": 2, "Questionable": 1}
FEED_TTL, FAIL_TTL, OFFICIAL_TTL, MAP_TTL = 600, 120, 600, 12 * 3600
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BigGameFantasy)", "Accept": "application/json"}
PRACTICE = {"Did Not Participate In Practice": "Did not practice", "Limited Participation in Practice": "Limited in practice",
            "Full Participation in Practice": "Full practice"}

_lock = threading.Lock()
_cache = {}                       # key -> (time, value)
_state = {"feed": {"ok": None, "checked": None, "count": 0, "newest": None, "unmatched": 0, "error": None}}


def _fresh(key, ttl):
    hit = _cache.get(key)
    return hit[1] if hit and time.time() - hit[0] < ttl else None


def norm_status(s):
    """ESPN's wording -> Out / Doubtful / Questionable / None."""
    t = (s or "").strip().lower()
    if not t:
        return None
    if "doubtful" in t:
        return "Doubtful"
    if "questionable" in t or "day-to-day" in t or "day to day" in t:
        return "Questionable"
    if any(k in t for k in ("out", "injured reserve", "reserve", "suspen", "unable to perform", "inactive")):
        return "Out"
    return None


def _clean(text, limit=420):
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text or ""))).strip()
    return (t[:limit].rstrip() + "\u2026") if len(t) > limit else (t or None)


def _when(s):
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


# ------------------------------------------------------------------ ids --
def id_maps(season):
    """ESPN id -> gsis id, and (name, team) -> gsis id as a fallback."""
    with _lock:
        hit = _fresh(("ids", season), MAP_TTL)
    if hit is not None:
        return hit
    by_id, by_name = {}, {}
    try:
        r = nfl.load_rosters(season).select(["gsis_id", "espn_id", "full_name", "team"]).to_pandas()
        for row in r.itertuples():
            if not isinstance(row.gsis_id, str):
                continue
            try:
                if row.espn_id == row.espn_id and row.espn_id is not None:
                    by_id[str(int(float(row.espn_id)))] = row.gsis_id
            except (TypeError, ValueError):
                pass
            by_name[(sleeper.norm(row.full_name), row.team)] = row.gsis_id
    except Exception:
        pass
    with _lock:
        _cache[("ids", season)] = (time.time(), (by_id, by_name))
    return by_id, by_name


def player_info(season):
    """gsis id -> (name, team, position), so feed-only players can be listed by name."""
    with _lock:
        hit = _fresh(("info", season), MAP_TTL)
    if hit is not None:
        return hit
    info = {}
    try:
        r = nfl.load_rosters(season).select(["gsis_id", "full_name", "team", "position"]).to_pandas()
        info = {row.gsis_id: (row.full_name, row.team, row.position) for row in r.itertuples() if isinstance(row.gsis_id, str)}
    except Exception:
        pass
    with _lock:
        _cache[("info", season)] = (time.time(), info)
    return info


def _espn_id(ath):
    if ath.get("id") not in (None, ""):
        return str(ath["id"])
    for link in ath.get("links") or []:
        m = re.search(r"/id/(\d+)", link.get("href", "") if isinstance(link, dict) else "")
        if m:
            return m.group(1)
    return None


# ------------------------------------------------------------------ the news feed --
def parse_feed(data, by_id, by_name):
    """Pull one newest record per player out of the feed's JSON. Tolerates missing fields."""
    out, unmatched = {}, 0
    for team in (data.get("injuries") or []) if isinstance(data, dict) else []:
        t_abbr = (team.get("abbreviation") or "") if isinstance(team, dict) else ""
        for it in (team.get("injuries") or []) if isinstance(team, dict) else []:
            ath = it.get("athlete") or {}
            status = norm_status(it.get("status") or (it.get("type") or {}).get("description"))
            if not status:
                continue
            g = by_id.get(_espn_id(ath) or "")
            if not g:
                nm = ath.get("displayName") or ath.get("fullName") or ""
                tm = ((ath.get("team") or {}).get("abbreviation")) or t_abbr
                g = by_name.get((sleeper.norm(nm), tm)) or by_name.get((sleeper.norm(nm), {"WSH": "WAS", "LAR": "LA"}.get(tm, tm)))
            if not g:
                unmatched += 1
                continue
            det = it.get("details") or {}
            injury = det.get("type") or det.get("location") or None
            rec = {"status": status, "injury": str(injury).strip().capitalize() if injury else None,
                   "news": _clean(it.get("shortComment")), "long": _clean(it.get("longComment"), 900),
                   "date": _when(it.get("date")), "return_date": det.get("returnDate") or None}
            old = out.get(g)
            if not old or (rec["date"] and (not old["date"] or rec["date"] > old["date"])):
                out[g] = rec
    return out, unmatched


def espn_feed(season):
    """gsis id -> newest feed record. Cached for 10 minutes; after a failure we stay quiet for 2 minutes."""
    with _lock:
        hit = _fresh(("feed", season), FEED_TTL)
        if hit is not None:
            return hit
        if _state["feed"]["checked"] and _state["feed"]["ok"] is False and time.time() - _state["feed"]["checked"] < FAIL_TTL:
            return {}
    by_id, by_name = id_maps(season)
    try:
        r = requests.get(FEED, headers=HEADERS, timeout=6)
        r.raise_for_status()
        recs, unmatched = parse_feed(r.json(), by_id, by_name)
        dates = [v["date"] for v in recs.values() if v["date"]]
        with _lock:
            _cache[("feed", season)] = (time.time(), recs)
            _state["feed"].update(ok=True, checked=time.time(), count=len(recs), unmatched=unmatched, error=None,
                                  newest=max(dates).isoformat() if dates else None)
        return recs
    except Exception as e:
        with _lock:
            _state["feed"].update(ok=False, checked=time.time(), error=type(e).__name__)
        return {}


# ------------------------------------------------------------------ the official report --
def official(season, week):
    """gsis id -> {status, injury, practice} from the official report for that week (designations and practice notes)."""
    with _lock:
        hit = _fresh(("official", season, week), OFFICIAL_TTL)
    if hit is not None:
        return hit
    out = {}
    try:
        inj = nfl.load_injuries([season]).to_pandas()
        inj = inj[(inj["week"] == week) & (inj["game_type"] == "REG") & inj["gsis_id"].notna()]
        for r in inj.itertuples():
            status = r.report_status if r.report_status in SEV else None
            practice = PRACTICE.get(r.practice_status)
            if not status and practice in (None, "Full practice"):
                continue
            injury = r.report_primary_injury if isinstance(r.report_primary_injury, str) else (
                r.practice_primary_injury if isinstance(r.practice_primary_injury, str) else None)
            out[r.gsis_id] = {"status": status, "injury": injury, "practice": practice}
    except Exception:
        pass
    with _lock:
        _cache[("official", season, week)] = (time.time(), out)
    return out


# ------------------------------------------------------------------ the merge --
def merge(o, f, now=None):
    """One player's latest picture from the official record `o` and the feed record `f` (either may be None)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stale = bool(f) and f["date"] is not None and now - f["date"] > dt.timedelta(days=10)
    use_feed = bool(f)
    if o and o.get("status") and f:
        so, sf = SEV[o["status"]], SEV[f["status"]]
        use_feed = not stale and sf >= so                       # fresh news can escalate; stale news never overrides this week's report
    elif o and o.get("status"):
        use_feed = False
    rec = {"status": None, "injury": None, "practice": (o or {}).get("practice"), "news": None, "long": None,
           "updated": None, "source": None, "return_date": None}
    if use_feed:
        rec.update(status=f["status"], injury=f["injury"] or (o or {}).get("injury"), news=f["news"], long=f["long"],
                   updated=f["date"].isoformat() if f["date"] else None, source="ESPN", return_date=f["return_date"])
    elif o and o.get("status"):
        rec.update(status=o["status"], injury=o.get("injury"), source="NFL report")
        if f and f["status"] == o["status"] and not stale:       # the news belongs to this status, and it is still recent
            rec.update(news=f["news"], long=f["long"], updated=f["date"].isoformat() if f["date"] else None, return_date=f["return_date"])
    elif o:                                                      # practice note only, no game designation yet
        rec.update(injury=o.get("injury"), source="NFL report")
    return rec


def latest(season, week):
    """gsis id -> merged record, for every player with any designation or practice note."""
    off, feed = official(season, week), espn_feed(season)
    return {g: merge(off.get(g), feed.get(g)) for g in set(off) | set(feed)}


def sources(season, week):
    """Which sources fed the latest answer, so the page can be honest about it."""
    off = official(season, week)
    with _lock:
        f = dict(_state["feed"])
    return {"nfl_report": {"ok": bool(off), "week": week, "players": len(off)},
            "feed": {"ok": f["ok"], "players": f["count"], "newest": f["newest"], "unmatched": f["unmatched"], "error": f["error"]}}
