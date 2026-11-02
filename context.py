"""Game context and weather for the big-game odds.

What is used, and why (each was tested on seasons it was not fitted on):
  * Game environment. The betting lines (spread and total) say how many points a
    team is expected to score. Players in high-scoring games hit big games more
    often than their form alone suggests, and players in low-scoring games less
    often. Fitted: +0.105 log-odds per 4 points of implied team points. Applied
    consistently improved accuracy in all three held-out seasons.
  * Wind. Strong wind hurts passing. For QB/WR/TE in open-air stadiums the odds
    drop 0.122 log-odds per 10 mph above 8 mph. Needs a forecast (see below).
What was tested and NOT used, because it showed no reliable effect: temperature,
dome versus outdoors, rest days, division games, field surface.

The forecast comes from Open-Meteo (free, no key). If it can't be reached, the
wind adjustment is simply skipped. Forecast wind is only used for upcoming games;
past forecasts weren't saved, so the track record leaves weather out.
"""
import datetime as dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np
import requests
import nflreadpy as nfl

BETA_IMPLIED = 0.105
BETA_WIND = 0.122
AVG_IMPLIED = 22.5
WIND_POS = {"QB", "WR", "TE"}
WIND_FLOOR = 8.0                 # mph below which wind is ignored
# Home stadiums we can forecast (fixed domes are skipped). Retractable roofs are included, but only
# open-air stadiums get the wind adjustment, because the roof decision isn't known until near kickoff.
STADIUMS = {
    "ARI": (33.528, -112.263), "ATL": (33.755, -84.401), "DAL": (32.747, -97.095), "HOU": (29.685, -95.411),
    "IND": (39.760, -86.164),
    "BAL": (39.278, -76.623), "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
    "CIN": (39.095, -84.516), "CLE": (41.506, -81.700), "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
    "JAX": (30.324, -81.637), "KC": (39.049, -94.484), "MIA": (25.958, -80.239), "NE": (42.091, -71.264),
    "NYG": (40.814, -74.074), "NYJ": (40.814, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
    "SEA": (47.595, -122.332), "SF": (37.403, -121.970), "TB": (27.976, -82.503), "TEN": (36.167, -86.771),
    "WAS": (38.908, -76.864)}
_cache, _lock = {}, threading.Lock()


def _kickoff_utc(gameday, gametime):
    try:
        local = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
        return local.astimezone(dt.timezone.utc)
    except Exception:
        return None


SURFACES = {"grass": "grass"}          # everything else in the data is some kind of artificial turf


def _severity(code):
    c = int(code)
    return (9 if 95 <= c <= 99 else 8 if c in (71, 73, 75, 77, 85, 86) else 7.5 if c in (66, 67)
            else 7 if c in (61, 63, 65, 80, 81, 82) else 6.5 if c in (56, 57) else 6 if c in (51, 53, 55)
            else 5 if c in (45, 48) else c if c in (0, 1, 2, 3) else 0)


def condition_text(code):
    return {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
            51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
            61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
            71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow", 80: "light showers", 81: "showers",
            82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
            96: "thunderstorms with hail", 99: "thunderstorms with hail"}.get(int(code), "unsettled")


def forecast_at(team, kick):
    """Forecast for the three hours after kickoff: temperature, wind, rain chance, conditions. None if unavailable."""
    if team not in STADIUMS or kick is None:
        return None
    key = (team, kick.isoformat())
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 3 * 3600:
            return hit[1]
    lat, lon = STADIUMS[team]
    out = None
    try:
        d0 = kick.date()
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=5, params={
            "latitude": lat, "longitude": lon, "timezone": "UTC",
            "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,weather_code",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
            "start_date": d0.isoformat(), "end_date": (d0 + dt.timedelta(days=1)).isoformat()})
        r.raise_for_status()
        h = r.json()["hourly"]
        idx = {t: i for i, t in enumerate(h["time"])}
        start = kick.replace(minute=0, second=0, microsecond=0)
        rows = [idx[t] for t in ((start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(3)) if t in idx]

        def vals(name):
            return [h[name][i] for i in rows if name in h and h[name][i] is not None]

        def avg(name):
            v = vals(name)
            return round(float(np.mean(v)), 1) if v else None
        codes = vals("weather_code")
        if rows and avg("wind_speed_10m") is not None:
            out = {"temp": avg("temperature_2m"), "feels": avg("apparent_temperature"),
                   "wind": avg("wind_speed_10m"), "gust": max(vals("wind_gusts_10m"), default=None),
                   "precip_prob": max(vals("precipitation_probability"), default=None),
                   "precip_in": round(float(sum(vals("precipitation"))), 2) if vals("precipitation") else None,
                   "condition": condition_text(max(codes, key=_severity)) if codes else None}
    except Exception:
        out = None
    with _lock:
        _cache[key] = (time.time(), out)
    return out


def forecast_wind(team, kick):
    """Average forecast wind (mph) at kickoff, or None."""
    f = forecast_at(team, kick)
    return None if not f else f["wind"]


def roof_type(roof, history):
    """'dome', 'retractable' or 'open-air', from the game row or, if missing, the team's history."""
    r = roof if roof == roof else None
    if r in ("dome",):
        return "dome"
    if r in ("closed", "open"):
        return "retractable"
    if r == "outdoors":
        return "open-air"
    hs = set(history or [])
    return "retractable" if hs & {"closed", "open"} else "dome" if "dome" in hs else "open-air"


def surface_text(surface):
    return None if not surface or surface != surface else ("grass" if surface == "grass" else "artificial turf")


def team_history(season):
    """Each home team's roof values and latest surface over the last few seasons."""
    s = nfl.load_schedules([season - 2, season - 1, season]).to_pandas()
    s = s[(s["game_type"] == "REG") & (s["location"] == "Home")]
    roofs = s.groupby("home_team")["roof"].agg(lambda v: sorted(set(v.dropna()))).to_dict()
    surf = s.groupby("home_team")["surface"].agg(lambda v: v[v.fillna("") != ""].iloc[-1] if (v.fillna("") != "").any() else None).to_dict()
    return roofs, surf


def game_context(season, week, weather=True):
    """team -> {opponent, home, spread, total, implied, roof, stadium, roof_type, surface, forecast, wind}"""
    s = nfl.load_schedules([season]).to_pandas()
    s = s[(s["game_type"] == "REG") & (s["week"] == week)]
    try:
        roofs, surf = team_history(season)
    except Exception:
        roofs, surf = {}, {}
    now = dt.datetime.now(dt.timezone.utc)
    jobs, out = [], {}
    for g in s.itertuples():
        spread, total = g.spread_line, g.total_line
        rt = roof_type(g.roof, roofs.get(g.home_team))
        sf = surface_text(g.surface) or surface_text(surf.get(g.home_team))
        kick = _kickoff_utc(g.gameday, g.gametime)
        for team, opp, home in ((g.home_team, g.away_team, True), (g.away_team, g.home_team, False)):
            sp = None if spread != spread else float(spread if home else -spread)
            tot = None if total != total else float(total)
            imp = None if sp is None or tot is None else tot / 2 + sp / 2
            out[team] = {"opponent": opp, "home": home, "spread": sp, "total": tot, "implied": imp,
                         "roof": None if g.roof != g.roof else g.roof, "stadium": g.stadium, "roof_type": rt,
                         "surface": sf, "forecast": None, "wind": None, "kickoff": kick}
        if (weather and g.location == "Home" and rt != "dome" and g.home_team in STADIUMS and kick is not None
                and now - dt.timedelta(hours=6) < kick < now + dt.timedelta(days=15)):
            jobs.append((g.home_team, g.away_team, kick, rt))
    if jobs:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for (home, away, _, rt), f in zip(jobs, ex.map(lambda j: forecast_at(j[0], j[2]), jobs)):
                out[home]["forecast"] = out[away]["forecast"] = f
                if f and rt == "open-air":                     # wind only moves the odds in open-air stadiums
                    out[home]["wind"] = out[away]["wind"] = f["wind"]
    return out


def _fmt_spread(sp):
    return "a pick'em" if abs(sp) < 0.25 else (f"favored by {abs(sp):g}" if sp > 0 else f"an underdog by {abs(sp):g}")


def forecast_line(f, retractable=False):
    parts = [f"{f['temp']:.0f}\u00b0F" + (f" (feels like {f['feels']:.0f}\u00b0F)" if f.get("feels") is not None and abs(f["feels"] - f["temp"]) >= 4 else "")]
    wind = f"wind {f['wind']:.0f} mph" + (f", gusts to {f['gust']:.0f}" if f.get("gust") and f["gust"] - f["wind"] >= 5 else "")
    parts.append(wind)
    if f.get("precip_prob") is not None:
        parts.append(f"{f['precip_prob']:.0f}% chance of precipitation")
    if f.get("condition"):
        parts.append(f["condition"])
    return "Forecast at kickoff" + (" (only if the roof is open)" if retractable else "") + ": " + ", ".join(parts) + "."


def stadium_line(c):
    if not c.get("stadium"):
        return None
    sf = f" with a {c['surface']} surface" if c.get("surface") else ""
    return {"dome": f"{c['stadium']} is a dome{sf}, so weather doesn't matter.",
            "retractable": f"{c['stadium']} has a retractable roof{sf}, so weather only matters if it's open. That is decided close to kickoff.",
            "open-air": f"{c['stadium']} is open-air{sf}."}[c["roof_type"]]


def adjust(position, team, ctx):
    """(log-odds shift, wind part of it, explanation lines) for one player."""
    c = (ctx or {}).get(team)
    if not c:
        return 0.0, 0.0, []
    notes, shift, wind_part = [], 0.0, 0.0
    if c["implied"] is not None:
        shift += BETA_IMPLIED * (c["implied"] - AVG_IMPLIED) / 4.0
        notes.append(f"{team} is {_fmt_spread(c['spread'])} in a {c['total']:g}-point game, so about "
                     f"{c['implied']:.0f} points are expected for them.")
    line = stadium_line(c)
    if line:
        notes.append(line)
    f = c.get("forecast")
    if f and c.get("roof_type") != "dome":
        notes.append(forecast_line(f, c.get("roof_type") == "retractable"))
    if c.get("wind") is not None and position in WIND_POS:
        wind_part = -BETA_WIND * max(c["wind"] - WIND_FLOOR, 0.0) / 10.0
        shift += wind_part
        if wind_part < -0.01:
            notes.append("Wind like this tends to hurt passing, so his chance is lowered a little.")
    return shift, wind_part, notes


def label(shift, wind_part, ctx_team):
    """Short tag for the row when the context moves the odds noticeably."""
    if wind_part <= -0.06 and abs(wind_part) >= abs(shift - wind_part):
        return "windy forecast"
    imp = (ctx_team or {}).get("implied")
    if imp is None:
        return "game context"
    return "high-scoring game" if imp >= 24.2 else "low-scoring game" if imp <= 20.5 else "game script"
