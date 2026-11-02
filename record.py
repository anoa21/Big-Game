"""Track record: grade our own picks, week by week, against what happened.

For each finished week we rebuild the picks the app would have shown BEFORE that
week (model trained only on earlier games, injury report for that week), then
check who actually had a big game. Nothing here peeks at the results.
"""
import os
import threading
import time

import numpy as np
import pandas as pd
import nflreadpy as nfl

import model
import sim

VERSION = "v2"                     # bump when the model changes so old grades are redone
STRONG, SOLID = 0.40, 0.20         # same cutoffs as the Strong / Solid / Risky tags
BUCKETS = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]
_state = {}                        # season -> {"weeks": {w: DataFrame}, "building": bool, "error": str, "doing": int}
_lock = threading.Lock()


def finished_weeks(season):
    s = nfl.load_schedules([season]).to_pandas()
    s = s[s["game_type"] == "REG"]
    done = s.groupby("week").apply(
        lambda g: bool(g["home_score"].notna().all() and g["away_score"].notna().all()))
    # Week 1 is skipped: with no games yet this season, picks lean on last year's rosters.
    return sorted(int(w) for w, ok in done.items() if ok and int(w) >= 2)


def grade_week(stats, season, week):
    """One row per player we would have shown that week: our odds and what happened."""
    seasons = sorted(stats["season"].unique().tolist())
    preds = model.predict_week(seasons, season, week, stats=stats)
    hist = stats[(stats["season"] < season) | (stats["week"] < week)]
    e = sim.enrich(preds, hist, season, week, with_props=False, weather=False)   # past forecasts weren't saved
    e = e[e["injury_status"] != "Out"]                       # the app hides players ruled out
    played = stats[(stats["season"] == season) & (stats["week"] == week)] \
        .set_index("player_id")["fantasy_points_ppr"]
    pts = e["player_id"].map(played)                          # NaN = didn't play, so no big game
    hit = (pts >= e["position"].map(model.BIG_GAME)).astype(int)
    return pd.DataFrame({"week": week, "player_id": e["player_id"].values,
                         "name": e["player_display_name"].values, "position": e["position"].values,
                         "prob": e["final_prob"].values, "hit": hit.values})


def _path(season):
    return f"/tmp/bg_record_{VERSION}_{season}"


def _worker(season):
    st = _state[season]
    try:
        stats = model.load_stats([season - 2, season - 1, season])
        for w in finished_weeks(season):
            if w in st["weeks"]:
                continue
            st["doing"] = w
            f = f"{_path(season)}_{w}.pkl"
            try:
                df = pd.read_pickle(f)
            except Exception:
                df = grade_week(stats, season, w)
                try:
                    df.to_pickle(f)
                except Exception:
                    pass
            st["weeks"][w] = df
    except Exception as e:
        st["error"] = str(e)
    finally:
        st["building"] = False
        st["doing"] = None


def start(season):
    """Kick off grading in the background (once) and return the current state."""
    with _lock:
        st = _state.setdefault(season, {"weeks": {}, "building": False, "error": None, "doing": None})
        total = len(finished_weeks(season))
        if not st["building"] and len(st["weeks"]) < total and not st["error"]:
            st["building"] = True
            threading.Thread(target=_worker, args=(season,), daemon=True).start()
        return st, total


def tag_of(p):
    return "strong" if p >= STRONG else "solid" if p >= SOLID else "risky"


def summarize(frames):
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["tag"] = df["prob"].map(tag_of)
    tags = {}
    for t in ("strong", "solid", "risky"):
        d = df[df["tag"] == t]
        tags[t] = {"n": int(len(d)), "predicted": round(float(d["prob"].mean()), 3) if len(d) else None,
                   "actual": round(float(d["hit"].mean()), 3) if len(d) else None}
    cal = []
    for lo, hi in BUCKETS:
        d = df[(df["prob"] >= lo) & (df["prob"] < hi)]
        if len(d) >= 5:
            cal.append({"low": lo, "high": min(hi, 1.0), "n": int(len(d)),
                        "predicted": round(float(d["prob"].mean()), 3),
                        "actual": round(float(d["hit"].mean()), 3)})
    weekly = []
    for w, d in df.groupby("week"):
        top = d.nlargest(10, "prob")
        weekly.append({"week": int(w), "hits": int(top["hit"].sum()), "of": int(len(top))})
    hits, of = sum(x["hits"] for x in weekly), sum(x["of"] for x in weekly)
    return {"players": int(len(df)), "base_rate": round(float(df["hit"].mean()), 3), "tags": tags,
            "calibration": cal, "weekly": weekly,
            "top10": {"hits": hits, "of": of, "rate": round(hits / of, 3) if of else None}}


def report(season):
    st, total = start(season)
    frames = [st["weeks"][w] for w in sorted(st["weeks"])]
    return {"season": season, "weeks_done": len(frames), "weeks_total": total,
            "building": st["building"], "doing": st["doing"], "error": st["error"],
            "summary": summarize(frames)}
