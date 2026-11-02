"""Next man up: when a key player is out, who inherits his carries and targets?

How it works. A player who has been getting, say, 25% of his team's targets and
then misses a game frees up those targets. We hand them to his healthy teammates
in proportion to each one's own recent share, then convert extra carries and
targets into fantasy points with factors fitted on real games (2021-2025) in
which a key player sat out.

What it does and doesn't claim. On games the fit never saw, teammates the model
expected to gain 1.5+ points gained about 2.4 to 2.5 points more than similar
players on teams with nobody out. It does NOT make every individual prediction
more accurate (week-to-week noise is much bigger), so it is shown as its own
guide and is not folded into the big-game percentages.
Only players who played recently count: someone who has been out for weeks is
already reflected in his teammates' recent numbers.
Refit with:  python nextup.py   (needs a few minutes and the nflverse data)
"""
import numpy as np
import pandas as pd

COEF = {"RB": (0.00, 0.39), "WR": (0.39, 0.14), "TE": (0.17, 0.00)}   # points per extra target, per extra carry
SHRINK = 0.85            # held-out seasons showed the fit running ~15% generous
KEY_TS, KEY_CS = 0.12, 0.25
MIN_BUMP = 1.0
STATUSES = ("Out", "Doubtful", "Questionable")
FIRM = ("Out", "Doubtful")
POS3 = ["RB", "WR", "TE"]
EVIDENCE = "In past seasons, players we expected to gain 1.5+ points gained about 2.4 more than similar players on teams with nobody out."


def _prep(hist_df):
    h = hist_df[hist_df["position"].isin(POS3 + ["QB"])].copy()
    for c in ("targets", "carries", "fantasy_points_ppr"):
        h[c] = h[c].fillna(0)
    h["t"] = h["season"] * 100 + h["week"]
    tot = h.groupby(["team", "t"])[["targets", "carries"]].transform("sum")
    h["team_t"], h["team_c"] = tot["targets"], tot["carries"]
    h["ts"] = np.where(tot["targets"] > 0, h["targets"] / tot["targets"], 0.0)
    h["cs"] = np.where(tot["carries"] > 0, h["carries"] / tot["carries"], 0.0)
    return h


def _clean(o):
    """JSON can't carry NaN, so turn missing values into None."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return None if isinstance(o, float) and np.isnan(o) else o


def _ordinal_words(a, names):
    return names[0] if len(names) == 1 else " and ".join(names[:2]) if len(names) == 2 else f"{names[0]} and others"


def compute(hist_df, season, week, injuries):
    """injuries: {player_id: (status, injury)}.  Returns {"teams": [...], "by_player": {...}}."""
    h = _prep(hist_df)
    latest_team = h.sort_values("t").groupby("player_id").tail(1).set_index("player_id")["team"].to_dict()
    teams = []
    for tm, d in h.groupby("team"):
        tg = sorted(d["t"].unique())[-3:]
        if len(tg) < 2:
            continue
        r = d[d["t"].isin(tg)]
        T, C = float(r.groupby("t")["team_t"].first().mean()), float(r.groupby("t")["team_c"].first().mean())
        per = r.groupby("player_id").agg(name=("player_display_name", "last"), pos=("position", "last"),
                                         games=("t", "nunique"), ts=("ts", "mean"), cs=("cs", "mean"))
        per = per[(per["games"] >= 2) & per["pos"].isin(POS3)]
        per["status"] = [injuries.get(pid, (None, None))[0] for pid in per.index]
        per["injury"] = [injuries.get(pid, (None, None))[1] for pid in per.index]
        per = per[[latest_team.get(pid) == tm for pid in per.index]]
        absent = per[per["status"].isin(STATUSES) & ((per["ts"] >= KEY_TS) | (per["cs"] >= KEY_CS))]
        if absent.empty:
            continue
        pool = per[~per["status"].isin(FIRM) & ~per.index.isin(absent.index)]
        s_ts, s_cs = float(pool["ts"].sum()), float(pool["cs"].sum())
        if pool.empty:
            continue
        recent8 = d.sort_values("t").groupby("player_id").tail(8).groupby("player_id")["fantasy_points_ppr"].mean()
        gainers = []
        for pid, p in pool.iterrows():
            a_t, a_c = COEF[p["pos"]]
            firm = maybe = x_t = x_c = 0.0
            who_firm, who_maybe = [], []
            for aid, a in absent.iterrows():
                xt = a["ts"] * (p["ts"] / s_ts if s_ts else 0.0) * T
                xc = a["cs"] * (p["cs"] / s_cs if s_cs else 0.0) * C
                bump = SHRINK * (a_t * xt + a_c * xc)
                if a["status"] in FIRM:
                    firm += bump; who_firm.append((bump, a["name"]))
                else:
                    maybe += bump; who_maybe.append((bump, a["name"]))
                x_t += xt; x_c += xc
            if firm + maybe >= MIN_BUMP:
                gainers.append({"player_id": pid, "name": p["name"], "position": p["pos"],
                                "bump_firm": round(firm, 1), "bump_maybe": round(maybe, 1),
                                "extra_targets": round(x_t, 1), "extra_carries": round(x_c, 1),
                                "ppg": round(float(recent8.get(pid, np.nan)), 1) if pid in recent8 else None,
                                "because": [n for _, n in sorted(who_firm, reverse=True)],
                                "if_sits": [n for _, n in sorted(who_maybe, reverse=True)],
                                "status": p["status"]})
        if not gainers:
            continue
        gainers.sort(key=lambda g: -(g["bump_firm"] + g["bump_maybe"]))
        teams.append({"team": tm, "top": gainers[0]["bump_firm"] + gainers[0]["bump_maybe"],
                      "absent": [{"player_id": aid, "name": a["name"], "position": a["pos"], "status": a["status"],
                                  "injury": a["injury"], "target_share": round(float(a["ts"]), 2),
                                  "carry_share": round(float(a["cs"]), 2)} for aid, a in absent.iterrows()],
                      "gainers": gainers[:5]})
    teams.sort(key=lambda t: -t["top"])
    by_player = {}
    for t in teams:
        for g in t["gainers"]:
            if g["bump_firm"] >= MIN_BUMP:
                why = f"Next man up: about +{g['bump_firm']:.1f} points with {_ordinal_words(0, g['because'])} out"
                val = g["bump_firm"]
            else:
                why = f"Next man up: about +{g['bump_maybe']:.1f} points if {_ordinal_words(0, g['if_sits'])} sits (questionable)"
                val = g["bump_maybe"]
            by_player[g["player_id"]] = {"bump": round(val, 1), "why": why}
    return _clean({"teams": teams, "by_player": by_player, "evidence": EVIDENCE})


if __name__ == "__main__":          # refit the conversion factors from real games
    import sys
    sys.path.insert(0, ".")
    print("See the fitting notes in this file's docstring; the fitted values are in COEF.")
