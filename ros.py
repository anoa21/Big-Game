"""Rest-of-season projections and the trade analyzer.

Each player gets projected points per game from recent form (recent games
weighted more), nudged by how generous the opposing defenses are over his
remaining schedule, times the games left (byes and this week's injury report
included). A trade is judged by how much it changes your best starting lineup
over the rest of the season; with no roster, by "points above replacement".
These are estimates with a wide margin, and the analyzer says so.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl

import sim

POS = ["QB", "RB", "WR", "TE"]
LAST_WEEK = 18
# Replacement level without league info: 12-team, 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX
REPL_RANK = {"QB": 13, "RB": 30, "WR": 36, "TE": 13}
DEFAULT_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = {"REC_FLEX": {"WR", "TE"}, "WRRB_FLEX": {"WR", "RB"},
                 "FLEX": {"RB", "WR", "TE"}, "SUPER_FLEX": {"QB", "RB", "WR", "TE"}}


def remaining_schedule(season, week):
    """team -> [(week, opponent)] for the regular-season weeks still to play."""
    s = nfl.load_schedules([season]).to_pandas()
    s = s[(s["game_type"] == "REG") & (s["week"] >= week) & (s["week"] <= LAST_WEEK)]
    out = {}
    for r in s.itertuples():
        out.setdefault(r.home_team, []).append((int(r.week), r.away_team))
        out.setdefault(r.away_team, []).append((int(r.week), r.home_team))
    return out


def schedule_label(f):
    return "favorable schedule" if f >= 1.03 else "tough schedule" if f <= 0.97 else "average schedule"


def build(hist_df, season, week, injuries=None):
    """One row per active QB/RB/WR/TE with rest-of-season numbers."""
    inj = injuries or {}
    hist = hist_df[hist_df["position"].isin(POS)].sort_values(["season", "week"])
    dtable = sim.defense_table(hist_df)
    pos_avg = {p: float(np.mean([v[1] for (t, q), v in dtable.items() if q == p])) for p in POS}
    sched = remaining_schedule(season, week)
    last = hist.groupby("player_id").tail(1).set_index("player_id")
    recent = hist.groupby("player_id").tail(sim.HIST_N).groupby("player_id")["fantasy_points_ppr"]
    rows = []
    for pid, pts in recent:
        pts = pts.dropna().to_numpy(float)
        if len(pts) < 3:
            continue
        r = last.loc[pid]
        team, pos = r["team"], r["position"]
        opps = sched.get(team, [])
        if not opps:
            continue
        w = sim.DECAY ** np.arange(len(pts))[::-1]
        w /= w.sum()
        ppg = float((pts * w).sum())
        sd = max(float(np.sqrt((w * (pts - ppg) ** 2).sum())), 3.0)
        fac = float(np.mean([sim.matchup_factor(dtable.get((o, pos), (None, None))[1], pos_avg[pos])
                             for _, o in opps]))
        status, injury = inj.get(pid, (None, None))
        play = sim.PLAY_PROB.get(status, 1.0)
        games = len(opps)
        exp_games = games - ((1 - play) if any(wk == week for wk, _ in opps) else 0)
        weeks = {wk for wk, _ in opps}
        bye = next((wk for wk in range(week, LAST_WEEK + 1) if wk not in weeks), None)
        flag = None
        if int(r["season"]) < season:
            if week > 3:          # not seen all season: retired, cut, or out for the year
                continue
        elif week - int(r["week"]) > 3:
            flag = f"Hasn't played since week {int(r['week'])}"
        ros_ppg = ppg * fac
        rows.append({
            "player_id": pid, "name": r["player_display_name"], "position": pos, "team": team,
            "ppg": ppg, "ros_ppg": ros_ppg, "games": games, "exp_games": exp_games,
            "ros_pts": ros_ppg * exp_games, "sd": sd * np.sqrt(max(exp_games, 1)),
            "sched": fac, "sched_label": schedule_label(fac), "bye": bye,
            "injury_status": status, "injury": injury, "flag": flag})
    return pd.DataFrame(rows).set_index("player_id", drop=False)


def replacement_ppg(table, free_agents=None):
    """Points per game a free agent would give you, by position."""
    t = table[table["flag"].isna() & (table["injury_status"] != "Out")]
    out = {}
    for pos in POS:
        if free_agents is not None:
            fa = t[(t["position"] == pos) & t.index.isin(free_agents)]
            if len(fa):
                out[pos] = float(fa["ros_ppg"].nlargest(2).mean())
                continue
        s = t[t["position"] == pos]["ros_ppg"].sort_values(ascending=False)
        out[pos] = float(s.iloc[min(REPL_RANK[pos], len(s)) - 1]) if len(s) else 0.0
    return out


def best_lineup(players, slots, repl=None, ref_games=0.0, with_empty=False):
    """Greedy fill: dedicated slots first, then the most restrictive flex first.
    A slot nobody on the roster can fill is credited at replacement level (you
    would pick up a free agent), so a thin roster isn't punished unfairly."""
    pool = sorted(players, key=lambda p: -p["ros_pts"])
    used, starters, extra, empty = set(), [], [0.0], []

    def take(elig, n, label):
        for _ in range(n):
            c = next((p for p in pool if p["position"] in elig and p["player_id"] not in used), None)
            if c:
                used.add(c["player_id"]); starters.append(c)
            else:
                empty.append(label)
                if repl:
                    extra[0] += max(repl.get(pos, 0) for pos in elig) * ref_games
    for pos in POS:
        take({pos}, slots.get(pos, 0), pos)
    for name in ("REC_FLEX", "WRRB_FLEX", "FLEX", "SUPER_FLEX"):
        take(FLEX_ELIGIBLE[name], slots.get(name, 0), name.replace("_", " ").title())
    total = float(sum(p["ros_pts"] for p in starters) + extra[0])
    return (starters, total, empty) if with_empty else (starters, total)


def _clean(d):
    """pandas stores missing values as NaN; turn them back into None."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def _row(p, repl):
    value = max(p["ros_ppg"] - repl.get(p["position"], 0), 0) * p["exp_games"]
    return {"player_id": p["player_id"], "name": p["name"], "position": p["position"],
            "team": p["team"], "ppg": round(p["ppg"], 1), "ros_ppg": round(p["ros_ppg"], 1),
            "games": int(p["games"]), "exp_games": round(float(p["exp_games"]), 1),
            "ros_pts": round(p["ros_pts"]), "low": round(max(p["ros_pts"] - 1.28 * p["sd"], 0)),
            "high": round(p["ros_pts"] + 1.28 * p["sd"]), "sched_label": p["sched_label"],
            "bye": None if p["bye"] is None else int(p["bye"]), "injury_status": p["injury_status"],
            "injury": p["injury"], "flag": p["flag"], "value": round(value)}


def _level(z):
    return ("good" if z >= 0.8 else "lean_good" if z >= 0.3 else "even" if z > -0.3
            else "lean_bad" if z > -0.8 else "bad")


def _sign(level):
    return 1 if level in ("good", "lean_good") else -1 if level in ("bad", "lean_bad") else 0


def _mutual(mine, theirs):
    """One plain-English read on whether both managers come out ahead."""
    m, t = _sign(mine), _sign(theirs)
    table = {(1, 1): ("Works for both sides", "good"), (1, 0): ("Works for both sides", "good"),
             (0, 1): ("Works for both sides", "good"), (0, 0): ("Close to even for both of you", "even"),
             (1, -1): ("Favors you, so expect pushback", "even"), (0, -1): ("Leans your way, so expect pushback", "even"),
             (-1, 1): ("Favors them, so they will likely say yes", "bad"), (-1, 0): ("Leans their way", "bad"),
             (-1, -1): ("Hurts both of you", "bad")}
    text, kind = table[(m, t)]
    return {"text": text, "kind": kind}


def _partner_side(table, partner_ids, give, get, get_ids, slots, repl, sd, name, my_level):
    """The same trade, seen from the other manager's roster."""
    roster = [_clean(table.loc[i].to_dict()) for i in partner_ids if i in table.index]
    if not roster:
        return None
    ref = float(table["exp_games"].median())
    lost = set(get_ids)
    kept = [p for p in roster if p["player_id"] not in lost]
    after_roster = kept + [g for g in give if g["player_id"] not in {p["player_id"] for p in kept}]
    before, tb, eb = best_lineup(roster, slots, repl, ref, True)
    after, ta, ea = best_lineup(after_roster, slots, repl, ref, True)
    bi, ai = {p["player_id"] for p in before}, {p["player_id"] for p in after}
    diff = ta - tb
    lvl = _level(diff / sd)
    thin = []
    for label in set(ea):
        extra = ea.count(label) - eb.count(label)
        if extra > 0:
            thin.append(label)
    notes = [f"{p['name']} isn't on {name}'s roster, so their numbers may be off."
             for p in get if p["player_id"] not in {r["player_id"] for r in roster}]
    return {"name": name, "level": lvl, "diff": round(diff), "range": [round(diff - 1.28 * sd), round(diff + 1.28 * sd)],
            "lineup": {"before": round(tb), "after": round(ta), "diff": round(diff),
                       "enters": [p["name"] for p in after if p["player_id"] not in bi],
                       "leaves": [p["name"] for p in before if p["player_id"] not in ai]},
            "thin": thin, "notes": notes, "mutual": _mutual(my_level, lvl)}


def analyze(table, give_ids, get_ids, roster_ids=None, slots=None, replacement=None,
            partner_ids=None, partner_name="Their team"):
    missing = [i for i in list(give_ids) + list(get_ids) if i not in table.index]
    if missing:
        raise KeyError("Couldn't find one of those players. Pick them from the search list.")
    if set(give_ids) & set(get_ids):
        raise ValueError("The same player is on both sides of the trade.")
    mine = set(roster_ids or [])
    owned = [table.loc[i, "name"] for i in get_ids if i in mine]
    if owned:
        raise ValueError(f"{owned[0]} is already on your team. Pick a player from the other side.")
    repl = {**replacement_ppg(table), **(replacement or {})}
    give = [_clean(table.loc[i].to_dict()) for i in give_ids]
    get = [_clean(table.loc[i].to_dict()) for i in get_ids]
    give_rows, get_rows = [_row(p, repl) for p in give], [_row(p, repl) for p in get]
    value_diff = sum(r["value"] for r in get_rows) - sum(r["value"] for r in give_rows)
    sd = float(np.sqrt(sum(p["sd"] ** 2 for p in give + get))) or 1.0

    lineup, diff = None, float(value_diff)
    if roster_ids:
        roster = [_clean(table.loc[i].to_dict()) for i in roster_ids if i in table.index]
        slots = slots or DEFAULT_SLOTS
        ref = float(table["exp_games"].median())
        before, tb = best_lineup(roster, slots, repl, ref)
        kept = [p for p in roster if p["player_id"] not in set(give_ids)]
        after, ta = best_lineup(kept + [g for g in get if g["player_id"] not in {p["player_id"] for p in kept}], slots, repl, ref)
        bi, ai = {p["player_id"] for p in before}, {p["player_id"] for p in after}
        lineup = {"before": round(tb), "after": round(ta), "diff": round(ta - tb),
                  "enters": [p["name"] for p in after if p["player_id"] not in bi],
                  "leaves": [p["name"] for p in before if p["player_id"] not in ai]}
        diff = ta - tb

    z = diff / sd
    level = _level(z)
    notes = []
    if roster_ids:
        for p in give:
            if p["player_id"] not in mine:
                notes.append(f"{p['name']} wasn't found on your roster, so the lineup numbers may be off.")
    for p, side in [(p, "give") for p in give] + [(p, "get") for p in get]:
        who = p["name"]
        if p["flag"]:
            notes.append(f"{who}: {p['flag'].lower()}, so check his status before trading.")
        if p["injury_status"]:
            notes.append(f"{who} is listed {p['injury_status']} this week"
                         + (f" ({str(p['injury']).lower()})." if p["injury"] else "."))
        if p["bye"] is not None:
            notes.append(f"{who}'s team has its bye in week {int(p['bye'])}.")
    partner = None
    if partner_ids:
        partner = _partner_side(table, partner_ids, give, get, get_ids, slots or DEFAULT_SLOTS, repl,
                                sd, partner_name, level)
    return {"give": give_rows, "get": get_rows, "lineup": lineup, "level": level,
            "diff": round(diff), "range": [round(diff - 1.28 * sd), round(diff + 1.28 * sd)],
            "basis": "lineup" if lineup else "value", "notes": notes, "partner": partner}
