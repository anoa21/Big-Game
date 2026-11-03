"""League outlook: roster strength for every team, and playoff odds.

Roster strength. Each team's best starting lineup is built from the same rest-of-season
projections the trade analyzer uses, using your league's real lineup slots, with bye
weeks and long-term absences taken out week by week.

Playoff odds. The rest of the regular season is simulated 5,000 times. Every team's
weekly score is drawn from a bell curve centered on a blend of (a) what its roster is
projected to score that week and (b) how much it has actually been scoring, then the
league's real remaining schedule decides each game. Teams are ranked by wins, then points
scored, and the top teams make the playoffs. A shared "season shock" for each team keeps
the odds from sounding surer than they are.
Simplified on purpose: no division rules, ties broken by points scored, scores treated as
a bell curve, rosters held as they are today (no future trades or pickups).
"""
import numpy as np

import ros

POS = ["QB", "RB", "WR", "TE"]
TYPICAL = {"K": 8.0, "DEF": 7.0}          # typical points for a kicker / defense slot
REPL_SD = 7.0                             # weekly spread of a replacement-level starter
SEASON_SHOCK_SD = 6.0                     # per-team uncertainty about its true weekly level, in points
SIMS = 5000


def _clean(row):
    return ros._clean(row)


def _players(table, ids):
    return [_clean(table.loc[i].to_dict()) for i in ids if i in table.index]


def _lineup(players, slots, repl):
    """Best lineup by points per game. Returns (starters, ppg total, sd, empty slots)."""
    ranked = [dict(p, ros_pts=p["ros_ppg"]) for p in players]
    starters, total, empty = ros.best_lineup(ranked, slots, repl, 1.0, True)
    var = sum((p["sd"] / np.sqrt(max(p["exp_games"], 1))) ** 2 for p in starters) + len(empty) * REPL_SD ** 2
    return starters, float(total), float(np.sqrt(var)), empty


def team_numbers(table, ids, slots, repl, weeks, current_week):
    """One team's strength profile and its projected score for each remaining week."""
    roster = [p for p in _players(table, ids) if not p.get("flag")]
    starters, ppg, sd, empty = _lineup(roster, slots, repl)
    group = {pos: round(sum(p["ros_ppg"] for p in starters if p["position"] == pos), 1) for pos in POS}
    bench = sorted((p for p in roster if p["player_id"] not in {s["player_id"] for s in starters}),
                   key=lambda p: -(p["ros_ppg"] - repl.get(p["position"], 0)))
    depth = round(sum(max(p["ros_ppg"] - repl.get(p["position"], 0), 0) for p in bench[:4]), 1)
    kd = sum(TYPICAL[k] for k in ("K", "DEF") for _ in range(slots.get(k, 0)))
    by_week = {}
    for w in weeks:
        avail = [p for p in roster if p.get("bye") != w and not (w == current_week and p.get("injury_status") == "Out")]
        by_week[w] = _lineup(avail, slots, repl)[1] + kd
    return {"ppg": round(ppg + kd, 1), "sd": sd, "group": group, "depth": depth, "empty": empty,
            "by_week": by_week, "top": [p["name"] for p in sorted(starters, key=lambda p: -p["ros_ppg"])[:3]],
            "n_players": len(roster)}


def rank_desc(values):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    r = [0] * len(values)
    for place, i in enumerate(order, 1):
        r[i] = place
    return r


def strength(teams):
    """teams: [{name, mine, numbers}] -> adds overall and per-position ranks, and your strengths/weaknesses."""
    n = len(teams)
    for key, label in (("ppg", "overall"), ("depth", "depth")):
        for t, rk in zip(teams, rank_desc([t["numbers"][key] for t in teams])):
            t.setdefault("rank", {})[label] = rk
    for pos in POS:
        for t, rk in zip(teams, rank_desc([t["numbers"]["group"][pos] for t in teams])):
            t["rank"][pos] = rk
    avg = {pos: float(np.mean([t["numbers"]["group"][pos] for t in teams])) for pos in POS}
    avg["ppg"] = float(np.mean([t["numbers"]["ppg"] for t in teams]))
    avg["depth"] = float(np.mean([t["numbers"]["depth"] for t in teams]))
    return avg, n


def verdict(mine, avg, n):
    """Plain-English strengths and weaknesses for one team."""
    strong = [p for p in POS if mine["rank"][p] <= max(2, n // 4)]
    weak = [p for p in POS if mine["rank"][p] >= n - max(1, n // 4) + 1]
    lines = []
    if strong:
        lines.append("Strongest spots: " + ", ".join(f"{p} ({_ord(mine['rank'][p])} of {n})" for p in strong) + ".")
    if weak:
        lines.append("Weakest spots: " + ", ".join(f"{p} ({_ord(mine['rank'][p])} of {n})" for p in weak)
                     + ". Those are the places to look first on the waiver wire or in a trade.")
    if mine["rank"]["depth"] >= n - 2:
        lines.append(f"Your bench is thin ({_ord(mine['rank']['depth'])} of {n}), so an injury would hurt more than for most teams.")
    elif mine["rank"]["depth"] <= 3:
        lines.append(f"Your bench is among the deepest in the league ({_ord(mine['rank']['depth'])} of {n}).")
    return lines


def _ord(k):
    return f"{k}{'th' if 11 <= k % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def simulate(teams, schedule, playoff_teams, my_index, sims=SIMS, seed=7):
    """teams: [{wins, ties, pf, mean_by_week:{w:..}, sd}], schedule: {week: [(i, j), ...]}.
    Returns playoff probability, expected wins, and how each of your remaining games swings your odds."""
    rng = np.random.default_rng(seed)
    T = len(teams)
    wins = np.tile(np.array([t["wins"] + 0.5 * t["ties"] for t in teams], float), (sims, 1))
    pf = np.tile(np.array([t["pf"] for t in teams], float), (sims, 1))
    shock = rng.normal(0, SEASON_SHOCK_SD, (sims, T))
    sd = np.array([max(t["sd"], 12.0) for t in teams])
    mine_results = []
    for w in sorted(schedule):
        score = np.zeros((sims, T))
        for i, t in enumerate(teams):
            score[:, i] = rng.normal(t["mean_by_week"][w] + shock[:, i], sd[i])
        for a, b in schedule[w]:
            win_a = score[:, a] > score[:, b]
            wins[:, a] += win_a
            wins[:, b] += ~win_a
            if my_index in (a, b):
                mine_results.append((w, a if b == my_index else b, win_a if a == my_index else ~win_a))
        pf += score
    key = wins * 1e5 + pf + rng.random((sims, T)) * 1e-3                 # wins, then points scored
    rank = (-key).argsort(axis=1).argsort(axis=1) + 1
    made = rank <= playoff_teams
    swing = []
    for w, opp, won in mine_results:
        if 20 <= won.sum() <= sims - 20:
            p_win, p_lose = float(made[won, my_index].mean()), float(made[~won, my_index].mean())
            swing.append({"week": w, "opponent_index": opp, "if_win": round(p_win, 3), "if_lose": round(p_lose, 3),
                          "swing": round(p_win - p_lose, 3), "win_chance": round(float(won.mean()), 3)})
    return {"playoff": made.mean(axis=0), "wins": wins.mean(axis=0), "swing": sorted(swing, key=lambda s: -s["swing"]),
            "seed_median": np.median(rank, axis=0)}
