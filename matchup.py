"""Matchup play-calling: your chance to beat this week's opponent, and which
lineup changes raise it.

Every player's week is simulated thousands of times (recent form, the defense he
faces, the injury report). Add up each team's starters in each simulation and
count how often you come out ahead. A swap is judged by how much it moves that
chance, so the advice automatically favors upside when you're the underdog and
a steady floor when you're the favorite.
Waiver help: every free agent in your league is tested the same way against this
week's opponent, using only players whose games haven't kicked off, with a suggested
drop if your roster is full. Players whose games have started are locked and ignored.
Simplifications: players are treated as independent, kickers and defenses use
typical scores, and only reception scoring is adjusted for your league.
"""
import numpy as np

import ros
import sim
import sleeper

N = 4000
POS = ["QB", "RB", "WR", "TE"]
TYPICAL = {"K": (8.0, 3.5), "DEF": (7.0, 4.0)}          # typical weekly fantasy points
SLOT_OK = {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, **ros.FLEX_ELIGIBLE}


def league_points(df, rec=1.0, te_bonus=0.0):
    """Per-game points under your league's reception scoring (PPR is the baseline)."""
    recv = df["receptions"].fillna(0)
    pts = df["fantasy_points_ppr"].fillna(0) - (1.0 - rec) * recv
    if te_bonus:
        pts = pts + np.where(df["position"] == "TE", te_bonus * recv, 0.0)
    return pts


def _nn(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


class Sampler:
    def __init__(self, hist_df, preds, table, week, rec=1.0, te_bonus=0.0, n=N, kickoffs=None, now=None):
        h = hist_df[hist_df["position"].isin(POS)].sort_values(["season", "week"]).copy()
        h["adj"] = league_points(h, rec, te_bonus)
        self.recent = {pid: g["adj"].to_numpy(float)
                       for pid, g in h.groupby("player_id").tail(sim.HIST_N).groupby("player_id")}
        self.preds = preds.set_index("player_id", drop=False)
        self.pos_avg = preds.groupby("position")["opp_pts_allowed_L6"].mean().to_dict()
        self.table, self.week, self.n = table, week, n
        self.kickoffs, self.now = kickoffs or {}, now
        self.repl = ros.replacement_ppg(table)
        self.cache = {}

    def locked(self, team):
        """True once that team's game has kicked off (lineups lock player by player)."""
        k = self.kickoffs.get(team)
        return bool(self.now is not None and k is not None and k <= self.now)

    def empty(self):
        z = np.zeros(self.n)
        return {"key": "empty", "name": "Empty slot", "position": None, "team": None, "opp": None,
                "status": None, "note": "Nobody is starting in this slot.", "s": z, "locked": False, "ros": None,
                "mean": 0.0, "low": 0.0, "high": 0.0}

    def player(self, pid, name, position):
        key = pid or f"x:{name}"
        if key in self.cache:
            return self.cache[key]
        import zlib
        rng = np.random.default_rng(zlib.crc32(str(key).encode()))
        n, note, team, opp, status = self.n, None, None, None, None
        if position in TYPICAL:
            m, sd = TYPICAL[position]
            s = np.clip(rng.normal(m, sd, n), 0, None)
        elif pid in self.preds.index and len(self.recent.get(pid, [])) >= 3:
            pr = self.preds.loc[pid]
            pos, team, opp = pr["position"], pr["team"], pr["opponent_team"]
            status = _nn(pr["injury_status"])
            hist = self.recent[pid]
            w = sim.DECAY ** np.arange(len(hist))[::-1]
            w /= w.sum()
            f = sim.matchup_factor(pr["opp_pts_allowed_L6"], self.pos_avg.get(pos))
            play = float(_nn(pr["play_prob"]) if _nn(pr["play_prob"]) is not None else 1.0)
            draw = rng.choice(hist, size=n, p=w) * rng.normal(1.0, sim.NOISE, n) * f
            s = np.where(rng.random(n) < play, np.maximum(draw, 0), 0.0)
            if play == 0:
                note = "Ruled out this week."
        elif pid in self.table.index:
            t = self.table.loc[pid]
            team, s = t["team"], np.zeros(n)
            note = "On a bye this week." if _nn(t["bye"]) == self.week else "Hasn't played recently, so counted as 0."
        elif position in POS:
            r = self.repl.get(position, 8.0)
            s = np.clip(rng.normal(r, 0.6 * r, n), 0, None)
            note = "Not enough game history, so counted at replacement level."
        else:
            s, note = np.zeros(n), "Not scored by this model, so counted as 0."
        long_term = None
        if pid in self.table.index and _nn(self.table.loc[pid, "ros_pts"]) is not None:
            long_term = float(self.table.loc[pid, "ros_pts"])
        out = {"key": key, "name": name, "position": position, "team": team, "opp": opp,
               "status": status, "note": note, "s": s, "locked": self.locked(team), "ros": long_term,
               "mean": float(s.mean()), "low": float(np.percentile(s, 10)), "high": float(np.percentile(s, 90))}
        self.cache[key] = out
        return out


def resolve(sl_id, sl_players, table, sampler):
    """Sleeper player id -> simulated player."""
    if sl_id in (None, "", "0", 0):
        return sampler.empty()
    p = sl_players.get(str(sl_id)) or {}
    pos = p.get("position")
    name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or str(sl_id)
    gsis = p.get("gsis_id")
    if pos in POS and (not gsis or gsis not in table.index):
        key = sleeper.norm(name)
        hit = table[(table["name"].map(sleeper.norm) == key) & (table["position"] == pos)]
        gsis = hit.index[0] if len(hit) else None
    return sampler.player(gsis if pos in POS else None, name, pos)


def free_agent_pool(sampler, preds, rostered_ids, rostered_names, per_pos=10):
    """The best unrostered players on this week's slate, simulated and ready to test."""
    pool = []
    for r in preds.itertuples():
        if r.position not in POS or r.player_id in rostered_ids or sleeper.norm(r.player_display_name) in rostered_names:
            continue
        if r.injury_status == "Out":
            continue
        p = sampler.player(r.player_id, r.player_display_name, r.position)
        if p["note"] or not p["opp"]:        # no real history, or no game to play: never recommend a guess
            continue
        if not p["locked"]:
            pool.append(p)
    out = []
    for pos in POS:
        out += sorted([p for p in pool if p["position"] == pos], key=lambda p: -p["mean"])[:per_pos]
    return out


def _win(mine, opp):
    return float((mine > opp).mean())


def _row(slot, p):
    return {"slot": slot, "name": p["name"], "position": p["position"], "team": p["team"],
            "opp": p["opp"], "expected": round(p["mean"], 1), "low": round(p["low"], 1),
            "high": round(p["high"], 1), "status": p["status"], "note": p["note"]}


def _why(inn, out, wp):
    d = inn["mean"] - out["mean"]
    up = (inn["high"] - out["high"]) - (inn["low"] - out["low"])
    if d >= 1.0:
        return f"{d:.1f} more projected points"
    if d <= -1.0:
        return f"{abs(d):.1f} fewer projected points, but the bigger upside helps your chances" if up > 0 \
            else f"{abs(d):.1f} fewer projected points, but a steadier floor helps your chances"
    return "about the same points with a higher ceiling" if up > 0 else "about the same points with a steadier floor"


def analyze(my_slots, bench, opp_slots, my_name="You", opp_name="Your opponent", week=None,
            free_agents=None, roster_full=False):
    """my_slots / opp_slots: [(slot label, player)], bench: [player]."""
    mine = sum(p["s"] for _, p in my_slots)
    opp = sum(p["s"] for _, p in opp_slots)
    wp = _win(mine, opp)

    # hill-climb: apply the swap that helps most, up to three times
    cur, pool, total, cur_wp, moves = list(my_slots), list(bench), mine.copy(), wp, []
    for _ in range(3):
        best = None
        for i, (slot, p) in enumerate(cur):
            if slot not in SLOT_OK or p["position"] not in POS or p.get("locked"):
                continue
            for b in pool:
                if b["position"] not in SLOT_OK[slot] or b.get("locked"):
                    continue
                new = total - p["s"] + b["s"]
                gain = _win(new, opp) - cur_wp
                if best is None or gain > best[0]:
                    best = (gain, i, b, new)
        if not best or best[0] < 0.01:
            break
        gain, i, b, new = best
        slot, out_p = cur[i]
        moves.append({"slot": slot, "start": _row(slot, b), "sit": _row(slot, out_p),
                      "gain": round(100 * gain), "why": _why(b, out_p, cur_wp)})
        pool = [x for x in pool if x["key"] != b["key"]] + [out_p]
        cur[i] = (slot, b)
        total, cur_wp = new, cur_wp + gain

    # waiver wire: test each free agent against the lineup you'd have after the bench moves
    pickups = []
    for fa in free_agents or []:
        best = None
        for slot, p in cur:
            if slot not in SLOT_OK or p["position"] not in POS or p.get("locked") or fa["position"] not in SLOT_OK[slot]:
                continue
            gain = _win(total - p["s"] + fa["s"], opp) - cur_wp
            if best is None or gain > best[0]:
                best = (gain, slot, p)
        if best and best[0] >= 0.01:
            gain, slot, p = best
            pickups.append({"add": _row(slot, fa), "slot": slot, "replaces": _row(slot, p), "gain": round(100 * gain),
                            "why": _why(fa, p, cur_wp), "_g": gain})
    pickups = sorted(pickups, key=lambda x: -x["_g"])
    grouped = {}
    for x in pickups:                                   # one best pickup per starter, the rest listed as alternatives
        key = x["replaces"]["name"]
        if key not in grouped:
            grouped[key] = dict(x, also=[])
        else:
            grouped[key]["also"].append({"name": x["add"]["name"], "position": x["add"]["position"],
                                         "team": x["add"]["team"], "gain": x["gain"]})
    pickups = list(grouped.values())[:3]
    for x in pickups:
        x.pop("_g"); x["also"] = x["also"][:2]
    drop = None
    if roster_full and pickups:
        droppable = [b for b in pool if b["position"] in POS and not b.get("locked")]
        if droppable:
            d = min(droppable, key=lambda b: b["ros"] if b["ros"] is not None else b["mean"] * 8)
            drop = {"name": d["name"], "position": d["position"], "team": d["team"], "expected": round(d["mean"], 1)}

    stance = "favorite" if wp >= 0.58 else "underdog" if wp <= 0.42 else "toss-up"
    advice = {"favorite": "You're the favorite, so protecting your floor matters more than chasing upside.",
              "underdog": "You're the underdog, so upside matters more than a safe floor this week.",
              "toss-up": "This one is close to a coin flip, so projected points matter most."}[stance]

    warnings = []
    for slot, p in my_slots:
        if p["note"] and p["position"] in POS + ["K", "DEF"]:
            warnings.append(f"{p['name']} ({slot}): {p['note']}")
    return {
        "week": week, "win_prob": round(wp, 3), "stance": stance, "advice": advice,
        "you": {"name": my_name, "expected": round(float(mine.mean()), 1),
                "low": round(float(np.percentile(mine, 10)), 1), "high": round(float(np.percentile(mine, 90)), 1)},
        "opponent": {"name": opp_name, "expected": round(float(opp.mean()), 1),
                     "low": round(float(np.percentile(opp, 10)), 1), "high": round(float(np.percentile(opp, 90)), 1)},
        "lineup": [_row(s, p) for s, p in my_slots],
        "opp_lineup": [_row(s, p) for s, p in opp_slots],
        "moves": moves, "win_prob_after": round(cur_wp, 3) if moves else None,
        "pickups": pickups, "drop": drop, "roster_full": roster_full,
        "locked": [p["name"] for _, p in my_slots if p.get("locked") and p["position"] in POS],
        "warnings": warnings}
