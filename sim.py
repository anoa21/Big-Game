"""Monte Carlo simulation layer + injury report.

For each player we simulate thousands of possible games by resampling their
recent scores (recent games weighted more), nudging for how generous the
opposing defense is, and zeroing out games where the injury report says they
likely won't play. "Big game chance" is the share of simulated games over the
position's threshold.

PLAY_PROB values are judgment calls, not fitted; tune them if you disagree.
"""
import zlib

import numpy as np
import pandas as pd
import nflreadpy as nfl
import context
import model

PLAY_PROB = {"Out": 0.0, "Doubtful": 0.05, "Questionable": 0.7}
N_SIMS = 5000
DECAY = 0.85          # weight on each older game
NOISE = 0.12          # game-to-game wobble on top of resampling
HIST_N = 8            # games of history used


def simulate(hist, threshold, factor=1.0, play_prob=1.0, n=N_SIMS, rng=None):
    """Return (p_big, p10, p50, p90) for one player."""
    rng = rng or np.random.default_rng(0)
    hist = np.asarray(hist, dtype=float)
    if len(hist) < 3:
        return (np.nan,) * 4
    w = DECAY ** np.arange(len(hist))[::-1]
    w /= w.sum()
    pts = rng.choice(hist, size=n, p=w) * rng.normal(1.0, NOISE, n) * factor
    pts = np.where(rng.random(n) < play_prob, np.maximum(pts, 0), 0.0)
    return ((pts >= threshold).mean(), *np.percentile(pts, [10, 50, 90]))


def matchup_factor(opp_allowed, pos_avg):
    if pd.isna(opp_allowed) or not pos_avg:
        return 1.0
    return float(np.clip((opp_allowed / pos_avg) ** 0.5, 0.85, 1.15))


# ---- Projected stat lines ("props") ------------------------------------
# (stat column, label, thresholds to report). "td" = rushing + receiving TDs.
PROPS = {
    "QB": [("passing_yards", "Passing yards", [225, 275, 325]),
           ("passing_tds", "Passing TDs", [1, 2, 3]),
           ("rushing_yards", "Rushing yards", [20, 40])],
    "RB": [("rushing_yards", "Rushing yards", [50, 75, 100]),
           ("receiving_yards", "Receiving yards", [20, 40, 60]),
           ("receptions", "Receptions", [3, 5]),
           ("td", "Touchdown", [1])],
    "WR": [("receiving_yards", "Receiving yards", [50, 75, 100]),
           ("receptions", "Receptions", [4, 6, 8]),
           ("td", "Touchdown", [1])],
    "TE": [("receiving_yards", "Receiving yards", [30, 50, 70]),
           ("receptions", "Receptions", [3, 5, 7]),
           ("td", "Touchdown", [1])],
}
COUNT_STATS = {"td", "passing_tds"}   # modeled as Poisson, not resampled
RATE_BLEND = 0.7                      # weight on the player's own TD rate
STAT_COLS = ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
             "receiving_yards", "receiving_tds", "receptions"]


def add_td_col(df):
    df = df.copy()
    df["td"] = df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)
    return df


def _poisson_at_least(k, lam):
    """P(X >= k) for a Poisson with mean lam."""
    term, cdf = np.exp(-lam), 0.0
    for i in range(k):
        cdf += term
        term *= lam / (i + 1)
    return 1.0 - cdf


def project_props(g, pos, factor, play, rate_avg, n=4000, rng=None):
    """g: the player's last few games (oldest first) with stat columns.
    Returns [{label, median, lines:[{line, p}]}]; p includes the chance he
    doesn't play, median is for a game he plays."""
    rng = rng or np.random.default_rng(0)
    if pos not in PROPS or len(g) < 3 or play <= 0:
        return []
    w = DECAY ** np.arange(len(g))[::-1]
    w /= w.sum()
    idx = rng.choice(len(g), size=n, p=w)             # same games for every stat
    mult = rng.normal(1.0, NOISE, n) * factor
    plays = rng.random(n) < play
    out = []
    for stat, label, lines in PROPS[pos]:
        col = g[stat].fillna(0).to_numpy(dtype=float)
        if stat in COUNT_STATS:
            own = float((col * w).sum())
            lam = (RATE_BLEND * own + (1 - RATE_BLEND) * rate_avg.get((pos, stat), own)) * factor
            out.append({"label": label, "median": None, "lines": [
                {"line": k, "p": round(play * _poisson_at_least(k, lam), 3)}
                for k in lines]})
            continue
        vals = col[idx] * mult
        if stat == "receptions":
            vals = np.round(vals)
        vals = np.where(plays, np.maximum(vals, 0), 0.0)
        active = vals[plays]
        out.append({"label": label,
                    "median": round(float(np.median(active)), 1) if len(active) else None,
                    "lines": [{"line": k, "p": round(float((vals >= k).mean()), 3)}
                              for k in lines]})
    return out


def td_rate_avg(hist_df):
    """League-average TDs per game by position (the prior for TD props)."""
    d = add_td_col(hist_df)
    return {(p, s): float(d.loc[d["position"] == p, s].mean())
            for p in PROPS for s in COUNT_STATS if s in d}


def defense_table(hist_df, last_n=6):
    """Points each defense allowed per game to each position over its last
    `last_n` games, ranked across all teams (rank 1 = allows the MOST)."""
    allowed = (hist_df[hist_df["fantasy_points_ppr"].notna()]
               .groupby(["opponent_team", "position", "season", "week"])
               ["fantasy_points_ppr"].sum().reset_index()
               .sort_values(["season", "week"]))
    t = (allowed.groupby(["opponent_team", "position"])["fantasy_points_ppr"]
         .apply(lambda x: x.tail(last_n).mean()).reset_index()
         .rename(columns={"fantasy_points_ppr": "allowed"}))
    t["rank"] = t.groupby("position")["allowed"].rank(ascending=False, method="min").astype(int)
    return {(r.opponent_team, r.position): (int(r.rank), float(r.allowed))
            for r in t.itertuples()}


def injury_report(season, week):
    """gsis_id -> (status, injury) for players on the official report."""
    try:
        inj = nfl.load_injuries([season]).to_pandas()
    except Exception:
        return {}
    inj = inj[(inj["week"] == week) & (inj["game_type"] == "REG")
              & inj["report_status"].isin(PLAY_PROB)]
    return {r.gsis_id: (r.report_status, r.report_primary_injury)
            for r in inj.itertuples()}


def enrich(preds, hist_df, season, week, blend=0.5, with_props=True, weather=True):
    """Add simulation odds, point range and injury status to model output."""
    inj = injury_report(season, week)
    hist_df = hist_df.sort_values(["season", "week"])
    recent = (hist_df.groupby("player_id")["fantasy_points_ppr"]
              .apply(lambda s: s.tail(HIST_N).tolist()).to_dict())
    pos_avg = preds.groupby("position")["opp_pts_allowed_L6"].mean().to_dict()
    tail = add_td_col(hist_df).groupby("player_id").tail(HIST_N)
    games = {pid: g for pid, g in tail.groupby("player_id")}
    rate_avg = td_rate_avg(hist_df)
    dtable = defense_table(hist_df)
    try:
        ctx = context.game_context(season, week, weather=weather)
    except Exception:
        ctx = {}                              # no schedule context: odds just stay as they were
    rows = []
    for r in preds.itertuples():
        status, injury = inj.get(r.player_id, (None, None))
        play = PLAY_PROB.get(status, 1.0)
        f = matchup_factor(r.opp_pts_allowed_L6, pos_avg.get(r.position))
        p_sim, p10, p50, p90 = simulate(
            recent.get(r.player_id, []), model.BIG_GAME[r.position], f, play,
            rng=np.random.default_rng(zlib.crc32(r.player_id.encode())))
        p_model = r.big_game_prob * play
        final = p_model if np.isnan(p_sim) else blend * p_model + (1 - blend) * p_sim
        shift, wind_part, ctx_notes = context.adjust(r.position, r.team, ctx)
        ctx_pts, ctx_line = 0.0, None
        if play > 0 and shift and 0 < final < 1:
            adj = 1 / (1 + np.exp(-(np.log(final / (1 - final)) + shift)))
            ctx_pts = round(100 * (adj - final), 1)
            final = adj
            if abs(ctx_pts) >= 2.0:                      # only call it out on the row when it matters
                ctx_line = f"Game context: {context.label(shift, wind_part, ctx.get(r.team))}, {ctx_pts:+.1f} points to his chance"
        props = (project_props(games[r.player_id], r.position, f, play, rate_avg,
                               rng=np.random.default_rng(zlib.crc32(r.player_id.encode())))
                 if with_props and r.player_id in games else [])
        d_rank, d_allowed = dtable.get((r.opponent_team, r.position), (None, None))
        rows.append((p_sim, p10, p50, p90, status, injury, play, final, props,
                     d_rank, d_allowed, ctx_pts, ctx_line, ctx_notes))
    out = preds.copy()
    out[["p_sim", "floor", "median", "ceiling", "injury_status",
         "injury", "play_prob", "final_prob", "props",
         "def_rank", "def_allowed", "ctx_pts", "ctx_line", "ctx_notes"]] = pd.DataFrame(rows, index=out.index)
    return out.sort_values("final_prob", ascending=False).reset_index(drop=True)


def backtest(seasons, blends=(0, 0.25, 0.5, 0.75, 1.0)):
    """Compare model-only, simulation-only and blends on the last season.
    (No injuries here: historical reports weren't saved as-of each week.)"""
    from sklearn.metrics import roc_auc_score
    df = model._mark_home(model.load_stats(seasons), seasons)
    df = model.add_labels(model.build_features(df))
    df = df.sort_values(["player_id", "season", "week"])
    df["hist"] = df.groupby("player_id")["fantasy_points_ppr"].transform(
        lambda s: pd.Series([s.iloc[max(0, i - HIST_N):i].tolist()
                             for i in range(len(s))], index=s.index))
    df = df[df["games_played"] >= 3]
    test_season = max(seasons)
    tr, te = df[df["season"] < test_season], df[df["season"] == test_season].copy()
    te["p_model"] = model._make_clf().fit(
        tr[model.FEATURES], tr["big_game"]).predict_proba(te[model.FEATURES])[:, 1]
    avg = tr.groupby("position")["opp_pts_allowed_L6"].mean().to_dict()
    sims = [simulate(r.hist, model.BIG_GAME[r.position],
                     matchup_factor(r.opp_pts_allowed_L6, avg.get(r.position)),
                     rng=np.random.default_rng(i))[0]
            for i, r in enumerate(te.itertuples())]
    te["p_sim"] = sims
    for b in blends:
        p = b * te["p_model"] + (1 - b) * te["p_sim"]
        top = te.assign(p=p).nlargest(int(len(te) * 0.05), "p")["big_game"].mean()
        print(f"blend={b:.2f} (1=model only, 0=sim only): "
              f"AUC {roc_auc_score(te['big_game'], p):.3f}  top-5% hit rate {top:.1%}")


def backtest_props(seasons):
    """Calibration check: when we say X% to reach a line, how often did it
    happen in the last season? (No injuries; players assumed to play.)"""
    df = model._mark_home(model.load_stats(seasons), seasons)
    df = add_td_col(model.build_features(df)).sort_values(["player_id", "season", "week"])
    test_season = max(seasons)
    train = df[df["season"] < test_season]
    avg = train.groupby("position")["opp_pts_allowed_L6"].mean().to_dict()
    rate_avg = td_rate_avg(train)
    rec = []
    for pid, g in df.groupby("player_id"):
        g = g.reset_index(drop=True)
        for i in range(3, len(g)):
            row = g.iloc[i]
            if row["season"] != test_season:
                continue
            f = matchup_factor(row["opp_pts_allowed_L6"], avg.get(row["position"]))
            props = project_props(g.iloc[max(0, i - HIST_N):i], row["position"], f, 1.0,
                                  rate_avg, rng=np.random.default_rng(i))
            for p in props:
                stat = next(x[0] for x in PROPS[row["position"]] if x[1] == p["label"])
                for ln in p["lines"]:
                    rec.append((row["position"], p["label"], ln["line"], ln["p"],
                                float(row[stat] >= ln["line"])))
    r = pd.DataFrame(rec, columns=["pos", "stat", "line", "pred", "hit"])
    t = r.groupby(["pos", "stat", "line"]).agg(n=("hit", "size"),
        predicted=("pred", "mean"), actual=("hit", "mean")).round(3)
    print(t.to_string())


if __name__ == "__main__":
    import sys
    backtest_props([2022, 2023, 2024]) if "props" in sys.argv else backtest([2022, 2023, 2024])
