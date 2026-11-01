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


def enrich(preds, hist_df, season, week, blend=0.5):
    """Add simulation odds, point range and injury status to model output."""
    inj = injury_report(season, week)
    hist_df = hist_df.sort_values(["season", "week"])
    recent = (hist_df.groupby("player_id")["fantasy_points_ppr"]
              .apply(lambda s: s.tail(HIST_N).tolist()).to_dict())
    pos_avg = preds.groupby("position")["opp_pts_allowed_L6"].mean().to_dict()
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
        rows.append((p_sim, p10, p50, p90, status, injury, play, final))
    out = preds.copy()
    out[["p_sim", "floor", "median", "ceiling", "injury_status",
         "injury", "play_prob", "final_prob"]] = pd.DataFrame(rows, index=out.index)
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


if __name__ == "__main__":
    backtest([2022, 2023, 2024])
