"""Big-game probability model built on nflverse weekly stats.

A "big game" is a position-specific PPR threshold (see BIG_GAME). Features use
only information available BEFORE the game (everything is shifted by one game),
so backtests are not leaking the future.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

POSITIONS = ["QB", "RB", "WR", "TE"]
BIG_GAME = {"QB": 25, "RB": 18, "WR": 18, "TE": 14}  # PPR points; tune to taste

USAGE = ["fantasy_points_ppr", "targets", "carries", "target_share",
         "wopr", "air_yards_share", "receiving_yards", "rushing_yards"]
FEATURES = (
    [f"{c}_L3" for c in USAGE] + [f"{c}_L8" for c in USAGE]
    + ["games_played", "opp_pts_allowed_L6", "is_home", "pos_code"]
)


def load_stats(seasons):
    df = nfl.load_player_stats(seasons).to_pandas()
    df = df[(df["season_type"] == "REG") & df["position"].isin(POSITIONS)].copy()
    return df.sort_values(["season", "week"]).reset_index(drop=True)


def _add_upcoming_rows(df, season, week, lookback=4):
    """Append empty rows for `week` for players seen recently, so the same
    shifted-rolling code produces their pre-game features."""
    sched = nfl.load_schedules([season]).to_pandas()
    sched = sched[(sched["week"] == week) & (sched["game_type"] == "REG")]
    opp = {}
    for _, g in sched.iterrows():
        opp[g["home_team"]] = (g["away_team"], 1)
        opp[g["away_team"]] = (g["home_team"], 0)

    recent = df[(df["season"] == season) & (df["week"] >= week - lookback)]
    last = recent.sort_values("week").groupby("player_id").tail(1)
    rows = []
    for _, r in last.iterrows():
        if r["team"] in opp:
            o, home = opp[r["team"]]
            rows.append({"player_id": r["player_id"],
                         "player_display_name": r["player_display_name"],
                         "position": r["position"], "team": r["team"],
                         "opponent_team": o, "season": season, "week": week,
                         "is_home": home, "upcoming": True})
    new = pd.DataFrame(rows)
    return pd.concat([df.assign(upcoming=False), new], ignore_index=True)


def build_features(df):
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    # Per-player rolling form, shifted so the current game is excluded.
    g = df.groupby("player_id", group_keys=False)
    for c in USAGE:
        s = g[c].shift(1)
        df[f"{c}_L3"] = s.groupby(df["player_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{c}_L8"] = s.groupby(df["player_id"]).transform(
            lambda x: x.rolling(8, min_periods=1).mean())
    df["games_played"] = g.cumcount()

    # Opponent defense vs position: fantasy points allowed per game, last 6.
    df["_t"] = df["season"] * 100 + df["week"]
    allowed = (df[df["fantasy_points_ppr"].notna()]
               .groupby(["opponent_team", "position", "_t"])
               ["fantasy_points_ppr"].sum().reset_index()
               .sort_values("_t"))
    allowed["opp_pts_allowed_L6"] = (
        allowed.groupby(["opponent_team", "position"])["fantasy_points_ppr"]
        .transform(lambda x: x.rolling(6, min_periods=2).mean()))
    # As-of join: most recent value strictly BEFORE the row's week.
    df = df.sort_values("_t")
    df = pd.merge_asof(df, allowed[["opponent_team", "position", "_t",
                                    "opp_pts_allowed_L6"]],
                       on="_t", by=["opponent_team", "position"],
                       allow_exact_matches=False).drop(columns="_t")
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["is_home"] = df["is_home"].fillna(0)
    df["pos_code"] = df["position"].map({p: i for i, p in enumerate(POSITIONS)})
    return df


def add_labels(df):
    thr = df["position"].map(BIG_GAME)
    df["big_game"] = (df["fantasy_points_ppr"] >= thr).astype(int)
    return df


def _mark_home(df, seasons):
    sched = nfl.load_schedules(seasons).to_pandas()
    home = sched[["season", "week", "home_team"]].assign(is_home_g=1)
    home = home.rename(columns={"home_team": "team"})
    out = df.merge(home, on=["season", "week", "team"], how="left")
    if "is_home" in out:
        out["is_home"] = out["is_home"].fillna(out["is_home_g"]).fillna(0)
    else:
        out["is_home"] = out["is_home_g"].fillna(0)
    return out.drop(columns="is_home_g")


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=4, l2_regularization=1.0,
                                          random_state=0)


def backtest(seasons):
    """Train on all but the last season, test on the last. Compares the model
    to a naive 'recent average points' baseline."""
    df = _mark_home(load_stats(seasons), seasons)
    df = add_labels(build_features(df))
    df = df[df["games_played"] >= 3]
    test_season = max(seasons)
    tr, te = df[df["season"] < test_season], df[df["season"] == test_season]
    clf = _make_clf().fit(tr[FEATURES], tr["big_game"])
    te = te.assign(p=clf.predict_proba(te[FEATURES])[:, 1])
    print(f"Train rows {len(tr)}, test rows {len(te)}, "
          f"base rate {te['big_game'].mean():.1%}")
    print(f"AUC model:    {roc_auc_score(te['big_game'], te['p']):.3f}")
    print(f"AUC baseline: {roc_auc_score(te['big_game'], te['fantasy_points_ppr_L8']):.3f}"
          "  (L8 avg points)")
    top = te.sort_values("p", ascending=False).head(int(len(te) * 0.05))
    print(f"Top 5% picks hit rate: {top['big_game'].mean():.1%}")
    return clf


def predict_week(seasons, season, week, as_of_only=True, stats=None):
    """Return big-game probabilities for every active skill player in `week`."""
    df = load_stats(seasons) if stats is None else stats
    if as_of_only:                       # don't peek at the week being predicted
        df = df[(df["season"] < season) | (df["week"] < week)]
    train = _mark_home(df, seasons)
    train = add_labels(build_features(train))
    train = train[train["games_played"] >= 3]
    clf = _make_clf().fit(train[FEATURES], train["big_game"])

    full = _add_upcoming_rows(df, season, week)
    full = build_features(full)
    up = full[full["upcoming"]].copy()
    up["big_game_prob"] = clf.predict_proba(up[FEATURES])[:, 1]
    return up[["player_id", "player_display_name", "position", "team",
               "opponent_team", "big_game_prob", "fantasy_points_ppr_L3",
               "fantasy_points_ppr_L8", "opp_pts_allowed_L6"]] \
        .sort_values("big_game_prob", ascending=False).reset_index(drop=True)


def predict_with_history(seasons, season, week):
    """predict_week plus the raw stats frame (needed by the simulator)."""
    stats = load_stats(seasons)
    return predict_week(seasons, season, week, stats=stats), \
        stats[(stats["season"] < season) | (stats["week"] < week)]
