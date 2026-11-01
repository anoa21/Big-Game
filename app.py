"""CLI: rank your Sleeper roster by big-game probability for a given week.

  python app.py backtest
  python app.py top --season 2025 --week 9                  # league-wide top 25
  python app.py roster --username YOU --season 2025 --week 9
"""
import argparse
import pandas as pd
import model
import sleeper

pd.set_option("display.width", 140)


def seasons_for(season):
    return [season - 2, season - 1, season]


def show(df):
    d = df.copy()
    d["big_game_prob"] = (d["big_game_prob"] * 100).round(1).astype(str) + "%"
    cols = ["player_display_name", "position", "team", "opponent_team",
            "big_game_prob", "fantasy_points_ppr_L3", "opp_pts_allowed_L6"]
    print(d[cols].round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "top", "roster"])
    ap.add_argument("--username")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int, default=9)
    a = ap.parse_args()

    if a.cmd == "backtest":
        model.backtest([a.season - 2, a.season - 1, a.season])
        return

    preds = model.predict_week(seasons_for(a.season), a.season, a.week)

    if a.cmd == "top":
        show(preds.head(25))
        return

    uid = sleeper.get_user_id(a.username)
    leagues = sleeper.get_leagues(uid, a.season)
    players = sleeper.get_players()
    for lg in leagues:
        roster = sleeper.get_my_roster(lg["league_id"], uid)
        mine = sleeper.roster_names(roster, players)
        ids = {m[0] for m in mine if m[0]}
        names = {sleeper.norm(m[1]) for m in mine}
        sub = preds[preds["player_id"].isin(ids) |
                    preds["player_display_name"].map(sleeper.norm).isin(names)]
        print(f"\n=== {lg['name']} ===")
        show(sub)


if __name__ == "__main__":
    main()
