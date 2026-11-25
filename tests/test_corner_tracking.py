import os, sys
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "tests", "."]
import features as F
import test_ratings as TR


def rows_with_corners(n=900):
    rows, _ = TR.world(n, comps=1, T=10)
    for i, r in enumerate(rows):
        r["home_corner_kicks"] = 5 + (i * 7) % 5
        r["away_corner_kicks"] = 4 + (i * 3) % 4
    return rows


def test_corner_stats_reach_the_tracker_and_forecast_is_available():
    rows = rows_with_corners()
    for r in rows:
        r["stat_home_corner_kicks"] = r["home_corner_kicks"]
        r["stat_away_corner_kicks"] = r["away_corner_kicks"]
    import api
    transformed = [api.transform_bzzoiro_match(dict(r, home_team=r["homeTeam"]["name"], away_team=r["awayTeam"]["name"],
                   league_id=1, event_date=r["utcDate"], home_team_id=r["homeTeam"]["id"], away_team_id=r["awayTeam"]["id"],
                   home_score=r["score"]["fullTime"]["home"], away_score=r["score"]["fullTime"]["away"])) for r in rows]
    for t, r in zip(transformed, rows):
        t["home_corner_kicks"], t["away_corner_kicks"] = r["home_corner_kicks"], r["away_corner_kicks"]
    df = F.process_matches(transformed, allowed_statuses=("FINISHED",))
    assert df["home_corner_kicks"].notna().mean() > 0.9
    tracker, _ = F.build_tracking_state(df)
    first = df.iloc[0]
    forecast = tracker.corner_forecast(first["home_key"], first["away_key"], first["competition_code"])
    assert forecast["available"] and 6 < forecast["total"] < 14
