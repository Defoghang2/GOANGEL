import os, sys
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "."]
import api
import features as F


def raw_event(**extra):
    base = {"id": 7, "league_id": 1, "status": "finished", "event_date": "2025-09-01T18:00:00+00:00",
            "home_team": "A", "away_team": "B", "home_team_id": 1, "away_team_id": 2,
            "home_score": 2, "away_score": 1, "home_score_ht": 1, "away_score_ht": 0}
    base.update(extra)
    return base


def test_transform_reads_odds_fields_from_the_event():
    out = api.transform_bzzoiro_match(raw_event(odds_home=1.5, odds_draw=4.2, odds_away=6.5, odds_over_25=1.7))
    assert out["odds_markets"]["home"] == 1.5 and out["odds_markets"]["over_2_5"] == 1.7


def test_historical_enrichment_no_longer_calls_the_odds_endpoint():
    calls = []
    api.get_event_odds = lambda eid: calls.append(eid) or {"odds": {"home_win": 1.5}}
    api.get_event_stats = lambda eid: None
    raw = raw_event()
    out = api._enrich_single(raw)
    assert calls == [] and out.get("odds_markets") is None


def test_odds_reach_the_dataframe_and_features():
    transformed = api.transform_bzzoiro_match(raw_event(odds_home=1.4, odds_draw=4.8, odds_away=7.5))
    df = F.process_matches([transformed], allowed_statuses=("FINISHED",))
    assert df.iloc[0]["odds_markets"]["home"] == 1.4
    X, *_ = F.create_features_with_context(df, strict_competition=True)
    names = F.FEATURE_NAMES
    assert X[0, names.index("mkt_p_home")] > 0.6
