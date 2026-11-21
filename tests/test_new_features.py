import os, sys
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "."]
import numpy as np
import pandas as pd
import features as F
from odds import market_features, MARKET_FEATURE_NAMES
from config import FEATURES_SELECTED

NAMES = F.FEATURE_NAMES
IDX = {n: i for i, n in enumerate(NAMES)}


def ts(day):
    return pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=day)


def vec(rs, home, away, day, comp="1"):
    start = IDX["home_pr_att"]
    return dict(zip(NAMES[start:start + 36], rs.features(home, away, comp, ts(day))))


def test_all_selected_features_exist_and_no_duplicates():
    assert len(set(FEATURES_SELECTED)) == len(FEATURES_SELECTED)
    missing = [n for n in FEATURES_SELECTED if n not in F.FEATURE_NAMES_FULL]
    assert not missing, missing
    print("sélection :", len(FEATURES_SELECTED), "sur", F.EXPECTED_FEATURE_COUNT_FULL)


def test_feature_vector_length_matches_names():
    rs = F.OnlineRatings()
    assert len(rs.features("a", "b", "1", ts(0))) == 36
    assert len(MARKET_FEATURE_NAMES) == 8
    assert F.EXPECTED_FEATURE_COUNT == len(NAMES)


def test_congestion_streaks_and_season_ppg():
    rs = F.OnlineRatings()
    for d, (gh, ga) in enumerate([(2, 0), (1, 1), (0, 1)]):
        rs.update("A", f"X{d}", gh, ga, "1", ts(d * 5), 1.2, 0.8)
    out = vec(rs, "A", "B", 16)
    assert out["home_games_14d"] == 2.0
    assert out["home_unbeaten"] == 0.0 and out["home_winless"] == 2.0
    assert abs(out["home_season_ppg"] - (3 + 1 + 0) / 3) < 1e-9
    assert np.isnan(out["away_season_ppg"])
    late = vec(rs, "A", "B", 400)
    assert np.isnan(late["home_season_ppg"]) and late["home_games_14d"] == 0.0


def test_xg_ewm_and_overperformance():
    rs = F.OnlineRatings()
    rs.update("A", "B", 3, 0, "1", ts(0), 1.0, 1.0)
    out = vec(rs, "A", "B", 7)
    assert abs(out["home_ewm_xgf"] - 1.0) < 1e-9 and abs(out["home_xg_overperf"] - 2.0) < 1e-9
    assert abs(out["away_xg_overperf"] - (-1.0)) < 1e-9
    rs2 = F.OnlineRatings()
    rs2.update("A", "B", 3, 0, "1", ts(0))
    assert np.isnan(vec(rs2, "A", "B", 7)["home_ewm_xgf"])


def test_comp_level_features_start_from_priors():
    rs = F.OnlineRatings()
    out = vec(rs, "A", "B", 0)
    assert abs(out["comp_avg_goals"] - 2.6) < 1e-9 and abs(out["comp_home_win_rate"] - 0.45) < 1e-9


def test_market_features_values_and_missing():
    f = dict(zip(MARKET_FEATURE_NAMES, market_features({"home": 1.5, "draw": 4.0, "away": 7.0, "over_2_5": 1.8, "under_2_5": 2.0, "btts_yes": 1.7, "btts_no": 2.1})))
    assert abs(f["mkt_p_home"] + f["mkt_p_draw"] + f["mkt_p_away"] - 1.0) < 1e-9
    assert f["mkt_p_home"] > 0.6 and 1.0 < f["mkt_overround"] < 1.3 and f["mkt_log_home_away"] > 0
    assert 0.4 < f["mkt_p_over_2_5"] < 0.6 and np.isnan(f["mkt_p_over_1_5"])
    assert all(np.isnan(v) for v in market_features(None)) and all(np.isnan(v) for v in market_features({}))
    assert all(np.isnan(v) for v in market_features(float("nan")))
