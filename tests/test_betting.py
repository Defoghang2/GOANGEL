import sys
sys.path[:0] = ["."]
import numpy as np
from betting import build_candidates, recommend
from corners import LINES


def corners_info(total=10.2):
    lines = {l: {"p_over": float(np.clip(0.5 + (total - l) * 0.12, 0.05, 0.95)), "p_fast": 0.5, "p_slow": 0.5} for l in LINES}
    return {"available": True, "home": total * 0.55, "away": total * 0.45, "total": total, "alpha": 0.1, "lines": lines}


def cands(p, o15=.75, o25=.55, btts=.55, ht=None, scores=None, corners=None):
    return build_candidates(p, "H", "A", o15, o25, btts, ht_hda=ht, ht_over05=.7, ht_over15=.3, top_scores=scores, corners=corners)


def test_no_three_and_a_half_goal_markets_anymore():
    keys = [c["key"] for c in cands([.4, .3, .3], corners=corners_info())]
    assert not any("3_5" in k for k in keys)
    assert "over_2_5" in keys and "over_1_5" in keys


def test_corner_markets_replace_them_with_their_own_probabilities():
    cs = cands([.4, .3, .3], corners=corners_info(10.2))
    corner = {c["key"]: c for c in cs if c["kind"] == "corners"}
    assert set(corner) == {f"corners_{s}_{l}" for s in ("over", "under") for l in ("7_5", "8_5", "9_5", "10_5", "11_5", "12_5")}
    assert abs(corner["corners_over_9_5"]["p_model"] + corner["corners_under_9_5"]["p_model"] - 1) < 1e-9
    assert corner["corners_over_9_5"]["p_model"] > corner["corners_over_11_5"]["p_model"]


def test_corners_absent_when_the_model_has_no_data():
    assert not any(c["kind"] == "corners" for c in cands([.4, .3, .3], corners={"available": False}))
    assert not any(c["kind"] == "corners" for c in cands([.4, .3, .3], corners=None))


def test_correct_score_candidates_use_their_odds():
    p = [0.40, 0.28, 0.32]
    scores = [("1-1", 0.12), ("1-0", 0.10), ("0-1", 0.09)]
    r = recommend(cands(p, scores=scores), p, {"cs_1_1": 7.5, "cs_1_0": 9.0, "home": 2.4, "draw": 3.3, "away": 3.1})
    assert r["n_priced"] == 5
    assert any(c["key"] == "cs_1_1" for c in cands(p, scores=scores))


def test_corner_odds_are_priced_and_can_be_selected():
    p = [0.40, 0.28, 0.32]
    odds = {"corners_over_9_5": 1.60, "corners_under_9_5": 2.45, "corners_over_10_5": 2.1, "corners_under_10_5": 1.75}
    r = recommend(cands(p, corners=corners_info(11.5)), p, odds)
    assert r["n_priced"] >= 4


def test_always_returns_a_pick_and_no_alternatives_key():
    for p in ([0.28, 0.25, 0.47], [0.34, 0.32, 0.34], [0.62, 0.22, 0.16]):
        r = recommend(cands(p), p, None)
        assert r["pick"]["name"] and r["n_priced"] == 0 and "alternatives" not in r


def test_small_odds_are_used_not_ignored():
    p = [0.70, 0.19, 0.11]
    odds = {"home": 1.30, "draw": 5.5, "away": 9.0, "1x": 1.08, "over_1_5": 1.18, "under_1_5": 4.6}
    r = recommend(cands(p, o15=.82), p, odds)
    assert r["pick"]["price"] is not None and r["pick"]["price"] < 1.5 and r["n_priced"] >= 4


def test_value_bet_detected_with_ev_and_kelly():
    p = [0.60, 0.23, 0.17]
    r = recommend(cands(p), p, {"home": 2.10, "draw": 3.4, "away": 4.4})
    pk = r["pick"]
    assert pk["tier"] == 0 and pk["ev"] >= 0.04 and pk["kelly"] > 0


def test_longshot_not_recommended_as_value():
    p = [0.159, 0.186, 0.655]
    r = recommend(cands(p), p, {"home": 9.30, "draw": 5.6, "away": 1.22})
    assert r["pick"]["key"] != "home"


def test_half_time_markets_priced_when_odds_exist():
    p = [0.40, 0.30, 0.30]
    ht = [0.30, 0.45, 0.25]
    r = recommend(cands(p, ht=ht), p, {"ht_draw": 2.4, "ht_home": 3.4, "ht_away": 4.2}, ht_hda=ht)
    assert r["n_priced"] == 3
