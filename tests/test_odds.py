import sys
sys.path[:0] = ["."]
from odds import extract_markets, sample_paths

def test_flat_fields_with_small_prices():
    m = extract_markets({"odds_home": 1.22, "odds_draw": 6.5, "odds_away": 11.0, "odds_over_25": 1.45, "odds_under_25": 2.7, "odds_btts_yes": 1.5, "odds_btts_no": 2.4})
    assert m["home"] == 1.22 and m["draw"] == 6.5 and m["away"] == 11.0
    assert m["over_2_5"] == 1.45 and m["under_2_5"] == 2.7 and m["btts_yes"] == 1.5 and m["btts_no"] == 2.4

def test_nested_market_blocks():
    m = extract_markets({"odds": {"home_win": 1.3, "draw": 5.0, "away_win": 9.0, "over_15_goals": 1.15, "under_15_goals": 5.2,
                                  "home_win_or_draw": 1.08, "away_win_or_draw": 3.1}})
    assert m["home"] == 1.3 and m["1x"] == 1.08 and m["x2"] == 3.1 and m["over_1_5"] == 1.15

def test_bookmaker_dict_takes_best_price():
    m = extract_markets({"markets": {"home_win": {"bet365": 1.5, "pinnacle": 1.57}, "draw": {"bet365": 4.0}, "away_win": {"bet365": 6.0}}})
    assert m["home"] == 1.57 and m["draw"] == 4.0

def test_list_of_selections():
    m = extract_markets({"markets": [{"market": "Match Winner", "selection": "Home", "odds": 1.9},
                                      {"market": "Match Winner", "selection": "Draw", "odds": 3.4},
                                      {"market": "Match Winner", "selection": "Away", "odds": 4.2}]})
    assert m == {"home": 1.9, "draw": 3.4, "away": 4.2}

def test_half_time_not_confused_with_full_time():
    m = extract_markets({"odds_home": 1.8, "ht_home": 2.6, "ht_draw": 2.1, "ht_over_0_5": 1.35, "first_half_over_15_goals": 3.1})
    assert m["home"] == 1.8 and m["ht_home"] == 2.6 and m["ht_draw"] == 2.1 and m["ht_over_0_5"] == 1.35 and m["ht_over_1_5"] == 3.1
    assert "draw" not in m

def test_double_chance_not_taken_for_draw():
    m = extract_markets({"home_draw": 1.12, "away_draw": 2.0})
    assert "draw" not in m and m["1x"] == 1.12

def test_totals_by_line_and_garbage_ignored():
    m = extract_markets({"totals": {"1.5": {"over": 1.3, "under": 3.4}, "2.5": {"over": 1.8, "under": 2.0}}, "id": 12345, "note": "x", "score": 0.5})
    assert m["over_2_5"] == 1.8 and m["under_1_5"] == 3.4 and len(m) == 4

def test_string_prices_and_empty():
    assert extract_markets({"odds_home": "1.25"})["home"] == 1.25
    assert extract_markets(None, {}, []) == {}
    assert sample_paths({"a": {"b": 1.5}}) == ["a_b=1.5"]


def test_correct_score_odds_from_nested_flat_and_list_shapes():
    nested = extract_markets({"correct_score": {"1-0": 7.5, "2-1": 9.0, "0-0": 11.0}})
    assert nested == {"cs_1_0": 7.5, "cs_2_1": 9.0, "cs_0_0": 11.0}
    flat = extract_markets({"odds_correct_score_3_2": 41.0, "cs_1_1": 6.5})
    assert flat == {"cs_3_2": 41.0, "cs_1_1": 6.5}
    listed = extract_markets({"markets": [{"market": "Correct Score", "selection": "2:0", "odds": 8.0},
                                           {"market": "Correct Score", "selection": "1:2", "odds": 13.0}]})
    assert listed == {"cs_2_0": 8.0, "cs_1_2": 13.0}


def test_half_time_score_not_taken_for_full_time_score():
    assert extract_markets({"ht_correct_score_1_0": 4.0}) == {}


def test_corner_odds_all_naming_styles():
    m = extract_markets({"corners": {"over_9_5": 1.85, "under_9_5": 1.95}, "odds_over_105_corners": 2.4,
                         "total_corners_under_8_5": 2.6, "corner_over_11_5": 3.9})
    assert m["corners_over_9_5"] == 1.85 and m["corners_under_9_5"] == 1.95
    assert m["corners_over_10_5"] == 2.4 and m["corners_under_8_5"] == 2.6 and m["corners_over_11_5"] == 3.9


def test_goal_lines_and_corner_lines_never_mixed():
    m = extract_markets({"over_2_5": 1.9, "corners_over_2_5": 7.7, "over_9_5_corners": 1.8})
    assert m["over_2_5"] == 1.9 and m["corners_over_9_5"] == 1.8
