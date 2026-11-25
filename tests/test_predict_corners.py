import os, sys, io, contextlib
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "tests", "."]
import numpy as np
import pandas as pd
import api
import features as F
import predict as P
from goangel_core import GoPreprocessor
import test_ratings as TR
from test_printer import check_structure


class Fixed:
    classes_ = np.array([0, 1, 2])

    def __init__(self, p):
        self.p = np.array(p, dtype=float)

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))

    def predict_proba_1x2(self, home, away):
        return self.p


def history_with_corners():
    rows, _ = TR.world(900, comps=1, T=10)
    out = []
    for i, r in enumerate(rows):
        raw = dict(r, home_team=r["homeTeam"]["name"], away_team=r["awayTeam"]["name"], league_id=1,
                   event_date=r["utcDate"], home_team_id=r["homeTeam"]["id"], away_team_id=r["awayTeam"]["id"],
                   home_score=r["score"]["fullTime"]["home"], away_score=r["score"]["fullTime"]["away"])
        t = api.transform_bzzoiro_match(raw)
        t["home_corner_kicks"], t["away_corner_kicks"] = 5 + (i * 7) % 5, 4 + (i * 3) % 4
        out.append(t)
    return F.process_matches(out, allowed_statuses=("FINISHED",))


def run_prediction(odds):
    df = history_with_corners()
    tracker, elo = F.build_tracking_state(df)
    first = df.iloc[0]
    rng = np.random.default_rng(0)
    pipeline = GoPreprocessor(5.0, selected_idx=list(range(20))).fit(rng.normal(size=(200, F.EXPECTED_FEATURE_COUNT_FULL)))
    ml = {"A": Fixed([0.52, 0.26, 0.22]), "B": Fixed([0.50, 0.27, 0.23]), "C": Fixed([0.54, 0.25, 0.21])}
    pb = [("P1", Fixed([0.51, 0.26, 0.23])), ("P2", Fixed([0.53, 0.25, 0.22]))]
    weights = {"A": .2, "B": .2, "C": .2, "P1": .2, "P2": .2, "__calibration_logloss__": 1.0}
    markets = P.predict_match(first["home_key"], first["away_key"], first["home_raw"], first["away_raw"], first["competition_code"],
                              ml, pb, pipeline, df, tracker, elo, weights, pd.Timestamp("2027-01-01", tz="UTC"), 0.45, None,
                              bookmaker_odds=odds)
    return first, markets


def render(first, markets):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P._print_match_prediction(pd.Series({"home_raw": first["home_raw"], "away_raw": first["away_raw"]}), markets, None, None)
    return [l for l in buf.getvalue().split("\n") if l.strip()]


def test_predict_match_exposes_corners_and_per_model_probabilities():
    first, m = run_prediction({})
    assert m["corners"]["available"] and len(m["model_hda"]) == 5
    assert set(m["pb_probs"]) == {"over_1_5", "over_2_5", "btts_yes"} and "over35" not in m


def test_full_display_with_corner_and_correct_score_odds():
    odds = {"home": 1.9, "draw": 3.6, "away": 4.4, "over_2_5": 1.9, "under_2_5": 1.95, "cs_1_1": 6.5, "cs_1_0": 7.0,
            "corners_over_9_5": 1.85, "corners_under_9_5": 1.95, "corners_over_10_5": 2.2, "corners_under_10_5": 1.65}
    first, m = run_prediction(odds)
    lines = render(first, m)
    check_structure(lines, first["home_raw"].upper(), first["away_raw"].upper())
    corner_line = [l for l in lines if l.startswith("🔔 CORNERS")][0]
    assert "attendus" in corner_line and "cote" in corner_line
    print("\n".join(lines))
