import sys
sys.path[:0] = ["."]
from betting import assess, STATUS_RELIABLE, STATUS_WEAK, STATUS_NONE

ALLOWED = ("Value Bet à faible gain", "Value Bet à haut gain", "EV trop négatif : vous perdrez à long terme",
           "EV très intéressant mais risqué", "Les modèles ne sont pas d'accord pour ce type de pari", "Probable",
           "Statistiquement favorable", "Type de pari rentable, mais match piège", "Les modèles sont en conflit sur ce match",
           "La marge des bookmakers réduit fortement les gains",
           "Les bookmakers ont baissé la cote parce que les mises dépassent leur limite actuelle pour ce pari")
DATA_REASONS = ("données insuffisantes", "cote indisponible", "avantage statistique insuffisant")


def pick(ev=0.06, p=0.58, key="home", kind="result", price=2.0, gap=0.03, p_model=None):
    return {"key": key, "kind": kind, "p": p, "p_model": p if p_model is None else p_model, "price": price, "ev": ev, "gap": gap, "name": "X"}


def agreeing(n=6, center=0.58):
    return [[center + 0.01 * i, 0.25, 1 - 0.25 - center - 0.01 * i] for i in range(n)]


def ctx(hda=None, n_models=6, odds=None, **kw):
    hda = agreeing() if hda is None else hda
    out = {"n_models": n_models, "fallback": False, "calibration_validated": True, "model_hda": hda,
           "probas": [0.58, 0.25, 0.17], "pb_probs": {}, "corners": None,
           "odds": odds if odds is not None else {"home": 2.0, "draw": 3.5, "away": 4.4}}
    out.update(kw)
    return out


def run(p, c):
    status, reason = assess(p, c)
    assert reason.startswith(ALLOWED) or reason.startswith(DATA_REASONS), reason
    return status, reason


def test_low_gain_value_bet_is_reliable_when_everything_agrees():
    assert run(pick(ev=0.06), ctx()) == (STATUS_RELIABLE, "Value Bet à faible gain")


def test_high_gain_value_bet():
    assert run(pick(ev=0.12), ctx()) == (STATUS_RELIABLE, "Value Bet à haut gain")


def test_very_negative_ev_means_no_bet():
    assert run(pick(ev=-0.20, p=0.30), ctx()) == (STATUS_NONE, "EV trop négatif : vous perdrez à long terme")


def test_huge_ev_is_risky_not_reliable():
    assert run(pick(ev=0.45), ctx()) == (STATUS_WEAK, "EV très intéressant mais risqué")


def test_models_disagreeing_on_this_market_downgrades():
    spread = [[0.30, 0.3, 0.4], [0.72, 0.15, 0.13], [0.35, 0.3, 0.35], [0.70, 0.2, 0.1], [0.62, 0.2, 0.18], [0.38, 0.3, 0.32]]
    s, r = run(pick(ev=0.08, p_model=0.52), ctx(hda=spread, probas=[0.52, 0.25, 0.23]))
    assert s == STATUS_WEAK and r == "Les modèles ne sont pas d'accord pour ce type de pari"


def test_conflict_on_the_match_reports_the_number_of_models():
    conflict = [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.2, 0.2, 0.6], [0.2, 0.2, 0.6], [0.6, 0.2, 0.2]]
    s, r = run(pick(), ctx(hda=conflict, probas=[0.6, 0.2, 0.2]))
    assert s == STATUS_NONE and r == "Les modèles sont en conflit sur ce match (3 modèles en désaccord sur 6)"


def test_probable_when_ev_slightly_negative_but_probability_high():
    assert run(pick(ev=-0.02, p=0.70), ctx()) == (STATUS_WEAK, "Probable")


def test_statistically_favorable_when_ev_is_positive_but_small():
    assert run(pick(ev=0.02, p=0.58), ctx()) == (STATUS_WEAK, "Statistiquement favorable")


def test_bookmaker_margin_is_cited_only_when_it_is_really_high():
    high = {"home": 1.8, "draw": 3.4, "away": 3.1}
    assert run(pick(ev=-0.02, p=0.58, price=1.8), ctx(odds=high))[1] == "La marge des bookmakers réduit fortement les gains"
    assert run(pick(ev=0.02, p=0.58, price=1.8), ctx(odds=high)) == (STATUS_WEAK, "La marge des bookmakers réduit fortement les gains")
    low = {"home": 2.0, "draw": 3.6, "away": 4.4}
    assert "marge" not in run(pick(ev=-0.02, p=0.70), ctx(odds=low))[1]


def test_missing_price_means_no_bet_with_a_reason():
    s, r = run(pick(price=None, ev=None), ctx())
    assert s == STATUS_NONE and "cote indisponible" in r


def test_insufficient_models_or_degraded_mode_means_no_bet():
    assert run(pick(), ctx(n_models=2))[0] == STATUS_NONE
    assert run(pick(), ctx(fallback=True))[0] == STATUS_NONE


def test_never_reliable_without_enough_statistical_evidence():
    assert run(pick(ev=0.08), ctx(n_models=3))[0] == STATUS_WEAK
    assert run(pick(ev=0.08), ctx(calibration_validated=False))[0] == STATUS_WEAK
    assert run(pick(ev=0.08, gap=0.20), ctx())[0] != STATUS_RELIABLE
    assert run(pick(ev=0.08), ctx(hda=agreeing(2)))[0] == STATUS_WEAK


def test_correct_score_and_corners_are_never_reliable():
    sc = pick(ev=0.12, p=0.12, key="cs_1_1", kind="score")
    assert run(sc, ctx())[0] == STATUS_WEAK
    corners = {"available": True, "lines": {9.5: {"p_over": 0.62, "p_fast": 0.60, "p_slow": 0.64}}}
    cp = pick(ev=0.08, p=0.62, key="corners_over_9_5", kind="corners")
    assert run(cp, ctx(corners=corners))[0] == STATUS_WEAK


def test_two_corner_estimators_that_disagree_are_flagged():
    corners = {"available": True, "lines": {9.5: {"p_over": 0.60, "p_fast": 0.78, "p_slow": 0.42}}}
    cp = pick(ev=0.08, p=0.60, key="corners_over_9_5", kind="corners")
    assert run(cp, ctx(corners=corners)) == (STATUS_WEAK, "Les modèles ne sont pas d'accord pour ce type de pari")


def test_unconfirmable_justifications_are_never_produced():
    combos = [pick(ev=e, p=p) for e in (-0.3, -0.05, 0.0, 0.03, 0.06, 0.2, 0.5) for p in (0.2, 0.5, 0.8)]
    for c in combos:
        _, reason = assess(c, ctx())
        assert "match piège" not in reason and "limite actuelle" not in reason
