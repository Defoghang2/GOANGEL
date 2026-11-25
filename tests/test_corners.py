import sys
sys.path[:0] = ["."]
import numpy as np
from sklearn.metrics import log_loss
from corners import CornerModel, LINES


def corner_world(n=6000, T=16, seed=0):
    r = np.random.default_rng(seed)
    att = r.normal(0, .22, T)
    dfn = r.normal(0, .18, T)
    rows = []
    for i in range(n):
        h, a = r.choice(T, 2, replace=False)
        mh, ma = 5.6 * np.exp(att[h] + dfn[a]), 4.6 * np.exp(att[a] + dfn[h])
        total_mean = mh + ma
        k = 1 / 0.10
        ch = int(r.negative_binomial(k, k / (k + mh)))
        ca = int(r.negative_binomial(k, k / (k + ma)))
        rows.append((f"t{h}", f"t{a}", ch, ca, mh + ma))
    return rows


def evaluate(line=9.5, split=0.7):
    rows = corner_world()
    model = CornerModel()
    cut = int(len(rows) * split)
    for h, a, ch, ca, _ in rows[:cut]:
        model.update(h, a, "L", ch, ca)
    y, p, p_fast, p_slow = [], [], [], []
    for h, a, ch, ca, _ in rows[cut:]:
        f = model.forecast(h, a, "L")
        y.append(int(ch + ca > line))
        p.append(f["lines"][line]["p_over"])
        p_fast.append(f["lines"][line]["p_fast"])
        p_slow.append(f["lines"][line]["p_slow"])
        model.update(h, a, "L", ch, ca)
    return np.array(y), np.array(p), np.array(p_fast), np.array(p_slow)


def test_corner_probabilities_beat_a_constant_baseline():
    y, p, _, _ = evaluate()
    base = np.full_like(p, y.mean())
    ll, ll0 = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), log_loss(y, base)
    print("corners over 9.5 :", round(ll, 4), "vs constant", round(ll0, 4))
    assert ll < ll0 - 0.01


def test_probabilities_are_calibrated_on_average():
    y, p, _, _ = evaluate()
    assert abs(float(p.mean()) - float(y.mean())) < 0.04


def test_over_probability_decreases_with_the_line():
    model = CornerModel()
    for h, a, ch, ca, _ in corner_world(800):
        model.update(h, a, "L", ch, ca)
    lines = model.forecast("t1", "t2", "L")["lines"]
    probs = [lines[l]["p_over"] for l in LINES]
    assert all(x > y for x, y in zip(probs, probs[1:])) and 0 < probs[-1] < probs[0] < 1


def test_unknown_or_unseen_teams_are_not_forecast():
    model = CornerModel()
    assert model.forecast("a", "b", "L") == {"available": False}
    for _ in range(3):
        model.update("a", "b", "L", 6, 4)
    assert model.forecast("a", "b", "L")["available"] is False


def test_missing_corner_data_does_not_update_the_state():
    model = CornerModel()
    model.update("a", "b", "L", None, 5)
    model.update("a", "b", "L", float("nan"), 5)
    model.update("a", "b", "L", -1, 5)
    assert model.team == {} and model.comp == {}


def test_forecast_uses_only_past_matches():
    rows = corner_world(500)
    a, b = CornerModel(), CornerModel()
    for h, aw, ch, ca, _ in rows[:300]:
        a.update(h, aw, "L", ch, ca)
        b.update(h, aw, "L", ch, ca)
    before = a.forecast("t1", "t2", "L")["total"]
    for h, aw, ch, ca, _ in rows[300:]:
        b.update(h, aw, "L", ch + 30, ca + 30)
    assert a.forecast("t1", "t2", "L")["total"] == before
    assert b.forecast("t1", "t2", "L")["total"] > before + 5


def test_fast_and_slow_estimators_exist_for_agreement_checks():
    model = CornerModel()
    for h, a, ch, ca, _ in corner_world(1500):
        model.update(h, a, "L", ch, ca)
    line = model.forecast("t3", "t4", "L")["lines"][9.5]
    assert 0 < line["p_fast"] < 1 and 0 < line["p_slow"] < 1 and abs(line["p_fast"] - line["p_slow"]) < 0.25
