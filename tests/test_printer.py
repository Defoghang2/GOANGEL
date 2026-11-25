import os, sys, io, contextlib, re
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "."]
import numpy as np
import pandas as pd
import predict as P
from odds import extract_markets
from corners import LINES

EXPECTED_PREFIXES = [
    "⚽ {H} VS {A}", "🤔 XG :", "🎯 SCORES PROBABLES :", "⚽ {H} :", "🤝 NUL :", "⚽ {A} :",
    "🏆 +1.5 buts :", "🏆 -1.5 buts :", "🏆 +2.5 buts :", "🏆 -2.5 buts :", "🎯 BTTS (Oui) :", "🎯 BTTS (Non) :",
    "MARCHÉS 1ÈRE MI-TEMPS :", "⚽ {H} :", "🤝 NUL :", "⚽ {A} :", "🏆 +1.5 :", "🏆 -1.5 :",
    "💰 COTE :", "🔔 CORNERS :", "💎 EV :", "💼 MISE KELLY :", "🚨 CONSEILLER :", "🤖 STATUT DU PARI :",
]


def corners_info(total=10.3):
    lines = {l: {"p_over": float(np.clip(0.5 + (total - l) * 0.12, 0.05, 0.95)), "p_fast": 0.55, "p_slow": 0.5} for l in LINES}
    return {"available": True, "home": total * 0.55, "away": total * 0.45, "total": total, "alpha": 0.1, "lines": lines}


def make_markets(odds=None, corners=True, n_models=(5, 5), fallback=False):
    m = P._build_output_from_lambdas(1.8, 1.1, 0.45)
    m["bookmaker_odds"] = odds if odds is not None else {}
    m["model_count_ml"], m["model_count_pb"] = n_models
    m["model_count_total"] = sum(n_models)
    m["fallback"] = fallback
    h = m["home_draw_away"]
    rng = np.random.default_rng(1)
    m["model_hda"] = [list(np.clip(np.array(h) + rng.normal(0, 0.015, 3), 0.01, 0.98)) for _ in range(sum(n_models))]
    m["pb_probs"] = {"over_1_5": [m["over15"]] * 4, "over_2_5": [m["over25"]] * 4, "btts_yes": [m["btts_yes"]] * 4}
    m["corners"] = corners_info() if corners else {"available": False}
    return m


def render(markets, home="Grêmio", away="Bahia"):
    row = pd.Series({"home_raw": home, "away_raw": away})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P._print_match_prediction(row, markets, None, None)
    return [l for l in buf.getvalue().split("\n") if l.strip()]


def check_structure(lines, home="GRÊMIO", away="BAHIA"):
    assert len(lines) == len(EXPECTED_PREFIXES), lines
    for line, prefix in zip(lines, EXPECTED_PREFIXES):
        assert line.startswith(prefix.format(H=home, A=away)), (line, prefix)


def test_exact_structure_with_odds():
    odds = extract_markets({"odds_home": 1.95, "odds_draw": 3.6, "odds_away": 4.4, "odds_over_15": 1.3, "odds_under_15": 3.5,
                            "corners_over_9_5": 1.85, "corners_under_9_5": 1.95})
    lines = render(make_markets(odds))
    check_structure(lines)
    print("\n".join(lines))


def test_exact_structure_without_any_odds():
    lines = render(make_markets({}))
    check_structure(lines)
    assert "cote indisponible" in lines[-1] and lines[-1].startswith("🤖 STATUT DU PARI : ⛔ Aucun pari")


def test_exact_structure_when_corner_data_is_missing_and_when_data_is_insufficient():
    check_structure(render(make_markets({}, corners=False)))
    insufficient = P._build_insufficient_output("Aucune donnée historique exploitable")
    lines = render(insufficient)
    check_structure(lines)
    assert "⛔ Aucun pari" in lines[-1] and "Aucune donnée" in lines[-1]


def test_status_line_always_uses_one_of_the_three_categories():
    for odds in ({}, {"home": 1.4, "draw": 4.8, "away": 8.0}, {"home": 3.5, "draw": 3.3, "away": 2.0}):
        for fallback in (False, True):
            last = render(make_markets(odds, fallback=fallback))[-1]
            assert re.match(r"^🤖 STATUT DU PARI : (✅ Pari fiable|⚠️ Pari peu fiable|⛔ Aucun pari) — .+", last), last


def test_no_alternative_lines_and_no_three_and_a_half_lines_are_printed():
    text = "\n".join(render(make_markets({"home": 1.9, "draw": 3.5, "away": 4.5})))
    assert "alternative" not in text.lower() and "3.5" not in text and "cotes disponibles" not in text


def test_corner_line_shows_expectation_probabilities_and_prices():
    odds = {"corners_over_10_5": 1.9, "corners_under_10_5": 1.9}
    corner_line = [l for l in render(make_markets(odds)) if l.startswith("🔔 CORNERS")][0]
    assert "attendus" in corner_line and "+10.5" in corner_line and "-10.5" in corner_line and "cote 1.90" in corner_line


def test_two_matches_have_identical_layout():
    a = render(make_markets({}), "Alpha FC", "Beta FC")
    b = render(make_markets({"home": 1.6, "draw": 4.0, "away": 6.0}), "Gamma", "Delta")
    check_structure(a, "ALPHA FC", "BETA FC")
    check_structure(b, "GAMMA", "DELTA")
