import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from goangel_core import blend_with_market, devig

MIN_EV = 0.04
MAX_EV = 0.30
MIN_EV_PROBABLE = -0.06
BLEND_W_MODEL = 0.40
MAX_GAP_VALUE = 0.15
MAX_GAP_PROBABLE = 0.22
KELLY_FRACTION = 0.25
MAX_KELLY = 0.05
SURE_MIN_PROB = 0.50
FLOORS = {"result": 0.40, "double_chance": 0.62, "goals": 0.55, "ht": 0.58, "corners": 0.55, "score": 0.08}
TYPICAL = {
    "home": 0.45, "away": 0.30, "draw": 0.26, "1x": 0.70, "x2": 0.55, "12": 0.74,
    "dnb_home": 0.62, "dnb_away": 0.42, "over_1_5": 0.75, "under_1_5": 0.25,
    "over_2_5": 0.52, "under_2_5": 0.48, "btts_yes": 0.52, "btts_no": 0.48,
}
PAIRS = {
    "over_1_5": "under_1_5", "under_1_5": "over_1_5",
    "over_2_5": "under_2_5", "under_2_5": "over_2_5",
    "btts_yes": "btts_no", "btts_no": "btts_yes",
    "ht_over_0_5": "ht_under_0_5", "ht_under_0_5": "ht_over_0_5",
    "ht_over_1_5": "ht_under_1_5", "ht_under_1_5": "ht_over_1_5",
}
for _line in ("7_5", "8_5", "9_5", "10_5", "11_5", "12_5"):
    PAIRS[f"corners_over_{_line}"] = f"corners_under_{_line}"
    PAIRS[f"corners_under_{_line}"] = f"corners_over_{_line}"

RESULT_KEYS = ("home", "draw", "away")
DOUBLE_CHANCE_KEYS = ("1x", "x2", "12", "dnb_home", "dnb_away")

STATUS_RELIABLE = "✅ Pari fiable"
STATUS_WEAK = "⚠️ Pari peu fiable"
STATUS_NONE = "⛔ Aucun pari"

MIN_MODELS = 3
MIN_MODELS_RELIABLE = 4
AGREE_TOLERANCE = 0.12
AGREE_SPREAD = 0.12
CONFLICT_RATIO = 0.40
EV_TOO_NEGATIVE = -0.06
EV_HIGH_GAIN = 0.10
EV_RISKY = 0.15
HIGH_MARGIN = 0.08
TOP_SCORES_AS_BETS = 5


def expected_value(prob: float, price: float) -> float:
    return float(prob * price - 1.0)


def kelly_stake(prob: float, price: float) -> float:
    b = price - 1.0
    if b <= 0 or prob <= 0:
        return 0.0
    k = (b * prob - (1.0 - prob)) / b
    return float(np.clip(k * KELLY_FRACTION, 0.0, MAX_KELLY)) if k > 0 else 0.0


def _kind(key: str) -> str:
    if key.startswith("ht_"):
        return "ht"
    if key.startswith("cs_"):
        return "score"
    if key.startswith("corners_"):
        return "corners"
    if key in RESULT_KEYS:
        return "result"
    if key in DOUBLE_CHANCE_KEYS:
        return "double_chance"
    return "goals"


def _family(key: str) -> str:
    kind = _kind(key)
    if kind in ("ht", "score", "corners"):
        return kind
    if key in ("home", "draw", "away", "dnb_home", "dnb_away"):
        return "result"
    if key in ("1x", "x2", "12"):
        return "double_chance"
    return "btts" if key.startswith("btts") else "goals"


def _derived(p3: Sequence[float], prefix: str) -> Dict[str, float]:
    h, d, a = float(p3[0]), float(p3[1]), float(p3[2])
    out = {f"{prefix}home": h, f"{prefix}draw": d, f"{prefix}away": a,
           f"{prefix}1x": h + d, f"{prefix}x2": d + a, f"{prefix}12": h + a}
    if not prefix and h + a > 0:
        out["dnb_home"], out["dnb_away"] = h / (h + a), a / (h + a)
    return out


def _blend2(p: float, m: float) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    m = float(np.clip(m, 1e-4, 1 - 1e-4))
    a = BLEND_W_MODEL * np.log(p) + (1 - BLEND_W_MODEL) * np.log(m)
    b = BLEND_W_MODEL * np.log(1 - p) + (1 - BLEND_W_MODEL) * np.log(1 - m)
    return float(1.0 / (1.0 + np.exp(b - a)))


def _triple_reference(model3: Sequence[float], odds: Dict[str, float], prefix: str):
    keys = [f"{prefix}home", f"{prefix}draw", f"{prefix}away"]
    if not all(k in odds for k in keys):
        return _derived(model3, prefix), None
    market3 = devig([odds[k] for k in keys])
    adjusted = blend_with_market(model3, market3, BLEND_W_MODEL)
    return _derived(adjusted, prefix), _derived(market3, prefix)


def _line_key(line: float) -> str:
    return f"{line:.1f}".replace(".", "_")


def build_candidates(
    proba: Sequence[float],
    home: str,
    away: str,
    over15: float,
    over25: float,
    btts_yes: float,
    ht_hda: Optional[Sequence[float]] = None,
    ht_over05: Optional[float] = None,
    ht_over15: Optional[float] = None,
    top_scores: Optional[Sequence[Tuple[str, float]]] = None,
    corners: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    d = _derived(proba, "")
    rows = [
        ("home", f"Victoire {home}", d["home"]),
        ("draw", "Match Nul", d["draw"]),
        ("away", f"Victoire {away}", d["away"]),
        ("1x", f"Double chance 1X ({home} ou Nul)", d["1x"]),
        ("x2", f"Double chance X2 (Nul ou {away})", d["x2"]),
        ("12", f"Double chance 12 ({home} ou {away})", d["12"]),
        ("dnb_home", f"Draw No Bet {home}", d.get("dnb_home", 0.0)),
        ("dnb_away", f"Draw No Bet {away}", d.get("dnb_away", 0.0)),
        ("over_1_5", "+1.5 buts", over15), ("under_1_5", "-1.5 buts", 1.0 - over15),
        ("over_2_5", "+2.5 buts", over25), ("under_2_5", "-2.5 buts", 1.0 - over25),
        ("btts_yes", "BTTS — Les deux marquent (Oui)", btts_yes),
        ("btts_no", "BTTS — Les deux marquent (Non)", 1.0 - btts_yes),
    ]
    if ht_hda is not None and len(ht_hda) == 3:
        h = _derived(ht_hda, "ht_")
        rows += [
            ("ht_home", f"Victoire {home} (1ère MT)", h["ht_home"]),
            ("ht_draw", "Match Nul (1ère MT)", h["ht_draw"]),
            ("ht_away", f"Victoire {away} (1ère MT)", h["ht_away"]),
            ("ht_1x", f"Double chance 1X ({home} ou Nul) (1ère MT)", h["ht_1x"]),
            ("ht_x2", f"Double chance X2 (Nul ou {away}) (1ère MT)", h["ht_x2"]),
        ]
    if ht_over05 is not None:
        rows += [("ht_over_0_5", "+0.5 buts (1ère MT)", ht_over05), ("ht_under_0_5", "-0.5 buts (1ère MT)", 1.0 - ht_over05)]
    if ht_over15 is not None:
        rows += [("ht_over_1_5", "+1.5 buts (1ère MT)", ht_over15), ("ht_under_1_5", "-1.5 buts (1ère MT)", 1.0 - ht_over15)]
    for label, prob in list(top_scores or [])[:TOP_SCORES_AS_BETS]:
        goals_home, goals_away = label.split("-")
        rows.append((f"cs_{int(goals_home)}_{int(goals_away)}", f"Score exact {label}", prob))
    if corners and corners.get("available"):
        for line, entry in corners["lines"].items():
            suffix = _line_key(line)
            rows += [(f"corners_over_{suffix}", f"+{line} corners", entry["p_over"]),
                     (f"corners_under_{suffix}", f"-{line} corners", 1.0 - entry["p_over"])]
    out = []
    for key, name, p in rows:
        p = float(p)
        if np.isfinite(p) and 0.0 <= p <= 1.0:
            out.append({"key": key, "name": name, "p_model": p, "kind": _kind(key), "family": _family(key), "is_ht": key.startswith("ht_")})
    return out


def _score_rows(cands, proba, odds, ht_hda):
    ft_adj, ft_mkt = _triple_reference(proba, odds, "")
    ht_adj, ht_mkt = _triple_reference(ht_hda, odds, "ht_") if ht_hda is not None and len(ht_hda) == 3 else ({}, None)
    rows = []
    for c in cands:
        key, p_model = c["key"], c["p_model"]
        p, ref = p_model, None
        if key in ft_adj:
            p = ft_adj[key]
            ref = ft_mkt.get(key) if ft_mkt else None
        elif key in ht_adj:
            p = ht_adj[key]
            ref = ht_mkt.get(key) if ht_mkt else None
        elif key in PAIRS and key in odds and PAIRS[key] in odds:
            ref = float(devig([odds[key], odds[PAIRS[key]]])[0])
            p = _blend2(p_model, ref)
        price = odds.get(key)
        ev = expected_value(p, price) if price else None
        rows.append(dict(c, p=float(p), price=price, ev=ev, ref=ref, gap=abs(p_model - ref) if ref is not None else 0.0,
                         kelly=kelly_stake(p, price) if price else 0.0))
    return rows


def _rank(r):
    priced = r["price"] is not None
    if priced and MIN_EV <= r["ev"] <= MAX_EV and r["p"] >= FLOORS[r["kind"]] and r["gap"] <= MAX_GAP_VALUE:
        return 0, r["kelly"]
    if priced and r["p"] >= FLOORS[r["kind"]] and r["gap"] <= MAX_GAP_PROBABLE and r["ev"] >= MIN_EV_PROBABLE:
        return 1, r["ev"] + 0.1 * r["p"]
    if r["kind"] not in ("ht", "score") and r["p"] >= SURE_MIN_PROB:
        return 2, r["p"] - TYPICAL.get(r["key"], 0.5)
    if r["p"] >= SURE_MIN_PROB:
        return 2, r["p"] - 0.7
    return 3, r["p"]


def recommend(
    cands: List[Dict[str, Any]],
    proba: Sequence[float],
    odds: Optional[Dict[str, float]],
    ht_hda: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    odds = odds or {}
    rows = _score_rows(cands, proba, odds, ht_hda)
    best = min(rows, key=lambda r: (_rank(r)[0], -_rank(r)[1]))
    return {
        "pick": dict(best, tier=_rank(best)[0]),
        "n_priced": sum(1 for r in rows if r["price"] is not None),
        "n_markets": len(rows),
    }


def model_probabilities(key: str, ctx: Dict[str, Any]) -> List[float]:
    hda = ctx.get("model_hda")
    pb = ctx.get("pb_probs") or {}
    if key in RESULT_KEYS + DOUBLE_CHANCE_KEYS and hda is not None and len(hda):
        arr = np.asarray(hda, dtype=float)
        h, d, a = arr[:, 0], arr[:, 1], arr[:, 2]
        table = {"home": h, "draw": d, "away": a, "1x": h + d, "x2": d + a, "12": h + a,
                 "dnb_home": h / np.maximum(h + a, 1e-9), "dnb_away": a / np.maximum(h + a, 1e-9)}
        return [float(x) for x in table[key]]
    if key in ("over_1_5", "over_2_5", "btts_yes"):
        return [float(x) for x in pb.get(key, [])]
    if key in ("under_1_5", "under_2_5", "btts_no"):
        opposite = "btts_yes" if key == "btts_no" else key.replace("under", "over")
        return [1.0 - float(x) for x in pb.get(opposite, [])]
    corners = ctx.get("corners")
    if key.startswith("corners_") and corners and corners.get("available"):
        parts = key.split("_")
        line = float(f"{parts[-2]}.{parts[-1]}")
        entry = corners["lines"].get(line)
        if entry is not None:
            probs = [entry["p_fast"], entry["p_slow"]]
            return probs if parts[1] == "over" else [1.0 - p for p in probs]
    return []


def agreement(pick: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    probs = model_probabilities(pick["key"], ctx)
    n = len(probs)
    if n < 2:
        return {"n": n, "disagree": False, "n_disagree": 0, "sufficient": False}
    ref = pick["p_model"]
    n_disagree = sum(1 for p in probs if abs(p - ref) > AGREE_TOLERANCE)
    spread = float(np.std(probs))
    disagree = n_disagree >= math.ceil(0.34 * n) or spread > AGREE_SPREAD
    return {"n": n, "disagree": bool(disagree), "n_disagree": n_disagree, "sufficient": n >= MIN_MODELS}


def match_conflict(ctx: Dict[str, Any]) -> Tuple[int, int]:
    hda = ctx.get("model_hda")
    ensemble = ctx.get("probas")
    if hda is None or ensemble is None or len(hda) < MIN_MODELS:
        return 0, 0 if hda is None else len(hda)
    arr = np.asarray(hda, dtype=float)
    leader = int(np.argmax(ensemble))
    return int(np.sum(np.argmax(arr, axis=1) != leader)), int(len(arr))


def bookmaker_margin(key: str, odds: Dict[str, float]) -> Optional[float]:
    if key in RESULT_KEYS + DOUBLE_CHANCE_KEYS:
        if all(k in odds for k in RESULT_KEYS):
            return float(sum(1.0 / odds[k] for k in RESULT_KEYS) - 1.0)
        return None
    other = PAIRS.get(key)
    if other and key in odds and other in odds:
        return float(1.0 / odds[key] + 1.0 / odds[other] - 1.0)
    return None


def assess(pick: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[str, str]:
    n_models = int(ctx.get("n_models", 0))
    if ctx.get("fallback") or n_models < MIN_MODELS:
        return STATUS_NONE, "données insuffisantes : modèles actifs insuffisants ou mode dégradé"
    if pick["price"] is None or pick["ev"] is None:
        return STATUS_NONE, "cote indisponible pour ce marché : EV non évaluable"
    conflict, total = match_conflict(ctx)
    if total >= MIN_MODELS and conflict >= max(2, math.ceil(CONFLICT_RATIO * total)):
        return STATUS_NONE, f"Les modèles sont en conflit sur ce match ({conflict} modèles en désaccord sur {total})"

    ev, p, kind = pick["ev"], pick["p"], pick["kind"]
    margin = bookmaker_margin(pick["key"], ctx.get("odds") or {})
    high_margin = margin is not None and margin >= HIGH_MARGIN
    agree = agreement(pick, ctx)

    if ev < EV_TOO_NEGATIVE:
        return STATUS_NONE, "EV trop négatif : vous perdrez à long terme"
    if ev < 0:
        if high_margin:
            return STATUS_NONE, "La marge des bookmakers réduit fortement les gains"
        if p >= FLOORS[kind] + 0.08 and not agree["disagree"]:
            return STATUS_WEAK, "Probable"
        return STATUS_NONE, "EV trop négatif : vous perdrez à long terme"
    if ev < MIN_EV and high_margin:
        return STATUS_WEAK, "La marge des bookmakers réduit fortement les gains"
    if ev > MAX_EV or (ev >= EV_RISKY and pick["gap"] > MAX_GAP_VALUE):
        return STATUS_WEAK, "EV très intéressant mais risqué"
    if agree["disagree"]:
        return STATUS_WEAK, "Les modèles ne sont pas d'accord pour ce type de pari"
    if ev < MIN_EV:
        if p >= FLOORS[kind]:
            return STATUS_WEAK, "Statistiquement favorable"
        return STATUS_NONE, "avantage statistique insuffisant : EV inférieur à 4 % et probabilité trop faible"
    justification = "Value Bet à haut gain" if ev >= EV_HIGH_GAIN else "Value Bet à faible gain"
    reliable = (
        kind not in ("score", "corners")
        and agree["sufficient"]
        and n_models >= MIN_MODELS_RELIABLE
        and ctx.get("calibration_validated", True) is not False
        and p >= FLOORS[kind]
        and pick["gap"] <= MAX_GAP_VALUE
        and not high_margin
    )
    return (STATUS_RELIABLE if reliable else STATUS_WEAK), justification
