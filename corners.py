import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import nbinom

LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)


class CornerModel:
    FAST = 0.93
    SLOW = 0.985
    PRIOR_HOME = 5.6
    PRIOR_AWAY = 4.6
    PRIOR_WEIGHT = 30.0
    SHRINK = 8.0
    MIN_TEAM_MATCHES = 5
    BLEND_FAST = 0.6
    ALPHA_PRIOR = 0.12
    ALPHA_MIN = 0.02
    ALPHA_MAX = 0.6
    MIN_ALPHA_MATCHES = 30

    def __init__(self) -> None:
        self.team: Dict[str, List[Any]] = {}
        self.comp: Dict[str, List[float]] = {}

    @staticmethod
    def _valid(value: Optional[float]) -> bool:
        return value is not None and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0

    def update(self, home: str, away: str, comp: str, corners_home: Optional[float], corners_away: Optional[float]) -> None:
        if not (self._valid(corners_home) and self._valid(corners_away)):
            return
        for team, won, conceded in ((home, corners_home, corners_away), (away, corners_away, corners_home)):
            state = self.team.setdefault(team, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0])
            for idx, decay in ((0, self.FAST), (1, self.SLOW)):
                cell = state[idx]
                cell[0] = cell[0] * decay + won
                cell[1] = cell[1] * decay + conceded
                cell[2] = cell[2] * decay + 1.0
            state[2] += 1
        c = self.comp.setdefault(comp, [0.0, 0.0, 0.0, 0.0, 0.0])
        total = corners_home + corners_away
        c[0] += 1.0
        c[1] += corners_home
        c[2] += corners_away
        delta = total - c[3]
        c[3] += delta / c[0]
        c[4] += delta * (total - c[3])

    def _league(self, comp: str) -> Tuple[float, float]:
        c = self.comp.get(comp)
        n = c[0] if c else 0.0
        w = self.PRIOR_WEIGHT
        return ((self.PRIOR_HOME * w + (c[1] if c else 0.0)) / (w + n),
                (self.PRIOR_AWAY * w + (c[2] if c else 0.0)) / (w + n))

    def _alpha(self, comp: str, mean_total: float) -> float:
        c = self.comp.get(comp)
        if not c or c[0] < self.MIN_ALPHA_MATCHES:
            return self.ALPHA_PRIOR
        variance = c[4] / (c[0] - 1.0)
        alpha = (variance - mean_total) / max(mean_total ** 2, 1e-9)
        return float(np.clip(alpha, self.ALPHA_MIN, self.ALPHA_MAX))

    def _team_rates(self, team: str, idx: int, avg: float) -> Tuple[float, float]:
        state = self.team.get(team)
        if state is None or state[idx][2] <= 0:
            return avg, avg
        won, conceded, weight = state[idx]
        trust = weight / (weight + self.SHRINK)
        return trust * won / weight + (1 - trust) * avg, trust * conceded / weight + (1 - trust) * avg

    def _expected(self, home: str, away: str, comp: str, idx: int) -> Tuple[float, float]:
        league_home, league_away = self._league(comp)
        avg = (league_home + league_away) / 2.0
        home_for, home_against = self._team_rates(home, idx, avg)
        away_for, away_against = self._team_rates(away, idx, avg)
        return (league_home * (home_for / avg) * (away_against / avg),
                league_away * (away_for / avg) * (home_against / avg))

    @staticmethod
    def probability_over(line: float, mean_total: float, alpha: float) -> float:
        mean_total = float(np.clip(mean_total, 1.0, 30.0))
        r = 1.0 / alpha
        return float(1.0 - nbinom.cdf(int(math.floor(line)), r, r / (r + mean_total)))

    def team_matches(self, team: str) -> int:
        state = self.team.get(team)
        return int(state[2]) if state else 0

    def forecast(self, home: str, away: str, comp: str) -> Dict[str, Any]:
        if min(self.team_matches(home), self.team_matches(away)) < self.MIN_TEAM_MATCHES:
            return {"available": False}
        fast_h, fast_a = self._expected(home, away, comp, 0)
        slow_h, slow_a = self._expected(home, away, comp, 1)
        lam_h = self.BLEND_FAST * fast_h + (1 - self.BLEND_FAST) * slow_h
        lam_a = self.BLEND_FAST * fast_a + (1 - self.BLEND_FAST) * slow_a
        total = lam_h + lam_a
        alpha = self._alpha(comp, total)
        lines = {
            line: {
                "p_over": self.probability_over(line, total, alpha),
                "p_fast": self.probability_over(line, fast_h + fast_a, alpha),
                "p_slow": self.probability_over(line, slow_h + slow_a, alpha),
            }
            for line in LINES
        }
        return {"available": True, "home": lam_h, "away": lam_a, "total": total, "alpha": alpha, "lines": lines}
