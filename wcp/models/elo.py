"""Elo rating system for international football."""

from __future__ import annotations

import math

from wcp.config import ELO_HOME_ADV, ELO_INITIAL, ELO_K


class EloRatings:
    def __init__(self, k: float = ELO_K, home_adv: float = ELO_HOME_ADV, initial: float = ELO_INITIAL):
        self.k = k
        self.home_adv = home_adv
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def expected(self, home: str, away: str, neutral: bool = False) -> float:
        diff = self.get(home) - self.get(away) + (0 if neutral else self.home_adv)
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def update(self, home: str, away: str, hg: int, ag: int, neutral: bool = False) -> None:
        exp = self.expected(home, away, neutral=neutral)
        if hg > ag:
            actual = 1.0
        elif hg < ag:
            actual = 0.0
        else:
            actual = 0.5
        delta = self.k * (actual - exp)
        self.ratings[home] = self.get(home) + delta
        self.ratings[away] = self.get(away) - delta

    def fit(self, matches) -> "EloRatings":
        for _, m in matches.iterrows():
            self.update(m["home_team"], m["away_team"], m["home_goals"], m["away_goals"],
                        neutral=bool(m.get("neutral", 0)))
        return self

    def win_prob(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        """Return (P_home_win, P_draw, P_away_win) via logistic draw model."""
        diff = self.get(home) - self.get(away) + (0 if neutral else self.home_adv)
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400))
        # Draw probability decreases with rating gap (empirical fit)
        draw_base = 0.26
        draw = draw_base * math.exp(-abs(diff) / 600)
        draw = min(draw, 0.35)
        rem = 1.0 - draw
        p_away = rem * (1 - p_home)
        p_home = rem * p_home
        total = p_home + draw + p_away
        return p_home / total, draw / total, p_away / total
