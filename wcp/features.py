"""Feature engineering for match outcome prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wcp.models.elo import EloRatings


def match_outcome(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals < away_goals:
        return 2
    return 1


def build_training_frame(matches: pd.DataFrame, elo: EloRatings) -> pd.DataFrame:
    """Build feature matrix from historical matches with point-in-time Elo."""
    rows = []
    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        neutral = bool(m.get("neutral", 0))
        h_elo = elo.get(home)
        a_elo = elo.get(away)
        diff = h_elo - a_elo + (0 if neutral else elo.home_adv)
        rows.append({
            "home_team": home,
            "away_team": away,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": diff,
            "neutral": int(neutral),
            "is_world_cup": int(m.get("tournament") == "World Cup"),
            "outcome": match_outcome(m["home_goals"], m["away_goals"]),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
        })
        elo.update(home, away, m["home_goals"], m["away_goals"], neutral=neutral)
    return pd.DataFrame(rows)


FEATURE_COLS = ["home_elo", "away_elo", "elo_diff", "neutral", "is_world_cup"]


def match_features(home: str, away: str, elo: EloRatings, neutral: bool = True, is_wc: bool = True) -> np.ndarray:
    h, a = elo.get(home), elo.get(away)
    diff = h - a + (0 if neutral else elo.home_adv)
    return np.array([[h, a, diff, int(neutral), int(is_wc)]], dtype=np.float64)
