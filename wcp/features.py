"""Feature engineering for match outcome prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wcp.config import HOST_NATIONS
from wcp.models.elo import EloRatings
from wcp.team_features import TeamFeatureStore


def match_outcome(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals < away_goals:
        return 2
    return 1


MATCHUP_COLS = [
    "form_diff_5",
    "form_diff_10",
    "attack_diff_10",
    "defense_diff_10",
    "win_rate_diff_10",
    "goals_scored_diff_10",
    "squad_value_log_ratio",
    "experience_gap",
]

BASE_COLS = ["home_elo", "away_elo", "elo_diff", "neutral", "is_world_cup", "host_boost"]

FEATURE_COLS = BASE_COLS + MATCHUP_COLS


def build_training_frame(
    matches: pd.DataFrame,
    elo: EloRatings,
    team_store: TeamFeatureStore | None = None,
) -> tuple[pd.DataFrame, TeamFeatureStore]:
    """Build feature matrix with point-in-time Elo and rolling team stats."""
    store = team_store or TeamFeatureStore()
    store._histories.clear()
    store.latest.clear()

    rows = []
    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        neutral = bool(m.get("neutral", 0))
        is_wc = m.get("tournament") == "World Cup"

        h_snap = store.snapshot_before(home)
        a_snap = store.snapshot_before(away)
        matchup = store.matchup_features(home, away, home_snap=h_snap, away_snap=a_snap)

        h_elo = elo.get(home)
        a_elo = elo.get(away)
        diff = h_elo - a_elo + (0 if neutral else elo.home_adv)
        host_boost = int(home in HOST_NATIONS and is_wc)

        row = {
            "home_team": home,
            "away_team": away,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": diff,
            "neutral": int(neutral),
            "is_world_cup": int(is_wc),
            "host_boost": host_boost,
            "outcome": match_outcome(m["home_goals"], m["away_goals"]),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            **matchup,
        }
        rows.append(row)

        elo.update(home, away, m["home_goals"], m["away_goals"], neutral=neutral)
        store._update_team(home, int(m["home_goals"]), int(m["away_goals"]))
        store._update_team(away, int(m["away_goals"]), int(m["home_goals"]))

    return pd.DataFrame(rows), store


def match_features(
    home: str,
    away: str,
    elo: EloRatings,
    team_store: TeamFeatureStore,
    neutral: bool = True,
    is_wc: bool = True,
) -> pd.DataFrame:
    h, a = elo.get(home), elo.get(away)
    diff = h - a + (0 if neutral else elo.home_adv)
    matchup = team_store.matchup_features(home, away)
    host_boost = int(home in HOST_NATIONS and is_wc)

    data = {
        "home_elo": h,
        "away_elo": a,
        "elo_diff": diff,
        "neutral": int(neutral),
        "is_world_cup": int(is_wc),
        "host_boost": host_boost,
        **matchup,
    }
    return pd.DataFrame([{c: data[c] for c in FEATURE_COLS}])
