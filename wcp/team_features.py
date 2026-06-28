"""Rolling team-level features computed without data leakage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wcp.priors import TEAM_SQUAD_VALUE


@dataclass
class TeamSnapshot:
    """Point-in-time team statistics before a match."""

    points_last5: float = 1.0
    points_last10: float = 1.0
    goals_for_last5: float = 1.2
    goals_for_last10: float = 1.2
    goals_against_last5: float = 1.2
    goals_against_last10: float = 1.2
    win_rate_last5: float = 0.33
    win_rate_last10: float = 0.33
    clean_sheet_rate_last10: float = 0.25
    matches_played: int = 0
    squad_value: float = 100.0  # millions EUR (prior)


@dataclass
class _MatchRecord:
    points: float
    goals_for: int
    goals_against: int
    win: int
    clean_sheet: int


def _rolling_mean(records: deque[_MatchRecord], attr: str) -> float:
    if not records:
        return 0.0
    return sum(getattr(r, attr) for r in records) / len(records)


class TeamFeatureStore:
    """Maintains rolling histories and serves point-in-time snapshots."""

    def __init__(self, windows: tuple[int, ...] = (5, 10)):
        self.windows = windows
        self._histories: dict[str, dict[int, deque[_MatchRecord]]] = {}
        self.latest: dict[str, TeamSnapshot] = {}

    def _ensure_team(self, team: str) -> None:
        if team not in self._histories:
            self._histories[team] = {w: deque(maxlen=w) for w in self.windows}

    def _snapshot(self, team: str) -> TeamSnapshot:
        self._ensure_team(team)
        h5 = self._histories[team][5]
        h10 = self._histories[team][10]
        n = len(h10)
        return TeamSnapshot(
            points_last5=_rolling_mean(h5, "points") if h5 else 1.0,
            points_last10=_rolling_mean(h10, "points") if h10 else 1.0,
            goals_for_last5=_rolling_mean(h5, "goals_for") if h5 else 1.2,
            goals_for_last10=_rolling_mean(h10, "goals_for") if h10 else 1.2,
            goals_against_last5=_rolling_mean(h5, "goals_against") if h5 else 1.2,
            goals_against_last10=_rolling_mean(h10, "goals_against") if h10 else 1.2,
            win_rate_last5=_rolling_mean(h5, "win") if h5 else 0.33,
            win_rate_last10=_rolling_mean(h10, "win") if h10 else 0.33,
            clean_sheet_rate_last10=_rolling_mean(h10, "clean_sheet") if h10 else 0.25,
            matches_played=n,
            squad_value=TEAM_SQUAD_VALUE.get(team, 50.0),
        )

    def get(self, team: str) -> TeamSnapshot:
        return self.latest.get(team, self._snapshot(team))

    def snapshot_before(self, team: str) -> TeamSnapshot:
        """Snapshot using current history state (call before update)."""
        snap = self._snapshot(team)
        snap.squad_value = TEAM_SQUAD_VALUE.get(team, 50.0)
        return snap

    def _update_team(self, team: str, gf: int, ga: int) -> None:
        self._ensure_team(team)
        win = int(gf > ga)
        pts = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
        rec = _MatchRecord(pts, gf, ga, win, int(ga == 0))
        for w in self.windows:
            self._histories[team][w].append(rec)
        self.latest[team] = self._snapshot(team)
        self.latest[team].squad_value = TEAM_SQUAD_VALUE.get(team, 50.0)

    def fit(self, matches: pd.DataFrame) -> "TeamFeatureStore":
        self._histories.clear()
        self.latest.clear()
        for _, m in matches.iterrows():
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            self._update_team(m["home_team"], hg, ag)
            self._update_team(m["away_team"], ag, hg)
        return self

    def matchup_features(
        self,
        home: str,
        away: str,
        *,
        home_snap: TeamSnapshot | None = None,
        away_snap: TeamSnapshot | None = None,
    ) -> dict[str, float]:
        h = home_snap or self.get(home)
        a = away_snap or self.get(away)
        return {
            "form_diff_5": h.points_last5 - a.points_last5,
            "form_diff_10": h.points_last10 - a.points_last10,
            "attack_diff_10": h.goals_for_last10 - a.goals_against_last10,
            "defense_diff_10": a.goals_for_last10 - h.goals_against_last10,
            "win_rate_diff_10": h.win_rate_last10 - a.win_rate_last10,
            "goals_scored_diff_10": h.goals_for_last10 - a.goals_for_last10,
            "squad_value_log_ratio": _log_ratio(h.squad_value, a.squad_value),
            "experience_gap": min(h.matches_played, 50) - min(a.matches_played, 50),
        }


def _log_ratio(a: float, b: float) -> float:
    return float(np.log1p(a) - np.log1p(b))
