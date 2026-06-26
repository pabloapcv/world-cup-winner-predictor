"""Ensemble predictor combining Elo, Dixon-Coles, and LightGBM."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from wcp.config import ENSEMBLE_WEIGHTS, MODELS_DIR
from wcp.features import match_features
from wcp.models.dixon_coles import DixonColesModel
from wcp.models.elo import EloRatings
from wcp.models.lightgbm_model import LightGBMPredictor


class EnsemblePredictor:
    def __init__(
        self,
        elo: EloRatings | None = None,
        dixon_coles: DixonColesModel | None = None,
        lightgbm: LightGBMPredictor | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.elo = elo or EloRatings()
        self.dixon_coles = dixon_coles or DixonColesModel()
        self.lightgbm = lightgbm or LightGBMPredictor()
        self.weights = weights or ENSEMBLE_WEIGHTS.copy()

    def fit(self, matches, train_df) -> "EnsemblePredictor":
        self.elo.fit(matches)
        self.dixon_coles.fit(matches)
        self.lightgbm.fit(train_df, train_df["outcome"])
        return self

    def match_prob(
        self, home: str, away: str, neutral: bool = True, is_wc: bool = True
    ) -> tuple[float, float, float]:
        w = self.weights
        p_elo = self.elo.win_prob(home, away, neutral=neutral)
        p_dc = self.dixon_coles.win_prob(home, away, neutral=neutral)
        feats = match_features(home, away, self.elo, neutral=neutral, is_wc=is_wc)
        p_lgb = self.lightgbm.predict_proba(feats)

        p_home = w["elo"] * p_elo[0] + w["dixon_coles"] * p_dc[0] + w["lightgbm"] * p_lgb[0]
        p_draw = w["elo"] * p_elo[1] + w["dixon_coles"] * p_dc[1] + w["lightgbm"] * p_lgb[1]
        p_away = w["elo"] * p_elo[2] + w["dixon_coles"] * p_dc[2] + w["lightgbm"] * p_lgb[2]
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def sample_match(
        self, home: str, away: str, neutral: bool, rng: np.random.Generator, is_wc: bool = True
    ) -> tuple[int, int]:
        """Sample score using Dixon-Coles (best for goal simulation)."""
        return self.dixon_coles.sample_score(home, away, neutral, rng)

    def save(self, path: Path | None = None) -> None:
        path = path or MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.elo, path / "elo.joblib")
        joblib.dump(self.dixon_coles, path / "dixon_coles.joblib")
        self.lightgbm.save(path / "lightgbm.joblib")

    @classmethod
    def load(cls, path: Path | None = None) -> "EnsemblePredictor":
        path = path or MODELS_DIR
        elo = joblib.load(path / "elo.joblib")
        dc = joblib.load(path / "dixon_coles.joblib")
        lgb = LightGBMPredictor().load(path / "lightgbm.joblib")
        return cls(elo=elo, dixon_coles=dc, lightgbm=lgb)

    def ratings_table(self) -> list[tuple[str, float, float, float]]:
        teams = sorted(set(self.elo.ratings) | set(self.dixon_coles.attack))
        rows = []
        for t in teams:
            elo = self.elo.get(t)
            att = self.dixon_coles.attack.get(t, 0)
            deff = self.dixon_coles.defense.get(t, 0)
            strength = elo + (att - deff) * 100
            rows.append((t, elo, att, deff))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows
