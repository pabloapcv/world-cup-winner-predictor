"""Dixon-Coles Poisson model for goal-based match prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DixonColesModel:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.25
    rho: float = -0.13
    xi: float = 0.0018

    def fit(self, matches, max_iter: int = 150) -> "DixonColesModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        hi = matches["home_team"].map(idx).values
        ai = matches["away_team"].map(idx).values
        xg = matches["home_goals"].values.astype(int)
        yg = matches["away_goals"].values.astype(int)
        neutral = matches["neutral"].values.astype(bool)
        dates = matches["date"].astype("int64").values
        t_max = dates.max()
        weights = np.exp(-self.xi * (t_max - dates) / (365.25 * 24 * 3600 * 1e9))

        def neg_ll(params: np.ndarray) -> float:
            att = params[:n] - params[:n].mean()
            deff = params[n:2 * n] - params[n:2 * n].mean()
            ha, rho = params[2 * n], params[2 * n + 1]

            ha_eff = np.where(neutral, 0.0, ha)
            lam_h = np.exp(att[hi] - deff[ai] + ha_eff)
            lam_a = np.exp(att[ai] - deff[hi])

            ll = weights * (poisson.logpmf(xg, lam_h) + poisson.logpmf(yg, lam_a))
            # Low-score correction
            mask00 = (xg == 0) & (yg == 0)
            mask01 = (xg == 0) & (yg == 1)
            mask10 = (xg == 1) & (yg == 0)
            mask11 = (xg == 1) & (yg == 1)
            if mask00.any():
                ll[mask00] += weights[mask00] * np.log(np.maximum(1 - lam_h[mask00] * lam_a[mask00] * rho, 1e-12))
            if mask01.any():
                ll[mask01] += weights[mask01] * np.log(np.maximum(1 + lam_h[mask01] * rho, 1e-12))
            if mask10.any():
                ll[mask10] += weights[mask10] * np.log(np.maximum(1 + lam_a[mask10] * rho, 1e-12))
            if mask11.any():
                ll[mask11] += weights[mask11] * np.log(np.maximum(1 - rho, 1e-12))
            return -float(ll.sum())

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25
        x0[2 * n + 1] = -0.13
        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": max_iter, "ftol": 1e-6})
        att = res.x[:n] - res.x[:n].mean()
        deff = res.x[n:2 * n] - res.x[n:2 * n].mean()
        self.attack = {t: float(att[idx[t]]) for t in teams}
        self.defense = {t: float(deff[idx[t]]) for t in teams}
        self.home_adv = float(res.x[2 * n])
        self.rho = float(res.x[2 * n + 1])
        return self

    @staticmethod
    def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1 - lam_h * lam_a * rho
        if x == 0 and y == 1:
            return 1 + lam_h * rho
        if x == 1 and y == 0:
            return 1 + lam_a * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def expected_goals(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        ha = 0.0 if neutral else self.home_adv
        lam_h = math.exp(self.attack.get(home, 0) - self.defense.get(away, 0) + ha)
        lam_a = math.exp(self.attack.get(away, 0) - self.defense.get(home, 0))
        return lam_h, lam_a

    def score_matrix(self, home: str, away: str, neutral: bool = True, max_goals: int = 8) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        mat = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
                if i <= 1 and j <= 1:
                    p *= self._tau(i, j, lam_h, lam_a, self.rho)
                mat[i, j] = p
        mat /= mat.sum()
        return mat

    def win_prob(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        mat = self.score_matrix(home, away, neutral)
        p_draw = float(np.trace(mat))
        p_home = float(np.tril(mat, -1).sum())
        p_away = float(np.triu(mat, 1).sum())
        return p_home, p_draw, p_away

    def sample_score(self, home: str, away: str, neutral: bool, rng: np.random.Generator) -> tuple[int, int]:
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        return min(int(rng.poisson(lam_h)), 8), min(int(rng.poisson(lam_a)), 8)
