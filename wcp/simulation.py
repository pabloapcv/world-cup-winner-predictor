"""Monte Carlo tournament simulation for World Cup 2026."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from wcp.config import DEFAULT_SIMULATIONS, HOST_NATIONS, WORLD_CUP_2026_GROUPS
from wcp.models.ensemble import EnsemblePredictor


def _play_match(
    predictor: EnsemblePredictor,
    home: str,
    away: str,
    rng: np.random.Generator,
    neutral: bool = True,
) -> tuple[str, int, int]:
    hg, ag = predictor.dixon_coles.sample_score(home, away, neutral=neutral, rng=rng)
    if hg > ag:
        return home, hg, ag
    if ag > hg:
        return away, hg, ag
    # Penalties — use Elo (fast) for knockout tiebreaks
    p_h, _, p_a = predictor.elo.win_prob(home, away, neutral=neutral)
    winner = home if rng.random() < p_h / (p_h + p_a) else away
    return winner, hg, ag


def _group_standings(
    teams: list[str],
    predictor: EnsemblePredictor,
    rng: np.random.Generator,
) -> list[dict]:
    stats = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0} for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            hg, ag = predictor.dixon_coles.sample_score(h, a, neutral=True, rng=rng)
            stats[h]["gf"] += hg
            stats[a]["gf"] += ag
            stats[h]["gd"] += hg - ag
            stats[a]["gd"] += ag - hg
            if hg > ag:
                stats[h]["pts"] += 3
            elif hg < ag:
                stats[a]["pts"] += 3
            else:
                stats[h]["pts"] += 1
                stats[a]["pts"] += 1
    return sorted(stats.values(), key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))


def _advance_from_groups(
    groups: dict[str, list[str]],
    predictor: EnsemblePredictor,
    rng: np.random.Generator,
) -> list[str]:
    group_results: dict[str, list[dict]] = {}
    third_places: list[dict] = []

    for g, teams in groups.items():
        standings = _group_standings(teams, predictor, rng)
        group_results[g] = standings
        third_places.append({**standings[2], "group": g})

    # Top 2 from each group
    qualified = []
    for g, standings in group_results.items():
        qualified.extend([standings[0]["team"], standings[1]["team"]])

    # 8 best third-place teams
    third_places.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    qualified.extend([t["team"] for t in third_places[:8]])
    return qualified


def _knockout_bracket(teams: list[str], predictor: EnsemblePredictor, rng: np.random.Generator) -> str:
    """Knockout phase with Elo-based seeding."""
    current = sorted(teams, key=lambda t: predictor.elo.get(t), reverse=True)
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            if i + 1 >= len(current):
                next_round.append(current[i])
                continue
            h, a = current[i], current[i + 1]
            winner, _, _ = _play_match(predictor, h, a, rng, neutral=True)
            next_round.append(winner)
        current = next_round
    return current[0]


def simulate_tournament(
    predictor: EnsemblePredictor,
    n_sims: int = DEFAULT_SIMULATIONS,
    groups: dict[str, list[str]] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    groups = groups or WORLD_CUP_2026_GROUPS
    wins: dict[str, int] = defaultdict(int)
    rng = np.random.default_rng(seed)

    for _ in range(n_sims):
        qualified = _advance_from_groups(groups, predictor, rng)
        champion = _knockout_bracket(qualified, predictor, rng)
        wins[champion] += 1

    return {team: count / n_sims for team, count in wins.items()}


def simulate_with_host_boost(
    predictor: EnsemblePredictor,
    n_sims: int = DEFAULT_SIMULATIONS,
    host_boost: float = 50.0,
) -> dict[str, float]:
    """Apply Elo boost to host nations for home-adjacent advantage."""
    original = {}
    for host in HOST_NATIONS:
        if host in predictor.elo.ratings:
            original[host] = predictor.elo.ratings[host]
        predictor.elo.ratings[host] = predictor.elo.get(host) + host_boost

    results = simulate_tournament(predictor, n_sims=n_sims)

    for host, elo in original.items():
        predictor.elo.ratings[host] = elo
    for host in HOST_NATIONS:
        if host not in original:
            predictor.elo.ratings.pop(host, None)

    return results
