"""Monte Carlo tournament simulation for World Cup 2026."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wcp.config import DEFAULT_SIMULATIONS, HOST_NATIONS, WORLD_CUP_2026_GROUPS
from wcp.models.ensemble import EnsemblePredictor

STAGES = ("r32", "r16", "quarter", "semi", "final", "champion")


@dataclass
class SimulationResult:
    win_probs: dict[str, float] = field(default_factory=dict)
    stage_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    n_sims: int = 0


def _play_match(
    predictor: EnsemblePredictor,
    home: str,
    away: str,
    rng: np.random.Generator,
    neutral: bool = True,
) -> str:
    hg, ag = predictor.dixon_coles.sample_score(home, away, neutral=neutral, rng=rng)
    if hg > ag:
        return home
    if ag > hg:
        return away
    p_h, _, p_a = predictor.elo.win_prob(home, away, neutral=neutral)
    return home if rng.random() < p_h / (p_h + p_a) else away


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
    third_places: list[dict] = []
    qualified: list[str] = []

    for teams in groups.values():
        standings = _group_standings(teams, predictor, rng)
        qualified.extend([standings[0]["team"], standings[1]["team"]])
        third_places.append(standings[2])

    third_places.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
    qualified.extend([t["team"] for t in third_places[:8]])
    return qualified


def _knockout_round(teams: list[str], predictor: EnsemblePredictor, rng: np.random.Generator) -> list[str]:
    seeded = sorted(teams, key=lambda t: predictor.elo.get(t), reverse=True)
    winners = []
    for i in range(0, len(seeded), 2):
        if i + 1 >= len(seeded):
            winners.append(seeded[i])
            continue
        winners.append(_play_match(predictor, seeded[i], seeded[i + 1], rng))
    return winners


def _simulate_one(
    predictor: EnsemblePredictor,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
    stage_counts: dict[str, dict[str, int]],
) -> str:
    qualified = _advance_from_groups(groups, predictor, rng)
    for t in qualified:
        stage_counts[t]["r32"] += 1

    rounds = [32, 16, 8, 4, 2]
    stage_keys = ["r16", "quarter", "semi", "final", "champion"]
    current = qualified

    for n_teams, stage in zip(rounds, stage_keys):
        current = _knockout_round(current, predictor, rng)
        for t in current:
            stage_counts[t][stage] += 1

    champion = current[0]
    return champion


def simulate_tournament(
    predictor: EnsemblePredictor,
    n_sims: int = DEFAULT_SIMULATIONS,
    groups: dict[str, list[str]] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    return simulate_tournament_detailed(predictor, n_sims, groups, seed).win_probs


def simulate_tournament_detailed(
    predictor: EnsemblePredictor,
    n_sims: int = DEFAULT_SIMULATIONS,
    groups: dict[str, list[str]] | None = None,
    seed: int = 42,
) -> SimulationResult:
    groups = groups or WORLD_CUP_2026_GROUPS
    all_teams = sorted({t for grp in groups.values() for t in grp})
    stage_counts: dict[str, dict[str, int]] = {
        t: {s: 0 for s in STAGES} for t in all_teams
    }
    wins: dict[str, int] = defaultdict(int)
    rng = np.random.default_rng(seed)

    for _ in range(n_sims):
        champion = _simulate_one(predictor, groups, rng, stage_counts)
        wins[champion] += 1

    win_probs = {t: wins.get(t, 0) / n_sims for t in all_teams}
    stage_probs = {
        t: {s: stage_counts[t][s] / n_sims for s in STAGES}
        for t in all_teams
    }
    return SimulationResult(win_probs=win_probs, stage_probs=stage_probs, n_sims=n_sims)


def stage_probs_to_dataframe(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for team, stages in result.stage_probs.items():
        row = {"team": team, **stages}
        row["win_probability"] = stages["champion"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("win_probability", ascending=False)


def simulate_with_host_boost(
    predictor: EnsemblePredictor,
    n_sims: int = DEFAULT_SIMULATIONS,
    host_boost: float = 50.0,
) -> dict[str, float]:
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
