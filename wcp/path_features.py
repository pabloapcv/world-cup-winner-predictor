"""Path-to-final and bracket difficulty features."""

from __future__ import annotations

import numpy as np

from wcp.config import HOST_NATIONS, WORLD_CUP_2026_GROUPS


def _find_group(team: str, groups: dict[str, list[str]]) -> str | None:
    for g, teams in groups.items():
        if team in teams:
            return g
    return None


def group_difficulty(team: str, elo_get, groups: dict[str, list[str]] | None = None) -> float:
    """Average Elo of group opponents (higher = harder group)."""
    groups = groups or WORLD_CUP_2026_GROUPS
    g = _find_group(team, groups)
    if g is None:
        return 1500.0
    opponents = [t for t in groups[g] if t != team]
    return float(np.mean([elo_get(t) for t in opponents]))


def group_strength_spread(team: str, elo_get, groups: dict[str, list[str]] | None = None) -> float:
    """Gap between team Elo and weakest group opponent."""
    groups = groups or WORLD_CUP_2026_GROUPS
    g = _find_group(team, groups)
    if g is None:
        return 0.0
    elos = [elo_get(t) for t in groups[g] if t != team]
    return float(elo_get(team) - min(elos))


def knockout_path_difficulty(team: str, elo_get, groups: dict[str, list[str]] | None = None) -> float:
    """
    Expected cumulative opponent strength through the bracket.
    Uses group difficulty + estimated R32/R16 foes (top teams from other groups).
    """
    groups = groups or WORLD_CUP_2026_GROUPS
    gd = group_difficulty(team, elo_get, groups)

    all_teams = [t for grp in groups.values() for t in grp]
    elite_threshold = np.percentile([elo_get(t) for t in all_teams], 75)
    n_elite = sum(1 for t in all_teams if t != team and elo_get(t) >= elite_threshold)

    # Rough estimate: face ~2 group matches vs quality, then 1-2 elite before final
    expected_knockout_load = n_elite * 0.15 * elite_threshold
    return gd + expected_knockout_load


def host_advantage_feature(team: str) -> int:
    return int(team in HOST_NATIONS)


def build_path_table(elo_get, groups: dict[str, list[str]] | None = None) -> list[dict]:
    groups = groups or WORLD_CUP_2026_GROUPS
    rows = []
    for team in sorted({t for grp in groups.values() for t in grp}):
        gd = group_difficulty(team, elo_get, groups)
        path = knockout_path_difficulty(team, elo_get, groups)
        rows.append({
            "team": team,
            "group": _find_group(team, groups),
            "elo": elo_get(team),
            "group_difficulty": round(gd, 1),
            "path_difficulty": round(path, 1),
            "host": team in HOST_NATIONS,
            "easy_path_index": round(elo_get(team) - path, 1),
        })
    rows.sort(key=lambda x: -x["easy_path_index"])
    return rows
